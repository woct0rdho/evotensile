import time
from collections import defaultdict, deque

import pytest

from evotensile.campaign.acquisition import AcquisitionPlan
from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.evaluator import (
    EvaluationResult,
    PairEvaluationOutcome,
    RealEvaluator,
    RealEvaluatorContext,
    ReplayEvaluator,
)
from evotensile.campaign.round_controller import (
    StagedRoundConfiguration,
    run_staged_round,
)
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.protocol import DEFAULT_BENCHMARK_PROTOCOL
from evotensile.scheduling.models import PairRequest
from evotensile.search.replay import ExactOracleReplayState, OracleRecord
from evotensile.shapes import pilot_100_shapes
from tests.helpers import fake_build_tensile, fake_structured_runner, sample_candidates


class FakeClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, duration):
        self.value += duration


class TimedEvaluator:
    def __init__(self, clock, durations, *, interrupt=False):
        self.clock = clock
        self.durations = deque(durations)
        self.interrupt = interrupt
        self.calls = []

    def evaluate(self, requests, *, artifact_shapes_by_candidate=None):
        self.calls.append((tuple(requests), artifact_shapes_by_candidate))
        if self.interrupt:
            raise RuntimeError("interrupted after pending checkpoint")
        self.clock.advance(self.durations.popleft())
        return EvaluationResult(
            mode="test",
            outcomes=tuple(
                PairEvaluationOutcome(
                    request=request,
                    provenance="test",
                    source_ref="staged-round-test",
                    status="ok",
                    known=True,
                    disclosed=True,
                    samples=request.min_samples,
                    performance=100.0,
                )
                for request in requests
            ),
            prepared_artifact_shapes={
                candidate_hash: tuple(shape.id for shape in shapes)
                for candidate_hash, shapes in (artifact_shapes_by_candidate or {}).items()
            },
        )


class ScriptedPlanner:
    def __init__(self, plans):
        self.plans = {phase: deque(items) for phase, items in plans.items()}
        self.plan_calls = defaultdict(int)
        self.observations = []

    def plan_wave(self, phase, controller):
        self.plan_calls[phase] += 1
        queue = self.plans.get(phase)
        return queue.popleft() if queue else None

    def observe(self, phase, result):
        self.observations.append((phase, result))


def _plan(candidate, shape, *, predicted_cost, preparation=True):
    return AcquisitionPlan(
        selected=(),
        preparation_order=((candidate.hash,) if preparation else ()),
        timing_requests=(PairRequest(candidate, shape),),
        artifact_shapes_by_candidate={candidate.hash: (shape,)},
        predicted_cost_s=predicted_cost,
    )


def _configuration(**fractions):
    return StagedRoundConfiguration(
        phase_fractions=tuple(
            (phase, fractions.get(phase, 0.0))
            for phase in ("broad", "promotion", "repair", "stabilization", "confirmation")
        ),
        no_new_preparation_guard_s=fractions.get("guard", 0.0),
    )


def _controller(shape, *, budget=10.0):
    return CampaignControllerState(
        shape_ids=(shape.id,),
        time_budget_s=budget,
        session_started_at=0.0,
    )


def test_admitted_wave_drains_past_soft_deadline_and_blocks_later_work():
    shape = pilot_100_shapes()[0]
    candidates = sample_candidates(2)
    clock = FakeClock()
    controller = _controller(shape)
    planner = ScriptedPlanner(
        {"broad": [_plan(candidates[0], shape, predicted_cost=4.0), _plan(candidates[1], shape, predicted_cost=1.0)]}
    )
    evaluator = TimedEvaluator(clock, [11.0])

    result = run_staged_round(
        controller,
        round_id="round-0",
        configuration=_configuration(broad=1.0),
        model_identity="model-a",
        candidates={candidate.hash: candidate for candidate in candidates},
        shapes={shape.id: shape},
        planner=planner,
        evaluator=evaluator,
        now=clock,
    )

    assert len(evaluator.calls) == 1
    assert result.state.stop_reason == "overrun"
    assert controller.overrun_s(now=clock()) == pytest.approx(1.0)
    assert planner.plan_calls["broad"] == 1


def test_phase_deadline_skips_one_phase_without_becoming_a_hard_timeout():
    shape = pilot_100_shapes()[0]
    candidates = sample_candidates(2)
    clock = FakeClock()
    controller = _controller(shape)
    planner = ScriptedPlanner(
        {
            "broad": [_plan(candidates[0], shape, predicted_cost=6.0)],
            "promotion": [_plan(candidates[1], shape, predicted_cost=2.0, preparation=False)],
        }
    )
    evaluator = TimedEvaluator(clock, [1.0])

    result = run_staged_round(
        controller,
        round_id="round-0",
        configuration=_configuration(broad=0.5, promotion=0.3, confirmation=0.2),
        model_identity="model-a",
        candidates={candidate.hash: candidate for candidate in candidates},
        shapes={shape.id: shape},
        planner=planner,
        evaluator=evaluator,
        now=clock,
    )

    assert [admission["reason"] for admission in result.state.admissions] == ["phase_deadline", "admitted"]
    assert evaluator.calls[0][0][0].candidate.hash == candidates[1].hash
    assert result.state.stop_reason == "completed"


def test_final_guard_rejects_new_preparation_but_allows_prepared_confirmation():
    shape = pilot_100_shapes()[0]
    candidates = sample_candidates(2)
    clock = FakeClock(8.0)
    controller = _controller(shape)
    planner = ScriptedPlanner(
        {
            "stabilization": [_plan(candidates[0], shape, predicted_cost=1.0, preparation=True)],
            "confirmation": [_plan(candidates[1], shape, predicted_cost=1.0, preparation=False)],
        }
    )
    evaluator = TimedEvaluator(clock, [1.0])

    result = run_staged_round(
        controller,
        round_id="round-0",
        configuration=_configuration(stabilization=0.9, confirmation=0.1, guard=3.0),
        model_identity="model-a",
        candidates={candidate.hash: candidate for candidate in candidates},
        shapes={shape.id: shape},
        planner=planner,
        evaluator=evaluator,
        now=clock,
    )

    assert [admission["reason"] for admission in result.state.admissions] == [
        "no_new_preparation_guard",
        "admitted",
    ]
    assert evaluator.calls[0][0][0].candidate.hash == candidates[1].hash


def test_replay_round_charges_simulated_phase_time_and_stops_after_overrun(tmp_path):
    shape = pilot_100_shapes()[0]
    candidates = sample_candidates(2)
    clock = FakeClock()
    db = EvoTensileDB.connect(
        tmp_path / "replay.sqlite",
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    evaluator = ReplayEvaluator(
        ExactOracleReplayState(
            db=db,
            shapes=[shape],
            oracle={
                (shape.id, candidate.hash): OracleRecord(
                    candidate=candidate,
                    status="ok",
                    screening_gflops=100.0,
                )
                for candidate in candidates
            },
            profile=DEFAULT_PROFILE,
        ),
        prepare_seconds_per_candidate=2.0,
    )
    controller = CampaignControllerState(
        shape_ids=(shape.id,),
        time_budget_s=1.0,
        session_started_at=0.0,
    )
    planner = ScriptedPlanner(
        {"broad": [_plan(candidates[0], shape, predicted_cost=0.5), _plan(candidates[1], shape, predicted_cost=0.1)]}
    )

    result = run_staged_round(
        controller,
        round_id="round-0",
        configuration=_configuration(broad=1.0),
        model_identity="model-a",
        candidates={candidate.hash: candidate for candidate in candidates},
        shapes={shape.id: shape},
        planner=planner,
        evaluator=evaluator,
        now=clock,
        charge_result_time=lambda evaluation: clock.advance(sum(evaluation.phase_time_s.values())),
    )

    assert result.state.stop_reason == "overrun"
    assert len(controller.queried_pairs) == 1
    assert planner.plan_calls["broad"] == 1
    assert controller.overrun_s(now=clock()) > 0.0


def test_real_round_executes_exact_persisted_wave_through_scheduler(tmp_path):
    shape = pilot_100_shapes()[0]
    candidate = sample_candidates(1)[0]
    protocol = DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1)
    db = EvoTensileDB.connect(
        tmp_path / "real.sqlite",
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    evaluator = RealEvaluator(
        RealEvaluatorContext(
            db=db,
            output_root=tmp_path / "real-round",
            target_profile=DEFAULT_PROFILE,
            protocol=protocol,
            runner_bin=fake_structured_runner(tmp_path),
            tensilelite_bin=fake_build_tensile(tmp_path),
            candidate_batch_size=1,
            shape_batch_size=1,
            compile_cache_root=tmp_path / "compile-cache",
        )
    )
    controller = CampaignControllerState(
        shape_ids=(shape.id,),
        time_budget_s=30.0,
        session_started_at=time.monotonic(),
    )
    planner = ScriptedPlanner({"broad": [_plan(candidate, shape, predicted_cost=1.0)]})

    result = run_staged_round(
        controller,
        round_id="round-0",
        configuration=_configuration(broad=1.0),
        model_identity="model-a",
        candidates={candidate.hash: candidate},
        shapes={shape.id: shape},
        planner=planner,
        evaluator=evaluator,
    )

    assert result.state.stop_reason == "completed"
    assert len(result.evaluation_results) == 1
    assert len(result.evaluation_results[0].schedules) == 1
    assert (shape.id, candidate.hash) in controller.disclosed_pairs
    assert controller.active_round is not None
    assert controller.active_round["pending"] is None


def test_pending_exact_wave_resumes_without_regenerating_requests():
    shape = pilot_100_shapes()[0]
    candidate = sample_candidates(1)[0]
    clock = FakeClock()
    controller = _controller(shape)
    plan = _plan(candidate, shape, predicted_cost=2.0)
    planner = ScriptedPlanner({"broad": [plan]})
    checkpoints = []
    interrupting = TimedEvaluator(clock, [], interrupt=True)

    with pytest.raises(RuntimeError, match="interrupted"):
        run_staged_round(
            controller,
            round_id="round-0",
            configuration=_configuration(broad=1.0),
            model_identity="model-a",
            candidates={candidate.hash: candidate},
            shapes={shape.id: shape},
            planner=planner,
            evaluator=interrupting,
            checkpoint=lambda state: checkpoints.append(state.to_checkpoint(now=clock())),
            now=clock,
        )

    assert controller.active_round is not None
    assert controller.active_round["pending"]["requests"][0]["candidate_hash"] == candidate.hash
    resumed = TimedEvaluator(clock, [1.0])
    result = run_staged_round(
        controller,
        round_id="round-0",
        configuration=_configuration(broad=1.0),
        model_identity="model-a",
        candidates={candidate.hash: candidate},
        shapes={shape.id: shape},
        planner=planner,
        evaluator=resumed,
        checkpoint=lambda state: checkpoints.append(state.to_checkpoint(now=clock())),
        now=clock,
    )

    assert planner.plan_calls["broad"] == 2
    assert resumed.calls[0][0][0].candidate.hash == candidate.hash
    assert result.state.completed_waves[0]["request_pairs"] == [[shape.id, candidate.hash]]
    assert result.state.stop_reason == "completed"
    assert checkpoints
