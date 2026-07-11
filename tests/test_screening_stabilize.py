from pathlib import Path

import pytest

from evotensile.adaptive_retime import timing_stats_from_times
from evotensile.artifacts import CandidateArtifact
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.protocol import DEFAULT_BENCHMARK_PROTOCOL
from evotensile.scheduler import execute_schedule
from evotensile.search import screening_stabilize
from evotensile.search.screening_stabilize import (
    ScreeningStabilizationPolicy,
    plan_screening_stabilization,
    stabilize_screening_leaders,
)
from evotensile.shapes import pilot_100_shapes
from evotensile.structured_runner import RunnablePair, StructuredRunOutput
from tests.helpers import fake_build_tensile, fake_structured_runner, sample_candidates


def _screening_protocol():
    return DEFAULT_BENCHMARK_PROTOCOL.with_overrides(
        num_warmups=1,
        num_benchmarks=2,
        enqueues_per_sync=1,
    )


def test_stabilization_reports_capped_timer_and_uncertainty_criteria():
    shape = pilot_100_shapes()[0]
    stats = {
        shape.id: [
            timing_stats_from_times(shape.id, "fast-noisy", [1.0, 2.0]),
        ]
    }

    plan = plan_screening_stabilization(
        stats,
        shapes=[shape],
        protocol=_screening_protocol(),
        policy=ScreeningStabilizationPolicy(),
    )

    finalist = plan.finalists[0]
    assert finalist.target_samples == 24
    assert finalist.uncapped_target_samples > finalist.target_samples
    assert finalist.required_timer_target > finalist.target_samples
    assert finalist.required_uncertainty_target > finalist.target_samples
    assert finalist.capped_criteria == ("timer_resolution", "uncertainty")


def test_stabilization_does_not_report_criteria_already_met_above_cap():
    shape = pilot_100_shapes()[0]
    times = [1.0 if index % 2 == 0 else 2.0 for index in range(200)]

    plan = plan_screening_stabilization(
        {shape.id: [timing_stats_from_times(shape.id, "measured", times)]},
        shapes=[shape],
        protocol=_screening_protocol(),
        policy=ScreeningStabilizationPolicy(),
    )

    finalist = plan.finalists[0]
    assert finalist.current_samples == 200
    assert not finalist.needs_topup
    assert not finalist.capped_criteria


def test_grid_stabilization_fair_queues_clusters_and_shapes():
    shapes = pilot_100_shapes()[:3]
    stats = {
        shape.id: [timing_stats_from_times(shape.id, f"candidate-{index}", [25.0, 25.0])]
        for index, shape in enumerate(shapes)
    }
    clusters = {
        shapes[0].id: "cluster-a",
        shapes[1].id: "cluster-a",
        shapes[2].id: "cluster-b",
    }

    plan = plan_screening_stabilization(
        stats,
        shapes=shapes,
        protocol=_screening_protocol(),
        policy=ScreeningStabilizationPolicy(
            top_k=1,
            min_samples=8,
            max_samples=8,
            min_launches=1,
            min_timer_ticks=0,
            uncertainty_half_width_pct=0.0,
        ),
        shape_clusters=clusters,
    )

    assert plan.shape_queues == 3
    assert plan.cluster_queues == 2
    assert [(request.cluster_id, request.shape_id) for request in plan.requests] == [
        ("cluster-a", shapes[0].id),
        ("cluster-b", shapes[2].id),
        ("cluster-a", shapes[1].id),
    ]
    assert [request.queue_index for request in plan.requests] == [0, 1, 2]


def test_grid_stabilization_requires_exact_cluster_coverage():
    shapes = pilot_100_shapes()[:2]

    with pytest.raises(ValueError, match="cover exactly"):
        plan_screening_stabilization(
            {},
            shapes=shapes,
            protocol=_screening_protocol(),
            policy=ScreeningStabilizationPolicy(),
            shape_clusters={shapes[0].id: "cluster-a"},
        )


def test_screening_stabilization_runner_budget_skips_later_fair_pair(tmp_path: Path, monkeypatch):
    db = EvoTensileDB.connect(tmp_path / "campaign.sqlite")
    db.init()
    shapes = pilot_100_shapes()[:2]
    stats = {
        shape.id: [timing_stats_from_times(shape.id, f"candidate-{index}", [25.0, 25.0])]
        for index, shape in enumerate(shapes)
    }
    artifacts = {
        (shape.id, f"candidate-{index}"): CandidateArtifact(
            runnable_pair=RunnablePair(
                shape_id=shape.id,
                candidate_hash=f"candidate-{index}",
                problem_index=index,
                requested_solution_index=index,
                library_solution_index=index,
                manifest_solution_index=index,
            ),
            build_run_id=f"build-{index}",
            build_output_dir=tmp_path,
            library_dir=tmp_path / f"library-{index}",
            solution_yaml_paths=(tmp_path / f"solution-{index}.yaml",),
            manifest_path=None,
            code_object_identity=f"artifact-{index}",
        )
        for index, shape in enumerate(shapes)
    }
    monkeypatch.setattr(screening_stabilize, "load_timing_stats", lambda *args, **kwargs: stats)
    monkeypatch.setattr(
        db,
        "validated_cache_entries",
        lambda **kwargs: set(artifacts),
    )
    monkeypatch.setattr(screening_stabilize, "load_candidate_artifacts", lambda *args, **kwargs: artifacts)

    calls = []

    def fake_run_structured_phase(**kwargs):
        calls.extend(pair.shape_id for pair in kwargs["pairs"])
        return StructuredRunOutput(
            mode="benchmark",
            run_id="benchmark-budget",
            returncode=1,
            samples=[],
            stdout_path=tmp_path / "stdout",
            stderr_path=tmp_path / "stderr",
            results_path=tmp_path / "results",
            duration_s=0.02,
            command=["fake"],
        )

    monkeypatch.setattr(screening_stabilize, "run_structured_phase", fake_run_structured_phase)
    result = stabilize_screening_leaders(
        db,
        shapes=shapes,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        screening_protocol=_screening_protocol(),
        validation_protocol_hash=_screening_protocol().validation_protocol_hash(),
        output_dir=tmp_path / "stabilize",
        runner_bin="fake",
        policy=ScreeningStabilizationPolicy(
            top_k=1,
            min_samples=8,
            max_samples=8,
            min_launches=1,
            min_timer_ticks=0,
            uncertainty_half_width_pct=0.0,
            max_runner_duration_s=0.01,
        ),
    )

    assert calls == [shapes[0].id]
    assert result.runner_budget_exhausted
    assert result.skipped_pairs[-1].shape_id == shapes[1].id
    assert result.skipped_pairs[-1].reason == "runner_duration_budget"


def test_screening_stabilization_reuses_prior_artifacts(tmp_path: Path, monkeypatch):
    fake_tensile = fake_build_tensile(tmp_path)
    fake_runner = fake_structured_runner(tmp_path)
    db = EvoTensileDB.connect(tmp_path / "campaign.sqlite")
    candidates = sample_candidates(2)
    shape = pilot_100_shapes()[0]
    protocol = _screening_protocol()
    events_path = tmp_path / "events.log"
    monkeypatch.setenv("EVOTENSILE_TEST_PHASE_EVENTS", str(events_path))

    execute_schedule(
        db,
        shapes=[shape],
        candidates=candidates,
        output_root=tmp_path / "round_00",
        protocol=protocol,
        candidate_batch_size=2,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=fake_runner,
        keep_going=True,
        prepare_workers=2,
    )
    result = stabilize_screening_leaders(
        db,
        shapes=[shape],
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        screening_protocol=protocol,
        validation_protocol_hash=protocol.validation_protocol_hash(),
        output_dir=tmp_path / "leader_stabilization",
        runner_bin=fake_runner,
        policy=ScreeningStabilizationPolicy(
            top_k=2,
            contender_epsilon_pct=100.0,
            min_samples=8,
            max_samples=8,
            min_launches=1,
            min_timer_ticks=0,
            uncertainty_half_width_pct=0.0,
        ),
    )

    ranked = db.rank_evaluations(
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=protocol.protocol_hash(),
        shape_id=shape.id,
    )
    events = events_path.read_text(encoding="utf-8").splitlines()
    assert result.runs == 1
    assert result.added_samples == 12
    assert result.runner_duration_s > 0.0
    assert not result.errors
    assert not result.skipped_pairs
    assert result.completed_pairs == tuple(request.pair for request in result.plan.requests)
    assert [summary.samples for summary in ranked] == [8, 8]
    assert events.count("compile_start") == 1
    assert events.count("validate_start") == 1
    assert events.count("benchmark_start") == 2
