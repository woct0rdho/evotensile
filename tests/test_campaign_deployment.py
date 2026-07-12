from dataclasses import dataclass

import pytest

from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.deployment import (
    DeploymentSelection,
    FinalistConfirmationPolicy,
    plan_confirmation_finalists,
    plan_stabilization_finalists,
    run_final_confirmation,
    select_deployment_solution_bank,
)
from evotensile.campaign.evaluator import EvaluationResult, PairEvaluationOutcome
from evotensile.scheduling.models import EvidenceStage, PairRequest
from evotensile.search.pair_model import PairPrediction
from evotensile.shapes import pilot_100_shapes
from tests.helpers import sample_candidates


def _prediction(candidate, shape, samples, *, validity=1.0):
    return PairPrediction(
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        mean_normalized_log_performance=sum(samples) / len(samples),
        epistemic_std_log_performance=0.1,
        validity_probability=validity,
        posterior_samples=tuple(samples),
        reference_performance=100.0,
    )


def _outcome(candidate, shape, performance, *, known=True):
    return PairEvaluationOutcome(
        request=PairRequest(candidate, shape, evidence_stage=EvidenceStage.CONFIRMATION, min_samples=10),
        provenance="replay",
        source_ref="deployment-test",
        status="ok" if known else "unknown",
        known=known,
        disclosed=known,
        samples=10 if known else 0,
        performance=performance if known else None,
    )


def test_finalist_plan_keeps_incumbents_and_posterior_close_competitors():
    shapes = pilot_100_shapes()[:2]
    candidates = sample_candidates(3)
    predictions = []
    for shape in shapes:
        predictions.extend(
            (
                _prediction(candidates[0], shape, (0.00, 0.02, 0.01)),
                _prediction(candidates[1], shape, (-0.01, 0.01, 0.00)),
                _prediction(candidates[2], shape, (-1.00, -1.00, -1.00)),
            )
        )

    arguments = {
        "candidates": {candidate.hash: candidate for candidate in candidates},
        "shapes": {shape.id: shape for shape in shapes},
        "incumbent_candidates_by_shape": {shape.id: candidates[2].hash for shape in shapes},
        "shape_weights": {shapes[0].id: 1.8, shapes[1].id: 0.2},
        "policy": FinalistConfirmationPolicy(
            relative_tolerance=0.03,
            minimum_close_probability=0.1,
            maximum_finalists_per_shape=3,
            stabilization_samples=5,
            min_samples=12,
        ),
    }
    stabilization = plan_stabilization_finalists(predictions, **arguments)
    plan = plan_confirmation_finalists(predictions, **arguments)

    assert all(
        finalist.request.evidence_stage == EvidenceStage.STABILIZATION and finalist.request.min_samples == 5
        for finalist in stabilization.finalists
    )
    assert len(plan.finalists) == 6
    assert all(finalist.request.evidence_stage == EvidenceStage.CONFIRMATION for finalist in plan.finalists)
    assert all(finalist.request.min_samples == 12 for finalist in plan.finalists)
    assert {
        finalist.request.shape.id
        for finalist in plan.finalists
        if finalist.incumbent and finalist.request.candidate.hash == candidates[2].hash
    } == {shape.id for shape in shapes}
    first_priorities = [
        finalist.request.priority for finalist in plan.finalists if finalist.request.shape.id == shapes[0].id
    ]
    second_priorities = [
        finalist.request.priority for finalist in plan.finalists if finalist.request.shape.id == shapes[1].id
    ]
    assert min(first_priorities) > max(second_priorities)


def test_solution_bank_zero_tolerance_preserves_exact_winners_and_nonzero_consolidates():
    shapes = pilot_100_shapes()[:3]
    candidates = sample_candidates(2)
    outcomes = [
        _outcome(candidates[0], shapes[0], 100.0),
        _outcome(candidates[0], shapes[1], 100.0),
        _outcome(candidates[0], shapes[2], 90.0),
        _outcome(candidates[1], shapes[0], 99.0),
        _outcome(candidates[1], shapes[1], 99.0),
        _outcome(candidates[1], shapes[2], 100.0),
    ]

    exact = select_deployment_solution_bank(
        outcomes,
        shape_ids=[shape.id for shape in shapes],
        tolerance_fraction=0.0,
    )
    consolidated = select_deployment_solution_bank(
        outcomes,
        shape_ids=[shape.id for shape in shapes],
        tolerance_fraction=0.02,
        shape_weights={shapes[0].id: 2.5, shapes[1].id: 0.4, shapes[2].id: 0.1},
        code_object_identity_by_candidate={
            candidates[0].hash: "code-a",
            candidates[1].hash: "code-b",
        },
    )

    assert exact.assignments == exact.exact_winners
    assert exact.solution_count == 2
    assert exact.worst_shape_loss_fraction == 0.0
    assert set(consolidated.assignments.values()) == {candidates[1].hash}
    assert consolidated.generalist_coverage[candidates[1].hash] == tuple(shape.id for shape in shapes)
    assert consolidated.specialist_shape_ids == ()
    assert consolidated.solution_count == 1
    assert consolidated.code_object_count == 1
    assert not consolidated.code_object_count_conservative
    assert consolidated.uniform_mean_loss_fraction == pytest.approx(0.02 / 3.0)
    assert consolidated.workload_weighted_mean_loss_fraction == pytest.approx(0.009666666666666667)
    assert consolidated.worst_shape_loss_fraction == pytest.approx(0.01)
    assert DeploymentSelection.from_dict(consolidated.to_dict()) == consolidated


def test_singleton_solution_bank_returns_confirmed_winner():
    shape = pilot_100_shapes()[0]
    candidates = sample_candidates(2)

    selection = select_deployment_solution_bank(
        [
            _outcome(candidates[0], shape, 100.0),
            _outcome(candidates[1], shape, 99.0),
        ],
        shape_ids=(shape.id,),
        tolerance_fraction=0.05,
    )

    assert selection.assignments == {shape.id: candidates[0].hash}
    assert selection.solution_count == 1
    assert selection.code_object_count == 1
    assert selection.uniform_mean_loss_fraction == 0.0


@dataclass
class _Clock:
    value: float = 0.0


class _ConfirmationEvaluator:
    def evaluate(self, requests, *, artifact_shapes_by_candidate=None):
        return EvaluationResult(
            mode="test",
            outcomes=tuple(_outcome(request.candidate, request.shape, 100.0) for request in requests),
            phase_time_s={"confirmation": 6.0},
        )


def test_confirmation_soft_deadline_lets_admitted_group_drain_then_stops():
    shapes = pilot_100_shapes()[:2]
    candidates = sample_candidates(2)
    predictions = [
        _prediction(candidate, shape, (0.0, 0.0)) for candidate, shape in zip(candidates, shapes, strict=True)
    ]
    plan = plan_confirmation_finalists(
        predictions,
        candidates={candidate.hash: candidate for candidate in candidates},
        shapes={shape.id: shape for shape in shapes},
        incumbent_candidates_by_shape={
            shape.id: candidate.hash for candidate, shape in zip(candidates, shapes, strict=True)
        },
        policy=FinalistConfirmationPolicy(
            maximum_finalists_per_shape=1,
            fallback_group_cost_s=3.0,
        ),
    )
    controller = CampaignControllerState(
        shape_ids=tuple(shape.id for shape in shapes),
        time_budget_s=5.0,
        session_started_at=0.0,
    )
    clock = _Clock()

    run = run_final_confirmation(
        _ConfirmationEvaluator(),
        controller,
        plan,
        now=lambda: clock.value,
        charge_result_time=lambda result: setattr(
            clock,
            "value",
            clock.value + sum(result.phase_time_s.values()),
        ),
    )

    assert run.stop_reason == "overrun"
    assert len(run.results) == 1
    assert run.admissions[0]["admitted"] is True
    assert clock.value == 6.0
    assert controller.overrun_s(now=clock.value) == 1.0
