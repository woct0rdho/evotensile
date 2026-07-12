import math

import pytest

from evotensile.campaign.acquisition import (
    BundleAcquisitionPolicy,
    BundleCostModel,
    plan_candidate_bundles,
)
from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.workload import ResolvedWorkloadWeights, ShapeWorkload
from evotensile.profile import DEFAULT_PROFILE
from evotensile.search.measured_cost import CandidateMeasuredCost
from evotensile.search.pair_model import PairPrediction
from evotensile.shapes import pilot_100_shapes
from tests.helpers import sample_candidates


def _prediction(candidate, shape, samples, *, validity=1.0, uncertainty=0.0, reference=100.0):
    return PairPrediction(
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        mean_normalized_log_performance=sum(samples) / len(samples),
        epistemic_std_log_performance=uncertainty,
        validity_probability=validity,
        posterior_samples=tuple(samples),
        reference_performance=reference,
    )


def _controller(shapes):
    controller = CampaignControllerState(
        shape_ids=tuple(shape.id for shape in shapes),
        time_budget_s=300.0,
        session_started_at=0.0,
    )
    for shape in shapes:
        controller.record_query(shape.id, "incumbent", known=True)
        controller.disclose(shape.id, "incumbent", performance=100.0)
    return controller


def _cost_model(**overrides):
    return BundleCostModel(
        workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
        fallback_preparation_s=overrides.get("fallback_preparation_s", 10.0),
        fallback_validation_s=overrides.get("fallback_validation_s", 1.0),
        fallback_timing_s=overrides.get("fallback_timing_s", 1.0),
        min_fit_rows=overrides.get("min_fit_rows", 12),
    )


def test_bundle_cost_counts_preparation_once_and_pair_cost_per_exact_request():
    shapes = pilot_100_shapes()[:2]
    candidate = sample_candidates(1)[0]
    model = _cost_model()

    first = model.estimate(candidate, shapes[:1], prepared_shape_ids=set())
    second = model.estimate(candidate, shapes[1:], prepared_shape_ids=set())
    bundle = model.estimate(candidate, shapes, prepared_shape_ids=set())

    assert bundle.preparation_required
    assert bundle.preparation_s < first.preparation_s + second.preparation_s
    assert bundle.validation_s == pytest.approx(first.validation_s + second.validation_s)
    assert bundle.timing_s == pytest.approx(first.timing_s + second.timing_s)


def test_lazy_greedy_uses_posterior_samples_to_avoid_duplicate_shape_gain():
    shapes = pilot_100_shapes()[:2]
    candidates = sample_candidates(3)
    controller = _controller(shapes)
    predictions = [
        _prediction(candidates[0], shapes[0], (0.20, 0.20, 0.20)),
        _prediction(candidates[1], shapes[0], (0.19, 0.19, 0.19)),
        _prediction(candidates[2], shapes[1], (0.12, 0.12, 0.12)),
    ]

    plan = plan_candidate_bundles(
        controller,
        candidates=candidates,
        shapes=shapes,
        predictions=predictions,
        cost_model=_cost_model(fallback_preparation_s=1.0, fallback_validation_s=0.1, fallback_timing_s=0.1),
        policy=BundleAcquisitionPolicy(
            coverage_weight=0.0,
            information_weight=0.0,
            bundle_sizes=(1,),
            max_pairs=2,
            max_bundles=2,
            max_predicted_cost_s=10.0,
        ),
    )

    selected_hashes = {score.bundle.candidate.hash for score in plan.selected}
    assert candidates[0].hash in selected_hashes
    assert candidates[2].hash in selected_hashes
    assert candidates[1].hash not in selected_hashes


def test_acquisition_uses_controller_workload_weights_for_timing_priority():
    shapes = pilot_100_shapes()[:2]
    candidate = sample_candidates(1)[0]
    controller = _controller(shapes)
    controller.set_workload(
        ResolvedWorkloadWeights.workload(
            controller.shape_ids,
            (
                ShapeWorkload(shapes[0].id, call_count=1.0, baseline_latency_us=1.0),
                ShapeWorkload(shapes[1].id, call_count=1.0, baseline_latency_us=9.0),
            ),
            provenance={
                "call_count_source": "test-workload",
                "baseline_label": "test-baseline",
                "baseline_source": "test.sqlite",
                "benchmark_protocol_hash": "bproto_test",
                "environment_compatibility_tag": "test-environment",
            },
        )
    )
    predictions = [_prediction(candidate, shape, (0.1, 0.1, 0.1)) for shape in shapes]

    plan = plan_candidate_bundles(
        controller,
        candidates=[candidate],
        shapes=shapes,
        predictions=predictions,
        cost_model=_cost_model(
            fallback_preparation_s=1.0,
            fallback_validation_s=0.0,
            fallback_timing_s=0.0,
        ),
        policy=BundleAcquisitionPolicy(
            coverage_weight=0.0,
            information_weight=0.0,
            bundle_sizes=(1,),
            max_pairs=1,
            max_bundles=1,
            max_predicted_cost_s=100.0,
        ),
    )

    assert [request.key for request in plan.timing_requests] == [(shapes[1].id, candidate.hash)]
    assert plan.timing_requests[0].priority > 0.0


def test_acquisition_emits_exact_requests_and_preserves_expanded_artifact_scope():
    shapes = pilot_100_shapes()[:3]
    candidate = sample_candidates(1)[0]
    controller = _controller(shapes)
    controller.record_prepared(candidate.hash, [shapes[0].id])
    predictions = [_prediction(candidate, shapes[1], (0.1, 0.2, 0.3))]

    plan = plan_candidate_bundles(
        controller,
        candidates=[candidate],
        shapes=shapes,
        predictions=predictions,
        cost_model=_cost_model(),
        policy=BundleAcquisitionPolicy(
            coverage_weight=0.0,
            information_weight=0.0,
            bundle_sizes=(1,),
            max_pairs=1,
            max_bundles=1,
            max_predicted_cost_s=100.0,
        ),
        artifact_shapes_by_target={
            shapes[0].id: (shapes[0],),
            shapes[1].id: (shapes[1], shapes[2]),
            shapes[2].id: (shapes[2],),
        },
    )

    assert [request.key for request in plan.timing_requests] == [(shapes[1].id, candidate.hash)]
    assert {shape.id for shape in plan.artifact_shapes_by_candidate[candidate.hash]} == {
        shapes[0].id,
        shapes[1].id,
        shapes[2].id,
    }
    assert plan.selected[0].bundle.cost.artifact_expansion_required
    assert plan.preparation_order == (candidate.hash,)


def test_singleton_bundle_reduces_to_expected_improvement_per_cost():
    shape = pilot_100_shapes()[0]
    candidate = sample_candidates(1)[0]
    controller = _controller([shape])
    prediction = _prediction(candidate, shape, (0.0, 0.1, 0.2), validity=0.5)
    model = _cost_model(fallback_preparation_s=1.0, fallback_validation_s=0.0, fallback_timing_s=0.0)

    plan = plan_candidate_bundles(
        controller,
        candidates=[candidate],
        shapes=[shape],
        predictions=[prediction],
        cost_model=model,
        policy=BundleAcquisitionPolicy(
            coverage_weight=0.0,
            information_weight=0.0,
            bundle_sizes=(1,),
            max_pairs=1,
            max_bundles=1,
            max_predicted_cost_s=100.0,
        ),
    )

    score = plan.selected[0]
    assert score.expected_improvement == pytest.approx(0.05)
    assert score.unresolved_coverage == 0.0
    assert score.marginal_utility == pytest.approx(0.05)
    assert score.utility_per_s == pytest.approx(score.marginal_utility / score.bundle.cost.total_s)


def test_cost_model_fits_measured_preparation_validation_and_timing_components():
    shapes = pilot_100_shapes()[:2]
    candidates = sample_candidates(14, seed=12345)
    candidate_by_hash = {candidate.hash: candidate for candidate in candidates}
    shapes_by_candidate = {candidate.hash: shapes for candidate in candidates}
    measured = {
        candidate.hash: CandidateMeasuredCost(
            prepare_s=1.0 + index,
            validation_s=0.2 + 0.02 * index,
            probe_s=0.1,
            screening_s=0.3 + 0.01 * index,
        )
        for index, candidate in enumerate(candidates)
    }
    model = _cost_model(min_fit_rows=12)

    summary = model.fit(
        candidates=candidate_by_hash,
        shapes_by_candidate=shapes_by_candidate,
        measured_costs=measured,
    )
    estimate = model.estimate(candidates[-1], shapes, prepared_shape_ids=set())

    assert summary.preparation_fitted
    assert summary.validation_fitted
    assert summary.timing_fitted
    assert summary.preparation_rows == len(candidates)
    assert summary.validation_rows == len(candidates) * len(shapes)
    assert summary.timing_rows == len(candidates) * len(shapes)
    assert math.isfinite(estimate.total_s)
    assert estimate.total_s > 0.0
