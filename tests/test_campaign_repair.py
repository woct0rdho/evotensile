import math

from evotensile.campaign.acquisition import BundleAcquisitionPolicy, BundleCostModel
from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.evaluator import PairEvaluationOutcome
from evotensile.campaign.repair import (
    RepairPolicy,
    assess_repair_deficits,
    build_repair_candidate_pool,
    plan_repair_acquisition,
    summarize_repair,
)
from evotensile.scheduling.models import EvidenceStage, PairRequest
from evotensile.search.pair_model import PairPrediction
from evotensile.search.shape_clustering import ShapeClusteringConfiguration, cluster_shapes
from evotensile.shapes import pilot_100_shapes
from tests.helpers import sample_candidates


def _controller(shapes, candidates, performances):
    controller = CampaignControllerState(
        shape_ids=tuple(shape.id for shape in shapes),
        time_budget_s=300.0,
        session_started_at=0.0,
    )
    for shape, candidate, performance in zip(shapes, candidates, performances, strict=True):
        controller.record_query(shape.id, candidate.hash, known=True)
        controller.disclose(shape.id, candidate.hash, performance=performance)
    return controller


def _prediction(candidate, shape, *, samples, reference=1200.0, validity=1.0, uncertainty=0.1):
    return PairPrediction(
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        mean_normalized_log_performance=sum(samples) / len(samples),
        epistemic_std_log_performance=uncertainty,
        validity_probability=validity,
        posterior_samples=tuple(samples),
        reference_performance=reference,
    )


def test_singleton_grid_repair_detection_is_noop():
    shapes = pilot_100_shapes()[:1]
    candidate = sample_candidates(1)[0]
    controller = _controller(shapes, [candidate], [900.0])
    clustering = cluster_shapes(
        shapes,
        ShapeClusteringConfiguration(workgroup_processor_count=20, cluster_count=1),
    )

    deficits = assess_repair_deficits(
        controller,
        shapes=shapes,
        clustering=clustering,
        reference_performance={shapes[0].id: 1200.0},
    )

    assert deficits == {}


def test_repair_deficit_and_pair_probability_gate_weak_shape():
    shapes = pilot_100_shapes()[:3]
    incumbents = sample_candidates(3)
    proposals = sample_candidates(5)[3:]
    controller = _controller(shapes, incumbents, [900.0, 1200.0, 1180.0])
    clustering = cluster_shapes(
        shapes,
        ShapeClusteringConfiguration(workgroup_processor_count=20, cluster_count=1),
    )
    predictions = [
        _prediction(proposals[0], shapes[0], samples=(0.0, 0.05, 0.1)),
        _prediction(proposals[1], shapes[0], samples=(-1.0, -0.9, -0.8)),
    ]
    policy = RepairPolicy(minimum_close_probability=0.2, mutation_candidates_per_shape=0)

    deficits = assess_repair_deficits(
        controller,
        shapes=shapes,
        clustering=clustering,
        predictions=predictions,
        reference_performance={shape.id: 1200.0 for shape in shapes},
        policy=policy,
    )
    acquisition = plan_repair_acquisition(
        controller,
        candidates=proposals,
        shapes=shapes,
        deficits=deficits,
        predictions=predictions,
        cost_model=BundleCostModel(
            workgroup_processor_count=20,
            fallback_preparation_s=1.0,
            fallback_validation_s=0.1,
            fallback_timing_s=0.1,
        ),
        acquisition_policy=BundleAcquisitionPolicy(
            improvement_weight=0.0,
            coverage_weight=0.0,
            information_weight=0.0,
            repair_weight=1.0,
            bundle_sizes=(1,),
            max_pairs=1,
            max_bundles=1,
            max_predicted_cost_s=10.0,
            evidence_stage=EvidenceStage.PROBE,
        ),
        repair_policy=policy,
    )

    assert set(deficits) == {shapes[0].id}
    assert math.isclose(deficits[shapes[0].id].capped_deficit_fraction, 0.30)
    assert acquisition.pair_close_probabilities[(shapes[0].id, proposals[0].hash)] == 1.0
    assert acquisition.pair_close_probabilities[(shapes[0].id, proposals[1].hash)] == 0.0
    assert [request.key for request in acquisition.plan.timing_requests] == [(shapes[0].id, proposals[0].hash)]


def test_repair_candidate_pool_audits_all_seed_lanes_and_mutations():
    shapes = pilot_100_shapes()[:3]
    candidates = sample_candidates(6)
    controller = _controller(shapes, candidates[:3], [800.0, 1100.0, 1050.0])
    clustering = cluster_shapes(
        shapes,
        ShapeClusteringConfiguration(workgroup_processor_count=20, cluster_count=1),
    )
    deficits = assess_repair_deficits(
        controller,
        shapes=shapes,
        clustering=clustering,
        reference_performance={shape.id: 1100.0 for shape in shapes},
        policy=RepairPolicy(uncertainty_weight=0.0),
    )
    observations = tuple(
        PairEvaluationOutcome(
            request=PairRequest(candidate, shape),
            provenance="test",
            source_ref="test",
            status="ok",
            known=True,
            disclosed=True,
            samples=3,
            performance=performance,
        )
        for shape, candidate, performance in zip(shapes, candidates[:3], [800.0, 1100.0, 1050.0], strict=True)
    )
    pool = build_repair_candidate_pool(
        controller,
        shapes=shapes,
        clustering=clustering,
        deficits=deficits,
        observations=observations,
        candidate_catalog={candidate.hash: candidate for candidate in candidates},
        broad_candidates=[candidates[3]],
        policy=RepairPolicy(
            neighbor_count=2,
            neighbor_candidates_per_shape=1,
            cluster_candidates=1,
            mutation_candidates_per_shape=2,
            seed=7,
        ),
    )

    lanes = {lane for origin in pool.origins for lane in origin.lanes}
    assert {"incumbent", "neighbor", "cluster", "broad", "mutation"} <= lanes
    assert all(origin.target_shape_ids == (shapes[0].id,) for origin in pool.origins)


def test_repair_report_tracks_reuse_resolution_gain_and_false_cost():
    shapes = pilot_100_shapes()[:3]
    candidates = sample_candidates(4)
    controller = _controller(shapes, candidates[:3], [900.0, 1200.0, 1180.0])
    clustering = cluster_shapes(
        shapes,
        ShapeClusteringConfiguration(workgroup_processor_count=20, cluster_count=1),
    )
    prediction = _prediction(candidates[3], shapes[0], samples=(0.0, 0.0, 0.0))
    policy = RepairPolicy(mutation_candidates_per_shape=0)
    deficits = assess_repair_deficits(
        controller,
        shapes=shapes,
        clustering=clustering,
        predictions=[prediction],
        reference_performance={shape.id: 1200.0 for shape in shapes},
        policy=policy,
    )
    acquisition = plan_repair_acquisition(
        controller,
        candidates=[candidates[3]],
        shapes=shapes,
        deficits=deficits,
        predictions=[prediction],
        cost_model=BundleCostModel(
            workgroup_processor_count=20,
            fallback_preparation_s=1.0,
            fallback_validation_s=0.1,
            fallback_timing_s=0.1,
        ),
        acquisition_policy=BundleAcquisitionPolicy(
            improvement_weight=0.0,
            coverage_weight=0.0,
            information_weight=0.0,
            repair_weight=1.0,
            bundle_sizes=(1,),
            max_pairs=1,
            max_bundles=1,
            max_predicted_cost_s=10.0,
        ),
        repair_policy=policy,
    )
    controller.record_query(shapes[0].id, candidates[3].hash, known=True)
    controller.disclose(shapes[0].id, candidates[3].hash, performance=1100.0)

    report = summarize_repair(
        acquisition,
        controller_after=controller,
        prepared_artifact_shapes_before={candidates[3].hash: {shapes[0].id}},
    )

    assert report.repair_queries == 1
    assert report.preparation_reuse_queries == 1
    assert report.resolved_outliers == 1
    assert report.mean_gain_fraction is not None and report.mean_gain_fraction > 0.20
    assert report.false_repair_queries == 0
    assert report.false_repair_predicted_cost_s == 0.0
