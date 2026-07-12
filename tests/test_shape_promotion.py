from pathlib import Path

from evotensile.campaign.baselines import evaluate_representative_first_baseline
from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.evaluator import ReplayEvaluator
from evotensile.campaign.promotion import PromotionPolicy, execute_promotion_race, plan_shape_promotions
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.scheduling.models import PairRequest
from evotensile.search.replay import ExactOracleReplayState, OracleRecord
from evotensile.search.shape_clustering import ShapeClusteringConfiguration, cluster_shapes
from evotensile.shapes import pilot_100_shapes
from tests.helpers import sample_candidates


def _setup(tmp_path: Path):
    shapes = pilot_100_shapes()[:8]
    candidates = sample_candidates(5)
    clustering = cluster_shapes(
        shapes,
        ShapeClusteringConfiguration(
            workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
            cluster_count=3,
            macro_tile_family_count=6,
        ),
    )
    first_cluster = next(cluster for cluster in clustering.clusters if len(cluster.shape_ids) > 1)
    specialist_shape_ids = set(first_cluster.shape_ids)
    second_medoid = next(
        cluster.medoid_shape_id for cluster in clustering.clusters if cluster.cluster_id != first_cluster.cluster_id
    )
    oracle = {}
    for shape in shapes:
        for index, candidate in enumerate(candidates):
            performance = 100.0
            if index == 1:
                performance = 105.0 if shape.id in specialist_shape_ids else 88.0
            elif index == 2:
                performance = 140.0 if shape.id == second_medoid else 75.0
            elif index == 3:
                performance = 90.0
            elif index == 4:
                performance = 92.0
            oracle[(shape.id, candidate.hash)] = OracleRecord(
                candidate=candidate,
                status="ok",
                screening_gflops=performance,
                order=0.01,
            )
    db = EvoTensileDB.connect(
        tmp_path / "promotion.sqlite",
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    state = ExactOracleReplayState(
        db=db,
        shapes=shapes,
        oracle=oracle,
        profile=DEFAULT_PROFILE,
        source_ref="promotion-test",
    )
    evaluator = ReplayEvaluator(state, prepare_seconds_per_candidate=0.1)
    controller = CampaignControllerState(
        shape_ids=tuple(shape.id for shape in shapes),
        time_budget_s=10.0,
        session_started_at=0.0,
    )
    controller.set_clustering(clustering.to_dict())
    seed = evaluate_representative_first_baseline(
        evaluator,
        controller,
        candidates=candidates,
        shapes=shapes,
        clustering=clustering,
    )
    return shapes, candidates, clustering, evaluator, controller, seed.result, first_cluster


def test_promotion_planner_uses_specialist_representative_nearest_and_broad_lanes(tmp_path: Path):
    shapes, candidates, clustering, _, controller, seed, first_cluster = _setup(tmp_path)
    policy = PromotionPolicy(
        representative_finalist_count=3,
        max_promotions_per_shape=6,
        broad_candidate_slots=2,
    )

    plans = plan_shape_promotions(
        controller,
        shapes=shapes,
        clustering=clustering,
        observations=seed.outcomes,
        policy=policy,
    )

    lanes = {plan.lane for plan in plans}
    assert {"specialist", "representative", "nearest", "broad"}.issubset(lanes)
    assert all(plan.pair not in controller.queried_pairs for plan in plans)
    assert all(plan.source_shape_id != plan.destination_shape_id for plan in plans if plan.lane != "representative")
    assert not any(
        plan.candidate.hash == candidates[2].hash and plan.destination_cluster_id == first_cluster.cluster_id
        for plan in plans
    )


def test_promotion_race_probes_then_tops_up_survivors_with_shared_artifact_scopes(tmp_path: Path):
    shapes, _, clustering, evaluator, controller, seed, _ = _setup(tmp_path)
    policy = PromotionPolicy(
        max_promotions_per_shape=5,
        broad_candidate_slots=1,
        probe_samples=1,
        main_samples=3,
        probe_survivor_regret_fraction=0.08,
    )

    result = execute_promotion_race(
        evaluator,
        controller,
        shapes=shapes,
        clustering=clustering,
        observations=seed.outcomes,
        policy=policy,
    )

    assert result.probe_result is not None
    assert result.main_result is not None
    assert result.probe_pairs == len(result.plans)
    assert 0 < result.main_pairs <= result.probe_pairs
    assert all(outcome.request.evidence_stage.value == "probe" for outcome in result.probe_result.outcomes)
    assert all(outcome.request.evidence_stage.value == "screening" for outcome in result.main_result.outcomes)
    assert all(outcome.samples == policy.main_samples for outcome in result.main_result.outcomes)
    assert controller.phase_time_s["probe"] > 0.0
    assert controller.phase_time_s["screening"] > 0.0
    assert any(event.preparation_reused for event in result.events)
    assert any(len(event.artifact_scope_shape_ids) > 1 for event in result.events)
    assert any(event.promotion_stage == "probe_rejected" for event in result.events)
    assert any(event.success for event in result.events)
    assert sum(item["event"] == "promotion" for item in controller.trace) == len(result.events)


def test_singleton_promotion_is_an_exact_no_op(tmp_path: Path):
    shape = pilot_100_shapes()[0]
    candidate = sample_candidates(1)[0]
    clustering = cluster_shapes(
        [shape],
        ShapeClusteringConfiguration(
            workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
            cluster_count=8,
        ),
    )
    db = EvoTensileDB.connect(
        tmp_path / "singleton.sqlite",
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
            },
            profile=DEFAULT_PROFILE,
        )
    )
    controller = CampaignControllerState(
        shape_ids=(shape.id,),
        time_budget_s=10.0,
        session_started_at=0.0,
    )
    controller.set_clustering(clustering.to_dict())
    seed = evaluator.evaluate([PairRequest(candidate, shape)])
    seed.apply(controller)

    result = execute_promotion_race(
        evaluator,
        controller,
        shapes=[shape],
        clustering=clustering,
        observations=seed.outcomes,
        policy=PromotionPolicy(),
    )

    assert result.plans == ()
    assert result.probe_result is None
    assert result.main_result is None
