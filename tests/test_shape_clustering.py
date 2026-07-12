from pathlib import Path

import pytest

from evotensile.campaign.baselines import (
    characterize_representative_promotions,
    evaluate_global_candidate_dense_baseline,
    evaluate_representative_first_baseline,
)
from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.evaluator import ReplayEvaluator
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.search.replay import ExactOracleReplayState, OracleRecord
from evotensile.search.shape_clustering import ShapeClusteringConfiguration, cluster_shapes
from evotensile.shapes import pilot_100_shapes
from tests.helpers import sample_candidates


def _configuration(*, count: int | None = None, threshold: float | None = None):
    return ShapeClusteringConfiguration(
        workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
        cluster_count=count,
        distance_threshold=threshold,
        macro_tile_family_count=6,
    )


def test_fixed_count_clustering_is_deterministic_and_uses_member_medoids():
    shapes = pilot_100_shapes()[:12]

    first = cluster_shapes(shapes, _configuration(count=4))
    reordered = cluster_shapes(list(reversed(shapes)), _configuration(count=4))

    assert first.to_dict() == reordered.to_dict()
    assert len(first.clusters) == 4
    assert {shape_id for cluster in first.clusters for shape_id in cluster.shape_ids} == {shape.id for shape in shapes}
    assert all(cluster.medoid_shape_id in cluster.shape_ids for cluster in first.clusters)
    assert all(cluster.distances_to_medoid[cluster.medoid_shape_id] == 0.0 for cluster in first.clusters)
    feature_names = next(iter(first.descriptors.values())).features
    assert "shape:arithmetic_intensity" in feature_names
    assert "shape:reduction_depth" in feature_names
    assert any(name.endswith(":wgp_granularity") for name in feature_names)
    assert any(name.endswith(":fill_m") for name in feature_names)


def test_threshold_and_singleton_clustering_have_deterministic_degenerations():
    shapes = pilot_100_shapes()[:8]
    separated = cluster_shapes(shapes, _configuration(threshold=0.0))
    merged = cluster_shapes(shapes, _configuration(threshold=1000.0))

    assert len(separated.clusters) == len(shapes)
    assert len(merged.clusters) == 1

    singleton = shapes[:1]
    fixed = cluster_shapes(singleton, _configuration(count=7))
    threshold = cluster_shapes(singleton, _configuration(threshold=0.0))
    assert fixed.medoid_shape_ids == threshold.medoid_shape_ids == (singleton[0].id,)
    assert fixed.cluster_by_shape == threshold.cluster_by_shape == {singleton[0].id: "cluster_000"}


def test_controller_persists_and_validates_shape_clustering():
    shapes = pilot_100_shapes()[:5]
    clustering = cluster_shapes(shapes, _configuration(count=2))
    controller = CampaignControllerState(
        shape_ids=tuple(shape.id for shape in shapes),
        time_budget_s=10.0,
        session_started_at=0.0,
    )

    controller.set_clustering(clustering.to_dict())
    checkpoint = controller.to_checkpoint(now=0.0)
    restored = CampaignControllerState.from_checkpoint(checkpoint, session_started_at=20.0)

    assert restored.clustering == clustering.to_dict()
    invalid = clustering.to_dict()
    invalid["shape_ids"] = list(clustering.shape_ids[:-1])
    with pytest.raises(ValueError, match="exact registered shape set"):
        controller.set_clustering(invalid)


def _replay_evaluator(
    tmp_path: Path,
    *,
    name: str,
    shapes,
    oracle,
) -> ReplayEvaluator:
    db = EvoTensileDB.connect(
        tmp_path / f"{name}.sqlite",
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    state = ExactOracleReplayState(
        db=db,
        shapes=shapes,
        oracle=oracle,
        profile=DEFAULT_PROFILE,
        source_ref=name,
    )
    return ReplayEvaluator(state, prepare_seconds_per_candidate=0.1)


def test_dense_and_representative_replay_baselines_use_explicit_pair_sets(tmp_path: Path):
    shapes = pilot_100_shapes()[:4]
    candidates = sample_candidates(2)
    clustering = cluster_shapes(shapes, _configuration(count=2))
    specialist_shape = next(
        shape_id
        for cluster in clustering.clusters
        for shape_id in cluster.shape_ids
        if shape_id != cluster.medoid_shape_id
    )
    oracle = {}
    for shape in shapes:
        for index, candidate in enumerate(candidates):
            performance = 100.0 if index == 0 else 90.0
            if shape.id == specialist_shape and index == 1:
                performance = 200.0
            oracle[(shape.id, candidate.hash)] = OracleRecord(
                candidate=candidate,
                status="ok",
                screening_gflops=performance,
                order=0.01,
            )

    representative_controller = CampaignControllerState(
        shape_ids=tuple(shape.id for shape in shapes),
        time_budget_s=10.0,
        session_started_at=0.0,
    )
    representative_controller.set_clustering(clustering.to_dict())
    representative = evaluate_representative_first_baseline(
        _replay_evaluator(tmp_path, name="representative", shapes=shapes, oracle=oracle),
        representative_controller,
        candidates=candidates,
        shapes=shapes,
        clustering=clustering,
    )
    assert representative.requested_pairs == len(candidates) * len(clustering.clusters)
    assert len(representative_controller.queried_pairs) == representative.requested_pairs
    assert set(representative_controller.incumbents) == set(clustering.medoid_shape_ids)

    dense_controller = CampaignControllerState(
        shape_ids=tuple(shape.id for shape in shapes),
        time_budget_s=10.0,
        session_started_at=0.0,
    )
    dense = evaluate_global_candidate_dense_baseline(
        _replay_evaluator(tmp_path, name="dense", shapes=shapes, oracle=oracle),
        dense_controller,
        candidates=candidates,
        shapes=shapes,
    )
    assert dense.requested_pairs == len(candidates) * len(shapes)
    assert len(dense_controller.queried_pairs) == dense.requested_pairs
    assert set(dense_controller.incumbents) == {shape.id for shape in shapes}

    diagnostics = characterize_representative_promotions(
        clustering,
        oracle,
        candidate_hashes=[candidate.hash for candidate in candidates],
        tolerance_fraction=0.05,
    )
    assert diagnostics.medoid_pairs == representative.requested_pairs
    assert diagnostics.missed_specialists >= 1
    assert diagnostics.promotion_precision is not None
    assert diagnostics.worst_regret_fraction == pytest.approx(1.0)
