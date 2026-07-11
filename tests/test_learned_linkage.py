from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.search.encoding import PARAM_NAMES
from evotensile.search.evidence import load_proposal_evidence_snapshot
from evotensile.search.learned_linkage import (
    DEFAULT_ORDINAL_PARAM_NAMES,
    ScoredGenome,
    evidence_to_scored_genomes,
    hybrid_mi_matrix,
    leader_clusters,
    learn_linkage_models,
    learn_linkage_models_from_snapshot,
    load_candidate_evidence,
    minimum_evidence_for_truncation,
    nearest_linkage_model,
    ordinal_gene_indices,
    select_truncation_pool,
    upgma_fos,
)
from evotensile.shapes import pilot_100_shapes
from tests.helpers import insert_test_benchmark_event, sample_candidates


def _scored(genome: tuple[int, ...], score: float, name: str | None = None) -> ScoredGenome:
    return ScoredGenome(genome=genome, score=score, candidate_hash=name, samples=1)


def test_load_candidate_evidence_uses_shape_local_ranks_and_positive_rows(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(3)
    shapes = pilot_100_shapes()[:2]
    db.register_candidates(candidates)
    db.register_shapes(shapes)
    problem_hash = DEFAULT_PROFILE.problem_type_hash
    protocol_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    insert_test_benchmark_event(
        db,
        shape_id=shapes[0].id,
        candidate_hash=candidates[0].hash,
        run_id="run",
        status="ok",
        problem_type_hash=problem_hash,
        benchmark_protocol_hash=protocol_hash,
        time_us=1.0,
    )
    insert_test_benchmark_event(
        db,
        shape_id=shapes[0].id,
        candidate_hash=candidates[1].hash,
        run_id="run",
        status="ok",
        problem_type_hash=problem_hash,
        benchmark_protocol_hash=protocol_hash,
        time_us=2.0,
    )
    insert_test_benchmark_event(
        db,
        shape_id=shapes[1].id,
        candidate_hash=candidates[1].hash,
        run_id="run",
        status="ok",
        problem_type_hash=problem_hash,
        benchmark_protocol_hash=protocol_hash,
        time_us=1.0,
    )
    insert_test_benchmark_event(
        db,
        shape_id=shapes[1].id,
        candidate_hash=candidates[2].hash,
        run_id="run",
        status="build_failed",
        problem_type_hash=problem_hash,
        benchmark_protocol_hash=protocol_hash,
    )

    snapshot = load_proposal_evidence_snapshot(
        db,
        problem_type_hash=problem_hash,
        benchmark_protocol_hash=protocol_hash,
        shapes=shapes,
    )
    evidence = load_candidate_evidence(
        snapshot,
        shapes=shapes,
        elite_per_shape=8,
    )
    scored = evidence_to_scored_genomes(evidence)

    assert [item.candidate.hash for item in evidence] == [candidates[1].hash, candidates[0].hash]
    assert evidence[0].aggregate_score == 0.5
    assert evidence[0].coverage_fraction == 1.0
    assert evidence[0].unresolved_shape_count == 0
    assert evidence[1].aggregate_score == 0.5
    assert evidence[1].coverage_fraction == 0.5
    assert evidence[1].unresolved_shape_count == 1
    assert scored[0].candidate_hash == candidates[1].hash
    assert scored[0].samples == 2


def test_minimum_evidence_for_truncation_satisfies_selected_sample_floor():
    assert minimum_evidence_for_truncation(truncation_tau=1.0, min_samples=8) == 8
    assert minimum_evidence_for_truncation(truncation_tau=0.75, min_samples=8) == 11
    assert minimum_evidence_for_truncation(truncation_tau=0.5, min_samples=8) == 16


def test_linkage_db_loader_expands_evidence_for_truncation(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "linkage.sqlite")
    db.init()
    candidates = sample_candidates(20, seed=1153)
    shape = pilot_100_shapes()[0]
    problem_hash = DEFAULT_PROFILE.problem_type_hash
    protocol_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([shape])
    for index, candidate in enumerate(candidates):
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="run",
            status="ok",
            problem_type_hash=problem_hash,
            benchmark_protocol_hash=protocol_hash,
            time_us=1.0 + index,
        )

    snapshot = load_proposal_evidence_snapshot(
        db,
        problem_type_hash=problem_hash,
        benchmark_protocol_hash=protocol_hash,
        shapes=[shape],
    )
    _, summary = learn_linkage_models_from_snapshot(
        snapshot,
        shapes=[shape],
        truncation_tau=0.5,
        min_samples=8,
    )

    assert summary.evidence_count == 16
    assert summary.selected_count == 8
    assert summary.fallback_reason is None


def test_truncation_selection_keeps_best_finite_scores():
    rows = [
        _scored((0, 0), 3.0, "slow"),
        _scored((1, 1), 1.0, "best"),
        _scored((2, 2), float("inf"), "failed"),
        _scored((3, 3), 2.0, "mid"),
    ]

    selected = select_truncation_pool(rows, truncation_tau=0.5, min_samples=1)

    assert [row.candidate_hash for row in selected] == ["best"]


def test_truncation_selection_falls_back_when_too_few_survivors():
    rows = [_scored((0, 0), 1.0, "only"), _scored((1, 1), 2.0, "second")]

    assert select_truncation_pool(rows, truncation_tau=0.5, min_samples=2) == []


def test_leader_clustering_separates_structural_basins():
    rows = [
        _scored((0, 0, 0, 0), 1.0, "a0"),
        _scored((0, 0, 0, 1), 1.1, "a1"),
        _scored((3, 3, 3, 3), 1.2, "b0"),
        _scored((3, 3, 3, 2), 1.3, "b1"),
    ]

    clusters = leader_clusters(rows, max_clusters=4, hamming_threshold=1)

    assert sorted(len(cluster) for cluster in clusters) == [2, 2]
    assert {cluster[0].candidate_hash for cluster in clusters} == {"a0", "b0"}


def test_num_elements_per_batch_store_is_nominal_because_zero_is_special():
    assert "NumElementsPerBatchStore" not in DEFAULT_ORDINAL_PARAM_NAMES
    assert PARAM_NAMES.index("NumElementsPerBatchStore") not in ordinal_gene_indices()


def test_hybrid_mi_groups_correlated_ordinal_genes():
    genomes = [(idx, idx, idx % 2) for idx in range(8)]

    matrix = hybrid_mi_matrix(genomes, ordinal_indices=frozenset({0, 1}), ordinal_bins=4)

    assert matrix[0][1] > matrix[0][2]


def test_upgma_stops_when_mi_is_below_floor():
    matrix = [
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]

    fos = upgma_fos(matrix, mi_floor=1e-6)

    assert (0, 1) in fos
    assert all(group != (0, 1, 2) for group in fos)


def test_learn_linkage_models_falls_back_without_enough_evidence():
    rows = [_scored((0, 0), 1.0, "a"), _scored((1, 1), 2.0, "b")]

    models, summary = learn_linkage_models(rows, min_samples=8)

    assert models == []
    assert summary.enabled is False
    assert summary.fallback_reason == "insufficient_validated_evidence"


def test_learn_linkage_models_and_nearest_assignment():
    rows = [
        _scored((0, 0, 0, 0), 1.0, "a0"),
        _scored((0, 0, 0, 1), 1.1, "a1"),
        _scored((0, 0, 1, 0), 1.2, "a2"),
        _scored((3, 3, 3, 3), 1.3, "b0"),
        _scored((3, 3, 3, 2), 1.4, "b1"),
        _scored((3, 3, 2, 3), 1.5, "b2"),
    ]

    models, summary = learn_linkage_models(
        rows,
        truncation_tau=1.0,
        min_samples=4,
        max_clusters=4,
        hamming_threshold=1,
        ordinal_indices=frozenset({0, 1, 2, 3}),
    )

    assert summary.enabled is True
    assert summary.model_count == 2
    assert sorted(model.cluster_size for model in models) == [3, 3]
    nearest = nearest_linkage_model((3, 3, 3, 1), models)
    assert nearest is not None
    assert nearest.leader_candidate_hash == "b0"
