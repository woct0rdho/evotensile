import random

from evotensile.candidate import Shape
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.scheduler import propose_candidates
from evotensile.search.family import family_descriptor_counts
from evotensile.search.surrogate import candidate_shape_features, select_surrogate_pool
from evotensile.search_space import macro_tile, random_candidate


def _shape_candidates(shape: Shape, start_seed: int, count: int):
    out = {}
    seed = start_seed
    while len(out) < count:
        candidate = random_candidate(random.Random(seed), target_shapes=[shape])
        out[candidate.hash] = candidate
        seed += 1
    return list(out.values())


def test_candidate_shape_features_include_generic_mechanics():
    shape = Shape(8192, 4096, 1, 2048)
    candidate = random_candidate(random.Random(1151), target_shapes=[shape])

    features = candidate_shape_features(candidate, shape)
    macro_tile0, macro_tile1 = macro_tile(candidate.canonical_params()["MatrixInstruction"])

    assert features["shape:log2_m"] == 13.0
    tile_area = features["tile:log2_area"]
    assert isinstance(tile_area, (int, float))
    assert tile_area > 0.0
    assert features["tile:m_remainder_fraction"] == (shape.m % macro_tile0) / macro_tile0
    assert features["tile:n_remainder_fraction"] == (shape.n % macro_tile1) / macro_tile1
    assert "gene:MatrixInstruction" in features
    assert "resource:valu_vgpr_lower_bound" in features


def test_surrogate_pool_learns_synthetic_tlds_performance_from_queried_rows(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "surrogate.sqlite")
    db.init()
    shape = Shape(8192, 8192, 1, 8192)
    problem_hash = DEFAULT_PROFILE.problem_type_hash
    protocol_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    training = _shape_candidates(shape, 1000, 64)
    db.register_candidates(training)
    db.register_shapes([shape])
    for candidate in training:
        params = candidate.canonical_params()
        time_us = 100.0 if params["TransposeLDS"] == 0 else 400.0
        db.insert_evaluation(
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="queried",
            status="ok",
            problem_type_hash=problem_hash,
            benchmark_protocol_hash=protocol_hash,
            time_us=time_us,
            validation="PASSED",
        )
    pool = _shape_candidates(shape, 5000, 160)

    selected = select_surrogate_pool(
        pool,
        db=db,
        problem_type_hash=problem_hash,
        benchmark_protocol_hash=protocol_hash,
        shapes=[shape],
        count=32,
        seed=20260710,
        min_evidence=24,
    )

    selected_tlds0 = sum(candidate.canonical_params()["TransposeLDS"] == 0 for candidate in selected)
    pool_tlds0_fraction = sum(candidate.canonical_params()["TransposeLDS"] == 0 for candidate in pool) / len(pool)
    assert len(selected) == 32
    assert selected_tlds0 >= 20
    assert selected_tlds0 / len(selected) > pool_tlds0_fraction


def test_surrogate_pool_falls_back_to_family_diversity_without_evidence(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "surrogate.sqlite")
    db.init()
    shape = Shape(8192, 8192, 1, 8192)
    pool = _shape_candidates(shape, 9000, 96)

    selected = select_surrogate_pool(
        pool,
        db=db,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        shapes=[shape],
        count=24,
        seed=20260710,
        min_evidence=24,
    )

    assert len(selected) == 24
    assert len(family_descriptor_counts(selected)) >= 12


def test_scheduler_surrogate_multiplier_preserves_cold_measurement_budget(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "surrogate.sqlite")
    db.init()
    shape = Shape(8192, 8192, 1, 8192)

    proposed = propose_candidates(
        db,
        proposal="family-qd",
        num_random=16,
        local_count=8,
        de_count=4,
        gomea_count=12,
        target_shapes=[shape],
        surrogate_pool_multiplier=4,
        seed=20260710,
    )

    assert len(proposed) == 16
    assert {candidate.source for candidate in proposed} == {"random"}
    assert len(family_descriptor_counts(proposed)) >= 8
