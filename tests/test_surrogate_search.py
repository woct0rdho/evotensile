import random

from evotensile.candidate import Candidate, Shape
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.scheduler import propose_candidates
from evotensile.search.family import family_descriptor_counts
from evotensile.search.mechanics import (
    candidate_shape_mechanics,
    mechanical_coverage_tokens,
    mechanical_prior_score,
)
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
    cu_granularity = features["grid:cu_granularity"]
    tile_fill_m = features["tile:fill_m"]
    tile_fill_n = features["tile:fill_n"]
    lds_bytes = features["resource:lds_bytes"]
    assert isinstance(cu_granularity, (int, float))
    assert isinstance(tile_fill_m, (int, float))
    assert isinstance(tile_fill_n, (int, float))
    assert isinstance(lds_bytes, (int, float))
    assert 0.0 < cu_granularity <= 1.0
    assert 0.0 < tile_fill_m <= 1.0
    assert 0.0 < tile_fill_n <= 1.0
    assert lds_bytes > 0.0

    mechanics = candidate_shape_mechanics(candidate, shape)
    assert mechanics["tiles_per_cu"] > 0.0
    assert mechanics["cu_rounds"] >= mechanics["tiles_per_cu"]
    assert mechanics["arithmetic_intensity"] > 0.0


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


def test_mechanical_prior_penalizes_single_instruction_workgroups():
    shape = Shape(8192, 8192, 1, 8192)
    base = _shape_candidates(shape, 11_000, 1)[0]
    tiny_params = base.canonical_params()
    tiny_params["MatrixInstruction"] = [16, 16, 16, 1, 1, 1, 1, 1, 1]
    broad_params = base.canonical_params()
    broad_params["MatrixInstruction"] = [16, 16, 16, 1, 1, 4, 4, 2, 2]
    tiny = Candidate(params=tiny_params)
    broad = Candidate(params=broad_params)

    assert candidate_shape_mechanics(tiny, shape)["dispatch_efficiency"] == 0.0
    assert candidate_shape_mechanics(broad, shape)["dispatch_efficiency"] > 0.8
    assert mechanical_prior_score(broad, shape) > mechanical_prior_score(tiny, shape)


def test_covering_cold_start_increases_mechanical_token_coverage(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "surrogate.sqlite")
    db.init()
    shape = Shape(8192, 8192, 1, 8192)
    pool = _shape_candidates(shape, 12_000, 256)

    baseline = select_surrogate_pool(
        pool,
        db=db,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        shapes=[shape],
        count=48,
        seed=20260710,
        min_evidence=24,
    )
    covering = select_surrogate_pool(
        pool,
        db=db,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        shapes=[shape],
        count=48,
        seed=20260710,
        min_evidence=24,
        covering_cold_start=True,
    )

    baseline_tokens = set().union(*(mechanical_coverage_tokens(candidate, shape) for candidate in baseline))
    covering_tokens = set().union(*(mechanical_coverage_tokens(candidate, shape) for candidate in covering))
    baseline_instructions = {tuple(candidate.canonical_params()["MatrixInstruction"]) for candidate in baseline}
    covering_instructions = {tuple(candidate.canonical_params()["MatrixInstruction"]) for candidate in covering}
    assert len(covering_tokens) > len(baseline_tokens)
    assert len(covering_instructions) >= len(baseline_instructions)


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
        covering_cold_start=True,
        seed=20260710,
    )

    assert len(proposed) == 16
    assert {candidate.source for candidate in proposed} == {"random"}
    assert len(family_descriptor_counts(proposed)) >= 8
