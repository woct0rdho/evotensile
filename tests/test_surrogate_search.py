import random

from evotensile.candidate import Candidate, Shape
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.search.acquisition import propose_candidates
from evotensile.search.evidence import load_proposal_evidence_snapshot
from evotensile.search.family import family_descriptor_counts
from evotensile.search.mechanics import (
    candidate_shape_mechanics,
    mechanical_coverage_tokens,
    mechanical_prior_score,
)
from evotensile.search.surrogate import (
    GridCandidatePrediction,
    ShapePrediction,
    TrainingObservation,
    _marginal_grid_gain_order,
    candidate_shape_features,
    select_surrogate_pool,
    surrogate_model_shape_ids,
)
from evotensile.search_space import macro_tile, random_candidate
from tests.helpers import insert_test_benchmark_event

WORKGROUP_PROCESSOR_COUNT = DEFAULT_PROFILE.workgroup_processor_count
SURROGATE_JOBS = DEFAULT_PROFILE.default_surrogate_jobs


def _evidence(db: EvoTensileDB, shapes: list[Shape]):
    return load_proposal_evidence_snapshot(
        db,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        shapes=shapes,
    )


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

    features = candidate_shape_features(candidate, shape, workgroup_processor_count=WORKGROUP_PROCESSOR_COUNT)
    macro_tile0, macro_tile1 = macro_tile(candidate.canonical_params()["MatrixInstruction"])

    assert features["shape:log2_m"] == 13.0
    tile_area = features["tile:log2_area"]
    assert isinstance(tile_area, (int, float))
    assert tile_area > 0.0
    assert features["tile:m_remainder_fraction"] == (shape.m % macro_tile0) / macro_tile0
    assert features["tile:n_remainder_fraction"] == (shape.n % macro_tile1) / macro_tile1
    assert "gene:MatrixInstruction" in features
    assert "resource:valu_vgpr_lower_bound" in features
    wgp_granularity = features["grid:wgp_granularity"]
    tile_fill_m = features["tile:fill_m"]
    tile_fill_n = features["tile:fill_n"]
    lds_bytes = features["resource:lds_bytes"]
    assert isinstance(wgp_granularity, (int, float))
    assert isinstance(tile_fill_m, (int, float))
    assert isinstance(tile_fill_n, (int, float))
    assert isinstance(lds_bytes, (int, float))
    assert 0.0 < wgp_granularity <= 1.0
    assert 0.0 < tile_fill_m <= 1.0
    assert 0.0 < tile_fill_n <= 1.0
    assert lds_bytes > 0.0

    mechanics = candidate_shape_mechanics(candidate, shape, workgroup_processor_count=WORKGROUP_PROCESSOR_COUNT)
    assert mechanics["workgroups_per_wgp"] > 0.0
    assert mechanics["wgp_rounds"] >= mechanics["workgroups_per_wgp"]
    assert mechanics["arithmetic_intensity"] > 0.0


def test_gfx1151_dispatch_mechanics_use_wgps_not_physical_cus():
    assert DEFAULT_PROFILE.compute_unit_count == 40
    assert DEFAULT_PROFILE.workgroup_processor_count == 20
    assert DEFAULT_PROFILE.compute_units_per_workgroup_processor == 2

    base = random_candidate(random.Random(1151))
    params = base.canonical_params()
    params["GlobalSplitU"] = 1
    candidate = Candidate(params=params)
    macro_tile0, macro_tile1 = macro_tile(params["MatrixInstruction"])
    shape = Shape(macro_tile0 * 40, macro_tile1, 1, params["DepthU"])

    mechanics = candidate_shape_mechanics(
        candidate,
        shape,
        workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
    )

    assert mechanics["workgroups"] == 40.0
    assert mechanics["workgroups_per_wgp"] == 2.0
    assert mechanics["wgp_rounds"] == 2.0
    assert mechanics["wgp_granularity"] == 1.0


def test_surrogate_activation_requires_unique_candidate_variation():
    shapes = [Shape(512, 128, 1, 256), Shape(1024, 1024, 1, 1024)]
    candidate = _shape_candidates(shapes[0], 1000, 1)[0]
    observations = [
        TrainingObservation(
            candidate_hash=candidate.hash,
            shape_id=shape.id,
            features=candidate_shape_features(candidate, shape, workgroup_processor_count=WORKGROUP_PROCESSOR_COUNT),
            log_time=float(index + 1),
        )
        for index, shape in enumerate(shapes)
        for _ in range(24)
    ]

    assert surrogate_model_shape_ids(observations, shapes=shapes, min_evidence=24) == ()


def test_grid_acquisition_preserves_complementary_shape_specialists():
    candidates = _shape_candidates(Shape(512, 128, 1, 256), 2000, 3)
    first_shape = "m512_n128_b1_k256"
    second_shape = "m1024_n1024_b1_k1024"
    predictions = [
        GridCandidatePrediction(
            candidate=candidates[0],
            by_shape=(
                ShapePrediction(first_shape, 8.0, 0.0, 10.0),
                ShapePrediction(second_shape, 11.0, 0.0, 10.0),
            ),
        ),
        GridCandidatePrediction(
            candidate=candidates[1],
            by_shape=(
                ShapePrediction(first_shape, 11.0, 0.0, 10.0),
                ShapePrediction(second_shape, 8.0, 0.0, 10.0),
            ),
        ),
        GridCandidatePrediction(
            candidate=candidates[2],
            by_shape=(
                ShapePrediction(first_shape, 9.4, 0.0, 10.0),
                ShapePrediction(second_shape, 9.4, 0.0, 10.0),
            ),
        ),
    ]

    ordered = _marginal_grid_gain_order(predictions)

    assert {candidate.hash for candidate in ordered[:2]} == {candidates[0].hash, candidates[1].hash}


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
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="queried",
            status="ok",
            problem_type_hash=problem_hash,
            benchmark_protocol_hash=protocol_hash,
            time_us=time_us,
        )
    pool = _shape_candidates(shape, 5000, 160)

    selected = select_surrogate_pool(
        pool,
        evidence=_evidence(db, [shape]),
        shapes=[shape],
        count=32,
        seed=20260710,
        min_evidence=24,
        surrogate_jobs=SURROGATE_JOBS,
        workgroup_processor_count=WORKGROUP_PROCESSOR_COUNT,
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
        evidence=_evidence(db, [shape]),
        shapes=[shape],
        count=24,
        seed=20260710,
        min_evidence=24,
        surrogate_jobs=SURROGATE_JOBS,
        workgroup_processor_count=WORKGROUP_PROCESSOR_COUNT,
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

    assert (
        candidate_shape_mechanics(tiny, shape, workgroup_processor_count=WORKGROUP_PROCESSOR_COUNT)[
            "dispatch_efficiency"
        ]
        == 0.0
    )
    assert (
        candidate_shape_mechanics(broad, shape, workgroup_processor_count=WORKGROUP_PROCESSOR_COUNT)[
            "dispatch_efficiency"
        ]
        > 0.8
    )
    assert mechanical_prior_score(
        broad,
        shape,
        workgroup_processor_count=WORKGROUP_PROCESSOR_COUNT,
    ) > mechanical_prior_score(
        tiny,
        shape,
        workgroup_processor_count=WORKGROUP_PROCESSOR_COUNT,
    )


def test_covering_cold_start_increases_mechanical_token_coverage(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "surrogate.sqlite")
    db.init()
    shape = Shape(8192, 8192, 1, 8192)
    pool = _shape_candidates(shape, 12_000, 256)

    baseline = select_surrogate_pool(
        pool,
        evidence=_evidence(db, [shape]),
        shapes=[shape],
        count=48,
        seed=20260710,
        min_evidence=24,
        surrogate_jobs=SURROGATE_JOBS,
        workgroup_processor_count=WORKGROUP_PROCESSOR_COUNT,
    )
    covering = select_surrogate_pool(
        pool,
        evidence=_evidence(db, [shape]),
        shapes=[shape],
        count=48,
        seed=20260710,
        min_evidence=24,
        covering_cold_start=True,
        surrogate_jobs=SURROGATE_JOBS,
        workgroup_processor_count=WORKGROUP_PROCESSOR_COUNT,
    )

    baseline_tokens = set().union(
        *(
            mechanical_coverage_tokens(candidate, shape, workgroup_processor_count=WORKGROUP_PROCESSOR_COUNT)
            for candidate in baseline
        )
    )
    covering_tokens = set().union(
        *(
            mechanical_coverage_tokens(candidate, shape, workgroup_processor_count=WORKGROUP_PROCESSOR_COUNT)
            for candidate in covering
        )
    )
    baseline_instructions = {tuple(candidate.canonical_params()["MatrixInstruction"]) for candidate in baseline}
    covering_instructions = {tuple(candidate.canonical_params()["MatrixInstruction"]) for candidate in covering}
    assert len(covering_tokens) > len(baseline_tokens)
    assert len(covering_instructions) >= len(baseline_instructions)


def test_scheduler_surrogate_multiplier_preserves_cold_measurement_budget(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "surrogate.sqlite")
    db.init()
    shape = Shape(1024, 1024, 1, 1024)

    proposal = propose_candidates(
        db,
        proposal="family-qd",
        num_random=3,
        local_count=0,
        de_count=0,
        gomea_count=0,
        target_shapes=[shape],
        surrogate_pool_multiplier=2,
        covering_cold_start=True,
        seed=20260710,
    )

    assert len(proposal.generated) == 6
    assert len(proposal.selected) == 3
    assert {candidate.source for candidate in proposal.selected} == {"random"}
    assert len(family_descriptor_counts(proposal.selected)) >= 2
