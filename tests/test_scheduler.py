import json
from pathlib import Path
from textwrap import dedent

from evotensile.candidate import Shape
from evotensile.cli import main as cli_main
from evotensile.database import EvoTensileDB, ValidationInsert
from evotensile.profile import DEFAULT_PROFILE
from evotensile.scheduler import (
    default_prepare_workers,
    detect_underperforming_shapes,
    execute_schedule,
    plan_batches,
    production_candidate_batch_size,
    propose_candidates,
    repair_seed_candidates,
)
from evotensile.search_space import cheap_constraints, make_candidate
from evotensile.shapes import pilot_100_shapes
from evotensile.tensilelite_diagnostics import DiagnosticRecord, DiagnosticRunResult
from tests.helpers import REFERENCE_CANDIDATE, sample_candidates


def _time_us_for_gflops(shape: Shape, gflops: float) -> float:
    return 2.0 * shape.m * shape.n * shape.batch * shape.k / (gflops * 1e9) * 1e6


def _record_validation(db: EvoTensileDB, shape: Shape, candidate_hash: str, detail: str = "PASSED") -> None:
    db.insert_validations(
        [
            ValidationInsert(
                shape_id=shape.id,
                candidate_hash=candidate_hash,
                run_id="cached_validation",
                status="passed",
                problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
                validation_protocol_hash=DEFAULT_PROFILE.default_protocol.validation_protocol_hash(),
                detail=detail,
            )
        ]
    )


def test_production_candidate_batch_size_maximizes_size_while_saturating_workers():
    assert (
        production_candidate_batch_size(
            candidate_count=64,
            shape_count=100,
            shape_batch_size=100,
            prepare_workers=32,
            max_candidate_batch_size=32,
        )
        == 2
    )
    assert (
        production_candidate_batch_size(
            candidate_count=64,
            shape_count=100,
            shape_batch_size=25,
            prepare_workers=32,
            max_candidate_batch_size=32,
        )
        == 9
    )
    assert (
        production_candidate_batch_size(
            candidate_count=16,
            shape_count=1,
            shape_batch_size=100,
            prepare_workers=32,
            max_candidate_batch_size=32,
        )
        == 1
    )


def test_plan_batches_skips_cached_ok_pairs(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(2)
    shapes = pilot_100_shapes()[:2]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.insert_evaluation(
        shape_id=shapes[0].id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=1.0,
        validation="PASSED prior_validation",
    )

    batches = plan_batches(
        db,
        shapes=shapes,
        candidates=candidates,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        validation_protocol_hash=DEFAULT_PROFILE.default_protocol.validation_protocol_hash(),
        candidate_batch_size=2,
        shape_batch_size=2,
    )

    assert len(batches) == 2
    assert sum(batch.missing_pairs for batch in batches) == 3
    assert sum(batch.nominal_pairs for batch in batches) == 3
    assert all(batch.extra_pairs == 0 for batch in batches)
    assert [candidate.hash for candidate in batches[0].candidates] == [candidates[1].hash]
    assert [shape.id for shape in batches[0].shapes] == [shapes[0].id]
    assert [candidate.hash for candidate in batches[1].candidates] == [candidate.hash for candidate in candidates]
    assert [shape.id for shape in batches[1].shapes] == [shapes[1].id]


def test_plan_batches_requests_only_missing_sample_count(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(1)
    shapes = pilot_100_shapes()[:1]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.insert_evaluation(
        shape_id=shapes[0].id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=1.0,
        validation="PASSED prior_validation",
    )
    _record_validation(db, shapes[0], candidates[0].hash)

    batches = plan_batches(
        db,
        shapes=shapes,
        candidates=candidates,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        validation_protocol_hash=DEFAULT_PROFILE.default_protocol.validation_protocol_hash(),
        min_samples=3,
        candidate_batch_size=1,
        shape_batch_size=1,
    )

    assert len(batches) == 1
    assert batches[0].missing_pairs == 1
    assert batches[0].samples_per_pair == 2
    assert batches[0].missing_samples == 2
    assert not batches[0].requires_validation


def test_plan_batches_reuses_detailed_hipblaslt_validation_evidence(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(1)
    shapes = pilot_100_shapes()[:1]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.insert_evaluation(
        shape_id=shapes[0].id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=1.0,
        validation="PASSED prior_validation",
    )
    _record_validation(
        db,
        shapes[0],
        candidates[0].hash,
        detail="PASSED checked=1048576 stride=1 backend=hipblaslt_gpu_compare",
    )

    batches = plan_batches(
        db,
        shapes=shapes,
        candidates=candidates,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        validation_protocol_hash=DEFAULT_PROFILE.default_protocol.validation_protocol_hash(),
        min_samples=3,
        candidate_batch_size=1,
        shape_batch_size=1,
    )

    assert len(batches) == 1
    assert not batches[0].requires_validation


def test_plan_batches_requires_validation_without_prior_validation_evidence(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(1)
    shapes = pilot_100_shapes()[:1]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.insert_evaluation(
        shape_id=shapes[0].id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=1.0,
        validation="NO_CHECK",
    )

    batches = plan_batches(
        db,
        shapes=shapes,
        candidates=candidates,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        validation_protocol_hash=DEFAULT_PROFILE.default_protocol.validation_protocol_hash(),
        min_samples=3,
        candidate_batch_size=1,
        shape_batch_size=1,
    )

    assert len(batches) == 1
    assert batches[0].requires_validation


def test_plan_batches_skips_reusable_negative_cache_entries(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(2)
    shapes = pilot_100_shapes()[:1]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.insert_evaluation(
        shape_id=shapes[0].id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="rejected",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )
    db.insert_evaluation(
        shape_id=shapes[0].id,
        candidate_hash=candidates[1].hash,
        run_id="cached",
        status="build_failed",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )

    batches = plan_batches(
        db,
        shapes=shapes,
        candidates=candidates,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        validation_protocol_hash=DEFAULT_PROFILE.default_protocol.validation_protocol_hash(),
        candidate_batch_size=2,
        shape_batch_size=1,
    )

    assert batches == []


def test_local_proposal_mutates_cached_elites(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(2)
    shapes = pilot_100_shapes()[:1]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes(shapes)
    db.insert_evaluation(
        shape_id=shapes[0].id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=10.0,
        validation="PASSED prior_validation",
    )

    proposed = propose_candidates(
        db,
        proposal="local",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        local_count=1,
        elite_count=1,
        seed=7,
    )

    mutations = [candidate for candidate in proposed if candidate.source == "mutation"]
    assert len(mutations) == 1
    assert mutations[0].parent_hashes == (candidates[0].hash,)
    assert {candidate.source for candidate in proposed} == {"mutation"}


def test_exact_shape_transfer_seeds_cached_winner(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(2)
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([shape])
    db.insert_evaluation(
        shape_id=shape.id,
        candidate_hash=candidates[1].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=1.0,
        validation="PASSED",
    )

    proposed = propose_candidates(
        db,
        proposal="seed-random-gomea",
        num_random=0,
        gomea_count=0,
        target_shapes=[shape],
        transfer_shape_count=1,
        transfer_per_shape=1,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )

    transfer = [candidate for candidate in proposed if candidate.source == "transfer"]
    assert [candidate.hash for candidate in transfer] == [candidates[1].hash]


def test_nearest_shape_transfer_seeds_cached_winners(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(3)
    target = pilot_100_shapes()[0]
    near_shape = pilot_100_shapes()[1]
    far_shape = pilot_100_shapes()[-1]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([target, near_shape, far_shape])
    for shape, candidate, time_us in ((near_shape, candidates[1], 5.0), (far_shape, candidates[2], 3.0)):
        db.insert_evaluation(
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=time_us,
            validation="PASSED prior_validation",
        )

    proposed = propose_candidates(
        db,
        proposal="seed-random-gomea",
        num_random=0,
        gomea_count=0,
        target_shapes=[target],
        transfer_shape_count=1,
        transfer_per_shape=1,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )

    transfer = [candidate for candidate in proposed if candidate.source == "transfer"]
    assert [candidate.hash for candidate in transfer] == [candidates[1].hash]
    assert transfer[0].parent_hashes == (candidates[1].hash,)


def test_detect_underperforming_shapes_flags_local_envelope_gap(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(4)
    shapes = [
        Shape(m=512, n=512, batch=1, k=512),
        Shape(m=640, n=512, batch=1, k=512),
        Shape(m=768, n=512, batch=1, k=512),
        Shape(m=896, n=512, batch=1, k=512),
        Shape(m=1024, n=512, batch=1, k=512),
    ]
    target = shapes[2]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes(shapes)
    for shape in shapes:
        candidate = candidates[0] if shape == target else candidates[1]
        gflops = 800.0 if shape == target else 1000.0
        db.insert_evaluation(
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=_time_us_for_gflops(shape, gflops),
            validation="PASSED",
        )

    outliers = detect_underperforming_shapes(
        db,
        shapes=shapes,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        min_samples=1,
        neighbor_count=4,
        envelope_quantile=0.75,
        threshold_pct=10.0,
    )

    assert [outlier.shape.id for outlier in outliers] == [target.id]
    assert outliers[0].candidate_hash == candidates[0].hash
    assert outliers[0].predicted_neighbor_gflops > 990.0
    assert outliers[0].residual_pct > 20.0


def test_repair_seed_candidates_include_neighbor_top_candidates(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(4)
    target = Shape(m=768, n=512, batch=1, k=512)
    neighbor = Shape(m=896, n=512, batch=1, k=512)
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([target, neighbor])
    db.insert_evaluation(
        shape_id=target.id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=_time_us_for_gflops(target, 800.0),
        validation="PASSED",
    )
    for candidate, gflops in ((candidates[1], 1000.0), (candidates[2], 950.0), (candidates[3], 900.0)):
        db.insert_evaluation(
            shape_id=neighbor.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=_time_us_for_gflops(neighbor, gflops),
            validation="PASSED",
        )
    outlier = detect_underperforming_shapes(
        db,
        shapes=[target],
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        min_samples=1,
        neighbor_count=1,
        envelope_quantile=0.75,
        threshold_pct=10.0,
    )[0]

    seeds = repair_seed_candidates(
        db,
        outliers=[outlier],
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        min_samples=1,
        neighbor_per_shape=2,
    )

    assert [seed.source for seed in seeds] == ["repair-transfer", "repair-transfer", "repair-transfer"]
    assert [seed.parent_hashes[0] for seed in seeds] == [
        candidates[0].hash,
        candidates[1].hash,
        candidates[2].hash,
    ]


def test_repair_outliers_cli_writes_metadata(tmp_path: Path):
    db_path = tmp_path / "sched.sqlite"
    output_dir = tmp_path / "repair"
    db = EvoTensileDB.connect(db_path)
    db.init()
    candidates = sample_candidates(3)
    shapes = [
        Shape(m=512, n=512, batch=1, k=512),
        Shape(m=768, n=512, batch=1, k=512),
        Shape(m=1024, n=512, batch=1, k=512),
    ]
    target = shapes[1]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes(shapes)
    for shape in shapes:
        candidate = candidates[0] if shape == target else candidates[1]
        gflops = 700.0 if shape == target else 1000.0
        db.insert_evaluation(
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=_time_us_for_gflops(shape, gflops),
            validation="PASSED",
        )

    rc = cli_main(
        [
            "repair-outliers",
            "--db",
            str(db_path),
            "--output-dir",
            str(output_dir),
            "--shapes",
            target.id.replace("m", "", 1).replace("_n", ",").replace("_b", ",").replace("_k", ","),
            "--outlier-min-samples",
            "1",
            "--neighbor-count",
            "2",
            "--neighbor-per-shape",
            "1",
            "--num-random",
            "0",
            "--gomea-count",
            "0",
            "--local-count",
            "0",
            "--de-count",
            "0",
            "--dry-run",
        ]
    )

    assert rc == 0
    metadata = json.loads((output_dir / "repair_metadata.json").read_text(encoding="utf-8"))
    assert metadata["outliers"][0]["shape_id"] == target.id
    assert metadata["repair_seed_candidates"] == 2
    assert metadata["planned_missing_pairs"] >= 1
    assert metadata["executed_batches"] == []


def test_exact_shape_elites_disable_nearest_shape_transfer(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(2)
    target = pilot_100_shapes()[0]
    exact_shape = pilot_100_shapes()[1]
    transfer_shape = pilot_100_shapes()[2]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([target, exact_shape, transfer_shape])
    db.insert_evaluation(
        shape_id=exact_shape.id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=1.0,
        validation="PASSED prior_validation",
    )
    db.insert_evaluation(
        shape_id=transfer_shape.id,
        candidate_hash=candidates[1].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=0.5,
        validation="PASSED prior_validation",
    )

    proposed = propose_candidates(
        db,
        proposal="gomea",
        gomea_count=4,
        target_shapes=[target],
        shape_id=exact_shape.id,
        transfer_shape_count=4,
        transfer_per_shape=1,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )

    assert all(candidate.source != "transfer" for candidate in proposed)
    assert {candidates[0].hash} & {parent for candidate in proposed for parent in candidate.parent_hashes}
    assert candidates[1].hash not in {parent for candidate in proposed for parent in candidate.parent_hashes}


def test_gomea_proposal_uses_learned_linkage_by_default_when_evidence_exists(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(6)
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([shape])
    for idx, candidate in enumerate(candidates):
        db.insert_evaluation(
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=1.0 + idx,
            validation="PASSED",
        )

    proposed = propose_candidates(
        db,
        proposal="gomea",
        gomea_count=4,
        elite_count=6,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        linkage_min_samples=3,
        linkage_truncation_tau=1.0,
    )

    assert proposed
    assert all(candidate.source == "gomea" for candidate in proposed)
    assert all(cheap_constraints(candidate.canonical_params()) for candidate in proposed)


def test_gomea_proposal_accepts_learned_linkage_from_db_evidence(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(6)
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([shape])
    for idx, candidate in enumerate(candidates):
        db.insert_evaluation(
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=1.0 + idx,
            validation="PASSED",
        )

    proposed = propose_candidates(
        db,
        proposal="gomea",
        gomea_count=4,
        elite_count=6,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        learned_linkage=True,
        linkage_min_samples=3,
        linkage_truncation_tau=1.0,
    )

    assert proposed
    assert all(candidate.source == "gomea" for candidate in proposed)
    assert all(cheap_constraints(candidate.canonical_params()) for candidate in proposed)


def test_evolutionary_proposal_uses_cached_elites(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(4)
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([shape])
    for idx, candidate in enumerate(candidates):
        db.insert_evaluation(
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=1.0 + idx,
            validation="PASSED prior_validation",
        )

    proposed = propose_candidates(
        db,
        proposal="evolutionary",
        num_random=2,
        local_count=2,
        de_count=2,
        gomea_count=2,
        elite_count=4,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        seed=7,
    )

    sources = {candidate.source for candidate in proposed}
    assert {"random", "mutation", "de", "gomea"} & sources == {"random", "mutation", "de", "gomea"}
    assert "seed" not in sources


def test_random_proposal_does_not_include_fixed_controls(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()

    proposed = propose_candidates(db, proposal="seed-random", num_random=12, seed=1151)

    assert {candidate.source for candidate in proposed} == {"random"}
    assert REFERENCE_CANDIDATE.hash not in {candidate.hash for candidate in proposed}


def test_random_proposals_respect_target_shape_rules(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    shape = Shape(m=8192, n=8192, batch=1, k=8192)

    proposed = propose_candidates(db, proposal="random", num_random=16, seed=20260701, target_shapes=[shape])

    assert len(proposed) == 16
    assert all(cheap_constraints(candidate.canonical_params(), shape=shape) for candidate in proposed)


def test_non_random_proposals_return_empty_without_evidence(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()

    for proposal in ("local", "de", "gomea"):
        proposed = propose_candidates(
            db,
            proposal=proposal,
            local_count=64,
            de_count=64,
            gomea_count=64,
            seed=1151,
        )

        assert proposed == []


def test_mixed_random_proposals_do_not_use_random_as_evolution_parents(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()

    for proposal in ("seed-random-local", "seed-random-de", "seed-random-gomea", "evolutionary"):
        proposed = propose_candidates(
            db,
            proposal=proposal,
            num_random=4,
            local_count=64,
            de_count=64,
            gomea_count=64,
            seed=1151,
        )

        assert {candidate.source for candidate in proposed} == {"random"}


def test_family_qd_cold_start_uses_balanced_random_family_coverage(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    shape = Shape(m=8192, n=8192, batch=1, k=8192)

    proposed = propose_candidates(
        db,
        proposal="family-qd",
        num_random=16,
        local_count=8,
        de_count=4,
        gomea_count=12,
        target_shapes=[shape],
        seed=819200,
    )

    assert {candidate.source for candidate in proposed} == {"random"}
    assert len(proposed) == 16
    assert {candidate.params["TransposeLDS"] for candidate in proposed} == {0, 2}


def test_family_qd_proposal_preserves_family_archive_leaders(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(4, seed=1151)
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([shape])
    for idx, candidate in enumerate(candidates):
        db.insert_evaluation(
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=1.0 + idx,
            validation="PASSED",
        )

    proposed = propose_candidates(
        db,
        proposal="family-qd",
        num_random=3,
        local_count=2,
        de_count=0,
        gomea_count=2,
        elite_count=4,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        target_shapes=[shape],
        seed=1151,
    )

    proposed_hashes = {candidate.hash for candidate in proposed}
    assert candidates[0].hash in proposed_hashes
    assert "random" in {candidate.source for candidate in proposed}
    assert len(proposed_hashes) == len(proposed)


def test_family_qd_adaptive_operators_use_separate_semantic_arms(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(12, seed=20260710)
    shape = Shape(m=8192, n=8192, batch=1, k=8192)
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([shape])
    for index, candidate in enumerate(candidates):
        db.insert_evaluation(
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=100.0 + index,
            validation="PASSED",
        )

    proposed = propose_candidates(
        db,
        proposal="family-qd",
        num_random=0,
        local_count=4,
        de_count=4,
        gomea_count=8,
        elite_count=12,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        target_shapes=[shape],
        adaptive_operators=True,
        adaptive_group_credit=True,
        micro_exhaustive_neighborhoods=True,
        adaptive_donor_selection=True,
        seed=20260711,
    )

    parent_hashes = {candidate.hash for candidate in candidates}
    generated_sources = {candidate.source for candidate in proposed if candidate.hash not in parent_hashes}
    assert "mutation" not in generated_sources
    assert "gomea" not in generated_sources
    assert {"semantic-mutation", "de", "gomea-neighborhood", "gomea-mixing"} <= generated_sources
    metadata_by_source = {
        candidate.source: candidate.proposal_metadata for candidate in proposed if candidate.hash not in parent_hashes
    }
    assert "semantic_group" in metadata_by_source["semantic-mutation"]
    assert metadata_by_source["gomea-neighborhood"]["enumerated_neighborhood"] is True
    assert metadata_by_source["gomea-mixing"]["donor_mode"] in {"quality", "diverse", "random"}


def test_execute_schedule_records_shape_rule_rejection_without_build(tmp_path: Path):
    fake_tensile = tmp_path / "fail_if_called.py"
    fake_tensile.write_text("#!/usr/bin/env python3\nraise SystemExit(99)\n", encoding="utf-8")
    fake_tensile.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidate = make_candidate(
        {**REFERENCE_CANDIDATE.canonical_params(), "GlobalSplitU": 2, "DepthU": 32},
        source="shape_rule",
    )
    shape = Shape(m=8192, n=8192, batch=1, k=8192)

    result = execute_schedule(
        db,
        shapes=[shape],
        candidates=[candidate],
        output_root=tmp_path / "batches",
        candidate_batch_size=1,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=tmp_path / "unused_runner",
    )

    assert result.executed_batches == []
    assert db.cache_summary() == {"rejected": 1}


def test_execute_schedule_records_single_candidate_build_timeout(tmp_path: Path):
    fake_tensile = tmp_path / "slow_tensile.py"
    fake_tensile.write_text(
        "#!/usr/bin/env python3\nimport time\ntime.sleep(10)\n",
        encoding="utf-8",
    )
    fake_tensile.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()

    result = execute_schedule(
        db,
        shapes=[shape],
        candidates=[candidate],
        output_root=tmp_path / "batches",
        candidate_batch_size=1,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=tmp_path / "unused_runner",
        build_timeout_s=0.1,
    )

    assert len(result.executed_batches) == 1
    assert result.executed_batches[0].build_returncode == 124
    assert db.cache_summary() == {"build_timeout": 1}
    assert (
        len(
            plan_batches(
                db,
                shapes=[shape],
                candidates=[candidate],
                problem_type_hash=p_hash,
                benchmark_protocol_hash=b_hash,
                validation_protocol_hash=DEFAULT_PROFILE.default_protocol.validation_protocol_hash(),
                candidate_batch_size=1,
                shape_batch_size=1,
            )
        )
        == 1
    )


def test_execute_schedule_salvages_final_yaml_and_uses_diagnostics_for_nonzero_build(tmp_path: Path, monkeypatch):
    fake_tensile = tmp_path / "fake_tensile.py"
    fake_tensile.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import sys
            from pathlib import Path
            import yaml

            config_path, out = Path(sys.argv[1]), Path(sys.argv[2])
            out.mkdir(parents=True, exist_ok=True)
            config = yaml.safe_load(config_path.read_text())
            problem = config["BenchmarkProblems"][0][1]
            problem_sizes = problem["BenchmarkFinalParameters"][0]["ProblemSizes"]
            item = problem["ForkParameters"][0]["Groups"][0][1]
            sol = dict(item)
            mi = sol["MatrixInstruction"]
            sol["MatrixInstruction"] = mi[:4]
            sol["MIWaveTile"] = [mi[5], mi[6]]
            sol["MIWaveGroup"] = [mi[7], mi[8]]
            sol["SolutionIndex"] = 0
            sol["KernelNameMin"] = "Kernel0"
            (out / "1_BenchmarkProblems" / "Cijk_Ailk_Bjlk_HHS_BH_Bias_H_HA_S_SAV_UserArgs_00" / "Data").mkdir(parents=True, exist_ok=True)
            (out / "1_BenchmarkProblems" / "Cijk_Ailk_Bjlk_HHS_BH_Bias_H_HA_S_SAV_UserArgs_00" / "Data" / "00_Final.yaml").write_text(
                yaml.safe_dump(
                    [{"MinimumRequiredVersion": "5.0.0"}, {"ProblemSizes": problem_sizes}, sol],
                    sort_keys=False,
                )
            )
            lib = out / "4_LibraryClient" / "library" / "gfx1151"
            lib.mkdir(parents=True, exist_ok=True)
            (lib / "TensileLibrary_gfx1151.yaml").write_text("---\\nsolutions: []\\n")
            (lib / "Kernels.so-000-gfx1151.hsaco").write_bytes(b"fake")
            sys.exit(2)
            """
        ),
        encoding="utf-8",
    )
    fake_tensile.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidates = [sample_candidates(1)[0], REFERENCE_CANDIDATE]
    shape = pilot_100_shapes()[0]

    fake_runner = tmp_path / "fake_runner.py"
    fake_runner.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import argparse
            import json

            parser = argparse.ArgumentParser()
            parser.add_argument("--mode", required=True)
            parser.add_argument("--pairs")
            parser.add_argument("--output")
            args, _ = parser.parse_known_args()
            with open(args.pairs) as src, open(args.output, "w") as out:
                for line in src:
                    pair = json.loads(line)
                    if args.mode == "validate":
                        out.write(json.dumps({
                            "shape_id": pair["shape_id"],
                            "candidate_hash": pair["candidate_hash"],
                            "status": "ok",
                            "sample_index": 0,
                            "time_us": None,
                            "validation": "PASSED",
                            "solution_index": pair["library_solution_index"],
                        }) + "\\n")
                    else:
                        for sample_index in range(pair.get("num_benchmarks", 1)):
                            out.write(json.dumps({
                                "shape_id": pair["shape_id"],
                                "candidate_hash": pair["candidate_hash"],
                                "status": "ok",
                                "sample_index": sample_index,
                                "time_us": 1.0 + sample_index * 0.001,
                                "validation": "NO_CHECK",
                                "solution_index": pair["library_solution_index"],
                            }) + "\\n")
            """
        ),
        encoding="utf-8",
    )
    fake_runner.chmod(0o755)

    def fake_diagnostics(*args, **kwargs):
        diagnostics_path = tmp_path / "diagnostics.jsonl"
        diagnostics_path.write_text("", encoding="utf-8")
        return DiagnosticRunResult(
            run_id="diagnostics_run",
            returncode=0,
            records=[
                DiagnosticRecord(
                    candidate_hash=candidates[0].hash,
                    candidate_index=0,
                    status="kernelwriter_failed",
                    phase="kernelwriter",
                    reason="KernelWriter returned errcode -2",
                    shape_ids=(shape.id,),
                )
            ],
            results_path=diagnostics_path,
            stdout_path=diagnostics_path,
            stderr_path=diagnostics_path,
            command=["diagnostics"],
            duration_s=0.0,
        )

    monkeypatch.setattr("evotensile.scheduler.run_tensilelite_diagnostics", fake_diagnostics)

    result = execute_schedule(
        db,
        shapes=[shape],
        candidates=candidates,
        output_root=tmp_path / "batches",
        candidate_batch_size=2,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=fake_runner,
        keep_going=True,
    )

    assert len(result.executed_batches) == 1
    assert result.executed_batches[0].ingest is not None
    assert result.executed_batches[0].ingest.status_counts == {"ok": 10, "build_failed": 1}
    assert db.cache_summary() == {"build_failed": 1, "ok": 10}


def test_multi_candidate_build_failure_unattributed_is_not_reusable_cache(tmp_path: Path, monkeypatch):
    fake_tensile = tmp_path / "fake_tensile.py"
    fake_tensile.write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(2)\n", encoding="utf-8")
    fake_tensile.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidates = sample_candidates(2)
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()

    def fake_diagnostics(*args, **kwargs):
        diagnostics_path = tmp_path / "empty_diagnostics.jsonl"
        diagnostics_path.write_text("", encoding="utf-8")
        return DiagnosticRunResult(
            run_id="diagnostics_unattributed",
            returncode=0,
            records=[],
            results_path=diagnostics_path,
            stdout_path=diagnostics_path,
            stderr_path=diagnostics_path,
            command=["diagnostics"],
            duration_s=0.0,
        )

    monkeypatch.setattr("evotensile.scheduler.run_tensilelite_diagnostics", fake_diagnostics)

    result = execute_schedule(
        db,
        shapes=[shape],
        candidates=candidates,
        output_root=tmp_path / "batches",
        candidate_batch_size=2,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=tmp_path / "unused_runner",
        keep_going=True,
    )

    assert len(result.executed_batches) == 1
    assert result.executed_batches[0].ingest is not None
    assert result.executed_batches[0].ingest.status_counts == {"build_failed_unattributed": 2}
    assert db.cache_summary() == {"build_failed_unattributed": 2}
    assert (
        len(
            plan_batches(
                db,
                shapes=[shape],
                candidates=candidates,
                problem_type_hash=p_hash,
                benchmark_protocol_hash=b_hash,
                validation_protocol_hash=DEFAULT_PROFILE.default_protocol.validation_protocol_hash(),
                candidate_batch_size=2,
                shape_batch_size=1,
            )
        )
        == 1
    )


def test_execute_schedule_records_single_candidate_build_failure(tmp_path: Path):
    fake_tensile = tmp_path / "fake_tensile.py"
    fake_tensile.write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(2)\n", encoding="utf-8")
    fake_tensile.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()

    result = execute_schedule(
        db,
        shapes=[shape],
        candidates=[candidate],
        output_root=tmp_path / "batches",
        candidate_batch_size=1,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=tmp_path / "unused_runner",
    )

    assert len(result.executed_batches) == 1
    assert db.cache_summary() == {"build_failed": 1}
    assert (
        plan_batches(
            db,
            shapes=[shape],
            candidates=[candidate],
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            validation_protocol_hash=DEFAULT_PROFILE.default_protocol.validation_protocol_hash(),
            candidate_batch_size=1,
            shape_batch_size=1,
        )
        == []
    )


def test_schedule_cli_metadata_records_operational_modes(tmp_path: Path):
    def run_cli(output_dir: Path, *extra_args: str) -> dict:
        rc = cli_main(
            [
                "schedule-batches",
                "--db",
                str(tmp_path / "sched.sqlite"),
                "--output-dir",
                str(output_dir),
                "--num-random",
                "1",
                "--limit-shapes",
                "1",
                "--shape-batch-size",
                "1",
                "--dry-run",
                *extra_args,
            ]
        )
        assert rc == 0
        return json.loads((output_dir / "schedule_metadata.json").read_text(encoding="utf-8"))

    default_metadata = run_cli(tmp_path / "default")
    assert default_metadata["profile"] == DEFAULT_PROFILE.name
    assert default_metadata["planned_batches"] >= 1
    assert default_metadata["executed_batches"] == []
    assert default_metadata["runner_bin"] == DEFAULT_PROFILE.default_runner_bin
    assert default_metadata["candidate_batch_size"] == production_candidate_batch_size(
        candidate_count=default_metadata["candidates"],
        shape_count=default_metadata["shapes"],
        shape_batch_size=default_metadata["shape_batch_size"],
        prepare_workers=default_metadata["prepare_workers"],
        max_candidate_batch_size=DEFAULT_PROFILE.default_candidate_batch_size,
    )
    assert default_metadata["prepare_workers"] == default_prepare_workers()
    assert default_metadata["adaptive_sampling"] is True
    assert default_metadata["stop_on_error"] is False
    assert default_metadata["learned_linkage_requested"] is True
    assert default_metadata["learned_linkage_enabled"] is False
    assert default_metadata["linkage_fallback_reason"] == "insufficient_validated_evidence"
    assert default_metadata["candidate_family_count"] >= 1
    assert sum(default_metadata["candidate_family_counts"].values()) == default_metadata["candidates"]
    assert default_metadata["archive_family_count"] == 0

    assert default_metadata["compile_cache_enabled"] is True
    assert default_metadata["compile_cache_root"] == str(tmp_path / "default" / "compile_cache")

    large_batch_metadata = run_cli(
        tmp_path / "large_batch",
        "--num-random",
        "16",
        "--prepare-workers",
        "8",
    )
    assert large_batch_metadata["candidate_batch_size"] > 1
    assert large_batch_metadata["planned_batches"] >= large_batch_metadata["prepare_workers"]

    debug_singleton_metadata = run_cli(tmp_path / "debug_singleton", "--candidate-batch-size", "1")
    assert debug_singleton_metadata["candidate_batch_size"] == 1

    learned_metadata = run_cli(tmp_path / "learned", "--learned-linkage")
    assert learned_metadata["learned_linkage_requested"] is True
    assert learned_metadata["learned_linkage_enabled"] is False
    assert learned_metadata["linkage_fallback_reason"] == "insufficient_validated_evidence"
    assert learned_metadata["linkage_min_samples"] == 8

    no_learned_metadata = run_cli(tmp_path / "no_learned", "--no-learned-linkage")
    assert no_learned_metadata["learned_linkage_requested"] is False
    assert no_learned_metadata["linkage_fallback_reason"] == "disabled"

    no_compile_cache_metadata = run_cli(tmp_path / "no_compile_cache", "--no-compile-cache")
    assert no_compile_cache_metadata["compile_cache_enabled"] is False
    assert no_compile_cache_metadata["compile_cache_root"] is None

    fail_fast_metadata = run_cli(tmp_path / "fail_fast", "--stop-on-error")
    fixed_sampling_metadata = run_cli(tmp_path / "fixed", "--fixed-sampling")
    assert fail_fast_metadata["stop_on_error"] is True
    assert fixed_sampling_metadata["adaptive_sampling"] is False


def test_execute_schedule_generate_only_writes_batch_inputs(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidates = sample_candidates(2)
    shapes = pilot_100_shapes()[:1]

    result = execute_schedule(
        db,
        shapes=shapes,
        candidates=candidates,
        output_root=tmp_path / "batches",
        candidate_batch_size=1,
        shape_batch_size=1,
        generate_only=True,
    )

    assert len(result.planned_batches) == 2
    assert len(result.executed_batches) == 2
    for executed in result.executed_batches:
        assert executed.yaml_path.exists()
        assert executed.manifest_path.exists()
        assert executed.output_dir.name == "run"
