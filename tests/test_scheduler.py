import fcntl
import json
import os
from dataclasses import replace
from pathlib import Path
from textwrap import dedent

import pytest

from evotensile.candidate import Shape
from evotensile.cli import main as cli_main
from evotensile.database import EvoTensileDB, ValidationInsert
from evotensile.profile import DEFAULT_PROFILE, PROFILES
from evotensile.scheduler import (
    _compile_cache_lock,
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


def _record_validation(
    db: EvoTensileDB,
    shape: Shape,
    candidate_hash: str,
    detail: str = "PASSED",
    *,
    status: str = "passed",
    validation_protocol_hash: str | None = None,
) -> None:
    db.insert_validations(
        [
            ValidationInsert(
                shape_id=shape.id,
                candidate_hash=candidate_hash,
                run_id="cached_validation",
                status=status,
                problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
                validation_protocol_hash=(
                    validation_protocol_hash or DEFAULT_PROFILE.default_protocol.validation_protocol_hash()
                ),
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


def test_compile_cache_lock_reuses_file_after_dead_owner_release(tmp_path: Path):
    cache_dir = tmp_path / "ccache_test"
    lock_path = tmp_path / ".ccache_test.lock"
    lock_path.write_text('{"pid": 999999999, "token": "dead"}\n', encoding="utf-8")

    with _compile_cache_lock(cache_dir, wait_timeout_s=0.1):
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
        assert owner["pid"] == os.getpid()
        assert owner["token"] != "dead"

    assert lock_path.exists()
    assert lock_path.read_text(encoding="utf-8") == ""


def test_compile_cache_lock_times_out_on_live_owner(tmp_path: Path):
    cache_dir = tmp_path / "ccache_test"
    lock_path = tmp_path / ".ccache_test.lock"
    with lock_path.open("a+", encoding="utf-8") as owner_file:
        fcntl.flock(owner_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner_file.write('{"pid": 1, "token": "live"}\n')
        owner_file.flush()
        with pytest.raises(TimeoutError, match="waiting for compile-cache lock"):
            with _compile_cache_lock(cache_dir, wait_timeout_s=0.01):
                pass

    assert lock_path.exists()


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


def test_validation_failure_is_scoped_to_validation_protocol(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    gpu_vhash = DEFAULT_PROFILE.default_protocol.validation_protocol_hash()
    cpu_vhash = DEFAULT_PROFILE.default_protocol.with_overrides(validation_backend="cpu").validation_protocol_hash()
    _record_validation(
        db,
        shape,
        candidate.hash,
        "FAILED gpu mismatch",
        status="failed",
        validation_protocol_hash=gpu_vhash,
    )
    db.insert_evaluation(
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="legacy_failure",
        status="validation_fail",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )

    assert (
        plan_batches(
            db,
            shapes=[shape],
            candidates=[candidate],
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            validation_protocol_hash=gpu_vhash,
            candidate_batch_size=1,
            shape_batch_size=1,
        )
        == []
    )
    cpu_batches = plan_batches(
        db,
        shapes=[shape],
        candidates=[candidate],
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        validation_protocol_hash=cpu_vhash,
        candidate_batch_size=1,
        shape_batch_size=1,
    )
    assert len(cpu_batches) == 1
    assert cpu_batches[0].requires_validation


def test_positive_timing_allows_topup_despite_older_or_newer_negative(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    v_hash = DEFAULT_PROFILE.default_protocol.validation_protocol_hash()
    db.insert_evaluation(
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="old_rejection",
        status="rejected",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )
    db.insert_evaluation(
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="timing",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=1.0,
        validation="PASSED prior_validation",
    )
    db.insert_evaluation(
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="new_build_failure",
        status="build_failed",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )
    _record_validation(db, shape, candidate.hash, "PASSED")

    batches = plan_batches(
        db,
        shapes=[shape],
        candidates=[candidate],
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        validation_protocol_hash=v_hash,
        min_samples=2,
        candidate_batch_size=1,
        shape_batch_size=1,
    )

    assert len(batches) == 1
    assert batches[0].samples_per_pair == 1
    assert not batches[0].requires_validation


def test_latest_validation_result_resolves_pass_fail_conflicts(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    v_hash = DEFAULT_PROFILE.default_protocol.validation_protocol_hash()
    db.insert_evaluation(
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="timing",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=1.0,
        validation="PASSED prior_validation",
    )
    _record_validation(db, shape, candidate.hash, "FAILED first", status="failed")
    _record_validation(db, shape, candidate.hash, "PASSED second")

    pass_batches = plan_batches(
        db,
        shapes=[shape],
        candidates=[candidate],
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        validation_protocol_hash=v_hash,
        min_samples=2,
        candidate_batch_size=1,
        shape_batch_size=1,
    )
    assert len(pass_batches) == 1
    assert not pass_batches[0].requires_validation

    _record_validation(db, shape, candidate.hash, "FAILED latest", status="failed")
    assert (
        plan_batches(
            db,
            shapes=[shape],
            candidates=[candidate],
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            validation_protocol_hash=v_hash,
            min_samples=2,
            candidate_batch_size=1,
            shape_batch_size=1,
        )
        == []
    )


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
    ).selected

    mutations = [candidate for candidate in proposed if candidate.source == "mutation"]
    assert len(mutations) == 1
    assert mutations[0].parent_hashes == (candidates[0].hash,)
    assert {candidate.source for candidate in proposed} == {"mutation"}


def test_multi_shape_elites_include_shape_normalized_specialists(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(4)
    small = Shape(512, 128, 1, 256)
    large = Shape(8192, 8192, 1, 8192)
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([small, large])
    for shape, rows in (
        (small, ((candidates[0], 1.0), (candidates[1], 2.0))),
        (large, ((candidates[2], 1000.0), (candidates[3], 1200.0))),
    ):
        for candidate, time_us in rows:
            db.insert_evaluation(
                shape_id=shape.id,
                candidate_hash=candidate.hash,
                run_id="cached",
                status="ok",
                problem_type_hash=p_hash,
                benchmark_protocol_hash=b_hash,
                time_us=time_us,
                validation="PASSED",
            )

    proposed = propose_candidates(
        db,
        proposal="local",
        local_count=8,
        elite_count=2,
        target_shapes=[small, large],
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        seed=7,
    ).selected

    parent_hashes = {parent_hash for candidate in proposed for parent_hash in candidate.parent_hashes}
    assert candidates[0].hash in parent_hashes
    assert candidates[2].hash in parent_hashes


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
    ).selected

    transfer = [candidate for candidate in proposed if candidate.source == "transfer"]
    assert [candidate.hash for candidate in transfer] == [candidates[1].hash]


def test_multi_target_transfer_round_robins_target_neighborhoods(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(2)
    first = Shape(512, 128, 1, 256)
    second = Shape(8192, 8192, 1, 8192)
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([first, second])
    for shape, candidate, time_us in ((first, candidates[0], 1.0), (second, candidates[1], 1000.0)):
        db.insert_evaluation(
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=time_us,
            validation="PASSED",
        )

    proposed = propose_candidates(
        db,
        proposal="seed-random-gomea",
        num_random=0,
        gomea_count=0,
        target_shapes=[first, second],
        transfer_shape_count=2,
        transfer_per_shape=1,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    ).selected

    transfer = [candidate for candidate in proposed if candidate.source == "transfer"]
    assert {candidate.hash for candidate in transfer} == {candidate.hash for candidate in candidates}
    assert set().union(*(set(candidate.proposal_metadata["transfer_target_shape_ids"]) for candidate in transfer)) == {
        first.id,
        second.id,
    }
    assert all(candidate.proposal_metadata["transfer_source_shape_ids"] for candidate in transfer)


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
    ).selected

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
    ).selected

    assert all(candidate.source != "transfer" for candidate in proposed)
    assert {candidates[0].hash} & {parent for candidate in proposed for parent in candidate.parent_hashes}
    assert candidates[1].hash not in {parent for candidate in proposed for parent in candidate.parent_hashes}


def test_family_qd_builds_one_evidence_snapshot_per_proposal(tmp_path: Path, monkeypatch):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(8)
    shape = pilot_100_shapes()[0]
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
            time_us=1.0 + index,
            validation="PASSED",
        )

    rank_calls = 0
    original_rank = db.rank_evaluations

    def counted_rank(*args, **kwargs):
        nonlocal rank_calls
        rank_calls += 1
        return original_rank(*args, **kwargs)

    monkeypatch.setattr(db, "rank_evaluations", counted_rank)
    proposed = propose_candidates(
        db,
        proposal="family-qd",
        num_random=2,
        local_count=2,
        de_count=2,
        gomea_count=4,
        elite_count=8,
        target_shapes=[shape],
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        adaptive_operators=True,
        adaptive_group_credit=True,
        adaptive_donor_selection=True,
        surrogate_pool_multiplier=2,
        linkage_min_samples=4,
        linkage_truncation_tau=1.0,
    ).selected

    assert proposed
    assert rank_calls == 1


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
    ).selected

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
    ).selected

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
    ).selected

    sources = {candidate.source for candidate in proposed}
    assert {"random", "mutation", "de", "gomea"} & sources == {"random", "mutation", "de", "gomea"}
    assert "seed" not in sources


def test_random_proposal_does_not_include_fixed_controls(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()

    proposed = propose_candidates(db, proposal="seed-random", num_random=12, seed=1151).selected

    assert {candidate.source for candidate in proposed} == {"random"}
    assert REFERENCE_CANDIDATE.hash not in {candidate.hash for candidate in proposed}


def test_random_proposals_respect_target_shape_rules(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    shape = Shape(m=8192, n=8192, batch=1, k=8192)

    proposed = propose_candidates(db, proposal="random", num_random=16, seed=20260701, target_shapes=[shape]).selected

    assert len(proposed) == 16
    assert all(cheap_constraints(candidate.canonical_params(), shape=shape) for candidate in proposed)


def test_multi_shape_random_proposal_keeps_scoped_specialists(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    small = Shape(m=512, n=128, batch=1, k=256)
    large = Shape(m=8192, n=8192, batch=1, k=8192)

    result = propose_candidates(
        db,
        proposal="random",
        num_random=64,
        seed=20260701,
        target_shapes=[small, large],
        scope_kind="cluster",
    )

    specialists = [
        candidate
        for candidate in result.selected
        if cheap_constraints(candidate.canonical_params(), shape=small)
        and not cheap_constraints(candidate.canonical_params(), shape=large)
    ]
    assert specialists
    assert result.scope.kind == "cluster"
    assert result.scope.shape_ids == (small.id, large.id)
    assert all(
        candidate.proposal_metadata["proposal_scope_kind"] == "cluster"
        and candidate.proposal_metadata["proposal_scope_shape_ids"] == [small.id, large.id]
        for candidate in result.generated
    )


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
        ).selected

        assert proposed == ()


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
        ).selected

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
    ).selected

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
    ).selected

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
    ).selected

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
    assert default_metadata["prepare_workers"] == DEFAULT_PROFILE.default_prepare_workers == 32
    assert default_metadata["validation_workers"] == DEFAULT_PROFILE.default_validation_workers == 1
    assert default_metadata["surrogate_jobs"] == DEFAULT_PROFILE.default_surrogate_jobs
    assert default_metadata["compute_unit_count"] == DEFAULT_PROFILE.compute_unit_count == 40
    assert default_metadata["workgroup_processor_count"] == DEFAULT_PROFILE.workgroup_processor_count == 20
    assert default_metadata["compute_units_per_workgroup_processor"] == 2
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

    cached_batch_metadata = run_cli(
        tmp_path / "cached_batch",
        "--num-random",
        "16",
        "--prepare-workers",
        "8",
    )
    assert cached_batch_metadata["candidate_batch_size"] == 1
    assert cached_batch_metadata["planned_batches"] >= cached_batch_metadata["prepare_workers"]

    large_batch_metadata = run_cli(
        tmp_path / "large_batch",
        "--num-random",
        "16",
        "--prepare-workers",
        "8",
        "--no-compile-cache",
    )
    assert large_batch_metadata["candidate_batch_size"] > 1

    debug_singleton_metadata = run_cli(tmp_path / "debug_singleton", "--candidate-batch-size", "1")
    assert debug_singleton_metadata["candidate_batch_size"] == 1

    production_policy_metadata = run_cli(
        tmp_path / "production_policy",
        "--search-policy",
        "gfx1151-grid-v1",
        "--num-random",
        "0",
        "--no-adaptive-donor-selection",
    )
    assert production_policy_metadata["search_policy"] == "gfx1151-grid-v1"
    assert production_policy_metadata["search_policy_settings"]["proposal"] == "family-qd"
    assert production_policy_metadata["proposal"] == "family-qd"
    assert production_policy_metadata["surrogate_pool_multiplier"] == 8
    assert production_policy_metadata["adaptive_operators"] is True
    assert production_policy_metadata["adaptive_group_credit"] is True
    assert production_policy_metadata["adaptive_donor_selection"] is False
    assert production_policy_metadata["cost_aware_operator_credit"] is True
    assert production_policy_metadata["cost_aware_scheduling"] is True

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


def test_schedule_cli_resolves_selected_profile_defaults(tmp_path: Path):
    profile = replace(
        DEFAULT_PROFILE,
        name="test-profile",
        default_proposal="random",
        default_num_random=3,
        default_elite_count=5,
        default_local_count=7,
        default_de_count=9,
        default_gomea_count=11,
        default_transfer_shapes=2,
        default_transfer_per_shape=3,
        default_mutation_rate=0.15,
        default_crossover_rate=0.65,
        default_random_gene_rate=0.05,
        default_candidate_batch_size=4,
        default_shape_batch_size=2,
        default_prepare_workers=6,
        default_validation_workers=1,
        default_surrogate_jobs=2,
        compute_unit_count=24,
        workgroup_processor_count=12,
    )
    PROFILES[profile.name] = profile
    try:
        output_dir = tmp_path / "selected_profile"
        assert (
            cli_main(
                [
                    "schedule-batches",
                    "--db",
                    str(tmp_path / "sched.sqlite"),
                    "--output-dir",
                    str(output_dir),
                    "--profile",
                    profile.name,
                    "--limit-shapes",
                    "1",
                    "--dry-run",
                ]
            )
            == 0
        )
    finally:
        del PROFILES[profile.name]

    metadata = json.loads((output_dir / "schedule_metadata.json").read_text(encoding="utf-8"))
    assert metadata["profile"] == profile.name
    assert metadata["proposal"] == profile.default_proposal
    assert metadata["candidates"] == profile.default_num_random
    assert metadata["shape_batch_size"] == profile.default_shape_batch_size
    assert metadata["prepare_workers"] == profile.default_prepare_workers
    assert metadata["validation_workers"] == profile.default_validation_workers
    assert metadata["surrogate_jobs"] == profile.default_surrogate_jobs
    assert metadata["compute_unit_count"] == profile.compute_unit_count
    assert metadata["workgroup_processor_count"] == profile.workgroup_processor_count
    assert metadata["compute_units_per_workgroup_processor"] == profile.compute_units_per_workgroup_processor


def test_search_policy_rejects_incompatible_profile(tmp_path: Path):
    profile = replace(DEFAULT_PROFILE, name="test-profile")
    PROFILES[profile.name] = profile
    try:
        with pytest.raises(ValueError, match="requires profile gfx1151-nt-hhs"):
            cli_main(
                [
                    "schedule-batches",
                    "--db",
                    str(tmp_path / "sched.sqlite"),
                    "--output-dir",
                    str(tmp_path / "output"),
                    "--profile",
                    profile.name,
                    "--search-policy",
                    "gfx1151-grid-v1",
                    "--dry-run",
                ]
            )
    finally:
        del PROFILES[profile.name]


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
