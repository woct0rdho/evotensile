import fcntl
import json
import os
from pathlib import Path

import pytest

from evotensile.candidate import Shape
from evotensile.database import EvoTensileDB, ValidationInsert
from evotensile.profile import DEFAULT_PROFILE
from evotensile.scheduling.compile_cache import compile_cache_lock
from evotensile.scheduling.planning import plan_batches, production_candidate_batch_size
from evotensile.shapes import pilot_100_shapes
from tests.helpers import insert_test_benchmark_event, sample_candidates


def _register_pairs(db: EvoTensileDB, candidates, shapes) -> None:
    db.register_candidates(list(candidates))
    db.register_shapes(list(shapes))


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
                source_kind="replay",
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

    with compile_cache_lock(cache_dir, wait_timeout_s=0.1):
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
            with compile_cache_lock(cache_dir, wait_timeout_s=0.01):
                pass

    assert lock_path.exists()


def test_plan_batches_skips_cached_ok_pairs(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(2)
    shapes = pilot_100_shapes()[:2]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    _register_pairs(db, candidates, shapes)
    insert_test_benchmark_event(
        db,
        shape_id=shapes[0].id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=1.0,
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
    _register_pairs(db, candidates, shapes)
    insert_test_benchmark_event(
        db,
        shape_id=shapes[0].id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=1.0,
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
    _register_pairs(db, candidates, shapes)
    insert_test_benchmark_event(
        db,
        shape_id=shapes[0].id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=1.0,
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
    _register_pairs(db, candidates, shapes)
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
    _register_pairs(db, [candidate], [shape])
    _record_validation(
        db,
        shape,
        candidate.hash,
        "FAILED gpu mismatch",
        status="failed",
        validation_protocol_hash=gpu_vhash,
    )
    insert_test_benchmark_event(
        db,
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
    _register_pairs(db, [candidate], [shape])
    insert_test_benchmark_event(
        db,
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="old_rejection",
        status="rejected",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )
    insert_test_benchmark_event(
        db,
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="timing",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=1.0,
    )
    insert_test_benchmark_event(
        db,
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
    _register_pairs(db, [candidate], [shape])
    insert_test_benchmark_event(
        db,
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="timing",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=1.0,
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
    _register_pairs(db, candidates, shapes)
    insert_test_benchmark_event(
        db,
        shape_id=shapes[0].id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="rejected",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )
    insert_test_benchmark_event(
        db,
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
