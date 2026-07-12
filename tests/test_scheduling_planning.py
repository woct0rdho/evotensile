import fcntl
import json
import os
from pathlib import Path

import pytest

from evotensile.candidate import Candidate, Shape
from evotensile.database import EvoTensileDB, ValidationInsert
from evotensile.profile import DEFAULT_PROFILE
from evotensile.scheduling.compile_cache import compile_cache_lock
from evotensile.scheduling.models import EvidenceStage, PairRequest
from evotensile.scheduling.planning import plan_pair_requests, production_candidate_batch_size
from evotensile.shapes import pilot_100_shapes
from tests.helpers import insert_test_benchmark_event, sample_candidates


def _requests(
    candidates: list[Candidate],
    shapes: list[Shape],
    *,
    min_samples: int = 1,
) -> list[PairRequest]:
    return [
        PairRequest(candidate=candidate, shape=shape, min_samples=min_samples)
        for shape in shapes
        for candidate in candidates
    ]


def _register_pairs(db: EvoTensileDB, candidates: list[Candidate], shapes: list[Shape]) -> None:
    db.register_candidates(candidates)
    db.register_shapes(shapes)


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


def _plan(
    db: EvoTensileDB,
    requests: list[PairRequest],
    **kwargs,
):
    return plan_pair_requests(
        db,
        requests=requests,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        validation_protocol_hash=DEFAULT_PROFILE.default_protocol.validation_protocol_hash(),
        **kwargs,
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


def test_pair_requests_deduplicate_identical_and_reject_conflicts(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    request = PairRequest(candidate, shape, EvidenceStage.SCREENING, 3, 2.0)

    batches = _plan(db, [request, request], candidate_batch_size=1, shape_batch_size=1)
    assert len(batches) == 1
    assert batches[0].requested_pairs == 1

    with pytest.raises(ValueError, match="conflicting exact pair requests"):
        _plan(
            db,
            [request, PairRequest(candidate, shape, EvidenceStage.SCREENING, 4, 2.0)],
            candidate_batch_size=1,
            shape_batch_size=1,
        )


def test_sparse_requests_never_materialize_cross_product_pairs(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(2)
    shapes = pilot_100_shapes()[:2]
    requests = [
        PairRequest(candidates[0], shapes[0], priority=2.0),
        PairRequest(candidates[1], shapes[1], priority=1.0),
    ]

    batches = _plan(db, requests, candidate_batch_size=2, shape_batch_size=2)
    planned_keys = {pair.key for batch in batches for pair in batch.pairs}

    assert planned_keys == {request.key for request in requests}
    assert (shapes[0].id, candidates[1].hash) not in planned_keys
    assert (shapes[1].id, candidates[0].hash) not in planned_keys


def test_explicit_artifact_scope_can_cover_unrequested_shapes(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
    shapes = pilot_100_shapes()[:2]
    request = PairRequest(candidate, shapes[0])

    batches = _plan(
        db,
        [request],
        candidate_batch_size=1,
        shape_batch_size=1,
        artifact_shapes_by_candidate={candidate.hash: shapes},
    )

    assert len(batches) == 1
    assert [shape.id for shape in batches[0].artifact_shapes] == sorted(shape.id for shape in shapes)
    assert [pair.key for pair in batches[0].pairs] == [request.key]


def test_cache_filters_only_requested_exact_pairs(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(2)
    shapes = pilot_100_shapes()[:2]
    requests = [PairRequest(candidates[0], shapes[0]), PairRequest(candidates[1], shapes[1])]
    _register_pairs(db, candidates, shapes)
    insert_test_benchmark_event(
        db,
        shape_id=shapes[0].id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        time_us=1.0,
    )

    batches = _plan(db, requests, candidate_batch_size=2, shape_batch_size=2)

    assert {pair.key for batch in batches for pair in batch.pairs} == {requests[1].key}


def test_planning_requests_only_remaining_samples_and_reuses_validation(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    _register_pairs(db, [candidate], [shape])
    insert_test_benchmark_event(
        db,
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="cached",
        status="ok",
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        time_us=1.0,
    )
    _record_validation(db, shape, candidate.hash, "PASSED checked=1048576 backend=hipblaslt")

    batch = _plan(db, [PairRequest(candidate, shape, min_samples=3)])[0]

    assert batch.pairs[0].samples_to_collect == 2
    assert not batch.pairs[0].requires_validation
    assert batch.requested_samples == 2


def test_mixed_validation_state_is_preserved_per_exact_pair(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(2)
    shape = pilot_100_shapes()[0]
    _register_pairs(db, candidates, [shape])
    _record_validation(db, shape, candidates[0].hash)

    batch = _plan(
        db,
        [PairRequest(candidate, shape) for candidate in candidates],
        candidate_batch_size=2,
        shape_batch_size=1,
    )[0]
    requirements = {pair.request.candidate.hash: pair.requires_validation for pair in batch.pairs}

    assert requirements == {candidates[0].hash: False, candidates[1].hash: True}
    assert batch.requires_validation


def test_validation_failure_is_scoped_to_validation_protocol(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
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

    request = PairRequest(candidate, shape)
    assert _plan(db, [request]) == []
    cpu_batches = plan_pair_requests(
        db,
        requests=[request],
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        validation_protocol_hash=cpu_vhash,
    )
    assert len(cpu_batches) == 1
    assert cpu_batches[0].pairs[0].requires_validation


def test_positive_timing_allows_topup_despite_reusable_negative(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    _register_pairs(db, [candidate], [shape])
    for run_id, status in (("old_rejection", "rejected"), ("new_build_failure", "build_failed")):
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id=run_id,
            status=status,
            problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
            benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        )
    insert_test_benchmark_event(
        db,
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="timing",
        status="ok",
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        time_us=1.0,
    )
    _record_validation(db, shape, candidate.hash)

    batch = _plan(db, [PairRequest(candidate, shape, min_samples=2)])[0]

    assert batch.pairs[0].samples_to_collect == 1
    assert not batch.pairs[0].requires_validation


def test_reusable_negative_cache_entries_skip_only_their_requests(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(2)
    shape = pilot_100_shapes()[0]
    _register_pairs(db, candidates, [shape])
    for candidate, status in zip(candidates, ("rejected", "build_failed"), strict=True):
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id=f"cached_{status}",
            status=status,
            problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
            benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        )

    assert _plan(db, [PairRequest(candidate, shape) for candidate in candidates]) == []
