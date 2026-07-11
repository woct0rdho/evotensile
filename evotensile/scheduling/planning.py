import math
from typing import TypeVar

from evotensile.adaptive_retime import ProbePolicy, decide_shape_probe, load_timing_stats
from evotensile.candidate import Candidate, Shape
from evotensile.database import BenchmarkEventInsert, EvoTensileDB
from evotensile.scheduling.models import PlannedBatch
from evotensile.search_space import explain_invalid_nt_hhs

T = TypeVar("T")


def production_candidate_batch_size(
    *,
    candidate_count: int,
    shape_count: int,
    shape_batch_size: int,
    prepare_workers: int,
    max_candidate_batch_size: int,
) -> int:
    if candidate_count <= 0 or prepare_workers <= 0:
        return 1
    shape_batches = max(1, math.ceil(max(1, shape_count) / shape_batch_size))
    max_size = max(1, min(candidate_count, max_candidate_batch_size))
    for candidate_batch_size in range(max_size, 0, -1):
        if math.ceil(candidate_count / candidate_batch_size) * shape_batches >= prepare_workers:
            return candidate_batch_size
    return 1


def _chunks(items: list[T], size: int) -> list[list[T]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    return [items[i : i + size] for i in range(0, len(items), size)]


def record_shape_rule_rejections(
    db: EvoTensileDB,
    *,
    shapes: list[Shape],
    candidates: list[Candidate],
    problem_type_hash: str,
    benchmark_protocol_hash: str,
) -> int:
    states = db.benchmark_evidence_states(
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        shape_ids=[shape.id for shape in shapes],
        candidate_hashes=[candidate.hash for candidate in candidates],
    )
    events: list[BenchmarkEventInsert] = []
    for shape in shapes:
        for candidate in candidates:
            if (shape.id, candidate.hash) in states:
                continue
            if any(
                reason.shape_dependent for reason in explain_invalid_nt_hhs(candidate.canonical_params(), shape=shape)
            ):
                events.append(
                    BenchmarkEventInsert(
                        shape_id=shape.id,
                        candidate_hash=candidate.hash,
                        run_id=None,
                        status="rejected",
                        source_kind="static_rule",
                        problem_type_hash=problem_type_hash,
                        benchmark_protocol_hash=benchmark_protocol_hash,
                    )
                )
    db.insert_benchmark_events(events)
    return len(events)


def _missing_candidate_indices_by_shape(
    db: EvoTensileDB,
    *,
    shapes: list[Shape],
    candidates: list[Candidate],
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    validation_protocol_hash: str,
    min_samples: int,
    ignore_cache: bool = False,
) -> dict[int, tuple[tuple[int, int, bool], ...]]:
    benchmark_states = {}
    validation_states: dict[tuple[str, str], str] = {}
    shape_ids = [shape.id for shape in shapes]
    candidate_hashes = [candidate.hash for candidate in candidates]
    if not ignore_cache:
        benchmark_states = db.benchmark_evidence_states(
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            shape_ids=shape_ids,
            candidate_hashes=candidate_hashes,
        )
        validation_states = db.validation_cache_states(
            problem_type_hash=problem_type_hash,
            validation_protocol_hash=validation_protocol_hash,
            shape_ids=shape_ids,
            candidate_hashes=candidate_hashes,
        )

    missing: dict[int, tuple[tuple[int, int, bool], ...]] = {}
    for shape_index, shape in enumerate(shapes):
        missing_items: list[tuple[int, int, bool]] = []
        for candidate_index, candidate in enumerate(candidates):
            if any(
                reason.shape_dependent for reason in explain_invalid_nt_hhs(candidate.canonical_params(), shape=shape)
            ):
                continue
            key = (shape.id, candidate.hash)
            benchmark_state = None if ignore_cache else benchmark_states.get(key)
            if (benchmark_state is not None and benchmark_state.reusable_negative) or validation_states.get(
                key
            ) == "failed":
                continue
            ok_count = 0 if benchmark_state is None else benchmark_state.ok_samples
            remaining = max(0, min_samples - ok_count)
            if remaining > 0:
                missing_items.append((candidate_index, remaining, validation_states.get(key) != "passed"))
        if missing_items:
            missing[shape_index] = tuple(missing_items)
    return missing


def _pair_exact_batches(
    *,
    batch_index_start: int,
    shapes: list[Shape],
    candidates: list[Candidate],
    missing_by_shape: dict[int, tuple[tuple[int, int, bool], ...]],
    max_batches: int | None = None,
) -> list[PlannedBatch]:
    grouped_shapes: dict[tuple[int, bool, tuple[int, ...]], list[Shape]] = {}
    for shape_index, missing_items in missing_by_shape.items():
        by_remaining: dict[tuple[int, bool], list[int]] = {}
        for candidate_index, remaining, requires_validation in missing_items:
            by_remaining.setdefault((remaining, requires_validation), []).append(candidate_index)
        for (remaining, requires_validation), missing_indices in by_remaining.items():
            grouped_shapes.setdefault((remaining, requires_validation, tuple(missing_indices)), []).append(
                shapes[shape_index]
            )

    planned: list[PlannedBatch] = []
    batch_index = batch_index_start
    for (samples_per_pair, requires_validation, missing_indices), group_shapes in grouped_shapes.items():
        group_candidates = [candidates[idx] for idx in missing_indices]
        # This rectangular cover is exact because every shape in the group has
        # the same missing candidate subset. Empty-cache runs still collapse to
        # the dense candidate-chunk x shape-chunk rectangle.
        planned.append(
            PlannedBatch(
                batch_index=batch_index,
                candidates=group_candidates,
                shapes=group_shapes,
                missing_pairs=len(group_candidates) * len(group_shapes),
                nominal_pairs=len(group_candidates) * len(group_shapes),
                samples_per_pair=samples_per_pair,
                requires_validation=requires_validation,
            )
        )
        batch_index += 1
        if max_batches is not None and len(planned) >= max_batches:
            break
    return planned


def preprepare_probe_screened_pairs(
    db: EvoTensileDB,
    *,
    shapes: list[Shape],
    candidates: list[Candidate],
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    probe_protocol_hash: str,
    policy: ProbePolicy,
) -> set[tuple[str, str]]:
    shape_ids = {shape.id for shape in shapes}
    candidate_hashes = {candidate.hash for candidate in candidates}
    probe_stats = load_timing_stats(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hashes=[probe_protocol_hash],
        min_samples=policy.initial_samples,
        shape_ids=shape_ids,
        candidate_hashes=candidate_hashes,
    )
    reference_stats = load_timing_stats(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hashes=[benchmark_protocol_hash],
        min_samples=1,
        shape_ids=shape_ids,
    )
    screened: set[tuple[str, str]] = set()
    for shape_id, stats in probe_stats.items():
        if len(stats) <= policy.min_survivors:
            continue
        decision = decide_shape_probe(
            shape_id,
            stats,
            policy=policy,
            reference_stats=reference_stats.get(shape_id, ()),
        )
        screened.update((shape_id, candidate_hash) for candidate_hash in decision.screened_hashes)
    return screened


def plan_batches(
    db: EvoTensileDB,
    *,
    shapes: list[Shape],
    candidates: list[Candidate],
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    validation_protocol_hash: str,
    min_samples: int = 1,
    candidate_batch_size: int = 32,
    shape_batch_size: int = 100,
    ignore_cache: bool = False,
    max_batches: int | None = None,
    excluded_pairs: set[tuple[str, str]] | None = None,
) -> list[PlannedBatch]:
    planned: list[PlannedBatch] = []
    batch_index = 0
    for candidate_chunk in _chunks(candidates, candidate_batch_size):
        for shape_chunk in _chunks(shapes, shape_batch_size):
            missing_by_shape = _missing_candidate_indices_by_shape(
                db,
                shapes=shape_chunk,
                candidates=candidate_chunk,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                validation_protocol_hash=validation_protocol_hash,
                min_samples=min_samples,
                ignore_cache=ignore_cache,
            )
            if excluded_pairs:
                missing_by_shape = {
                    shape_index: tuple(
                        item
                        for item in missing_items
                        if (shape_chunk[shape_index].id, candidate_chunk[item[0]].hash) not in excluded_pairs
                    )
                    for shape_index, missing_items in missing_by_shape.items()
                }
                missing_by_shape = {key: value for key, value in missing_by_shape.items() if value}
            if not missing_by_shape:
                continue
            new_batches = _pair_exact_batches(
                batch_index_start=batch_index,
                shapes=shape_chunk,
                candidates=candidate_chunk,
                missing_by_shape=missing_by_shape,
                max_batches=None if max_batches is None else max_batches - len(planned),
            )
            planned.extend(new_batches)
            batch_index += len(new_batches)
            if max_batches is not None and len(planned) >= max_batches:
                return planned
    return planned
