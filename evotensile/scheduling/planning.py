import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from evotensile.adaptive_retime import ProbePolicy, decide_shape_probe, load_timing_stats
from evotensile.candidate import Candidate, Shape
from evotensile.database import BenchmarkEventInsert, EvoTensileDB
from evotensile.scheduling.models import EvidenceStage, PairRequest, PlannedBatch, PlannedPair
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
    return [items[index : index + size] for index in range(0, len(items), size)]


def normalize_pair_requests(requests: Sequence[PairRequest]) -> list[PairRequest]:
    by_key: dict[tuple[str, str], PairRequest] = {}
    for request in requests:
        current = by_key.get(request.key)
        if current is None:
            by_key[request.key] = request
            continue
        requirements = (request.evidence_stage, request.min_samples, request.priority)
        current_requirements = (current.evidence_stage, current.min_samples, current.priority)
        if requirements != current_requirements:
            raise ValueError(
                "conflicting exact pair requests for "
                f"shape={request.shape.id} candidate={request.candidate.hash}: "
                f"{current_requirements} != {requirements}"
            )
    return sorted(by_key.values(), key=lambda item: -item.priority)


def requested_candidates(requests: Sequence[PairRequest]) -> list[Candidate]:
    return list({request.candidate.hash: request.candidate for request in requests}.values())


def requested_shapes(requests: Sequence[PairRequest]) -> list[Shape]:
    return list({request.shape.id: request.shape for request in requests}.values())


def record_shape_rule_rejections(
    db: EvoTensileDB,
    *,
    requests: Sequence[PairRequest],
    problem_type_hash: str,
    benchmark_protocol_hash: str,
) -> int:
    normalized = normalize_pair_requests(requests)
    states = db.benchmark_evidence_states(
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        shape_ids=[shape.id for shape in requested_shapes(normalized)],
        candidate_hashes=[candidate.hash for candidate in requested_candidates(normalized)],
    )
    events = [
        BenchmarkEventInsert(
            shape_id=request.shape.id,
            candidate_hash=request.candidate.hash,
            run_id=None,
            status="rejected",
            source_kind="static_rule",
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
        )
        for request in normalized
        if request.key not in states
        and any(
            reason.shape_dependent
            for reason in explain_invalid_nt_hhs(request.candidate.canonical_params(), shape=request.shape)
        )
    ]
    db.insert_benchmark_events(events)
    return len(events)


def _planned_pairs(
    db: EvoTensileDB,
    *,
    requests: Sequence[PairRequest],
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    validation_protocol_hash: str,
    ignore_cache: bool,
    excluded_pairs: set[tuple[str, str]],
) -> list[PlannedPair]:
    shapes = requested_shapes(requests)
    candidates = requested_candidates(requests)
    benchmark_states = {}
    validation_states: dict[tuple[str, str], str] = {}
    if not ignore_cache:
        benchmark_states = db.benchmark_evidence_states(
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            shape_ids=[shape.id for shape in shapes],
            candidate_hashes=[candidate.hash for candidate in candidates],
        )
        validation_states = db.validation_cache_states(
            problem_type_hash=problem_type_hash,
            validation_protocol_hash=validation_protocol_hash,
            shape_ids=[shape.id for shape in shapes],
            candidate_hashes=[candidate.hash for candidate in candidates],
        )

    planned: list[PlannedPair] = []
    for request in requests:
        if request.key in excluded_pairs:
            continue
        if any(
            reason.shape_dependent
            for reason in explain_invalid_nt_hhs(request.candidate.canonical_params(), shape=request.shape)
        ):
            continue
        benchmark_state = None if ignore_cache else benchmark_states.get(request.key)
        validation_state = None if ignore_cache else validation_states.get(request.key)
        if (benchmark_state is not None and benchmark_state.reusable_negative) or validation_state == "failed":
            continue
        ok_samples = 0 if benchmark_state is None else benchmark_state.ok_samples
        remaining = request.min_samples - ok_samples
        if remaining <= 0:
            continue
        planned.append(
            PlannedPair(
                request=request,
                samples_to_collect=remaining,
                requires_validation=validation_state != "passed",
            )
        )
    return planned


@dataclass(frozen=True)
class _CandidateUnit:
    candidate: Candidate
    artifact_shapes: tuple[Shape, ...]
    pairs: tuple[PlannedPair, ...]
    evidence_stage: EvidenceStage

    @property
    def priority(self) -> float:
        return max(pair.request.priority for pair in self.pairs)


def _artifact_scope_by_candidate(
    requests: Sequence[PairRequest],
    artifact_shapes_by_candidate: Mapping[str, Sequence[Shape]] | None,
) -> dict[str, tuple[Shape, ...]]:
    requested_by_candidate: dict[str, dict[str, Shape]] = {}
    for request in requests:
        requested_by_candidate.setdefault(request.candidate.hash, {})[request.shape.id] = request.shape

    scopes: dict[str, tuple[Shape, ...]] = {}
    for candidate_hash, requested in requested_by_candidate.items():
        supplied = None if artifact_shapes_by_candidate is None else artifact_shapes_by_candidate.get(candidate_hash)
        if supplied is None:
            scopes[candidate_hash] = tuple(sorted(requested.values(), key=lambda shape: shape.id))
            continue
        scope_by_id: dict[str, Shape] = {}
        for shape in supplied:
            current = scope_by_id.get(shape.id)
            if current is not None and current != shape:
                raise ValueError(f"conflicting artifact shapes for {shape.id}")
            scope_by_id[shape.id] = shape
        missing = sorted(set(requested) - set(scope_by_id))
        if missing:
            raise ValueError(
                f"artifact scope for candidate {candidate_hash} does not cover requested shapes: {missing}"
            )
        scopes[candidate_hash] = tuple(sorted(scope_by_id.values(), key=lambda shape: shape.id))
    if artifact_shapes_by_candidate is not None:
        unknown = sorted(set(artifact_shapes_by_candidate) - set(requested_by_candidate))
        if unknown:
            raise ValueError(f"artifact scopes reference candidates without pair requests: {unknown}")
    return scopes


def _candidate_units(
    planned_pairs: Sequence[PlannedPair],
    *,
    scopes: Mapping[str, tuple[Shape, ...]],
    explicit_scope_hashes: set[str],
    shape_batch_size: int,
) -> list[_CandidateUnit]:
    by_candidate_stage: dict[tuple[str, EvidenceStage], list[PlannedPair]] = {}
    for pair in planned_pairs:
        key = (pair.request.candidate.hash, pair.request.evidence_stage)
        by_candidate_stage.setdefault(key, []).append(pair)

    units: list[_CandidateUnit] = []
    for (candidate_hash, evidence_stage), pairs in by_candidate_stage.items():
        candidate = pairs[0].request.candidate
        scope = scopes[candidate_hash]
        scope_chunks = (
            [list(scope)] if candidate_hash in explicit_scope_hashes else _chunks(list(scope), shape_batch_size)
        )
        for scope_chunk in scope_chunks:
            shape_ids = {shape.id for shape in scope_chunk}
            chunk_pairs = tuple(pair for pair in pairs if pair.request.shape.id in shape_ids)
            if chunk_pairs:
                units.append(
                    _CandidateUnit(
                        candidate=candidate,
                        artifact_shapes=tuple(scope_chunk),
                        pairs=chunk_pairs,
                        evidence_stage=evidence_stage,
                    )
                )
    return units


def plan_pair_requests(
    db: EvoTensileDB,
    *,
    requests: Sequence[PairRequest],
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    validation_protocol_hash: str,
    candidate_batch_size: int = 32,
    shape_batch_size: int = 100,
    ignore_cache: bool = False,
    max_batches: int | None = None,
    excluded_pairs: set[tuple[str, str]] | None = None,
    artifact_shapes_by_candidate: Mapping[str, Sequence[Shape]] | None = None,
) -> list[PlannedBatch]:
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    if shape_batch_size <= 0:
        raise ValueError("shape_batch_size must be positive")
    normalized = normalize_pair_requests(requests)
    if not normalized:
        return []
    scopes = _artifact_scope_by_candidate(normalized, artifact_shapes_by_candidate)
    planned_pairs = _planned_pairs(
        db,
        requests=normalized,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        validation_protocol_hash=validation_protocol_hash,
        ignore_cache=ignore_cache,
        excluded_pairs=excluded_pairs or set(),
    )
    units = _candidate_units(
        planned_pairs,
        scopes=scopes,
        explicit_scope_hashes=set(artifact_shapes_by_candidate or ()),
        shape_batch_size=shape_batch_size,
    )
    grouped: dict[tuple[EvidenceStage, tuple[str, ...]], list[_CandidateUnit]] = {}
    for unit in units:
        key = (unit.evidence_stage, tuple(shape.id for shape in unit.artifact_shapes))
        grouped.setdefault(key, []).append(unit)

    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            -max(unit.priority for unit in item[1]),
            item[0][0].value,
            item[0][1],
        ),
    )
    batches: list[PlannedBatch] = []
    for (evidence_stage, _), group_units in ordered_groups:
        ordered_units = sorted(group_units, key=lambda unit: -unit.priority)
        for chunk in _chunks(ordered_units, candidate_batch_size):
            pairs = tuple(
                sorted(
                    (pair for unit in chunk for pair in unit.pairs),
                    key=lambda pair: -pair.request.priority,
                )
            )
            batches.append(
                PlannedBatch(
                    batch_index=len(batches),
                    pairs=pairs,
                    artifact_candidates=tuple(unit.candidate for unit in chunk),
                    artifact_shapes=chunk[0].artifact_shapes,
                    evidence_stage=evidence_stage,
                )
            )
            if max_batches is not None and len(batches) >= max_batches:
                return batches
    return batches


def preprepare_probe_screened_pairs(
    db: EvoTensileDB,
    *,
    requests: Sequence[PairRequest],
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    probe_protocol_hash: str,
    policy: ProbePolicy,
) -> set[tuple[str, str]]:
    normalized = normalize_pair_requests(requests)
    allowed = {request.key for request in normalized}
    shape_ids = {request.shape.id for request in normalized}
    candidate_hashes = {request.candidate.hash for request in normalized}
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
        eligible = [item for item in stats if (shape_id, item.candidate_hash) in allowed]
        if len(eligible) <= policy.min_survivors:
            continue
        decision = decide_shape_probe(
            shape_id,
            eligible,
            policy=policy,
            reference_stats=reference_stats.get(shape_id, ()),
        )
        screened.update((shape_id, candidate_hash) for candidate_hash in decision.screened_hashes)
    return screened
