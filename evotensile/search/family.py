import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..candidate import Candidate, Shape
from ..database import EvaluationSummary, EvoTensileDB
from ..search_space import (
    DOMAINS,
    MATRIX_INSTRUCTIONS,
    NT_HHS_RANDOM_VALU_VGPR_HEADROOM,
    _valu_vgpr_lower_bound,
    cheap_constraints,
    macro_tile,
    make_candidate,
    random_candidate,
    repair_linked_overrides,
)

FAMILY_DESCRIPTOR_VERSION = "nt_hhs_v2"
NT_HHS_PROFILE = "gfx1151-nt-hhs"


@dataclass(frozen=True)
class FamilyDescriptor:
    profile: str
    version: str
    fields: tuple[tuple[str, Any], ...]

    @property
    def key(self) -> str:
        items = ",".join(f"{name}={_format_value(value)}" for name, value in self.fields)
        return f"{self.profile}:{self.version}:{items}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "version": self.version,
            "key": self.key,
            "fields": {name: value for name, value in self.fields},
        }


@dataclass(frozen=True)
class FamilyArchiveEntry:
    descriptor: FamilyDescriptor
    leader: Candidate
    aggregate_score: float
    samples: int
    shape_count: int
    observed_candidate_count: int
    status_counts: dict[str, int]

    @property
    def leader_candidate_hash(self) -> str:
        return self.leader.hash

    def summary(self) -> dict[str, Any]:
        return {
            "descriptor": self.descriptor.as_dict(),
            "leader_candidate_hash": self.leader_candidate_hash,
            "aggregate_score": self.aggregate_score,
            "samples": self.samples,
            "shape_count": self.shape_count,
            "observed_candidate_count": self.observed_candidate_count,
            "status_counts": dict(sorted(self.status_counts.items())),
        }


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (tuple, list)):
        return "x".join(str(item) for item in value)
    return str(value)


def _field(params: Mapping[str, Any], name: str, default: Any) -> Any:
    return params.get(name, default)


def _tile_area_log2(macro_tile0: int, macro_tile1: int) -> int:
    return int(math.floor(math.log2(macro_tile0 * macro_tile1)))


def _tile_aspect(macro_tile0: int, macro_tile1: int) -> str:
    ratio = macro_tile0 / macro_tile1
    if ratio >= 2.0:
        return "m_major"
    if ratio <= 0.5:
        return "n_major"
    return "balanced"


def nt_hhs_family_descriptor(candidate: Candidate | Mapping[str, Any]) -> FamilyDescriptor:
    params = candidate.canonical_params() if isinstance(candidate, Candidate) else dict(candidate)
    macro_tile0, macro_tile1 = macro_tile(params["MatrixInstruction"])
    fields: tuple[tuple[str, Any], ...] = (
        ("TileAreaLog2", _tile_area_log2(macro_tile0, macro_tile1)),
        ("TileAspect", _tile_aspect(macro_tile0, macro_tile1)),
        ("TransposeLDS", int(_field(params, "TransposeLDS", 0))),
        ("GlobalSplitU", int(_field(params, "GlobalSplitU", 1))),
    )
    return FamilyDescriptor(profile=NT_HHS_PROFILE, version=FAMILY_DESCRIPTOR_VERSION, fields=fields)


def family_descriptor(candidate: Candidate | Mapping[str, Any], *, profile: str = NT_HHS_PROFILE) -> FamilyDescriptor:
    if profile != NT_HHS_PROFILE:
        raise ValueError(f"unsupported family descriptor profile: {profile}")
    return nt_hhs_family_descriptor(candidate)


def family_descriptor_counts(
    candidates: Iterable[Candidate],
    *,
    profile: str = NT_HHS_PROFILE,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        descriptor = family_descriptor(candidate, profile=profile)
        counts[descriptor.key] = counts.get(descriptor.key, 0) + 1
    return counts


def _target_tile_aspect(shapes: Sequence[Shape] | None) -> str:
    if not shapes:
        return "balanced"
    mean_log_ratio = sum(math.log2(shape.m / shape.n) for shape in shapes) / len(shapes)
    if mean_log_ratio >= 1.0:
        return "m_major"
    if mean_log_ratio <= -1.0:
        return "n_major"
    return "balanced"


def nt_hhs_family_cells(*, target_shapes: Sequence[Shape] | None = None) -> list[FamilyDescriptor]:
    tile_cells = {
        (_tile_area_log2(macro_tile0, macro_tile1), _tile_aspect(macro_tile0, macro_tile1))
        for macro_tile0, macro_tile1 in (macro_tile(instruction) for instruction in MATRIX_INSTRUCTIONS)
    }
    global_split_u_values = (1,) if target_shapes else tuple(int(value) for value in DOMAINS["GlobalSplitU"])
    return [
        FamilyDescriptor(
            profile=NT_HHS_PROFILE,
            version=FAMILY_DESCRIPTOR_VERSION,
            fields=(
                ("TileAreaLog2", area),
                ("TileAspect", aspect),
                ("TransposeLDS", transpose_lds),
                ("GlobalSplitU", global_split_u),
            ),
        )
        for area, aspect in sorted(tile_cells)
        for transpose_lds in (0, 2)
        for global_split_u in global_split_u_values
    ]


def load_family_attempt_counts(
    db: EvoTensileDB,
    *,
    problem_type_hash: str | None = None,
    benchmark_protocol_hash: str | None = None,
    shapes: Sequence[Shape] | None = None,
    profile: str = NT_HHS_PROFILE,
) -> dict[str, int]:
    clauses: list[str] = []
    params: list[str] = []
    if problem_type_hash is not None:
        clauses.append("problem_type_hash = ?")
        params.append(problem_type_hash)
    if benchmark_protocol_hash is not None:
        clauses.append("benchmark_protocol_hash = ?")
        params.append(benchmark_protocol_hash)
    if shapes:
        placeholders = ",".join("?" for _ in shapes)
        clauses.append(f"shape_id IN ({placeholders})")
        params.extend(shape.id for shape in shapes)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    with db.connection() as con:
        rows = con.execute(
            f"""
            SELECT DISTINCT candidate_hash
            FROM evaluations
            {where}
            """,
            params,
        ).fetchall()
    candidates = db.get_candidates(sorted(str(row["candidate_hash"]) for row in rows))
    return family_descriptor_counts(candidates, profile=profile)


def _candidate_for_family(
    descriptor: FamilyDescriptor,
    *,
    rng: random.Random,
    target_shapes: Sequence[Shape] | None,
    exclude: set[str],
) -> Candidate | None:
    fields = dict(descriptor.fields)
    matching_instructions = [
        instruction
        for instruction in MATRIX_INSTRUCTIONS
        if _tile_area_log2(*macro_tile(instruction)) == fields["TileAreaLog2"]
        and _tile_aspect(*macro_tile(instruction)) == fields["TileAspect"]
    ]
    rng.shuffle(matching_instructions)
    for _ in range(256):
        base = random_candidate(
            rng,
            target_shapes=target_shapes,
            transpose_lds=int(fields["TransposeLDS"]),
        )
        params = base.canonical_params()
        params["MatrixInstruction"] = rng.choice(matching_instructions)
        params["GlobalSplitU"] = int(fields["GlobalSplitU"])
        params["TransposeLDS"] = int(fields["TransposeLDS"])
        params = repair_linked_overrides(params)
        if _valu_vgpr_lower_bound(params) > NT_HHS_RANDOM_VALU_VGPR_HEADROOM:
            continue
        if target_shapes and not all(cheap_constraints(params, shape=shape) for shape in target_shapes):
            continue
        try:
            candidate = make_candidate(params, source="random")
        except ValueError:
            continue
        if candidate.hash in exclude or family_descriptor(candidate) != descriptor:
            continue
        return candidate
    return None


def family_stratified_random_candidates(
    db: EvoTensileDB,
    count: int,
    *,
    seed: int,
    target_shapes: Sequence[Shape] | None,
    problem_type_hash: str | None,
    benchmark_protocol_hash: str | None,
    profile: str = NT_HHS_PROFILE,
) -> list[Candidate]:
    if count <= 0:
        return []
    if profile != NT_HHS_PROFILE:
        raise ValueError(f"unsupported family descriptor profile: {profile}")

    rng = random.Random(seed)
    cells = nt_hhs_family_cells(target_shapes=target_shapes)
    counts = load_family_attempt_counts(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        shapes=target_shapes,
        profile=profile,
    )
    target_aspect = _target_tile_aspect(target_shapes)
    positive_family_keys = {
        entry.descriptor.key
        for entry in load_family_archive(
            db,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            shapes=target_shapes,
            min_samples=1,
            profile=profile,
            limit=None,
        )
    }
    out: dict[str, Candidate] = {}
    unavailable: set[str] = set()
    while len(out) < count:
        available = [cell for cell in cells if cell.key not in unavailable]
        prefer_target_aspect = bool(target_shapes) and len(out) % 2 == 0
        if prefer_target_aspect:
            target_cells = [cell for cell in available if dict(cell.fields)["TileAspect"] == target_aspect]
            if target_cells:
                available = target_cells
        if not available:
            break
        tie_break = {cell.key: rng.random() for cell in available}

        def priority(cell: FamilyDescriptor) -> tuple[int, int, float]:
            attempts = counts.get(cell.key, 0)
            has_positive = cell.key in positive_family_keys
            if not has_positive and attempts == 1:
                tier = 0
            elif not has_positive and attempts == 0:
                tier = 1
            elif not has_positive:
                tier = 2
            else:
                tier = 3
            return (tier, attempts, tie_break[cell.key])

        cell = min(available, key=priority)
        candidate = _candidate_for_family(
            cell,
            rng=rng,
            target_shapes=target_shapes,
            exclude=set(out),
        )
        if candidate is None:
            unavailable.add(cell.key)
            continue
        out[candidate.hash] = candidate
        counts[cell.key] = counts.get(cell.key, 0) + 1
    if len(out) < count:
        raise RuntimeError(f"failed to generate {count} family-stratified candidates; generated {len(out)}")
    return list(out.values())


def _rank_percentiles(summaries: Sequence[EvaluationSummary]) -> dict[str, float]:
    ordered = [summary for summary in summaries if summary.median_gflops is not None and summary.median_gflops > 0.0]
    if not ordered:
        return {}
    ordered.sort(key=lambda summary: (summary.median_gflops or 0.0, -(summary.median_time_us or 0.0)), reverse=True)
    denominator = max(len(ordered) - 1, 1)
    return {summary.candidate_hash: rank / denominator for rank, summary in enumerate(ordered)}


def _archive_shape_ids(
    db: EvoTensileDB,
    *,
    problem_type_hash: str | None,
    benchmark_protocol_hash: str | None,
    shapes: Sequence[Shape] | None,
) -> list[str]:
    if shapes is not None:
        return [shape.id for shape in shapes]
    clauses: list[str] = []
    params: list[str] = []
    if problem_type_hash is not None:
        clauses.append("problem_type_hash = ?")
        params.append(problem_type_hash)
    if benchmark_protocol_hash is not None:
        clauses.append("benchmark_protocol_hash = ?")
        params.append(benchmark_protocol_hash)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    with db.connection() as con:
        rows = con.execute(
            f"""
            SELECT DISTINCT shape_id
            FROM evaluations
            {where}
            ORDER BY shape_id
            """,
            params,
        ).fetchall()
    return [str(row["shape_id"]) for row in rows]


def _family_status_counts(
    db: EvoTensileDB,
    *,
    problem_type_hash: str | None,
    benchmark_protocol_hash: str | None,
    shape_ids: Sequence[str],
    profile: str,
) -> dict[str, dict[str, int]]:
    clauses: list[str] = []
    params: list[str] = []
    if problem_type_hash is not None:
        clauses.append("problem_type_hash = ?")
        params.append(problem_type_hash)
    if benchmark_protocol_hash is not None:
        clauses.append("benchmark_protocol_hash = ?")
        params.append(benchmark_protocol_hash)
    if shape_ids:
        placeholders = ",".join("?" for _ in shape_ids)
        clauses.append(f"shape_id IN ({placeholders})")
        params.extend(shape_ids)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    with db.connection() as con:
        rows = con.execute(
            f"""
            SELECT candidate_hash, status, COUNT(*) AS n
            FROM evaluations
            {where}
            GROUP BY candidate_hash, status
            """,
            params,
        ).fetchall()

    candidate_hashes = sorted({str(row["candidate_hash"]) for row in rows})
    candidates = {candidate.hash: candidate for candidate in db.get_candidates(candidate_hashes)}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        candidate = candidates.get(str(row["candidate_hash"]))
        if candidate is None:
            continue
        descriptor_key = family_descriptor(candidate, profile=profile).key
        counts[descriptor_key][str(row["status"])] += int(row["n"])
    return {key: dict(status_counts) for key, status_counts in counts.items()}


def load_family_archive(
    db: EvoTensileDB,
    *,
    problem_type_hash: str | None = None,
    benchmark_protocol_hash: str | None = None,
    shapes: Sequence[Shape] | None = None,
    min_samples: int = 1,
    profile: str = NT_HHS_PROFILE,
    limit: int | None = None,
) -> list[FamilyArchiveEntry]:
    shape_ids = _archive_shape_ids(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        shapes=shapes,
    )
    if not shape_ids:
        return []

    summaries_by_shape: dict[str, list[EvaluationSummary]] = {}
    candidate_hashes: set[str] = set()
    for shape_id in shape_ids:
        summaries = db.rank_evaluations(
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            shape_id=shape_id,
            min_samples=min_samples,
            limit=None,
        )
        summaries_by_shape[shape_id] = summaries
        candidate_hashes.update(summary.candidate_hash for summary in summaries)

    candidates = {candidate.hash: candidate for candidate in db.get_candidates(sorted(candidate_hashes))}
    status_counts = _family_status_counts(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        shape_ids=shape_ids,
        profile=profile,
    )

    candidate_scores: dict[str, dict[str, list[tuple[str, float, int]]]] = defaultdict(lambda: defaultdict(list))
    family_descriptors: dict[str, FamilyDescriptor] = {}
    for shape_id, summaries in summaries_by_shape.items():
        percentiles = _rank_percentiles(summaries)
        for summary in summaries:
            candidate = candidates.get(summary.candidate_hash)
            percentile = percentiles.get(summary.candidate_hash)
            if candidate is None or percentile is None:
                continue
            descriptor = family_descriptor(candidate, profile=profile)
            family_descriptors[descriptor.key] = descriptor
            candidate_scores[descriptor.key][candidate.hash].append((shape_id, percentile, summary.samples))

    entries: list[FamilyArchiveEntry] = []
    for descriptor_key, scores_by_candidate in candidate_scores.items():
        best_hash = min(
            scores_by_candidate,
            key=lambda candidate_hash: (
                sum(score for _, score, _ in scores_by_candidate[candidate_hash])
                / len(scores_by_candidate[candidate_hash]),
                -sum(samples for _, _, samples in scores_by_candidate[candidate_hash]),
                candidate_hash,
            ),
        )
        best_items = scores_by_candidate[best_hash]
        leader = candidates[best_hash]
        entries.append(
            FamilyArchiveEntry(
                descriptor=family_descriptors[descriptor_key],
                leader=leader,
                aggregate_score=sum(score for _, score, _ in best_items) / len(best_items),
                samples=sum(samples for _, _, samples in best_items),
                shape_count=len({shape_id for shape_id, _, _ in best_items}),
                observed_candidate_count=len(scores_by_candidate),
                status_counts=status_counts.get(descriptor_key, {}),
            )
        )
    entries.sort(
        key=lambda entry: (
            entry.aggregate_score,
            -entry.samples,
            -entry.shape_count,
            entry.leader_candidate_hash,
        )
    )
    return entries[:limit] if limit is not None else entries
