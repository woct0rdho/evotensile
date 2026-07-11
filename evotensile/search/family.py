import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..candidate import Candidate, Shape
from ..database import EvaluationSummary, EvoTensileDB
from ..search.encoding import candidate_to_genome, hamming_distance
from ..search.grid_evidence import CandidateGridScore, GridObjective, candidate_grid_scores
from ..search_space import (
    DOMAINS,
    MATRIX_INSTRUCTIONS,
    NT_HHS_RANDOM_VALU_VGPR_HEADROOM,
    _valu_vgpr_lower_bound,
    eligible_for_shape_scope,
    macro_tile,
    make_candidate,
    random_candidate,
    repair_linked_overrides,
)

FAMILY_DESCRIPTOR_VERSION = "nt_hhs_v2"
NT_HHS_PROFILE = "gfx1151-nt-hhs"
DEFAULT_FAMILY_ELITES_PER_CELL = 4
DEFAULT_FAMILY_DIVERSITY_SCORE_SLACK = 0.25


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
    objective: str
    aggregate_score: float
    specialist_score: float
    generalist_score: float
    coverage_fraction: float
    unresolved_shape_count: int
    samples: int
    shape_count: int
    observed_candidate_count: int
    status_counts: dict[str, int]
    family_rank: int = 1
    novelty_distance: int = 0

    @property
    def leader_candidate_hash(self) -> str:
        return self.leader.hash

    def summary(self) -> dict[str, Any]:
        return {
            "descriptor": self.descriptor.as_dict(),
            "leader_candidate_hash": self.leader_candidate_hash,
            "objective": self.objective,
            "aggregate_score": self.aggregate_score,
            "specialist_score": self.specialist_score,
            "generalist_score": self.generalist_score,
            "coverage_fraction": self.coverage_fraction,
            "unresolved_shape_count": self.unresolved_shape_count,
            "samples": self.samples,
            "shape_count": self.shape_count,
            "observed_candidate_count": self.observed_candidate_count,
            "family_rank": self.family_rank,
            "novelty_distance": self.novelty_distance,
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


def nt_hhs_family_cells() -> list[FamilyDescriptor]:
    tile_cells = {
        (_tile_area_log2(macro_tile0, macro_tile1), _tile_aspect(macro_tile0, macro_tile1))
        for macro_tile0, macro_tile1 in (macro_tile(instruction) for instruction in MATRIX_INSTRUCTIONS)
    }
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
        for global_split_u in DOMAINS["GlobalSplitU"]
    ]


def load_family_attempt_counts(
    db: EvoTensileDB,
    *,
    problem_type_hash: str | None = None,
    benchmark_protocol_hash: str | None = None,
    shapes: Sequence[Shape] | None = None,
    profile: str = NT_HHS_PROFILE,
) -> dict[str, int]:
    evaluation_clauses: list[str] = []
    evaluation_params: list[str] = []
    validation_clauses: list[str] = []
    validation_params: list[str] = []
    if problem_type_hash is not None:
        evaluation_clauses.append("problem_type_hash = ?")
        evaluation_params.append(problem_type_hash)
        validation_clauses.append("problem_type_hash = ?")
        validation_params.append(problem_type_hash)
    if benchmark_protocol_hash is not None:
        evaluation_clauses.append("benchmark_protocol_hash = ?")
        evaluation_params.append(benchmark_protocol_hash)
    if shapes:
        placeholders = ",".join("?" for _ in shapes)
        evaluation_clauses.append(f"shape_id IN ({placeholders})")
        evaluation_params.extend(shape.id for shape in shapes)
        validation_clauses.append(f"shape_id IN ({placeholders})")
        validation_params.extend(shape.id for shape in shapes)
    evaluation_where = "WHERE " + " AND ".join(evaluation_clauses) if evaluation_clauses else ""
    validation_where = "WHERE " + " AND ".join(validation_clauses) if validation_clauses else ""
    with db.connection() as con:
        evaluation_rows = con.execute(
            f"""
            SELECT DISTINCT candidate_hash
            FROM evaluations
            {evaluation_where}
            """,
            evaluation_params,
        ).fetchall()
        validation_rows = con.execute(
            f"""
            SELECT DISTINCT candidate_hash
            FROM validations
            {validation_where}
            """,
            validation_params,
        ).fetchall()
    candidate_hashes = {str(row["candidate_hash"]) for row in [*evaluation_rows, *validation_rows]}
    candidates = db.get_candidates(sorted(candidate_hashes))
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
        if not eligible_for_shape_scope(params, target_shapes):
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
    cells = nt_hhs_family_cells()
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
            objective=GridObjective.SPECIALIST,
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
    evaluation_clauses: list[str] = []
    evaluation_params: list[str] = []
    validation_clauses: list[str] = []
    validation_params: list[str] = []
    if problem_type_hash is not None:
        evaluation_clauses.append("problem_type_hash = ?")
        evaluation_params.append(problem_type_hash)
        validation_clauses.append("problem_type_hash = ?")
        validation_params.append(problem_type_hash)
    if benchmark_protocol_hash is not None:
        evaluation_clauses.append("benchmark_protocol_hash = ?")
        evaluation_params.append(benchmark_protocol_hash)
    if shape_ids:
        placeholders = ",".join("?" for _ in shape_ids)
        evaluation_clauses.append(f"shape_id IN ({placeholders})")
        evaluation_params.extend(shape_ids)
        validation_clauses.append(f"shape_id IN ({placeholders})")
        validation_params.extend(shape_ids)
    evaluation_where = "WHERE " + " AND ".join(evaluation_clauses) if evaluation_clauses else ""
    validation_where = "WHERE " + " AND ".join(validation_clauses) if validation_clauses else ""
    with db.connection() as con:
        evaluation_rows = con.execute(
            f"""
            SELECT candidate_hash, status, COUNT(*) AS n
            FROM evaluations
            {evaluation_where}
            GROUP BY candidate_hash, status
            """,
            evaluation_params,
        ).fetchall()
        validation_rows = con.execute(
            f"""
            SELECT candidate_hash, status, COUNT(*) AS n
            FROM validations
            {validation_where}
            GROUP BY candidate_hash, status
            """,
            validation_params,
        ).fetchall()

    candidate_hashes = sorted({str(row["candidate_hash"]) for row in [*evaluation_rows, *validation_rows]})
    candidates = {candidate.hash: candidate for candidate in db.get_candidates(candidate_hashes)}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in evaluation_rows:
        candidate = candidates.get(str(row["candidate_hash"]))
        if candidate is None:
            continue
        descriptor_key = family_descriptor(candidate, profile=profile).key
        counts[descriptor_key][str(row["status"])] += int(row["n"])
    for row in validation_rows:
        candidate = candidates.get(str(row["candidate_hash"]))
        if candidate is None:
            continue
        descriptor_key = family_descriptor(candidate, profile=profile).key
        counts[descriptor_key][f"validation_{row['status']}"] += int(row["n"])
    return {key: dict(status_counts) for key, status_counts in counts.items()}


@dataclass(frozen=True)
class _FamilyCandidateScore:
    grid_score: CandidateGridScore
    aggregate_score: float

    @property
    def candidate_hash(self) -> str:
        return self.grid_score.candidate_hash

    @property
    def samples(self) -> int:
        return self.grid_score.samples

    @property
    def shape_count(self) -> int:
        return self.grid_score.shape_count


def _select_diverse_family_scores(
    scores: Sequence[_FamilyCandidateScore],
    *,
    candidates: Mapping[str, Candidate],
    objective: str,
    count: int,
    score_slack: float,
) -> list[tuple[_FamilyCandidateScore, int]]:
    if count <= 0 or not scores:
        return []

    def quality_key(item: _FamilyCandidateScore) -> tuple[float, int, int, str]:
        coverage_tie = (
            item.shape_count
            if objective in {GridObjective.SPECIALIST, GridObjective.UNCERTAINTY}
            else -item.shape_count
        )
        return (item.aggregate_score, coverage_tie, -item.samples, item.candidate_hash)

    remaining = sorted(scores, key=quality_key)
    selected: list[tuple[_FamilyCandidateScore, int]] = [(remaining.pop(0), 0)]
    quality_limit = selected[0][0].aggregate_score + max(0.0, score_slack)
    while remaining and len(selected) < count:
        eligible = [item for item in remaining if item.aggregate_score <= quality_limit] or remaining
        selected_hashes = [item.candidate_hash for item, _ in selected]

        def selection_key(item: _FamilyCandidateScore) -> tuple[int, float, int, int, str]:
            genome = candidate_to_genome(candidates[item.candidate_hash])
            novelty = min(
                hamming_distance(genome, candidate_to_genome(candidates[selected_hash]))
                for selected_hash in selected_hashes
            )
            return (-novelty, *quality_key(item))

        chosen = min(eligible, key=selection_key)
        chosen_genome = candidate_to_genome(candidates[chosen.candidate_hash])
        novelty = min(
            hamming_distance(chosen_genome, candidate_to_genome(candidates[selected_hash]))
            for selected_hash in selected_hashes
        )
        selected.append((chosen, novelty))
        remaining.remove(chosen)
    return selected


def load_family_archive(
    db: EvoTensileDB,
    *,
    problem_type_hash: str | None = None,
    benchmark_protocol_hash: str | None = None,
    shapes: Sequence[Shape] | None = None,
    min_samples: int = 1,
    objective: str,
    profile: str = NT_HHS_PROFILE,
    limit: int | None = None,
    elites_per_family: int = 1,
    diversity_score_slack: float = DEFAULT_FAMILY_DIVERSITY_SCORE_SLACK,
) -> list[FamilyArchiveEntry]:
    if objective not in {
        GridObjective.SPECIALIST,
        GridObjective.GENERALIST,
        GridObjective.COVERAGE,
        GridObjective.UNCERTAINTY,
    }:
        raise ValueError(f"unknown family archive objective: {objective}")
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

    grid_scores = candidate_grid_scores(summaries_by_shape, target_shape_ids=shape_ids)
    candidate_scores: dict[str, dict[str, CandidateGridScore]] = defaultdict(dict)
    family_descriptors: dict[str, FamilyDescriptor] = {}
    for candidate_hash, grid_score in grid_scores.items():
        candidate = candidates.get(candidate_hash)
        if candidate is None:
            continue
        descriptor = family_descriptor(candidate, profile=profile)
        family_descriptors[descriptor.key] = descriptor
        candidate_scores[descriptor.key][candidate_hash] = grid_score

    entries: list[FamilyArchiveEntry] = []
    for descriptor_key, scores_by_candidate in candidate_scores.items():
        family_scores = [
            _FamilyCandidateScore(
                grid_score=grid_score,
                aggregate_score=grid_score.objective_score(objective),
            )
            for grid_score in scores_by_candidate.values()
        ]
        selected = _select_diverse_family_scores(
            family_scores,
            candidates=candidates,
            objective=objective,
            count=elites_per_family,
            score_slack=diversity_score_slack,
        )
        for family_rank, (score, novelty_distance) in enumerate(selected, start=1):
            entries.append(
                FamilyArchiveEntry(
                    descriptor=family_descriptors[descriptor_key],
                    leader=candidates[score.candidate_hash],
                    objective=objective,
                    aggregate_score=score.aggregate_score,
                    specialist_score=score.grid_score.specialist_score,
                    generalist_score=score.grid_score.generalist_score,
                    coverage_fraction=score.grid_score.coverage_fraction,
                    unresolved_shape_count=score.grid_score.unresolved_shape_count,
                    samples=score.samples,
                    shape_count=score.shape_count,
                    observed_candidate_count=len(scores_by_candidate),
                    status_counts=status_counts.get(descriptor_key, {}),
                    family_rank=family_rank,
                    novelty_distance=novelty_distance,
                )
            )
    entries.sort(
        key=lambda entry: (
            entry.family_rank,
            entry.aggregate_score,
            -entry.samples,
            -entry.shape_count,
            entry.leader_candidate_hash,
        )
    )
    return entries[:limit] if limit is not None else entries
