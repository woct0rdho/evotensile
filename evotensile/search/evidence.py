from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from evotensile.candidate import Candidate, Shape
from evotensile.database import EvaluationSummary, EvoTensileDB, ProposalOccurrence
from evotensile.search.evaluation_cost import CandidateEvaluationCost, load_candidate_evaluation_costs


@dataclass(frozen=True)
class ProposalEvidenceSnapshot:
    problem_type_hash: str | None
    benchmark_protocol_hash: str | None
    shape_ids: tuple[str, ...]
    summaries: tuple[EvaluationSummary, ...]
    summaries_by_shape: Mapping[str, tuple[EvaluationSummary, ...]]
    candidates: Mapping[str, Candidate]
    selected_occurrences: tuple[ProposalOccurrence, ...]
    latest_positive_times: Mapping[tuple[str, str], float]
    candidate_costs: Mapping[str, CandidateEvaluationCost]
    evaluation_status_counts: Mapping[str, Mapping[str, int]]

    def shape_summaries(self, shape_id: str | None = None) -> tuple[EvaluationSummary, ...]:
        if shape_id is None:
            return self.summaries
        return self.summaries_by_shape.get(shape_id, ())


def load_proposal_evidence_snapshot(
    db: EvoTensileDB,
    *,
    problem_type_hash: str | None,
    benchmark_protocol_hash: str | None,
    shapes: Sequence[Shape] | None,
) -> ProposalEvidenceSnapshot:
    allowed_shape_ids = None if shapes is None else {shape.id for shape in shapes}
    summaries = tuple(
        db.rank_evaluations(
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            min_samples=1,
            limit=None,
        )
    )
    by_shape: dict[str, list[EvaluationSummary]] = defaultdict(list)
    for summary in summaries:
        by_shape[summary.shape_id].append(summary)

    if problem_type_hash is None or benchmark_protocol_hash is None:
        occurrences: tuple[ProposalOccurrence, ...] = ()
        latest_positive_times: dict[tuple[str, str], float] = {}
    else:
        occurrences = tuple(
            db.proposal_occurrences(
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                selected_only=True,
            )
        )
        latest_positive_times = db.latest_positive_evaluation_times(
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
        )

    evaluation_status_counts: dict[str, dict[str, int]] = defaultdict(dict)
    if problem_type_hash is not None:
        clauses = ["problem_type_hash = ?"]
        params: list[str] = [problem_type_hash]
        if benchmark_protocol_hash is not None:
            clauses.append("benchmark_protocol_hash = ?")
            params.append(benchmark_protocol_hash)
        if allowed_shape_ids is not None:
            if allowed_shape_ids:
                placeholders = ",".join("?" for _ in allowed_shape_ids)
                clauses.append(f"shape_id IN ({placeholders})")
                params.extend(sorted(allowed_shape_ids))
            else:
                clauses.append("0")
        with db.connection() as con:
            rows = con.execute(
                f"""
                SELECT candidate_hash, status, COUNT(*) AS n
                FROM evaluations
                WHERE {" AND ".join(clauses)}
                GROUP BY candidate_hash, status
                """,
                params,
            ).fetchall()
        for row in rows:
            evaluation_status_counts[str(row["candidate_hash"])][str(row["status"])] = int(row["n"])

        validation_clauses = ["problem_type_hash = ?"]
        validation_params: list[str] = [problem_type_hash]
        if allowed_shape_ids is not None:
            if allowed_shape_ids:
                placeholders = ",".join("?" for _ in allowed_shape_ids)
                validation_clauses.append(f"shape_id IN ({placeholders})")
                validation_params.extend(sorted(allowed_shape_ids))
            else:
                validation_clauses.append("0")
        with db.connection() as con:
            validation_rows = con.execute(
                f"""
                SELECT candidate_hash, status, COUNT(*) AS n
                FROM validations
                WHERE {" AND ".join(validation_clauses)}
                GROUP BY candidate_hash, status
                """,
                validation_params,
            ).fetchall()
        for row in validation_rows:
            evaluation_status_counts[str(row["candidate_hash"])][f"validation_{row['status']}"] = int(row["n"])

    candidate_hashes = (
        {summary.candidate_hash for summary in summaries}
        | {occurrence.candidate_hash for occurrence in occurrences}
        | {parent_hash for occurrence in occurrences for parent_hash in occurrence.parent_hashes}
        | set(evaluation_status_counts)
    )
    candidates = {candidate.hash: candidate for candidate in db.get_candidates(sorted(candidate_hashes))}
    return ProposalEvidenceSnapshot(
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        shape_ids=tuple(shape.id for shape in shapes or ()),
        summaries=summaries,
        summaries_by_shape=MappingProxyType({shape_id: tuple(items) for shape_id, items in by_shape.items()}),
        candidates=MappingProxyType(candidates),
        selected_occurrences=occurrences,
        latest_positive_times=MappingProxyType(latest_positive_times),
        candidate_costs=MappingProxyType(load_candidate_evaluation_costs(db)),
        evaluation_status_counts=MappingProxyType(
            {
                candidate_hash: MappingProxyType(dict(status_counts))
                for candidate_hash, status_counts in evaluation_status_counts.items()
            }
        ),
    )
