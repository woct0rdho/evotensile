from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from evotensile.candidate import Candidate, Shape
from evotensile.database import BenchmarkSummary, EvoTensileDB, ProposalOccurrence
from evotensile.search.measured_cost import CandidateMeasuredCost, load_candidate_measured_costs


@dataclass(frozen=True)
class ProposalEvidenceSnapshot:
    problem_type_hash: str
    benchmark_protocol_hash: str
    shape_ids: tuple[str, ...]
    summaries: tuple[BenchmarkSummary, ...]
    summaries_by_shape: Mapping[str, tuple[BenchmarkSummary, ...]]
    candidates: Mapping[str, Candidate]
    selected_occurrences: tuple[ProposalOccurrence, ...]
    latest_positive_times: Mapping[tuple[str, str], float]
    candidate_costs: Mapping[str, CandidateMeasuredCost]
    evidence_status_counts: Mapping[str, Mapping[str, int]]

    def shape_summaries(self, shape_id: str | None = None) -> tuple[BenchmarkSummary, ...]:
        if shape_id is None:
            return self.summaries
        return self.summaries_by_shape.get(shape_id, ())


def load_proposal_evidence_snapshot(
    db: EvoTensileDB,
    *,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    shapes: Sequence[Shape] | None,
) -> ProposalEvidenceSnapshot:
    allowed_shape_ids = None if shapes is None else {shape.id for shape in shapes}
    summaries = tuple(
        db.rank_benchmarks(
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            min_samples=1,
            limit=None,
        )
    )
    by_shape: dict[str, list[BenchmarkSummary]] = defaultdict(list)
    for summary in summaries:
        by_shape[summary.shape_id].append(summary)

    occurrences = tuple(
        db.proposal_candidate_occurrences(
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            selected_only=True,
        )
    )
    latest_positive_times = db.latest_positive_benchmark_times(
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
    )
    evidence_status_counts = db.evidence_status_counts(
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        shape_ids=allowed_shape_ids,
    )

    candidate_hashes = (
        {summary.candidate_hash for summary in summaries}
        | {occurrence.candidate_hash for occurrence in occurrences}
        | {parent_hash for occurrence in occurrences for parent_hash in occurrence.parent_hashes}
        | set(evidence_status_counts)
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
        candidate_costs=MappingProxyType(load_candidate_measured_costs(db)),
        evidence_status_counts=MappingProxyType(
            {
                candidate_hash: MappingProxyType(dict(status_counts))
                for candidate_hash, status_counts in evidence_status_counts.items()
            }
        ),
    )
