from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evotensile.database import EvaluationSummary


class GridObjective:
    SPECIALIST = "specialist"
    GENERALIST = "generalist"
    COVERAGE = "coverage"
    UNCERTAINTY = "uncertainty"


GRID_OBJECTIVES = (
    GridObjective.SPECIALIST,
    GridObjective.GENERALIST,
    GridObjective.COVERAGE,
    GridObjective.UNCERTAINTY,
)


@dataclass(frozen=True)
class CandidateGridScore:
    candidate_hash: str
    specialist_score: float
    generalist_score: float
    coverage_fraction: float
    unresolved_shape_count: int
    samples: int
    shape_count: int

    def objective_score(self, objective: str) -> float:
        if objective is GridObjective.SPECIALIST:
            return self.specialist_score
        if objective is GridObjective.GENERALIST:
            return self.generalist_score
        if objective is GridObjective.COVERAGE:
            return 1.0 - self.coverage_fraction
        if objective is GridObjective.UNCERTAINTY:
            return self.coverage_fraction
        raise ValueError(f"unsupported grid objective: {objective}")


def rank_percentiles(summaries: Sequence[EvaluationSummary]) -> dict[str, float]:
    ordered = [summary for summary in summaries if summary.median_gflops is not None and summary.median_gflops > 0.0]
    ordered.sort(key=lambda summary: (summary.median_gflops or 0.0, -(summary.median_time_us or 0.0)), reverse=True)
    denominator = max(len(ordered) - 1, 1)
    return {summary.candidate_hash: rank / denominator for rank, summary in enumerate(ordered)}


def candidate_grid_scores(
    summaries_by_shape: Mapping[str, Sequence[EvaluationSummary]],
    *,
    target_shape_ids: Sequence[str],
    elite_per_shape: int | None = None,
) -> dict[str, CandidateGridScore]:
    items_by_candidate: dict[str, list[tuple[str, float, int]]] = {}
    for shape_id in target_shape_ids:
        summaries = summaries_by_shape.get(shape_id, ())
        percentiles = rank_percentiles(summaries)
        selected = summaries if elite_per_shape is None else summaries[:elite_per_shape]
        for summary in selected:
            percentile = percentiles.get(summary.candidate_hash)
            if percentile is None:
                continue
            items_by_candidate.setdefault(summary.candidate_hash, []).append((shape_id, percentile, summary.samples))

    target_count = max(len(set(target_shape_ids)), 1)
    scores: dict[str, CandidateGridScore] = {}
    for candidate_hash, items in items_by_candidate.items():
        represented = len({shape_id for shape_id, _, _ in items})
        percentiles = [percentile for _, percentile, _ in items]
        unresolved = max(0, target_count - represented)
        scores[candidate_hash] = CandidateGridScore(
            candidate_hash=candidate_hash,
            specialist_score=min(percentiles),
            generalist_score=(sum(percentiles) + unresolved) / target_count,
            coverage_fraction=represented / target_count,
            unresolved_shape_count=unresolved,
            samples=sum(samples for _, _, samples in items),
            shape_count=represented,
        )
    return scores
