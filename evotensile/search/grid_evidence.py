import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evotensile.database import BenchmarkSummary


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


def rank_percentiles(summaries: Sequence[BenchmarkSummary]) -> dict[str, float]:
    ordered = [summary for summary in summaries if summary.median_gflops is not None and summary.median_gflops > 0.0]
    ordered.sort(key=lambda summary: (summary.median_gflops or 0.0, -(summary.median_time_us or 0.0)), reverse=True)
    denominator = max(len(ordered) - 1, 1)
    return {summary.candidate_hash: rank / denominator for rank, summary in enumerate(ordered)}


def candidate_grid_scores(
    summaries_by_shape: Mapping[str, Sequence[BenchmarkSummary]],
    *,
    target_shape_ids: Sequence[str],
    elite_per_shape: int | None = None,
    shape_weights: Mapping[str, float] | None = None,
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

    unique_shape_ids = tuple(dict.fromkeys(target_shape_ids))
    target_count = max(len(unique_shape_ids), 1)
    weights = {shape_id: 1.0 for shape_id in unique_shape_ids}
    if shape_weights is not None:
        if set(shape_weights) != set(unique_shape_ids):
            raise ValueError("grid score weights must cover the exact target shape set")
        weights = {shape_id: float(shape_weights[shape_id]) for shape_id in unique_shape_ids}
        if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
            raise ValueError("grid score weights must be finite and nonnegative")
    total_weight = sum(weights.values())
    if total_weight <= 0.0:
        raise ValueError("grid score weights must contain positive mass")
    scores: dict[str, CandidateGridScore] = {}
    for candidate_hash, items in items_by_candidate.items():
        represented_shape_ids = {shape_id for shape_id, _, _ in items}
        represented = len(represented_shape_ids)
        percentiles = [percentile for _, percentile, _ in items]
        unresolved = max(0, target_count - represented)
        represented_weight = sum(weights[shape_id] for shape_id in represented_shape_ids)
        weighted_percentiles = sum(percentile * weights[shape_id] for shape_id, percentile, _ in items)
        scores[candidate_hash] = CandidateGridScore(
            candidate_hash=candidate_hash,
            specialist_score=min(percentiles),
            generalist_score=(weighted_percentiles + total_weight - represented_weight) / total_weight,
            coverage_fraction=represented_weight / total_weight,
            unresolved_shape_count=unresolved,
            samples=sum(samples for _, _, samples in items),
            shape_count=represented,
        )
    return scores
