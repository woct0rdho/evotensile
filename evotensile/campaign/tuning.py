import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from evotensile.search.shape_clustering import ShapeClustering


@dataclass(frozen=True)
class PolicyTrialObservation:
    configuration_id: str
    seed: int
    ordering_id: str
    fold_id: str
    mean_log_regret: float
    p95_log_regret: float
    worst_log_regret: float
    unresolved_shapes: int
    queried_pairs: int
    unknown_pairs: int
    prepared_candidates: int

    def __post_init__(self) -> None:
        metrics = (self.mean_log_regret, self.p95_log_regret, self.worst_log_regret)
        if any(not math.isfinite(value) or value < 0.0 for value in metrics):
            raise ValueError("policy trial regrets must be finite and nonnegative")
        counts = (
            self.unresolved_shapes,
            self.queried_pairs,
            self.unknown_pairs,
            self.prepared_candidates,
        )
        if any(value < 0 for value in counts):
            raise ValueError("policy trial counts must be nonnegative")
        if not self.configuration_id or not self.ordering_id or not self.fold_id:
            raise ValueError("policy trial identities must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "configuration_id": self.configuration_id,
            "seed": self.seed,
            "ordering_id": self.ordering_id,
            "fold_id": self.fold_id,
            "mean_log_regret": self.mean_log_regret,
            "p95_log_regret": self.p95_log_regret,
            "worst_log_regret": self.worst_log_regret,
            "unresolved_shapes": self.unresolved_shapes,
            "queried_pairs": self.queried_pairs,
            "unknown_pairs": self.unknown_pairs,
            "prepared_candidates": self.prepared_candidates,
        }


@dataclass(frozen=True)
class PolicyAggregate:
    configuration_id: str
    trials: int
    seeds: int
    mean_log_regret: float
    p95_log_regret: float
    worst_log_regret: float
    mean_unresolved_shapes: float
    mean_prepared_candidates: float
    mean_unknown_pairs: float
    seed_mean_regret_variance: float
    pareto_optimal: bool = False
    robust_score: float | None = None

    def objective_values(self) -> tuple[float, ...]:
        return (
            self.mean_log_regret,
            self.p95_log_regret,
            self.worst_log_regret,
            self.mean_unresolved_shapes,
            self.mean_prepared_candidates,
            self.mean_unknown_pairs,
            self.seed_mean_regret_variance,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "configuration_id": self.configuration_id,
            "trials": self.trials,
            "seeds": self.seeds,
            "mean_log_regret": self.mean_log_regret,
            "p95_log_regret": self.p95_log_regret,
            "worst_log_regret": self.worst_log_regret,
            "mean_unresolved_shapes": self.mean_unresolved_shapes,
            "mean_prepared_candidates": self.mean_prepared_candidates,
            "mean_unknown_pairs": self.mean_unknown_pairs,
            "seed_mean_regret_variance": self.seed_mean_regret_variance,
            "pareto_optimal": self.pareto_optimal,
            "robust_score": self.robust_score,
        }


def mechanically_stratified_folds(
    clustering: ShapeClustering,
    *,
    fold_count: int,
) -> dict[str, tuple[str, ...]]:
    if fold_count <= 0:
        raise ValueError("mechanical fold count must be positive")
    folds: list[list[str]] = [[] for _ in range(fold_count)]
    for cluster in clustering.clusters:
        ordered = sorted(
            cluster.shape_ids,
            key=lambda shape_id: (cluster.distances_to_medoid[shape_id], shape_id),
        )
        for index, shape_id in enumerate(ordered):
            folds[index % fold_count].append(shape_id)
    return {f"fold_{index:02d}": tuple(sorted(shape_ids)) for index, shape_ids in enumerate(folds) if shape_ids}


def fold_regret_metrics(
    *,
    shape_ids: Sequence[str],
    oracle_best: Mapping[str, float],
    incumbent_performance: Mapping[str, float],
) -> tuple[float, float, float, int]:
    regrets = []
    unresolved = 0
    for shape_id in shape_ids:
        oracle = float(oracle_best[shape_id])
        incumbent = incumbent_performance.get(shape_id)
        if incumbent is None:
            unresolved += 1
            continue
        regrets.append(max(0.0, math.log(oracle / incumbent)))
    if not regrets:
        penalty = math.log(2.0)
        return penalty, penalty, penalty, unresolved
    return (
        statistics.fmean(regrets),
        _percentile(regrets, 0.95),
        max(regrets),
        unresolved,
    )


def aggregate_policy_trials(
    observations: Sequence[PolicyTrialObservation],
) -> tuple[PolicyAggregate, ...]:
    grouped: dict[str, list[PolicyTrialObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.configuration_id].append(observation)
    aggregates = []
    for configuration_id, rows in sorted(grouped.items()):
        seed_means: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            seed_means[row.seed].append(row.mean_log_regret)
        per_seed = [statistics.fmean(values) for values in seed_means.values()]
        aggregates.append(
            PolicyAggregate(
                configuration_id=configuration_id,
                trials=len(rows),
                seeds=len(seed_means),
                mean_log_regret=statistics.fmean(row.mean_log_regret for row in rows),
                p95_log_regret=statistics.fmean(row.p95_log_regret for row in rows),
                worst_log_regret=max(row.worst_log_regret for row in rows),
                mean_unresolved_shapes=statistics.fmean(row.unresolved_shapes for row in rows),
                mean_prepared_candidates=statistics.fmean(row.prepared_candidates for row in rows),
                mean_unknown_pairs=statistics.fmean(row.unknown_pairs for row in rows),
                seed_mean_regret_variance=statistics.pvariance(per_seed) if len(per_seed) > 1 else 0.0,
            )
        )
    pareto_ids = {
        aggregate.configuration_id
        for aggregate in aggregates
        if not any(
            _dominates(other.objective_values(), aggregate.objective_values())
            for other in aggregates
            if other.configuration_id != aggregate.configuration_id
        )
    }
    robust_scores = _robust_scores(aggregates)
    return tuple(
        replace(
            aggregate,
            pareto_optimal=aggregate.configuration_id in pareto_ids,
            robust_score=robust_scores[aggregate.configuration_id],
        )
        for aggregate in aggregates
    )


def select_robust_default(aggregates: Sequence[PolicyAggregate]) -> PolicyAggregate:
    if not aggregates:
        raise ValueError("robust policy selection requires aggregate results")
    pareto = [aggregate for aggregate in aggregates if aggregate.pareto_optimal]
    candidates = pareto or list(aggregates)
    return min(
        candidates,
        key=lambda aggregate: (
            math.inf if aggregate.robust_score is None else aggregate.robust_score,
            aggregate.worst_log_regret,
            aggregate.p95_log_regret,
            aggregate.configuration_id,
        ),
    )


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    tolerance = 1e-12
    return all(
        left_value <= right_value + tolerance for left_value, right_value in zip(left, right, strict=True)
    ) and any(left_value < right_value - tolerance for left_value, right_value in zip(left, right, strict=True))


def _robust_scores(aggregates: Sequence[PolicyAggregate]) -> dict[str, float]:
    if not aggregates:
        return {}
    objectives = [aggregate.objective_values() for aggregate in aggregates]
    minima = [min(values) for values in zip(*objectives, strict=True)]
    maxima = [max(values) for values in zip(*objectives, strict=True)]
    scores = {}
    for aggregate in aggregates:
        normalized = [
            0.0 if maximum == minimum else (value - minimum) / (maximum - minimum)
            for value, minimum, maximum in zip(
                aggregate.objective_values(),
                minima,
                maxima,
                strict=True,
            )
        ]
        scores[aggregate.configuration_id] = max(normalized) + 0.1 * statistics.fmean(normalized)
    return scores


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
