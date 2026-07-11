import math
from dataclasses import dataclass

from evotensile.candidate import Candidate, Shape
from evotensile.database import EvaluationSummary, EvoTensileDB
from evotensile.search.shape_neighborhoods import shape_distance, shape_feature_delta
from evotensile.shapes import shape_from_id


@dataclass(frozen=True)
class ShapeOutlier:
    shape: Shape
    candidate_hash: str
    samples: int
    median_gflops: float
    predicted_neighbor_gflops: float
    residual_pct: float
    neighbor_shape_ids: tuple[str, ...]
    neighbor_candidate_hashes: tuple[str, ...]


def _weighted_quantile(values: list[tuple[float, float]], quantile: float) -> float | None:
    if not values:
        return None
    q = min(max(quantile, 0.0), 1.0)
    ordered = sorted(values, key=lambda item: item[0])
    total_weight = sum(weight for _, weight in ordered)
    if total_weight <= 0:
        return ordered[len(ordered) // 2][0]
    threshold = total_weight * q
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    size = len(rhs)
    augmented = [row[:] + [rhs_value] for row, rhs_value in zip(matrix, rhs, strict=True)]
    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row_index: abs(augmented[row_index][pivot_index]))
        pivot = augmented[pivot_row][pivot_index]
        if abs(pivot) < 1e-12:
            return None
        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        scale = augmented[pivot_index][pivot_index]
        for col in range(pivot_index, size + 1):
            augmented[pivot_index][col] /= scale
        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            if factor == 0.0:
                continue
            for col in range(pivot_index, size + 1):
                augmented[row_index][col] -= factor * augmented[pivot_index][col]
    return [augmented[row_index][size] for row_index in range(size)]


def _weighted_local_linear_prediction(
    target: Shape,
    nearest: list[tuple[float, Shape, EvaluationSummary]],
) -> float | None:
    if len(nearest) < 3:
        return None
    dimension = len(shape_feature_delta(target, nearest[0][1])) + 1
    matrix = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    rhs = [0.0 for _ in range(dimension)]
    total_weight = 0.0
    neighbor_logs: list[float] = []
    for distance, other_shape, summary in nearest:
        if summary.median_gflops is None or summary.median_gflops <= 0:
            continue
        y = math.log(summary.median_gflops)
        row = [1.0, *shape_feature_delta(target, other_shape)]
        weight = 1.0 / max(distance, 0.125)
        total_weight += weight
        neighbor_logs.append(y)
        for row_index, row_value in enumerate(row):
            rhs[row_index] += weight * row_value * y
            for col_index, col_value in enumerate(row):
                matrix[row_index][col_index] += weight * row_value * col_value
    if total_weight <= 0.0 or not neighbor_logs:
        return None
    ridge = total_weight * 1e-3
    for index in range(1, dimension):
        matrix[index][index] += ridge
    coefficients = _solve_linear_system(matrix, rhs)
    if coefficients is None:
        return None
    return min(max(coefficients[0], min(neighbor_logs)), max(neighbor_logs))


def _winner_summaries_by_shape(
    db: EvoTensileDB,
    *,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    min_samples: int,
) -> dict[str, EvaluationSummary]:
    winners: dict[str, EvaluationSummary] = {}
    for summary in db.rank_evaluations(
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        min_samples=min_samples,
    ):
        if summary.median_gflops is None or summary.median_gflops <= 0:
            continue
        winners.setdefault(summary.shape_id, summary)
    return winners


def detect_underperforming_shapes(
    db: EvoTensileDB,
    *,
    shapes: list[Shape],
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    min_samples: int = 1,
    neighbor_count: int = 8,
    envelope_quantile: float = 0.75,
    threshold_pct: float = 5.0,
    max_shapes: int | None = None,
) -> list[ShapeOutlier]:
    """Find shapes whose best measured candidate is below a local neighbor envelope."""
    if neighbor_count <= 0:
        return []
    winners = _winner_summaries_by_shape(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        min_samples=min_samples,
    )
    shape_by_id: dict[str, Shape] = {}
    for shape_id in winners:
        try:
            shape_by_id[shape_id] = shape_from_id(shape_id)
        except ValueError:
            continue

    outliers: list[ShapeOutlier] = []
    threshold_log = math.log1p(threshold_pct / 100.0)
    targets = shapes or sorted(shape_by_id.values(), key=lambda shape: shape.id)
    for shape in targets:
        summary = winners.get(shape.id)
        if summary is None:
            continue
        median_gflops = summary.median_gflops
        if median_gflops is None or median_gflops <= 0:
            continue
        neighbor_items: list[tuple[float, Shape, EvaluationSummary]] = []
        for other_id, other_summary in winners.items():
            if other_id == shape.id or other_summary.median_gflops is None or other_summary.median_gflops <= 0:
                continue
            other_shape = shape_by_id.get(other_id)
            if other_shape is None:
                continue
            neighbor_items.append((shape_distance(shape, other_shape), other_shape, other_summary))
        nearest = sorted(neighbor_items, key=lambda item: (item[0], item[1].id))[:neighbor_count]
        if not nearest:
            continue
        weighted_logs = [
            (math.log(other_summary.median_gflops), 1.0 / max(distance, 0.125))
            for distance, _, other_summary in nearest
            if other_summary.median_gflops is not None and other_summary.median_gflops > 0
        ]
        envelope_log = _weighted_quantile(weighted_logs, envelope_quantile)
        predicted_log = _weighted_local_linear_prediction(shape, nearest)
        if envelope_log is None and predicted_log is None:
            continue
        if envelope_log is not None and predicted_log is not None:
            predicted_log = min(predicted_log, envelope_log)
        elif predicted_log is None:
            predicted_log = envelope_log
        if predicted_log is None:
            continue
        residual_log = predicted_log - math.log(median_gflops)
        if residual_log <= threshold_log:
            continue
        outliers.append(
            ShapeOutlier(
                shape=shape,
                candidate_hash=summary.candidate_hash,
                samples=summary.samples,
                median_gflops=median_gflops,
                predicted_neighbor_gflops=math.exp(predicted_log),
                residual_pct=(math.exp(residual_log) - 1.0) * 100.0,
                neighbor_shape_ids=tuple(other_shape.id for _, other_shape, _ in nearest),
                neighbor_candidate_hashes=tuple(other_summary.candidate_hash for _, _, other_summary in nearest),
            )
        )
    outliers.sort(key=lambda item: (-item.residual_pct, item.shape.id))
    return outliers[:max_shapes] if max_shapes is not None else outliers


def repair_seed_candidates(
    db: EvoTensileDB,
    *,
    outliers: list[ShapeOutlier],
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    min_samples: int = 1,
    neighbor_per_shape: int = 4,
) -> list[Candidate]:
    """Seed repair searches from each outlier's winner and nearest-neighbor top candidates."""
    if not outliers:
        return []
    hashes: list[str] = []
    seen: set[str] = set()
    for outlier in outliers:
        for candidate_hash in (outlier.candidate_hash, *outlier.neighbor_candidate_hashes):
            if candidate_hash not in seen:
                hashes.append(candidate_hash)
                seen.add(candidate_hash)
        for shape_id in outlier.neighbor_shape_ids:
            for summary in db.rank_evaluations(
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                shape_id=shape_id,
                min_samples=min_samples,
                limit=neighbor_per_shape,
            ):
                if summary.candidate_hash in seen:
                    continue
                hashes.append(summary.candidate_hash)
                seen.add(summary.candidate_hash)
    return [
        Candidate(params=candidate.canonical_params(), source="repair-transfer", parent_hashes=(candidate.hash,))
        for candidate in db.get_candidates(hashes)
    ]
