import contextlib
import errno
import math
import os
import random
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from .adaptive_retime import AdaptivePolicy, decide_retime_by_shape, load_timing_stats
from .cache import POSITIVE_CACHE_STATUSES
from .candidate import Candidate, Shape, stable_hash
from .database import EvaluationInsert, EvaluationSummary, EvoTensileDB
from .manifest import write_manifest
from .profile import DEFAULT_PROFILE, TargetProfile
from .protocol import BenchmarkProtocol
from .runner import DEFAULT_TENSILELITE_BIN
from .search.differential_evolution import differential_evolution_candidates
from .search.gomea import gomea_candidates, gomea_neighborhood_candidates
from .search.learned_linkage import (
    DEFAULT_MAX_CLUSTERS,
    DEFAULT_MIN_LINKAGE_SAMPLES,
    DEFAULT_ORDINAL_BINS,
    DEFAULT_TRUNCATION_TAU,
    LinkageLearningSummary,
    LinkageModel,
    learn_linkage_models_from_db,
)
from .search.local_search import mutate_elites
from .search.random_search import initial_random_batch
from .search_space import cheap_constraints, explain_invalid_nt_hhs, random_candidate
from .shapes import shape_from_id
from .structured_runner import build_then_structured_benchmark
from .tensilelite_diagnostics import attribution_inserts_from_diagnostics, run_tensilelite_diagnostics
from .yaml_writer import write_tensilelite_yaml


@dataclass(frozen=True)
class PlannedBatch:
    batch_index: int
    candidates: list[Candidate]
    shapes: list[Shape]
    missing_pairs: int
    nominal_pairs: int
    samples_per_pair: int
    requires_validation: bool = True

    @property
    def extra_pairs(self) -> int:
        return self.nominal_pairs - self.missing_pairs

    @property
    def missing_samples(self) -> int:
        return self.missing_pairs * self.samples_per_pair

    @property
    def nominal_samples(self) -> int:
        return self.nominal_pairs * self.samples_per_pair


@dataclass(frozen=True)
class BatchIngestResult:
    inserted: int
    unmapped: int
    status_counts: dict[str, int]
    rejected: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class ExecutedBatch:
    planned: PlannedBatch
    yaml_path: Path
    manifest_path: Path
    output_dir: Path
    build_returncode: int | None = None
    runner_returncode: int | None = None
    ingest: BatchIngestResult | None = None
    build_output_dir: Path | None = None


@dataclass(frozen=True)
class ScheduleResult:
    planned_batches: list[PlannedBatch]
    executed_batches: list[ExecutedBatch] = field(default_factory=list)
    adaptive_rounds: int = 0

    @property
    def missing_pairs(self) -> int:
        return sum(batch.missing_pairs for batch in self.planned_batches)

    @property
    def nominal_pairs(self) -> int:
        return sum(batch.nominal_pairs for batch in self.planned_batches)


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


T = TypeVar("T")

PROPOSAL_MODES = (
    "random",
    "seed-random",
    "local",
    "seed-random-local",
    "de",
    "seed-random-de",
    "gomea",
    "seed-random-gomea",
    "evolutionary",
)
DEFAULT_PROPOSAL = DEFAULT_PROFILE.default_proposal
DEFAULT_NUM_RANDOM = DEFAULT_PROFILE.default_num_random
DEFAULT_COMPILE_THREADS = 1
_COMPILE_CACHE_LOCK_POLL_S = 0.1


def default_batch_workers() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return os.cpu_count() or 1


def production_candidate_batch_size(
    *,
    candidate_count: int,
    shape_count: int,
    shape_batch_size: int,
    batch_workers: int,
    max_candidate_batch_size: int,
) -> int:
    if candidate_count <= 0 or batch_workers <= 0:
        return 1
    shape_batches = max(1, math.ceil(max(1, shape_count) / shape_batch_size))
    max_size = max(1, min(candidate_count, max_candidate_batch_size))
    for candidate_batch_size in range(max_size, 0, -1):
        if math.ceil(candidate_count / candidate_batch_size) * shape_batches >= batch_workers:
            return candidate_batch_size
    return 1


DEFAULT_ELITE_COUNT = DEFAULT_PROFILE.default_elite_count
DEFAULT_LOCAL_COUNT = DEFAULT_PROFILE.default_local_count
DEFAULT_DE_COUNT = DEFAULT_PROFILE.default_de_count
DEFAULT_GOMEA_COUNT = DEFAULT_PROFILE.default_gomea_count
DEFAULT_TRANSFER_SHAPES = DEFAULT_PROFILE.default_transfer_shapes
DEFAULT_TRANSFER_PER_SHAPE = DEFAULT_PROFILE.default_transfer_per_shape
DEFAULT_MUTATION_RATE = DEFAULT_PROFILE.default_mutation_rate
DEFAULT_CROSSOVER_RATE = DEFAULT_PROFILE.default_crossover_rate
DEFAULT_RANDOM_GENE_RATE = DEFAULT_PROFILE.default_random_gene_rate
DEFAULT_LEARNED_LINKAGE_ENABLED = True
DEFAULT_LINKAGE_TRUNCATION_TAU = DEFAULT_TRUNCATION_TAU
DEFAULT_LINKAGE_MIN_SAMPLES = DEFAULT_MIN_LINKAGE_SAMPLES
DEFAULT_LINKAGE_MAX_CLUSTERS = DEFAULT_MAX_CLUSTERS
DEFAULT_LINKAGE_ORDINAL_BINS = DEFAULT_ORDINAL_BINS


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    by_hash: dict[str, Candidate] = {}
    for candidate in candidates:
        by_hash.setdefault(candidate.hash, candidate)
    return list(by_hash.values())


def _ranked_elites(
    db: EvoTensileDB,
    *,
    problem_type_hash: str | None,
    benchmark_protocol_hash: str | None,
    shape_id: str | None,
    elite_count: int,
) -> list[Candidate]:
    summaries = db.rank_evaluations(
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        shape_id=shape_id,
        min_samples=1,
        limit=elite_count,
    )
    return db.get_candidates([summary.candidate_hash for summary in summaries])


def _shape_distance(left: Shape, right: Shape) -> float:
    left_features = left.features()
    right_features = right.features()
    keys = ("log2_m", "log2_n", "log2_k", "log2_m_over_n", "log2_k_over_m", "log2_k_over_n")
    return math.sqrt(sum((left_features[key] - right_features[key]) ** 2 for key in keys))


def _learned_linkage_models_for_proposal(
    db: EvoTensileDB,
    *,
    enabled: bool,
    problem_type_hash: str | None,
    benchmark_protocol_hash: str | None,
    target_shapes: list[Shape] | None,
    min_samples: int,
    truncation_tau: float,
    max_clusters: int,
    ordinal_bins: int,
) -> tuple[list[LinkageModel], LinkageLearningSummary]:
    if not enabled:
        return [], LinkageLearningSummary(False, 0, 0, 0, "disabled")
    return learn_linkage_models_from_db(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        shapes=target_shapes,
        truncation_tau=truncation_tau,
        min_samples=min_samples,
        max_clusters=max_clusters,
        ordinal_bins=ordinal_bins,
    )


def _nearest_shape_ids(targets: list[Shape], source_shape_ids: set[str], *, limit: int) -> list[str]:
    if limit <= 0 or not targets or not source_shape_ids:
        return []
    source_shapes: list[Shape] = []
    for shape_id in source_shape_ids:
        try:
            source_shapes.append(shape_from_id(shape_id))
        except ValueError:
            continue

    best_by_shape: dict[str, float] = {}
    for source in source_shapes:
        # Use nearest target distance so exact target-shape winners and nearby-shape winners seed new work.
        best_by_shape[source.id] = min(_shape_distance(source, target) for target in targets)
    return [shape_id for shape_id, _ in sorted(best_by_shape.items(), key=lambda item: (item[1], item[0]))[:limit]]


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


def _local_feature_delta(target: Shape, other: Shape) -> list[float]:
    target_features = target.features()
    other_features = other.features()
    keys = ("log2_m", "log2_n", "log2_k", "log2_m_over_n", "log2_k_over_m", "log2_k_over_n")
    return [other_features[key] - target_features[key] for key in keys]


def _weighted_local_linear_prediction(
    target: Shape,
    nearest: list[tuple[float, Shape, EvaluationSummary]],
) -> float | None:
    if len(nearest) < 3:
        return None
    dimension = len(_local_feature_delta(target, nearest[0][1])) + 1
    matrix = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    rhs = [0.0 for _ in range(dimension)]
    total_weight = 0.0
    neighbor_logs: list[float] = []
    for distance, other_shape, summary in nearest:
        if summary.median_gflops is None or summary.median_gflops <= 0:
            continue
        y = math.log(summary.median_gflops)
        row = [1.0, *_local_feature_delta(target, other_shape)]
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
            neighbor_items.append((_shape_distance(shape, other_shape), other_shape, other_summary))
        nearest = sorted(neighbor_items, key=lambda item: (item[0], item[1].id))[:neighbor_count]
        if not nearest:
            continue
        weighted_logs: list[tuple[float, float]] = []
        for distance, _, other_summary in nearest:
            other_median_gflops = other_summary.median_gflops
            if other_median_gflops is None or other_median_gflops <= 0:
                continue
            weighted_logs.append((math.log(other_median_gflops), 1.0 / max(distance, 0.125)))
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
        actual_log = math.log(median_gflops)
        residual_log = predicted_log - actual_log
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
    seeds: list[Candidate] = []
    for candidate in db.get_candidates(hashes):
        seeds.append(
            Candidate(params=candidate.canonical_params(), source="repair-transfer", parent_hashes=(candidate.hash,))
        )
    return seeds


def _transfer_elites(
    db: EvoTensileDB,
    *,
    target_shapes: list[Shape],
    problem_type_hash: str | None,
    benchmark_protocol_hash: str | None,
    nearest_shape_count: int,
    per_shape: int,
) -> list[Candidate]:
    if nearest_shape_count <= 0 or per_shape <= 0 or not target_shapes:
        return []
    summaries = db.rank_evaluations(
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        min_samples=1,
    )
    source_shape_ids = {summary.shape_id for summary in summaries}
    nearest_shape_ids = _nearest_shape_ids(target_shapes, source_shape_ids, limit=nearest_shape_count)
    hashes: list[str] = []
    seen_hashes: set[str] = set()
    for source_shape_id in nearest_shape_ids:
        source_summaries = db.rank_evaluations(
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            shape_id=source_shape_id,
            min_samples=1,
            limit=per_shape,
        )
        for summary in source_summaries:
            if summary.candidate_hash in seen_hashes:
                continue
            hashes.append(summary.candidate_hash)
            seen_hashes.add(summary.candidate_hash)
    transfer = []
    for candidate in db.get_candidates(hashes):
        transfer.append(
            Candidate(params=candidate.canonical_params(), source="transfer", parent_hashes=(candidate.hash,))
        )
    return transfer


def _shape_aware_random_batch(num_random: int, *, seed: int, target_shapes: list[Shape] | None) -> list[Candidate]:
    if not target_shapes:
        return initial_random_batch(num_random, seed=seed)
    rng = random.Random(seed)
    out: dict[str, Candidate] = {}
    attempts = 0
    max_attempts = max(1000, num_random * 1000)
    while len(out) < num_random and attempts < max_attempts:
        attempts += 1
        candidate = random_candidate(rng, target_shapes=target_shapes)
        params = candidate.canonical_params()
        if all(cheap_constraints(params, shape=shape) for shape in target_shapes):
            out[candidate.hash] = candidate
    if len(out) < num_random:
        raise RuntimeError(f"failed to generate {num_random} shape-valid random candidates after {attempts} attempts")
    return list(out.values())


def propose_candidates(
    db: EvoTensileDB,
    *,
    proposal: str = DEFAULT_PROPOSAL,
    num_random: int = DEFAULT_NUM_RANDOM,
    seed: int = 1,
    problem_type_hash: str | None = None,
    benchmark_protocol_hash: str | None = None,
    shape_id: str | None = None,
    target_shapes: list[Shape] | None = None,
    transfer_shape_count: int = DEFAULT_TRANSFER_SHAPES,
    transfer_per_shape: int = DEFAULT_TRANSFER_PER_SHAPE,
    elite_count: int = DEFAULT_ELITE_COUNT,
    local_count: int = DEFAULT_LOCAL_COUNT,
    de_count: int = DEFAULT_DE_COUNT,
    gomea_count: int = DEFAULT_GOMEA_COUNT,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    random_gene_rate: float = DEFAULT_RANDOM_GENE_RATE,
    learned_linkage: bool = DEFAULT_LEARNED_LINKAGE_ENABLED,
    linkage_truncation_tau: float = DEFAULT_LINKAGE_TRUNCATION_TAU,
    linkage_min_samples: int = DEFAULT_LINKAGE_MIN_SAMPLES,
    linkage_max_clusters: int = DEFAULT_LINKAGE_MAX_CLUSTERS,
    linkage_ordinal_bins: int = DEFAULT_LINKAGE_ORDINAL_BINS,
) -> list[Candidate]:
    """Build candidates from random proposals and/or cached/imported elites."""
    if proposal not in PROPOSAL_MODES:
        raise ValueError(f"unknown proposal mode: {proposal}")

    candidates: list[Candidate] = []
    uses_random = proposal in {
        "random",
        "seed-random",
        "seed-random-local",
        "seed-random-de",
        "seed-random-gomea",
        "evolutionary",
    }
    needs_elites = proposal in {
        "local",
        "seed-random-local",
        "de",
        "seed-random-de",
        "gomea",
        "seed-random-gomea",
        "evolutionary",
    }
    elites = (
        _ranked_elites(
            db,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            shape_id=shape_id,
            elite_count=elite_count,
        )
        if needs_elites
        else []
    )
    transfer_elites = (
        _transfer_elites(
            db,
            target_shapes=target_shapes or [],
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            nearest_shape_count=transfer_shape_count,
            per_shape=transfer_per_shape,
        )
        if needs_elites and shape_id is None
        else []
    )
    if transfer_elites:
        # Nearby winners should be evaluated before random restarts, especially when candidate batches are truncated.
        candidates.extend(transfer_elites)
        elites = _dedupe_candidates([*elites, *transfer_elites])

    if uses_random:
        candidates.extend(_shape_aware_random_batch(num_random, seed=seed, target_shapes=target_shapes))

    if proposal in {"local", "seed-random-local", "evolutionary"} and local_count > 0:
        candidates.extend(mutate_elites(elites, count=local_count, seed=seed + 1009, mutation_rate=mutation_rate))

    if proposal in {"de", "seed-random-de", "evolutionary"} and de_count > 0:
        parents = _dedupe_candidates(elites)
        candidates.extend(
            differential_evolution_candidates(
                parents,
                count=de_count,
                seed=seed + 2003,
                crossover_rate=crossover_rate,
                random_gene_rate=random_gene_rate,
                exclude={candidate.hash for candidate in candidates},
            )
        )

    if proposal in {"gomea", "seed-random-gomea", "evolutionary"} and gomea_count > 0:
        parents = _dedupe_candidates(elites)
        linkage_models, _ = _learned_linkage_models_for_proposal(
            db,
            enabled=learned_linkage,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            target_shapes=target_shapes,
            min_samples=linkage_min_samples,
            truncation_tau=linkage_truncation_tau,
            max_clusters=linkage_max_clusters,
            ordinal_bins=linkage_ordinal_bins,
        )
        neighborhood_parents = parents
        gomea_budget = max(0, gomea_count)
        neighborhood_budget = gomea_budget // 2
        candidates.extend(
            gomea_neighborhood_candidates(
                neighborhood_parents,
                count=neighborhood_budget,
                max_elites=None,
                exclude={candidate.hash for candidate in candidates},
            )
        )
        candidates.extend(
            gomea_candidates(
                parents,
                count=gomea_budget - neighborhood_budget,
                seed=seed + 3001,
                elite_count=elite_count,
                exclude={candidate.hash for candidate in candidates},
                target_shapes=target_shapes,
                linkage_models=linkage_models,
            )
        )

    return _dedupe_candidates(candidates)


def _chunks(items: list[T], size: int) -> list[list[T]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    return [items[i : i + size] for i in range(0, len(items), size)]


def _resolve_timeout(value: float | None, default: float | None) -> float | None:
    if value is None:
        return default
    if value <= 0:
        return None
    return value


def _record_shape_rule_rejections(
    db: EvoTensileDB,
    *,
    shapes: list[Shape],
    candidates: list[Candidate],
    problem_type_hash: str,
    benchmark_protocol_hash: str,
) -> int:
    counts = db.reusable_cache_entry_counts(
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        shape_ids=[shape.id for shape in shapes],
        candidate_hashes=[candidate.hash for candidate in candidates],
    )
    evaluations: list[EvaluationInsert] = []
    for shape in shapes:
        for candidate in candidates:
            if counts.get((shape.id, candidate.hash)):
                continue
            if any(
                reason.shape_dependent for reason in explain_invalid_nt_hhs(candidate.canonical_params(), shape=shape)
            ):
                evaluations.append(
                    EvaluationInsert(
                        shape_id=shape.id,
                        candidate_hash=candidate.hash,
                        run_id=None,
                        status="rejected",
                        problem_type_hash=problem_type_hash,
                        benchmark_protocol_hash=benchmark_protocol_hash,
                    )
                )
    db.insert_evaluations(evaluations)
    return len(evaluations)


def _missing_candidate_indices_by_shape(
    db: EvoTensileDB,
    *,
    shapes: list[Shape],
    candidates: list[Candidate],
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    min_samples: int,
    ignore_cache: bool = False,
) -> dict[int, tuple[tuple[int, int, bool], ...]]:
    counts: dict[tuple[str, str], dict[str, int]] = {}
    validated: set[tuple[str, str]] = set()
    shape_ids = [shape.id for shape in shapes]
    candidate_hashes = [candidate.hash for candidate in candidates]
    if not ignore_cache:
        counts = db.reusable_cache_entry_counts(
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            shape_ids=shape_ids,
            candidate_hashes=candidate_hashes,
        )
        validated = db.validated_cache_entries(
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
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
            status_counts = {} if ignore_cache else counts.get((shape.id, candidate.hash), {})
            negative_count = sum(
                count for status, count in status_counts.items() if status not in POSITIVE_CACHE_STATUSES
            )
            if negative_count > 0:
                continue
            ok_count = sum(status_counts.get(status, 0) for status in POSITIVE_CACHE_STATUSES)
            remaining = max(0, min_samples - ok_count)
            if remaining > 0:
                missing_items.append((candidate_index, remaining, (shape.id, candidate.hash) not in validated))
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


def plan_batches(
    db: EvoTensileDB,
    *,
    shapes: list[Shape],
    candidates: list[Candidate],
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    min_samples: int = 1,
    candidate_batch_size: int = 32,
    shape_batch_size: int = 100,
    ignore_cache: bool = False,
    max_batches: int | None = None,
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
                min_samples=min_samples,
                ignore_cache=ignore_cache,
            )
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


def _batch_fingerprint(batch: PlannedBatch) -> str:
    payload = {
        "candidates": [candidate.hash for candidate in batch.candidates],
        "requires_validation": batch.requires_validation,
        "samples_per_pair": batch.samples_per_pair,
        "shapes": [shape.id for shape in batch.shapes],
    }
    return stable_hash(payload, prefix="batch_")[:18]


def _compile_cache_global_parameters(target_profile: TargetProfile, protocol: BenchmarkProtocol) -> dict[str, object]:
    protocol_keys = set(protocol.global_parameters())
    return {
        key: value
        for key, value in target_profile.global_parameters(protocol).items()
        if key not in protocol_keys
        and key
        not in {
            "ForceRedoBenchmarkProblems",
            "ForceRedoLibraryLogic",
            "ValidationMaxToPrint",
            "ValidationPrintValids",
        }
    }


def _compile_cache_key(
    candidates: list[Candidate],
    *,
    target_profile: TargetProfile,
    protocol: BenchmarkProtocol,
) -> str:
    payload = {
        "candidates": [candidate.hash for candidate in candidates],
        "global_parameters": _compile_cache_global_parameters(target_profile, protocol),
        "library_logic": target_profile.library_logic,
        "problem_type_hash": target_profile.problem_type_hash,
    }
    return stable_hash(payload, prefix="ccache_")[:22]


def _compile_cache_dir(
    compile_cache_root: str | Path | None,
    current: PlannedBatch,
    *,
    target_profile: TargetProfile,
    protocol: BenchmarkProtocol,
) -> Path | None:
    if compile_cache_root is None:
        return None
    return Path(compile_cache_root) / _compile_cache_key(
        current.candidates,
        target_profile=target_profile,
        protocol=protocol,
    )


def _compile_cache_success_marker(path: Path) -> Path:
    return path / ".evotensile_compile_cache_ok"


def _has_tensilelite_cache(path: Path) -> bool:
    return _compile_cache_success_marker(path).exists() and any(path.glob("**/caches/*/cache.yaml"))


def _mark_compile_cache_success(path: Path) -> None:
    _compile_cache_success_marker(path).write_text("ok\n", encoding="utf-8")


@contextlib.contextmanager
def _compile_cache_lock(path: Path) -> Iterator[None]:
    lock_dir = path.parent / f".{path.name}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            time.sleep(_COMPILE_CACHE_LOCK_POLL_S)
    try:
        yield
    finally:
        try:
            lock_dir.rmdir()
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                raise


def write_batch_inputs(
    batch: PlannedBatch,
    output_root: str | Path,
    *,
    target_profile: TargetProfile,
    protocol: BenchmarkProtocol,
    unique_run_dir: bool = False,
) -> tuple[Path, Path, Path]:
    batch_dir = Path(output_root) / f"batch_{batch.batch_index:04d}_{_batch_fingerprint(batch)}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = batch_dir / "config.yaml"
    manifest_path = batch_dir / "config.manifest.csv"
    run_dir = batch_dir / (f"run_{uuid.uuid4().hex[:8]}" if unique_run_dir else "run")
    write_tensilelite_yaml(
        yaml_path,
        batch.candidates,
        batch.shapes,
        global_parameters=target_profile.global_parameters(protocol),
        library_logic=target_profile.library_logic,
        problem_type=target_profile.problem_type,
    )
    write_manifest(manifest_path, batch.candidates, batch.shapes)
    return yaml_path, manifest_path, run_dir


def _record_batch_status(
    db: EvoTensileDB,
    batch: PlannedBatch,
    *,
    status: str,
    run_id: str | None,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
) -> int:
    evaluations = [
        EvaluationInsert(
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id=run_id,
            status=status,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
        )
        for shape in batch.shapes
        for candidate in batch.candidates
    ]
    db.insert_evaluations(evaluations)
    return len(evaluations)


def _ingest_result_from_inserts(
    inserts: list[EvaluationInsert], *, errors: list[str] | None = None
) -> BatchIngestResult:
    status_counts: dict[str, int] = {}
    rejected = 0
    unmapped = 0
    inserted = 0
    for item in inserts:
        status_counts[item.status] = status_counts.get(item.status, 0) + 1
        if item.status == "rejected":
            rejected += 1
        elif item.status == "unmapped":
            unmapped += 1
        else:
            inserted += 1
    return BatchIngestResult(
        inserted=inserted,
        unmapped=unmapped,
        status_counts=status_counts,
        rejected=rejected,
        errors=errors or [],
    )


def _adaptive_topup_groups(
    db: EvoTensileDB,
    *,
    shapes: list[Shape],
    candidates: list[Candidate],
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    policy: AdaptivePolicy,
    min_samples: int,
) -> list[tuple[int, list[Shape], list[Candidate]]]:
    shape_by_id = {shape.id: shape for shape in shapes}
    candidate_by_hash = {candidate.hash: candidate for candidate in candidates}
    stats_by_shape = load_timing_stats(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hashes=[benchmark_protocol_hash],
        min_samples=min_samples,
        shape_ids=set(shape_by_id),
        candidate_hashes=set(candidate_by_hash),
    )
    decisions = decide_retime_by_shape(stats_by_shape, policy=policy)
    grouped: dict[tuple[int, tuple[str, ...]], list[Shape]] = {}
    rank_order_by_key: dict[tuple[int, tuple[str, ...]], dict[str, int]] = {}
    for decision in decisions.values():
        if not decision.needs_retime or decision.target_samples <= 0:
            continue
        available_hashes = tuple(
            candidate_hash for candidate_hash in decision.retime_candidate_hashes if candidate_hash in candidate_by_hash
        )
        if len(available_hashes) < 2:
            continue
        key = (decision.target_samples, tuple(sorted(available_hashes)))
        shape = shape_by_id.get(decision.shape_id)
        if shape is None:
            continue
        grouped.setdefault(key, []).append(shape)
        ranks = rank_order_by_key.setdefault(key, {})
        for rank, candidate_hash in enumerate(available_hashes):
            ranks.setdefault(candidate_hash, rank)

    groups: list[tuple[int, list[Shape], list[Candidate]]] = []
    for (target_samples, candidate_hashes), group_shapes in sorted(
        grouped.items(), key=lambda item: (item[0][0], len(item[1]), item[0][1]), reverse=True
    ):
        ranks = rank_order_by_key[(target_samples, candidate_hashes)]
        ordered_hashes = sorted(candidate_hashes, key=lambda candidate_hash: (ranks[candidate_hash], candidate_hash))
        groups.append(
            (
                target_samples,
                sorted(group_shapes, key=lambda shape: shape.id),
                [candidate_by_hash[h] for h in ordered_hashes],
            )
        )
    return groups


def _execute_current_batch(
    db: EvoTensileDB,
    current: PlannedBatch,
    *,
    output_root: str | Path,
    target_profile: TargetProfile,
    protocol: BenchmarkProtocol,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    tensilelite_bin: str | Path,
    compile_threads: int | None,
    runner_bin: str | Path | None,
    build_timeout_s: float | None,
    runner_timeout_s: float | None,
    generate_only: bool = False,
    compile_cache_root: str | Path | None = None,
) -> tuple[ExecutedBatch, bool]:
    current_protocol = protocol.with_overrides(
        num_benchmarks=current.samples_per_pair,
        num_elements_to_validate=protocol.num_elements_to_validate if current.requires_validation else 0,
    )
    yaml_path, manifest_path, run_dir = write_batch_inputs(
        current,
        output_root,
        target_profile=target_profile,
        protocol=current_protocol,
        unique_run_dir=not generate_only,
    )
    if generate_only:
        return ExecutedBatch(current, yaml_path, manifest_path, run_dir), False

    compile_cache_dir = _compile_cache_dir(
        compile_cache_root,
        current,
        target_profile=target_profile,
        protocol=current_protocol,
    )
    build_dir = compile_cache_dir or run_dir
    if compile_cache_dir is not None:
        with _compile_cache_lock(compile_cache_dir):
            build_result, structured_result, inserts, structured_errors = build_then_structured_benchmark(
                yaml_path,
                manifest_path,
                run_dir,
                shapes=current.shapes,
                candidates=current.candidates,
                db=db,
                tensilelite_bin=tensilelite_bin,
                compile_threads=compile_threads,
                target_profile=target_profile,
                protocol=current_protocol,
                runner_bin=runner_bin,
                trust_prior_validation=not current.requires_validation,
                build_timeout_s=build_timeout_s,
                runner_timeout_s=runner_timeout_s,
                build_dir=compile_cache_dir,
                use_build_cache=_has_tensilelite_cache(compile_cache_dir),
            )
            if build_result.ok:
                _mark_compile_cache_success(compile_cache_dir)
    else:
        build_result, structured_result, inserts, structured_errors = build_then_structured_benchmark(
            yaml_path,
            manifest_path,
            run_dir,
            shapes=current.shapes,
            candidates=current.candidates,
            db=db,
            tensilelite_bin=tensilelite_bin,
            compile_threads=compile_threads,
            target_profile=target_profile,
            protocol=current_protocol,
            runner_bin=runner_bin,
            trust_prior_validation=not current.requires_validation,
            build_timeout_s=build_timeout_s,
            runner_timeout_s=runner_timeout_s,
        )
    if build_result.timed_out and len(current.candidates) == 1:
        recorded = _record_batch_status(
            db,
            current,
            status="build_timeout",
            run_id=build_result.run_id,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
        )
        ingest = BatchIngestResult(inserted=0, unmapped=0, status_counts={"build_timeout": recorded})
    elif not build_result.ok and len(current.candidates) == 1:
        recorded = _record_batch_status(
            db,
            current,
            status="build_failed",
            run_id=build_result.run_id,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
        )
        ingest = BatchIngestResult(inserted=0, unmapped=0, status_counts={"build_failed": recorded})
    elif structured_errors:
        status_counts = {}
        if structured_result is not None and structured_result.timed_out:
            recorded = _record_batch_status(
                db,
                current,
                status="runner_timeout",
                run_id=None,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
            )
            status_counts["runner_timeout"] = recorded
        ingest = BatchIngestResult(inserted=0, unmapped=0, status_counts=status_counts, errors=structured_errors)
    else:
        db.insert_evaluations(inserts)
        ingest = _ingest_result_from_inserts(inserts)
    runner_returncode = structured_result.returncode if structured_result is not None else None
    failed = not build_result.ok or (structured_result is not None and not structured_result.ok) or not ingest.ok
    accepted_candidate_hashes = {item.candidate_hash for item in inserts if item.status == "ok"}
    failed_candidate_hashes = {
        candidate.hash for candidate in current.candidates if candidate.hash not in accepted_candidate_hashes
    }
    if (
        not build_result.ok
        and not structured_errors
        and (structured_result is None or structured_result.ok)
        and len(current.candidates) > 1
        and failed_candidate_hashes
    ):
        diagnostics = run_tensilelite_diagnostics(
            yaml_path,
            manifest_path,
            build_dir,
            tensilelite_bin=tensilelite_bin,
            db=db,
            target_profile=target_profile,
            protocol=current_protocol,
            timeout_s=build_timeout_s,
        )
        diagnostic_inserts = attribution_inserts_from_diagnostics(
            diagnostics.records,
            planned_shape_ids=[shape.id for shape in current.shapes],
            failed_candidate_hashes=failed_candidate_hashes,
            run_id=diagnostics.run_id,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            unattributed_status="build_timeout_unattributed" if build_result.timed_out else "build_failed_unattributed",
        )
        db.insert_evaluations(diagnostic_inserts)
        ingest = _ingest_result_from_inserts([*inserts, *diagnostic_inserts])
        failed = (
            failed or not diagnostics.ok or any(item.status.endswith("_unattributed") for item in diagnostic_inserts)
        )
    executed = ExecutedBatch(
        planned=current,
        yaml_path=yaml_path,
        manifest_path=manifest_path,
        output_dir=run_dir,
        build_returncode=build_result.returncode,
        runner_returncode=runner_returncode,
        ingest=ingest,
        build_output_dir=build_dir,
    )
    return executed, failed


def execute_schedule(
    db: EvoTensileDB,
    *,
    shapes: list[Shape],
    candidates: list[Candidate],
    output_root: str | Path,
    target_profile: TargetProfile = DEFAULT_PROFILE,
    protocol: BenchmarkProtocol | None = None,
    min_samples: int = 1,
    candidate_batch_size: int = 32,
    shape_batch_size: int = 100,
    ignore_cache: bool = False,
    max_batches: int | None = None,
    dry_run: bool = False,
    generate_only: bool = False,
    tensilelite_bin: str | Path = DEFAULT_TENSILELITE_BIN,
    compile_threads: int | None = DEFAULT_COMPILE_THREADS,
    keep_going: bool = False,
    runner_bin: str | Path | None = None,
    build_timeout_s: float | None = None,
    runner_timeout_s: float | None = None,
    adaptive_policy: AdaptivePolicy | None = None,
    adaptive_initial_samples: int = 3,
    adaptive_max_rounds: int = 4,
    batch_workers: int | None = None,
    compile_cache_root: str | Path | None = None,
) -> ScheduleResult:
    if not dry_run and not generate_only and runner_bin is None:
        raise ValueError("--runner-bin is required")

    protocol = protocol or target_profile.default_protocol
    resolved_batch_workers = default_batch_workers() if batch_workers is None else batch_workers
    problem_type_hash = target_profile.problem_type_hash
    benchmark_protocol_hash = target_profile.benchmark_protocol_hash(protocol)
    build_timeout_s = _resolve_timeout(build_timeout_s, target_profile.default_build_timeout_s)
    runner_timeout_s = _resolve_timeout(runner_timeout_s, target_profile.default_runner_timeout_s)

    if adaptive_policy is not None:
        if adaptive_initial_samples <= 0:
            raise ValueError("adaptive_initial_samples must be positive")
        if adaptive_max_rounds < 0:
            raise ValueError("adaptive_max_rounds must be non-negative")
        initial_protocol = protocol.with_overrides(num_benchmarks=adaptive_initial_samples)
        initial = execute_schedule(
            db,
            shapes=shapes,
            candidates=candidates,
            output_root=output_root,
            target_profile=target_profile,
            protocol=initial_protocol,
            min_samples=adaptive_initial_samples,
            candidate_batch_size=candidate_batch_size,
            shape_batch_size=shape_batch_size,
            ignore_cache=ignore_cache,
            max_batches=max_batches,
            dry_run=dry_run,
            generate_only=generate_only,
            tensilelite_bin=tensilelite_bin,
            compile_threads=compile_threads,
            keep_going=keep_going,
            runner_bin=runner_bin,
            build_timeout_s=build_timeout_s,
            runner_timeout_s=runner_timeout_s,
            batch_workers=resolved_batch_workers,
            compile_cache_root=compile_cache_root,
        )
        planned = list(initial.planned_batches)
        executed = list(initial.executed_batches)
        if dry_run or generate_only:
            return ScheduleResult(planned_batches=planned, executed_batches=executed, adaptive_rounds=0)
        completed_rounds = 0
        for _ in range(adaptive_max_rounds):
            groups = _adaptive_topup_groups(
                db,
                shapes=shapes,
                candidates=candidates,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                policy=adaptive_policy,
                min_samples=adaptive_initial_samples,
            )
            if not groups:
                break
            ran_round = False
            for target_samples, group_shapes, group_candidates in groups:
                topup = execute_schedule(
                    db,
                    shapes=group_shapes,
                    candidates=group_candidates,
                    output_root=output_root,
                    target_profile=target_profile,
                    protocol=protocol.with_overrides(num_benchmarks=target_samples),
                    min_samples=target_samples,
                    candidate_batch_size=candidate_batch_size,
                    shape_batch_size=shape_batch_size,
                    ignore_cache=False,
                    max_batches=None,
                    dry_run=False,
                    generate_only=False,
                    tensilelite_bin=tensilelite_bin,
                    compile_threads=compile_threads,
                    keep_going=keep_going,
                    runner_bin=runner_bin,
                    build_timeout_s=build_timeout_s,
                    runner_timeout_s=runner_timeout_s,
                    batch_workers=resolved_batch_workers,
                    compile_cache_root=compile_cache_root,
                )
                planned.extend(topup.planned_batches)
                executed.extend(topup.executed_batches)
                if topup.executed_batches:
                    ran_round = True
                if topup.executed_batches and not keep_going:
                    last_ingest = topup.executed_batches[-1].ingest
                    if last_ingest is not None and not last_ingest.ok:
                        return ScheduleResult(
                            planned_batches=planned, executed_batches=executed, adaptive_rounds=completed_rounds
                        )
            if not ran_round:
                break
            completed_rounds += 1
        return ScheduleResult(planned_batches=planned, executed_batches=executed, adaptive_rounds=completed_rounds)

    if resolved_batch_workers <= 0:
        raise ValueError("batch_workers must be positive")

    target_samples = max(min_samples, protocol.num_benchmarks)

    db.init()
    db.register_candidates(candidates)
    db.register_shapes(shapes)
    if not dry_run and not generate_only:
        _record_shape_rule_rejections(
            db,
            shapes=shapes,
            candidates=candidates,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
        )
    planned = plan_batches(
        db,
        shapes=shapes,
        candidates=candidates,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        min_samples=target_samples,
        candidate_batch_size=candidate_batch_size,
        shape_batch_size=shape_batch_size,
        ignore_cache=ignore_cache,
        max_batches=max_batches,
    )
    if dry_run:
        return ScheduleResult(planned_batches=planned)

    executed: list[ExecutedBatch] = []
    planned_batches = list(planned)
    batch_cursor = 0
    pending_current_batches: list[PlannedBatch] = []
    stop_requested = False

    def current_batches_for_planned(batch: PlannedBatch) -> list[PlannedBatch]:
        # Recheck just before execution so a resumed run skips observations ingested by earlier batches.
        missing_by_shape = _missing_candidate_indices_by_shape(
            db,
            shapes=batch.shapes,
            candidates=batch.candidates,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            min_samples=target_samples,
            ignore_cache=ignore_cache,
        )
        if not missing_by_shape:
            return []
        return _pair_exact_batches(
            batch_index_start=batch.batch_index,
            shapes=batch.shapes,
            candidates=batch.candidates,
            missing_by_shape=missing_by_shape,
        )

    def execute_current(current: PlannedBatch) -> tuple[ExecutedBatch, bool]:
        return _execute_current_batch(
            db,
            current,
            output_root=output_root,
            target_profile=target_profile,
            protocol=protocol,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            tensilelite_bin=tensilelite_bin,
            compile_threads=compile_threads,
            runner_bin=runner_bin,
            build_timeout_s=build_timeout_s,
            runner_timeout_s=runner_timeout_s,
            generate_only=generate_only,
            compile_cache_root=compile_cache_root,
        )

    def handle_result(
        current: PlannedBatch,
        result: tuple[ExecutedBatch, bool],
    ) -> None:
        nonlocal stop_requested
        executed_batch, failed = result
        executed.append(executed_batch)
        if failed and not keep_going:
            stop_requested = True

    effective_workers = resolved_batch_workers if keep_going and not generate_only else 1
    if effective_workers == 1:
        while batch_cursor < len(planned_batches) and not stop_requested:
            batch = planned_batches[batch_cursor]
            batch_cursor += 1
            for current in current_batches_for_planned(batch):
                handle_result(current, execute_current(current))
                if stop_requested:
                    break
        return ScheduleResult(planned_batches=planned_batches, executed_batches=executed)

    futures: dict[Future[tuple[ExecutedBatch, bool]], PlannedBatch] = {}
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        while (batch_cursor < len(planned_batches) or pending_current_batches or futures) and not stop_requested:
            while len(futures) < effective_workers and not stop_requested:
                if pending_current_batches:
                    current = pending_current_batches.pop(0)
                elif batch_cursor < len(planned_batches):
                    batch = planned_batches[batch_cursor]
                    batch_cursor += 1
                    pending_current_batches.extend(current_batches_for_planned(batch))
                    continue
                else:
                    break
                futures[executor.submit(execute_current, current)] = current
            if not futures:
                continue
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                current = futures.pop(future)
                handle_result(current, future.result())
                if stop_requested:
                    break
    return ScheduleResult(planned_batches=planned_batches, executed_batches=executed)
