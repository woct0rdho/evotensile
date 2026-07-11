import contextlib
import errno
import json
import math
import os
import random
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from .adaptive_retime import AdaptivePolicy, ProbePolicy, decide_retime_by_shape, decide_shape_probe, load_timing_stats
from .artifacts import register_candidate_artifacts
from .cache import POSITIVE_CACHE_STATUSES
from .candidate import Candidate, Shape, stable_hash
from .database import EvaluationInsert, EvaluationSummary, EvoTensileDB
from .manifest import write_manifest
from .profile import DEFAULT_PROFILE, TargetProfile
from .protocol import BenchmarkProtocol
from .runner import DEFAULT_TENSILELITE_BIN, RunResult, run_tensilelite
from .search.cost_model import predicted_batch_prepare_weight
from .search.differential_evolution import differential_evolution_candidates
from .search.family import (
    DEFAULT_FAMILY_ELITES_PER_CELL,
    family_stratified_random_candidates,
    load_family_archive,
)
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
from .search.local_search import mutate_elites, semantic_mutation_candidates
from .search.operator_credit import (
    allocate_operator_budget,
    credit_ucb_scores,
    load_donor_mode_credits,
    load_operator_credits,
    load_semantic_group_credits,
)
from .search.random_search import initial_random_batch
from .search.surrogate import DEFAULT_SURROGATE_MIN_EVIDENCE, select_surrogate_pool
from .search_space import cheap_constraints, explain_invalid_nt_hhs, random_candidate
from .shapes import shape_from_id
from .solution_mapping import find_solution_yamls
from .structured_runner import (
    RunnablePair,
    StructuredRunOutput,
    build_runnable_pairs,
    library_dir_from_build,
    run_structured_phase,
    validate_benchmark_samples,
    validate_validation_samples,
)
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
class PreparedBatch:
    planned: PlannedBatch
    yaml_path: Path
    manifest_path: Path
    output_dir: Path
    build_output_dir: Path
    build_result: RunResult
    library_dir: Path | None
    validated_pairs: list[RunnablePair]
    preparation_inserts: list[EvaluationInsert]
    validation_result: StructuredRunOutput | None = None
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutedBatch:
    planned: PlannedBatch
    yaml_path: Path
    manifest_path: Path
    output_dir: Path
    build_returncode: int | None = None
    validation_returncode: int | None = None
    runner_returncode: int | None = None
    ingest: BatchIngestResult | None = None
    build_output_dir: Path | None = None
    phase: str = "initial"


@dataclass(frozen=True)
class ScheduleResult:
    planned_batches: list[PlannedBatch]
    executed_batches: list[ExecutedBatch] = field(default_factory=list)
    adaptive_rounds: int = 0
    probe_protocol_hash: str | None = None
    probe_survivor_pairs: int = 0
    probe_screened_pairs: int = 0

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
    "family-qd",
)
DEFAULT_PROPOSAL = DEFAULT_PROFILE.default_proposal
DEFAULT_NUM_RANDOM = DEFAULT_PROFILE.default_num_random
DEFAULT_COMPILE_THREADS = 1
_COMPILE_CACHE_LOCK_POLL_S = 0.1


def default_prepare_workers() -> int:
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


def _family_archive_leaders(
    db: EvoTensileDB,
    *,
    problem_type_hash: str | None,
    benchmark_protocol_hash: str | None,
    shape_id: str | None,
    target_shapes: list[Shape] | None,
    elite_count: int,
) -> list[Candidate]:
    if elite_count <= 0:
        return []
    archive_shapes = target_shapes
    if shape_id is not None and target_shapes:
        archive_shapes = [shape for shape in target_shapes if shape.id == shape_id]
    entries = load_family_archive(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        shapes=archive_shapes,
        min_samples=1,
        limit=None,
        elites_per_family=min(DEFAULT_FAMILY_ELITES_PER_CELL, elite_count),
    )
    return _dedupe_candidates([entry.leader for entry in entries])


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
    adaptive_operators: bool = False,
    surrogate_pool_multiplier: int = 1,
    surrogate_min_evidence: int = DEFAULT_SURROGATE_MIN_EVIDENCE,
    covering_cold_start: bool = False,
    adaptive_group_credit: bool = False,
    micro_exhaustive_neighborhoods: bool = False,
    adaptive_donor_selection: bool = False,
    cost_aware_operator_credit: bool = False,
    parent_candidates: Sequence[Candidate] | None = None,
    cold_start_precovered_tokens: set[str] | None = None,
) -> list[Candidate]:
    """Build candidates from random proposals and/or cached/imported elites."""
    if proposal not in PROPOSAL_MODES:
        raise ValueError(f"unknown proposal mode: {proposal}")
    pool_multiplier = max(1, surrogate_pool_multiplier)
    pool_num_random = num_random * pool_multiplier
    pool_local_count = local_count * pool_multiplier
    pool_de_count = de_count * pool_multiplier
    pool_gomea_count = gomea_count * pool_multiplier

    candidates: list[Candidate] = []
    uses_random = proposal in {
        "random",
        "seed-random",
        "seed-random-local",
        "seed-random-de",
        "seed-random-gomea",
        "evolutionary",
        "family-qd",
    }
    needs_elites = proposal in {
        "local",
        "seed-random-local",
        "de",
        "seed-random-de",
        "gomea",
        "seed-random-gomea",
        "evolutionary",
        "family-qd",
    }
    supplied_parents = _dedupe_candidates(list(parent_candidates or ()))
    elites = (
        supplied_parents
        if needs_elites and parent_candidates is not None
        else _ranked_elites(
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
        if needs_elites and shape_id is None and parent_candidates is None
        else []
    )
    if transfer_elites:
        # Nearby winners should be evaluated before random restarts, especially when candidate batches are truncated.
        candidates.extend(transfer_elites)
        elites = _dedupe_candidates([*elites, *transfer_elites])

    family_leaders: list[Candidate] = []
    if proposal == "family-qd" and parent_candidates is None:
        family_leaders = _family_archive_leaders(
            db,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            shape_id=shape_id,
            target_shapes=target_shapes,
            elite_count=elite_count,
        )
        candidates.extend(family_leaders)
        elites = _dedupe_candidates([*family_leaders, *elites])
    elif supplied_parents:
        candidates.extend(supplied_parents)

    operator_allocation: dict[str, int] | None = None
    semantic_group_weights: dict[str, float] | None = None
    donor_mode_weights: dict[str, float] | None = None
    if adaptive_operators and proposal == "family-qd":
        operator_credits = load_operator_credits(
            db,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            shapes=target_shapes,
        )
        operator_allocation = allocate_operator_budget(
            pool_local_count + pool_de_count + pool_gomea_count,
            operator_credits,
            cost_aware=cost_aware_operator_credit,
        )
        if adaptive_group_credit:
            semantic_group_weights = credit_ucb_scores(
                load_semantic_group_credits(
                    db,
                    problem_type_hash=problem_type_hash,
                    benchmark_protocol_hash=benchmark_protocol_hash,
                    shapes=target_shapes,
                ),
                cost_aware=cost_aware_operator_credit,
            )
        if adaptive_donor_selection:
            donor_mode_weights = credit_ucb_scores(
                load_donor_mode_credits(
                    db,
                    problem_type_hash=problem_type_hash,
                    benchmark_protocol_hash=benchmark_protocol_hash,
                    shapes=target_shapes,
                ),
                cost_aware=cost_aware_operator_credit,
            )

    if uses_random:
        random_batch = (
            family_stratified_random_candidates(
                db,
                pool_num_random,
                seed=seed,
                target_shapes=target_shapes,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
            )
            if proposal == "family-qd"
            else _shape_aware_random_batch(pool_num_random, seed=seed, target_shapes=target_shapes)
        )
        candidates.extend(random_batch)

    mutation_budget = operator_allocation["semantic-mutation"] if operator_allocation is not None else pool_local_count
    if proposal in {"local", "seed-random-local", "evolutionary", "family-qd"} and mutation_budget > 0:
        if operator_allocation is not None:
            candidates.extend(
                semantic_mutation_candidates(
                    elites,
                    count=mutation_budget,
                    seed=seed + 1009,
                    target_shapes=target_shapes,
                    exclude={candidate.hash for candidate in candidates},
                    group_weights=semantic_group_weights,
                )
            )
        else:
            candidates.extend(
                mutate_elites(elites, count=mutation_budget, seed=seed + 1009, mutation_rate=mutation_rate)
            )

    de_budget = operator_allocation["de"] if operator_allocation is not None else pool_de_count
    if proposal in {"de", "seed-random-de", "evolutionary", "family-qd"} and de_budget > 0:
        parents = _dedupe_candidates(elites)
        candidates.extend(
            differential_evolution_candidates(
                parents,
                count=de_budget,
                seed=seed + 2003,
                crossover_rate=crossover_rate,
                random_gene_rate=random_gene_rate,
                exclude={candidate.hash for candidate in candidates},
            )
        )

    adaptive_gomea_budget = (
        0
        if operator_allocation is None
        else operator_allocation["gomea-neighborhood"] + operator_allocation["gomea-mixing"]
    )
    if proposal in {"gomea", "seed-random-gomea", "evolutionary", "family-qd"} and (
        gomea_count > 0 or adaptive_gomea_budget > 0
    ):
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
        if operator_allocation is None:
            gomea_budget = max(0, pool_gomea_count)
            neighborhood_budget = gomea_budget // 2
            mixing_budget = gomea_budget - neighborhood_budget
        else:
            neighborhood_budget = operator_allocation["gomea-neighborhood"]
            mixing_budget = operator_allocation["gomea-mixing"]
        candidates.extend(
            gomea_neighborhood_candidates(
                neighborhood_parents,
                count=neighborhood_budget,
                max_elites=None,
                exclude={candidate.hash for candidate in candidates},
                seed=seed + 2903,
                source="gomea-neighborhood" if operator_allocation is not None else "gomea",
                target_shapes=target_shapes,
                group_weights=semantic_group_weights,
                micro_exhaustive=micro_exhaustive_neighborhoods,
            )
        )
        candidates.extend(
            gomea_candidates(
                parents,
                count=mixing_budget,
                seed=seed + 3001,
                elite_count=elite_count,
                exclude={candidate.hash for candidate in candidates},
                target_shapes=target_shapes,
                linkage_models=linkage_models,
                family_local_probability=0.8 if operator_allocation is not None else 0.0,
                source="gomea-mixing" if operator_allocation is not None else "gomea",
                donor_mode_weights=donor_mode_weights,
                adaptive_donor_selection=adaptive_donor_selection,
            )
        )

    deduped = _dedupe_candidates(candidates)
    if pool_multiplier <= 1:
        return deduped
    preserved_hashes = (
        {candidate.hash for candidate in [*transfer_elites, *family_leaders, *supplied_parents]}
        if proposal == "family-qd"
        else {candidate.hash for candidate in [*transfer_elites, *supplied_parents]}
    )
    preserved = [candidate for candidate in deduped if candidate.hash in preserved_hashes]
    generated = [candidate for candidate in deduped if candidate.hash not in preserved_hashes]
    variation_budget = local_count + de_count + gomea_count if elites else 0
    selection_count = num_random + variation_budget
    selected = select_surrogate_pool(
        generated,
        db=db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        shapes=target_shapes or [],
        count=selection_count,
        seed=seed + 4001,
        min_evidence=surrogate_min_evidence,
        covering_cold_start=covering_cold_start,
        cold_start_precovered_tokens=cold_start_precovered_tokens,
    )
    return _dedupe_candidates([*preserved, *selected])


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
    validation_protocol_hash: str,
    min_samples: int,
    ignore_cache: bool = False,
) -> dict[int, tuple[tuple[int, int, bool], ...]]:
    counts: dict[tuple[str, str], dict[str, int]] = {}
    validation_states: dict[tuple[str, str], str] = {}
    shape_ids = [shape.id for shape in shapes]
    candidate_hashes = [candidate.hash for candidate in candidates]
    if not ignore_cache:
        counts = db.reusable_cache_entry_counts(
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            shape_ids=shape_ids,
            candidate_hashes=candidate_hashes,
        )
        validation_states = db.validation_cache_states(
            problem_type_hash=problem_type_hash,
            validation_protocol_hash=validation_protocol_hash,
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
            key = (shape.id, candidate.hash)
            status_counts = {} if ignore_cache else counts.get(key, {})
            negative_count = sum(
                count for status, count in status_counts.items() if status not in POSITIVE_CACHE_STATUSES
            )
            if negative_count > 0 or validation_states.get(key) == "failed":
                continue
            ok_count = sum(status_counts.get(status, 0) for status in POSITIVE_CACHE_STATUSES)
            remaining = max(0, min_samples - ok_count)
            if remaining > 0:
                missing_items.append((candidate_index, remaining, validation_states.get(key) != "passed"))
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
    validation_protocol_hash: str,
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
                validation_protocol_hash=validation_protocol_hash,
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


def _probe_survivor_keys(
    db: EvoTensileDB,
    *,
    shapes: list[Shape],
    candidates: list[Candidate],
    available_pairs: set[tuple[str, str]],
    problem_type_hash: str,
    probe_protocol_hash: str,
    benchmark_protocol_hash: str,
    policy: ProbePolicy,
    min_samples: int,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    shape_ids = {shape.id for shape in shapes}
    candidate_hashes = {candidate.hash for candidate in candidates}
    probe_stats = load_timing_stats(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hashes=[probe_protocol_hash],
        min_samples=min_samples,
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
    survivors: set[tuple[str, str]] = set()
    for shape_id, stats in probe_stats.items():
        eligible = [stats_item for stats_item in stats if (shape_id, stats_item.candidate_hash) in available_pairs]
        if not eligible:
            continue
        decision = decide_shape_probe(
            shape_id,
            eligible,
            policy=policy,
            reference_stats=reference_stats.get(shape_id, ()),
        )
        survivors.update((shape_id, candidate_hash) for candidate_hash in decision.survivor_hashes)
    return survivors, available_pairs - survivors


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


def _record_structured_run(
    db: EvoTensileDB,
    output: StructuredRunOutput,
    *,
    yaml_path: Path,
    output_dir: Path,
    pair_count: int,
) -> None:
    db.insert_run(
        output.run_id,
        yaml_path=str(yaml_path),
        output_dir=str(output_dir),
        status="timeout" if output.timed_out else "ok" if output.ok else "failed",
        returncode=output.returncode,
        metadata_json=json.dumps(
            {
                "command": output.command,
                "duration_s": output.duration_s,
                "mode": output.mode,
                "pair_count": pair_count,
                "results_path": str(output.results_path),
                "stderr_path": str(output.stderr_path),
                "stdout_path": str(output.stdout_path),
                "timed_out": output.timed_out,
            },
            sort_keys=True,
        ),
    )


def _prepare_current_batch(
    db: EvoTensileDB,
    current: PlannedBatch,
    *,
    output_root: str | Path,
    target_profile: TargetProfile,
    protocol: BenchmarkProtocol,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    validation_protocol_hash: str,
    tensilelite_bin: str | Path,
    compile_threads: int | None,
    runner_bin: str | Path,
    build_timeout_s: float | None,
    runner_timeout_s: float | None,
    compile_cache_root: str | Path | None,
    validation_gate: threading.Semaphore | None,
) -> PreparedBatch:
    build_protocol = protocol.with_overrides(num_benchmarks=current.samples_per_pair)
    yaml_path, manifest_path, run_dir = write_batch_inputs(
        current,
        output_root,
        target_profile=target_profile,
        protocol=build_protocol,
        unique_run_dir=True,
    )
    compile_cache_dir = _compile_cache_dir(
        compile_cache_root,
        current,
        target_profile=target_profile,
        protocol=build_protocol,
    )
    build_dir = compile_cache_dir or run_dir

    def build() -> RunResult:
        return run_tensilelite(
            yaml_path,
            build_dir,
            tensilelite_bin=tensilelite_bin,
            db=db,
            build_only=True,
            cpu_threads=compile_threads,
            global_parameters=target_profile.global_parameter_items(build_protocol),
            timeout_s=build_timeout_s,
            use_cache=compile_cache_dir is not None and _has_tensilelite_cache(compile_cache_dir),
        )

    if compile_cache_dir is None:
        build_result = build()
    else:
        with _compile_cache_lock(compile_cache_dir):
            build_result = build()
            if build_result.ok:
                _mark_compile_cache_success(compile_cache_dir)

    preparation_inserts: list[EvaluationInsert] = []
    errors: list[str] = []
    planned_pairs = {(shape.id, candidate.hash) for shape in current.shapes for candidate in current.candidates}
    solution_yamls = [str(path) for path in find_solution_yamls([build_dir])]
    runnable, missing = build_runnable_pairs(
        manifest_path=manifest_path,
        solution_yaml_paths=solution_yamls,
        planned_pairs=planned_pairs,
    )
    library_dir = library_dir_from_build(build_dir)

    if not build_result.ok and len(current.candidates) == 1 and not runnable:
        status = "build_timeout" if build_result.timed_out else "build_failed"
        preparation_inserts = [
            EvaluationInsert(
                shape_id=shape.id,
                candidate_hash=current.candidates[0].hash,
                run_id=build_result.run_id,
                status=status,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
            )
            for shape in current.shapes
        ]
        runnable = []
    elif build_result.ok:
        preparation_inserts.extend(
            EvaluationInsert(
                shape_id=item.shape_id,
                candidate_hash=item.candidate_hash,
                run_id=build_result.run_id,
                status=item.status,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
            )
            for item in missing
        )
    elif len(current.candidates) > 1:
        accepted_hashes = {pair.candidate_hash for pair in runnable}
        failed_hashes = {candidate.hash for candidate in current.candidates} - accepted_hashes
        if failed_hashes:
            diagnostics = run_tensilelite_diagnostics(
                yaml_path,
                manifest_path,
                build_dir,
                tensilelite_bin=tensilelite_bin,
                db=db,
                target_profile=target_profile,
                protocol=build_protocol,
                timeout_s=build_timeout_s,
            )
            diagnostic_inserts = attribution_inserts_from_diagnostics(
                diagnostics.records,
                planned_shape_ids=[shape.id for shape in current.shapes],
                failed_candidate_hashes=failed_hashes,
                run_id=diagnostics.run_id,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                unattributed_status=(
                    "build_timeout_unattributed" if build_result.timed_out else "build_failed_unattributed"
                ),
            )
            preparation_inserts.extend(diagnostic_inserts)

    validated_pairs: list[RunnablePair] = []
    validation_result: StructuredRunOutput | None = None
    if runnable and library_dir is None:
        errors.append("compiled artifact has no runnable library directory")
    elif runnable:
        assert library_dir is not None
        try:
            register_candidate_artifacts(
                db,
                problem_type_hash=problem_type_hash,
                runnable_pairs=runnable,
                build_run_id=build_result.run_id,
                build_output_dir=build_dir,
                library_dir=library_dir,
                solution_yaml_paths=solution_yamls,
                manifest_path=manifest_path,
            )
        except (OSError, ValueError) as exc:
            errors.append(f"candidate artifact registration failed: {exc}")

    if runnable and library_dir is not None and not errors:
        if current.requires_validation:
            validation_protocol = protocol.with_overrides(num_benchmarks=1)

            def run_validation() -> StructuredRunOutput:
                return run_structured_phase(
                    mode="validate",
                    run_dir=run_dir,
                    pairs=runnable,
                    shapes=current.shapes,
                    protocol=validation_protocol,
                    runner_bin=runner_bin,
                    library_dir=library_dir,
                    timeout_s=runner_timeout_s,
                )

            if validation_gate is None:
                validation_result = run_validation()
            else:
                with validation_gate:
                    validation_result = run_validation()
            _record_structured_run(
                db,
                validation_result,
                yaml_path=yaml_path,
                output_dir=run_dir,
                pair_count=len(runnable),
            )
            try:
                outcome = validate_validation_samples(
                    validation_result.samples,
                    runnable_pairs=runnable,
                    problem_type_hash=problem_type_hash,
                    validation_protocol_hash=validation_protocol_hash,
                    run_id=validation_result.run_id,
                    runner_returncode=validation_result.returncode,
                )
            except Exception as exc:
                errors.append(str(exc))
            else:
                db.insert_validations(outcome.validations)
                validated_pairs = outcome.passed_pairs
        else:
            cached = db.validated_cache_entries(
                problem_type_hash=problem_type_hash,
                validation_protocol_hash=validation_protocol_hash,
                shape_ids=[shape.id for shape in current.shapes],
                candidate_hashes=[candidate.hash for candidate in current.candidates],
            )
            validated_pairs = [pair for pair in runnable if (pair.shape_id, pair.candidate_hash) in cached]
            if len(validated_pairs) != len(runnable):
                errors.append("prepared artifact contains pairs without cached correctness verification")

    if preparation_inserts:
        db.insert_evaluations(preparation_inserts)
    return PreparedBatch(
        planned=current,
        yaml_path=yaml_path,
        manifest_path=manifest_path,
        output_dir=run_dir,
        build_output_dir=build_dir,
        build_result=build_result,
        library_dir=library_dir,
        validated_pairs=validated_pairs,
        preparation_inserts=preparation_inserts,
        validation_result=validation_result,
        errors=errors,
    )


def _benchmark_prepared_pairs(
    db: EvoTensileDB,
    prepared: PreparedBatch,
    *,
    pairs: list[RunnablePair],
    protocol: BenchmarkProtocol,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    runner_bin: str | Path,
    runner_timeout_s: float | None,
    phase: str,
    include_preparation: bool = False,
) -> ExecutedBatch:
    preparation_inserts = prepared.preparation_inserts if include_preparation else []
    preparation_ingest = _ingest_result_from_inserts(preparation_inserts, errors=prepared.errors)
    if not pairs or prepared.library_dir is None or prepared.errors:
        return ExecutedBatch(
            planned=prepared.planned,
            yaml_path=prepared.yaml_path,
            manifest_path=prepared.manifest_path,
            output_dir=prepared.output_dir,
            build_returncode=prepared.build_result.returncode,
            validation_returncode=(
                prepared.validation_result.returncode if prepared.validation_result is not None else None
            ),
            ingest=preparation_ingest,
            build_output_dir=prepared.build_output_dir,
            phase=phase,
        )

    benchmark_protocol = protocol.with_overrides(num_elements_to_validate=0)
    output = run_structured_phase(
        mode="benchmark",
        run_dir=prepared.output_dir,
        pairs=pairs,
        shapes=prepared.planned.shapes,
        protocol=benchmark_protocol,
        runner_bin=runner_bin,
        library_dir=prepared.library_dir,
        timeout_s=runner_timeout_s,
    )
    _record_structured_run(
        db,
        output,
        yaml_path=prepared.yaml_path,
        output_dir=prepared.output_dir,
        pair_count=len(pairs),
    )
    errors = list(prepared.errors)
    if output.timed_out:
        timing_inserts = [
            EvaluationInsert(
                shape_id=pair.shape_id,
                candidate_hash=pair.candidate_hash,
                run_id=output.run_id,
                status="runner_timeout",
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                solution_index=pair.library_solution_index,
            )
            for pair in pairs
        ]
        errors.append(f"benchmark phase timed out after {runner_timeout_s} seconds")
    else:
        try:
            timing_inserts = validate_benchmark_samples(
                output.samples,
                runnable_pairs=pairs,
                protocol=benchmark_protocol,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                run_id=output.run_id,
                runner_returncode=output.returncode,
            )
        except Exception as exc:
            timing_inserts = []
            errors.append(str(exc))
    if timing_inserts:
        db.insert_evaluations(timing_inserts)
    combined = [*preparation_inserts, *timing_inserts]
    return ExecutedBatch(
        planned=prepared.planned,
        yaml_path=prepared.yaml_path,
        manifest_path=prepared.manifest_path,
        output_dir=prepared.output_dir,
        build_returncode=prepared.build_result.returncode,
        validation_returncode=(
            prepared.validation_result.returncode if prepared.validation_result is not None else None
        ),
        runner_returncode=output.returncode,
        ingest=_ingest_result_from_inserts(combined, errors=errors),
        build_output_dir=prepared.build_output_dir,
        phase=phase,
    )


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
    probe_policy: ProbePolicy | None = None,
    adaptive_max_rounds: int = 4,
    prepare_workers: int | None = None,
    compile_cache_root: str | Path | None = None,
    cost_aware_scheduling: bool = False,
    validation_workers: int | None = None,
) -> ScheduleResult:
    if not dry_run and not generate_only and runner_bin is None:
        raise ValueError("--runner-bin is required")
    if prepare_workers is not None and prepare_workers <= 0:
        raise ValueError("prepare_workers must be positive")
    if validation_workers is not None and validation_workers <= 0:
        raise ValueError("validation_workers must be positive")
    if adaptive_policy is not None and probe_policy is None:
        raise ValueError("probe_policy is required when adaptive sampling is enabled")
    if adaptive_max_rounds < 0:
        raise ValueError("adaptive_max_rounds must be non-negative")

    protocol = protocol or target_profile.default_protocol
    if protocol.role != "main":
        raise ValueError("execute_schedule requires a main benchmark protocol")
    resolved_prepare_workers = default_prepare_workers() if prepare_workers is None else prepare_workers
    problem_type_hash = target_profile.problem_type_hash
    benchmark_protocol_hash = target_profile.benchmark_protocol_hash(protocol)
    validation_protocol_hash = protocol.validation_protocol_hash()
    build_timeout_s = _resolve_timeout(build_timeout_s, target_profile.default_build_timeout_s)
    runner_timeout_s = _resolve_timeout(runner_timeout_s, target_profile.default_runner_timeout_s)
    initial_samples = max(min_samples, protocol.num_benchmarks)

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
        validation_protocol_hash=validation_protocol_hash,
        min_samples=initial_samples,
        candidate_batch_size=candidate_batch_size,
        shape_batch_size=shape_batch_size,
        ignore_cache=ignore_cache,
        max_batches=max_batches,
    )
    if dry_run:
        return ScheduleResult(planned_batches=planned)
    if generate_only:
        generated = []
        for batch in planned:
            batch_protocol = protocol.with_overrides(num_benchmarks=batch.samples_per_pair)
            yaml_path, manifest_path, run_dir = write_batch_inputs(
                batch,
                output_root,
                target_profile=target_profile,
                protocol=batch_protocol,
            )
            generated.append(
                ExecutedBatch(
                    planned=batch,
                    yaml_path=yaml_path,
                    manifest_path=manifest_path,
                    output_dir=run_dir,
                    phase="generated",
                )
            )
        return ScheduleResult(planned_batches=planned, executed_batches=generated)

    assert runner_bin is not None
    validation_gate = None if validation_workers is None else threading.Semaphore(validation_workers)

    def prepare(batch: PlannedBatch) -> PreparedBatch:
        return _prepare_current_batch(
            db,
            batch,
            output_root=output_root,
            target_profile=target_profile,
            protocol=protocol,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            validation_protocol_hash=validation_protocol_hash,
            tensilelite_bin=tensilelite_bin,
            compile_threads=compile_threads,
            runner_bin=runner_bin,
            build_timeout_s=build_timeout_s,
            runner_timeout_s=runner_timeout_s,
            compile_cache_root=compile_cache_root,
            validation_gate=validation_gate,
        )

    # Phase 1: all compilation, mapping, diagnostics, and correctness verification.
    # Exiting the executor is the hard barrier before any timing begins.
    prepare_order = planned
    if cost_aware_scheduling:
        prepare_order = sorted(
            planned,
            key=lambda batch: (
                -predicted_batch_prepare_weight(batch.candidates, batch.shapes),
                batch.batch_index,
            ),
        )
    with ThreadPoolExecutor(max_workers=resolved_prepare_workers) as executor:
        prepared = list(executor.map(prepare, prepare_order))

    executed: list[ExecutedBatch] = []
    pair_owner: dict[tuple[str, str], tuple[PreparedBatch, RunnablePair]] = {}
    for item in prepared:
        for pair in item.validated_pairs:
            pair_owner[(pair.shape_id, pair.candidate_hash)] = (item, pair)

    if adaptive_policy is None:
        for item in prepared:
            benchmark = _benchmark_prepared_pairs(
                db,
                item,
                pairs=item.validated_pairs,
                protocol=protocol.with_overrides(num_benchmarks=item.planned.samples_per_pair),
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                runner_bin=runner_bin,
                runner_timeout_s=runner_timeout_s,
                phase="initial",
                include_preparation=True,
            )
            executed.append(benchmark)
            if not keep_going and benchmark.ingest is not None and not benchmark.ingest.ok:
                return ScheduleResult(planned_batches=planned, executed_batches=executed)
        return ScheduleResult(planned_batches=planned, executed_batches=executed)

    assert probe_policy is not None
    probe_protocol = protocol.with_overrides(
        role="probe",
        num_warmups=0,
        num_benchmarks=probe_policy.samples,
        enqueues_per_sync=1,
        syncs_per_benchmark=1,
        num_elements_to_validate=0,
    )
    probe_protocol_hash = target_profile.benchmark_protocol_hash(probe_protocol)

    for item in prepared:
        if not item.validated_pairs:
            continue
        counts = db.reusable_cache_entry_counts(
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=probe_protocol_hash,
            shape_ids=[shape.id for shape in item.planned.shapes],
            candidate_hashes=[candidate.hash for candidate in item.planned.candidates],
        )
        initial_pairs_by_remaining: dict[int, list[RunnablePair]] = {}
        for pair in item.validated_pairs:
            current_samples = counts.get((pair.shape_id, pair.candidate_hash), {}).get("ok", 0)
            remaining = probe_policy.initial_samples - current_samples
            if remaining > 0:
                initial_pairs_by_remaining.setdefault(remaining, []).append(pair)
        for remaining, pairs in sorted(initial_pairs_by_remaining.items()):
            probe = _benchmark_prepared_pairs(
                db,
                item,
                pairs=pairs,
                protocol=probe_protocol.with_overrides(num_benchmarks=remaining),
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=probe_protocol_hash,
                runner_bin=runner_bin,
                runner_timeout_s=runner_timeout_s,
                phase="probe-initial",
            )
            executed.append(probe)
            if not keep_going and probe.ingest is not None and not probe.ingest.ok:
                return ScheduleResult(
                    planned_batches=planned,
                    executed_batches=executed,
                    probe_protocol_hash=probe_protocol_hash,
                )

    provisional_survivor_keys, provisional_screened_keys = _probe_survivor_keys(
        db,
        shapes=shapes,
        candidates=candidates,
        available_pairs=set(pair_owner),
        problem_type_hash=problem_type_hash,
        probe_protocol_hash=probe_protocol_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        policy=probe_policy,
        min_samples=probe_policy.initial_samples,
    )

    for item in prepared:
        topup_pairs = [
            pair for pair in item.validated_pairs if (pair.shape_id, pair.candidate_hash) in provisional_survivor_keys
        ]
        if not topup_pairs:
            continue
        counts = db.reusable_cache_entry_counts(
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=probe_protocol_hash,
            shape_ids=[shape.id for shape in item.planned.shapes],
            candidate_hashes=[candidate.hash for candidate in item.planned.candidates],
        )
        topup_pairs_by_remaining: dict[int, list[RunnablePair]] = {}
        for pair in topup_pairs:
            current_samples = counts.get((pair.shape_id, pair.candidate_hash), {}).get("ok", 0)
            remaining = probe_policy.samples - current_samples
            if remaining > 0:
                topup_pairs_by_remaining.setdefault(remaining, []).append(pair)
        for remaining, pairs in sorted(topup_pairs_by_remaining.items()):
            probe = _benchmark_prepared_pairs(
                db,
                item,
                pairs=pairs,
                protocol=probe_protocol.with_overrides(num_benchmarks=remaining),
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=probe_protocol_hash,
                runner_bin=runner_bin,
                runner_timeout_s=runner_timeout_s,
                phase="probe-topup",
            )
            executed.append(probe)
            if not keep_going and probe.ingest is not None and not probe.ingest.ok:
                return ScheduleResult(
                    planned_batches=planned,
                    executed_batches=executed,
                    probe_protocol_hash=probe_protocol_hash,
                )

    survivor_keys, final_screened_keys = _probe_survivor_keys(
        db,
        shapes=shapes,
        candidates=candidates,
        available_pairs=provisional_survivor_keys,
        problem_type_hash=problem_type_hash,
        probe_protocol_hash=probe_protocol_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        policy=probe_policy,
        min_samples=probe_policy.samples,
    )
    screened_keys = provisional_screened_keys | final_screened_keys

    # Phase 3: run the main timing protocol only for probe survivors.
    for item in prepared:
        main_pairs = [pair for pair in item.validated_pairs if (pair.shape_id, pair.candidate_hash) in survivor_keys]
        benchmark = _benchmark_prepared_pairs(
            db,
            item,
            pairs=main_pairs,
            protocol=protocol.with_overrides(num_benchmarks=item.planned.samples_per_pair),
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            runner_bin=runner_bin,
            runner_timeout_s=runner_timeout_s,
            phase="initial",
            include_preparation=True,
        )
        executed.append(benchmark)
        if not keep_going and benchmark.ingest is not None and not benchmark.ingest.ok:
            return ScheduleResult(
                planned_batches=planned,
                executed_batches=executed,
                probe_protocol_hash=probe_protocol_hash,
                probe_survivor_pairs=len(survivor_keys),
                probe_screened_pairs=len(screened_keys),
            )

    completed_rounds = 0
    for _ in range(adaptive_max_rounds):
        groups = _adaptive_topup_groups(
            db,
            shapes=shapes,
            candidates=candidates,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            policy=adaptive_policy,
            min_samples=initial_samples,
        )
        requests: dict[tuple[int, int], tuple[PreparedBatch, list[RunnablePair]]] = {}
        for target_samples, group_shapes, group_candidates in groups:
            counts = db.reusable_cache_entry_counts(
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                shape_ids=[shape.id for shape in group_shapes],
                candidate_hashes=[candidate.hash for candidate in group_candidates],
            )
            for shape in group_shapes:
                for candidate in group_candidates:
                    owner = pair_owner.get((shape.id, candidate.hash))
                    if owner is None:
                        continue
                    current_samples = counts.get((shape.id, candidate.hash), {}).get("ok", 0)
                    remaining = target_samples - current_samples
                    if remaining <= 0:
                        continue
                    prepared_batch, pair = owner
                    key = (id(prepared_batch), remaining)
                    request = requests.setdefault(key, (prepared_batch, []))
                    request[1].append(pair)
        if not requests:
            break

        ran_round = False
        for (_, remaining), (prepared_batch, pairs) in requests.items():
            topup = _benchmark_prepared_pairs(
                db,
                prepared_batch,
                pairs=pairs,
                protocol=protocol.with_overrides(num_benchmarks=remaining),
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                runner_bin=runner_bin,
                runner_timeout_s=runner_timeout_s,
                phase="adaptive",
            )
            executed.append(topup)
            ran_round = True
            if not keep_going and topup.ingest is not None and not topup.ingest.ok:
                return ScheduleResult(
                    planned_batches=planned,
                    executed_batches=executed,
                    adaptive_rounds=completed_rounds,
                    probe_protocol_hash=probe_protocol_hash,
                    probe_survivor_pairs=len(survivor_keys),
                    probe_screened_pairs=len(screened_keys),
                )
        if not ran_round:
            break
        completed_rounds += 1

    return ScheduleResult(
        planned_batches=planned,
        executed_batches=executed,
        adaptive_rounds=completed_rounds,
        probe_protocol_hash=probe_protocol_hash,
        probe_survivor_pairs=len(survivor_keys),
        probe_screened_pairs=len(screened_keys),
    )
