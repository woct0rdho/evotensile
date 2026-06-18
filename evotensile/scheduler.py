import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from .cache import CacheKey, normalize_version_name
from .candidate import Candidate, Shape, stable_hash
from .database import EvoTensileDB
from .ingest import IngestResult, ingest_results
from .manifest import write_manifest
from .runner import DEFAULT_TENSILELITE_BIN, build_then_benchmark
from .search.differential_evolution import differential_evolution_candidates
from .search.gomea import gomea_candidates, gomea_neighborhood_candidates
from .search.local_search import mutate_elites
from .search.random_search import initial_random_batch
from .shapes import shape_from_id
from .solution_mapping import find_solution_yamls
from .yaml_writer import write_tensilelite_yaml


@dataclass(frozen=True)
class PlannedBatch:
    batch_index: int
    candidates: list[Candidate]
    shapes: list[Shape]
    missing_pairs: int
    nominal_pairs: int

    @property
    def extra_pairs(self) -> int:
        return self.nominal_pairs - self.missing_pairs


@dataclass(frozen=True)
class ExecutedBatch:
    planned: PlannedBatch
    yaml_path: Path
    manifest_path: Path
    output_dir: Path
    build_returncode: int | None = None
    benchmark_returncode: int | None = None
    ingest: IngestResult | None = None


@dataclass(frozen=True)
class ScheduleResult:
    planned_batches: list[PlannedBatch]
    executed_batches: list[ExecutedBatch] = field(default_factory=list)

    @property
    def missing_pairs(self) -> int:
        return sum(batch.missing_pairs for batch in self.planned_batches)

    @property
    def nominal_pairs(self) -> int:
        return sum(batch.nominal_pairs for batch in self.planned_batches)


T = TypeVar("T")

PROPOSAL_MODES = (
    "seed-random",
    "local",
    "seed-random-local",
    "de",
    "seed-random-de",
    "gomea",
    "seed-random-gomea",
    "evolutionary",
)
DEFAULT_PROPOSAL = "seed-random-gomea"
DEFAULT_NUM_RANDOM = 64
DEFAULT_ELITE_COUNT = 8
DEFAULT_LOCAL_COUNT = 32
DEFAULT_DE_COUNT = 32
DEFAULT_GOMEA_COUNT = 64
DEFAULT_TRANSFER_SHAPES = 4
DEFAULT_TRANSFER_PER_SHAPE = 2
DEFAULT_MUTATION_RATE = 0.25
DEFAULT_CROSSOVER_RATE = 0.8
DEFAULT_RANDOM_GENE_RATE = 0.1


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    by_hash: dict[str, Candidate] = {}
    for candidate in candidates:
        by_hash.setdefault(candidate.hash, candidate)
    return list(by_hash.values())


def _ranked_elites(
    db: EvoTensileDB,
    *,
    version_name: str | None,
    problem_type_hash: str | None,
    benchmark_protocol_hash: str | None,
    shape_id: str | None,
    elite_count: int,
) -> list[Candidate]:
    summaries = db.rank_evaluations(
        version_name=version_name,
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


def _nearest_shape_ids(targets: list[Shape], source_shape_ids: set[str], *, limit: int) -> list[str]:
    if limit <= 0 or not targets or not source_shape_ids:
        return []
    source_shapes: list[Shape] = []
    target_ids = {shape.id for shape in targets}
    for shape_id in source_shape_ids:
        if shape_id in target_ids:
            continue
        try:
            source_shapes.append(shape_from_id(shape_id))
        except ValueError:
            continue

    best_by_shape: dict[str, float] = {}
    for source in source_shapes:
        # Use nearest target distance so a 100-shape run imports winners from neighborhoods that cover the grid.
        best_by_shape[source.id] = min(_shape_distance(source, target) for target in targets)
    return [shape_id for shape_id, _ in sorted(best_by_shape.items(), key=lambda item: (item[1], item[0]))[:limit]]


def _transfer_elites(
    db: EvoTensileDB,
    *,
    target_shapes: list[Shape],
    version_name: str | None,
    problem_type_hash: str | None,
    benchmark_protocol_hash: str | None,
    nearest_shape_count: int,
    per_shape: int,
) -> list[Candidate]:
    if nearest_shape_count <= 0 or per_shape <= 0 or not target_shapes:
        return []
    summaries = db.rank_evaluations(
        version_name=version_name,
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
            version_name=version_name,
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


def propose_candidates(
    db: EvoTensileDB,
    *,
    proposal: str = DEFAULT_PROPOSAL,
    num_random: int = DEFAULT_NUM_RANDOM,
    seed: int = 1,
    version_name: str | None = None,
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
) -> list[Candidate]:
    """Build the candidate set for a scheduled run from random seeds and/or cached elites."""
    if proposal not in PROPOSAL_MODES:
        raise ValueError(f"unknown proposal mode: {proposal}")

    candidates: list[Candidate] = []
    include_random = proposal.startswith("seed-random") or proposal == "evolutionary"
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
            version_name=version_name,
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
            version_name=version_name,
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

    if include_random:
        candidates.extend(initial_random_batch(num_random, seed=seed))

    if proposal in {"local", "seed-random-local", "evolutionary"} and local_count > 0:
        candidates.extend(mutate_elites(elites, count=local_count, seed=seed + 1009, mutation_rate=mutation_rate))

    if proposal in {"de", "seed-random-de", "evolutionary"} and de_count > 0:
        parents = _dedupe_candidates([*elites, *candidates])
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
        parents = _dedupe_candidates([*elites, *candidates])
        neighborhood_parents = _dedupe_candidates([*candidates, *elites])
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
            )
        )

    return _dedupe_candidates(candidates)


def _chunks(items: list[T], size: int) -> list[list[T]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    return [items[i : i + size] for i in range(0, len(items), size)]


def _is_cached(
    db: EvoTensileDB,
    *,
    version_name: str,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    shape: Shape,
    candidate: Candidate,
    min_samples: int,
    ignore_cache: bool,
) -> bool:
    if ignore_cache:
        return False
    return db.has_reusable_cache_entry(
        CacheKey(
            version_name=version_name,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
        ),
        min_ok_samples=min_samples,
    )


def _missing_candidate_indices_by_shape(
    db: EvoTensileDB,
    *,
    shapes: list[Shape],
    candidates: list[Candidate],
    version_name: str,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    min_samples: int,
    ignore_cache: bool = False,
) -> dict[int, tuple[int, ...]]:
    missing: dict[int, tuple[int, ...]] = {}
    for shape_index, shape in enumerate(shapes):
        missing_indices: list[int] = []
        for candidate_index, candidate in enumerate(candidates):
            if not _is_cached(
                db,
                version_name=version_name,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                shape=shape,
                candidate=candidate,
                min_samples=min_samples,
                ignore_cache=ignore_cache,
            ):
                missing_indices.append(candidate_index)
        if missing_indices:
            missing[shape_index] = tuple(missing_indices)
    return missing


def _pair_exact_batches(
    *,
    batch_index_start: int,
    shapes: list[Shape],
    candidates: list[Candidate],
    missing_by_shape: dict[int, tuple[int, ...]],
    max_batches: int | None = None,
) -> list[PlannedBatch]:
    grouped_shapes: dict[tuple[int, ...], list[Shape]] = {}
    for shape_index, missing_indices in missing_by_shape.items():
        grouped_shapes.setdefault(missing_indices, []).append(shapes[shape_index])

    planned: list[PlannedBatch] = []
    batch_index = batch_index_start
    for missing_indices, group_shapes in grouped_shapes.items():
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
    version_name: str,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    min_samples: int = 1,
    candidate_batch_size: int = 32,
    shape_batch_size: int = 100,
    ignore_cache: bool = False,
    max_batches: int | None = None,
) -> list[PlannedBatch]:
    version = normalize_version_name(version_name)
    planned: list[PlannedBatch] = []
    batch_index = 0
    for candidate_chunk in _chunks(candidates, candidate_batch_size):
        for shape_chunk in _chunks(shapes, shape_batch_size):
            missing_by_shape = _missing_candidate_indices_by_shape(
                db,
                shapes=shape_chunk,
                candidates=candidate_chunk,
                version_name=version,
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
        "shapes": [shape.id for shape in batch.shapes],
    }
    return stable_hash(payload, prefix="batch_")[:18]


def write_batch_inputs(
    batch: PlannedBatch, output_root: str | Path, *, unique_run_dir: bool = False
) -> tuple[Path, Path, Path]:
    batch_dir = Path(output_root) / f"batch_{batch.batch_index:04d}_{_batch_fingerprint(batch)}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = batch_dir / "config.yaml"
    manifest_path = batch_dir / "config.manifest.csv"
    run_dir = batch_dir / (f"run_{uuid.uuid4().hex[:8]}" if unique_run_dir else "run")
    write_tensilelite_yaml(yaml_path, batch.candidates, batch.shapes)
    write_manifest(manifest_path, batch.candidates, batch.shapes)
    return yaml_path, manifest_path, run_dir


def _record_batch_status(
    db: EvoTensileDB,
    batch: PlannedBatch,
    *,
    status: str,
    run_id: str | None,
    version_name: str,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
) -> int:
    inserted = 0
    for shape in batch.shapes:
        for candidate in batch.candidates:
            db.insert_evaluation(
                shape_id=shape.id,
                candidate_hash=candidate.hash,
                run_id=run_id,
                status=status,
                version_name=version_name,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
            )
            inserted += 1
    return inserted


def execute_schedule(
    db: EvoTensileDB,
    *,
    shapes: list[Shape],
    candidates: list[Candidate],
    output_root: str | Path,
    version_name: str,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    min_samples: int = 1,
    candidate_batch_size: int = 32,
    shape_batch_size: int = 100,
    ignore_cache: bool = False,
    max_batches: int | None = None,
    dry_run: bool = False,
    generate_only: bool = False,
    tensilelite_bin: str | Path = DEFAULT_TENSILELITE_BIN,
    compile_threads: int | None = -1,
    benchmark_threads: int | None = 1,
    global_parameters: list[str] | None = None,
    extra_args: list[str] | None = None,
    keep_going: bool = False,
) -> ScheduleResult:
    db.init()
    db.register_candidates(candidates)
    db.register_shapes(shapes)
    version = normalize_version_name(version_name)
    planned = plan_batches(
        db,
        shapes=shapes,
        candidates=candidates,
        version_name=version,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        min_samples=min_samples,
        candidate_batch_size=candidate_batch_size,
        shape_batch_size=shape_batch_size,
        ignore_cache=ignore_cache,
        max_batches=max_batches,
    )
    if dry_run:
        return ScheduleResult(planned_batches=planned)

    executed: list[ExecutedBatch] = []
    for batch in planned:
        # Recheck just before execution so a resumed run skips observations ingested by earlier batches.
        missing_by_shape = _missing_candidate_indices_by_shape(
            db,
            shapes=batch.shapes,
            candidates=batch.candidates,
            version_name=version,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            min_samples=min_samples,
            ignore_cache=ignore_cache,
        )
        if not missing_by_shape:
            continue
        current_batches = _pair_exact_batches(
            batch_index_start=batch.batch_index,
            shapes=batch.shapes,
            candidates=batch.candidates,
            missing_by_shape=missing_by_shape,
        )
        for current in current_batches:
            yaml_path, manifest_path, run_dir = write_batch_inputs(
                current,
                output_root,
                unique_run_dir=not generate_only,
            )
            if generate_only:
                executed.append(ExecutedBatch(current, yaml_path, manifest_path, run_dir))
                continue

            # Keep compile and benchmark sequential. On Strix Halo the CPU compiler
            # and integrated GPU share power/thermal headroom, so we avoid deliberate
            # compile/benchmark overlap even though compilation itself may be threaded.
            build_result, bench_result = build_then_benchmark(
                yaml_path,
                run_dir,
                tensilelite_bin=tensilelite_bin,
                db=db,
                compile_threads=compile_threads,
                benchmark_threads=benchmark_threads,
                global_parameters=global_parameters,
                version_name=version,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                extra_args=extra_args,
            )
            ingest: IngestResult | None = None
            if bench_result is None and len(current.candidates) == 1:
                _record_batch_status(
                    db,
                    current,
                    status="build_failed",
                    run_id=build_result.run_id,
                    version_name=version,
                    problem_type_hash=problem_type_hash,
                    benchmark_protocol_hash=benchmark_protocol_hash,
                )
            if bench_result is not None:
                # Ingest only the benchmark stdout for timing rows. The same run
                # directory also contains the build-only stdout, and build-only can
                # emit cold/partial CSV rows that should not enter the hot-loop cache.
                ingest = ingest_results(
                    db=db,
                    paths=[bench_result.stdout_path],
                    manifest_path=manifest_path,
                    version_name=version,
                    problem_type_hash=problem_type_hash,
                    benchmark_protocol_hash=benchmark_protocol_hash,
                    run_id=bench_result.run_id,
                    include_logs=True,
                    solutions_yaml=[str(path) for path in find_solution_yamls([run_dir])],
                )
            executed.append(
                ExecutedBatch(
                    planned=current,
                    yaml_path=yaml_path,
                    manifest_path=manifest_path,
                    output_dir=run_dir,
                    build_returncode=build_result.returncode,
                    benchmark_returncode=bench_result.returncode if bench_result is not None else None,
                    ingest=ingest,
                )
            )
            failed = (
                not build_result.ok
                or bench_result is None
                or not bench_result.ok
                or (ingest is not None and not ingest.ok)
            )
            if failed and not keep_going:
                return ScheduleResult(planned_batches=planned, executed_batches=executed)
    return ScheduleResult(planned_batches=planned, executed_batches=executed)
