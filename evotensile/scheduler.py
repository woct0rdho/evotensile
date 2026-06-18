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
    if include_random:
        candidates.extend(initial_random_batch(num_random, seed=seed))

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
    return db.has_cached_evaluation(
        CacheKey(
            version_name=version_name,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
        ),
        min_samples=min_samples,
    )


def missing_shape_subset(
    db: EvoTensileDB,
    *,
    shapes: list[Shape],
    candidates: list[Candidate],
    version_name: str,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    min_samples: int,
    ignore_cache: bool = False,
) -> tuple[list[Shape], int]:
    missing_pairs = 0
    missing_shape_ids: set[str] = set()
    for shape in shapes:
        for candidate in candidates:
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
                missing_pairs += 1
                missing_shape_ids.add(shape.id)
    return [shape for shape in shapes if shape.id in missing_shape_ids], missing_pairs


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
            missing_shapes, missing_pairs = missing_shape_subset(
                db,
                shapes=shape_chunk,
                candidates=candidate_chunk,
                version_name=version,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                min_samples=min_samples,
                ignore_cache=ignore_cache,
            )
            if missing_pairs == 0:
                continue
            planned.append(
                PlannedBatch(
                    batch_index=batch_index,
                    candidates=list(candidate_chunk),
                    shapes=missing_shapes,
                    missing_pairs=missing_pairs,
                    nominal_pairs=len(candidate_chunk) * len(missing_shapes),
                )
            )
            batch_index += 1
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
        batch_shapes, missing_pairs = missing_shape_subset(
            db,
            shapes=batch.shapes,
            candidates=batch.candidates,
            version_name=version,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            min_samples=min_samples,
            ignore_cache=ignore_cache,
        )
        if missing_pairs == 0:
            continue
        current = PlannedBatch(
            batch_index=batch.batch_index,
            candidates=batch.candidates,
            shapes=batch_shapes,
            missing_pairs=missing_pairs,
            nominal_pairs=len(batch.candidates) * len(batch_shapes),
        )
        yaml_path, manifest_path, run_dir = write_batch_inputs(current, output_root, unique_run_dir=not generate_only)
        if generate_only:
            executed.append(ExecutedBatch(current, yaml_path, manifest_path, run_dir))
            continue

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
        if bench_result is not None:
            ingest = ingest_results(
                db=db,
                paths=[run_dir],
                manifest_path=manifest_path,
                version_name=version,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                run_id=bench_result.run_id,
                include_logs=True,
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
            not build_result.ok or bench_result is None or not bench_result.ok or (ingest is not None and not ingest.ok)
        )
        if failed and not keep_going:
            break
    return ScheduleResult(planned_batches=planned, executed_batches=executed)
