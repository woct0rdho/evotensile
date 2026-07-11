import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from evotensile.artifacts import register_artifact_bundle
from evotensile.candidate import stable_hash
from evotensile.database import BenchmarkEventInsert, EvoTensileDB
from evotensile.manifest import write_manifest
from evotensile.profile import TargetProfile
from evotensile.protocol import BenchmarkProtocol
from evotensile.runner import RunResult, run_tensilelite
from evotensile.scheduling.compile_cache import compile_cache_dir, compile_cache_lock, has_tensilelite_cache
from evotensile.scheduling.models import PlannedBatch, PreparedBatch
from evotensile.scheduling.structured import record_structured_run
from evotensile.search.cost_model import predicted_batch_prepare_weight
from evotensile.solution_mapping import find_solution_yamls
from evotensile.structured_runner import (
    RunnablePair,
    StructuredRunOutput,
    build_runnable_pairs,
    library_dir_from_build,
    run_structured_phase,
    validate_validation_samples,
)
from evotensile.tensilelite_diagnostics import attribution_inserts_from_diagnostics, run_tensilelite_diagnostics
from evotensile.yaml_writer import write_tensilelite_yaml


@dataclass(frozen=True)
class PreparationContext:
    db: EvoTensileDB
    output_root: str | Path
    target_profile: TargetProfile
    protocol: BenchmarkProtocol
    problem_type_hash: str
    benchmark_protocol_hash: str
    validation_protocol_hash: str
    tensilelite_bin: str | Path
    compile_threads: int | None
    runner_bin: str | Path
    build_timeout_s: float | None
    runner_timeout_s: float | None
    compile_cache_root: str | Path | None
    prepare_workers: int
    validation_workers: int
    cost_aware_scheduling: bool


def _batch_fingerprint(batch: PlannedBatch) -> str:
    payload = {
        "candidates": [candidate.hash for candidate in batch.candidates],
        "requires_validation": batch.requires_validation,
        "samples_per_pair": batch.samples_per_pair,
        "shapes": [shape.id for shape in batch.shapes],
    }
    return stable_hash(payload, prefix="batch_")[:18]


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


def _prepare_current_batch(
    context: PreparationContext,
    current: PlannedBatch,
    *,
    validation_gate: threading.Semaphore,
) -> PreparedBatch:
    db = context.db
    output_root = context.output_root
    target_profile = context.target_profile
    protocol = context.protocol
    problem_type_hash = context.problem_type_hash
    benchmark_protocol_hash = context.benchmark_protocol_hash
    validation_protocol_hash = context.validation_protocol_hash
    tensilelite_bin = context.tensilelite_bin
    compile_threads = context.compile_threads
    runner_bin = context.runner_bin
    build_timeout_s = context.build_timeout_s
    runner_timeout_s = context.runner_timeout_s
    compile_cache_root = context.compile_cache_root
    build_protocol = protocol.with_overrides(num_benchmarks=current.samples_per_pair)
    yaml_path, manifest_path, run_dir = write_batch_inputs(
        current,
        output_root,
        target_profile=target_profile,
        protocol=build_protocol,
        unique_run_dir=True,
    )
    cache_dir = compile_cache_dir(
        compile_cache_root,
        current,
        target_profile=target_profile,
        protocol=build_protocol,
    )
    build_dir = cache_dir or run_dir

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
            use_cache=cache_dir is not None and has_tensilelite_cache(cache_dir),
            candidate_hashes=[candidate.hash for candidate in current.candidates],
        )

    if cache_dir is None:
        build_result = build()
    else:
        with compile_cache_lock(cache_dir):
            build_result = build()
            if build_result.ok:
                (cache_dir / ".evotensile_compile_cache_ok").write_text("ok\n", encoding="utf-8")

    preparation_inserts: list[BenchmarkEventInsert] = []
    errors: list[str] = []
    planned_pairs = {(shape.id, candidate.hash) for shape in current.shapes for candidate in current.candidates}
    solution_yamls = [str(path) for path in find_solution_yamls([build_dir])]
    runnable, missing = build_runnable_pairs(
        manifest_path=manifest_path,
        solution_yaml_paths=solution_yamls,
        planned_pairs=planned_pairs,
        build_run_id=build_result.run_id,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
    )
    library_dir = library_dir_from_build(build_dir)

    if not build_result.ok and len(current.candidates) == 1 and not runnable:
        status = "build_timeout" if build_result.timed_out else "build_failed"
        preparation_inserts = [
            BenchmarkEventInsert(
                shape_id=shape.id,
                candidate_hash=current.candidates[0].hash,
                run_id=build_result.run_id,
                status=status,
                source_kind="native_run",
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
            )
            for shape in current.shapes
        ]
        runnable = []
    elif build_result.ok:
        preparation_inserts.extend(
            BenchmarkEventInsert(
                shape_id=item.shape_id,
                candidate_hash=item.candidate_hash,
                run_id=build_result.run_id,
                status=item.status,
                source_kind="native_run",
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
                candidate_hashes=[candidate.hash for candidate in current.candidates],
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
            register_artifact_bundle(
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
            record_structured_run(
                db,
                validation_result,
                yaml_path=yaml_path,
                output_dir=run_dir,
                pairs=runnable,
                cost_phase="validation",
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
        db.insert_benchmark_events(preparation_inserts)
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


def prepare_wave(context: PreparationContext, batches: list[PlannedBatch]) -> list[PreparedBatch]:
    validation_gate = threading.Semaphore(context.validation_workers)
    preparation_order = batches
    if context.cost_aware_scheduling:
        preparation_order = sorted(
            batches,
            key=lambda batch: (
                -predicted_batch_prepare_weight(
                    batch.candidates,
                    batch.shapes,
                    workgroup_processor_count=context.target_profile.workgroup_processor_count,
                ),
                batch.batch_index,
            ),
        )

    def prepare(batch: PlannedBatch) -> PreparedBatch:
        return _prepare_current_batch(context, batch, validation_gate=validation_gate)

    with ThreadPoolExecutor(max_workers=context.prepare_workers) as executor:
        prepared = list(executor.map(prepare, preparation_order))
    prepared_by_index = {item.planned.batch_index: item for item in prepared}
    return [prepared_by_index[batch.batch_index] for batch in batches]
