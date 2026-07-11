from collections.abc import Callable, Sequence
from pathlib import Path

from .adaptive_retime import AdaptivePolicy, ProbePolicy
from .candidate import Candidate, Shape
from .database import EvoTensileDB
from .profile import DEFAULT_PROFILE, TargetProfile
from .protocol import BenchmarkProtocol
from .runner import DEFAULT_TENSILELITE_BIN
from .scheduling.models import ExecutedBatch, PreparedBatch, ScheduleResult
from .scheduling.planning import plan_batches, preprepare_probe_screened_pairs, record_shape_rule_rejections
from .scheduling.preparation import PreparationContext, prepare_wave, write_batch_inputs
from .scheduling.timing import TimingContext, run_serial_timing
from .subprocess_utils import resolve_timeout

DEFAULT_COMPILE_THREADS = 1


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
    prepare_wave_batches: int | None = None,
    admit_next_wave: Callable[[ScheduleResult], bool] | None = None,
    timing_batch_order: Callable[[Sequence[PreparedBatch]], Sequence[PreparedBatch]] | None = None,
) -> ScheduleResult:
    if not dry_run and not generate_only and runner_bin is None:
        raise ValueError("--runner-bin is required")
    if prepare_workers is not None and prepare_workers <= 0:
        raise ValueError("prepare_workers must be positive")
    if validation_workers is not None and validation_workers <= 0:
        raise ValueError("validation_workers must be positive")
    if prepare_wave_batches is not None and prepare_wave_batches <= 0:
        raise ValueError("prepare_wave_batches must be positive")
    if adaptive_policy is not None and probe_policy is None:
        raise ValueError("probe_policy is required when adaptive sampling is enabled")
    if adaptive_max_rounds < 0:
        raise ValueError("adaptive_max_rounds must be non-negative")

    protocol = protocol or target_profile.default_protocol
    if protocol.role != "main":
        raise ValueError("execute_schedule requires a main benchmark protocol")
    resolved_prepare_workers = target_profile.default_prepare_workers if prepare_workers is None else prepare_workers
    resolved_validation_workers = (
        target_profile.default_validation_workers if validation_workers is None else validation_workers
    )
    resolved_wave_batches = (
        target_profile.default_prepare_wave_batches if prepare_wave_batches is None else prepare_wave_batches
    )
    problem_type_hash = target_profile.problem_type_hash
    benchmark_protocol_hash = target_profile.benchmark_protocol_hash(protocol)
    validation_protocol_hash = protocol.validation_protocol_hash()
    build_timeout_s = resolve_timeout(build_timeout_s, target_profile.default_build_timeout_s)
    runner_timeout_s = resolve_timeout(runner_timeout_s, target_profile.default_runner_timeout_s)
    initial_samples = max(min_samples, protocol.num_benchmarks)
    effective_candidate_batch_size = 1 if compile_cache_root is not None else candidate_batch_size

    probe_protocol = None
    probe_protocol_hash = None
    probe_policy_hash = None
    preprepare_screened_pairs: set[tuple[str, str]] = set()

    db.init()
    db.register_candidates(candidates)
    db.register_shapes(shapes)
    if not dry_run and not generate_only:
        record_shape_rule_rejections(
            db,
            shapes=shapes,
            candidates=candidates,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
        )
    if adaptive_policy is not None:
        assert probe_policy is not None
        probe_policy_hash = probe_policy.policy_hash
        probe_protocol = protocol.with_overrides(
            role="probe",
            num_warmups=0,
            num_benchmarks=probe_policy.samples,
            enqueues_per_sync=1,
            syncs_per_benchmark=1,
            num_elements_to_validate=0,
        )
        probe_protocol_hash = target_profile.benchmark_protocol_hash(probe_protocol)
        if not ignore_cache:
            preprepare_screened_pairs = preprepare_probe_screened_pairs(
                db,
                shapes=shapes,
                candidates=candidates,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                probe_protocol_hash=probe_protocol_hash,
                policy=probe_policy,
            )

    planned = plan_batches(
        db,
        shapes=shapes,
        candidates=candidates,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        validation_protocol_hash=validation_protocol_hash,
        min_samples=initial_samples,
        candidate_batch_size=effective_candidate_batch_size,
        shape_batch_size=shape_batch_size,
        ignore_cache=ignore_cache,
        max_batches=max_batches,
        excluded_pairs=preprepare_screened_pairs,
    )
    if dry_run:
        return ScheduleResult(
            planned_batches=planned,
            probe_protocol_hash=probe_protocol_hash,
            probe_policy_hash=probe_policy_hash,
            probe_screened_pairs=len(preprepare_screened_pairs),
            probe_preprepare_screened_pairs=len(preprepare_screened_pairs),
        )
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
        return ScheduleResult(
            planned_batches=planned,
            executed_batches=generated,
            probe_protocol_hash=probe_protocol_hash,
            probe_policy_hash=probe_policy_hash,
            probe_screened_pairs=len(preprepare_screened_pairs),
            probe_preprepare_screened_pairs=len(preprepare_screened_pairs),
        )

    assert runner_bin is not None
    executed: list[ExecutedBatch] = []
    completed_waves = 0
    adaptive_rounds = 0
    probe_survivor_pairs = 0
    probe_screened_pairs = len(preprepare_screened_pairs)
    for wave_start in range(0, len(planned), resolved_wave_batches):
        if completed_waves and admit_next_wave is not None:
            progress = ScheduleResult(
                planned_batches=planned,
                executed_batches=executed,
                completed_waves=completed_waves,
                adaptive_rounds=adaptive_rounds,
                probe_protocol_hash=probe_protocol_hash,
                probe_policy_hash=probe_policy_hash,
                probe_survivor_pairs=probe_survivor_pairs,
                probe_screened_pairs=probe_screened_pairs,
                probe_preprepare_screened_pairs=len(preprepare_screened_pairs),
            )
            if not admit_next_wave(progress):
                break

        wave = planned[wave_start : wave_start + resolved_wave_batches]
        wave_shapes = list({shape.id: shape for batch in wave for shape in batch.shapes}.values())
        wave_candidates = list({candidate.hash: candidate for batch in wave for candidate in batch.candidates}.values())
        prepared = prepare_wave(
            PreparationContext(
                db=db,
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
                prepare_workers=resolved_prepare_workers,
                validation_workers=resolved_validation_workers,
                cost_aware_scheduling=cost_aware_scheduling,
            ),
            wave,
        )
        wave_result = run_serial_timing(
            TimingContext(
                db=db,
                shapes=wave_shapes,
                candidates=wave_candidates,
                planned_batches=wave,
                protocol=protocol,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                runner_bin=runner_bin,
                runner_timeout_s=runner_timeout_s,
                keep_going=keep_going,
                initial_samples=initial_samples,
                adaptive_policy=adaptive_policy,
                probe_policy=probe_policy,
                probe_protocol=probe_protocol,
                probe_protocol_hash=probe_protocol_hash,
                probe_policy_hash=probe_policy_hash,
                adaptive_max_rounds=adaptive_max_rounds,
                timing_batch_order=timing_batch_order,
            ),
            prepared,
        )
        executed.extend(wave_result.executed_batches)
        completed_waves += 1
        adaptive_rounds += wave_result.adaptive_rounds
        probe_survivor_pairs += wave_result.probe_survivor_pairs
        probe_screened_pairs += wave_result.probe_screened_pairs
        if not keep_going and any(
            batch.ingest is not None and not batch.ingest.ok for batch in wave_result.executed_batches
        ):
            break

    return ScheduleResult(
        planned_batches=planned,
        executed_batches=executed,
        completed_waves=completed_waves,
        adaptive_rounds=adaptive_rounds,
        probe_protocol_hash=probe_protocol_hash,
        probe_policy_hash=probe_policy_hash,
        probe_survivor_pairs=probe_survivor_pairs,
        probe_screened_pairs=probe_screened_pairs,
        probe_preprepare_screened_pairs=len(preprepare_screened_pairs),
    )
