import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .adaptive_retime import AdaptivePolicy
from .candidate import Candidate, Shape
from .database import EvoTensileDB
from .profile import DEFAULT_PROFILE, PROFILES, TargetProfile, get_profile
from .protocol import BenchmarkProtocol
from .rejection_mining import classification_counts, summarize_rejection_logs
from .runner import DEFAULT_TENSILELITE_BIN
from .scheduler import (
    DEFAULT_COMPILE_THREADS,
    DEFAULT_CROSSOVER_RATE,
    DEFAULT_DE_COUNT,
    DEFAULT_ELITE_COUNT,
    DEFAULT_GOMEA_COUNT,
    DEFAULT_LOCAL_COUNT,
    DEFAULT_MUTATION_RATE,
    DEFAULT_NUM_RANDOM,
    DEFAULT_PROPOSAL,
    DEFAULT_RANDOM_GENE_RATE,
    DEFAULT_TRANSFER_PER_SHAPE,
    DEFAULT_TRANSFER_SHAPES,
    PROPOSAL_MODES,
    ScheduleResult,
    default_batch_workers,
    detect_underperforming_shapes,
    execute_schedule,
    propose_candidates,
    repair_seed_candidates,
)
from .search.coverage import candidate_coverage
from .search.random_search import initial_random_batch
from .search_space import DOMAINS, MATRIX_INSTRUCTIONS, macro_tile
from .shapes import parse_shape


def _profile(args: argparse.Namespace) -> TargetProfile:
    return get_profile(getattr(args, "profile", None))


def _protocol(args: argparse.Namespace, profile: TargetProfile) -> BenchmarkProtocol:
    protocol = profile.default_protocol.with_overrides(
        num_warmups=getattr(args, "num_warmups", None),
        num_benchmarks=getattr(args, "num_benchmarks", None),
        enqueues_per_sync=getattr(args, "enqueues_per_sync", None),
        syncs_per_benchmark=getattr(args, "syncs_per_benchmark", None),
        num_elements_to_validate=getattr(args, "num_elements_to_validate", None),
    )
    return protocol


def _parse_shapes(args: argparse.Namespace, profile: TargetProfile) -> list[Shape]:
    if getattr(args, "shapes", None):
        return [parse_shape(s) for s in args.shapes]
    shapes = profile.shapes()
    if getattr(args, "limit_shapes", None):
        return shapes[: args.limit_shapes]
    return shapes


def _adaptive_policy(args: argparse.Namespace) -> AdaptivePolicy | None:
    if getattr(args, "fixed_sampling", False):
        return None
    return AdaptivePolicy(
        epsilon_pct=args.adaptive_epsilon_pct,
        confidence=args.adaptive_confidence,
        min_retime_samples=args.adaptive_min_samples,
        max_retime_samples=args.adaptive_max_samples,
        sample_step=args.adaptive_sample_step,
        max_k=args.adaptive_max_k,
        min_effect_pct=args.adaptive_min_effect_pct,
    )


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    by_hash: dict[str, Candidate] = {}
    for candidate in candidates:
        by_hash.setdefault(candidate.hash, candidate)
    return list(by_hash.values())


def _timeout_arg(value: float | None, default: float | None) -> float | None:
    if value is None:
        return default
    if value <= 0:
        return None
    return value


def _candidates(args: argparse.Namespace):
    return initial_random_batch(args.num_random, seed=args.seed)


def _add_profile_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None, help="Target profile")


def _add_candidate_shape_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-random", type=int, default=DEFAULT_NUM_RANDOM)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit-shapes", type=int, default=None)
    parser.add_argument("--shapes", nargs="*")


def _add_cache_identity_args(parser: argparse.ArgumentParser) -> None:
    _add_profile_arg(parser)


def _add_protocol_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-warmups", type=int, default=None)
    parser.add_argument("--num-benchmarks", type=int, default=None)
    parser.add_argument("--enqueues-per-sync", type=int, default=None)
    parser.add_argument("--syncs-per-benchmark", type=int, default=None)
    parser.add_argument("--num-elements-to-validate", type=int, default=None)


def _add_timeout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--build-timeout",
        type=float,
        default=None,
        help="TensileLite build timeout in seconds; defaults to the target profile, 0 disables",
    )
    parser.add_argument(
        "--runner-timeout",
        type=float,
        default=None,
        help="Structured runner timeout in seconds; defaults to the target profile, 0 disables",
    )


@dataclass(frozen=True)
class ScheduleCliContext:
    profile: TargetProfile
    protocol: BenchmarkProtocol
    db: EvoTensileDB
    problem_hash: str
    protocol_hash: str
    shapes: list[Shape]
    runner_bin: str
    build_timeout: float | None
    runner_timeout: float | None
    adaptive_policy: AdaptivePolicy | None


def _resolve_candidate_batch_size(args: argparse.Namespace, profile: TargetProfile) -> int:
    if args.candidate_batch_size is not None:
        return args.candidate_batch_size
    return 1 if args.proposal in PROPOSAL_MODES else profile.default_candidate_batch_size


def _validate_schedule_args(args: argparse.Namespace) -> None:
    profile = _profile(args)
    args.candidate_batch_size = _resolve_candidate_batch_size(args, profile)
    positive_ints = (
        "candidate_batch_size",
        "shape_batch_size",
        "min_samples",
        "adaptive_initial_samples",
        "batch_workers",
    )
    for name in positive_ints:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    nonnegative_ints = (
        "num_random",
        "elite_count",
        "local_count",
        "de_count",
        "gomea_count",
        "transfer_shapes",
        "transfer_per_shape",
        "adaptive_max_rounds",
        "adaptive_min_samples",
        "adaptive_max_samples",
        "adaptive_sample_step",
        "adaptive_max_k",
    )
    for name in nonnegative_ints:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.adaptive_max_samples < args.adaptive_min_samples:
        raise ValueError("--adaptive-max-samples must be >= --adaptive-min-samples")


def _schedule_context(args: argparse.Namespace) -> ScheduleCliContext:
    _validate_schedule_args(args)
    profile = _profile(args)
    protocol = _protocol(args, profile)
    db = EvoTensileDB.connect(args.db)
    db.init()
    return ScheduleCliContext(
        profile=profile,
        protocol=protocol,
        db=db,
        problem_hash=profile.problem_type_hash,
        protocol_hash=profile.benchmark_protocol_hash(protocol),
        shapes=_parse_shapes(args, profile),
        runner_bin=args.runner_bin or profile.default_runner_bin,
        build_timeout=_timeout_arg(args.build_timeout, profile.default_build_timeout_s),
        runner_timeout=_timeout_arg(args.runner_timeout, profile.default_runner_timeout_s),
        adaptive_policy=_adaptive_policy(args),
    )


def _propose_candidates_for_shapes(
    db: EvoTensileDB,
    args: argparse.Namespace,
    *,
    problem_hash: str,
    protocol_hash: str,
    shapes: list[Shape],
    proposal_shape_id: str | None = None,
) -> list[Candidate]:
    return propose_candidates(
        db,
        proposal=args.proposal,
        num_random=args.num_random,
        seed=args.seed,
        problem_type_hash=problem_hash,
        benchmark_protocol_hash=protocol_hash,
        shape_id=proposal_shape_id,
        target_shapes=shapes,
        transfer_shape_count=args.transfer_shapes,
        transfer_per_shape=args.transfer_per_shape,
        elite_count=args.elite_count,
        local_count=args.local_count,
        de_count=args.de_count,
        gomea_count=args.gomea_count,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        random_gene_rate=args.random_gene_rate,
    )


def _execute_schedule_from_args(
    args: argparse.Namespace,
    context: ScheduleCliContext,
    *,
    shapes: list[Shape],
    candidates: list[Candidate],
    dry_run: bool | None = None,
) -> ScheduleResult:
    return execute_schedule(
        context.db,
        shapes=shapes,
        candidates=candidates,
        output_root=args.output_dir,
        target_profile=context.profile,
        protocol=context.protocol,
        min_samples=args.min_samples,
        candidate_batch_size=args.candidate_batch_size,
        shape_batch_size=args.shape_batch_size,
        ignore_cache=args.ignore_cache,
        max_batches=args.max_batches,
        dry_run=args.dry_run if dry_run is None else dry_run,
        generate_only=args.generate_only,
        tensilelite_bin=args.tensilelite_bin,
        compile_threads=args.compile_threads,
        keep_going=not args.stop_on_error,
        runner_bin=context.runner_bin,
        build_timeout_s=context.build_timeout,
        runner_timeout_s=context.runner_timeout,
        adaptive_policy=context.adaptive_policy,
        adaptive_initial_samples=args.adaptive_initial_samples,
        adaptive_max_rounds=args.adaptive_max_rounds,
        batch_workers=args.batch_workers,
    )


def _schedule_metadata_common(
    args: argparse.Namespace,
    context: ScheduleCliContext,
    *,
    result: ScheduleResult,
    candidates: list[Candidate],
    shapes: list[Shape],
) -> dict[str, object]:
    return {
        "db": args.db,
        "output_dir": args.output_dir,
        "profile": context.profile.name,
        "problem_type_hash": context.problem_hash,
        "benchmark_protocol_hash": context.protocol_hash,
        "protocol": context.protocol.global_parameters(),
        "proposal": args.proposal,
        "candidates": len(candidates),
        "shapes": len(shapes),
        "candidate_batch_size": args.candidate_batch_size,
        "shape_batch_size": args.shape_batch_size,
        "batch_workers": args.batch_workers,
        "min_samples": args.min_samples,
        "ignore_cache": args.ignore_cache,
        "dry_run": args.dry_run,
        "generate_only": args.generate_only,
        "stop_on_error": args.stop_on_error,
        "runner_bin": str(context.runner_bin) if context.runner_bin else None,
        "build_timeout_s": context.build_timeout,
        "runner_timeout_s": context.runner_timeout,
        "adaptive_sampling": context.adaptive_policy is not None,
        "adaptive_initial_samples": args.adaptive_initial_samples,
        "adaptive_max_rounds": args.adaptive_max_rounds,
        "adaptive_rounds": result.adaptive_rounds,
        "adaptive_policy": None if context.adaptive_policy is None else context.adaptive_policy.__dict__,
        "planned_batches": len(result.planned_batches),
        "planned_missing_pairs": result.missing_pairs,
        "planned_nominal_pairs": result.nominal_pairs,
        "planned_missing_samples": sum(batch.missing_samples for batch in result.planned_batches),
        "planned_nominal_samples": sum(batch.nominal_samples for batch in result.planned_batches),
    }


def _add_proposal_args(parser: argparse.ArgumentParser, *, repair: bool = False) -> None:
    parser.add_argument("--proposal", choices=PROPOSAL_MODES, default=DEFAULT_PROPOSAL)
    if not repair:
        parser.add_argument("--proposal-shape-id", default=None, help="Limit cached elite selection to one shape id")
    parser.add_argument(
        "--transfer-shapes",
        type=int,
        default=DEFAULT_TRANSFER_SHAPES,
        help="Seed normal proposal generation from this many nearest already-tuned shapes; 0 disables transfer",
    )
    parser.add_argument(
        "--transfer-per-shape",
        type=int,
        default=DEFAULT_TRANSFER_PER_SHAPE,
        help="Seed normal proposal generation with this many top candidates per nearest shape",
    )
    parser.add_argument("--elite-count", type=int, default=DEFAULT_ELITE_COUNT)
    parser.add_argument("--local-count", type=int, default=DEFAULT_LOCAL_COUNT)
    parser.add_argument("--de-count", type=int, default=DEFAULT_DE_COUNT)
    parser.add_argument("--gomea-count", type=int, default=DEFAULT_GOMEA_COUNT)
    parser.add_argument("--mutation-rate", type=float, default=DEFAULT_MUTATION_RATE)
    parser.add_argument("--crossover-rate", type=float, default=DEFAULT_CROSSOVER_RATE)
    parser.add_argument("--random-gene-rate", type=float, default=DEFAULT_RANDOM_GENE_RATE)


def _add_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=None,
        help="Candidates per TensileLite config; defaults to 1 for proposal-driven exploration",
    )
    parser.add_argument("--shape-batch-size", type=int, default=DEFAULT_PROFILE.default_shape_batch_size)
    parser.add_argument(
        "--batch-workers",
        type=int,
        default=default_batch_workers(),
        help="Parallel TensileLite batches to run; defaults to available CPU cores",
    )
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--tensilelite-bin", default=DEFAULT_TENSILELITE_BIN)
    parser.add_argument(
        "--compile-threads",
        type=int,
        default=DEFAULT_COMPILE_THREADS,
        help="TensileLite CpuThreads per batch; defaults to 1",
    )
    parser.add_argument(
        "--runner-bin", default=None, help="Structured runner executable; defaults to the target profile"
    )
    _add_timeout_args(parser)
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after the first failed batch")


def _add_adaptive_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fixed-sampling", action="store_true", help="Disable default adaptive finalist top-ups")
    parser.add_argument("--adaptive-initial-samples", type=int, default=3)
    parser.add_argument("--adaptive-max-rounds", type=int, default=4)
    parser.add_argument("--adaptive-epsilon-pct", type=float, default=2.0)
    parser.add_argument("--adaptive-confidence", type=float, default=0.90)
    parser.add_argument("--adaptive-min-samples", type=int, default=20)
    parser.add_argument("--adaptive-max-samples", type=int, default=80)
    parser.add_argument("--adaptive-sample-step", type=int, default=10)
    parser.add_argument("--adaptive-max-k", type=int, default=8)
    parser.add_argument("--adaptive-min-effect-pct", type=float, default=0.5)


def _add_schedule_args(parser: argparse.ArgumentParser, *, repair: bool = False) -> None:
    _add_candidate_shape_args(parser)
    _add_cache_identity_args(parser)
    _add_protocol_args(parser)
    _add_proposal_args(parser, repair=repair)
    _add_execution_args(parser)
    _add_adaptive_args(parser)


def cmd_proposal_coverage(args: argparse.Namespace) -> int:
    profile = _profile(args)
    db = EvoTensileDB.connect(args.db)
    db.init()
    shapes = _parse_shapes(args, profile)
    protocol = _protocol(args, profile)
    candidates = _propose_candidates_for_shapes(
        db,
        args,
        problem_hash=profile.problem_type_hash,
        protocol_hash=profile.benchmark_protocol_hash(protocol),
        shapes=shapes,
        proposal_shape_id=args.proposal_shape_id,
    )
    coverage = candidate_coverage(candidates)
    payload = {
        **coverage,
        "profile": profile.name,
        "proposal": args.proposal,
        "num_random": args.num_random,
        "gomea_count": args.gomea_count,
        "de_count": args.de_count,
        "local_count": args.local_count,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_summarize_space(args: argparse.Namespace) -> int:
    profile = _profile(args)
    candidates = _candidates(args)
    print("EvoTensile search-space summary")
    print(f"  profile: {profile.name}")
    print(f"  problem_type_hash: {profile.problem_type_hash}")
    print(f"  benchmark_protocol_hash: {profile.benchmark_protocol_hash()}")
    print(f"  MatrixInstruction choices: {len(MATRIX_INSTRUCTIONS)}")
    for mi in MATRIX_INSTRUCTIONS:
        mt0, mt1 = macro_tile(mi)
        print(f"    {mi} -> MT{mt0}x{mt1}")
    print("  Domain sizes:")
    product = 1
    for name, values in DOMAINS.items():
        product *= len(values)
        print(f"    {name}: {len(values)}")
    print(f"  Raw listed product before cheap constraints: {product:,}")
    print(f"  Generated candidates: {len(candidates)} ({args.num_random} requested random, deduped)")
    print(f"  Profile shapes: {len(profile.shapes())}")
    return 0


def cmd_summarize_cache(args: argparse.Namespace) -> int:
    profile = _profile(args)
    protocol = _protocol(args, profile)
    db = EvoTensileDB.connect(args.db)
    db.init()
    summary = db.cache_summary(
        problem_type_hash=profile.problem_type_hash,
        benchmark_protocol_hash=profile.benchmark_protocol_hash(protocol),
    )
    print(f"db: {args.db}")
    print(f"profile: {profile.name}")
    print(f"problem_type_hash: {profile.problem_type_hash}")
    print(f"benchmark_protocol_hash: {profile.benchmark_protocol_hash(protocol)}")
    print("status counts:")
    if summary:
        for status, count in summary.items():
            print(f"  {status}: {count}")
    else:
        print("  <none>")
    return 0


def cmd_rank_evals(args: argparse.Namespace) -> int:
    profile = _profile(args)
    protocol = _protocol(args, profile)
    db = EvoTensileDB.connect(args.db)
    summaries = db.rank_evaluations(
        problem_type_hash=profile.problem_type_hash,
        benchmark_protocol_hash=profile.benchmark_protocol_hash(protocol),
        shape_id=args.shape_id,
        min_samples=args.min_samples,
        limit=args.limit,
    )
    print("shape_id,candidate_hash,samples,median_gflops,best_gflops,median_time_us,best_time_us")
    for summary in summaries:
        print(
            f"{summary.shape_id},{summary.candidate_hash},{summary.samples},"
            f"{summary.median_gflops if summary.median_gflops is not None else ''},"
            f"{summary.best_gflops if summary.best_gflops is not None else ''},"
            f"{summary.median_time_us if summary.median_time_us is not None else ''},"
            f"{summary.best_time_us if summary.best_time_us is not None else ''}"
        )
    return 0


def cmd_summarize_rejections(args: argparse.Namespace) -> int:
    summaries = summarize_rejection_logs(args.paths)
    counts = classification_counts(summaries)
    if args.json:
        print(
            json.dumps(
                {
                    "counts": counts,
                    "logs": [summary.to_dict() for summary in summaries],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print("classification,count")
    for classification, count in sorted(counts.items()):
        print(f"{classification},{count}")
    if args.verbose:
        print("path,classification,actual_solutions,total_solutions,solution_stage,messages")
        for summary in summaries:
            print(
                f"{summary.path},{summary.classification},"
                f"{summary.actual_solutions if summary.actual_solutions is not None else ''},"
                f"{summary.total_solutions if summary.total_solutions is not None else ''},"
                f"{summary.solution_stage or ''},"
                f"{' | '.join(summary.messages)}"
            )
    return 0


def cmd_schedule_batches(args: argparse.Namespace) -> int:
    context = _schedule_context(args)
    candidates = _propose_candidates_for_shapes(
        context.db,
        args,
        problem_hash=context.problem_hash,
        protocol_hash=context.protocol_hash,
        shapes=context.shapes,
        proposal_shape_id=args.proposal_shape_id,
    )
    result = _execute_schedule_from_args(args, context, shapes=context.shapes, candidates=candidates)
    print(f"db: {args.db}")
    print(f"output_dir: {args.output_dir}")
    print(f"profile: {context.profile.name}")
    print(f"problem_type_hash: {context.problem_hash}")
    print(f"benchmark_protocol_hash: {context.protocol_hash}")
    print(f"proposal: {args.proposal}")
    print(f"candidates: {len(candidates)}")
    print(f"candidate_batch_size: {args.candidate_batch_size}")
    print(f"shape_batch_size: {args.shape_batch_size}")
    print(f"batch_workers: {args.batch_workers}")
    if context.runner_bin:
        print(f"runner_bin: {context.runner_bin}")
    print(f"planned batches: {len(result.planned_batches)}")
    print(f"planned missing pairs: {result.missing_pairs}")
    print(f"planned nominal pairs: {result.nominal_pairs}")
    print(f"planned missing samples: {sum(batch.missing_samples for batch in result.planned_batches)}")
    print(f"planned nominal samples: {sum(batch.nominal_samples for batch in result.planned_batches)}")
    for batch in result.planned_batches:
        print(
            f"batch {batch.batch_index:04d}: candidates={len(batch.candidates)} "
            f"shapes={len(batch.shapes)} missing_pairs={batch.missing_pairs} "
            f"nominal_pairs={batch.nominal_pairs} samples_per_pair={batch.samples_per_pair} "
            f"requires_validation={batch.requires_validation} "
            f"missing_samples={batch.missing_samples} extra_pairs={batch.extra_pairs}"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_counts_total: dict[str, int] = {}
    executed_batches = []
    for executed in result.executed_batches:
        status_counts = executed.ingest.status_counts if executed.ingest is not None else {}
        for status, count in status_counts.items():
            status_counts_total[status] = status_counts_total.get(status, 0) + count
        executed_batches.append(
            {
                "batch_index": executed.planned.batch_index,
                "build_returncode": executed.build_returncode,
                "runner_returncode": executed.runner_returncode,
                "requires_validation": executed.planned.requires_validation,
                "yaml_path": str(executed.yaml_path),
                "manifest_path": str(executed.manifest_path),
                "output_dir": str(executed.output_dir),
                "ingest": {
                    "inserted": executed.ingest.inserted if executed.ingest is not None else 0,
                    "rejected": executed.ingest.rejected if executed.ingest is not None else 0,
                    "unmapped": executed.ingest.unmapped if executed.ingest is not None else 0,
                    "status_counts": status_counts,
                    "errors": executed.ingest.errors if executed.ingest is not None else [],
                },
            }
        )
    metadata = _schedule_metadata_common(args, context, result=result, candidates=candidates, shapes=context.shapes)
    metadata.update(
        {
            "batches": [
                {
                    "batch_index": batch.batch_index,
                    "candidates": len(batch.candidates),
                    "shapes": len(batch.shapes),
                    "missing_pairs": batch.missing_pairs,
                    "nominal_pairs": batch.nominal_pairs,
                    "samples_per_pair": batch.samples_per_pair,
                    "requires_validation": batch.requires_validation,
                    "missing_samples": batch.missing_samples,
                    "nominal_samples": batch.nominal_samples,
                    "extra_pairs": batch.extra_pairs,
                }
                for batch in result.planned_batches
            ],
            "executed_batches": executed_batches,
            "status_counts": status_counts_total,
        }
    )
    metadata_path = output_dir / "schedule_metadata.json"
    metadata["metadata_path"] = str(metadata_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"metadata: {metadata_path}")
    if args.dry_run:
        return 0
    print(f"executed batches: {len(result.executed_batches)}")
    for executed in result.executed_batches:
        ingest = executed.ingest
        inserted = ingest.inserted if ingest is not None else 0
        rejected = ingest.rejected if ingest is not None else 0
        unmapped = ingest.unmapped if ingest is not None else 0
        print(
            f"executed {executed.planned.batch_index:04d}: build={executed.build_returncode} "
            f"runner={executed.runner_returncode} inserted={inserted} rejected={rejected} unmapped={unmapped} "
            f"yaml={executed.yaml_path}"
        )
    return 0


def cmd_repair_outliers(args: argparse.Namespace) -> int:
    context = _schedule_context(args)
    eligible_shapes = context.shapes
    outliers = detect_underperforming_shapes(
        context.db,
        shapes=eligible_shapes,
        problem_type_hash=context.problem_hash,
        benchmark_protocol_hash=context.protocol_hash,
        min_samples=args.outlier_min_samples,
        neighbor_count=args.neighbor_count,
        envelope_quantile=args.envelope_quantile,
        threshold_pct=args.outlier_threshold_pct,
        max_shapes=args.max_outliers,
    )
    repair_shapes = [outlier.shape for outlier in outliers]
    repair_seeds = repair_seed_candidates(
        context.db,
        outliers=outliers,
        problem_type_hash=context.problem_hash,
        benchmark_protocol_hash=context.protocol_hash,
        min_samples=args.outlier_min_samples,
        neighbor_per_shape=args.neighbor_per_shape,
    )
    proposal_candidates = _propose_candidates_for_shapes(
        context.db,
        args,
        problem_hash=context.problem_hash,
        protocol_hash=context.protocol_hash,
        shapes=repair_shapes,
    )
    candidates = _dedupe_candidates([*repair_seeds, *proposal_candidates])
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]

    result = _execute_schedule_from_args(
        args,
        context,
        shapes=repair_shapes,
        candidates=candidates,
        dry_run=args.dry_run or not repair_shapes or not candidates,
    )

    print(f"db: {args.db}")
    print(f"output_dir: {args.output_dir}")
    print(f"profile: {context.profile.name}")
    print(f"problem_type_hash: {context.problem_hash}")
    print(f"benchmark_protocol_hash: {context.protocol_hash}")
    print(f"eligible shapes: {len(eligible_shapes)}")
    print(f"outliers: {len(outliers)}")
    print(f"repair seed candidates: {len(repair_seeds)}")
    print(f"total candidates: {len(candidates)}")
    print(f"planned batches: {len(result.planned_batches)}")
    print(f"planned missing pairs: {result.missing_pairs}")
    print(f"planned missing samples: {sum(batch.missing_samples for batch in result.planned_batches)}")
    for outlier in outliers:
        print(
            f"outlier {outlier.shape.id}: actual={outlier.median_gflops:.3f}gflops "
            f"neighbor_prediction={outlier.predicted_neighbor_gflops:.3f}gflops "
            f"gap={outlier.residual_pct:.2f}% winner={outlier.candidate_hash}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_counts: dict[str, int] = {}
    executed_batches = []
    for executed in result.executed_batches:
        batch_status_counts = executed.ingest.status_counts if executed.ingest is not None else {}
        for status, count in batch_status_counts.items():
            status_counts[status] = status_counts.get(status, 0) + count
        executed_batches.append(
            {
                "batch_index": executed.planned.batch_index,
                "build_returncode": executed.build_returncode,
                "runner_returncode": executed.runner_returncode,
                "requires_validation": executed.planned.requires_validation,
                "yaml_path": str(executed.yaml_path),
                "manifest_path": str(executed.manifest_path),
                "output_dir": str(executed.output_dir),
                "ingest": None
                if executed.ingest is None
                else {
                    "inserted": executed.ingest.inserted,
                    "rejected": executed.ingest.rejected,
                    "unmapped": executed.ingest.unmapped,
                    "status_counts": batch_status_counts,
                    "errors": executed.ingest.errors,
                },
            }
        )
    metadata_path = output_dir / "repair_metadata.json"
    metadata = _schedule_metadata_common(args, context, result=result, candidates=candidates, shapes=repair_shapes)
    metadata.update(
        {
            "eligible_shapes": len(eligible_shapes),
            "outlier_min_samples": args.outlier_min_samples,
            "neighbor_count": args.neighbor_count,
            "neighbor_per_shape": args.neighbor_per_shape,
            "envelope_quantile": args.envelope_quantile,
            "outlier_threshold_pct": args.outlier_threshold_pct,
            "max_outliers": args.max_outliers,
            "outliers": [
                {
                    "shape_id": outlier.shape.id,
                    "candidate_hash": outlier.candidate_hash,
                    "samples": outlier.samples,
                    "median_gflops": outlier.median_gflops,
                    "predicted_neighbor_gflops": outlier.predicted_neighbor_gflops,
                    "residual_pct": outlier.residual_pct,
                    "neighbor_shape_ids": list(outlier.neighbor_shape_ids),
                    "neighbor_candidate_hashes": list(outlier.neighbor_candidate_hashes),
                }
                for outlier in outliers
            ],
            "repair_seed_candidates": len(repair_seeds),
            "candidate_hashes": [candidate.hash for candidate in candidates],
            "executed_batches": executed_batches,
            "status_counts": status_counts,
            "metadata_path": str(metadata_path),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"metadata: {metadata_path}")
    print(f"executed batches: {len(result.executed_batches)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evotensile")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cmd = sub.add_parser("summarize-space", help="Print search-space and generated-candidate summary")
    _add_profile_arg(cmd)
    cmd.add_argument("--num-random", type=int, default=DEFAULT_NUM_RANDOM)
    cmd.add_argument("--seed", type=int, default=1)
    cmd.set_defaults(func=cmd_summarize_space)

    cmd = sub.add_parser("proposal-coverage", help="Summarize generated proposal coverage without executing")
    cmd.add_argument("--db", required=True)
    _add_candidate_shape_args(cmd)
    _add_cache_identity_args(cmd)
    _add_protocol_args(cmd)
    _add_proposal_args(cmd)
    cmd.set_defaults(func=cmd_proposal_coverage)

    cmd = sub.add_parser("summarize-rejections", help="Classify TensileLite rejection logs")
    cmd.add_argument("paths", nargs="+", help="Log files or run directories to scan")
    cmd.add_argument("--json", action="store_true", help="Emit structured JSON")
    cmd.add_argument("--verbose", action="store_true", help="Print per-log classifications")
    cmd.set_defaults(func=cmd_summarize_rejections)

    cmd = sub.add_parser("schedule-batches", help="Cache-aware batch scheduling, build, runner, and ingestion")
    cmd.add_argument("--db", required=True)
    cmd.add_argument("--output-dir", required=True)
    _add_schedule_args(cmd)
    cmd.set_defaults(func=cmd_schedule_batches)

    cmd = sub.add_parser("repair-outliers", help="Rerun locally underperforming shapes with neighbor-seeded configs")
    cmd.add_argument("--db", required=True)
    cmd.add_argument("--output-dir", required=True)
    _add_schedule_args(cmd, repair=True)
    cmd.add_argument("--outlier-min-samples", type=int, default=10)
    cmd.add_argument("--neighbor-count", type=int, default=8)
    cmd.add_argument("--neighbor-per-shape", type=int, default=4)
    cmd.add_argument("--envelope-quantile", type=float, default=0.75)
    cmd.add_argument("--outlier-threshold-pct", type=float, default=10.0)
    cmd.add_argument("--max-outliers", type=int, default=None)
    cmd.add_argument("--max-candidates", type=int, default=None)
    cmd.set_defaults(func=cmd_repair_outliers)

    cmd = sub.add_parser("summarize-cache", help="Summarize cached evaluation statuses")
    cmd.add_argument("--db", required=True)
    _add_cache_identity_args(cmd)
    _add_protocol_args(cmd)
    cmd.set_defaults(func=cmd_summarize_cache)

    cmd = sub.add_parser("rank-evals", help="Rank only validation-passed cached evaluations")
    cmd.add_argument("--db", required=True)
    _add_cache_identity_args(cmd)
    _add_protocol_args(cmd)
    cmd.add_argument("--shape-id", default=None)
    cmd.add_argument("--min-samples", type=int, default=1)
    cmd.add_argument("--limit", type=int, default=20)
    cmd.set_defaults(func=cmd_rank_evals)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
