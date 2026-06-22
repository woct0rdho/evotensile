import argparse
import json
import sys
from pathlib import Path

from .adaptive_retime import AdaptivePolicy
from .candidate import Shape
from .database import EvoTensileDB
from .profile import DEFAULT_PROFILE, PROFILES, TargetProfile, get_profile
from .protocol import BenchmarkProtocol
from .runner import DEFAULT_TENSILELITE_BIN
from .scheduler import (
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
    execute_schedule,
    propose_candidates,
)
from .search.random_search import initial_random_batch
from .search_space import DOMAINS, MATRIX_INSTRUCTIONS, known_seed_candidates, macro_tile
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


def cmd_summarize_space(args: argparse.Namespace) -> int:
    profile = _profile(args)
    candidates = _candidates(args)
    seeds = known_seed_candidates()
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
    print(f"  Deterministic seeds: {len(seeds)}")
    print(f"  Generated candidates: {len(candidates)} ({args.num_random} requested random + seeds, deduped)")
    print(f"  Profile shapes: {len(profile.shapes())}")
    return 0


def cmd_cache_summary(args: argparse.Namespace) -> int:
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


def cmd_schedule_batches(args: argparse.Namespace) -> int:
    profile = _profile(args)
    protocol = _protocol(args, profile)
    db = EvoTensileDB.connect(args.db)
    db.init()
    problem_hash = profile.problem_type_hash
    protocol_hash = profile.benchmark_protocol_hash(protocol)
    shapes = _parse_shapes(args, profile)
    candidates = propose_candidates(
        db,
        proposal=args.proposal,
        num_random=args.num_random,
        seed=args.seed,
        problem_type_hash=problem_hash,
        benchmark_protocol_hash=protocol_hash,
        shape_id=args.proposal_shape_id,
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
    runner_bin = args.runner_bin or profile.default_runner_bin
    adaptive_policy = None
    if args.adaptive_sampling:
        adaptive_policy = AdaptivePolicy(
            epsilon_pct=args.adaptive_epsilon_pct,
            confidence=args.adaptive_confidence,
            min_retime_samples=args.adaptive_min_samples,
            max_retime_samples=args.adaptive_max_samples,
            sample_step=args.adaptive_sample_step,
            max_k=args.adaptive_max_k,
            min_effect_pct=args.adaptive_min_effect_pct,
        )
    result = execute_schedule(
        db,
        shapes=shapes,
        candidates=candidates,
        output_root=args.output_dir,
        target_profile=profile,
        protocol=protocol,
        min_samples=args.min_samples,
        candidate_batch_size=args.candidate_batch_size,
        shape_batch_size=args.shape_batch_size,
        ignore_cache=args.ignore_cache,
        max_batches=args.max_batches,
        dry_run=args.dry_run,
        generate_only=args.generate_only,
        tensilelite_bin=args.tensilelite_bin,
        compile_threads=args.compile_threads,
        keep_going=args.keep_going,
        runner_bin=runner_bin,
        build_timeout_s=args.build_timeout,
        runner_timeout_s=args.runner_timeout,
        adaptive_policy=adaptive_policy,
        adaptive_initial_samples=args.adaptive_initial_samples,
        adaptive_max_rounds=args.adaptive_max_rounds,
    )
    print(f"db: {args.db}")
    print(f"output_dir: {args.output_dir}")
    print(f"profile: {profile.name}")
    print(f"problem_type_hash: {problem_hash}")
    print(f"benchmark_protocol_hash: {protocol_hash}")
    print(f"proposal: {args.proposal}")
    print(f"candidates: {len(candidates)}")
    print(f"candidate_batch_size: {args.candidate_batch_size}")
    print(f"shape_batch_size: {args.shape_batch_size}")
    if runner_bin:
        print(f"runner_bin: {runner_bin}")
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
    metadata = {
        "db": args.db,
        "output_dir": args.output_dir,
        "profile": profile.name,
        "problem_type_hash": problem_hash,
        "benchmark_protocol_hash": protocol_hash,
        "protocol": protocol.global_parameters(),
        "proposal": args.proposal,
        "candidates": len(candidates),
        "shapes": len(shapes),
        "candidate_batch_size": args.candidate_batch_size,
        "shape_batch_size": args.shape_batch_size,
        "min_samples": args.min_samples,
        "ignore_cache": args.ignore_cache,
        "dry_run": args.dry_run,
        "generate_only": args.generate_only,
        "runner_bin": str(runner_bin) if runner_bin else None,
        "build_timeout_s": args.build_timeout,
        "runner_timeout_s": args.runner_timeout,
        "adaptive_sampling": args.adaptive_sampling,
        "adaptive_initial_samples": args.adaptive_initial_samples,
        "adaptive_max_rounds": args.adaptive_max_rounds,
        "adaptive_rounds": result.adaptive_rounds,
        "adaptive_policy": None if adaptive_policy is None else adaptive_policy.__dict__,
        "planned_batches": len(result.planned_batches),
        "planned_missing_pairs": result.missing_pairs,
        "planned_nominal_pairs": result.nominal_pairs,
        "planned_missing_samples": sum(batch.missing_samples for batch in result.planned_batches),
        "planned_nominal_samples": sum(batch.nominal_samples for batch in result.planned_batches),
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
        "executed_batches": [],
        "status_counts": {},
    }
    for executed in result.executed_batches:
        status_counts = executed.ingest.status_counts if executed.ingest is not None else {}
        for status, count in status_counts.items():
            metadata["status_counts"][status] = metadata["status_counts"].get(status, 0) + count
        metadata["executed_batches"].append(
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evotensile")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cmd = sub.add_parser("summarize-space", help="Print search-space and generated-candidate summary")
    _add_profile_arg(cmd)
    cmd.add_argument("--num-random", type=int, default=DEFAULT_NUM_RANDOM)
    cmd.add_argument("--seed", type=int, default=1)
    cmd.set_defaults(func=cmd_summarize_space)

    cmd = sub.add_parser("schedule-batches", help="Cache-aware batch scheduling, build, runner, and ingestion")
    cmd.add_argument("--db", required=True)
    cmd.add_argument("--output-dir", required=True)
    _add_candidate_shape_args(cmd)
    _add_cache_identity_args(cmd)
    _add_protocol_args(cmd)
    cmd.add_argument("--proposal", choices=PROPOSAL_MODES, default=DEFAULT_PROPOSAL)
    cmd.add_argument("--proposal-shape-id", default=None, help="Limit cached elite selection to one shape id")
    cmd.add_argument(
        "--transfer-shapes",
        type=int,
        default=DEFAULT_TRANSFER_SHAPES,
        help="Seed from winners of this many nearest already-tuned shapes; 0 disables transfer",
    )
    cmd.add_argument(
        "--transfer-per-shape",
        type=int,
        default=DEFAULT_TRANSFER_PER_SHAPE,
        help="Seed this many top candidates from each nearest shape",
    )
    cmd.add_argument("--elite-count", type=int, default=DEFAULT_ELITE_COUNT)
    cmd.add_argument("--local-count", type=int, default=DEFAULT_LOCAL_COUNT)
    cmd.add_argument("--de-count", type=int, default=DEFAULT_DE_COUNT)
    cmd.add_argument("--gomea-count", type=int, default=DEFAULT_GOMEA_COUNT)
    cmd.add_argument("--mutation-rate", type=float, default=DEFAULT_MUTATION_RATE)
    cmd.add_argument("--crossover-rate", type=float, default=DEFAULT_CROSSOVER_RATE)
    cmd.add_argument("--random-gene-rate", type=float, default=DEFAULT_RANDOM_GENE_RATE)
    cmd.add_argument("--candidate-batch-size", type=int, default=DEFAULT_PROFILE.default_candidate_batch_size)
    cmd.add_argument("--shape-batch-size", type=int, default=DEFAULT_PROFILE.default_shape_batch_size)
    cmd.add_argument("--min-samples", type=int, default=1)
    cmd.add_argument("--ignore-cache", action="store_true")
    cmd.add_argument("--max-batches", type=int, default=None)
    cmd.add_argument("--dry-run", action="store_true")
    cmd.add_argument("--generate-only", action="store_true")
    cmd.add_argument("--tensilelite-bin", default=DEFAULT_TENSILELITE_BIN)
    cmd.add_argument("--compile-threads", type=int, default=-1)
    cmd.add_argument("--runner-bin", default=None, help="Structured runner executable; defaults to the target profile")
    cmd.add_argument("--build-timeout", type=float, default=None, help="TensileLite build timeout in seconds")
    cmd.add_argument("--runner-timeout", type=float, default=None, help="Structured runner timeout in seconds")
    cmd.add_argument("--keep-going", action="store_true")
    cmd.add_argument("--adaptive-sampling", action="store_true")
    cmd.add_argument("--adaptive-initial-samples", type=int, default=3)
    cmd.add_argument("--adaptive-max-rounds", type=int, default=4)
    cmd.add_argument("--adaptive-epsilon-pct", type=float, default=2.0)
    cmd.add_argument("--adaptive-confidence", type=float, default=0.90)
    cmd.add_argument("--adaptive-min-samples", type=int, default=20)
    cmd.add_argument("--adaptive-max-samples", type=int, default=80)
    cmd.add_argument("--adaptive-sample-step", type=int, default=10)
    cmd.add_argument("--adaptive-max-k", type=int, default=8)
    cmd.add_argument("--adaptive-min-effect-pct", type=float, default=0.5)
    cmd.set_defaults(func=cmd_schedule_batches)

    cmd = sub.add_parser("cache-summary", help="Summarize cached evaluation statuses")
    cmd.add_argument("--db", required=True)
    _add_cache_identity_args(cmd)
    _add_protocol_args(cmd)
    cmd.set_defaults(func=cmd_cache_summary)

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
