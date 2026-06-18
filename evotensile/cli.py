import argparse
import sys

from .cache import normalize_version_name, problem_type_hash
from .candidate import Shape
from .database import EvoTensileDB
from .ingest import csv_paths, ingest_results, print_ingest_result
from .parser import evaluation_status, parse_tensilelite_csv
from .runner import DEFAULT_TENSILELITE_BIN, serial_benchmark_protocol_hash
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
    PROPOSAL_MODES,
    execute_schedule,
    propose_candidates,
)
from .search.random_search import initial_random_batch
from .search_space import DOMAINS, MATRIX_INSTRUCTIONS, known_seed_candidates, macro_tile
from .shapes import parse_shape, pilot_100_shapes


def _parse_shapes(args: argparse.Namespace) -> list[Shape]:
    if getattr(args, "shapes", None):
        return [parse_shape(s) for s in args.shapes]
    if getattr(args, "limit_shapes", None):
        return pilot_100_shapes()[: args.limit_shapes]
    return pilot_100_shapes()


def _candidates(args: argparse.Namespace):
    return initial_random_batch(args.num_random, seed=args.seed)


def _problem_hash_arg(args: argparse.Namespace) -> str:
    return getattr(args, "problem_type_hash", None) or problem_type_hash()


def _protocol_hash_arg(args: argparse.Namespace) -> str:
    # EvoTensile benchmark ingestion assumes serial GPU execution, matching
    # schedule-batches' forced ParallelGpuExecution=1 benchmark command.
    return serial_benchmark_protocol_hash(
        getattr(args, "global_parameter", None),
        benchmark_protocol_hash=getattr(args, "benchmark_protocol_hash", None),
    )


def _add_candidate_shape_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-random", type=int, default=DEFAULT_NUM_RANDOM)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit-shapes", type=int, default=None)
    parser.add_argument("--shapes", nargs="*")


def _add_cache_identity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--version-name",
        default="unversioned",
        help="Manual timing-cache namespace; use this to control cache refreshes",
    )
    parser.add_argument("--problem-type-hash", default=None)
    parser.add_argument("--benchmark-protocol-hash", default=None)
    parser.add_argument(
        "--global-parameter",
        action="append",
        default=[],
        help="Pass/consider a TensileLite --global-parameters KEY=VALUE item; repeatable",
    )


def cmd_summarize_space(args: argparse.Namespace) -> int:
    candidates = _candidates(args)
    seeds = known_seed_candidates()
    print("EvoTensile search-space summary")
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
    print(f"  Pilot shapes: {len(pilot_100_shapes())}")
    return 0


def cmd_cache_summary(args: argparse.Namespace) -> int:
    db = EvoTensileDB.connect(args.db)
    db.init()
    summary = db.cache_summary(
        version_name=args.version_name,
        problem_type_hash=args.problem_type_hash,
        benchmark_protocol_hash=args.benchmark_protocol_hash,
    )
    print(f"db: {args.db}")
    if args.version_name:
        print(f"version_name: {normalize_version_name(args.version_name)}")
    if args.problem_type_hash:
        print(f"problem_type_hash: {args.problem_type_hash}")
    if args.benchmark_protocol_hash:
        print(f"benchmark_protocol_hash: {args.benchmark_protocol_hash}")
    print("status counts:")
    if summary:
        for status, count in summary.items():
            print(f"  {status}: {count}")
    else:
        print("  <none>")
    print("known versions:", ", ".join(db.distinct_versions()) or "<none>")
    return 0


def cmd_rank_evals(args: argparse.Namespace) -> int:
    db = EvoTensileDB.connect(args.db)
    summaries = db.rank_evaluations(
        version_name=args.version_name,
        problem_type_hash=args.problem_type_hash,
        benchmark_protocol_hash=args.benchmark_protocol_hash,
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


def cmd_parse_csv(args: argparse.Namespace) -> int:
    paths = csv_paths(args.paths, include_logs=args.include_logs)
    total = 0
    status_counts: dict[str, int] = {}
    for path in paths:
        rows = parse_tensilelite_csv(path)
        total += len(rows)
        ok = sum(1 for r in rows if evaluation_status(r, require_validation=not args.allow_unknown_validation) == "ok")
        for row in rows:
            status = evaluation_status(row, require_validation=not args.allow_unknown_validation)
            status_counts[status] = status_counts.get(status, 0) + 1
        print(f"{path}: rows={len(rows)} validation_ok={ok}")
    print(f"total rows: {total}")
    print("status counts:")
    for status in sorted(status_counts):
        print(f"  {status}: {status_counts[status]}")
    return 0


def cmd_ingest_csv(args: argparse.Namespace) -> int:
    db = EvoTensileDB.connect(args.db)
    db.init()
    version = normalize_version_name(args.version_name)
    problem_hash = _problem_hash_arg(args)
    protocol_hash = _protocol_hash_arg(args)
    result = ingest_results(
        db=db,
        paths=args.paths,
        manifest_path=args.manifest,
        version_name=version,
        problem_type_hash=problem_hash,
        benchmark_protocol_hash=protocol_hash,
        run_id=args.run_id,
        include_logs=args.include_logs,
        solutions_yaml=args.solutions_yaml,
        allow_manifest_order_fallback=args.allow_manifest_order_fallback,
        allow_unknown_validation=args.allow_unknown_validation,
    )
    print_ingest_result(result, db_path=args.db, manifest_path=args.manifest)
    print(f"version_name: {version}")
    print(f"problem_type_hash: {problem_hash}")
    print(f"benchmark_protocol_hash: {protocol_hash}")
    return 0 if result.ok else 2


def cmd_schedule_batches(args: argparse.Namespace) -> int:
    db = EvoTensileDB.connect(args.db)
    db.init()
    version = normalize_version_name(args.version_name)
    problem_hash = _problem_hash_arg(args)
    protocol_hash = _protocol_hash_arg(args)
    candidates = propose_candidates(
        db,
        proposal=args.proposal,
        num_random=args.num_random,
        seed=args.seed,
        version_name=version,
        problem_type_hash=problem_hash,
        benchmark_protocol_hash=protocol_hash,
        shape_id=args.proposal_shape_id,
        elite_count=args.elite_count,
        local_count=args.local_count,
        de_count=args.de_count,
        gomea_count=args.gomea_count,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        random_gene_rate=args.random_gene_rate,
    )
    shapes = _parse_shapes(args)
    result = execute_schedule(
        db,
        shapes=shapes,
        candidates=candidates,
        output_root=args.output_dir,
        version_name=version,
        problem_type_hash=problem_hash,
        benchmark_protocol_hash=protocol_hash,
        min_samples=args.min_samples,
        candidate_batch_size=args.candidate_batch_size,
        shape_batch_size=args.shape_batch_size,
        ignore_cache=args.ignore_cache,
        max_batches=args.max_batches,
        dry_run=args.dry_run,
        generate_only=args.generate_only,
        tensilelite_bin=args.tensilelite_bin,
        compile_threads=args.compile_threads,
        benchmark_threads=args.benchmark_threads,
        global_parameters=args.global_parameter,
        extra_args=args.extra_arg,
        keep_going=args.keep_going,
    )
    print(f"db: {args.db}")
    print(f"output_dir: {args.output_dir}")
    print(f"version_name: {version}")
    print(f"problem_type_hash: {problem_hash}")
    print(f"benchmark_protocol_hash: {protocol_hash}")
    print(f"proposal: {args.proposal}")
    print(f"candidates: {len(candidates)}")
    print(f"candidate_batch_size: {args.candidate_batch_size}")
    print(f"shape_batch_size: {args.shape_batch_size}")
    print(f"planned batches: {len(result.planned_batches)}")
    print(f"planned missing evaluations: {result.missing_pairs}")
    print(f"planned nominal evaluations: {result.nominal_pairs}")
    for batch in result.planned_batches:
        print(
            f"batch {batch.batch_index:04d}: candidates={len(batch.candidates)} "
            f"shapes={len(batch.shapes)} missing={batch.missing_pairs} nominal={batch.nominal_pairs} "
            f"extra={batch.extra_pairs}"
        )
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
            f"bench={executed.benchmark_returncode} inserted={inserted} rejected={rejected} unmapped={unmapped} "
            f"yaml={executed.yaml_path}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evotensile")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summarize-space", help="Print search-space and generated-candidate summary")
    s.add_argument("--num-random", type=int, default=DEFAULT_NUM_RANDOM)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(func=cmd_summarize_space)

    s = sub.add_parser("schedule-batches", help="Cache-aware batch scheduling, build/bench, and ingestion")
    s.add_argument("--db", required=True)
    s.add_argument("--output-dir", required=True)
    _add_candidate_shape_args(s)
    _add_cache_identity_args(s)
    s.add_argument("--proposal", choices=PROPOSAL_MODES, default=DEFAULT_PROPOSAL)
    s.add_argument("--proposal-shape-id", default=None, help="Limit cached elite selection to one shape id")
    s.add_argument("--elite-count", type=int, default=DEFAULT_ELITE_COUNT)
    s.add_argument("--local-count", type=int, default=DEFAULT_LOCAL_COUNT)
    s.add_argument("--de-count", type=int, default=DEFAULT_DE_COUNT)
    s.add_argument("--gomea-count", type=int, default=DEFAULT_GOMEA_COUNT)
    s.add_argument("--mutation-rate", type=float, default=DEFAULT_MUTATION_RATE)
    s.add_argument("--crossover-rate", type=float, default=DEFAULT_CROSSOVER_RATE)
    s.add_argument("--random-gene-rate", type=float, default=DEFAULT_RANDOM_GENE_RATE)
    s.add_argument("--candidate-batch-size", type=int, default=32)
    s.add_argument("--shape-batch-size", type=int, default=100)
    s.add_argument("--min-samples", type=int, default=1)
    s.add_argument("--ignore-cache", action="store_true")
    s.add_argument("--max-batches", type=int, default=None)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--generate-only", action="store_true")
    s.add_argument("--tensilelite-bin", default=DEFAULT_TENSILELITE_BIN)
    s.add_argument("--compile-threads", type=int, default=-1)
    s.add_argument("--benchmark-threads", type=int, default=1)
    s.add_argument("--extra-arg", action="append", default=[])
    s.add_argument("--keep-going", action="store_true")
    s.set_defaults(func=cmd_schedule_batches)

    s = sub.add_parser("cache-summary", help="Summarize cached evaluation statuses")
    s.add_argument("--db", required=True)
    s.add_argument("--version-name", default=None)
    s.add_argument("--problem-type-hash", default=None)
    s.add_argument("--benchmark-protocol-hash", default=None)
    s.set_defaults(func=cmd_cache_summary)

    s = sub.add_parser("rank-evals", help="Rank only validation-passed cached evaluations")
    s.add_argument("--db", required=True)
    s.add_argument("--version-name", default=None)
    s.add_argument("--problem-type-hash", default=None)
    s.add_argument("--benchmark-protocol-hash", default=None)
    s.add_argument("--shape-id", default=None)
    s.add_argument("--min-samples", type=int, default=1)
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=cmd_rank_evals)

    s = sub.add_parser("parse-csv", help="Parse TensileLite CSV files, logs, or directories")
    s.add_argument("paths", nargs="+")
    s.add_argument("--include-logs", action="store_true")
    s.add_argument("--allow-unknown-validation", action="store_true")
    s.set_defaults(func=cmd_parse_csv)

    s = sub.add_parser("ingest-csv", help="Ingest validation-gated TensileLite CSV/log rows into SQLite")
    s.add_argument("paths", nargs="+")
    s.add_argument("--db", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--run-id")
    s.add_argument("--include-logs", action="store_true")
    s.add_argument(
        "--solutions-yaml",
        action="append",
        default=[],
        help="TensileLite *_Final.yaml/_CSVWinner.yaml; auto-detected for directories",
    )
    s.add_argument(
        "--allow-manifest-order-fallback",
        action="store_true",
        help="Debug-only fallback when final solution YAML is unavailable",
    )
    s.add_argument("--allow-unknown-validation", action="store_true")
    _add_cache_identity_args(s)
    s.set_defaults(func=cmd_ingest_csv)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
