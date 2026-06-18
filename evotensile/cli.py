import argparse
import json
import sys
from pathlib import Path

from .cache import (
    benchmark_protocol_hash_from_items,
    cache_keys,
    normalize_version_name,
    problem_type_hash,
)
from .candidate import Shape
from .database import EvoTensileDB
from .manifest import manifest_by_problem_solution, manifest_by_shape_solution, read_manifest, write_manifest
from .parser import evaluation_status, find_result_csvs, parse_tensile_csv
from .runner import DEFAULT_TENSILE_BIN, build_then_benchmark, run_tensile
from .search_space import (
    DOMAINS,
    MATRIX_INSTRUCTIONS,
    known_seed_candidates,
    macro_tile,
    seed_and_random_candidates,
)
from .shapes import parse_shape, pilot_100_shapes
from .yaml_writer import write_tensile_yaml


def _parse_shapes(args: argparse.Namespace) -> list[Shape]:
    if getattr(args, "shapes", None):
        return [parse_shape(s) for s in args.shapes]
    if getattr(args, "limit_shapes", None):
        return pilot_100_shapes()[: args.limit_shapes]
    return pilot_100_shapes()


def _candidates(args: argparse.Namespace):
    return seed_and_random_candidates(args.num_random, seed=args.seed)


def _problem_hash_arg(args: argparse.Namespace) -> str:
    return getattr(args, "problem_type_hash", None) or problem_type_hash()


def _protocol_hash_arg(args: argparse.Namespace) -> str:
    return getattr(args, "benchmark_protocol_hash", None) or benchmark_protocol_hash_from_items(
        getattr(args, "global_parameter", None)
    )


def _print_cache_identity(args: argparse.Namespace) -> None:
    print(f"version_name: {normalize_version_name(getattr(args, 'version_name', None))}")
    print(f"problem_type_hash: {_problem_hash_arg(args)}")
    print(f"benchmark_protocol_hash: {_protocol_hash_arg(args)}")


def _add_candidate_shape_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-random", type=int, default=32)
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
        help="Pass/consider a Tensile --global-parameters KEY=VALUE item; repeatable",
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


def cmd_pilot_yaml(args: argparse.Namespace) -> int:
    candidates = _candidates(args)
    shapes = _parse_shapes(args)
    out = write_tensile_yaml(args.output_yaml, candidates, shapes)
    manifest_path = Path(args.manifest) if args.manifest else Path(args.output_yaml).with_suffix(".manifest.csv")
    write_manifest(manifest_path, candidates, shapes)
    print(f"Wrote {out}")
    print(f"Wrote {manifest_path}")
    print(f"  candidates: {len(candidates)}")
    print(f"  shapes: {len(shapes)}")
    print(f"  nominal evaluations: {len(candidates) * len(shapes):,}")
    return 0


def cmd_init_db(args: argparse.Namespace) -> int:
    db = EvoTensileDB.connect(args.db)
    db.init()
    print(f"Initialized {args.db}")
    print(db.counts())
    return 0


def cmd_register_pilot(args: argparse.Namespace) -> int:
    db = EvoTensileDB.connect(args.db)
    db.init()
    candidates = _candidates(args)
    shapes = _parse_shapes(args)
    db.register_candidates(candidates)
    db.register_shapes(shapes)
    print(f"Registered into {args.db}")
    print(f"  candidates: {len(candidates)}")
    print(f"  shapes: {len(shapes)}")
    print(db.counts())
    return 0


def cmd_run_yaml(args: argparse.Namespace) -> int:
    db = EvoTensileDB.connect(args.db) if args.db else None
    if db is not None:
        db.init()
    result = run_tensile(
        args.yaml,
        args.output_dir,
        tensile_bin=args.tensile_bin,
        db=db,
        use_cache=args.use_cache,
        build_only=args.build_only,
        cpu_threads=args.cpu_threads,
        global_parameters=args.global_parameter,
        version_name=args.version_name,
        problem_type_hash=_problem_hash_arg(args),
        benchmark_protocol_hash=_protocol_hash_arg(args),
        extra_args=args.extra_arg,
    )
    print(f"run_id: {result.run_id}")
    print(f"returncode: {result.returncode}")
    print(f"stdout: {result.stdout_path}")
    print(f"stderr: {result.stderr_path}")
    print("command:", " ".join(result.command))
    print(f"version_name: {result.version_name}")
    print(f"problem_type_hash: {result.problem_type_hash}")
    print(f"benchmark_protocol_hash: {result.benchmark_protocol_hash}")
    return result.returncode


def cmd_build_bench_yaml(args: argparse.Namespace) -> int:
    db = EvoTensileDB.connect(args.db) if args.db else None
    if db is not None:
        db.init()
    build_result, bench_result = build_then_benchmark(
        args.yaml,
        args.output_dir,
        tensile_bin=args.tensile_bin,
        db=db,
        compile_threads=args.compile_threads,
        benchmark_threads=args.benchmark_threads,
        global_parameters=args.global_parameter,
        version_name=args.version_name,
        problem_type_hash=_problem_hash_arg(args),
        benchmark_protocol_hash=_protocol_hash_arg(args),
        extra_args=args.extra_arg,
    )
    print("build run_id:", build_result.run_id)
    print("build returncode:", build_result.returncode)
    print("build command:", " ".join(build_result.command))
    print(f"version_name: {build_result.version_name}")
    print(f"problem_type_hash: {build_result.problem_type_hash}")
    print(f"benchmark_protocol_hash: {build_result.benchmark_protocol_hash}")
    if bench_result is None:
        print("benchmark skipped because build failed")
        return build_result.returncode
    print("benchmark run_id:", bench_result.run_id)
    print("benchmark returncode:", bench_result.returncode)
    print("benchmark command:", " ".join(bench_result.command))
    return bench_result.returncode


def cmd_cache_key(args: argparse.Namespace) -> int:
    _print_cache_identity(args)
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


def cmd_cache_missing(args: argparse.Namespace) -> int:
    db = EvoTensileDB.connect(args.db)
    db.init()
    candidates = _candidates(args)
    shapes = _parse_shapes(args)
    problem_hash = _problem_hash_arg(args)
    protocol_hash = _protocol_hash_arg(args)
    keys = cache_keys(
        shapes,
        candidates,
        version_name=args.version_name,
        problem_hash=problem_hash,
        protocol_hash=protocol_hash,
    )
    missing = [key for key in keys if not db.has_cached_evaluation(key, min_samples=args.min_samples)]
    print(f"version_name: {normalize_version_name(args.version_name)}")
    print(f"problem_type_hash: {problem_hash}")
    print(f"benchmark_protocol_hash: {protocol_hash}")
    print(f"shapes: {len(shapes)}")
    print(f"candidates: {len(candidates)}")
    print(f"total evaluations: {len(keys)}")
    print(f"cached evaluations: {len(keys) - len(missing)}")
    print(f"missing evaluations: {len(missing)}")
    if args.print_missing:
        for key in missing:
            print(f"{key.shape_id} {key.candidate_hash}")
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


def _csv_paths(items: list[str], *, include_logs: bool = False) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            paths.extend(find_result_csvs(p, include_logs=include_logs))
        else:
            paths.append(p)
    return sorted(set(paths))


def cmd_parse_csv(args: argparse.Namespace) -> int:
    paths = _csv_paths(args.paths, include_logs=args.include_logs)
    total = 0
    status_counts: dict[str, int] = {}
    for path in paths:
        rows = parse_tensile_csv(path)
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
    manifest_entries = read_manifest(args.manifest)
    by_problem_solution = manifest_by_problem_solution(manifest_entries)
    by_shape_solution = manifest_by_shape_solution(manifest_entries)
    version = normalize_version_name(args.version_name)
    problem_hash = _problem_hash_arg(args)
    protocol_hash = _protocol_hash_arg(args)
    paths = _csv_paths(args.paths, include_logs=args.include_logs)
    inserted = 0
    unmapped = 0
    status_counts: dict[str, int] = {}
    for path in paths:
        for row in parse_tensile_csv(path):
            entry = None
            if row.problem_index is not None and row.solution_index is not None:
                entry = by_problem_solution.get((row.problem_index, row.solution_index))
            if entry is None and row.shape_id is not None and row.solution_index is not None:
                entry = by_shape_solution.get((row.shape_id, row.solution_index))
            if entry is None:
                unmapped += 1
                continue
            status = evaluation_status(row, require_validation=not args.allow_unknown_validation)
            status_counts[status] = status_counts.get(status, 0) + 1
            db.insert_evaluation(
                shape_id=entry.shape_id,
                candidate_hash=entry.candidate_hash,
                run_id=args.run_id,
                status=status,
                version_name=version,
                problem_type_hash=problem_hash,
                benchmark_protocol_hash=protocol_hash,
                time_us=row.time_us,
                gflops=row.gflops,
                validation=row.validation,
                solution_index=row.solution_index,
                raw_csv_row=json.dumps(row.raw, sort_keys=True),
            )
            inserted += 1
    print(f"db: {args.db}")
    print(f"manifest: {args.manifest}")
    print(f"version_name: {version}")
    print(f"problem_type_hash: {problem_hash}")
    print(f"benchmark_protocol_hash: {protocol_hash}")
    print(f"inserted evaluations: {inserted}")
    print(f"unmapped rows: {unmapped}")
    print("status counts:")
    for status in sorted(status_counts):
        print(f"  {status}: {status_counts[status]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evotensile")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summarize-space", help="Print search-space and generated-candidate summary")
    s.add_argument("--num-random", type=int, default=32)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(func=cmd_summarize_space)

    s = sub.add_parser("pilot-yaml", help="Generate a TensileLite YAML for pilot shapes")
    s.add_argument("--output-yaml", required=True)
    s.add_argument("--num-random", type=int, default=32)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--limit-shapes", type=int, default=None, help="Use first N pilot shapes")
    s.add_argument("--shapes", nargs="*", help="Explicit shapes as M,N,batch,K or MxNxBxK")
    s.add_argument("--manifest", default=None, help="Candidate/shape manifest path; defaults to OUTPUT.manifest.csv")
    s.set_defaults(func=cmd_pilot_yaml)

    s = sub.add_parser("init-db", help="Initialize an EvoTensile SQLite DB")
    s.add_argument("--db", required=True)
    s.set_defaults(func=cmd_init_db)

    s = sub.add_parser("register-pilot", help="Register pilot shapes and generated candidates in DB")
    s.add_argument("--db", required=True)
    _add_candidate_shape_args(s)
    s.set_defaults(func=cmd_register_pilot)

    s = sub.add_parser("run-yaml", help="Run TensileLite on an existing YAML")
    s.add_argument("--yaml", required=True)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--tensile-bin", default=DEFAULT_TENSILE_BIN)
    s.add_argument("--db")
    s.add_argument("--use-cache", action="store_true")
    s.add_argument("--build-only", action="store_true")
    s.add_argument("--cpu-threads", type=int, default=None, help="Pass CpuThreads=N to Tensile")
    _add_cache_identity_args(s)
    s.add_argument("--extra-arg", action="append", default=[])
    s.set_defaults(func=cmd_run_yaml)

    s = sub.add_parser(
        "build-bench-yaml",
        help="Compile with --build-only, then benchmark serially with --use-cache",
    )
    s.add_argument("--yaml", required=True)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--tensile-bin", default=DEFAULT_TENSILE_BIN)
    s.add_argument("--db")
    s.add_argument("--compile-threads", type=int, default=-1)
    s.add_argument("--benchmark-threads", type=int, default=1)
    _add_cache_identity_args(s)
    s.add_argument("--extra-arg", action="append", default=[])
    s.set_defaults(func=cmd_build_bench_yaml)

    s = sub.add_parser("cache-key", help="Print the current timing-cache identity")
    _add_cache_identity_args(s)
    s.set_defaults(func=cmd_cache_key)

    s = sub.add_parser("cache-summary", help="Summarize cached evaluation statuses")
    s.add_argument("--db", required=True)
    s.add_argument("--version-name", default=None)
    s.add_argument("--problem-type-hash", default=None)
    s.add_argument("--benchmark-protocol-hash", default=None)
    s.set_defaults(func=cmd_cache_summary)

    s = sub.add_parser("cache-missing", help="Count generated candidate/shape evaluations missing in cache")
    s.add_argument("--db", required=True)
    _add_candidate_shape_args(s)
    _add_cache_identity_args(s)
    s.add_argument("--min-samples", type=int, default=1)
    s.add_argument("--print-missing", action="store_true")
    s.set_defaults(func=cmd_cache_missing)

    s = sub.add_parser("rank-evals", help="Rank only validation-passed cached evaluations")
    s.add_argument("--db", required=True)
    s.add_argument("--version-name", default=None)
    s.add_argument("--problem-type-hash", default=None)
    s.add_argument("--benchmark-protocol-hash", default=None)
    s.add_argument("--shape-id", default=None)
    s.add_argument("--min-samples", type=int, default=1)
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=cmd_rank_evals)

    s = sub.add_parser("parse-csv", help="Parse Tensile CSV files, logs, or directories")
    s.add_argument("paths", nargs="+")
    s.add_argument("--include-logs", action="store_true")
    s.add_argument("--allow-unknown-validation", action="store_true")
    s.set_defaults(func=cmd_parse_csv)

    s = sub.add_parser("ingest-csv", help="Ingest validation-gated Tensile CSV/log rows into SQLite")
    s.add_argument("paths", nargs="+")
    s.add_argument("--db", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--run-id")
    s.add_argument("--include-logs", action="store_true")
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
