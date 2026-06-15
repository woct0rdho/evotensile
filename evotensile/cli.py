import argparse
import sys
from pathlib import Path

from .candidate import Shape
from .database import EvoTensileDB
from .parser import find_result_csvs, parse_tensile_csv
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
    print(f"Wrote {out}")
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
        extra_args=args.extra_arg,
    )
    print(f"run_id: {result.run_id}")
    print(f"returncode: {result.returncode}")
    print(f"stdout: {result.stdout_path}")
    print(f"stderr: {result.stderr_path}")
    print("command:", " ".join(result.command))
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
        extra_args=args.extra_arg,
    )
    print("build run_id:", build_result.run_id)
    print("build returncode:", build_result.returncode)
    print("build command:", " ".join(build_result.command))
    if bench_result is None:
        print("benchmark skipped because build failed")
        return build_result.returncode
    print("benchmark run_id:", bench_result.run_id)
    print("benchmark returncode:", bench_result.returncode)
    print("benchmark command:", " ".join(bench_result.command))
    return bench_result.returncode


def cmd_parse_csv(args: argparse.Namespace) -> int:
    paths: list[Path] = []
    for item in args.paths:
        p = Path(item)
        if p.is_dir():
            paths.extend(find_result_csvs(p))
        else:
            paths.append(p)
    total = 0
    for path in paths:
        rows = parse_tensile_csv(path)
        total += len(rows)
        ok = sum(1 for r in rows if r.time_us is not None)
        print(f"{path}: rows={len(rows)} rows_with_time={ok}")
    print(f"total rows: {total}")
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
    s.set_defaults(func=cmd_pilot_yaml)

    s = sub.add_parser("init-db", help="Initialize an EvoTensile SQLite DB")
    s.add_argument("--db", required=True)
    s.set_defaults(func=cmd_init_db)

    s = sub.add_parser("register-pilot", help="Register pilot shapes and generated candidates in DB")
    s.add_argument("--db", required=True)
    s.add_argument("--num-random", type=int, default=32)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--limit-shapes", type=int, default=None)
    s.add_argument("--shapes", nargs="*")
    s.set_defaults(func=cmd_register_pilot)

    s = sub.add_parser("run-yaml", help="Run TensileLite on an existing YAML")
    s.add_argument("--yaml", required=True)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--tensile-bin", default=DEFAULT_TENSILE_BIN)
    s.add_argument("--db")
    s.add_argument("--use-cache", action="store_true")
    s.add_argument("--build-only", action="store_true")
    s.add_argument("--cpu-threads", type=int, default=None, help="Pass CpuThreads=N to Tensile")
    s.add_argument(
        "--global-parameter",
        action="append",
        default=[],
        help="Pass a Tensile --global-parameters KEY=VALUE item; repeatable",
    )
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
    s.add_argument(
        "--global-parameter",
        action="append",
        default=[],
        help="Pass a Tensile --global-parameters KEY=VALUE item; repeatable",
    )
    s.add_argument("--extra-arg", action="append", default=[])
    s.set_defaults(func=cmd_build_bench_yaml)

    s = sub.add_parser("parse-csv", help="Parse Tensile CSV files or directories")
    s.add_argument("paths", nargs="+")
    s.set_defaults(func=cmd_parse_csv)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
