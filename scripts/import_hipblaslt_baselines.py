#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from evotensile.activity import apu_activity_lock
from evotensile.candidate import Candidate
from evotensile.database import EvoTensileDB
from evotensile.profile import PROFILES, get_profile
from evotensile.protocol import BenchmarkProtocol
from evotensile.runner import DEFAULT_TENSILELITE_BIN
from evotensile.scheduler import DEFAULT_COMPILE_THREADS, execute_schedule
from evotensile.shapes import Shape, parse_shape
from evotensile.tensilelite_keys import DIRECT_SOLUTION_MATCH_KEYS

DEFAULT_BENCH = Path.home() / "rocm-libraries/build/hipblaslt-bench/clients/hipblaslt-bench"
DEFAULT_ROCM_DEVEL = Path.home() / "venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel"
DEFAULT_ROCM_LIBRARIES = Path.home() / "venv_torch/lib/python3.14/site-packages/_rocm_sdk_libraries"
DEFAULT_LOGIC_YAML = (
    Path.home()
    / "rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/GridBased/gfx1151_Cijk_Ailk_Bjlk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml"
)

CSV_HEADER_MARKER = "hipblaslt-Gflops"
SOLUTION_INDEX_RE = re.compile(r"--Solution index:\s*(?P<value>-?\d+)")
SOLUTION_NAME_RE = re.compile(r"--Solution name:\s*(?P<value>\S.*)")
KERNEL_NAME_RE = re.compile(r"--kernel name:\s*(?P<value>\S.*)")
DEVICE_RE = re.compile(r"Device ID \d+ : (?P<name>.*?) (?P<arch>gfx\d+) ")

CANDIDATE_PARAM_KEYS = frozenset(
    {
        "1LDSBuffer",
        "ExpandPointerSwap",
        "PrefetchLocalRead",
        "StoreVectorWidth",
        "WorkGroup",
        *DIRECT_SOLUTION_MATCH_KEYS,
    }
)


@dataclass(frozen=True)
class BaselineSelection:
    shape: Shape
    candidate: Candidate
    solution_index: int
    solution_name: str | None
    kernel_name: str | None
    logic_solution_index: int | None
    logic_solution_name: str | None
    hipblaslt_gflops: float
    hipblaslt_time_us: float
    stdout_path: Path
    stderr_path: Path
    command: list[str]


def _profile_protocol(args: argparse.Namespace) -> tuple[Any, BenchmarkProtocol]:
    profile = get_profile(args.profile)
    protocol = profile.default_protocol.with_overrides(
        num_warmups=args.num_warmups,
        num_benchmarks=args.num_benchmarks,
        enqueues_per_sync=args.enqueues_per_sync,
        syncs_per_benchmark=args.syncs_per_benchmark,
        num_elements_to_validate=args.num_elements_to_validate,
    )
    return profile, protocol


def _parse_shapes(args: argparse.Namespace, profile) -> list[Shape]:
    if args.shapes:
        return [parse_shape(value) for value in args.shapes]
    shapes = profile.shapes()
    if args.limit_shapes is not None:
        return shapes[: args.limit_shapes]
    return shapes


def _csv_payload_lines(stdout: str) -> tuple[list[str], list[str]]:
    header: list[str] | None = None
    data: list[str] | None = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if CSV_HEADER_MARKER in stripped and "," in stripped:
            stripped = re.sub(r"^\[\d+\]:", "", stripped)
            header = [item.strip() for item in stripped.split(",")]
            continue
        if header and not stripped.startswith("--") and stripped.count(",") >= 10:
            values = [item.strip() for item in stripped.split(",")]
            if len(values) == len(header):
                data = values
    if header is None or data is None:
        raise ValueError("could not parse hipblaslt-bench CSV result block")
    return header, data


def _match_or_none(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group("value").strip() if match else None


def parse_bench_output(stdout: str) -> dict[str, Any]:
    header, data = _csv_payload_lines(stdout)
    row = dict(zip(header, data, strict=True))
    device_name = None
    device_arch = None
    device_match = DEVICE_RE.search(stdout)
    if device_match:
        device_name = device_match.group("name").strip()
        device_arch = device_match.group("arch").strip()
    return {
        "hipblaslt_gflops": float(row["hipblaslt-Gflops"]),
        "hipblaslt_time_us": float(row["us"]),
        "solution_index": int(_match_or_none(SOLUTION_INDEX_RE, stdout) or -1),
        "solution_name": _match_or_none(SOLUTION_NAME_RE, stdout),
        "kernel_name": _match_or_none(KERNEL_NAME_RE, stdout),
        "device_name": device_name,
        "device_arch": device_arch,
    }


def runtime_env(rocm_devel: Path, rocm_libraries: Path, tensile_libpath: Path | None) -> dict[str, str]:
    env = dict(os.environ)
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    ld_parts = [str(rocm_devel / "llvm/lib"), str(rocm_libraries / "lib"), str(rocm_devel / "lib")]
    if existing_ld:
        ld_parts.append(existing_ld)
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    if tensile_libpath is None:
        tensile_libpath = rocm_libraries / "lib/hipblaslt/library/gfx1151"
    env["HIPBLASLT_TENSILE_LIBPATH"] = str(tensile_libpath)
    return env


def command_for_shape(
    bench: Path,
    shape: Shape,
    *,
    alpha: float,
    beta: float,
    cold_iters: int,
    iters: int,
    requested_solution: int,
    initialization: str,
    use_gpu_timer: bool,
    print_kernel_info: bool,
) -> list[str]:
    cmd = [
        str(bench),
        "--function",
        "matmul",
        "--precision",
        "f16_r",
        "--compute_type",
        "f32_r",
        "--scale_type",
        "f32_r",
        "--transA",
        "N",
        "--transB",
        "T",
        "-m",
        str(shape.m),
        "-n",
        str(shape.n),
        "-k",
        str(shape.k),
        "--batch_count",
        str(shape.batch),
        "--alpha",
        f"{alpha:g}",
        "--beta",
        f"{beta:g}",
        "--bias_vector",
        "--bias_type",
        "f16_r",
        "--bias_source",
        "d",
        "--scaleAlpha_vector",
        "--activation_type",
        "none",
        "--initialization",
        initialization,
        "--cold_iters",
        str(cold_iters),
        "--iters",
        str(iters),
        "--requested_solution",
        str(requested_solution),
    ]
    if use_gpu_timer:
        cmd.append("--use_gpu_timer")
    if print_kernel_info:
        cmd.append("--print_kernel_info")
    return cmd


def _load_logic_solutions(logic_yaml: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(logic_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < 6 or not isinstance(data[5], list):
        raise ValueError(f"unsupported TensileLite logic YAML layout: {logic_yaml}")
    return [item for item in data[5] if isinstance(item, dict)]


def _name_tokens(value: str | None) -> list[str]:
    if not value:
        return []
    start = value.find("_MT")
    suffix = value[start + 1 :] if start >= 0 else value
    return suffix.split("_")


def _is_ordered_subsequence(needles: list[str], haystack: list[str]) -> bool:
    pos = 0
    for needle in needles:
        try:
            pos = haystack.index(needle, pos) + 1
        except ValueError:
            return False
    return True


def _match_logic_solution(
    parsed: dict[str, Any], logic_solutions: list[dict[str, Any]], shape_id: str
) -> dict[str, Any]:
    solution_tokens = _name_tokens(parsed.get("solution_name"))
    kernel_tokens = _name_tokens(parsed.get("kernel_name"))
    solution_matches: list[dict[str, Any]] = []
    kernel_matches: list[dict[str, Any]] = []
    for solution in logic_solutions:
        solution_name_min = str(solution.get("SolutionNameMin") or "")
        kernel_name_min = str(solution.get("KernelNameMin") or "")
        if solution_name_min and _is_ordered_subsequence(_name_tokens(solution_name_min), solution_tokens):
            solution_matches.append(solution)
        if kernel_name_min and _is_ordered_subsequence(_name_tokens(kernel_name_min), kernel_tokens):
            kernel_matches.append(solution)
    if solution_matches:
        return solution_matches[0]
    if kernel_matches:
        selected_mod = parsed.get("solution_index", -1) % len(logic_solutions)
        mod_matches = [
            solution for solution in kernel_matches if int(solution.get("SolutionIndex", -2)) == selected_mod
        ]
        if len(mod_matches) == 1:
            return mod_matches[0]
        return kernel_matches[0]
    raise ValueError(f"could not map hipblaslt-bench solution for {shape_id}: {parsed.get('solution_name')}")


def _solution_to_candidate(solution: dict[str, Any]) -> Candidate:
    params = {key: solution[key] for key in CANDIDATE_PARAM_KEYS if key in solution}
    matrix_instruction = solution.get("MatrixInstruction")
    if isinstance(matrix_instruction, list):
        mi = list(matrix_instruction)
        if len(mi) == 4:
            mi.append(1)
            mi.extend(solution.get("MIWaveTile") or [1, 1])
            mi.extend(solution.get("MIWaveGroup") or [1, 1])
        params["MatrixInstruction"] = mi
    if "StoreVectorWidth" not in params:
        params["StoreVectorWidth"] = solution.get("GlobalWriteVectorWidth", -1)
    return Candidate(params=params, source="installed_hipblaslt_baseline")


def query_baselines(args: argparse.Namespace, shapes: list[Shape], env: dict[str, str]) -> list[BaselineSelection]:
    logic_solutions = _load_logic_solutions(args.logic_yaml)
    logs_dir = args.output_dir / "hipblaslt_query_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    selections: list[BaselineSelection] = []
    for index, shape in enumerate(shapes, 1):
        cmd = command_for_shape(
            args.bench,
            shape,
            alpha=args.alpha,
            beta=args.beta,
            cold_iters=args.cold_iters,
            iters=args.iters,
            requested_solution=args.requested_solution,
            initialization=args.initialization,
            use_gpu_timer=not args.no_gpu_timer,
            print_kernel_info=True,
        )
        stdout_path = logs_dir / f"{shape.id}.stdout.log"
        stderr_path = logs_dir / f"{shape.id}.stderr.log"
        print(f"[{index}/{len(shapes)}] query {shape.id}", flush=True)
        with apu_activity_lock(exclusive=True):
            proc = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=args.timeout, check=False)
        stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
        try:
            if proc.returncode != 0:
                raise RuntimeError(f"hipblaslt-bench failed for {shape.id}: {stderr_path}")
            parsed = parse_bench_output(proc.stdout)
            solution = _match_logic_solution(parsed, logic_solutions, shape.id)
            candidate = _solution_to_candidate(solution)
        except Exception as exc:
            if args.stop_on_error:
                raise
            print(f"warning: skipping {shape.id}: {exc}", flush=True)
            continue
        selections.append(
            BaselineSelection(
                shape=shape,
                candidate=candidate,
                solution_index=int(parsed["solution_index"]),
                solution_name=parsed.get("solution_name"),
                kernel_name=parsed.get("kernel_name"),
                logic_solution_index=int(solution["SolutionIndex"]) if "SolutionIndex" in solution else None,
                logic_solution_name=solution.get("SolutionNameMin"),
                hipblaslt_gflops=float(parsed["hipblaslt_gflops"]),
                hipblaslt_time_us=float(parsed["hipblaslt_time_us"]),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                command=cmd,
            )
        )
    return selections


def write_selections(path: Path, selections: list[BaselineSelection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "shape_id",
                "candidate_hash",
                "hipblaslt_solution_index",
                "hipblaslt_solution_name",
                "hipblaslt_kernel_name",
                "logic_solution_index",
                "logic_solution_name",
                "hipblaslt_gflops",
                "hipblaslt_time_us",
                "stdout_path",
                "stderr_path",
                "command_json",
            ]
        )
        for item in selections:
            writer.writerow(
                [
                    item.shape.id,
                    item.candidate.hash,
                    item.solution_index,
                    item.solution_name or "",
                    item.kernel_name or "",
                    item.logic_solution_index if item.logic_solution_index is not None else "",
                    item.logic_solution_name or "",
                    item.hipblaslt_gflops,
                    item.hipblaslt_time_us,
                    item.stdout_path,
                    item.stderr_path,
                    json.dumps(item.command),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import current hipBLASLt-selected configs as EvoTensile baseline candidates"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--rocm-devel", type=Path, default=DEFAULT_ROCM_DEVEL)
    parser.add_argument("--rocm-libraries", type=Path, default=DEFAULT_ROCM_LIBRARIES)
    parser.add_argument("--tensile-libpath", type=Path, default=None)
    parser.add_argument("--logic-yaml", type=Path, default=DEFAULT_LOGIC_YAML)
    parser.add_argument("--runner-bin", default=None)
    parser.add_argument("--tensilelite-bin", default=DEFAULT_TENSILELITE_BIN)
    parser.add_argument(
        "--compile-threads",
        type=int,
        default=DEFAULT_COMPILE_THREADS,
        help="TensileLite CpuThreads per batch; defaults to 1",
    )
    parser.add_argument("--build-timeout", type=float, default=None, help="defaults to the target profile; 0 disables")
    parser.add_argument("--runner-timeout", type=float, default=None, help="defaults to the target profile; 0 disables")
    parser.add_argument("--limit-shapes", type=int, default=None)
    parser.add_argument("--shapes", nargs="*")
    parser.add_argument("--num-warmups", type=int, default=None)
    parser.add_argument("--num-benchmarks", type=int, default=None)
    parser.add_argument("--enqueues-per-sync", type=int, default=None)
    parser.add_argument("--syncs-per-benchmark", type=int, default=None)
    parser.add_argument("--num-elements-to-validate", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--initialization", default="hpl")
    parser.add_argument("--cold-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--requested-solution", type=int, default=1)
    parser.add_argument("--no-gpu-timer", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after the first query or schedule error")
    parser.add_argument("--query-only", action="store_true")
    args = parser.parse_args()

    if not args.bench.exists():
        raise FileNotFoundError(args.bench)
    if not args.logic_yaml.exists():
        raise FileNotFoundError(args.logic_yaml)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    profile, protocol = _profile_protocol(args)
    shapes = _parse_shapes(args, profile)
    runner_bin = args.runner_bin or profile.default_runner_bin
    build_timeout = profile.default_build_timeout_s if args.build_timeout is None else args.build_timeout
    runner_timeout = profile.default_runner_timeout_s if args.runner_timeout is None else args.runner_timeout
    if build_timeout is not None and build_timeout <= 0:
        build_timeout = None
    if runner_timeout is not None and runner_timeout <= 0:
        runner_timeout = None
    env = runtime_env(args.rocm_devel, args.rocm_libraries, args.tensile_libpath)
    selections = query_baselines(args, shapes, env)
    selections_csv = args.output_dir / "baseline_selections.csv"
    write_selections(selections_csv, selections)

    grouped: dict[str, tuple[Candidate, list[Shape]]] = {}
    for item in selections:
        _, group_shapes = grouped.setdefault(item.candidate.hash, (item.candidate, []))
        group_shapes.append(item.shape)

    executed_groups = []
    if not args.query_only:
        db = EvoTensileDB.connect(args.db)
        for group_index, (candidate, group_shapes) in enumerate(grouped.values()):
            group_dir = args.output_dir / f"baseline_group_{group_index:04d}_{candidate.hash}"
            result = execute_schedule(
                db,
                shapes=group_shapes,
                candidates=[candidate],
                output_root=group_dir,
                target_profile=profile,
                protocol=protocol,
                min_samples=protocol.num_benchmarks,
                candidate_batch_size=1,
                shape_batch_size=max(1, len(group_shapes)),
                tensilelite_bin=args.tensilelite_bin,
                compile_threads=args.compile_threads,
                keep_going=not args.stop_on_error,
                runner_bin=runner_bin,
                build_timeout_s=build_timeout,
                runner_timeout_s=runner_timeout,
            )
            status_counts: dict[str, int] = {}
            for executed in result.executed_batches:
                if executed.ingest is None:
                    continue
                for status, count in executed.ingest.status_counts.items():
                    status_counts[status] = status_counts.get(status, 0) + count
            executed_groups.append(
                {
                    "candidate_hash": candidate.hash,
                    "shape_count": len(group_shapes),
                    "planned_batches": len(result.planned_batches),
                    "executed_batches": len(result.executed_batches),
                    "status_counts": status_counts,
                }
            )

    metadata = {
        "db": args.db,
        "output_dir": str(args.output_dir),
        "profile": profile.name,
        "problem_type_hash": profile.problem_type_hash,
        "benchmark_protocol_hash": profile.benchmark_protocol_hash(protocol),
        "protocol": protocol.global_parameters(),
        "shape_count": len(shapes),
        "unique_candidate_count": len(grouped),
        "selections_csv": str(selections_csv),
        "query_only": args.query_only,
        "stop_on_error": args.stop_on_error,
        "build_timeout_s": build_timeout,
        "runner_timeout_s": runner_timeout,
        "executed_groups": executed_groups,
        "HIPBLASLT_TENSILE_LIBPATH": env["HIPBLASLT_TENSILE_LIBPATH"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
