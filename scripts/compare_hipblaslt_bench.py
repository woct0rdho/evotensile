#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evotensile.candidate import Candidate  # noqa: E402
from evotensile.shapes import shape_from_id  # noqa: E402
from evotensile.tensilelite_keys import DIRECT_SOLUTION_MATCH_KEYS  # noqa: E402
from evotensile.yaml_writer import write_tensilelite_yaml  # noqa: E402

DEFAULT_BENCH = Path.home() / "rocm-libraries/build/hipblaslt-bench-current/clients/hipblaslt-bench"
DEFAULT_ROCM_DEVEL = Path.home() / "venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel"
DEFAULT_ROCM_LIBRARIES = Path.home() / "venv_torch/lib/python3.14/site-packages/_rocm_sdk_libraries"
DEFAULT_WIN_CSV = Path("out/grid100_full_20260618_top4_retime_export/winners.csv")
DEFAULT_OUTPUT_DIR = Path("out/hipblaslt_bench_grid100_20260619")
DEFAULT_LOGIC_YAML = (
    Path.home()
    / "rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/GridBased/gfx1151_Cijk_Ailk_Bjlk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml"
)

CSV_HEADER_MARKER = "hipblaslt-Gflops"
SOLUTION_INDEX_RE = re.compile(r"--Solution index:\s*(?P<value>-?\d+)")
SOLUTION_NAME_RE = re.compile(r"--Solution name:\s*(?P<value>\S.*)")
KERNEL_NAME_RE = re.compile(r"--kernel name:\s*(?P<value>\S.*)")
HIPBLASLT_VERSION_RE = re.compile(r"hipBLASLt version:\s*(?P<value>\S+)")
HIPBLASLT_GIT_VERSION_RE = re.compile(r"hipBLASLt git version:\s*(?P<value>\S+)")
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
class WinnerRow:
    shape_id: str
    candidate_hash: str
    samples: int
    median_gflops: float
    best_gflops: float
    median_time_us: float
    best_time_us: float


def _as_float(value: str) -> float:
    if value == "":
        return float("nan")
    return float(value)


def load_winners(path: Path) -> list[WinnerRow]:
    winners: list[WinnerRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            winners.append(
                WinnerRow(
                    shape_id=row["shape_id"],
                    candidate_hash=row["candidate_hash"],
                    samples=int(row["samples"]),
                    median_gflops=_as_float(row["median_gflops"]),
                    best_gflops=_as_float(row["best_gflops"]),
                    median_time_us=_as_float(row["median_time_us"]),
                    best_time_us=_as_float(row["best_time_us"]),
                )
            )
    return winners


def _csv_payload_lines(stdout: str) -> tuple[list[str], list[str]]:
    header: list[str] | None = None
    data: list[str] | None = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if CSV_HEADER_MARKER in stripped and "," in stripped:
            # hipblaslt-bench prefixes the header with "[0]:".
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


def parse_bench_output(stdout: str) -> dict[str, Any]:
    header, data = _csv_payload_lines(stdout)
    row = dict(zip(header, data, strict=True))

    def match_or_none(pattern: re.Pattern[str]) -> str | None:
        match = pattern.search(stdout)
        return match.group("value").strip() if match else None

    device_name = None
    device_arch = None
    device_match = DEVICE_RE.search(stdout)
    if device_match:
        device_name = device_match.group("name").strip()
        device_arch = device_match.group("arch").strip()

    return {
        "hipblaslt_gflops": float(row["hipblaslt-Gflops"]),
        "hipblaslt_gbps": float(row["hipblaslt-GB/s"]),
        "hipblaslt_time_us": float(row["us"]),
        "solution_index": int(match_or_none(SOLUTION_INDEX_RE) or -1),
        "solution_name": match_or_none(SOLUTION_NAME_RE),
        "kernel_name": match_or_none(KERNEL_NAME_RE),
        "hipblaslt_version": match_or_none(HIPBLASLT_VERSION_RE),
        "hipblaslt_git_version": match_or_none(HIPBLASLT_GIT_VERSION_RE),
        "device_name": device_name,
        "device_arch": device_arch,
        "raw_csv_row": row,
    }


def command_for_shape(
    bench: Path,
    winner: WinnerRow,
    *,
    alpha: float,
    beta: float,
    cold_iters: int,
    iters: int,
    requested_solution: int,
    initialization: str,
    verify: bool,
    use_gpu_timer: bool,
    print_kernel_info: bool,
    extra_args: list[str],
) -> list[str]:
    shape = shape_from_id(winner.shape_id)
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
    if verify:
        cmd.append("--verify")
    cmd.extend(extra_args)
    return cmd


def runtime_env(rocm_devel: Path, rocm_libraries: Path, tensile_libpath: Path | None) -> dict[str, str]:
    env = dict(os.environ)
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    ld_parts = [str(rocm_devel / "llvm/lib"), str(rocm_libraries / "lib"), str(rocm_devel / "lib")]
    if existing_ld:
        ld_parts.append(existing_ld)
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)

    # With HIPBLASLT_TENSILE_LIBPATH set, hipBLASLt expects the per-arch directory directly.
    if tensile_libpath is None:
        tensile_libpath = rocm_libraries / "lib/hipblaslt/library/gfx1151"
    env["HIPBLASLT_TENSILE_LIBPATH"] = str(tensile_libpath)
    return env


def _read_existing_completed(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    completed: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                completed.add(row["shape_id"])
    return completed


def _write_csv_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "shape_id",
                "candidate_hash",
                "status",
                "evotensile_samples",
                "evotensile_median_gflops",
                "evotensile_best_gflops",
                "evotensile_median_time_us",
                "evotensile_best_time_us",
                "hipblaslt_gflops",
                "hipblaslt_gbps",
                "hipblaslt_time_us",
                "speedup_vs_hipblaslt",
                "time_ratio_vs_hipblaslt",
                "solution_index",
                "solution_name",
                "kernel_name",
                "returncode",
                "duration_s",
                "stdout_path",
                "stderr_path",
                "command_json",
            ]
        )


def append_result(csv_path: Path, row: list[Any]) -> None:
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(row)


def _summary_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def write_summary(csv_path: Path, metadata: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    speedups = [float(row["speedup_vs_hipblaslt"]) for row in ok_rows if row.get("speedup_vs_hipblaslt")]
    hip_gflops = [float(row["hipblaslt_gflops"]) for row in ok_rows if row.get("hipblaslt_gflops")]
    evo_gflops = [float(row["evotensile_median_gflops"]) for row in ok_rows if row.get("evotensile_median_gflops")]
    regressions = [value for value in speedups if value < 1.0]

    summary = {
        **metadata,
        "result_csv": str(csv_path),
        "rows": len(rows),
        "ok_rows": len(ok_rows),
        "failed_rows": len(rows) - len(ok_rows),
        "speedup_stats": _summary_stats(speedups),
        "hipblaslt_gflops_stats": _summary_stats(hip_gflops),
        "evotensile_median_gflops_stats": _summary_stats(evo_gflops),
        "regression_count": len(regressions),
        "regression_shape_ids": [
            row["shape_id"]
            for row in ok_rows
            if row.get("speedup_vs_hipblaslt") and float(row["speedup_vs_hipblaslt"]) < 1.0
        ],
        "wins_gt_5pct": sum(1 for value in speedups if value > 1.05),
        "wins_gt_10pct": sum(1 for value in speedups if value > 1.10),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _load_comparison_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _match_logic_solution(row: dict[str, str], logic_solutions: list[dict[str, Any]]) -> dict[str, Any]:
    solution_tokens = _name_tokens(row.get("solution_name"))
    kernel_tokens = _name_tokens(row.get("kernel_name"))
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

    # KernelNameMin intentionally omits some solution-level tokens (for example
    # GSU), so use it only as a fallback and disambiguate with the bench solution
    # index modulo the logic-solution count when possible.
    if kernel_matches:
        try:
            selected_mod = int(row.get("solution_index", "")) % len(logic_solutions)
        except ValueError:
            selected_mod = -1
        mod_matches = [
            solution for solution in kernel_matches if int(solution.get("SolutionIndex", -2)) == selected_mod
        ]
        if len(mod_matches) == 1:
            return mod_matches[0]
        return kernel_matches[0]

    raise ValueError(f"could not map hipblaslt-bench solution for {row.get('shape_id')}: {row.get('solution_name')}")


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
    return Candidate(params=params, source="installed_hipblaslt")


def _safe_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def export_hybrid_configs(
    *,
    comparison_csv: Path,
    winners_csv: Path,
    output_dir: Path,
    logic_yaml: Path,
    prefer_tuned_on_tie: bool,
) -> dict[str, Any]:
    comparison_rows = _load_comparison_rows(comparison_csv)
    logic_solutions = _load_logic_solutions(logic_yaml)
    tuned_rows_by_shape: dict[str, dict[str, str]] = {}
    with winners_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            tuned_rows_by_shape[row["shape_id"]] = row

    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_dir = output_dir / "per_shape_yaml"
    json_dir = output_dir / "candidates_json"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    rows_out: list[list[Any]] = []
    kept_tuned = 0
    replaced_with_hipblaslt = 0
    installed_candidates: dict[str, Candidate] = {}
    comparison_ok = [row for row in comparison_rows if row.get("status") == "ok"]

    for row in sorted(comparison_ok, key=lambda item: item["shape_id"]):
        shape_id = row["shape_id"]
        tuned = tuned_rows_by_shape.get(shape_id)
        if tuned is None:
            raise ValueError(f"comparison row has no tuned winner: {shape_id}")
        speedup = _safe_float(row.get("speedup_vs_hipblaslt", ""))
        if speedup is None:
            raise ValueError(f"comparison row has no speedup: {shape_id}")

        use_tuned = speedup >= 1.0 if prefer_tuned_on_tie else speedup > 1.0
        if use_tuned:
            kept_tuned += 1
            source = "evotensile"
            selected_hash = tuned["candidate_hash"]
            source_yaml = Path(tuned["yaml_path"])
            source_json = Path(tuned["candidate_json_path"])
            yaml_path = yaml_dir / source_yaml.name
            json_path = json_dir / source_json.name
            shutil.copy2(source_yaml, yaml_path)
            shutil.copy2(source_json, json_path)
            selected_gflops = row["evotensile_median_gflops"]
            selected_time_us = row["evotensile_median_time_us"]
            logic_solution_index = ""
            logic_solution_name = ""
        else:
            replaced_with_hipblaslt += 1
            source = "installed_hipblaslt"
            solution = _match_logic_solution(row, logic_solutions)
            candidate = _solution_to_candidate(solution)
            installed_candidates[candidate.hash] = candidate
            selected_hash = candidate.hash
            shape = shape_from_id(shape_id)
            yaml_path = yaml_dir / f"{shape_id}_{candidate.hash}.yaml"
            json_path = json_dir / f"{candidate.hash}.json"
            write_tensilelite_yaml(yaml_path, [candidate], [shape])
            json_path.write_text(candidate.to_json() + "\n", encoding="utf-8")
            selected_gflops = row["hipblaslt_gflops"]
            selected_time_us = row["hipblaslt_time_us"]
            logic_solution_index = solution.get("SolutionIndex", "")
            logic_solution_name = solution.get("SolutionNameMin", "")

        rows_out.append(
            [
                shape_id,
                source,
                selected_hash,
                row["candidate_hash"],
                row["evotensile_median_gflops"],
                row["evotensile_median_time_us"],
                row["hipblaslt_gflops"],
                row["hipblaslt_time_us"],
                row["speedup_vs_hipblaslt"],
                selected_gflops,
                selected_time_us,
                row["solution_index"],
                row["solution_name"],
                row["kernel_name"],
                logic_solution_index,
                logic_solution_name,
                yaml_path,
                json_path,
            ]
        )

    csv_path = output_dir / "winners.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "shape_id",
                "selected_source",
                "selected_candidate_hash",
                "tuned_candidate_hash",
                "evotensile_median_gflops",
                "evotensile_median_time_us",
                "hipblaslt_gflops",
                "hipblaslt_time_us",
                "speedup_vs_hipblaslt",
                "selected_gflops",
                "selected_time_us",
                "hipblaslt_solution_index",
                "hipblaslt_solution_name",
                "hipblaslt_kernel_name",
                "logic_solution_index",
                "logic_solution_name",
                "yaml_path",
                "candidate_json_path",
            ]
        )
        writer.writerows(rows_out)

    metadata = {
        "comparison_csv": str(comparison_csv),
        "tuned_winners_csv": str(winners_csv),
        "logic_yaml": str(logic_yaml),
        "output_dir": str(output_dir),
        "winner_count": len(rows_out),
        "kept_tuned": kept_tuned,
        "replaced_with_hipblaslt": replaced_with_hipblaslt,
        "installed_candidate_count": len(installed_candidates),
        "prefer_tuned_on_tie": prefer_tuned_on_tie,
        "policy": "Use EvoTensile when speedup_vs_hipblaslt is above threshold; otherwise export the matching installed hipBLASLt solution as a TensileLite Groups config.",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "winners_csv": str(csv_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winners-csv", type=Path, default=DEFAULT_WIN_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--rocm-devel", type=Path, default=DEFAULT_ROCM_DEVEL)
    parser.add_argument("--rocm-libraries", type=Path, default=DEFAULT_ROCM_LIBRARIES)
    parser.add_argument("--tensile-libpath", type=Path, default=None)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--cold-iters", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--initialization", default="hpl")
    parser.add_argument("--requested-solution", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-shape-id", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--no-gpu-timer", action="store_true")
    parser.add_argument("--no-print-kernel-info", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument(
        "--hybrid-export-dir",
        type=Path,
        default=None,
        help="Export per-shape configs that keep EvoTensile only when faster than installed hipBLASLt.",
    )
    parser.add_argument(
        "--logic-yaml",
        type=Path,
        default=DEFAULT_LOGIC_YAML,
        help="Current hipBLASLt GridBased logic YAML used to reconstruct replacement configs.",
    )
    parser.add_argument(
        "--prefer-tuned-on-tie",
        action="store_true",
        help="Keep EvoTensile when speedup_vs_hipblaslt is exactly 1.0; default requires strictly faster.",
    )
    args = parser.parse_args()

    if not args.bench.exists():
        raise FileNotFoundError(args.bench)
    if not args.winners_csv.exists():
        raise FileNotFoundError(args.winners_csv)

    output_dir = args.output_dir
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"
    summary_path = output_dir / "summary.json"
    metadata_path = output_dir / "metadata.json"
    _write_csv_header(csv_path)

    winners = load_winners(args.winners_csv)
    if args.only_shape_id:
        wanted = set(args.only_shape_id)
        winners = [winner for winner in winners if winner.shape_id in wanted]
    if args.limit is not None:
        winners = winners[: args.limit]

    completed = _read_existing_completed(csv_path) if args.resume else set()
    env = runtime_env(args.rocm_devel, args.rocm_libraries, args.tensile_libpath)

    metadata = {
        "bench": str(args.bench),
        "winners_csv": str(args.winners_csv),
        "output_dir": str(output_dir),
        "rocm_devel": str(args.rocm_devel),
        "rocm_libraries": str(args.rocm_libraries),
        "HIPBLASLT_TENSILE_LIBPATH": env["HIPBLASLT_TENSILE_LIBPATH"],
        "protocol": {
            "precision": "f16_r",
            "compute_type": "f32_r",
            "scale_type": "f32_r",
            "transA": "N",
            "transB": "T",
            "bias_vector": True,
            "bias_type": "f16_r",
            "bias_source": "d",
            "scaleAlpha_vector": True,
            "activation_type": "none",
            "alpha": args.alpha,
            "beta": args.beta,
            "initialization": args.initialization,
            "cold_iters": args.cold_iters,
            "iters": args.iters,
            "use_gpu_timer": not args.no_gpu_timer,
            "requested_solution": args.requested_solution,
            "verify": args.verify,
            "note": "hipblaslt-bench reports one average over iters launches; EvoTensile winner values are TensileLite medians over 10 benchmark groups.",
        },
        "shape_count_requested": len(winners),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for index, winner in enumerate(winners, start=1):
        if winner.shape_id in completed:
            print(f"[{index}/{len(winners)}] skip cached {winner.shape_id}", flush=True)
            continue

        cmd = command_for_shape(
            args.bench,
            winner,
            alpha=args.alpha,
            beta=args.beta,
            cold_iters=args.cold_iters,
            iters=args.iters,
            requested_solution=args.requested_solution,
            initialization=args.initialization,
            verify=args.verify,
            use_gpu_timer=not args.no_gpu_timer,
            print_kernel_info=not args.no_print_kernel_info,
            extra_args=args.extra_arg,
        )
        stdout_path = logs_dir / f"{winner.shape_id}.stdout.log"
        stderr_path = logs_dir / f"{winner.shape_id}.stderr.log"
        print(f"[{index}/{len(winners)}] run {winner.shape_id}", flush=True)
        started = time.time()
        try:
            proc = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=args.timeout, check=False)
            duration = time.time() - started
            stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
            stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
            if proc.returncode == 0:
                parsed = parse_bench_output(proc.stdout)
                speedup = winner.median_gflops / parsed["hipblaslt_gflops"]
                time_ratio = winner.median_time_us / parsed["hipblaslt_time_us"]
                status = "ok"
            else:
                parsed = {}
                speedup = ""
                time_ratio = ""
                status = "failed"
        except Exception as exc:  # Keep long grid runs resumable after parser/timeouts.
            duration = time.time() - started
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            parsed = {}
            speedup = ""
            time_ratio = ""
            status = "failed"
            proc = None  # type: ignore[assignment]

        append_result(
            csv_path,
            [
                winner.shape_id,
                winner.candidate_hash,
                status,
                winner.samples,
                winner.median_gflops,
                winner.best_gflops,
                winner.median_time_us,
                winner.best_time_us,
                parsed.get("hipblaslt_gflops", ""),
                parsed.get("hipblaslt_gbps", ""),
                parsed.get("hipblaslt_time_us", ""),
                speedup,
                time_ratio,
                parsed.get("solution_index", ""),
                parsed.get("solution_name", ""),
                parsed.get("kernel_name", ""),
                proc.returncode if proc is not None else "",
                f"{duration:.6f}",
                stdout_path,
                stderr_path,
                json.dumps(cmd),
            ],
        )
        if status == "ok":
            print(
                f"[{index}/{len(winners)}] ok {winner.shape_id}: "
                f"Evo {winner.median_gflops:.1f} vs hipBLASLt {parsed['hipblaslt_gflops']:.1f} GF/s "
                f"speedup {float(speedup):.3f}x",
                flush=True,
            )
        else:
            print(f"[{index}/{len(winners)}] failed {winner.shape_id}", flush=True)

    summary = write_summary(csv_path, metadata, summary_path)
    if args.hybrid_export_dir is not None:
        hybrid_metadata = export_hybrid_configs(
            comparison_csv=csv_path,
            winners_csv=args.winners_csv,
            output_dir=args.hybrid_export_dir,
            logic_yaml=args.logic_yaml,
            prefer_tuned_on_tie=args.prefer_tuned_on_tie,
        )
        summary["hybrid_export"] = hybrid_metadata
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
