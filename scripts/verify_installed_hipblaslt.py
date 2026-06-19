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

CSV_HEADER_MARKER = "hipblaslt-Gflops"
SOLUTION_INDEX_RE = re.compile(r"--Solution index:\s*(?P<value>-?\d+)")
SOLUTION_NAME_RE = re.compile(r"--Solution name:\s*(?P<value>\S.*)")
HIPBLASLT_VERSION_RE = re.compile(r"hipBLASLt version:\s*(?P<value>\S+)")
HIPBLASLT_GIT_VERSION_RE = re.compile(r"hipBLASLt git version:\s*(?P<value>\S+)")
SUPPORTED_RE = re.compile(r"Is supported\s+(?P<supported>\d+)\s*/\s*Total solutions:\s*(?P<total>\d+)")

DEFAULT_BENCH = Path.home() / "rocm-libraries/build/hipblaslt-bench/clients/hipblaslt-bench"
DEFAULT_ROCM_PATH = Path(
    os.environ.get("ROCM_PATH", Path.home() / "venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel")
)
DEFAULT_OUTPUT_DIR = Path("out/hipblaslt_correctness_smoke")


@dataclass(frozen=True)
class Case:
    name: str
    m: int
    n: int
    k: int
    on_tuned_grid: bool

    @property
    def shape_id(self) -> str:
        return f"m{self.m}_n{self.n}_b1_k{self.k}"


DEFAULT_CASES = (
    Case("on_grid_small_skinny", 512, 128, 256, True),
    Case("on_grid_square_1024", 1024, 1024, 1024, True),
    Case("on_grid_regression_guard", 640, 768, 4096, True),
    Case("off_grid_mid_square", 768, 768, 1024, False),
    Case("off_grid_k1536", 1152, 896, 1536, False),
    Case("off_grid_rect_large", 1536, 512, 2048, False),
)


def _parse_shape(value: str) -> Case:
    parts = value.split(",")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError("shape must be M,N,K or name,M,N,K")
    if len(parts) == 3:
        name = f"custom_{value.replace(',', 'x')}"
        m, n, k = (int(part) for part in parts)
    else:
        name = parts[0]
        m, n, k = (int(part) for part in parts[1:])
    return Case(name=name, m=m, n=n, k=k, on_tuned_grid=False)


def _csv_payload(stdout: str) -> dict[str, str]:
    header: list[str] | None = None
    row: list[str] | None = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if CSV_HEADER_MARKER in stripped and "," in stripped:
            stripped = re.sub(r"^\[\d+\]:", "", stripped)
            header = [item.strip() for item in stripped.split(",")]
            continue
        if header and not stripped.startswith("--") and stripped.count(",") >= 10:
            values = [item.strip() for item in stripped.split(",")]
            if len(values) == len(header):
                row = values
    if header is None or row is None:
        raise ValueError("could not parse hipblaslt-bench CSV result block")
    return dict(zip(header, row, strict=True))


def _match(pattern: re.Pattern[str], text: str, group: str = "value") -> str | None:
    found = pattern.search(text)
    return found.group(group).strip() if found else None


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _command(bench: Path, case: Case, args: argparse.Namespace) -> list[str]:
    return [
        str(bench),
        "--transA",
        "N",
        "--transB",
        "T",
        "-m",
        str(case.m),
        "-n",
        str(case.n),
        "-k",
        str(case.k),
        "--batch_count",
        "1",
        "--precision",
        "f16_r",
        "--a_type",
        "f16_r",
        "--b_type",
        "f16_r",
        "--c_type",
        "f16_r",
        "--d_type",
        "f16_r",
        "--compute_type",
        "f32_r",
        "--bias_vector",
        "--bias_type",
        "f16_r",
        "--bias_source",
        "d",
        "--scaleAlpha_vector",
        "--activation_type",
        "none",
        "--alpha",
        str(args.alpha),
        "--beta",
        str(args.beta),
        "--initialization",
        args.initialization,
        "--cold_iters",
        str(args.cold_iters),
        "--iters",
        str(args.iters),
        "--requested_solution",
        str(args.requested_solution),
        "--use_gpu_timer",
        "--verify",
        "--print_kernel_info",
    ]


def _env(rocm_path: Path, tensile_libpath: Path) -> dict[str, str]:
    env = os.environ.copy()
    old_ld = env.get("LD_LIBRARY_PATH", "")
    ld_parts = [str(rocm_path / "lib"), str(rocm_path / "llvm/lib")]
    if old_ld:
        ld_parts.append(old_ld)
    env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(ld_parts))
    env["HIPBLASLT_TENSILE_LIBPATH"] = str(tensile_libpath)
    return env


def _run_case(bench: Path, case: Case, args: argparse.Namespace, env: dict[str, str], logs_dir: Path) -> dict[str, Any]:
    cmd = _command(bench, case, args)
    started = time.perf_counter()
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=args.timeout)
    elapsed = time.perf_counter() - started

    log_prefix = logs_dir / case.name
    (log_prefix.with_suffix(".stdout.log")).write_text(proc.stdout, encoding="utf-8")
    (log_prefix.with_suffix(".stderr.log")).write_text(proc.stderr, encoding="utf-8")

    parsed: dict[str, Any] = {}
    error: str | None = None
    if proc.returncode == 0:
        try:
            row = _csv_payload(proc.stdout)
            supported = _match(SUPPORTED_RE, proc.stdout, "supported")
            total = _match(SUPPORTED_RE, proc.stdout, "total")
            parsed = {
                "supported": int(supported) if supported is not None else None,
                "total_solutions": int(total) if total is not None else None,
                "hipblaslt_gflops": _float(row, "hipblaslt-Gflops"),
                "hipblaslt_time_us": _float(row, "us"),
                "cpu_gflops": _float(row, "CPU-Gflops"),
                "cpu_time_us": _float(row, "CPU-us"),
                "norm_error": _float(row, "norm_error"),
                "atol": _float(row, "atol"),
                "rtol": _float(row, "rtol"),
                "solution_index": int(_match(SOLUTION_INDEX_RE, proc.stdout) or -1),
                "solution_name": _match(SOLUTION_NAME_RE, proc.stdout),
                "hipblaslt_version": _match(HIPBLASLT_VERSION_RE, proc.stdout),
                "hipblaslt_git_version": _match(HIPBLASLT_GIT_VERSION_RE, proc.stdout),
            }
        except Exception as exc:  # noqa: BLE001 - include parse failure in CSV/JSON output.
            error = f"parse_error: {exc}"
    else:
        error = f"returncode={proc.returncode}"

    status = "ok" if proc.returncode == 0 and error is None and parsed.get("supported") != 0 else "failed"
    if parsed.get("supported") == 0:
        error = "unsupported"

    return {
        "name": case.name,
        "shape_id": case.shape_id,
        "m": case.m,
        "n": case.n,
        "k": case.k,
        "on_tuned_grid": case.on_tuned_grid,
        "status": status,
        "error": error or "",
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        **parsed,
        "stdout_log": str(log_prefix.with_suffix(".stdout.log")),
        "stderr_log": str(log_prefix.with_suffix(".stderr.log")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--rocm-path", type=Path, default=DEFAULT_ROCM_PATH)
    parser.add_argument("--tensile-libpath", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--shape", action="append", type=_parse_shape, default=[], help="Custom case: M,N,K or name,M,N,K; repeatable"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--cold-iters", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--initialization", default="hpl")
    parser.add_argument("--requested-solution", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    bench = args.bench.expanduser().resolve()
    rocm_path = args.rocm_path.expanduser().resolve()
    tensile_libpath = (args.tensile_libpath or (rocm_path / "lib/hipblaslt/library/gfx1151")).expanduser().resolve()
    cases = args.shape or list(DEFAULT_CASES)
    if args.limit:
        cases = cases[: args.limit]

    if not bench.exists():
        raise FileNotFoundError(f"hipblaslt-bench not found: {bench}")
    if not tensile_libpath.exists():
        raise FileNotFoundError(f"HIPBLASLT_TENSILE_LIBPATH not found: {tensile_libpath}")

    output_dir = args.output_dir
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = _env(rocm_path, tensile_libpath)

    results = [_run_case(bench, case, args, env, logs_dir) for case in cases]

    fieldnames = [
        "name",
        "shape_id",
        "m",
        "n",
        "k",
        "on_tuned_grid",
        "status",
        "error",
        "returncode",
        "elapsed_s",
        "supported",
        "total_solutions",
        "solution_index",
        "hipblaslt_gflops",
        "hipblaslt_time_us",
        "cpu_gflops",
        "cpu_time_us",
        "norm_error",
        "atol",
        "rtol",
        "hipblaslt_version",
        "hipblaslt_git_version",
        "stdout_log",
        "stderr_log",
    ]
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "bench": str(bench),
        "rocm_path": str(rocm_path),
        "tensile_libpath": str(tensile_libpath),
        "case_count": len(results),
        "ok_count": sum(1 for row in results if row["status"] == "ok"),
        "failed_count": sum(1 for row in results if row["status"] != "ok"),
        "elapsed_s": sum(float(row["elapsed_s"]) for row in results),
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
