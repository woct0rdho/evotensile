#!/usr/bin/env python3

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import yaml

from evotensile.candidate import Candidate
from evotensile.manifest import write_manifest
from evotensile.profile import get_profile
from evotensile.runner import DEFAULT_TENSILELITE_BIN, run_tensilelite
from evotensile.scheduler import DEFAULT_COMPILE_THREADS
from evotensile.search_space import defaulted_params, explain_invalid_nt_hhs
from evotensile.shapes import parse_shape
from evotensile.solution_mapping import solution_matches_candidate
from evotensile.structured_runner import find_solution_yamls
from evotensile.yaml_writer import write_tensilelite_yaml


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_params_from_db(db_path: Path, candidate_hash: str) -> dict[str, Any]:
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT candidate_json FROM candidates WHERE candidate_hash = ?",
            (candidate_hash,),
        ).fetchone()
    if row is None:
        raise SystemExit(f"candidate not found in {db_path}: {candidate_hash}")
    value = json.loads(row[0])
    return defaulted_params(value.get("params", value))


def _candidate_params_from_file(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    return defaulted_params(value.get("params", value))


def _solution_records(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict) and ("SolutionIndex" in item or "KernelNameMin" in item)]


def _interesting_log_lines(paths: list[Path]) -> list[str]:
    markers = (
        "reject:",
        "fatal",
        "failed to generate assembly",
        "overflowed resources",
        "total vgpr",
        "not in [0, 256]",
        "no valid solutions",
        "runtimeerror",
        "did_not_satisfy_asserts",
        "validation",
        "wrong_hardware",
    )
    lines: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if any(marker in line.lower() for marker in markers):
                lines.append(f"{path.name}: {line.strip()}")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an offline TensileLite oracle check for one EvoTensile candidate")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--params-json", type=Path, help="JSON file containing candidate params or {'params': ...}")
    source.add_argument("--candidate-hash", help="Candidate hash to load from --db")
    parser.add_argument("--db", type=Path, help="EvoTensile DB used with --candidate-hash")
    parser.add_argument("--shape", default="8192,8192,1,8192", help="Shape as M,N,B,K")
    parser.add_argument("--profile", default="gfx1151-nt-hhs")
    parser.add_argument("--output-dir", type=Path, default=Path("out/tensilelite_oracle"))
    parser.add_argument("--tensilelite-bin", type=Path, default=Path(DEFAULT_TENSILELITE_BIN))
    parser.add_argument("--compile-threads", type=int, default=DEFAULT_COMPILE_THREADS)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--print-rejection-reasons", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_hash and args.db is None:
        raise SystemExit("--db is required with --candidate-hash")

    profile = get_profile(args.profile)
    shape = parse_shape(args.shape)
    params = (
        _candidate_params_from_db(args.db, args.candidate_hash)
        if args.candidate_hash
        else _candidate_params_from_file(args.params_json)
    )
    candidate = Candidate(params, source="oracle")
    output_dir = args.output_dir / candidate.hash
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "config.yaml"
    manifest_path = output_dir / "config.manifest.csv"

    global_parameters = profile.global_parameters(profile.default_protocol)
    if args.print_rejection_reasons:
        global_parameters["PrintSolutionRejectionReason"] = True
    write_tensilelite_yaml(
        yaml_path,
        [candidate],
        [shape],
        global_parameters=global_parameters,
        library_logic=profile.library_logic,
        problem_type=profile.problem_type,
    )
    write_manifest(manifest_path, [candidate], [shape])

    reasons = explain_invalid_nt_hhs(candidate.canonical_params(), shape=shape)
    result = run_tensilelite(
        yaml_path,
        output_dir / "build",
        tensilelite_bin=args.tensilelite_bin,
        build_only=True,
        cpu_threads=args.compile_threads,
        timeout_s=args.timeout,
    )
    solution_yamls = find_solution_yamls([result.output_dir])
    mapping = []
    for solution_yaml in solution_yamls:
        mapping.extend(
            {
                "path": str(solution_yaml),
                "solution_index": solution.get("SolutionIndex"),
                "kernel_name": solution.get("KernelNameMin"),
                "matches_candidate": solution_matches_candidate(solution, candidate.canonical_params()),
            }
            for solution in _solution_records(solution_yaml)
        )

    summary = {
        "candidate_hash": candidate.hash,
        "shape_id": shape.id,
        "yaml_path": str(yaml_path),
        "manifest_path": str(manifest_path),
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "stdout_path": str(result.stdout_path),
        "stderr_path": str(result.stderr_path),
        "invalid_reasons": [reason.__dict__ for reason in reasons],
        "solution_yamls": [str(path) for path in solution_yamls],
        "mapping": mapping,
        "interesting_log_lines": _interesting_log_lines([result.stdout_path, result.stderr_path]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
