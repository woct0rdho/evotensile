import csv
import json
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import TypedDict

from evotensile.database import EvoTensileDB


class HotConfirmationRecord(TypedDict):
    screen_rank: int
    candidate_hash: str
    returncode: int
    duration_s: float
    samples: int
    median_time_us: float
    best_time_us: float
    median_gflops: float
    best_gflops: float
    library_dir: str
    command: list[str]


def _artifact_map(db_path: str | Path, *, architecture: str) -> dict[str, tuple[dict[str, object], Path]]:
    found: dict[str, tuple[dict[str, object], Path]] = {}
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT metadata_json FROM runs WHERE metadata_json IS NOT NULL ORDER BY timestamp"
        ).fetchall()
    for (metadata_json,) in rows:
        metadata = json.loads(metadata_json)
        command = metadata.get("command") or []
        build_output_dir = metadata.get("build_output_dir")
        if "--pairs" not in command:
            continue
        pairs_path = Path(command[command.index("--pairs") + 1])
        if not pairs_path.exists():
            continue
        if "--library-dir" in command:
            library_dirs = [Path(command[command.index("--library-dir") + 1])]
        elif build_output_dir:
            library_dirs = sorted(Path(build_output_dir).glob(f"1_BenchmarkProblems/**/source/library/{architecture}"))
        else:
            library_dirs = []
        if not library_dirs or not library_dirs[0].exists():
            continue
        for line in pairs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                pair = json.loads(line)
                found[str(pair["candidate_hash"])] = (pair, library_dirs[0])
    return found


def hot_confirm_topk(
    *,
    db_path: str | Path,
    output_dir: str | Path,
    runner_bin: str | Path,
    shape_id: str,
    problem_type_hash: str,
    screening_protocol_hash: str,
    validation_protocol_hash: str,
    architecture: str = "gfx1151",
    top_k: int = 8,
    deadline: float | None = None,
    runner_timeout_s: float = 300.0,
) -> list[HotConfirmationRecord]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    db = EvoTensileDB.connect(db_path)
    summaries = db.rank_evaluations(
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=screening_protocol_hash,
        shape_id=shape_id,
        min_samples=2,
        limit=top_k,
    )
    hashes = [summary.candidate_hash for summary in summaries]
    validated = db.validated_cache_entries(
        problem_type_hash=problem_type_hash,
        validation_protocol_hash=validation_protocol_hash,
        shape_ids=[shape_id],
        candidate_hashes=hashes,
    )
    hashes = [candidate_hash for candidate_hash in hashes if (shape_id, candidate_hash) in validated]
    artifacts = _artifact_map(db_path, architecture=architecture)
    records: list[HotConfirmationRecord] = []
    for screen_rank, candidate_hash in enumerate(hashes, 1):
        if deadline is not None and time.monotonic() >= deadline:
            break
        artifact = artifacts.get(candidate_hash)
        if artifact is None:
            continue
        pair, library_dir = artifact
        hot_pair = dict(pair)
        hot_pair.update(
            {
                "num_warmups": 20,
                "num_benchmarks": 10,
                "enqueues_per_sync": 10,
                "syncs_per_benchmark": 1,
                "num_elements_to_validate": 0,
            }
        )
        candidate_dir = output / f"rank_{screen_rank:02d}_{candidate_hash}"
        candidate_dir.mkdir(exist_ok=True)
        pairs_path = candidate_dir / "pairs.jsonl"
        results_path = candidate_dir / "results.jsonl"
        stdout_path = candidate_dir / "stdout.log"
        stderr_path = candidate_dir / "stderr.log"
        pairs_path.write_text(json.dumps(hot_pair, sort_keys=True) + "\n", encoding="utf-8")
        command = [
            str(runner_bin),
            "--mode",
            "benchmark",
            "--pairs",
            str(pairs_path),
            "--output",
            str(results_path),
            "--validation-backend",
            "hipblaslt",
            "--library-dir",
            str(library_dir),
        ]
        timeout = runner_timeout_s
        if deadline is not None:
            timeout = min(timeout, max(1.0, deadline - time.monotonic()))
        start = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            try:
                process = subprocess.run(
                    command,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    check=False,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                break
        duration = time.perf_counter() - start
        if not results_path.exists():
            continue
        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        times = sorted(float(row["time_us"]) for row in rows if row.get("status") == "ok")
        gflops = sorted(float(row["gflops"]) for row in rows if row.get("status") == "ok")
        if process.returncode != 0 or len(times) != 10:
            continue
        records.append(
            {
                "screen_rank": screen_rank,
                "candidate_hash": candidate_hash,
                "returncode": process.returncode,
                "duration_s": duration,
                "samples": len(times),
                "median_time_us": (times[4] + times[5]) / 2.0,
                "best_time_us": min(times),
                "median_gflops": (gflops[4] + gflops[5]) / 2.0,
                "best_gflops": max(gflops),
                "library_dir": str(library_dir),
                "command": command,
            }
        )
    records.sort(key=lambda record: (record["median_time_us"], record["candidate_hash"]))
    payload = {
        "protocol": {
            "num_warmups": 20,
            "num_benchmarks": 10,
            "enqueues_per_sync": 10,
            "syncs_per_benchmark": 1,
            "num_elements_to_validate": 0,
            "validation_backend": "hipblaslt",
            "validation_disabled_by_num_elements": True,
            "validation_reused_from_screening_db": True,
        },
        "ranked": records,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "ranked.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_hash",
                "screen_rank",
                "samples",
                "median_time_us",
                "best_time_us",
                "median_gflops",
                "best_gflops",
                "duration_s",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "candidate_hash": record["candidate_hash"],
                    "screen_rank": record["screen_rank"],
                    "samples": record["samples"],
                    "median_time_us": record["median_time_us"],
                    "best_time_us": record["best_time_us"],
                    "median_gflops": record["median_gflops"],
                    "best_gflops": record["best_gflops"],
                    "duration_s": record["duration_s"],
                }
            )
    return records
