import csv
import json
import time
from pathlib import Path
from typing import TypedDict

from evotensile.artifacts import load_artifact_mappings
from evotensile.database import EvoTensileDB
from evotensile.metrics import gflops_from_us
from evotensile.protocol import BenchmarkProtocol
from evotensile.shapes import shape_from_id
from evotensile.structured_runner import run_structured_phase, validate_benchmark_samples


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


class HotConfirmationFailure(TypedDict):
    screen_rank: int
    candidate_hash: str
    returncode: int
    timed_out: bool
    reason: str


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def hot_confirm_topk(
    *,
    db_path: str | Path,
    environment_compatibility_tag: str | None = None,
    output_dir: str | Path,
    runner_bin: str | Path,
    shape_id: str,
    problem_type_hash: str,
    screening_protocol_hash: str,
    validation_protocol_hash: str,
    hot_protocol: BenchmarkProtocol,
    top_k: int = 8,
    admission_deadline: float | None = None,
    runner_timeout_s: float = 300.0,
) -> list[HotConfirmationRecord]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    db = EvoTensileDB.connect(
        db_path,
        environment_compatibility_tag=environment_compatibility_tag,
    )
    summaries = db.rank_benchmarks(
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
    artifacts = load_artifact_mappings(
        db,
        problem_type_hash=problem_type_hash,
        shape_ids=[shape_id],
        candidate_hashes=hashes,
    )
    shape = shape_from_id(shape_id)
    records: list[HotConfirmationRecord] = []
    failures: list[HotConfirmationFailure] = []
    for screen_rank, candidate_hash in enumerate(hashes, 1):
        if admission_deadline is not None and time.monotonic() >= admission_deadline:
            break
        artifact = artifacts.get((shape_id, candidate_hash))
        if artifact is None:
            failures.append(
                {
                    "screen_rank": screen_rank,
                    "candidate_hash": candidate_hash,
                    "returncode": 0,
                    "timed_out": False,
                    "reason": "registered screening artifact is unavailable",
                }
            )
            continue
        candidate_dir = output / f"rank_{screen_rank:02d}_{candidate_hash}"
        run_output = run_structured_phase(
            mode="benchmark",
            run_dir=candidate_dir,
            pairs=[artifact.runnable_pair],
            shapes=[shape],
            protocol=hot_protocol,
            runner_bin=runner_bin,
            library_dir=artifact.library_dir,
            timeout_s=runner_timeout_s,
        )
        if run_output.timed_out:
            failures.append(
                {
                    "screen_rank": screen_rank,
                    "candidate_hash": candidate_hash,
                    "returncode": run_output.returncode,
                    "timed_out": True,
                    "reason": f"hot confirmation timed out after {runner_timeout_s} seconds",
                }
            )
            continue
        try:
            inserts = validate_benchmark_samples(
                run_output.samples,
                runnable_pairs=[artifact.runnable_pair],
                protocol=hot_protocol,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=hot_protocol.protocol_hash(),
                run_id=run_output.run_id,
                runner_returncode=run_output.returncode,
            )
            if len(inserts) != 1 or inserts[0].status != "ok":
                raise ValueError("hot confirmation did not return one complete positive event")
            times = list(inserts[0].samples_us)
            if len(times) != hot_protocol.num_benchmarks:
                raise ValueError("hot confirmation did not return a complete positive sample set")
        except (TypeError, ValueError) as exc:
            failures.append(
                {
                    "screen_rank": screen_rank,
                    "candidate_hash": candidate_hash,
                    "returncode": run_output.returncode,
                    "timed_out": False,
                    "reason": str(exc),
                }
            )
            continue
        gflops = [gflops_from_us(shape, time_us) for time_us in times]
        records.append(
            {
                "screen_rank": screen_rank,
                "candidate_hash": candidate_hash,
                "returncode": run_output.returncode,
                "duration_s": run_output.duration_s,
                "samples": len(times),
                "median_time_us": _median(times),
                "best_time_us": min(times),
                "median_gflops": _median(gflops),
                "best_gflops": max(gflops),
                "library_dir": str(artifact.library_dir),
                "command": run_output.command,
            }
        )
    records.sort(key=lambda record: (record["median_time_us"], record["candidate_hash"]))
    payload = {
        "protocol": {
            **hot_protocol.global_parameters(),
            **hot_protocol.runner_parameters(),
            "benchmark_protocol_hash": hot_protocol.protocol_hash(),
            "validation_disabled_by_num_elements": True,
            "validation_reused_from_screening_db": True,
        },
        "ranked": records,
        "failures": failures,
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
