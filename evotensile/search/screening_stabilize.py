import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from evotensile.adaptive_retime import CandidateTimingStats, load_timing_stats
from evotensile.artifacts import CandidateArtifact, load_candidate_artifacts
from evotensile.candidate import Shape
from evotensile.database import EvoTensileDB
from evotensile.protocol import BenchmarkProtocol
from evotensile.structured_runner import run_structured_phase, validate_benchmark_samples


@dataclass(frozen=True)
class ScreeningStabilizationPolicy:
    top_k: int = 4
    contender_epsilon_pct: float = 3.0
    confidence: float = 0.90
    min_samples: int = 6
    max_samples: int = 10
    sample_step: int = 2
    min_timed_duration_us: float = 100_000.0

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("stabilization top-k must be positive")
        if self.contender_epsilon_pct < 0.0:
            raise ValueError("stabilization contender epsilon must be non-negative")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("stabilization confidence must be in (0, 1)")
        if self.min_samples <= 0:
            raise ValueError("stabilization minimum samples must be positive")
        if self.max_samples < self.min_samples:
            raise ValueError("stabilization maximum samples must be at least the minimum")
        if self.sample_step <= 0:
            raise ValueError("stabilization sample step must be positive")
        if self.min_timed_duration_us < 0.0:
            raise ValueError("stabilization minimum timed duration must be non-negative")

    @property
    def epsilon_log(self) -> float:
        return math.log1p(self.contender_epsilon_pct / 100.0)

    @property
    def z_value(self) -> float:
        return statistics.NormalDist().inv_cdf(0.5 + self.confidence / 2.0)


@dataclass(frozen=True)
class ScreeningTopupRequest:
    candidate_hash: str
    rank: int
    current_samples: int
    target_samples: int
    median_time_us: float
    gap_pct: float
    ci_low_pct: float

    @property
    def remaining_samples(self) -> int:
        return self.target_samples - self.current_samples


@dataclass(frozen=True)
class ScreeningStabilizationResult:
    requests: tuple[ScreeningTopupRequest, ...]
    completed_candidate_hashes: tuple[str, ...]
    skipped_candidate_hashes: tuple[str, ...]
    runs: int
    added_samples: int
    duration_s: float
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "requests": [asdict(request) for request in self.requests],
            "completed_candidate_hashes": list(self.completed_candidate_hashes),
            "skipped_candidate_hashes": list(self.skipped_candidate_hashes),
            "runs": self.runs,
            "added_samples": self.added_samples,
            "duration_s": self.duration_s,
            "errors": list(self.errors),
        }


def _round_up(value: int, step: int) -> int:
    return int(math.ceil(value / step) * step)


def screening_topup_requests(
    stats: list[CandidateTimingStats],
    *,
    protocol: BenchmarkProtocol,
    policy: ScreeningStabilizationPolicy,
) -> list[ScreeningTopupRequest]:
    ranked = sorted(stats, key=lambda item: (item.score_log_time, item.candidate_hash))
    if not ranked:
        return []
    leader = ranked[0]
    launches_per_sample = protocol.enqueues_per_sync * protocol.syncs_per_benchmark
    requests: list[ScreeningTopupRequest] = []
    for rank, contender in enumerate(ranked[: policy.top_k], 1):
        gap_log = contender.score_log_time - leader.score_log_time
        combined_se = math.sqrt(contender.stderr_median_log**2 + leader.stderr_median_log**2)
        ci_low_log = gap_log - policy.z_value * combined_se
        if rank > 1 and ci_low_log > policy.epsilon_log:
            continue
        duration_target = math.ceil(
            policy.min_timed_duration_us / max(contender.median_time_us * launches_per_sample, 1.0)
        )
        target_samples = max(policy.min_samples, duration_target)
        target_samples = min(policy.max_samples, _round_up(target_samples, policy.sample_step))
        if contender.samples >= target_samples:
            continue
        requests.append(
            ScreeningTopupRequest(
                candidate_hash=contender.candidate_hash,
                rank=rank,
                current_samples=contender.samples,
                target_samples=target_samples,
                median_time_us=contender.median_time_us,
                gap_pct=math.expm1(gap_log) * 100.0,
                ci_low_pct=math.expm1(ci_low_log) * 100.0,
            )
        )
    return requests


def stabilize_screening_leaders(
    db: EvoTensileDB,
    *,
    shape: Shape,
    problem_type_hash: str,
    screening_protocol: BenchmarkProtocol,
    validation_protocol_hash: str,
    output_dir: str | Path,
    runner_bin: str | Path,
    policy: ScreeningStabilizationPolicy | None = None,
    deadline: float | None = None,
    runner_timeout_s: float = 300.0,
) -> ScreeningStabilizationResult:
    started = time.monotonic()
    active_policy = policy or ScreeningStabilizationPolicy()
    protocol_hash = screening_protocol.protocol_hash()
    stats_by_shape = load_timing_stats(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hashes=[protocol_hash],
        min_samples=1,
        shape_ids={shape.id},
    )
    requests = screening_topup_requests(
        stats_by_shape.get(shape.id, []),
        protocol=screening_protocol,
        policy=active_policy,
    )
    if not requests:
        return ScreeningStabilizationResult((), (), (), 0, 0, time.monotonic() - started, ())

    candidate_hashes = [request.candidate_hash for request in requests]
    validated = db.validated_cache_entries(
        problem_type_hash=problem_type_hash,
        validation_protocol_hash=validation_protocol_hash,
        shape_ids=[shape.id],
        candidate_hashes=candidate_hashes,
    )
    artifacts = load_candidate_artifacts(
        db,
        problem_type_hash=problem_type_hash,
        shape_ids=[shape.id],
        candidate_hashes=candidate_hashes,
    )
    grouped: dict[tuple[Path, int], list[tuple[ScreeningTopupRequest, CandidateArtifact]]] = {}
    skipped: list[str] = []
    for request in requests:
        key = (shape.id, request.candidate_hash)
        artifact = artifacts.get(key)
        if key not in validated or artifact is None:
            skipped.append(request.candidate_hash)
            continue
        group_key = (artifact.library_dir, request.remaining_samples)
        grouped.setdefault(group_key, []).append((request, artifact))

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    errors: list[str] = []
    added_samples = 0
    run_count = 0
    for group_index, ((library_dir, remaining), items) in enumerate(
        sorted(grouped.items(), key=lambda item: (-item[0][1], str(item[0][0])))
    ):
        if deadline is not None and time.monotonic() >= deadline:
            skipped.extend(request.candidate_hash for request, _ in items)
            continue
        pairs = [artifact.runnable_pair for _, artifact in items]
        run_protocol = screening_protocol.with_overrides(
            num_benchmarks=remaining,
            num_elements_to_validate=0,
        )
        timeout = runner_timeout_s
        if deadline is not None:
            timeout = min(timeout, max(1.0, deadline - time.monotonic()))
        output = run_structured_phase(
            mode="benchmark",
            run_dir=output_root / f"group_{group_index:02d}",
            pairs=pairs,
            shapes=[shape],
            protocol=run_protocol,
            runner_bin=runner_bin,
            library_dir=library_dir,
            timeout_s=timeout,
        )
        run_count += 1
        db.insert_run(
            output.run_id,
            yaml_path=None,
            output_dir=str(output_root / f"group_{group_index:02d}"),
            status="timeout" if output.timed_out else "ok" if output.ok else "failed",
            returncode=output.returncode,
            metadata_json=json.dumps(
                {
                    "command": output.command,
                    "duration_s": output.duration_s,
                    "mode": output.mode,
                    "pair_count": len(pairs),
                    "phase": "screening_stabilization",
                    "results_path": str(output.results_path),
                    "stderr_path": str(output.stderr_path),
                    "stdout_path": str(output.stdout_path),
                    "timed_out": output.timed_out,
                },
                sort_keys=True,
            ),
        )
        try:
            inserts = validate_benchmark_samples(
                output.samples,
                runnable_pairs=pairs,
                protocol=run_protocol,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=protocol_hash,
                run_id=output.run_id,
                runner_returncode=output.returncode,
            )
        except Exception as exc:
            errors.append(str(exc))
            skipped.extend(request.candidate_hash for request, _ in items)
            continue
        db.insert_evaluations(inserts)
        added_samples += sum(1 for insert in inserts if insert.status == "ok")
        completed.extend(request.candidate_hash for request, _ in items)

    return ScreeningStabilizationResult(
        requests=tuple(requests),
        completed_candidate_hashes=tuple(completed),
        skipped_candidate_hashes=tuple(dict.fromkeys(skipped)),
        runs=run_count,
        added_samples=added_samples,
        duration_s=time.monotonic() - started,
        errors=tuple(errors),
    )
