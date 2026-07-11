import math
import statistics
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from evotensile.adaptive_retime import MEDIAN_SE_FACTOR, CandidateTimingStats, load_timing_stats
from evotensile.artifacts import CandidateArtifact, load_artifact_mappings
from evotensile.candidate import Shape
from evotensile.database import EvoTensileDB
from evotensile.protocol import BenchmarkProtocol
from evotensile.structured_runner import run_structured_phase, validate_benchmark_samples
from evotensile.utils import round_up


@dataclass(frozen=True)
class ScreeningStabilizationPolicy:
    top_k: int = 4
    contender_epsilon_pct: float = 3.0
    confidence: float = 0.90
    min_samples: int = 8
    max_samples: int = 24
    sample_step: int = 2
    min_launches: int = 8
    timer_resolution_us: float = 1.0
    min_timer_ticks: int = 100
    uncertainty_half_width_pct: float = 10.0
    noise_floor_pct: float = 1.0
    max_pairs_per_run: int = 16
    max_runner_duration_s: float = 30.0

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
        if self.min_launches <= 0:
            raise ValueError("stabilization minimum launches must be positive")
        if self.timer_resolution_us <= 0.0:
            raise ValueError("stabilization timer resolution must be positive")
        if self.min_timer_ticks < 0:
            raise ValueError("stabilization minimum timer ticks must be non-negative")
        if self.uncertainty_half_width_pct < 0.0:
            raise ValueError("stabilization uncertainty half-width must be non-negative")
        if self.noise_floor_pct < 0.0:
            raise ValueError("stabilization noise floor must be non-negative")
        if self.max_pairs_per_run <= 0:
            raise ValueError("stabilization maximum pairs per run must be positive")
        if self.max_runner_duration_s <= 0.0:
            raise ValueError("stabilization runner-duration budget must be positive")

    @property
    def epsilon_log(self) -> float:
        return math.log1p(self.contender_epsilon_pct / 100.0)

    @property
    def uncertainty_half_width_log(self) -> float:
        return math.log1p(self.uncertainty_half_width_pct / 100.0)

    @property
    def noise_floor_log(self) -> float:
        return math.log1p(self.noise_floor_pct / 100.0)

    @property
    def z_value(self) -> float:
        return statistics.NormalDist().inv_cdf(0.5 + self.confidence / 2.0)


@dataclass(frozen=True)
class ScreeningPair:
    shape_id: str
    candidate_hash: str


@dataclass(frozen=True)
class ScreeningFinalistPlan:
    shape_id: str
    cluster_id: str
    candidate_hash: str
    rank: int
    current_samples: int
    target_samples: int
    uncapped_target_samples: int
    median_time_us: float
    gap_pct: float
    ci_low_pct: float
    required_sample_target: int
    required_launch_target: int
    required_timer_target: int
    required_uncertainty_target: int
    capped_criteria: tuple[str, ...]
    queue_index: int | None = None

    @property
    def pair(self) -> ScreeningPair:
        return ScreeningPair(self.shape_id, self.candidate_hash)

    @property
    def remaining_samples(self) -> int:
        return max(0, self.target_samples - self.current_samples)

    @property
    def needs_topup(self) -> bool:
        return self.remaining_samples > 0


@dataclass(frozen=True)
class ScreeningStabilizationPlan:
    finalists: tuple[ScreeningFinalistPlan, ...]
    requests: tuple[ScreeningFinalistPlan, ...]
    shape_queues: int
    cluster_queues: int


@dataclass(frozen=True)
class ScreeningSkippedPair:
    shape_id: str
    candidate_hash: str
    reason: str


@dataclass(frozen=True)
class ScreeningStabilizationResult:
    plan: ScreeningStabilizationPlan
    completed_pairs: tuple[ScreeningPair, ...]
    skipped_pairs: tuple[ScreeningSkippedPair, ...]
    runs: int
    added_samples: int
    runner_duration_s: float
    duration_s: float
    runner_budget_exhausted: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "plan": asdict(self.plan),
            "completed_pairs": [asdict(pair) for pair in self.completed_pairs],
            "skipped_pairs": [asdict(pair) for pair in self.skipped_pairs],
            "runs": self.runs,
            "added_samples": self.added_samples,
            "runner_duration_s": self.runner_duration_s,
            "duration_s": self.duration_s,
            "runner_budget_exhausted": self.runner_budget_exhausted,
            "errors": list(self.errors),
        }


def _required_uncertainty_samples(
    stats: CandidateTimingStats,
    *,
    policy: ScreeningStabilizationPolicy,
) -> int:
    if policy.uncertainty_half_width_pct == 0.0:
        return 0
    sigma = max(stats.robust_sigma_log, policy.noise_floor_log)
    return math.ceil((policy.z_value * MEDIAN_SE_FACTOR * sigma / policy.uncertainty_half_width_log) ** 2)


def _shape_finalist_plans(
    stats: Sequence[CandidateTimingStats],
    *,
    cluster_id: str,
    protocol: BenchmarkProtocol,
    policy: ScreeningStabilizationPolicy,
) -> list[ScreeningFinalistPlan]:
    ranked = sorted(stats, key=lambda item: (item.score_log_time, item.candidate_hash))
    if not ranked:
        return []
    leader = ranked[0]
    launches_per_sample = protocol.enqueues_per_sync * protocol.syncs_per_benchmark
    plans: list[ScreeningFinalistPlan] = []
    for rank, contender in enumerate(ranked[: policy.top_k], 1):
        gap_log = contender.score_log_time - leader.score_log_time
        combined_se = math.sqrt(contender.stderr_median_log**2 + leader.stderr_median_log**2)
        ci_low_log = gap_log - policy.z_value * combined_se
        if rank > 1 and ci_low_log > policy.epsilon_log:
            continue
        required_sample_target = policy.min_samples
        required_launch_target = math.ceil(policy.min_launches / launches_per_sample)
        required_timer_target = math.ceil(
            policy.timer_resolution_us
            * policy.min_timer_ticks
            / max(contender.median_time_us * launches_per_sample, 1e-12)
        )
        required_uncertainty_target = _required_uncertainty_samples(contender, policy=policy)
        uncapped_target = max(
            required_sample_target,
            required_launch_target,
            required_timer_target,
            required_uncertainty_target,
        )
        rounded_target = round_up(uncapped_target, policy.sample_step)
        target_samples = min(policy.max_samples, rounded_target)
        required_targets = {
            "samples": required_sample_target,
            "launches": required_launch_target,
            "timer_resolution": required_timer_target,
            "uncertainty": required_uncertainty_target,
        }
        capped_criteria = tuple(
            name
            for name, target in required_targets.items()
            if target > policy.max_samples and contender.samples < target
        )
        plans.append(
            ScreeningFinalistPlan(
                shape_id=contender.shape_id,
                cluster_id=cluster_id,
                candidate_hash=contender.candidate_hash,
                rank=rank,
                current_samples=contender.samples,
                target_samples=target_samples,
                uncapped_target_samples=rounded_target,
                median_time_us=contender.median_time_us,
                gap_pct=math.expm1(gap_log) * 100.0,
                ci_low_pct=math.expm1(ci_low_log) * 100.0,
                required_sample_target=required_sample_target,
                required_launch_target=required_launch_target,
                required_timer_target=required_timer_target,
                required_uncertainty_target=required_uncertainty_target,
                capped_criteria=capped_criteria,
            )
        )
    return plans


def _fair_request_order(
    requests_by_shape: Mapping[str, Sequence[ScreeningFinalistPlan]],
    cluster_by_shape: Mapping[str, str],
) -> list[ScreeningFinalistPlan]:
    shape_queues = {shape_id: deque(requests) for shape_id, requests in requests_by_shape.items() if requests}
    cluster_shapes: dict[str, deque[str]] = {}
    for shape_id in sorted(shape_queues):
        cluster_shapes.setdefault(cluster_by_shape[shape_id], deque()).append(shape_id)
    ordered: list[ScreeningFinalistPlan] = []
    active_clusters = deque(sorted(cluster_shapes))
    while active_clusters:
        cluster_id = active_clusters.popleft()
        shapes = cluster_shapes[cluster_id]
        shape_id = shapes.popleft()
        ordered.append(shape_queues[shape_id].popleft())
        if shape_queues[shape_id]:
            shapes.append(shape_id)
        if shapes:
            active_clusters.append(cluster_id)
    return [replace(request, queue_index=index) for index, request in enumerate(ordered)]


def plan_screening_stabilization(
    stats_by_shape: Mapping[str, Sequence[CandidateTimingStats]],
    *,
    shapes: Sequence[Shape],
    protocol: BenchmarkProtocol,
    policy: ScreeningStabilizationPolicy,
    shape_clusters: Mapping[str, str] | None = None,
) -> ScreeningStabilizationPlan:
    shape_by_id = {shape.id: shape for shape in shapes}
    if not shape_by_id:
        raise ValueError("stabilization requires at least one shape")
    if len(shape_by_id) != len(shapes):
        raise ValueError("stabilization shapes must be unique")
    if shape_clusters is None:
        cluster_by_shape = {shape_id: shape_id for shape_id in shape_by_id}
    else:
        if set(shape_clusters) != set(shape_by_id):
            raise ValueError("stabilization cluster mapping must cover exactly the requested shapes")
        if any(not cluster_id for cluster_id in shape_clusters.values()):
            raise ValueError("stabilization cluster IDs must be non-empty")
        cluster_by_shape = dict(shape_clusters)

    finalists: list[ScreeningFinalistPlan] = []
    requests_by_shape: dict[str, list[ScreeningFinalistPlan]] = {}
    for shape_id in sorted(shape_by_id):
        shape_plans = _shape_finalist_plans(
            stats_by_shape.get(shape_id, ()),
            cluster_id=cluster_by_shape[shape_id],
            protocol=protocol,
            policy=policy,
        )
        finalists.extend(shape_plans)
        requests_by_shape[shape_id] = [plan for plan in shape_plans if plan.needs_topup]
    ordered_requests = _fair_request_order(requests_by_shape, cluster_by_shape)
    queue_index_by_pair = {
        (request.shape_id, request.candidate_hash): request.queue_index for request in ordered_requests
    }
    finalists = [
        replace(plan, queue_index=queue_index_by_pair.get((plan.shape_id, plan.candidate_hash))) for plan in finalists
    ]
    return ScreeningStabilizationPlan(
        finalists=tuple(finalists),
        requests=tuple(ordered_requests),
        shape_queues=sum(bool(requests) for requests in requests_by_shape.values()),
        cluster_queues=len({request.cluster_id for request in ordered_requests}),
    )


def _execution_groups(
    requests: Sequence[tuple[ScreeningFinalistPlan, CandidateArtifact]],
    *,
    max_pairs_per_run: int,
) -> list[list[tuple[ScreeningFinalistPlan, CandidateArtifact]]]:
    groups: list[list[tuple[ScreeningFinalistPlan, CandidateArtifact]]] = []
    for item in requests:
        item_key = (item[1].library_dir, item[0].remaining_samples)
        if groups:
            previous = groups[-1]
            previous_key = (previous[0][1].library_dir, previous[0][0].remaining_samples)
            if item_key == previous_key and len(previous) < max_pairs_per_run:
                previous.append(item)
                continue
        groups.append([item])
    return groups


def stabilize_screening_leaders(
    db: EvoTensileDB,
    *,
    shapes: Sequence[Shape],
    problem_type_hash: str,
    screening_protocol: BenchmarkProtocol,
    validation_protocol_hash: str,
    output_dir: str | Path,
    runner_bin: str | Path,
    policy: ScreeningStabilizationPolicy,
    runner_timeout_s: float,
    shape_clusters: Mapping[str, str] | None = None,
    admission_deadline: float | None = None,
) -> ScreeningStabilizationResult:
    started = time.monotonic()
    active_policy = policy
    shape_by_id = {shape.id: shape for shape in shapes}
    if not shape_by_id:
        raise ValueError("stabilization requires at least one shape")
    if len(shape_by_id) != len(shapes):
        raise ValueError("stabilization shapes must be unique")
    protocol_hash = screening_protocol.protocol_hash()
    stats_by_shape = load_timing_stats(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hashes=[protocol_hash],
        min_samples=1,
        shape_ids=set(shape_by_id),
    )
    plan = plan_screening_stabilization(
        stats_by_shape,
        shapes=shapes,
        protocol=screening_protocol,
        policy=active_policy,
        shape_clusters=shape_clusters,
    )
    if not plan.requests:
        return ScreeningStabilizationResult(plan, (), (), 0, 0, 0.0, time.monotonic() - started, False, ())

    candidate_hashes = list(dict.fromkeys(request.candidate_hash for request in plan.requests))
    shape_ids = list(shape_by_id)
    validated = db.validated_cache_entries(
        problem_type_hash=problem_type_hash,
        validation_protocol_hash=validation_protocol_hash,
        shape_ids=shape_ids,
        candidate_hashes=candidate_hashes,
    )
    artifacts = load_artifact_mappings(
        db,
        problem_type_hash=problem_type_hash,
        shape_ids=shape_ids,
        candidate_hashes=candidate_hashes,
    )
    eligible: list[tuple[ScreeningFinalistPlan, CandidateArtifact]] = []
    skipped: list[ScreeningSkippedPair] = []
    for request in plan.requests:
        key = (request.shape_id, request.candidate_hash)
        artifact = artifacts.get(key)
        if key not in validated:
            skipped.append(ScreeningSkippedPair(*key, "missing_compatible_validation"))
        elif artifact is None:
            skipped.append(ScreeningSkippedPair(*key, "missing_compatible_artifact"))
        else:
            eligible.append((request, artifact))

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    completed: list[ScreeningPair] = []
    errors: list[str] = []
    added_samples = 0
    run_count = 0
    runner_duration_s = 0.0
    runner_budget_exhausted = False
    groups = _execution_groups(eligible, max_pairs_per_run=active_policy.max_pairs_per_run)
    for group_index, items in enumerate(groups):
        if admission_deadline is not None and time.monotonic() >= admission_deadline:
            skipped.extend(
                ScreeningSkippedPair(request.shape_id, request.candidate_hash, "admission_deadline")
                for request, _ in items
            )
            continue
        if runner_duration_s >= active_policy.max_runner_duration_s:
            runner_budget_exhausted = True
            skipped.extend(
                ScreeningSkippedPair(request.shape_id, request.candidate_hash, "runner_duration_budget")
                for request, _ in items
            )
            continue
        remaining = items[0][0].remaining_samples
        pairs = [artifact.runnable_pair for _, artifact in items]
        run_protocol = screening_protocol.with_overrides(
            num_benchmarks=remaining,
            num_elements_to_validate=0,
        )
        run_dir = output_root / f"group_{group_index:04d}"
        output = run_structured_phase(
            mode="benchmark",
            run_dir=run_dir,
            pairs=pairs,
            shapes=[shape_by_id[pair.shape_id] for pair in pairs],
            protocol=run_protocol,
            runner_bin=runner_bin,
            library_dir=items[0][1].library_dir,
            timeout_s=runner_timeout_s,
        )
        run_count += 1
        runner_duration_s += output.duration_s
        db.insert_run(
            output.run_id,
            phase="screening",
            status="timeout" if output.timed_out else "ok" if output.ok else "failed",
            duration_s=output.duration_s,
            returncode=output.returncode,
            candidate_hashes=[pair.candidate_hash for pair in pairs],
        )
        try:
            inserts = validate_benchmark_samples(
                output.samples,
                runnable_pairs=pairs,
                protocol=run_protocol,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=protocol_hash,
                run_id=output.run_id,
                validation_protocol_hash=validation_protocol_hash,
                runner_returncode=output.returncode,
            )
        except Exception as exc:
            errors.append(str(exc))
            skipped.extend(
                ScreeningSkippedPair(request.shape_id, request.candidate_hash, "result_ingest_failed")
                for request, _ in items
            )
            continue
        db.insert_benchmark_events(inserts)
        added_samples += sum(len(insert.samples_us) for insert in inserts if insert.status == "ok")
        completed.extend(request.pair for request, _ in items)

    runner_budget_exhausted = runner_budget_exhausted or (runner_duration_s >= active_policy.max_runner_duration_s)
    return ScreeningStabilizationResult(
        plan=plan,
        completed_pairs=tuple(completed),
        skipped_pairs=tuple(skipped),
        runs=run_count,
        added_samples=added_samples,
        runner_duration_s=runner_duration_s,
        duration_s=time.monotonic() - started,
        runner_budget_exhausted=runner_budget_exhausted,
        errors=tuple(errors),
    )
