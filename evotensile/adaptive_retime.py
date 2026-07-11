import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .candidate import stable_hash
from .database import EvoTensileDB
from .utils import round_up

MEDIAN_SE_FACTOR = 1.2533141373155001


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute quantile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - position) + sorted_values[hi] * (position - lo)


@dataclass(frozen=True)
class CandidateTimingStats:
    shape_id: str
    candidate_hash: str
    samples: int
    median_time_us: float
    mean_log_time: float
    median_log_time: float
    stddev_log_time: float
    robust_sigma_log: float
    stderr_median_log: float
    mad_log: float
    iqr_log: float
    p10_time_us: float
    p90_time_us: float
    outlier_count: int

    @property
    def score_log_time(self) -> float:
        return self.median_log_time


@dataclass(frozen=True)
class PairDecision:
    candidate_hash: str
    rank: int
    gap_log: float
    gap_pct: float
    ci_low_log: float
    ci_high_log: float
    ci_low_pct: float
    ci_high_pct: float
    plausible: bool


@dataclass(frozen=True)
class ShapeRetimingDecision:
    shape_id: str
    status: str
    winner_hash: str | None
    retime_candidate_hashes: tuple[str, ...]
    target_samples: int
    source_candidates: int
    plausible_candidates: int
    truncated_plausible_candidates: int
    top_gap_pct: float | None
    top_gap_ci_low_pct: float | None
    top_gap_ci_high_pct: float | None
    pair_decisions: tuple[PairDecision, ...]

    @property
    def needs_retime(self) -> bool:
        return self.status == "needs_retime"


@dataclass(frozen=True)
class ShapeProbeDecision:
    shape_id: str
    reference_hash: str
    survivor_hashes: tuple[str, ...]
    screened_hashes: tuple[str, ...]


@dataclass(frozen=True)
class ProbePolicy:
    samples: int = 3
    initial_samples: int = 1
    max_slowdown_factor: float = 4.0
    confidence: float = 0.90
    noise_floor_pct: float = 5.0
    min_survivors: int = 8

    def __post_init__(self) -> None:
        if self.samples < 2:
            raise ValueError("probe samples must be at least 2")
        if not 1 <= self.initial_samples < self.samples:
            raise ValueError("probe initial samples must be positive and less than total samples")
        if self.max_slowdown_factor < 1.0:
            raise ValueError("probe max slowdown factor must be at least 1")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("probe confidence must be in (0, 1)")
        if self.noise_floor_pct < 0.0:
            raise ValueError("probe noise floor must be non-negative")
        if self.min_survivors <= 0:
            raise ValueError("probe minimum survivors must be positive")

    @property
    def policy_hash(self) -> str:
        return stable_hash(
            {
                "version": 1,
                "samples": self.samples,
                "initial_samples": self.initial_samples,
                "max_slowdown_factor": self.max_slowdown_factor,
                "confidence": self.confidence,
                "noise_floor_pct": self.noise_floor_pct,
                "min_survivors": self.min_survivors,
            },
            prefix="probe_policy_",
        )[:29]

    @property
    def max_slowdown_log(self) -> float:
        return math.log(self.max_slowdown_factor)

    @property
    def noise_floor_log(self) -> float:
        return math.log1p(self.noise_floor_pct / 100.0)

    @property
    def z_value(self) -> float:
        return statistics.NormalDist().inv_cdf(0.5 + self.confidence / 2.0)


@dataclass(frozen=True)
class AdaptivePolicy:
    epsilon_pct: float = 2.0
    confidence: float = 0.90
    min_retime_samples: int = 20
    max_retime_samples: int = 80
    sample_step: int = 10
    max_k: int = 8
    min_effect_pct: float = 0.5
    max_rounds: int = 4

    def __post_init__(self) -> None:
        if self.epsilon_pct < 0.0:
            raise ValueError("adaptive epsilon must be non-negative")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("adaptive confidence must be in (0, 1)")
        if self.min_retime_samples <= 0:
            raise ValueError("adaptive minimum samples must be positive")
        if self.max_retime_samples < self.min_retime_samples:
            raise ValueError("adaptive maximum samples must be at least the minimum")
        if self.sample_step <= 0:
            raise ValueError("adaptive sample step must be positive")
        if self.max_k <= 0:
            raise ValueError("adaptive max-k must be positive")
        if self.min_effect_pct <= 0.0:
            raise ValueError("adaptive minimum effect must be positive")
        if self.max_rounds < 0:
            raise ValueError("adaptive maximum rounds must be non-negative")

    @property
    def epsilon_log(self) -> float:
        return math.log1p(self.epsilon_pct / 100.0)

    @property
    def min_effect_log(self) -> float:
        return math.log1p(self.min_effect_pct / 100.0)

    @property
    def z_value(self) -> float:
        return statistics.NormalDist().inv_cdf(0.5 + self.confidence / 2.0)


def timing_stats_from_times(shape_id: str, candidate_hash: str, times_us: Sequence[float]) -> CandidateTimingStats:
    values = [float(value) for value in times_us if value is not None and math.isfinite(float(value)) and value > 0.0]
    if not values:
        raise ValueError(f"no positive timing samples for {shape_id}/{candidate_hash}")
    logs = [math.log(value) for value in values]
    sorted_logs = sorted(logs)
    sorted_times = sorted(values)
    median_log = statistics.median(sorted_logs)
    median_time = statistics.median(values)
    mean_log = statistics.fmean(logs)
    stddev_log = statistics.stdev(logs) if len(logs) >= 2 else 0.0
    mad_log = statistics.median(abs(value - median_log) for value in logs)
    q25 = _quantile(sorted_logs, 0.25)
    q75 = _quantile(sorted_logs, 0.75)
    iqr_log = q75 - q25
    sigma_candidates = [stddev_log]
    if mad_log > 0.0:
        sigma_candidates.append(1.4826 * mad_log)
    if iqr_log > 0.0:
        sigma_candidates.append(iqr_log / 1.349)
    robust_sigma = max(sigma_candidates) if sigma_candidates else 0.0
    stderr = MEDIAN_SE_FACTOR * robust_sigma / math.sqrt(len(values)) if len(values) else 0.0
    p10 = _quantile(sorted_times, 0.10)
    p90 = _quantile(sorted_times, 0.90)
    high_fence = q75 + 1.5 * iqr_log if iqr_log > 0.0 else float("inf")
    outliers = sum(value > high_fence for value in logs)
    return CandidateTimingStats(
        shape_id=shape_id,
        candidate_hash=candidate_hash,
        samples=len(values),
        median_time_us=median_time,
        mean_log_time=mean_log,
        median_log_time=median_log,
        stddev_log_time=stddev_log,
        robust_sigma_log=robust_sigma,
        stderr_median_log=stderr,
        mad_log=mad_log,
        iqr_log=iqr_log,
        p10_time_us=p10,
        p90_time_us=p90,
        outlier_count=outliers,
    )


def _gap_ci(
    left: CandidateTimingStats,
    right: CandidateTimingStats,
    *,
    z_value: float,
    sigma_floor_log: float = 0.0,
) -> tuple[float, float, float]:
    gap = left.score_log_time - right.score_log_time
    left_se = max(
        left.stderr_median_log,
        MEDIAN_SE_FACTOR * sigma_floor_log / math.sqrt(left.samples),
    )
    right_se = max(
        right.stderr_median_log,
        MEDIAN_SE_FACTOR * sigma_floor_log / math.sqrt(right.samples),
    )
    se = math.sqrt(left_se**2 + right_se**2)
    return gap, gap - z_value * se, gap + z_value * se


def _gap_pct(gap_log: float) -> float:
    return (math.exp(gap_log) - 1.0) * 100.0


def decide_shape_probe(
    shape_id: str,
    probe_stats: Sequence[CandidateTimingStats],
    *,
    policy: ProbePolicy,
    reference_stats: Sequence[CandidateTimingStats] = (),
) -> ShapeProbeDecision:
    ranked = sorted(probe_stats, key=lambda item: (item.score_log_time, item.candidate_hash))
    if not ranked:
        raise ValueError(f"no probe timing stats for {shape_id}")
    reference = min(
        (*ranked, *reference_stats),
        key=lambda item: (item.score_log_time, item.candidate_hash),
    )
    forced_survivors = {item.candidate_hash for item in ranked[: policy.min_survivors]}
    survivors: list[str] = []
    screened: list[str] = []
    for contender in ranked:
        _, ci_low, _ = _gap_ci(
            contender,
            reference,
            z_value=policy.z_value,
            sigma_floor_log=policy.noise_floor_log,
        )
        survives = ci_low <= policy.max_slowdown_log or contender.candidate_hash in forced_survivors
        (survivors if survives else screened).append(contender.candidate_hash)
    return ShapeProbeDecision(
        shape_id=shape_id,
        reference_hash=reference.candidate_hash,
        survivor_hashes=tuple(survivors),
        screened_hashes=tuple(screened),
    )


def _pairwise_equivalent(stats: Sequence[CandidateTimingStats], *, policy: AdaptivePolicy) -> bool:
    if len(stats) <= 1:
        return False
    for left_index, left in enumerate(stats):
        for right in stats[left_index + 1 :]:
            _, ci_low, ci_high = _gap_ci(left, right, z_value=policy.z_value)
            if ci_low < -policy.epsilon_log or ci_high > policy.epsilon_log:
                return False
    return True


def _required_samples(active: Sequence[CandidateTimingStats], *, policy: AdaptivePolicy) -> int:
    if len(active) <= 1:
        return 0
    best = active[0]
    required = policy.min_retime_samples
    z = policy.z_value
    for contender in active[1:]:
        gap = abs(contender.score_log_time - best.score_log_time)
        if gap > policy.epsilon_log:
            denominator = gap - policy.epsilon_log
        else:
            denominator = policy.epsilon_log - gap
        denominator = max(denominator, policy.min_effect_log)
        sigma_gap = MEDIAN_SE_FACTOR * math.sqrt(best.robust_sigma_log**2 + contender.robust_sigma_log**2)
        if sigma_gap <= 0.0:
            pair_required = policy.min_retime_samples
        else:
            pair_required = math.ceil((z * sigma_gap / denominator) ** 2)
        required = max(required, pair_required)
    required = min(required, policy.max_retime_samples)
    required = round_up(required, policy.sample_step)
    return max(policy.min_retime_samples, required)


def decide_shape_retime(
    shape_id: str,
    stats: Sequence[CandidateTimingStats],
    *,
    policy: AdaptivePolicy,
) -> ShapeRetimingDecision:
    ranked = sorted(stats, key=lambda item: (item.score_log_time, item.candidate_hash))
    if not ranked:
        return ShapeRetimingDecision(
            shape_id=shape_id,
            status="no_valid_candidates",
            winner_hash=None,
            retime_candidate_hashes=(),
            target_samples=0,
            source_candidates=0,
            plausible_candidates=0,
            truncated_plausible_candidates=0,
            top_gap_pct=None,
            top_gap_ci_low_pct=None,
            top_gap_ci_high_pct=None,
            pair_decisions=(),
        )
    if len(ranked) == 1:
        return ShapeRetimingDecision(
            shape_id=shape_id,
            status="resolved_winner",
            winner_hash=ranked[0].candidate_hash,
            retime_candidate_hashes=(),
            target_samples=0,
            source_candidates=1,
            plausible_candidates=1,
            truncated_plausible_candidates=0,
            top_gap_pct=None,
            top_gap_ci_low_pct=None,
            top_gap_ci_high_pct=None,
            pair_decisions=(),
        )

    best = ranked[0]
    pair_decisions: list[PairDecision] = []
    plausible = [best]
    for rank, contender in enumerate(ranked[1:], 2):
        gap, ci_low, ci_high = _gap_ci(contender, best, z_value=policy.z_value)
        is_plausible = ci_low <= policy.epsilon_log
        if is_plausible:
            plausible.append(contender)
        pair_decisions.append(
            PairDecision(
                candidate_hash=contender.candidate_hash,
                rank=rank,
                gap_log=gap,
                gap_pct=_gap_pct(gap),
                ci_low_log=ci_low,
                ci_high_log=ci_high,
                ci_low_pct=_gap_pct(ci_low),
                ci_high_pct=_gap_pct(ci_high),
                plausible=is_plausible,
            )
        )

    top_pair = pair_decisions[0]
    if len(plausible) == 1:
        return ShapeRetimingDecision(
            shape_id=shape_id,
            status="resolved_winner",
            winner_hash=best.candidate_hash,
            retime_candidate_hashes=(),
            target_samples=0,
            source_candidates=len(ranked),
            plausible_candidates=1,
            truncated_plausible_candidates=0,
            top_gap_pct=top_pair.gap_pct,
            top_gap_ci_low_pct=top_pair.ci_low_pct,
            top_gap_ci_high_pct=top_pair.ci_high_pct,
            pair_decisions=tuple(pair_decisions),
        )

    equivalence_candidates = [
        item for item in plausible if abs(item.score_log_time - best.score_log_time) <= policy.epsilon_log
    ]
    if len(equivalence_candidates) == len(plausible) and _pairwise_equivalent(equivalence_candidates, policy=policy):
        return ShapeRetimingDecision(
            shape_id=shape_id,
            status="resolved_equivalent",
            winner_hash=best.candidate_hash,
            retime_candidate_hashes=(),
            target_samples=0,
            source_candidates=len(ranked),
            plausible_candidates=len(plausible),
            truncated_plausible_candidates=0,
            top_gap_pct=top_pair.gap_pct,
            top_gap_ci_low_pct=top_pair.ci_low_pct,
            top_gap_ci_high_pct=top_pair.ci_high_pct,
            pair_decisions=tuple(pair_decisions),
        )

    active = plausible[: max(2, policy.max_k)]
    target_samples = _required_samples(active, policy=policy)
    return ShapeRetimingDecision(
        shape_id=shape_id,
        status="needs_retime",
        winner_hash=best.candidate_hash,
        retime_candidate_hashes=tuple(item.candidate_hash for item in active),
        target_samples=target_samples,
        source_candidates=len(ranked),
        plausible_candidates=len(plausible),
        truncated_plausible_candidates=max(0, len(plausible) - len(active)),
        top_gap_pct=top_pair.gap_pct,
        top_gap_ci_low_pct=top_pair.ci_low_pct,
        top_gap_ci_high_pct=top_pair.ci_high_pct,
        pair_decisions=tuple(pair_decisions),
    )


def load_timing_stats(
    db: EvoTensileDB,
    *,
    problem_type_hash: str,
    benchmark_protocol_hashes: Iterable[str] | None = None,
    min_samples: int = 1,
    shape_ids: set[str] | None = None,
    candidate_hashes: set[str] | None = None,
) -> dict[str, list[CandidateTimingStats]]:
    clauses = [
        "pt.problem_type_hash = ?",
        "be.status = 'ok'",
        "EXISTS (SELECT 1 FROM validations AS v "
        "WHERE v.validation_namespace_id = be.validation_namespace_id "
        "AND v.shape_key = be.shape_key AND v.candidate_id = be.candidate_id AND v.status = 'passed' "
        "AND NOT EXISTS (SELECT 1 FROM validations AS newer "
        "WHERE newer.validation_namespace_id = v.validation_namespace_id "
        "AND newer.shape_key = v.shape_key AND newer.candidate_id = v.candidate_id "
        "AND (newer.created_at > v.created_at OR "
        "(newer.created_at = v.created_at AND newer.validation_id > v.validation_id))))",
    ]
    params: list[str] = [problem_type_hash]
    protocol_hashes = list(benchmark_protocol_hashes or [])
    if protocol_hashes:
        placeholders = ",".join("?" for _ in protocol_hashes)
        clauses.append(f"bp.benchmark_protocol_hash IN ({placeholders})")
        params.extend(protocol_hashes)
    if shape_ids:
        placeholders = ",".join("?" for _ in shape_ids)
        clauses.append(f"s.shape_id IN ({placeholders})")
        params.extend(sorted(shape_ids))
    if candidate_hashes:
        placeholders = ",".join("?" for _ in candidate_hashes)
        clauses.append(f"c.candidate_hash IN ({placeholders})")
        params.extend(sorted(candidate_hashes))
    query = f"""
        SELECT s.shape_id, c.candidate_hash, bs.time_us
        FROM benchmark_events AS be
        JOIN benchmark_samples AS bs USING (event_id)
        JOIN benchmark_namespaces AS bn USING (benchmark_namespace_id)
        JOIN problem_types AS pt USING (problem_type_id)
        JOIN benchmark_protocols AS bp USING (benchmark_protocol_id)
        JOIN shapes AS s USING (shape_key)
        JOIN candidates AS c USING (candidate_id)
        WHERE {" AND ".join(clauses)}
        ORDER BY s.shape_id, c.candidate_hash, be.event_id, bs.sample_index
    """
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    with db.connection() as con:
        rows = con.execute(query, params).fetchall()
    for row in rows:
        grouped[(row["shape_id"], row["candidate_hash"])].append(float(row["time_us"]))

    by_shape: dict[str, list[CandidateTimingStats]] = defaultdict(list)
    for (shape_id, candidate_hash), samples in grouped.items():
        if len(samples) < min_samples:
            continue
        by_shape[shape_id].append(timing_stats_from_times(shape_id, candidate_hash, samples))
    return dict(by_shape)


def decide_retime_by_shape(
    stats_by_shape: dict[str, list[CandidateTimingStats]], *, policy: AdaptivePolicy
) -> dict[str, ShapeRetimingDecision]:
    return {
        shape_id: decide_shape_retime(shape_id, stats, policy=policy)
        for shape_id, stats in sorted(stats_by_shape.items())
    }
