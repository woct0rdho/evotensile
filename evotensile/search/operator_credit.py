import math
from collections.abc import Sequence
from dataclasses import dataclass

from evotensile.candidate import Shape
from evotensile.database import EvoTensileDB

ADAPTIVE_OPERATOR_ARMS = (
    "semantic-mutation",
    "de",
    "gomea-neighborhood",
    "gomea-mixing",
)


@dataclass(frozen=True)
class OperatorCredit:
    arm: str
    successes: int = 0
    failures: int = 0
    cumulative_log_speedup: float = 0.0

    @property
    def trials(self) -> int:
        return self.successes + self.failures

    @property
    def posterior_mean(self) -> float:
        return (self.successes + 1.0) / (self.trials + 2.0)

    def summary(self) -> dict[str, float | int | str]:
        return {
            "arm": self.arm,
            "successes": self.successes,
            "failures": self.failures,
            "trials": self.trials,
            "posterior_mean": self.posterior_mean,
            "cumulative_log_speedup": self.cumulative_log_speedup,
        }


def load_operator_credits(
    db: EvoTensileDB,
    *,
    problem_type_hash: str | None,
    benchmark_protocol_hash: str | None,
    shapes: Sequence[Shape] | None,
    min_improvement_fraction: float = 0.005,
) -> dict[str, OperatorCredit]:
    summaries = db.rank_evaluations(
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        min_samples=1,
        limit=None,
    )
    allowed_shape_ids = {shape.id for shape in shapes} if shapes is not None else None
    by_pair = {
        (summary.shape_id, summary.candidate_hash): summary
        for summary in summaries
        if allowed_shape_ids is None or summary.shape_id in allowed_shape_ids
    }
    candidates = {
        candidate.hash: candidate
        for candidate in db.get_candidates(sorted({candidate_hash for _, candidate_hash in by_pair}))
    }
    counts = {arm: [0, 0, 0.0] for arm in ADAPTIVE_OPERATOR_ARMS}
    for (shape_id, candidate_hash), summary in by_pair.items():
        candidate = candidates.get(candidate_hash)
        child_time = summary.median_time_us
        if candidate is None or candidate.source not in counts or child_time is None or child_time <= 0.0:
            continue
        parent_times = [
            parent_summary.median_time_us
            for parent_hash in candidate.parent_hashes
            if (parent_summary := by_pair.get((shape_id, parent_hash))) is not None
            and parent_summary.median_time_us is not None
            and parent_summary.median_time_us > 0.0
        ]
        if not parent_times:
            continue
        reference_time = min(parent_times)
        log_speedup = math.log(reference_time / child_time)
        success = child_time <= reference_time * (1.0 - max(0.0, min_improvement_fraction))
        bucket = counts[candidate.source]
        bucket[0 if success else 1] += 1
        bucket[2] += log_speedup
    return {
        arm: OperatorCredit(
            arm=arm,
            successes=int(values[0]),
            failures=int(values[1]),
            cumulative_log_speedup=float(values[2]),
        )
        for arm, values in counts.items()
    }


def allocate_operator_budget(
    total: int,
    credits: dict[str, OperatorCredit],
    *,
    minimum_per_arm: int = 1,
) -> dict[str, int]:
    arms = tuple(arm for arm in ADAPTIVE_OPERATOR_ARMS if arm in credits)
    allocation = {arm: 0 for arm in arms}
    if total <= 0 or not arms:
        return allocation
    minimum = max(0, minimum_per_arm)
    if total < minimum * len(arms):
        for arm in arms[:total]:
            allocation[arm] += 1
        return allocation
    for arm in arms:
        allocation[arm] = minimum
    remaining = total - minimum * len(arms)
    total_trials = sum(credits[arm].trials for arm in arms)
    scores = {}
    for arm in arms:
        credit = credits[arm]
        exploration = math.sqrt(2.0 * math.log(total_trials + 2.0) / (credit.trials + 1.0))
        scores[arm] = credit.posterior_mean + exploration
    score_sum = sum(scores.values())
    if score_sum <= 0.0:
        scores = {arm: 1.0 for arm in arms}
        score_sum = float(len(arms))
    exact = {arm: remaining * scores[arm] / score_sum for arm in arms}
    for arm in arms:
        allocation[arm] += int(math.floor(exact[arm]))
    assigned = sum(allocation.values())
    order = sorted(arms, key=lambda arm: (-(exact[arm] - math.floor(exact[arm])), arm))
    for arm in order[: total - assigned]:
        allocation[arm] += 1
    return allocation
