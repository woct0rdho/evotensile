import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from evotensile.candidate import Candidate, Shape
from evotensile.database import EvoTensileDB
from evotensile.search.cost_model import load_candidate_evaluation_costs
from evotensile.search.semantics import semantic_group_key, semantic_group_names

ADAPTIVE_OPERATOR_ARMS = (
    "semantic-mutation",
    "de",
    "gomea-neighborhood",
    "gomea-mixing",
)
DONOR_MODES = ("quality", "diverse", "random")


@dataclass(frozen=True)
class OperatorCredit:
    arm: str
    successes: int = 0
    failures: int = 0
    cumulative_log_speedup: float = 0.0
    cumulative_cost_s: float = 0.0

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
            "cumulative_cost_s": self.cumulative_cost_s,
        }


@dataclass(frozen=True)
class _ChildOutcome:
    candidate: Candidate
    success: bool
    log_speedup: float
    evaluation_cost_s: float


def _queried_child_outcomes(
    db: EvoTensileDB,
    *,
    problem_type_hash: str | None,
    benchmark_protocol_hash: str | None,
    shapes: Sequence[Shape] | None,
    min_improvement_fraction: float,
) -> list[_ChildOutcome]:
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
    candidate_costs = load_candidate_evaluation_costs(db)
    outcomes: list[_ChildOutcome] = []
    for (shape_id, candidate_hash), summary in by_pair.items():
        candidate = candidates.get(candidate_hash)
        child_time = summary.median_time_us
        if candidate is None or child_time is None or child_time <= 0.0:
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
        candidate_cost = candidate_costs.get(candidate_hash)
        outcomes.append(
            _ChildOutcome(
                candidate=candidate,
                success=child_time <= reference_time * (1.0 - max(0.0, min_improvement_fraction)),
                log_speedup=math.log(reference_time / child_time),
                evaluation_cost_s=0.0 if candidate_cost is None else candidate_cost.total_s,
            )
        )
    return outcomes


def _credits_from_outcomes(
    outcomes: Sequence[_ChildOutcome],
    *,
    keys: Sequence[str],
    classify: Callable[[Candidate], str | None],
) -> dict[str, OperatorCredit]:
    counts = {key: [0, 0, 0.0, 0.0] for key in keys}
    for outcome in outcomes:
        key = classify(outcome.candidate)
        if key not in counts:
            continue
        bucket = counts[key]
        bucket[0 if outcome.success else 1] += 1
        bucket[2] += outcome.log_speedup
        bucket[3] += outcome.evaluation_cost_s
    return {
        key: OperatorCredit(
            arm=key,
            successes=int(values[0]),
            failures=int(values[1]),
            cumulative_log_speedup=float(values[2]),
            cumulative_cost_s=float(values[3]),
        )
        for key, values in counts.items()
    }


def load_operator_credits(
    db: EvoTensileDB,
    *,
    problem_type_hash: str | None,
    benchmark_protocol_hash: str | None,
    shapes: Sequence[Shape] | None,
    min_improvement_fraction: float = 0.005,
) -> dict[str, OperatorCredit]:
    outcomes = _queried_child_outcomes(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        shapes=shapes,
        min_improvement_fraction=min_improvement_fraction,
    )
    return _credits_from_outcomes(
        outcomes,
        keys=ADAPTIVE_OPERATOR_ARMS,
        classify=lambda candidate: candidate.source,
    )


def load_semantic_group_credits(
    db: EvoTensileDB,
    *,
    problem_type_hash: str | None,
    benchmark_protocol_hash: str | None,
    shapes: Sequence[Shape] | None,
    min_improvement_fraction: float = 0.005,
) -> dict[str, OperatorCredit]:
    outcomes = _queried_child_outcomes(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        shapes=shapes,
        min_improvement_fraction=min_improvement_fraction,
    )
    keys = tuple(semantic_group_key(group) for group in semantic_group_names())

    def classify(candidate: Candidate) -> str | None:
        if candidate.source not in {"semantic-mutation", "gomea-neighborhood"}:
            return None
        value = candidate.proposal_metadata.get("semantic_group")
        return str(value) if value is not None else None

    return _credits_from_outcomes(outcomes, keys=keys, classify=classify)


def load_donor_mode_credits(
    db: EvoTensileDB,
    *,
    problem_type_hash: str | None,
    benchmark_protocol_hash: str | None,
    shapes: Sequence[Shape] | None,
    min_improvement_fraction: float = 0.005,
) -> dict[str, OperatorCredit]:
    outcomes = _queried_child_outcomes(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        shapes=shapes,
        min_improvement_fraction=min_improvement_fraction,
    )

    def classify(candidate: Candidate) -> str | None:
        if candidate.source != "gomea-mixing":
            return None
        value = candidate.proposal_metadata.get("donor_mode")
        return str(value) if value is not None else None

    return _credits_from_outcomes(outcomes, keys=DONOR_MODES, classify=classify)


def credit_ucb_scores(
    credits: dict[str, OperatorCredit],
    *,
    cost_aware: bool = False,
) -> dict[str, float]:
    if not credits:
        return {}
    total_trials = sum(credit.trials for credit in credits.values())
    scores = {
        key: credit.posterior_mean + math.sqrt(2.0 * math.log(total_trials + 2.0) / (credit.trials + 1.0))
        for key, credit in credits.items()
    }
    if not cost_aware:
        return scores
    average_costs = {key: (credit.cumulative_cost_s + 1.0) / (credit.trials + 1.0) for key, credit in credits.items()}
    reference_cost = sorted(average_costs.values())[len(average_costs) // 2]
    return {
        key: score * min(2.0, max(0.5, math.sqrt(reference_cost / max(average_costs[key], 1e-9))))
        for key, score in scores.items()
    }


def allocate_operator_budget(
    total: int,
    credits: dict[str, OperatorCredit],
    *,
    minimum_per_arm: int = 1,
    cost_aware: bool = False,
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
    scores = credit_ucb_scores(
        {arm: credits[arm] for arm in arms},
        cost_aware=cost_aware,
    )
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
