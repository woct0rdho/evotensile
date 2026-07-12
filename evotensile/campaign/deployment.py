import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.evaluator import EvaluationResult, PairEvaluationOutcome, PairEvaluator
from evotensile.candidate import Candidate, Shape
from evotensile.scheduling.models import EvidenceStage, PairRequest
from evotensile.search.pair_model import PairPrediction


class DeploymentSelectionPayload(TypedDict):
    tolerance_fraction: float
    shape_ids: list[str]
    assignments: dict[str, str]
    exact_winners: dict[str, str]
    confirmed_performance: dict[str, float]
    exact_winner_performance: dict[str, float]
    per_shape_loss_fraction: dict[str, float]
    candidate_coverage: dict[str, list[str]]
    generalist_coverage: dict[str, list[str]]
    specialist_shape_ids: list[str]
    uniform_mean_loss_fraction: float
    workload_weighted_mean_loss_fraction: float
    worst_shape_loss_fraction: float
    solution_count: int
    code_object_count: int
    code_object_count_conservative: bool
    shape_weights: dict[str, float]


@dataclass(frozen=True)
class FinalistConfirmationPolicy:
    relative_tolerance: float = 0.02
    minimum_close_probability: float = 0.10
    maximum_finalists_per_shape: int = 3
    stabilization_samples: int = 4
    min_samples: int = 10
    fallback_group_cost_s: float = 30.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.relative_tolerance < 1.0:
            raise ValueError("finalist relative tolerance must be in [0, 1)")
        if not 0.0 <= self.minimum_close_probability <= 1.0:
            raise ValueError("finalist close probability must be in [0, 1]")
        if self.maximum_finalists_per_shape <= 0 or self.stabilization_samples <= 0 or self.min_samples <= 0:
            raise ValueError("finalist counts and samples must be positive")
        if not math.isfinite(self.fallback_group_cost_s) or self.fallback_group_cost_s <= 0.0:
            raise ValueError("finalist fallback group cost must be finite and positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_tolerance": self.relative_tolerance,
            "minimum_close_probability": self.minimum_close_probability,
            "maximum_finalists_per_shape": self.maximum_finalists_per_shape,
            "stabilization_samples": self.stabilization_samples,
            "min_samples": self.min_samples,
            "fallback_group_cost_s": self.fallback_group_cost_s,
        }


@dataclass(frozen=True)
class ConfirmationFinalist:
    request: PairRequest
    close_probability: float
    predicted_performance: float | None
    incumbent: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "shape_id": self.request.shape.id,
            "candidate_hash": self.request.candidate.hash,
            "close_probability": self.close_probability,
            "predicted_performance": self.predicted_performance,
            "incumbent": self.incumbent,
            "priority": self.request.priority,
            "min_samples": self.request.min_samples,
        }


@dataclass(frozen=True)
class ConfirmationPlan:
    finalists: tuple[ConfirmationFinalist, ...]
    policy: FinalistConfirmationPolicy
    shape_weights: dict[str, float]

    @property
    def requests(self) -> tuple[PairRequest, ...]:
        return tuple(finalist.request for finalist in self.finalists)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy.to_dict(),
            "shape_weights": dict(sorted(self.shape_weights.items())),
            "finalists": [finalist.to_dict() for finalist in self.finalists],
            "pair_count": len(self.finalists),
            "candidate_count": len({finalist.request.candidate.hash for finalist in self.finalists}),
        }


@dataclass(frozen=True)
class ConfirmationRun:
    plan: ConfirmationPlan
    results: tuple[EvaluationResult, ...]
    admissions: tuple[dict[str, object], ...]
    stop_reason: str

    @property
    def outcomes(self) -> tuple[PairEvaluationOutcome, ...]:
        return tuple(outcome for result in self.results for outcome in result.outcomes)

    def to_dict(self) -> dict[str, object]:
        return {
            "plan": self.plan.to_dict(),
            "admissions": list(self.admissions),
            "stop_reason": self.stop_reason,
            "groups_completed": len(self.results),
            "known_pairs": sum(result.known_pairs for result in self.results),
            "unknown_pairs": sum(result.unknown_pairs for result in self.results),
            "outcomes": [
                {
                    "shape_id": outcome.request.shape.id,
                    "candidate_hash": outcome.request.candidate.hash,
                    "status": outcome.status,
                    "known": outcome.known,
                    "samples": outcome.samples,
                    "performance": outcome.performance,
                    "provenance": outcome.provenance,
                    "source_ref": outcome.source_ref,
                }
                for outcome in self.outcomes
            ],
        }


@dataclass(frozen=True)
class DeploymentSelection:
    tolerance_fraction: float
    shape_ids: tuple[str, ...]
    assignments: dict[str, str]
    exact_winners: dict[str, str]
    confirmed_performance: dict[str, float]
    exact_winner_performance: dict[str, float]
    per_shape_loss_fraction: dict[str, float]
    candidate_coverage: dict[str, tuple[str, ...]]
    generalist_coverage: dict[str, tuple[str, ...]]
    specialist_shape_ids: tuple[str, ...]
    uniform_mean_loss_fraction: float
    workload_weighted_mean_loss_fraction: float
    worst_shape_loss_fraction: float
    solution_count: int
    code_object_count: int
    code_object_count_conservative: bool
    shape_weights: dict[str, float]

    def __post_init__(self) -> None:
        shape_set = set(self.shape_ids)
        for mapping in (
            self.assignments,
            self.exact_winners,
            self.confirmed_performance,
            self.exact_winner_performance,
            self.per_shape_loss_fraction,
            self.shape_weights,
        ):
            if set(mapping) != shape_set:
                raise ValueError("deployment selection mappings must cover the exact shape set")
        if len(set(self.assignments.values())) != self.solution_count:
            raise ValueError("deployment solution count must match assigned candidates")

    @property
    def selected_candidate_hashes(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.assignments.values())))

    def to_dict(self) -> DeploymentSelectionPayload:
        return {
            "tolerance_fraction": self.tolerance_fraction,
            "shape_ids": list(self.shape_ids),
            "assignments": dict(sorted(self.assignments.items())),
            "exact_winners": dict(sorted(self.exact_winners.items())),
            "confirmed_performance": dict(sorted(self.confirmed_performance.items())),
            "exact_winner_performance": dict(sorted(self.exact_winner_performance.items())),
            "per_shape_loss_fraction": dict(sorted(self.per_shape_loss_fraction.items())),
            "candidate_coverage": {
                candidate_hash: list(shape_ids) for candidate_hash, shape_ids in sorted(self.candidate_coverage.items())
            },
            "generalist_coverage": {
                candidate_hash: list(shape_ids)
                for candidate_hash, shape_ids in sorted(self.generalist_coverage.items())
            },
            "specialist_shape_ids": list(self.specialist_shape_ids),
            "uniform_mean_loss_fraction": self.uniform_mean_loss_fraction,
            "workload_weighted_mean_loss_fraction": self.workload_weighted_mean_loss_fraction,
            "worst_shape_loss_fraction": self.worst_shape_loss_fraction,
            "solution_count": self.solution_count,
            "code_object_count": self.code_object_count,
            "code_object_count_conservative": self.code_object_count_conservative,
            "shape_weights": dict(sorted(self.shape_weights.items())),
        }

    @classmethod
    def from_dict(cls, payload: DeploymentSelectionPayload) -> "DeploymentSelection":
        shape_ids = tuple(payload["shape_ids"])
        return cls(
            tolerance_fraction=float(payload["tolerance_fraction"]),
            shape_ids=shape_ids,
            assignments=dict(payload["assignments"]),
            exact_winners=dict(payload["exact_winners"]),
            confirmed_performance=dict(payload["confirmed_performance"]),
            exact_winner_performance=dict(payload["exact_winner_performance"]),
            per_shape_loss_fraction=dict(payload["per_shape_loss_fraction"]),
            candidate_coverage={
                candidate_hash: tuple(shape_ids) for candidate_hash, shape_ids in payload["candidate_coverage"].items()
            },
            generalist_coverage={
                candidate_hash: tuple(shape_ids) for candidate_hash, shape_ids in payload["generalist_coverage"].items()
            },
            specialist_shape_ids=tuple(payload["specialist_shape_ids"]),
            uniform_mean_loss_fraction=float(payload["uniform_mean_loss_fraction"]),
            workload_weighted_mean_loss_fraction=float(payload["workload_weighted_mean_loss_fraction"]),
            worst_shape_loss_fraction=float(payload["worst_shape_loss_fraction"]),
            solution_count=int(payload["solution_count"]),
            code_object_count=int(payload["code_object_count"]),
            code_object_count_conservative=bool(payload["code_object_count_conservative"]),
            shape_weights=dict(payload["shape_weights"]),
        )


def plan_stabilization_finalists(
    predictions: Sequence[PairPrediction],
    *,
    candidates: Mapping[str, Candidate],
    shapes: Mapping[str, Shape],
    incumbent_candidates_by_shape: Mapping[str, str],
    shape_weights: Mapping[str, float] | None = None,
    policy: FinalistConfirmationPolicy | None = None,
) -> ConfirmationPlan:
    active_policy = policy or FinalistConfirmationPolicy()
    return _plan_finalists(
        predictions,
        candidates=candidates,
        shapes=shapes,
        incumbent_candidates_by_shape=incumbent_candidates_by_shape,
        shape_weights=shape_weights,
        policy=active_policy,
        evidence_stage=EvidenceStage.STABILIZATION,
        min_samples=active_policy.stabilization_samples,
    )


def plan_confirmation_finalists(
    predictions: Sequence[PairPrediction],
    *,
    candidates: Mapping[str, Candidate],
    shapes: Mapping[str, Shape],
    incumbent_candidates_by_shape: Mapping[str, str],
    shape_weights: Mapping[str, float] | None = None,
    policy: FinalistConfirmationPolicy | None = None,
) -> ConfirmationPlan:
    active_policy = policy or FinalistConfirmationPolicy()
    return _plan_finalists(
        predictions,
        candidates=candidates,
        shapes=shapes,
        incumbent_candidates_by_shape=incumbent_candidates_by_shape,
        shape_weights=shape_weights,
        policy=active_policy,
        evidence_stage=EvidenceStage.CONFIRMATION,
        min_samples=active_policy.min_samples,
    )


def _plan_finalists(
    predictions: Sequence[PairPrediction],
    *,
    candidates: Mapping[str, Candidate],
    shapes: Mapping[str, Shape],
    incumbent_candidates_by_shape: Mapping[str, str],
    shape_weights: Mapping[str, float] | None,
    policy: FinalistConfirmationPolicy,
    evidence_stage: EvidenceStage,
    min_samples: int,
) -> ConfirmationPlan:
    active_policy = policy
    if set(incumbent_candidates_by_shape) != set(shapes):
        raise ValueError("confirmation incumbents must cover the exact shape set")
    weights = _validated_weights(tuple(shapes), shape_weights)
    by_shape: dict[str, list[PairPrediction]] = {shape_id: [] for shape_id in shapes}
    for prediction in predictions:
        if prediction.shape_id not in shapes or prediction.candidate_hash not in candidates:
            continue
        by_shape[prediction.shape_id].append(prediction)
    finalists = []
    maximum_log_loss = -math.log1p(-active_policy.relative_tolerance)
    for shape_id, shape in shapes.items():
        shape_predictions = by_shape[shape_id]
        if not shape_predictions:
            raise ValueError(f"confirmation predictions are missing shape {shape_id}")
        close_probabilities = _posterior_close_probabilities(shape_predictions, maximum_log_loss)
        incumbent_hash = incumbent_candidates_by_shape[shape_id]
        if incumbent_hash not in candidates:
            raise ValueError(f"confirmation incumbent candidate is unavailable: {incumbent_hash}")
        ranked = sorted(
            shape_predictions,
            key=lambda prediction: (
                -close_probabilities[prediction.candidate_hash],
                -prediction.mean_normalized_log_performance,
                prediction.candidate_hash,
            ),
        )
        selected_hashes = [incumbent_hash]
        for prediction in ranked:
            if len(selected_hashes) >= active_policy.maximum_finalists_per_shape:
                break
            if prediction.candidate_hash in selected_hashes:
                continue
            if close_probabilities[prediction.candidate_hash] < active_policy.minimum_close_probability:
                continue
            selected_hashes.append(prediction.candidate_hash)
        prediction_by_candidate = {prediction.candidate_hash: prediction for prediction in shape_predictions}
        for candidate_hash in selected_hashes:
            prediction = prediction_by_candidate.get(candidate_hash)
            close_probability = 1.0 if candidate_hash == incumbent_hash else close_probabilities[candidate_hash]
            finalists.append(
                ConfirmationFinalist(
                    request=PairRequest(
                        candidates[candidate_hash],
                        shape,
                        evidence_stage=evidence_stage,
                        min_samples=min_samples,
                        priority=weights[shape_id] * close_probability,
                    ),
                    close_probability=close_probability,
                    predicted_performance=None if prediction is None else prediction.predicted_performance,
                    incumbent=candidate_hash == incumbent_hash,
                )
            )
    finalists.sort(
        key=lambda finalist: (
            -finalist.request.priority,
            finalist.request.candidate.hash,
            finalist.request.shape.id,
        )
    )
    return ConfirmationPlan(tuple(finalists), active_policy, weights)


def run_final_confirmation(
    evaluator: PairEvaluator,
    controller: CampaignControllerState,
    plan: ConfirmationPlan,
    *,
    predicted_group_cost_s: Mapping[str, float] | None = None,
    checkpoint: Callable[[CampaignControllerState], None] | None = None,
    now: Callable[[], float] = time.monotonic,
    charge_result_time: Callable[[EvaluationResult], None] | None = None,
) -> ConfirmationRun:
    if set(plan.shape_weights) != set(controller.shape_ids):
        raise ValueError("confirmation plan must match controller shapes")
    groups: dict[str, list[PairRequest]] = {}
    group_priority: dict[str, float] = {}
    for request in plan.requests:
        groups.setdefault(request.candidate.hash, []).append(request)
        group_priority[request.candidate.hash] = max(group_priority.get(request.candidate.hash, 0.0), request.priority)
    ordered_hashes = sorted(groups, key=lambda candidate_hash: (-group_priority[candidate_hash], candidate_hash))
    admissions = []
    results = []
    stop_reason = "completed"
    controller.transition("confirmation")
    for candidate_hash in ordered_hashes:
        predicted_cost_s = plan.policy.fallback_group_cost_s
        if predicted_group_cost_s is not None and candidate_hash in predicted_group_cost_s:
            predicted_cost_s = float(predicted_group_cost_s[candidate_hash])
        decision = controller.decide_admission(predicted_duration_s=predicted_cost_s, now=now())
        record = {
            **decision.to_dict(),
            "candidate_hash": candidate_hash,
            "request_pairs": [list(request.key) for request in groups[candidate_hash]],
        }
        admissions.append(record)
        controller.append_trace("confirmation_admission", record)
        if not decision.admitted:
            stop_reason = decision.reason
            break
        if checkpoint is not None:
            checkpoint(controller)
        requests = tuple(groups[candidate_hash])
        result = evaluator.evaluate(
            requests,
            artifact_shapes_by_candidate={candidate_hash: tuple(request.shape for request in requests)},
        )
        result.apply(controller)
        if charge_result_time is not None:
            charge_result_time(result)
        results.append(result)
        if checkpoint is not None:
            checkpoint(controller)
        if controller.overrun_s(now=now()) > 0.0:
            stop_reason = "overrun"
            break
    controller.append_trace(
        "confirmation_complete",
        {
            "stop_reason": stop_reason,
            "groups_completed": len(results),
            "known_pairs": sum(result.known_pairs for result in results),
            "unknown_pairs": sum(result.unknown_pairs for result in results),
        },
    )
    return ConfirmationRun(plan, tuple(results), tuple(admissions), stop_reason)


def select_deployment_solution_bank(
    outcomes: Sequence[PairEvaluationOutcome],
    *,
    shape_ids: Sequence[str],
    tolerance_fraction: float = 0.0,
    shape_weights: Mapping[str, float] | None = None,
    code_object_identity_by_candidate: Mapping[str, str] | None = None,
) -> DeploymentSelection:
    if not 0.0 <= tolerance_fraction < 1.0:
        raise ValueError("deployment tolerance must be in [0, 1)")
    ordered_shapes = tuple(str(shape_id) for shape_id in shape_ids)
    if not ordered_shapes or len(set(ordered_shapes)) != len(ordered_shapes):
        raise ValueError("deployment shapes must be non-empty and unique")
    weights = _validated_weights(ordered_shapes, shape_weights)
    performance_by_candidate: dict[str, dict[str, float]] = {}
    for outcome in outcomes:
        performance = outcome.performance
        if (
            outcome.request.shape.id in weights
            and outcome.request.evidence_stage == EvidenceStage.CONFIRMATION
            and outcome.known
            and outcome.disclosed
            and outcome.samples > 0
            and performance is not None
            and math.isfinite(performance)
            and performance > 0.0
        ):
            candidate_performance = performance_by_candidate.setdefault(outcome.request.candidate.hash, {})
            candidate_performance[outcome.request.shape.id] = max(
                candidate_performance.get(outcome.request.shape.id, 0.0), performance
            )
    exact_winners = {}
    exact_performance = {}
    for shape_id in ordered_shapes:
        options = [
            (values[shape_id], candidate_hash)
            for candidate_hash, values in performance_by_candidate.items()
            if shape_id in values
        ]
        if not options:
            raise ValueError(f"deployment selection lacks confirmed performance for {shape_id}")
        best_performance, best_candidate = min(options, key=lambda item: (-item[0], item[1]))
        exact_winners[shape_id] = best_candidate
        exact_performance[shape_id] = best_performance

    eligible = {
        candidate_hash: {
            shape_id
            for shape_id, performance in values.items()
            if performance >= exact_performance[shape_id] * (1.0 - tolerance_fraction)
        }
        for candidate_hash, values in performance_by_candidate.items()
    }
    eligible = {candidate_hash: shape_set for candidate_hash, shape_set in eligible.items() if shape_set}
    if tolerance_fraction == 0.0:
        selected = list(dict.fromkeys(exact_winners[shape_id] for shape_id in ordered_shapes))
    else:
        selected = _greedy_cover(
            ordered_shapes,
            eligible=eligible,
            performance_by_candidate=performance_by_candidate,
            exact_performance=exact_performance,
            weights=weights,
        )
    assignments = {}
    confirmed_performance = {}
    for shape_id in ordered_shapes:
        options = [
            (performance_by_candidate[candidate_hash][shape_id], candidate_hash)
            for candidate_hash in selected
            if shape_id in eligible[candidate_hash]
        ]
        if not options:
            raise ValueError(f"selected solution bank does not cover {shape_id}")
        performance, candidate_hash = min(options, key=lambda item: (-item[0], item[1]))
        assignments[shape_id] = candidate_hash
        confirmed_performance[shape_id] = performance
    candidate_coverage = {
        candidate_hash: tuple(shape_id for shape_id in ordered_shapes if assignments[shape_id] == candidate_hash)
        for candidate_hash in selected
    }
    candidate_coverage = {
        candidate_hash: shape_set for candidate_hash, shape_set in candidate_coverage.items() if shape_set
    }
    generalist_coverage = {
        candidate_hash: shape_set for candidate_hash, shape_set in candidate_coverage.items() if len(shape_set) > 1
    }
    specialist_shapes = tuple(shape_ids[0] for shape_ids in candidate_coverage.values() if len(shape_ids) == 1)
    losses = {
        shape_id: max(
            0.0,
            1.0 - confirmed_performance[shape_id] / exact_performance[shape_id],
        )
        for shape_id in ordered_shapes
    }
    total_weight = sum(weights.values())
    conservative_code_objects = code_object_identity_by_candidate is None
    if code_object_identity_by_candidate is not None:
        missing_code_objects = sorted(set(candidate_coverage) - set(code_object_identity_by_candidate))
        if missing_code_objects:
            raise ValueError(
                "deployment code-object identities are missing candidates: " + ", ".join(missing_code_objects)
            )
    code_object_identities = {
        candidate_hash
        if code_object_identity_by_candidate is None
        else code_object_identity_by_candidate[candidate_hash]
        for candidate_hash in candidate_coverage
    }
    return DeploymentSelection(
        tolerance_fraction=tolerance_fraction,
        shape_ids=ordered_shapes,
        assignments=assignments,
        exact_winners=exact_winners,
        confirmed_performance=confirmed_performance,
        exact_winner_performance=exact_performance,
        per_shape_loss_fraction=losses,
        candidate_coverage=candidate_coverage,
        generalist_coverage=generalist_coverage,
        specialist_shape_ids=tuple(sorted(specialist_shapes)),
        uniform_mean_loss_fraction=statistics.fmean(losses.values()),
        workload_weighted_mean_loss_fraction=sum(losses[shape_id] * weights[shape_id] for shape_id in ordered_shapes)
        / total_weight,
        worst_shape_loss_fraction=max(losses.values()),
        solution_count=len(candidate_coverage),
        code_object_count=len(code_object_identities),
        code_object_count_conservative=conservative_code_objects,
        shape_weights=weights,
    )


def _posterior_close_probabilities(
    predictions: Sequence[PairPrediction],
    maximum_log_loss: float,
) -> dict[str, float]:
    sample_count = min((len(prediction.posterior_samples) for prediction in predictions), default=0)
    if sample_count <= 0:
        leader = max(
            predictions,
            key=lambda prediction: (
                prediction.mean_normalized_log_performance,
                prediction.candidate_hash,
            ),
        )
        return {
            prediction.candidate_hash: (
                prediction.validity_probability if prediction.candidate_hash == leader.candidate_hash else 0.0
            )
            for prediction in predictions
        }
    counts = {prediction.candidate_hash: 0 for prediction in predictions}
    for index in range(sample_count):
        best = max(prediction.posterior_samples[index] for prediction in predictions)
        for prediction in predictions:
            if prediction.posterior_samples[index] >= best - maximum_log_loss:
                counts[prediction.candidate_hash] += 1
    return {
        prediction.candidate_hash: (counts[prediction.candidate_hash] / sample_count * prediction.validity_probability)
        for prediction in predictions
    }


def _greedy_cover(
    shape_ids: Sequence[str],
    *,
    eligible: Mapping[str, set[str]],
    performance_by_candidate: Mapping[str, Mapping[str, float]],
    exact_performance: Mapping[str, float],
    weights: Mapping[str, float],
) -> list[str]:
    uncovered = set(shape_ids)
    selected = []
    while uncovered:
        options = []
        for candidate_hash, covered_shapes in eligible.items():
            newly_covered = covered_shapes & uncovered
            if not newly_covered:
                continue
            mean_loss = statistics.fmean(
                max(
                    0.0,
                    1.0 - performance_by_candidate[candidate_hash][shape_id] / exact_performance[shape_id],
                )
                for shape_id in newly_covered
            )
            options.append(
                (
                    -len(newly_covered),
                    -sum(weights[shape_id] for shape_id in newly_covered),
                    mean_loss,
                    candidate_hash,
                )
            )
        if not options:
            raise ValueError(f"deployment candidates cannot cover shapes: {sorted(uncovered)}")
        candidate_hash = min(options)[3]
        selected.append(candidate_hash)
        uncovered -= eligible[candidate_hash]
    for candidate_hash in tuple(reversed(selected)):
        others = [value for value in selected if value != candidate_hash]
        if all(any(shape_id in eligible[value] for value in others) for shape_id in shape_ids):
            selected.remove(candidate_hash)
    return selected


def _validated_weights(
    shape_ids: Sequence[str],
    supplied: Mapping[str, float] | None,
) -> dict[str, float]:
    weights = {shape_id: 1.0 for shape_id in shape_ids}
    if supplied is not None:
        if set(supplied) != set(shape_ids):
            raise ValueError("shape weights must cover the exact deployment shape set")
        weights = {shape_id: float(supplied[shape_id]) for shape_id in shape_ids}
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights.values()):
        raise ValueError("shape weights must be finite and nonnegative")
    if sum(weights.values()) <= 0.0:
        raise ValueError("shape weights must have positive total mass")
    return weights
