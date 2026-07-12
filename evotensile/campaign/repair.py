import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evotensile.campaign.acquisition import (
    AcquisitionPlan,
    BundleAcquisitionPolicy,
    BundleCostModel,
    plan_candidate_bundles,
)
from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.evaluator import PairEvaluationOutcome
from evotensile.candidate import Candidate, Shape
from evotensile.search.local_search import semantic_mutation_candidates
from evotensile.search.pair_model import PairPrediction
from evotensile.search.shape_clustering import ShapeClustering, shape_descriptor_distances


@dataclass(frozen=True)
class RepairPolicy:
    neighbor_count: int = 8
    neighbor_quantile: float = 0.75
    cluster_quantile: float = 0.75
    uncertainty_weight: float = 0.25
    minimum_deficit_fraction: float = 0.05
    maximum_deficit_fraction: float = 0.30
    useful_close_fraction: float = 0.5
    minimum_close_probability: float = 0.10
    neighbor_candidates_per_shape: int = 3
    cluster_candidates: int = 4
    mutation_candidates_per_shape: int = 4
    mutation_max_changed_genes: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        counts = (
            self.neighbor_count,
            self.neighbor_candidates_per_shape,
            self.cluster_candidates,
            self.mutation_candidates_per_shape,
            self.mutation_max_changed_genes,
        )
        if any(value < 0 for value in counts):
            raise ValueError("repair counts must be nonnegative")
        fractions = (
            self.neighbor_quantile,
            self.cluster_quantile,
            self.uncertainty_weight,
            self.minimum_deficit_fraction,
            self.maximum_deficit_fraction,
            self.useful_close_fraction,
            self.minimum_close_probability,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in fractions):
            raise ValueError("repair fractions must be finite and nonnegative")
        if self.neighbor_quantile > 1.0 or self.cluster_quantile > 1.0:
            raise ValueError("repair quantiles must not exceed one")
        if self.useful_close_fraction > 1.0 or self.minimum_close_probability > 1.0:
            raise ValueError("repair probabilities and close fractions must not exceed one")
        if self.maximum_deficit_fraction < self.minimum_deficit_fraction:
            raise ValueError("maximum repair deficit must cover the minimum deficit")


@dataclass(frozen=True)
class ShapeRepairDeficit:
    shape_id: str
    incumbent_candidate_hash: str
    incumbent_performance: float
    reference_target: float | None
    neighbor_target: float | None
    cluster_target: float | None
    uncertainty_log: float
    evidence_target: float
    raw_deficit_log: float
    capped_deficit_log: float

    @property
    def raw_deficit_fraction(self) -> float:
        return math.expm1(self.raw_deficit_log)

    @property
    def capped_deficit_fraction(self) -> float:
        return math.expm1(self.capped_deficit_log)

    def useful_target(self, fraction: float) -> float:
        return self.incumbent_performance * math.exp(self.capped_deficit_log * fraction)

    def to_dict(self) -> dict[str, object]:
        return {
            "shape_id": self.shape_id,
            "incumbent_candidate_hash": self.incumbent_candidate_hash,
            "incumbent_performance": self.incumbent_performance,
            "reference_target": self.reference_target,
            "neighbor_target": self.neighbor_target,
            "cluster_target": self.cluster_target,
            "uncertainty_log": self.uncertainty_log,
            "evidence_target": self.evidence_target,
            "raw_deficit_fraction": self.raw_deficit_fraction,
            "capped_deficit_fraction": self.capped_deficit_fraction,
        }


@dataclass(frozen=True)
class RepairCandidateOrigin:
    candidate_hash: str
    lanes: tuple[str, ...]
    target_shape_ids: tuple[str, ...]
    parent_hashes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_hash": self.candidate_hash,
            "lanes": list(self.lanes),
            "target_shape_ids": list(self.target_shape_ids),
            "parent_hashes": list(self.parent_hashes),
        }


@dataclass(frozen=True)
class RepairCandidatePool:
    candidates: tuple[Candidate, ...]
    origins: tuple[RepairCandidateOrigin, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_hashes": [candidate.hash for candidate in self.candidates],
            "origins": [origin.to_dict() for origin in self.origins],
        }


@dataclass(frozen=True)
class RepairAcquisition:
    deficits: dict[str, ShapeRepairDeficit]
    pair_close_probabilities: dict[tuple[str, str], float]
    plan: AcquisitionPlan

    def to_dict(self) -> dict[str, object]:
        return {
            "deficits": {shape_id: deficit.to_dict() for shape_id, deficit in sorted(self.deficits.items())},
            "pair_close_probabilities": {
                f"{shape_id}|{candidate_hash}": probability
                for (shape_id, candidate_hash), probability in sorted(self.pair_close_probabilities.items())
            },
            "plan": self.plan.to_dict(),
        }


@dataclass(frozen=True)
class RepairReport:
    eligible_shapes: int
    repair_queries: int
    preparation_reuse_queries: int
    resolved_outliers: int
    mean_gain_fraction: float | None
    worst_shape_gain_fraction: float | None
    false_repair_queries: int
    false_repair_predicted_cost_s: float
    per_shape_gain_fraction: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "eligible_shapes": self.eligible_shapes,
            "repair_queries": self.repair_queries,
            "preparation_reuse_queries": self.preparation_reuse_queries,
            "resolved_outliers": self.resolved_outliers,
            "mean_gain_fraction": self.mean_gain_fraction,
            "worst_shape_gain_fraction": self.worst_shape_gain_fraction,
            "false_repair_queries": self.false_repair_queries,
            "false_repair_predicted_cost_s": self.false_repair_predicted_cost_s,
            "per_shape_gain_fraction": dict(sorted(self.per_shape_gain_fraction.items())),
        }


def assess_repair_deficits(
    controller: CampaignControllerState,
    *,
    shapes: Sequence[Shape],
    clustering: ShapeClustering,
    predictions: Sequence[PairPrediction] = (),
    reference_performance: Mapping[str, float] | None = None,
    policy: RepairPolicy | None = None,
) -> dict[str, ShapeRepairDeficit]:
    active_policy = policy or RepairPolicy()
    shape_by_id = {shape.id: shape for shape in shapes}
    if set(shape_by_id) != set(controller.shape_ids) or set(shape_by_id) != set(clustering.shape_ids):
        raise ValueError("repair shapes, clustering, and controller must have identical identity")
    if len(shape_by_id) == 1:
        return {}
    references = dict(reference_performance or {})
    if any(
        shape_id not in shape_by_id or not math.isfinite(value) or value <= 0.0
        for shape_id, value in references.items()
    ):
        raise ValueError("repair references must be positive finite values for campaign shapes")
    distances = shape_descriptor_distances(clustering)
    cluster_by_shape = clustering.cluster_by_shape
    predictions_by_shape: dict[str, list[PairPrediction]] = defaultdict(list)
    for prediction in predictions:
        if (
            prediction.shape_id in shape_by_id
            and (
                prediction.shape_id,
                prediction.candidate_hash,
            )
            not in controller.queried_pairs
        ):
            predictions_by_shape[prediction.shape_id].append(prediction)
    deficits = {}
    minimum_log = math.log1p(active_policy.minimum_deficit_fraction)
    maximum_log = math.log1p(active_policy.maximum_deficit_fraction)
    for shape_id in sorted(shape_by_id):
        incumbent = controller.incumbents.get(shape_id)
        if incumbent is None:
            continue
        neighbors = sorted(
            (
                (distances[(shape_id, other_shape_id)], other_incumbent.performance)
                for other_shape_id, other_incumbent in controller.incumbents.items()
                if other_shape_id != shape_id
            ),
            key=lambda item: item[0],
        )[: active_policy.neighbor_count]
        neighbor_target = _weighted_log_quantile(neighbors, active_policy.neighbor_quantile)
        cluster_values = [
            other_incumbent.performance
            for other_shape_id, other_incumbent in controller.incumbents.items()
            if other_shape_id != shape_id and cluster_by_shape[other_shape_id] == cluster_by_shape[shape_id]
        ]
        cluster_target = _log_quantile(cluster_values, active_policy.cluster_quantile)
        reference_target = references.get(shape_id)
        evidence_values = [
            value for value in (reference_target, neighbor_target, cluster_target) if value is not None and value > 0.0
        ]
        if not evidence_values:
            continue
        evidence_target = math.exp(statistics.median(math.log(value) for value in evidence_values))
        uncertainty_log = max(
            (
                prediction.epistemic_std_log_performance * prediction.validity_probability
                for prediction in predictions_by_shape[shape_id]
            ),
            default=0.0,
        )
        raw_deficit_log = max(
            0.0,
            math.log(evidence_target / incumbent.performance) + active_policy.uncertainty_weight * uncertainty_log,
        )
        if raw_deficit_log < minimum_log:
            continue
        capped_deficit_log = min(raw_deficit_log, maximum_log)
        deficits[shape_id] = ShapeRepairDeficit(
            shape_id=shape_id,
            incumbent_candidate_hash=incumbent.candidate_hash,
            incumbent_performance=incumbent.performance,
            reference_target=reference_target,
            neighbor_target=neighbor_target,
            cluster_target=cluster_target,
            uncertainty_log=uncertainty_log,
            evidence_target=evidence_target,
            raw_deficit_log=raw_deficit_log,
            capped_deficit_log=capped_deficit_log,
        )
    return deficits


def repair_pair_close_probabilities(
    controller: CampaignControllerState,
    *,
    deficits: Mapping[str, ShapeRepairDeficit],
    predictions: Sequence[PairPrediction],
    policy: RepairPolicy | None = None,
) -> dict[tuple[str, str], float]:
    active_policy = policy or RepairPolicy()
    probabilities = {}
    for prediction in predictions:
        key = prediction.shape_id, prediction.candidate_hash
        deficit = deficits.get(prediction.shape_id)
        if deficit is None or key in controller.queried_pairs:
            continue
        reference = prediction.reference_performance
        if reference is None or reference <= 0.0 or not prediction.posterior_samples:
            probability = 0.0
        else:
            threshold_log = math.log(deficit.useful_target(active_policy.useful_close_fraction) / reference)
            probability = (
                sum(sample >= threshold_log for sample in prediction.posterior_samples)
                / len(prediction.posterior_samples)
                * prediction.validity_probability
            )
        probabilities[key] = probability if probability >= active_policy.minimum_close_probability else 0.0
    return probabilities


def build_repair_candidate_pool(
    controller: CampaignControllerState,
    *,
    shapes: Sequence[Shape],
    clustering: ShapeClustering,
    deficits: Mapping[str, ShapeRepairDeficit],
    observations: Sequence[PairEvaluationOutcome],
    candidate_catalog: Mapping[str, Candidate],
    broad_candidates: Sequence[Candidate] = (),
    policy: RepairPolicy | None = None,
) -> RepairCandidatePool:
    active_policy = policy or RepairPolicy()
    shape_by_id = {shape.id: shape for shape in shapes}
    if set(shape_by_id) != set(controller.shape_ids) or set(shape_by_id) != set(clustering.shape_ids):
        raise ValueError("repair seed shapes, clustering, and controller must have identical identity")
    if not deficits:
        return RepairCandidatePool((), ())
    distances = shape_descriptor_distances(clustering)
    cluster_by_shape = clustering.cluster_by_shape
    successful: dict[str, list[PairEvaluationOutcome]] = defaultdict(list)
    for outcome in observations:
        if (
            outcome.request.shape.id in shape_by_id
            and outcome.known
            and outcome.disclosed
            and outcome.performance is not None
            and outcome.performance > 0.0
        ):
            successful[outcome.request.shape.id].append(outcome)
    for outcomes in successful.values():
        outcomes.sort(key=lambda outcome: (-(outcome.performance or 0.0), outcome.request.candidate.hash))
    selected: dict[str, Candidate] = {}
    lanes: dict[str, set[str]] = defaultdict(set)
    targets: dict[str, set[str]] = defaultdict(set)

    def add(candidate: Candidate | None, *, lane: str, target_shape_id: str) -> None:
        if candidate is None:
            return
        selected[candidate.hash] = candidate
        lanes[candidate.hash].add(lane)
        targets[candidate.hash].add(target_shape_id)

    broad = sorted(broad_candidates, key=lambda candidate: candidate.hash)
    mutation_parents: dict[str, list[Candidate]] = defaultdict(list)
    for shape_id, deficit in sorted(deficits.items()):
        add(candidate_catalog.get(deficit.incumbent_candidate_hash), lane="incumbent", target_shape_id=shape_id)
        neighbor_shape_ids = sorted(
            (other_shape_id for other_shape_id in successful if other_shape_id != shape_id),
            key=lambda other_shape_id: (distances[(shape_id, other_shape_id)], other_shape_id),
        )[: active_policy.neighbor_count]
        for neighbor_shape_id in neighbor_shape_ids:
            for outcome in successful[neighbor_shape_id][: active_policy.neighbor_candidates_per_shape]:
                add(outcome.request.candidate, lane="neighbor", target_shape_id=shape_id)
        cluster_outcomes = sorted(
            (
                outcome
                for other_shape_id, outcomes in successful.items()
                if cluster_by_shape[other_shape_id] == cluster_by_shape[shape_id]
                for outcome in outcomes
            ),
            key=lambda outcome: (
                -(outcome.performance or 0.0),
                outcome.request.shape.id,
                outcome.request.candidate.hash,
            ),
        )
        for outcome in cluster_outcomes[: active_policy.cluster_candidates]:
            add(outcome.request.candidate, lane="cluster", target_shape_id=shape_id)
        for candidate in broad:
            add(candidate, lane="broad", target_shape_id=shape_id)
        mutation_parents[shape_id] = [
            candidate for candidate_hash, candidate in selected.items() if shape_id in targets[candidate_hash]
        ]
    for target_index, (shape_id, parents) in enumerate(sorted(mutation_parents.items())):
        mutations = semantic_mutation_candidates(
            parents,
            count=active_policy.mutation_candidates_per_shape,
            seed=active_policy.seed + target_index,
            target_shapes=[shape_by_id[shape_id]],
            max_changed_genes=active_policy.mutation_max_changed_genes,
            exclude=set(selected),
        )
        for candidate in mutations:
            add(candidate, lane="mutation", target_shape_id=shape_id)
    candidates = tuple(selected[candidate_hash] for candidate_hash in sorted(selected))
    origins = tuple(
        RepairCandidateOrigin(
            candidate_hash=candidate.hash,
            lanes=tuple(sorted(lanes[candidate.hash])),
            target_shape_ids=tuple(sorted(targets[candidate.hash])),
            parent_hashes=candidate.parent_hashes,
        )
        for candidate in candidates
    )
    return RepairCandidatePool(candidates, origins)


def plan_repair_acquisition(
    controller: CampaignControllerState,
    *,
    candidates: Sequence[Candidate],
    shapes: Sequence[Shape],
    deficits: Mapping[str, ShapeRepairDeficit],
    predictions: Sequence[PairPrediction],
    cost_model: BundleCostModel,
    acquisition_policy: BundleAcquisitionPolicy,
    repair_policy: RepairPolicy | None = None,
) -> RepairAcquisition:
    active_policy = repair_policy or RepairPolicy()
    probabilities = repair_pair_close_probabilities(
        controller,
        deficits=deficits,
        predictions=predictions,
        policy=active_policy,
    )
    repair_values = {
        shape.id: deficits[shape.id].capped_deficit_log if shape.id in deficits else 0.0 for shape in shapes
    }
    plan = plan_candidate_bundles(
        controller,
        candidates=candidates,
        shapes=shapes,
        predictions=predictions,
        cost_model=cost_model,
        policy=acquisition_policy,
        repair_values=repair_values,
        repair_probabilities=probabilities,
    )
    return RepairAcquisition(dict(deficits), probabilities, plan)


def summarize_repair(
    acquisition: RepairAcquisition,
    *,
    controller_after: CampaignControllerState,
    prepared_artifact_shapes_before: Mapping[str, set[str]] | None = None,
    useful_close_fraction: float = 0.5,
) -> RepairReport:
    if not 0.0 <= useful_close_fraction <= 1.0:
        raise ValueError("repair report close fraction must be in [0, 1]")
    prepared_before = prepared_artifact_shapes_before or {}
    queries = [request.key for request in acquisition.plan.timing_requests]
    reuse_queries = sum(shape_id in prepared_before.get(candidate_hash, set()) for shape_id, candidate_hash in queries)
    gains = {}
    resolved = 0
    false_shapes = set()
    for shape_id, deficit in acquisition.deficits.items():
        incumbent_after = controller_after.incumbents.get(shape_id)
        performance_after = (
            deficit.incumbent_performance
            if incumbent_after is None
            else max(deficit.incumbent_performance, incumbent_after.performance)
        )
        gain = performance_after / deficit.incumbent_performance - 1.0
        gains[shape_id] = gain
        if performance_after >= deficit.useful_target(useful_close_fraction):
            resolved += 1
        if gain <= 0.0:
            false_shapes.add(shape_id)
    false_queries = sum(shape_id in false_shapes for shape_id, _ in queries)
    false_cost = 0.0
    for score in acquisition.plan.selected:
        false_count = sum(shape.id in false_shapes for shape in score.bundle.shapes)
        if false_count:
            false_cost += score.bundle.cost.total_s * false_count / len(score.bundle.shapes)
    gain_values = list(gains.values())
    return RepairReport(
        eligible_shapes=len(acquisition.deficits),
        repair_queries=len(queries),
        preparation_reuse_queries=reuse_queries,
        resolved_outliers=resolved,
        mean_gain_fraction=statistics.fmean(gain_values) if gain_values else None,
        worst_shape_gain_fraction=min(gain_values) if gain_values else None,
        false_repair_queries=false_queries,
        false_repair_predicted_cost_s=false_cost,
        per_shape_gain_fraction=gains,
    )


def _log_quantile(values: Sequence[float], quantile: float) -> float | None:
    positive = sorted(math.log(value) for value in values if math.isfinite(value) and value > 0.0)
    if not positive:
        return None
    position = quantile * (len(positive) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return math.exp(positive[lower])
    weight = position - lower
    return math.exp(positive[lower] * (1.0 - weight) + positive[upper] * weight)


def _weighted_log_quantile(values: Sequence[tuple[float, float]], quantile: float) -> float | None:
    weighted = sorted(
        (math.log(value), 1.0 / max(distance, 0.125))
        for distance, value in values
        if math.isfinite(value) and value > 0.0
    )
    if not weighted:
        return None
    total_weight = sum(weight for _, weight in weighted)
    threshold = total_weight * quantile
    cumulative = 0.0
    for value, weight in weighted:
        cumulative += weight
        if cumulative >= threshold:
            return math.exp(value)
    return math.exp(weighted[-1][0])
