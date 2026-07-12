import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.evaluator import EvaluationResult, PairEvaluationOutcome, PairEvaluator
from evotensile.candidate import Candidate, Shape
from evotensile.scheduling.models import EvidenceStage, PairRequest
from evotensile.search.shape_clustering import ShapeClustering, shape_descriptor_distances

PromotionLane = Literal["specialist", "representative", "nearest", "broad"]


@dataclass(frozen=True)
class PromotionPolicy:
    neighbor_depth: int = 2
    representative_finalist_count: int = 3
    max_promotions_per_shape: int = 6
    specialist_slots: int = 1
    survivor_floor: int = 1
    broad_candidate_slots: int = 2
    broad_candidate_min_shapes: int = 2
    adjacent_cluster_depth: int = 1
    source_near_winner_fraction: float = 0.02
    probe_survivor_regret_fraction: float = 0.05
    stop_regret_fraction: float = 0.30
    probe_samples: int = 1
    main_samples: int = 3

    def __post_init__(self) -> None:
        positive_values = (
            self.max_promotions_per_shape,
            self.survivor_floor,
            self.broad_candidate_min_shapes,
            self.probe_samples,
            self.main_samples,
        )
        if any(value <= 0 for value in positive_values):
            raise ValueError("promotion capacities and sample targets must be positive")
        optional_counts = (
            self.neighbor_depth,
            self.representative_finalist_count,
            self.specialist_slots,
            self.broad_candidate_slots,
            self.adjacent_cluster_depth,
        )
        if any(value < 0 for value in optional_counts):
            raise ValueError("promotion lane counts must be nonnegative")
        if self.specialist_slots > self.max_promotions_per_shape:
            raise ValueError("specialist slots cannot exceed promotions per shape")
        if self.survivor_floor > self.max_promotions_per_shape:
            raise ValueError("survivor floor cannot exceed promotions per shape")
        if self.main_samples <= self.probe_samples:
            raise ValueError("promotion main sample target must exceed probe target")
        fractions = (
            self.source_near_winner_fraction,
            self.probe_survivor_regret_fraction,
            self.stop_regret_fraction,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in fractions):
            raise ValueError("promotion regret fractions must be finite and nonnegative")


@dataclass(frozen=True)
class PromotionPlan:
    candidate: Candidate
    source_shape_id: str
    destination_shape_id: str
    source_cluster_id: str
    destination_cluster_id: str
    lane: PromotionLane
    source_performance: float
    destination_distance: float

    @property
    def pair(self) -> tuple[str, str]:
        return self.destination_shape_id, self.candidate.hash


@dataclass(frozen=True)
class PromotionEvent:
    source_shape_id: str
    destination_shape_id: str
    candidate_hash: str
    source_cluster_id: str
    destination_cluster_id: str
    lane: PromotionLane
    promotion_stage: str
    preparation_reused: bool
    artifact_scope_shape_ids: tuple[str, ...]
    success: bool
    realized_gain_fraction: float | None
    performance: float | None
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_shape_id": self.source_shape_id,
            "destination_shape_id": self.destination_shape_id,
            "candidate_hash": self.candidate_hash,
            "source_cluster_id": self.source_cluster_id,
            "destination_cluster_id": self.destination_cluster_id,
            "lane": self.lane,
            "promotion_stage": self.promotion_stage,
            "preparation_reused": self.preparation_reused,
            "artifact_scope_shape_ids": list(self.artifact_scope_shape_ids),
            "success": self.success,
            "realized_gain_fraction": self.realized_gain_fraction,
            "performance": self.performance,
            "status": self.status,
        }


@dataclass(frozen=True)
class PromotionRaceResult:
    plans: tuple[PromotionPlan, ...]
    probe_result: EvaluationResult | None
    main_result: EvaluationResult | None
    events: tuple[PromotionEvent, ...]

    @property
    def probe_pairs(self) -> int:
        return 0 if self.probe_result is None else len(self.probe_result.outcomes)

    @property
    def main_pairs(self) -> int:
        return 0 if self.main_result is None else len(self.main_result.outcomes)

    def to_dict(self) -> dict[str, object]:
        return {
            "planned_pairs": len(self.plans),
            "probe_pairs": self.probe_pairs,
            "main_pairs": self.main_pairs,
            "events": [event.to_dict() for event in self.events],
        }


def plan_shape_promotions(
    controller: CampaignControllerState,
    *,
    shapes: Sequence[Shape],
    clustering: ShapeClustering,
    observations: Sequence[PairEvaluationOutcome],
    policy: PromotionPolicy,
) -> tuple[PromotionPlan, ...]:
    shape_by_id = {shape.id: shape for shape in shapes}
    if set(shape_by_id) != set(clustering.shape_ids) or set(shape_by_id) != set(controller.shape_ids):
        raise ValueError("promotion shapes, clustering, and controller must have identical identity")
    successful = _successful_observations(observations)
    if len(shapes) == 1 or not successful:
        return ()
    distances = shape_descriptor_distances(clustering)
    cluster_by_shape = clustering.cluster_by_shape
    medoid_by_cluster = {cluster.cluster_id: cluster.medoid_shape_id for cluster in clustering.clusters}
    cluster_neighbors = _cluster_neighbors(clustering, distances, policy.adjacent_cluster_depth)
    blocked = _blocked_candidate_clusters(controller, successful, cluster_by_shape, policy.stop_regret_fraction)
    broad_candidates = _broad_candidates(successful, cluster_by_shape, policy.broad_candidate_min_shapes)
    plans = []
    for destination_shape_id in sorted(shape_by_id):
        proposals: list[PromotionPlan] = []
        source_shape_ids = sorted(
            (shape_id for shape_id in successful if shape_id != destination_shape_id),
            key=lambda shape_id: (distances[(destination_shape_id, shape_id)], shape_id),
        )[: policy.neighbor_depth]
        for source_index, source_shape_id in enumerate(source_shape_ids):
            near_winners = _near_winners(successful[source_shape_id], policy.source_near_winner_fraction)
            for winner_index, outcome in enumerate(near_winners):
                lane: PromotionLane = (
                    "specialist" if source_index == 0 and winner_index < policy.specialist_slots else "nearest"
                )
                proposals.append(
                    _promotion_plan(
                        outcome,
                        source_shape_id=source_shape_id,
                        destination_shape_id=destination_shape_id,
                        lane=lane,
                        cluster_by_shape=cluster_by_shape,
                        distances=distances,
                    )
                )
        destination_cluster = cluster_by_shape[destination_shape_id]
        medoid_shape_id = medoid_by_cluster[destination_cluster]
        for outcome in successful.get(medoid_shape_id, ())[: policy.representative_finalist_count]:
            proposals.append(
                _promotion_plan(
                    outcome,
                    source_shape_id=medoid_shape_id,
                    destination_shape_id=destination_shape_id,
                    lane="representative",
                    cluster_by_shape=cluster_by_shape,
                    distances=distances,
                )
            )
        adjacent_clusters = {destination_cluster, *cluster_neighbors[destination_cluster]}
        broad_added = 0
        proposed_candidate_hashes = {proposal.candidate.hash for proposal in proposals}
        for candidate_hash, outcomes in broad_candidates:
            if candidate_hash in proposed_candidate_hashes:
                continue
            source_outcome = min(
                (outcome for outcome in outcomes if cluster_by_shape[outcome.request.shape.id] in adjacent_clusters),
                key=lambda outcome: (
                    distances[(destination_shape_id, outcome.request.shape.id)],
                    outcome.request.shape.id,
                ),
                default=None,
            )
            if source_outcome is None:
                continue
            proposals.append(
                _promotion_plan(
                    source_outcome,
                    source_shape_id=source_outcome.request.shape.id,
                    destination_shape_id=destination_shape_id,
                    lane="broad",
                    cluster_by_shape=cluster_by_shape,
                    distances=distances,
                )
            )
            broad_added += 1
            if broad_added >= policy.broad_candidate_slots:
                break
        selected: dict[tuple[str, str], PromotionPlan] = {}
        lane_order = {"specialist": 0, "nearest": 1, "representative": 2, "broad": 3}
        for proposal in sorted(
            proposals,
            key=lambda item: (
                lane_order[item.lane],
                -item.source_performance,
                item.destination_distance,
                item.source_shape_id,
                item.candidate.hash,
            ),
        ):
            if proposal.pair in controller.queried_pairs:
                continue
            if (proposal.candidate.hash, destination_cluster) in blocked:
                continue
            selected.setdefault(proposal.pair, proposal)
            if len(selected) >= policy.max_promotions_per_shape:
                break
        plans.extend(selected.values())
    return tuple(sorted(plans, key=lambda item: (item.destination_shape_id, item.lane, item.candidate.hash)))


def execute_promotion_race(
    evaluator: PairEvaluator,
    controller: CampaignControllerState,
    *,
    shapes: Sequence[Shape],
    clustering: ShapeClustering,
    observations: Sequence[PairEvaluationOutcome],
    policy: PromotionPolicy,
) -> PromotionRaceResult:
    plans = plan_shape_promotions(
        controller,
        shapes=shapes,
        clustering=clustering,
        observations=observations,
        policy=policy,
    )
    if not plans:
        return PromotionRaceResult((), None, None, ())
    shape_by_id = {shape.id: shape for shape in shapes}
    initial_incumbents = {
        shape_id: None if incumbent is None else incumbent.performance
        for shape_id in shape_by_id
        for incumbent in [controller.incumbents.get(shape_id)]
    }
    artifact_shapes = _artifact_scopes(plans, shape_by_id)
    prepared_before = {
        candidate_hash: set(controller.prepared_artifact_shapes.get(candidate_hash, set()))
        for candidate_hash in artifact_shapes
    }
    probe_result = evaluator.evaluate(
        [
            PairRequest(
                plan.candidate,
                shape_by_id[plan.destination_shape_id],
                evidence_stage=EvidenceStage.PROBE,
                min_samples=policy.probe_samples,
            )
            for plan in plans
        ],
        artifact_shapes_by_candidate=artifact_shapes,
    )
    probe_result.apply(controller)
    probe_by_pair = {outcome.key: outcome for outcome in probe_result.outcomes}
    survivors = _probe_survivors(plans, probe_by_pair, controller, policy)
    main_result = None
    if survivors:
        main_result = evaluator.evaluate(
            [
                PairRequest(
                    plan.candidate,
                    shape_by_id[plan.destination_shape_id],
                    evidence_stage=EvidenceStage.SCREENING,
                    min_samples=policy.main_samples,
                )
                for plan in survivors
            ],
            artifact_shapes_by_candidate={
                candidate_hash: scope
                for candidate_hash, scope in artifact_shapes.items()
                if candidate_hash in {plan.candidate.hash for plan in survivors}
            },
        )
        main_result.apply(controller)
    main_by_pair = {} if main_result is None else {outcome.key: outcome for outcome in main_result.outcomes}
    survivor_pairs = {plan.pair for plan in survivors}
    events = []
    for plan in plans:
        outcome = main_by_pair.get(plan.pair, probe_by_pair[plan.pair])
        baseline = initial_incumbents[plan.destination_shape_id]
        gain = None
        if outcome.performance is not None and baseline is not None:
            gain = outcome.performance / baseline - 1.0
        final_incumbent = controller.incumbents.get(plan.destination_shape_id)
        success = bool(
            outcome.performance is not None
            and final_incumbent is not None
            and final_incumbent.candidate_hash == plan.candidate.hash
            and (baseline is None or outcome.performance > baseline)
        )
        scope_ids = tuple(shape.id for shape in artifact_shapes[plan.candidate.hash])
        preparation_reused = bool(prepared_before[plan.candidate.hash].intersection(scope_ids) or len(scope_ids) > 1)
        event = PromotionEvent(
            source_shape_id=plan.source_shape_id,
            destination_shape_id=plan.destination_shape_id,
            candidate_hash=plan.candidate.hash,
            source_cluster_id=plan.source_cluster_id,
            destination_cluster_id=plan.destination_cluster_id,
            lane=plan.lane,
            promotion_stage="main" if plan.pair in survivor_pairs else "probe_rejected",
            preparation_reused=preparation_reused,
            artifact_scope_shape_ids=scope_ids,
            success=success,
            realized_gain_fraction=gain,
            performance=outcome.performance,
            status=outcome.status,
        )
        events.append(event)
        controller.append_trace("promotion", event.to_dict())
    return PromotionRaceResult(plans, probe_result, main_result, tuple(events))


def _successful_observations(
    observations: Sequence[PairEvaluationOutcome],
) -> dict[str, tuple[PairEvaluationOutcome, ...]]:
    by_shape: dict[str, dict[str, PairEvaluationOutcome]] = defaultdict(dict)
    for outcome in observations:
        if outcome.known and outcome.performance is not None and outcome.performance > 0.0:
            existing = by_shape[outcome.request.shape.id].get(outcome.request.candidate.hash)
            if existing is None or (existing.performance or 0.0) < outcome.performance:
                by_shape[outcome.request.shape.id][outcome.request.candidate.hash] = outcome
    return {
        shape_id: tuple(
            sorted(
                outcomes.values(),
                key=lambda outcome: (-(outcome.performance or 0.0), outcome.request.candidate.hash),
            )
        )
        for shape_id, outcomes in by_shape.items()
    }


def _near_winners(
    outcomes: Sequence[PairEvaluationOutcome],
    fraction: float,
) -> tuple[PairEvaluationOutcome, ...]:
    if not outcomes:
        return ()
    threshold = (outcomes[0].performance or 0.0) / (1.0 + fraction)
    return tuple(outcome for outcome in outcomes if (outcome.performance or 0.0) >= threshold)


def _promotion_plan(
    outcome: PairEvaluationOutcome,
    *,
    source_shape_id: str,
    destination_shape_id: str,
    lane: PromotionLane,
    cluster_by_shape: Mapping[str, str],
    distances: Mapping[tuple[str, str], float],
) -> PromotionPlan:
    assert outcome.performance is not None
    return PromotionPlan(
        candidate=outcome.request.candidate,
        source_shape_id=source_shape_id,
        destination_shape_id=destination_shape_id,
        source_cluster_id=cluster_by_shape[source_shape_id],
        destination_cluster_id=cluster_by_shape[destination_shape_id],
        lane=lane,
        source_performance=outcome.performance,
        destination_distance=distances[(source_shape_id, destination_shape_id)],
    )


def _cluster_neighbors(
    clustering: ShapeClustering,
    distances: Mapping[tuple[str, str], float],
    depth: int,
) -> dict[str, tuple[str, ...]]:
    return {
        cluster.cluster_id: tuple(
            other.cluster_id
            for other in sorted(
                (other for other in clustering.clusters if other.cluster_id != cluster.cluster_id),
                key=lambda other: (
                    distances[(cluster.medoid_shape_id, other.medoid_shape_id)],
                    other.cluster_id,
                ),
            )[:depth]
        )
        for cluster in clustering.clusters
    }


def _broad_candidates(
    successful: Mapping[str, Sequence[PairEvaluationOutcome]],
    cluster_by_shape: Mapping[str, str],
    minimum_shapes: int,
) -> list[tuple[str, tuple[PairEvaluationOutcome, ...]]]:
    by_candidate: dict[str, list[PairEvaluationOutcome]] = defaultdict(list)
    for outcomes in successful.values():
        for outcome in outcomes:
            by_candidate[outcome.request.candidate.hash].append(outcome)
    broad = [
        (candidate_hash, tuple(outcomes))
        for candidate_hash, outcomes in by_candidate.items()
        if len({outcome.request.shape.id for outcome in outcomes}) >= minimum_shapes
        and len({cluster_by_shape[outcome.request.shape.id] for outcome in outcomes}) >= 2
    ]
    return sorted(
        broad,
        key=lambda item: (
            -len({outcome.request.shape.id for outcome in item[1]}),
            -sum(outcome.performance or 0.0 for outcome in item[1]) / len(item[1]),
            item[0],
        ),
    )


def _blocked_candidate_clusters(
    controller: CampaignControllerState,
    successful: Mapping[str, Sequence[PairEvaluationOutcome]],
    cluster_by_shape: Mapping[str, str],
    stop_regret_fraction: float,
) -> set[tuple[str, str]]:
    blocked = set()
    for shape_id, outcomes in successful.items():
        incumbent = controller.incumbents.get(shape_id)
        if incumbent is None:
            continue
        threshold = incumbent.performance / (1.0 + stop_regret_fraction)
        for outcome in outcomes:
            if (outcome.performance or 0.0) < threshold:
                blocked.add((outcome.request.candidate.hash, cluster_by_shape[shape_id]))
    return blocked


def _artifact_scopes(
    plans: Sequence[PromotionPlan],
    shape_by_id: Mapping[str, Shape],
) -> dict[str, tuple[Shape, ...]]:
    shape_ids_by_candidate: dict[str, set[str]] = defaultdict(set)
    for plan in plans:
        shape_ids_by_candidate[plan.candidate.hash].add(plan.destination_shape_id)
    return {
        candidate_hash: tuple(shape_by_id[shape_id] for shape_id in sorted(shape_ids))
        for candidate_hash, shape_ids in shape_ids_by_candidate.items()
    }


def _probe_survivors(
    plans: Sequence[PromotionPlan],
    outcomes: Mapping[tuple[str, str], PairEvaluationOutcome],
    controller: CampaignControllerState,
    policy: PromotionPolicy,
) -> tuple[PromotionPlan, ...]:
    plans_by_shape: dict[str, list[PromotionPlan]] = defaultdict(list)
    for plan in plans:
        plans_by_shape[plan.destination_shape_id].append(plan)
    survivors = []
    for shape_id, shape_plans in sorted(plans_by_shape.items()):
        positive = [
            plan
            for plan in shape_plans
            if outcomes[plan.pair].performance is not None and (outcomes[plan.pair].performance or 0.0) > 0.0
        ]
        positive.sort(
            key=lambda plan: (
                -(outcomes[plan.pair].performance or 0.0),
                plan.lane != "specialist",
                plan.candidate.hash,
            )
        )
        incumbent = controller.incumbents.get(shape_id)
        threshold = 0.0 if incumbent is None else incumbent.performance / (1.0 + policy.probe_survivor_regret_fraction)
        selected = [plan for plan in positive if (outcomes[plan.pair].performance or 0.0) >= threshold]
        specialist = [plan for plan in positive if plan.lane == "specialist"][: policy.specialist_slots]
        selected_by_pair = {plan.pair: plan for plan in (*specialist, *selected)}
        if len(selected_by_pair) < policy.survivor_floor:
            for plan in positive:
                selected_by_pair.setdefault(plan.pair, plan)
                if len(selected_by_pair) >= policy.survivor_floor:
                    break
        survivors.extend(selected_by_pair.values())
    return tuple(sorted(survivors, key=lambda item: (item.destination_shape_id, item.lane, item.candidate.hash)))
