import heapq
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.evaluator import PairEvaluationOutcome
from evotensile.candidate import Candidate, Shape
from evotensile.scheduling.models import EvidenceStage, PairRequest
from evotensile.search.cost_model import predicted_candidate_prepare_weight
from evotensile.search.evidence import ProposalEvidenceSnapshot
from evotensile.search.measured_cost import CandidateMeasuredCost
from evotensile.search.pair_model import ContextualPairModel, PairModelConfiguration, PairPrediction
from evotensile.search.surrogate import candidate_shape_features

if TYPE_CHECKING:
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.feature_extraction import DictVectorizer


@dataclass(frozen=True)
class BundleAcquisitionPolicy:
    improvement_weight: float = 1.0
    coverage_weight: float = 0.20
    information_weight: float = 0.10
    repair_weight: float = 0.0
    bundle_sizes: tuple[int, ...] = (1, 2, 4, 8)
    max_pairs: int = 64
    max_bundles: int = 16
    max_predicted_cost_s: float = 300.0
    min_utility_per_s: float = 0.0
    min_samples: int = 1
    evidence_stage: EvidenceStage = EvidenceStage.PROBE

    def __post_init__(self) -> None:
        weights = (
            self.improvement_weight,
            self.coverage_weight,
            self.information_weight,
            self.repair_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("bundle acquisition weights must be finite and nonnegative")
        if not self.bundle_sizes or any(value <= 0 for value in self.bundle_sizes):
            raise ValueError("bundle acquisition sizes must be positive")
        if tuple(sorted(set(self.bundle_sizes))) != self.bundle_sizes:
            raise ValueError("bundle acquisition sizes must be unique and increasing")
        if self.max_pairs <= 0 or self.max_bundles <= 0 or self.min_samples <= 0:
            raise ValueError("bundle acquisition capacities and samples must be positive")
        if not math.isfinite(self.max_predicted_cost_s) or self.max_predicted_cost_s <= 0.0:
            raise ValueError("bundle acquisition cost budget must be finite and positive")
        if not math.isfinite(self.min_utility_per_s) or self.min_utility_per_s < 0.0:
            raise ValueError("bundle acquisition minimum utility rate must be finite and nonnegative")


@dataclass(frozen=True)
class BundleCostFitSummary:
    preparation_rows: int
    validation_rows: int
    timing_rows: int
    preparation_fitted: bool
    validation_fitted: bool
    timing_fitted: bool
    preparation_fallback_s: float
    validation_fallback_s: float
    timing_fallback_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "preparation_rows": self.preparation_rows,
            "validation_rows": self.validation_rows,
            "timing_rows": self.timing_rows,
            "preparation_fitted": self.preparation_fitted,
            "validation_fitted": self.validation_fitted,
            "timing_fitted": self.timing_fitted,
            "preparation_fallback_s": self.preparation_fallback_s,
            "validation_fallback_s": self.validation_fallback_s,
            "timing_fallback_s": self.timing_fallback_s,
        }


@dataclass(frozen=True)
class BundleCostEstimate:
    preparation_s: float
    validation_s: float
    timing_s: float
    preparation_required: bool
    artifact_expansion_required: bool

    @property
    def total_s(self) -> float:
        return self.preparation_s + self.validation_s + self.timing_s

    def to_dict(self) -> dict[str, object]:
        return {
            "preparation_s": self.preparation_s,
            "validation_s": self.validation_s,
            "timing_s": self.timing_s,
            "total_s": self.total_s,
            "preparation_required": self.preparation_required,
            "artifact_expansion_required": self.artifact_expansion_required,
        }


@dataclass(frozen=True)
class CandidateBundle:
    candidate: Candidate
    shapes: tuple[Shape, ...]
    artifact_shapes: tuple[Shape, ...]
    predictions: tuple[PairPrediction, ...]
    cost: BundleCostEstimate

    @property
    def pair_keys(self) -> tuple[tuple[str, str], ...]:
        return tuple((shape.id, self.candidate.hash) for shape in self.shapes)


@dataclass(frozen=True)
class BundleScore:
    bundle: CandidateBundle
    expected_improvement: float
    unresolved_coverage: float
    information_value: float
    repair_value: float
    marginal_utility: float
    utility_per_s: float
    selection_index: int

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_hash": self.bundle.candidate.hash,
            "shape_ids": [shape.id for shape in self.bundle.shapes],
            "expected_improvement": self.expected_improvement,
            "unresolved_coverage": self.unresolved_coverage,
            "information_value": self.information_value,
            "repair_value": self.repair_value,
            "marginal_utility": self.marginal_utility,
            "utility_per_s": self.utility_per_s,
            "selection_index": self.selection_index,
            "cost": self.bundle.cost.to_dict(),
        }


@dataclass(frozen=True)
class AcquisitionPlan:
    selected: tuple[BundleScore, ...]
    preparation_order: tuple[str, ...]
    timing_requests: tuple[PairRequest, ...]
    artifact_shapes_by_candidate: dict[str, tuple[Shape, ...]]
    predicted_cost_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "selected": [score.to_dict() for score in self.selected],
            "preparation_order": list(self.preparation_order),
            "timing_pairs": [list(request.key) for request in self.timing_requests],
            "artifact_shapes_by_candidate": {
                candidate_hash: [shape.id for shape in shapes]
                for candidate_hash, shapes in sorted(self.artifact_shapes_by_candidate.items())
            },
            "predicted_cost_s": self.predicted_cost_s,
        }


class BundleCostModel:
    def __init__(
        self,
        *,
        workgroup_processor_count: int,
        fallback_preparation_s: float = 5.0,
        fallback_validation_s: float = 0.05,
        fallback_timing_s: float = 0.05,
        min_fit_rows: int = 12,
        seed: int = 0,
        jobs: int = 1,
    ) -> None:
        if workgroup_processor_count <= 0 or min_fit_rows <= 0 or jobs == 0:
            raise ValueError("bundle cost model topology, fit rows, and jobs must be valid")
        fallbacks = (fallback_preparation_s, fallback_validation_s, fallback_timing_s)
        if any(not math.isfinite(value) or value < 0.0 for value in fallbacks):
            raise ValueError("bundle cost fallbacks must be finite and nonnegative")
        self.workgroup_processor_count = workgroup_processor_count
        self.min_fit_rows = min_fit_rows
        self.seed = seed
        self.jobs = jobs
        self._fallbacks = fallbacks
        self._prepare_model = _CostRegressor(seed=seed, jobs=jobs, min_fit_rows=min_fit_rows)
        self._validation_model = _CostRegressor(seed=seed + 1, jobs=jobs, min_fit_rows=min_fit_rows)
        self._timing_model = _CostRegressor(seed=seed + 2, jobs=jobs, min_fit_rows=min_fit_rows)
        self._prepare_model.fit((), (), fallback=fallback_preparation_s)
        self._validation_model.fit((), (), fallback=fallback_validation_s)
        self._timing_model.fit((), (), fallback=fallback_timing_s)
        self._prepare_weight_reference = 1.0

    def fit(
        self,
        *,
        candidates: Mapping[str, Candidate],
        shapes_by_candidate: Mapping[str, Sequence[Shape]],
        measured_costs: Mapping[str, CandidateMeasuredCost],
    ) -> BundleCostFitSummary:
        preparation_rows = []
        preparation_targets = []
        validation_rows = []
        validation_targets = []
        timing_rows = []
        timing_targets = []
        preparation_weights = []
        for candidate_hash, cost in measured_costs.items():
            candidate = candidates.get(candidate_hash)
            shapes = tuple(shapes_by_candidate.get(candidate_hash, ()))
            if candidate is None or not shapes:
                continue
            preparation_weights.append(
                statistics.fmean(
                    predicted_candidate_prepare_weight(
                        candidate,
                        shape,
                        workgroup_processor_count=self.workgroup_processor_count,
                    )
                    for shape in shapes
                )
            )
            if cost.prepare_s > 0.0:
                preparation_rows.append(self._bundle_features(candidate, shapes))
                preparation_targets.append(cost.prepare_s)
            pair_count = len(shapes)
            for shape in shapes:
                features = candidate_shape_features(
                    candidate,
                    shape,
                    workgroup_processor_count=self.workgroup_processor_count,
                )
                if cost.validation_s > 0.0:
                    validation_rows.append(features)
                    validation_targets.append(cost.validation_s / pair_count)
                timing_cost = cost.probe_s + cost.screening_s
                if timing_cost > 0.0:
                    timing_rows.append(features)
                    timing_targets.append(timing_cost / pair_count)
        if preparation_weights:
            self._prepare_weight_reference = statistics.median(preparation_weights)
        self._prepare_model.fit(preparation_rows, preparation_targets, fallback=self._fallbacks[0])
        self._validation_model.fit(validation_rows, validation_targets, fallback=self._fallbacks[1])
        self._timing_model.fit(timing_rows, timing_targets, fallback=self._fallbacks[2])
        return BundleCostFitSummary(
            preparation_rows=len(preparation_rows),
            validation_rows=len(validation_rows),
            timing_rows=len(timing_rows),
            preparation_fitted=self._prepare_model.fitted,
            validation_fitted=self._validation_model.fitted,
            timing_fitted=self._timing_model.fitted,
            preparation_fallback_s=self._prepare_model.fallback,
            validation_fallback_s=self._validation_model.fallback,
            timing_fallback_s=self._timing_model.fallback,
        )

    def estimate(
        self,
        candidate: Candidate,
        shapes: Sequence[Shape],
        *,
        artifact_shapes: Sequence[Shape] | None = None,
        prepared_shape_ids: set[str],
    ) -> BundleCostEstimate:
        if not shapes:
            raise ValueError("bundle cost estimate requires shapes")
        active_artifact_shapes = tuple(artifact_shapes or shapes)
        requested_shape_ids = {shape.id for shape in active_artifact_shapes}
        new_artifact_shapes = requested_shape_ids - prepared_shape_ids
        preparation_required = bool(new_artifact_shapes)
        artifact_expansion_required = bool(prepared_shape_ids and new_artifact_shapes)
        preparation_s = 0.0
        if preparation_required:
            preparation_s = self._prepare_model.predict(self._bundle_features(candidate, active_artifact_shapes))
            if not self._prepare_model.fitted:
                weight = statistics.fmean(
                    predicted_candidate_prepare_weight(
                        candidate,
                        shape,
                        workgroup_processor_count=self.workgroup_processor_count,
                    )
                    for shape in active_artifact_shapes
                )
                preparation_s *= weight / max(self._prepare_weight_reference, 1e-12)
        validation_s = sum(
            self._validation_model.predict(
                candidate_shape_features(
                    candidate,
                    shape,
                    workgroup_processor_count=self.workgroup_processor_count,
                )
            )
            for shape in shapes
        )
        timing_s = sum(
            self._timing_model.predict(
                candidate_shape_features(
                    candidate,
                    shape,
                    workgroup_processor_count=self.workgroup_processor_count,
                )
            )
            for shape in shapes
        )
        return BundleCostEstimate(
            preparation_s=max(0.0, preparation_s),
            validation_s=max(0.0, validation_s),
            timing_s=max(0.0, timing_s),
            preparation_required=preparation_required,
            artifact_expansion_required=artifact_expansion_required,
        )

    def _bundle_features(self, candidate: Candidate, shapes: Sequence[Shape]) -> dict[str, float | str]:
        rows = [
            candidate_shape_features(
                candidate,
                shape,
                workgroup_processor_count=self.workgroup_processor_count,
            )
            for shape in shapes
        ]
        features: dict[str, float | str] = {}
        for name in rows[0]:
            values = [row[name] for row in rows]
            if all(isinstance(value, (int, float)) for value in values):
                numeric = [float(value) for value in values]
                features[f"mean:{name}"] = statistics.fmean(numeric)
                features[f"max:{name}"] = max(numeric)
            else:
                features[name] = str(values[0])
        features["scope:shape_count"] = float(len(shapes))
        return features


class _CostRegressor:
    def __init__(self, *, seed: int, jobs: int, min_fit_rows: int) -> None:
        self.seed = seed
        self.jobs = jobs
        self.min_fit_rows = min_fit_rows
        self._vectorizer: DictVectorizer | None = None
        self._model: ExtraTreesRegressor | None = None
        self._fallback = 0.0

    @property
    def fitted(self) -> bool:
        return self._model is not None

    @property
    def fallback(self) -> float:
        return self._fallback

    def fit(
        self,
        rows: Sequence[Mapping[str, float | str]],
        targets: Sequence[float],
        *,
        fallback: float,
    ) -> None:
        positive = [float(value) for value in targets if math.isfinite(value) and value > 0.0]
        if positive:
            median = statistics.median(positive)
            mad = statistics.median(abs(value - median) for value in positive)
            self._fallback = max(fallback, median + 2.0 * mad)
        else:
            self._fallback = fallback
        if len(rows) < self.min_fit_rows or len(rows) != len(targets):
            self._vectorizer = None
            self._model = None
            return
        try:
            from sklearn.ensemble import ExtraTreesRegressor
            from sklearn.feature_extraction import DictVectorizer
        except ImportError as exc:  # pragma: no cover - exercised only in minimal installations
            raise RuntimeError("bundle cost fitting requires scikit-learn") from exc
        self._vectorizer = DictVectorizer(sparse=False)
        matrix = self._vectorizer.fit_transform(rows)
        self._model = ExtraTreesRegressor(
            n_estimators=96,
            min_samples_leaf=2,
            max_features=0.7,
            bootstrap=True,
            max_samples=0.8,
            random_state=self.seed,
            n_jobs=self.jobs,
        )
        self._model.fit(matrix, targets)

    def predict(self, row: Mapping[str, float | str]) -> float:
        if self._model is None or self._vectorizer is None:
            return self._fallback
        matrix = self._vectorizer.transform([row])
        tree_values = [float(estimator.predict(matrix)[0]) for estimator in self._model.estimators_]
        median = statistics.median(tree_values)
        mad = statistics.median(abs(value - median) for value in tree_values)
        return max(self._fallback * 0.25, median + 2.0 * mad)


def select_singleton_bundle_pool(
    candidates: Sequence[Candidate],
    *,
    evidence: ProposalEvidenceSnapshot,
    shape: Shape,
    count: int,
    seed: int,
    workgroup_processor_count: int,
    jobs: int,
    min_evidence: int = 24,
    information_weight: float = 0.05,
) -> tuple[Candidate, ...] | None:
    summaries = [
        summary
        for summary in evidence.shape_summaries(shape.id)
        if summary.median_gflops is not None and summary.median_gflops > 0.0
    ]
    if len(summaries) < min_evidence or count <= 0 or not candidates:
        return None
    outcomes = tuple(
        PairEvaluationOutcome(
            request=PairRequest(
                evidence.candidates[summary.candidate_hash],
                shape,
                evidence_stage=EvidenceStage.SCREENING,
            ),
            provenance="campaign-db",
            source_ref="proposal-evidence",
            status="ok",
            known=True,
            disclosed=True,
            samples=summary.samples,
            performance=summary.median_gflops,
        )
        for summary in summaries
        if summary.candidate_hash in evidence.candidates
    )
    if len(outcomes) < min_evidence:
        return None
    model = ContextualPairModel(
        workgroup_processor_count=workgroup_processor_count,
        configuration=PairModelConfiguration(
            min_performance_rows=min_evidence,
            seed=seed,
            jobs=jobs,
        ),
    )
    model.fit(outcomes)
    predictions = model.predict([(candidate, shape) for candidate in candidates])
    controller = CampaignControllerState(
        shape_ids=(shape.id,),
        time_budget_s=300.0,
        session_started_at=0.0,
    )
    for outcome in outcomes:
        controller.record_query(shape.id, outcome.request.candidate.hash, known=True)
        controller.disclose(
            shape.id,
            outcome.request.candidate.hash,
            performance=outcome.performance,
        )
        controller.record_prepared(outcome.request.candidate.hash, (shape.id,))
    cost_model = BundleCostModel(
        workgroup_processor_count=workgroup_processor_count,
        fallback_preparation_s=0.1,
        fallback_validation_s=0.0,
        fallback_timing_s=0.001,
        seed=seed,
        jobs=jobs,
    )
    cost_model.fit(
        candidates=evidence.candidates,
        shapes_by_candidate={candidate_hash: (shape,) for candidate_hash in evidence.candidate_costs},
        measured_costs=evidence.candidate_costs,
    )
    plan = plan_candidate_bundles(
        controller,
        candidates=candidates,
        shapes=(shape,),
        predictions=predictions,
        cost_model=cost_model,
        policy=BundleAcquisitionPolicy(
            coverage_weight=0.0,
            information_weight=information_weight,
            bundle_sizes=(1,),
            max_pairs=count,
            max_bundles=count,
            max_predicted_cost_s=300.0,
            evidence_stage=EvidenceStage.SCREENING,
        ),
    )
    return tuple(request.candidate for request in plan.timing_requests)


def plan_candidate_bundles(
    controller: CampaignControllerState,
    *,
    candidates: Sequence[Candidate],
    shapes: Sequence[Shape],
    predictions: Sequence[PairPrediction],
    cost_model: BundleCostModel,
    policy: BundleAcquisitionPolicy,
    repair_values: Mapping[str, float] | None = None,
    repair_probabilities: Mapping[tuple[str, str], float] | None = None,
    shape_weights: Mapping[str, float] | None = None,
    artifact_shapes_by_target: Mapping[str, Sequence[Shape]] | None = None,
) -> AcquisitionPlan:
    shape_by_id = {shape.id: shape for shape in shapes}
    candidate_by_hash = {candidate.hash: candidate for candidate in candidates}
    if set(shape_by_id) != set(controller.shape_ids):
        raise ValueError("bundle acquisition shapes must match controller identity")
    artifact_scope = {shape_id: (shape,) for shape_id, shape in shape_by_id.items()}
    if artifact_shapes_by_target is not None:
        if set(artifact_shapes_by_target) != set(shape_by_id):
            raise ValueError("bundle artifact scopes must cover exact target shapes")
        artifact_scope = {}
        for shape_id, scope in artifact_shapes_by_target.items():
            scope_by_id = {shape.id: shape for shape in scope}
            if shape_id not in scope_by_id or any(scope_id not in shape_by_id for scope_id in scope_by_id):
                raise ValueError("bundle artifact scopes must include their target and use campaign shapes")
            artifact_scope[shape_id] = tuple(scope_by_id[scope_id] for scope_id in sorted(scope_by_id))
    weights = controller.shape_weights
    if shape_weights is not None:
        if set(shape_weights) != set(shape_by_id):
            raise ValueError("bundle acquisition weights must cover exact shapes")
        weights = {shape_id: float(shape_weights[shape_id]) for shape_id in shape_by_id}
        if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
            raise ValueError("bundle acquisition weights must be finite and nonnegative")
    repair = {shape_id: 0.0 for shape_id in shape_by_id}
    if repair_values is not None:
        if set(repair_values) != set(shape_by_id):
            raise ValueError("bundle repair values must cover exact shapes")
        repair = {shape_id: float(repair_values[shape_id]) for shape_id in shape_by_id}
        if any(not math.isfinite(value) or value < 0.0 for value in repair.values()):
            raise ValueError("bundle repair values must be finite and nonnegative")
    repair_probability = {}
    if repair_probabilities is not None:
        repair_probability = {key: float(value) for key, value in repair_probabilities.items()}
        prediction_keys = {(prediction.shape_id, prediction.candidate_hash) for prediction in predictions}
        if any(key not in prediction_keys for key in repair_probability):
            raise ValueError("bundle repair probabilities must refer to predicted pairs")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in repair_probability.values()):
            raise ValueError("bundle repair probabilities must be finite values in [0, 1]")
    prediction_by_key = {(prediction.shape_id, prediction.candidate_hash): prediction for prediction in predictions}
    options = []
    for candidate_hash, candidate in sorted(candidate_by_hash.items()):
        pair_predictions = [
            prediction_by_key[(shape.id, candidate_hash)]
            for shape in shapes
            if (shape.id, candidate_hash) in prediction_by_key
            and (shape.id, candidate_hash) not in controller.queried_pairs
        ]
        ranked = sorted(
            pair_predictions,
            key=lambda prediction: (
                -_pair_priority(
                    prediction,
                    controller,
                    weights,
                    repair,
                    repair_probability,
                    policy,
                ),
                prediction.shape_id,
            ),
        )
        sizes = [size for size in policy.bundle_sizes if size <= len(ranked)]
        if ranked and len(ranked) not in sizes:
            sizes.append(len(ranked))
        for size in sizes:
            selected_predictions = tuple(ranked[:size])
            selected_shapes = tuple(shape_by_id[prediction.shape_id] for prediction in selected_predictions)
            prepared_shape_ids = set(controller.prepared_artifact_shapes.get(candidate_hash, set()))
            artifact_shape_ids = prepared_shape_ids | {
                artifact_shape.id for shape in selected_shapes for artifact_shape in artifact_scope[shape.id]
            }
            artifact_shapes = tuple(shape_by_id[shape_id] for shape_id in sorted(artifact_shape_ids))
            options.append(
                CandidateBundle(
                    candidate=candidate,
                    shapes=selected_shapes,
                    artifact_shapes=artifact_shapes,
                    predictions=selected_predictions,
                    cost=cost_model.estimate(
                        candidate,
                        selected_shapes,
                        artifact_shapes=artifact_shapes,
                        prepared_shape_ids=prepared_shape_ids,
                    ),
                )
            )
    selected = _lazy_greedy_select(
        controller,
        options,
        policy,
        weights,
        repair,
        repair_probability,
    )
    preparation_order = tuple(
        score.bundle.candidate.hash
        for score in sorted(
            selected,
            key=lambda score: (
                -score.bundle.cost.preparation_s,
                score.selection_index,
                score.bundle.candidate.hash,
            ),
        )
        if score.bundle.cost.preparation_required
    )
    timing_requests = tuple(
        PairRequest(
            score.bundle.candidate,
            shape,
            evidence_stage=policy.evidence_stage,
            min_samples=policy.min_samples,
            priority=score.utility_per_s,
        )
        for score in selected
        for shape in score.bundle.shapes
    )
    artifact_shapes = {score.bundle.candidate.hash: score.bundle.artifact_shapes for score in selected}
    return AcquisitionPlan(
        selected=tuple(selected),
        preparation_order=preparation_order,
        timing_requests=timing_requests,
        artifact_shapes_by_candidate=artifact_shapes,
        predicted_cost_s=sum(score.bundle.cost.total_s for score in selected),
    )


def _pair_priority(
    prediction: PairPrediction,
    controller: CampaignControllerState,
    weights: Mapping[str, float],
    repair: Mapping[str, float],
    repair_probability: Mapping[tuple[str, str], float],
    policy: BundleAcquisitionPolicy,
) -> float:
    incumbent = controller.incumbents.get(prediction.shape_id)
    expected_gain = _expected_pair_gain(prediction, None if incumbent is None else incumbent.performance)
    unresolved = float(incumbent is None) * _unresolved_coverage_value(prediction)
    information = prediction.epistemic_std_log_performance * prediction.validity_probability
    close_probability = repair_probability.get(
        (prediction.shape_id, prediction.candidate_hash),
        prediction.validity_probability,
    )
    return weights[prediction.shape_id] * (
        policy.improvement_weight * expected_gain
        + policy.coverage_weight * unresolved
        + policy.information_weight * information
        + policy.repair_weight * repair[prediction.shape_id] * close_probability
    )


def _lazy_greedy_select(
    controller: CampaignControllerState,
    options: Sequence[CandidateBundle],
    policy: BundleAcquisitionPolicy,
    weights: Mapping[str, float],
    repair: Mapping[str, float],
    repair_probability: Mapping[tuple[str, str], float],
) -> list[BundleScore]:
    covered_gain: dict[tuple[str, int], float] = {}
    active_by_candidate: dict[str, set[int]] = {}
    heap = []
    for index, bundle in enumerate(options):
        active_by_candidate.setdefault(bundle.candidate.hash, set()).add(index)
        components = _bundle_components(
            bundle,
            controller,
            covered_gain,
            weights,
            repair,
            repair_probability,
            policy,
        )
        rate = components[-1]
        heapq.heappush(heap, (-rate, bundle.candidate.hash, tuple(shape.id for shape in bundle.shapes), index))
    selected = []
    selected_indices: set[int] = set()
    selected_candidates: set[str] = set()
    total_pairs = 0
    total_cost = 0.0
    while heap and len(selected) < policy.max_bundles:
        _, _, _, index = heapq.heappop(heap)
        if index in selected_indices:
            continue
        bundle = options[index]
        if bundle.candidate.hash in selected_candidates:
            continue
        components = _bundle_components(
            bundle,
            controller,
            covered_gain,
            weights,
            repair,
            repair_probability,
            policy,
        )
        rate = components[-1]
        next_rate = -heap[0][0] if heap else -math.inf
        if rate + 1e-15 < next_rate:
            heapq.heappush(
                heap,
                (-rate, bundle.candidate.hash, tuple(shape.id for shape in bundle.shapes), index),
            )
            continue
        if rate < policy.min_utility_per_s or components[4] <= 0.0:
            break
        if total_pairs + len(bundle.shapes) > policy.max_pairs:
            selected_indices.add(index)
            continue
        if total_cost + bundle.cost.total_s > policy.max_predicted_cost_s:
            selected_indices.add(index)
            continue
        score = BundleScore(
            bundle=bundle,
            expected_improvement=components[0],
            unresolved_coverage=components[1],
            information_value=components[2],
            repair_value=components[3],
            marginal_utility=components[4],
            utility_per_s=rate,
            selection_index=len(selected),
        )
        selected.append(score)
        selected_indices.update(active_by_candidate[bundle.candidate.hash])
        selected_candidates.add(bundle.candidate.hash)
        total_pairs += len(bundle.shapes)
        total_cost += bundle.cost.total_s
        _update_covered_gain(bundle, controller, covered_gain)
    return selected


def _bundle_components(
    bundle: CandidateBundle,
    controller: CampaignControllerState,
    covered_gain: Mapping[tuple[str, int], float],
    weights: Mapping[str, float],
    repair: Mapping[str, float],
    repair_probability: Mapping[tuple[str, str], float],
    policy: BundleAcquisitionPolicy,
) -> tuple[float, float, float, float, float, float]:
    improvement = 0.0
    coverage = 0.0
    information = 0.0
    repair_value = 0.0
    for prediction in bundle.predictions:
        weight = weights[prediction.shape_id]
        incumbent = controller.incumbents.get(prediction.shape_id)
        gains = _pair_sample_gains(prediction, None if incumbent is None else incumbent.performance)
        if gains:
            improvement += weight * statistics.fmean(
                max(0.0, gain - covered_gain.get((prediction.shape_id, index), 0.0)) for index, gain in enumerate(gains)
            )
        if incumbent is None:
            coverage += weight * _unresolved_coverage_value(prediction)
        information += weight * prediction.epistemic_std_log_performance * prediction.validity_probability
        repair_value += (
            weight
            * repair[prediction.shape_id]
            * repair_probability.get(
                (prediction.shape_id, prediction.candidate_hash),
                prediction.validity_probability,
            )
        )
    utility = (
        policy.improvement_weight * improvement
        + policy.coverage_weight * coverage
        + policy.information_weight * information
        + policy.repair_weight * repair_value
    )
    rate = utility / max(bundle.cost.total_s, 1e-12)
    return improvement, coverage, information, repair_value, utility, rate


def _unresolved_coverage_value(prediction: PairPrediction) -> float:
    if not prediction.posterior_samples:
        return 0.0
    relative_quality = statistics.fmean(math.exp(min(0.0, sample)) for sample in prediction.posterior_samples)
    return relative_quality * prediction.validity_probability


def _expected_pair_gain(prediction: PairPrediction, incumbent_performance: float | None) -> float:
    gains = _pair_sample_gains(prediction, incumbent_performance)
    return statistics.fmean(gains) if gains else 0.0


def _pair_sample_gains(
    prediction: PairPrediction,
    incumbent_performance: float | None,
) -> tuple[float, ...]:
    if not prediction.posterior_samples:
        return ()
    if incumbent_performance is None:
        return tuple(0.0 for _ in prediction.posterior_samples)
    reference = prediction.reference_performance
    if reference is None or reference <= 0.0:
        return tuple(0.0 for _ in prediction.posterior_samples)
    incumbent_log = math.log(incumbent_performance / reference)
    return tuple(
        max(0.0, sample - incumbent_log) * prediction.validity_probability for sample in prediction.posterior_samples
    )


def _update_covered_gain(
    bundle: CandidateBundle,
    controller: CampaignControllerState,
    covered_gain: dict[tuple[str, int], float],
) -> None:
    for prediction in bundle.predictions:
        incumbent = controller.incumbents.get(prediction.shape_id)
        gains = _pair_sample_gains(prediction, None if incumbent is None else incumbent.performance)
        for index, gain in enumerate(gains):
            key = prediction.shape_id, index
            covered_gain[key] = max(covered_gain.get(key, 0.0), gain)
