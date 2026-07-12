import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_extraction import DictVectorizer

from evotensile.campaign.evaluator import PairEvaluationOutcome
from evotensile.candidate import Candidate, Shape
from evotensile.search.surrogate import candidate_shape_features


class ExtraTreesRegressorParameters(TypedDict):
    n_estimators: int
    min_samples_leaf: int
    max_features: float
    bootstrap: bool
    max_samples: float
    random_state: int
    n_jobs: int


@dataclass(frozen=True)
class PairModelConfiguration:
    n_estimators: int = 192
    min_samples_leaf: int = 2
    max_features: float = 0.7
    bootstrap_fraction: float = 0.8
    min_performance_rows: int = 24
    seed: int = 12345
    jobs: int = 1

    def __post_init__(self) -> None:
        if self.n_estimators <= 0 or self.min_samples_leaf <= 0 or self.min_performance_rows <= 0:
            raise ValueError("pair-model estimator and evidence counts must be positive")
        if not 0.0 < self.max_features <= 1.0:
            raise ValueError("pair-model max_features must be in (0, 1]")
        if not 0.0 < self.bootstrap_fraction <= 1.0:
            raise ValueError("pair-model bootstrap fraction must be in (0, 1]")
        if self.jobs == 0:
            raise ValueError("pair-model jobs cannot be zero")

    def to_dict(self) -> dict[str, object]:
        return {
            "n_estimators": self.n_estimators,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap_fraction": self.bootstrap_fraction,
            "min_performance_rows": self.min_performance_rows,
            "seed": self.seed,
            "jobs": self.jobs,
        }


@dataclass(frozen=True)
class PairModelFitSummary:
    performance_rows: int
    validity_rows: int
    valid_rows: int
    invalid_rows: int
    candidate_count: int
    shape_count: int
    feature_count: int
    feature_contract_hash: str
    uncertainty_scale: float
    residual_floor: float
    shape_references: dict[str, float]
    configuration: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "performance_rows": self.performance_rows,
            "validity_rows": self.validity_rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
            "candidate_count": self.candidate_count,
            "shape_count": self.shape_count,
            "feature_count": self.feature_count,
            "feature_contract_hash": self.feature_contract_hash,
            "uncertainty_scale": self.uncertainty_scale,
            "residual_floor": self.residual_floor,
            "shape_references": dict(sorted(self.shape_references.items())),
            "configuration": self.configuration,
        }


@dataclass(frozen=True)
class PairPrediction:
    shape_id: str
    candidate_hash: str
    mean_normalized_log_performance: float
    epistemic_std_log_performance: float
    validity_probability: float
    posterior_samples: tuple[float, ...]
    reference_performance: float | None

    @property
    def predicted_performance(self) -> float | None:
        if self.reference_performance is None:
            return None
        return self.reference_performance * math.exp(self.mean_normalized_log_performance)

    def probability_of_improvement(
        self,
        *,
        incumbent_performance: float | None = None,
        minimum_gain_fraction: float = 0.0,
    ) -> float:
        if minimum_gain_fraction < 0.0:
            raise ValueError("minimum gain fraction must be nonnegative")
        if not self.posterior_samples:
            return 0.0
        reference = self.reference_performance
        if incumbent_performance is None:
            if reference is None:
                threshold = 0.0
            else:
                incumbent_performance = reference
        if incumbent_performance is not None:
            if reference is None or incumbent_performance <= 0.0:
                raise ValueError("probability of improvement requires a positive compatible reference")
            threshold = math.log(incumbent_performance * (1.0 + minimum_gain_fraction) / reference)
        else:
            threshold = math.log1p(minimum_gain_fraction)
        posterior_probability = sum(sample > threshold for sample in self.posterior_samples) / len(
            self.posterior_samples
        )
        return posterior_probability * self.validity_probability


@dataclass(frozen=True)
class PairModelMetrics:
    pairs: int
    shapes: int
    candidates: int
    mean_absolute_normalized_log_error: float | None
    interval_50_coverage: float | None
    interval_80_coverage: float | None
    interval_90_coverage: float | None
    mean_shape_rank_correlation: float | None
    top_k: int
    mean_shape_top_k_recall: float | None
    validity_brier: float | None
    probability_improvement_brier: float | None
    probability_improvement_calibration_error: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "pairs": self.pairs,
            "shapes": self.shapes,
            "candidates": self.candidates,
            "mean_absolute_normalized_log_error": self.mean_absolute_normalized_log_error,
            "interval_50_coverage": self.interval_50_coverage,
            "interval_80_coverage": self.interval_80_coverage,
            "interval_90_coverage": self.interval_90_coverage,
            "mean_shape_rank_correlation": self.mean_shape_rank_correlation,
            "top_k": self.top_k,
            "mean_shape_top_k_recall": self.mean_shape_top_k_recall,
            "validity_brier": self.validity_brier,
            "probability_improvement_brier": self.probability_improvement_brier,
            "probability_improvement_calibration_error": self.probability_improvement_calibration_error,
        }


class ContextualPairModel:
    def __init__(
        self,
        *,
        workgroup_processor_count: int,
        configuration: PairModelConfiguration | None = None,
    ) -> None:
        if workgroup_processor_count <= 0:
            raise ValueError("pair model requires a positive work-group processor count")
        self.workgroup_processor_count = workgroup_processor_count
        self.configuration = configuration or PairModelConfiguration()
        self._vectorizer: DictVectorizer | None = None
        self._performance_model: ExtraTreesRegressor | None = None
        self._validity_model: ExtraTreesClassifier | None = None
        self._constant_validity: float | None = None
        self._references: dict[str, float] = {}
        self._training_candidate_hashes: set[str] = set()
        self._training_shape_ids: set[str] = set()
        self._uncertainty_scale = 1.0
        self._residual_floor = 0.0
        self._fit_summary: PairModelFitSummary | None = None

    @property
    def fitted(self) -> bool:
        return self._fit_summary is not None

    @property
    def fit_summary(self) -> PairModelFitSummary:
        if self._fit_summary is None:
            raise ValueError("pair model has not been fitted")
        return self._fit_summary

    def fit(self, outcomes: Sequence[PairEvaluationOutcome]) -> PairModelFitSummary:
        visible = _latest_visible_outcomes(outcomes)
        validity_rows = [outcome for outcome in visible if outcome.known and outcome.disclosed]
        performance_rows = [
            outcome for outcome in validity_rows if outcome.performance is not None and outcome.performance > 0.0
        ]
        if len(performance_rows) < self.configuration.min_performance_rows:
            raise ValueError(
                f"pair model requires at least {self.configuration.min_performance_rows} positive disclosed pairs"
            )
        self._training_candidate_hashes = {outcome.request.candidate.hash for outcome in validity_rows}
        self._training_shape_ids = {outcome.request.shape.id for outcome in validity_rows}
        self._references = {}
        for outcome in performance_rows:
            shape_id = outcome.request.shape.id
            self._references[shape_id] = max(self._references.get(shape_id, 0.0), outcome.performance or 0.0)
        all_features = [self._features(outcome.request.candidate, outcome.request.shape) for outcome in validity_rows]
        self._vectorizer = DictVectorizer(sparse=False)
        all_matrix = self._vectorizer.fit_transform(all_features)
        performance_indices = [
            index
            for index, outcome in enumerate(validity_rows)
            if outcome.performance is not None and outcome.performance > 0.0
        ]
        performance_matrix = all_matrix[performance_indices]
        targets = [
            math.log(
                (validity_rows[index].performance or 0.0) / self._references[validity_rows[index].request.shape.id]
            )
            for index in performance_indices
        ]
        calibration_indices = [
            index
            for index, outcome_index in enumerate(performance_indices)
            if _stable_fold(validity_rows[outcome_index].key, self.configuration.seed, folds=5) == 0
        ]
        calibration_index_set = set(calibration_indices)
        fit_indices = [index for index in range(len(targets)) if index not in calibration_index_set]
        if len(calibration_indices) >= 8 and len(fit_indices) >= self.configuration.min_performance_rows:
            calibration_model = ExtraTreesRegressor(**self._regressor_parameters(seed=self.configuration.seed + 2))
            calibration_model.fit(performance_matrix[fit_indices], [targets[index] for index in fit_indices])
            calibration_tree_predictions = Parallel(n_jobs=self.configuration.jobs, prefer="threads")(
                delayed(estimator.predict)(performance_matrix[calibration_indices])
                for estimator in calibration_model.estimators_
            )
            calibration_means, calibration_stds = _column_moments(calibration_tree_predictions)
            residuals = [
                abs(targets[index] - mean) for index, mean in zip(calibration_indices, calibration_means, strict=True)
            ]
            ratios = [
                residual / standard_deviation
                for residual, standard_deviation in zip(residuals, calibration_stds, strict=True)
                if standard_deviation > 1e-12
            ]
            self._uncertainty_scale = 1.15 * min(8.0, max(0.25, _quantile(ratios, 0.90) / 1.6449)) if ratios else 1.15
            median_residual = statistics.median(residuals)
            self._residual_floor = 1.15 * median_residual / 0.6745 if median_residual > 0.0 else 0.0
        else:
            self._uncertainty_scale = 1.0
            self._residual_floor = 0.0
        self._performance_model = ExtraTreesRegressor(**self._regressor_parameters(seed=self.configuration.seed))
        self._performance_model.fit(performance_matrix, targets)

        validity_targets = [
            int(outcome.performance is not None and outcome.performance > 0.0) for outcome in validity_rows
        ]
        classes = set(validity_targets)
        if len(classes) == 1:
            self._constant_validity = float(validity_targets[0])
            self._validity_model = None
        else:
            self._constant_validity = None
            self._validity_model = ExtraTreesClassifier(
                n_estimators=self.configuration.n_estimators,
                min_samples_leaf=self.configuration.min_samples_leaf,
                max_features=self.configuration.max_features,
                bootstrap=True,
                max_samples=self.configuration.bootstrap_fraction,
                random_state=self.configuration.seed + 1,
                n_jobs=self.configuration.jobs,
                class_weight="balanced",
            )
            self._validity_model.fit(all_matrix, validity_targets)
        feature_names = tuple(str(name) for name in self._vectorizer.get_feature_names_out())
        feature_contract_hash = hashlib.sha256(
            json.dumps(
                {
                    "features": feature_names,
                    "configuration": self.configuration.to_dict(),
                    "workgroup_processor_count": self.workgroup_processor_count,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        valid_rows = sum(validity_targets)
        self._fit_summary = PairModelFitSummary(
            performance_rows=len(performance_rows),
            validity_rows=len(validity_rows),
            valid_rows=valid_rows,
            invalid_rows=len(validity_rows) - valid_rows,
            candidate_count=len({outcome.request.candidate.hash for outcome in validity_rows}),
            shape_count=len({outcome.request.shape.id for outcome in validity_rows}),
            feature_count=len(feature_names),
            feature_contract_hash=feature_contract_hash,
            uncertainty_scale=self._uncertainty_scale,
            residual_floor=self._residual_floor,
            shape_references=dict(self._references),
            configuration=self.configuration.to_dict(),
        )
        return self._fit_summary

    def predict(self, requests: Sequence[tuple[Candidate, Shape]]) -> tuple[PairPrediction, ...]:
        if self._fit_summary is None or self._vectorizer is None or self._performance_model is None:
            raise ValueError("pair model has not been fitted")
        matrix = self._vectorizer.transform([self._features(candidate, shape) for candidate, shape in requests])
        tree_predictions = Parallel(n_jobs=self.configuration.jobs, prefer="threads")(
            delayed(estimator.predict)(matrix) for estimator in self._performance_model.estimators_
        )
        means, raw_stds = _column_moments(tree_predictions)
        if self._constant_validity is not None:
            validity = [self._constant_validity] * len(requests)
        elif self._validity_model is not None:
            probabilities = self._validity_model.predict_proba(matrix)
            positive_index = list(self._validity_model.classes_).index(1)
            validity = [float(row[positive_index]) for row in probabilities]
        else:
            raise ValueError("pair model validity state is incomplete")
        predictions = []
        for index, ((candidate, shape), mean, raw_std, valid_probability) in enumerate(
            zip(requests, means, raw_stds, validity, strict=True)
        ):
            novelty_factor = 1.0
            if candidate.hash not in self._training_candidate_hashes:
                novelty_factor *= 1.35
            if shape.id not in self._training_shape_ids:
                novelty_factor *= 1.15
            calibrated_std = max(raw_std * self._uncertainty_scale, self._residual_floor) * novelty_factor
            spread_factor = calibrated_std / raw_std if raw_std > 1e-12 else 1.0
            predictions.append(
                PairPrediction(
                    shape_id=shape.id,
                    candidate_hash=candidate.hash,
                    mean_normalized_log_performance=mean,
                    epistemic_std_log_performance=calibrated_std,
                    validity_probability=valid_probability,
                    posterior_samples=tuple(
                        mean + (float(tree[index]) - mean) * spread_factor for tree in tree_predictions
                    ),
                    reference_performance=self._references.get(shape.id),
                )
            )
        return tuple(predictions)

    def _regressor_parameters(self, *, seed: int) -> ExtraTreesRegressorParameters:
        return {
            "n_estimators": self.configuration.n_estimators,
            "min_samples_leaf": self.configuration.min_samples_leaf,
            "max_features": self.configuration.max_features,
            "bootstrap": True,
            "max_samples": self.configuration.bootstrap_fraction,
            "random_state": seed,
            "n_jobs": self.configuration.jobs,
        }

    def _features(self, candidate: Candidate, shape: Shape) -> dict[str, float | str]:
        return candidate_shape_features(
            candidate,
            shape,
            workgroup_processor_count=self.workgroup_processor_count,
        )


def evaluate_pair_predictions(
    predictions: Sequence[PairPrediction],
    outcomes: Sequence[PairEvaluationOutcome],
    *,
    top_k: int = 3,
    minimum_gain_fraction: float = 0.0,
) -> PairModelMetrics:
    if top_k <= 0:
        raise ValueError("pair-model top-k must be positive")
    prediction_by_key = {(prediction.shape_id, prediction.candidate_hash): prediction for prediction in predictions}
    visible = [outcome for outcome in _latest_visible_outcomes(outcomes) if outcome.known and outcome.disclosed]
    valid = [
        outcome
        for outcome in visible
        if outcome.performance is not None and outcome.performance > 0.0 and outcome.key in prediction_by_key
    ]
    best_by_shape: dict[str, float] = {}
    for outcome in valid:
        best_by_shape[outcome.request.shape.id] = max(
            best_by_shape.get(outcome.request.shape.id, 0.0), outcome.performance or 0.0
        )
    errors = []
    standardized_errors = []
    by_shape: dict[str, list[tuple[PairEvaluationOutcome, PairPrediction]]] = defaultdict(list)
    for outcome in valid:
        prediction = prediction_by_key[outcome.key]
        actual = math.log((outcome.performance or 0.0) / best_by_shape[outcome.request.shape.id])
        error = actual - prediction.mean_normalized_log_performance
        errors.append(abs(error))
        standardized_errors.append(abs(error) / max(prediction.epistemic_std_log_performance, 1e-12))
        by_shape[outcome.request.shape.id].append((outcome, prediction))
    rank_correlations = []
    recalls = []
    for pairs in by_shape.values():
        if len(pairs) >= 2:
            actual_values = [outcome.performance or 0.0 for outcome, _ in pairs]
            predicted_values = [prediction.mean_normalized_log_performance for _, prediction in pairs]
            rank_correlations.append(_rank_correlation(actual_values, predicted_values))
        active_k = min(top_k, len(pairs))
        if active_k > 0:
            actual_top = {
                outcome.request.candidate.hash
                for outcome, _ in sorted(
                    pairs,
                    key=lambda item: (-(item[0].performance or 0.0), item[0].request.candidate.hash),
                )[:active_k]
            }
            predicted_top = {
                outcome.request.candidate.hash
                for outcome, _ in sorted(
                    pairs,
                    key=lambda item: (
                        -item[1].mean_normalized_log_performance,
                        item[0].request.candidate.hash,
                    ),
                )[:active_k]
            }
            recalls.append(len(actual_top & predicted_top) / active_k)
    validity_terms = []
    improvement_terms = []
    calibration_rows = []
    for outcome in visible:
        prediction = prediction_by_key.get(outcome.key)
        if prediction is None:
            continue
        actual_valid = float(outcome.performance is not None and outcome.performance > 0.0)
        validity_terms.append((prediction.validity_probability - actual_valid) ** 2)
        if prediction.reference_performance is None or not actual_valid:
            continue
        probability = prediction.probability_of_improvement(minimum_gain_fraction=minimum_gain_fraction)
        threshold = prediction.reference_performance * (1.0 + minimum_gain_fraction)
        actual_improvement = float((outcome.performance or 0.0) > threshold)
        improvement_terms.append((probability - actual_improvement) ** 2)
        calibration_rows.append((probability, actual_improvement))
    return PairModelMetrics(
        pairs=len(valid),
        shapes=len(by_shape),
        candidates=len({outcome.request.candidate.hash for outcome in valid}),
        mean_absolute_normalized_log_error=statistics.fmean(errors) if errors else None,
        interval_50_coverage=_coverage(standardized_errors, 0.6745),
        interval_80_coverage=_coverage(standardized_errors, 1.2816),
        interval_90_coverage=_coverage(standardized_errors, 1.6449),
        mean_shape_rank_correlation=(statistics.fmean(rank_correlations) if rank_correlations else None),
        top_k=top_k,
        mean_shape_top_k_recall=statistics.fmean(recalls) if recalls else None,
        validity_brier=statistics.fmean(validity_terms) if validity_terms else None,
        probability_improvement_brier=(statistics.fmean(improvement_terms) if improvement_terms else None),
        probability_improvement_calibration_error=_calibration_error(calibration_rows),
    )


def _latest_visible_outcomes(
    outcomes: Sequence[PairEvaluationOutcome],
) -> tuple[PairEvaluationOutcome, ...]:
    latest = {}
    for outcome in outcomes:
        if outcome.disclosed:
            latest[outcome.key] = outcome
    return tuple(latest[key] for key in sorted(latest))


def _stable_fold(key: tuple[str, str], seed: int, *, folds: int) -> int:
    payload = f"{seed}/{key[0]}/{key[1]}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % folds


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _column_moments(columns: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    matrix = np.asarray(columns, dtype=float)
    return matrix.mean(axis=0).tolist(), matrix.std(axis=0).tolist()


def _rank_correlation(actual: Sequence[float], predicted: Sequence[float]) -> float:
    actual_ranks = _ranks(actual)
    predicted_ranks = _ranks(predicted)
    actual_mean = statistics.fmean(actual_ranks)
    predicted_mean = statistics.fmean(predicted_ranks)
    numerator = sum(
        (actual_rank - actual_mean) * (predicted_rank - predicted_mean)
        for actual_rank, predicted_rank in zip(actual_ranks, predicted_ranks, strict=True)
    )
    actual_scale = math.sqrt(sum((rank - actual_mean) ** 2 for rank in actual_ranks))
    predicted_scale = math.sqrt(sum((rank - predicted_mean) ** 2 for rank in predicted_ranks))
    denominator = actual_scale * predicted_scale
    return numerator / denominator if denominator > 0.0 else 0.0


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + end - 1) / 2.0
        for original_index, _ in ordered[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


def _coverage(standardized_errors: Sequence[float], threshold: float) -> float | None:
    if not standardized_errors:
        return None
    return sum(value <= threshold for value in standardized_errors) / len(standardized_errors)


def _calibration_error(rows: Sequence[tuple[float, float]], *, bins: int = 10) -> float | None:
    if not rows:
        return None
    total = len(rows)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [row for row in rows if lower <= row[0] < upper or (index == bins - 1 and row[0] == 1.0)]
        if not members:
            continue
        predicted = statistics.fmean(row[0] for row in members)
        actual = statistics.fmean(row[1] for row in members)
        error += len(members) / total * abs(predicted - actual)
    return error
