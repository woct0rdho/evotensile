import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from evotensile.candidate import Candidate, Shape
from evotensile.search.encoding import PARAM_NAMES, candidate_to_genome, hamming_distance
from evotensile.search.evidence import ProposalEvidenceSnapshot
from evotensile.search.family import family_descriptor
from evotensile.search.mechanics import candidate_shape_mechanics, mechanical_prior_score, select_covering_cold_pool
from evotensile.search_space import _valu_vgpr_lower_bound, macro_tile

DEFAULT_SURROGATE_MIN_EVIDENCE = 24


@dataclass(frozen=True)
class TrainingObservation:
    candidate_hash: str
    shape_id: str
    features: Mapping[str, Any]
    log_time: float


@dataclass(frozen=True)
class ShapePrediction:
    shape_id: str
    mean_log_time: float
    epistemic_std_log_time: float
    incumbent_log_time: float

    @property
    def predicted_gain(self) -> float:
        return max(0.0, self.incumbent_log_time - self.mean_log_time)

    @property
    def optimistic_gain(self) -> float:
        return max(0.0, self.incumbent_log_time - self.mean_log_time + 0.5 * self.epistemic_std_log_time)


@dataclass(frozen=True)
class GridCandidatePrediction:
    candidate: Candidate
    by_shape: tuple[ShapePrediction, ...]

    @property
    def mean_regret(self) -> float:
        return sum(item.mean_log_time - item.incumbent_log_time for item in self.by_shape) / len(self.by_shape)

    @property
    def mean_gain(self) -> float:
        return sum(item.predicted_gain for item in self.by_shape) / len(self.by_shape)

    @property
    def maximum_uncertainty(self) -> float:
        return max(item.epistemic_std_log_time for item in self.by_shape)


class ExtraTreesSurrogate:
    def __init__(self, *, seed: int = 0, jobs: int) -> None:
        try:
            from sklearn.ensemble import ExtraTreesRegressor
            from sklearn.feature_extraction import DictVectorizer
        except ImportError as exc:  # pragma: no cover - exercised only in minimal installations
            raise RuntimeError("surrogate search requires scikit-learn") from exc
        self._vectorizer = DictVectorizer(sparse=False)
        self._model = ExtraTreesRegressor(
            n_estimators=192,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=seed,
            n_jobs=jobs,
        )

    def fit(self, rows: Sequence[Mapping[str, Any]], targets: Sequence[float]) -> None:
        matrix = self._vectorizer.fit_transform(rows)
        self._model.fit(matrix, targets)

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float]]:
        matrix = self._vectorizer.transform(rows)
        tree_predictions = [estimator.predict(matrix) for estimator in self._model.estimators_]
        means = []
        standard_deviations = []
        for column in zip(*tree_predictions, strict=True):
            mean = sum(float(value) for value in column) / len(column)
            variance = sum((float(value) - mean) ** 2 for value in column) / len(column)
            means.append(mean)
            standard_deviations.append(math.sqrt(variance))
        return means, standard_deviations


def candidate_shape_features(
    candidate: Candidate,
    shape: Shape,
    *,
    effective_cu_count: int,
) -> dict[str, float | str]:
    params = candidate.canonical_params()
    macro_tile0, macro_tile1 = macro_tile(params["MatrixInstruction"])
    workgroup_threads = math.prod(params["WorkGroup"])
    mechanics = candidate_shape_mechanics(candidate, shape, effective_cu_count=effective_cu_count)
    features: dict[str, float | str] = {
        f"gene:{name}": json.dumps(params[name], sort_keys=True, separators=(",", ":")) for name in PARAM_NAMES
    }
    features.update(
        {
            "shape:log2_m": math.log2(shape.m),
            "shape:log2_n": math.log2(shape.n),
            "shape:log2_k": math.log2(shape.k),
            "shape:log2_batch": math.log2(shape.batch),
            "shape:log2_m_over_n": math.log2(shape.m / shape.n),
            "tile:log2_m": math.log2(macro_tile0),
            "tile:log2_n": math.log2(macro_tile1),
            "tile:log2_area": math.log2(macro_tile0 * macro_tile1),
            "tile:aspect_log2": math.log2(macro_tile0 / macro_tile1),
            "tile:m_remainder_fraction": (shape.m % macro_tile0) / macro_tile0,
            "tile:n_remainder_fraction": (shape.n % macro_tile1) / macro_tile1,
            "tile:fill_m": mechanics["tile_fill_m"],
            "tile:fill_n": mechanics["tile_fill_n"],
            "grid:log2_tiles": math.log2(mechanics["output_tiles"]),
            "grid:log2_workgroups": math.log2(mechanics["workgroups"]),
            "grid:tiles_per_cu": mechanics["tiles_per_cu"],
            "grid:log2_cu_rounds": math.log2(mechanics["cu_rounds"]),
            "grid:cu_granularity": mechanics["cu_granularity"],
            "reduction:log2_iterations": math.log2(mechanics["reduction_iterations"]),
            "reduction:k_fill": mechanics["k_fill"],
            "workgroup:log2_threads": math.log2(workgroup_threads),
            "workgroup:waves": mechanics["waves_per_workgroup"],
            "workgroup:wave_tile_area": mechanics["wave_tile_area"],
            "workgroup:wave_group_size": mechanics["wave_group_size"],
            "resource:valu_vgpr_lower_bound": float(_valu_vgpr_lower_bound(params)),
            "resource:valu_vgpr_fraction": mechanics["valu_vgpr_fraction"],
            "resource:lds_bytes": mechanics["lds_bytes"],
            "resource:lds_fraction": mechanics["lds_fraction"],
            "resource:workspace_fraction": mechanics["workspace_fraction"],
            "shape:arithmetic_intensity": mechanics["arithmetic_intensity"],
            "vector:a_bytes": float(params["VectorWidthA"] * 2),
            "vector:b_bytes": float(params["VectorWidthB"] * 2),
            "vector:global_a_bytes": float(params["GlobalReadVectorWidthA"] * 2),
            "vector:global_b_bytes": float(params["GlobalReadVectorWidthB"] * 2),
            "store:auto_batch": "1" if params["NumElementsPerBatchStore"] == 0 else "0",
        }
    )
    return features


def _training_observations(
    snapshot: ProposalEvidenceSnapshot,
    *,
    shapes: Sequence[Shape] | None,
    effective_cu_count: int,
) -> list[TrainingObservation]:
    allowed_shapes = {shape.id: shape for shape in shapes} if shapes is not None else {}
    summaries = snapshot.summaries
    candidates = snapshot.candidates
    observations = []
    for summary in summaries:
        candidate = candidates.get(summary.candidate_hash)
        if candidate is None or summary.median_time_us is None or summary.median_time_us <= 0.0:
            continue
        shape = allowed_shapes.get(summary.shape_id)
        if shapes is not None and shape is None:
            continue
        if shape is None:
            continue
        observations.append(
            TrainingObservation(
                candidate_hash=candidate.hash,
                shape_id=shape.id,
                features=candidate_shape_features(
                    candidate,
                    shape,
                    effective_cu_count=effective_cu_count,
                ),
                log_time=math.log(summary.median_time_us),
            )
        )
    return observations


def _has_candidate_feature_diversity(observations: Sequence[TrainingObservation]) -> bool:
    if not observations:
        return False
    varying_genes = 0
    for name in PARAM_NAMES:
        values = {observation.features[f"gene:{name}"] for observation in observations}
        if len(values) > 1:
            varying_genes += 1
    return varying_genes >= min(4, len(PARAM_NAMES))


def surrogate_model_shape_ids(
    observations: Sequence[TrainingObservation],
    *,
    shapes: Sequence[Shape],
    min_evidence: int,
) -> tuple[str, ...]:
    by_shape: dict[str, list[TrainingObservation]] = defaultdict(list)
    for observation in observations:
        by_shape[observation.shape_id].append(observation)
    return tuple(
        shape.id
        for shape in shapes
        if len({observation.candidate_hash for observation in by_shape.get(shape.id, [])}) >= min_evidence
        and _has_candidate_feature_diversity(by_shape[shape.id])
    )


def _shape_models(
    observations: Sequence[TrainingObservation],
    *,
    shapes: Sequence[Shape],
    min_evidence: int,
    seed: int,
    surrogate_jobs: int,
) -> tuple[dict[str, ExtraTreesSurrogate], dict[str, float]]:
    by_shape: dict[str, list[TrainingObservation]] = defaultdict(list)
    for observation in observations:
        by_shape[observation.shape_id].append(observation)
    modeled_shape_ids = set(surrogate_model_shape_ids(observations, shapes=shapes, min_evidence=min_evidence))
    models: dict[str, ExtraTreesSurrogate] = {}
    incumbents: dict[str, float] = {}
    for shape_index, shape in enumerate(shapes):
        if shape.id not in modeled_shape_ids:
            continue
        shape_observations = by_shape[shape.id]
        model = ExtraTreesSurrogate(seed=seed + shape_index, jobs=surrogate_jobs)
        model.fit(
            [observation.features for observation in shape_observations],
            [observation.log_time for observation in shape_observations],
        )
        models[shape.id] = model
        incumbents[shape.id] = min(observation.log_time for observation in shape_observations)
    return models, incumbents


def _grid_predictions(
    candidates: Sequence[Candidate],
    *,
    shapes: Sequence[Shape],
    models: Mapping[str, ExtraTreesSurrogate],
    incumbents: Mapping[str, float],
    effective_cu_count: int,
) -> list[GridCandidatePrediction]:
    predictions_by_hash: dict[str, list[ShapePrediction]] = {candidate.hash: [] for candidate in candidates}
    for shape in shapes:
        model = models.get(shape.id)
        if model is None:
            continue
        means, standard_deviations = model.predict(
            [
                candidate_shape_features(
                    candidate,
                    shape,
                    effective_cu_count=effective_cu_count,
                )
                for candidate in candidates
            ]
        )
        for candidate, mean, standard_deviation in zip(candidates, means, standard_deviations, strict=True):
            predictions_by_hash[candidate.hash].append(
                ShapePrediction(
                    shape_id=shape.id,
                    mean_log_time=mean,
                    epistemic_std_log_time=standard_deviation,
                    incumbent_log_time=incumbents[shape.id],
                )
            )
    return [
        GridCandidatePrediction(candidate=candidate, by_shape=tuple(predictions_by_hash[candidate.hash]))
        for candidate in candidates
    ]


def _diverse_fallback(candidates: Sequence[Candidate], *, count: int, seed: int) -> list[Candidate]:
    if len(candidates) <= count:
        return list(candidates)
    rng = random.Random(seed)
    by_family: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_family[family_descriptor(candidate).key].append(candidate)
    for family_candidates in by_family.values():
        rng.shuffle(family_candidates)
    selected = []
    family_keys = list(by_family)
    rng.shuffle(family_keys)
    while len(selected) < count and family_keys:
        next_keys = []
        for key in family_keys:
            if by_family[key] and len(selected) < count:
                selected.append(by_family[key].pop())
            if by_family[key]:
                next_keys.append(key)
        family_keys = next_keys
    return selected


def _append_unique(selected: list[Candidate], candidates: Sequence[Candidate], count: int) -> None:
    seen = {candidate.hash for candidate in selected}
    for candidate in candidates:
        if len(selected) >= count:
            return
        if candidate.hash not in seen:
            selected.append(candidate)
            seen.add(candidate.hash)


def _marginal_grid_gain_order(predictions: Sequence[GridCandidatePrediction]) -> list[Candidate]:
    remaining = list(predictions)
    covered_gain: dict[str, float] = defaultdict(float)
    ordered: list[Candidate] = []
    while remaining:
        chosen = max(
            remaining,
            key=lambda prediction: (
                sum(max(0.0, item.optimistic_gain - covered_gain[item.shape_id]) for item in prediction.by_shape),
                prediction.mean_gain,
                -prediction.mean_regret,
                prediction.candidate.hash,
            ),
        )
        ordered.append(chosen.candidate)
        for item in chosen.by_shape:
            covered_gain[item.shape_id] = max(covered_gain[item.shape_id], item.optimistic_gain)
        remaining.remove(chosen)
    return ordered


def _unresolved_shape_order(
    candidates: Sequence[Candidate],
    shapes: Sequence[Shape],
    *,
    effective_cu_count: int,
) -> list[Candidate]:
    queues = [
        sorted(
            candidates,
            key=lambda candidate: (
                -mechanical_prior_score(candidate, shape, effective_cu_count=effective_cu_count),
                candidate.hash,
            ),
        )
        for shape in shapes
    ]
    selected: list[Candidate] = []
    seen: set[str] = set()
    rank = 0
    while len(seen) < len(candidates):
        added = False
        for queue in queues:
            if rank >= len(queue):
                continue
            candidate = queue[rank]
            if candidate.hash not in seen:
                selected.append(candidate)
                seen.add(candidate.hash)
                added = True
        if not added and all(rank >= len(queue) - 1 for queue in queues):
            break
        rank += 1
    return selected


def select_surrogate_pool(
    candidates: Sequence[Candidate],
    *,
    evidence: ProposalEvidenceSnapshot,
    shapes: Sequence[Shape],
    count: int,
    seed: int,
    min_evidence: int = DEFAULT_SURROGATE_MIN_EVIDENCE,
    covering_cold_start: bool = False,
    cold_start_precovered_tokens: set[str] | None = None,
    surrogate_jobs: int,
    effective_cu_count: int,
) -> list[Candidate]:
    deduped = list({candidate.hash: candidate for candidate in candidates}.values())
    if count <= 0:
        return []
    if len(deduped) <= count:
        return deduped
    if not shapes:
        return _diverse_fallback(deduped, count=count, seed=seed)
    observations = _training_observations(
        evidence,
        shapes=shapes,
        effective_cu_count=effective_cu_count,
    )
    models, incumbents = _shape_models(
        observations,
        shapes=shapes,
        min_evidence=min_evidence,
        seed=seed,
        surrogate_jobs=surrogate_jobs,
    )
    if not models:
        if covering_cold_start and len(shapes) == 1:
            return select_covering_cold_pool(
                deduped,
                shape=shapes[0],
                count=count,
                seed=seed,
                precovered_tokens=cold_start_precovered_tokens,
                effective_cu_count=effective_cu_count,
            )
        if len(shapes) > 1:
            selected: list[Candidate] = []
            _append_unique(
                selected,
                _unresolved_shape_order(
                    deduped,
                    shapes,
                    effective_cu_count=effective_cu_count,
                ),
                max(1, count // 2),
            )
            remaining = [candidate for candidate in deduped if candidate.hash not in {item.hash for item in selected}]
            _append_unique(selected, _diverse_fallback(remaining, count=count, seed=seed), count)
            return selected
        return _diverse_fallback(deduped, count=count, seed=seed)

    predictions = _grid_predictions(
        deduped,
        shapes=shapes,
        models=models,
        incumbents=incumbents,
        effective_cu_count=effective_cu_count,
    )
    specialist_count = max(1, int(count * 0.35))
    generalist_count = max(1, int(count * 0.25))
    uncertainty_count = max(1, int(count * 0.15))
    unresolved_count = max(1, int(count * 0.10))
    selected: list[Candidate] = []
    _append_unique(selected, _marginal_grid_gain_order(predictions), specialist_count)
    _append_unique(
        selected,
        [
            item.candidate
            for item in sorted(predictions, key=lambda item: (item.mean_regret, -item.mean_gain, item.candidate.hash))
        ],
        specialist_count + generalist_count,
    )
    _append_unique(
        selected,
        [
            item.candidate
            for item in sorted(
                predictions,
                key=lambda item: (-item.maximum_uncertainty, item.mean_regret, item.candidate.hash),
            )
        ],
        specialist_count + generalist_count + uncertainty_count,
    )
    unresolved_shapes = [shape for shape in shapes if shape.id not in models]
    if unresolved_shapes:
        _append_unique(
            selected,
            _unresolved_shape_order(
                deduped,
                unresolved_shapes,
                effective_cu_count=effective_cu_count,
            ),
            specialist_count + generalist_count + uncertainty_count + unresolved_count,
        )
    remaining = [candidate for candidate in deduped if candidate.hash not in {item.hash for item in selected}]
    diversity_target = max(len(selected), min(count, int(count * 0.90)))
    _append_unique(
        selected,
        _diverse_fallback(remaining, count=min(count - len(selected), len(remaining)), seed=seed + 1),
        diversity_target,
    )
    remaining = [candidate for candidate in deduped if candidate.hash not in {item.hash for item in selected}]
    random.Random(seed + 2).shuffle(remaining)
    _append_unique(selected, remaining, count)
    if len(selected) < count:
        genomes = {candidate.hash: candidate_to_genome(candidate) for candidate in deduped}
        remaining = [candidate for candidate in deduped if candidate.hash not in {item.hash for item in selected}]
        while remaining and len(selected) < count:
            chosen = max(
                remaining,
                key=lambda candidate: min(
                    hamming_distance(genomes[candidate.hash], genomes[selected_candidate.hash])
                    for selected_candidate in selected
                ),
            )
            selected.append(chosen)
            remaining.remove(chosen)
    return selected[:count]
