import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from evotensile.candidate import Candidate, Shape
from evotensile.database import EvoTensileDB
from evotensile.search.encoding import PARAM_NAMES, candidate_to_genome, hamming_distance
from evotensile.search.family import family_descriptor
from evotensile.search.mechanics import candidate_shape_mechanics, select_covering_cold_pool
from evotensile.search_space import _valu_vgpr_lower_bound, macro_tile

DEFAULT_SURROGATE_MIN_EVIDENCE = 24
DEFAULT_SURROGATE_POOL_MULTIPLIER = 1


@dataclass(frozen=True)
class CandidatePrediction:
    candidate: Candidate
    mean_log_time: float
    std_log_time: float

    @property
    def acquisition(self) -> float:
        return self.mean_log_time - 0.5 * self.std_log_time


class ExtraTreesSurrogate:
    def __init__(self, *, seed: int = 0) -> None:
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
            n_jobs=-1,
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


def candidate_shape_features(candidate: Candidate, shape: Shape) -> dict[str, float | str]:
    params = candidate.canonical_params()
    macro_tile0, macro_tile1 = macro_tile(params["MatrixInstruction"])
    workgroup_threads = math.prod(params["WorkGroup"])
    mechanics = candidate_shape_mechanics(candidate, shape)
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


def _training_rows(
    db: EvoTensileDB,
    *,
    problem_type_hash: str | None,
    benchmark_protocol_hash: str | None,
    shapes: Sequence[Shape] | None,
) -> tuple[list[dict[str, float | str]], list[float]]:
    allowed_shapes = {shape.id: shape for shape in shapes} if shapes is not None else {}
    summaries = db.rank_evaluations(
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        min_samples=1,
        limit=None,
    )
    candidate_hashes = sorted({summary.candidate_hash for summary in summaries})
    candidates = {candidate.hash: candidate for candidate in db.get_candidates(candidate_hashes)}
    rows = []
    targets = []
    for summary in summaries:
        candidate = candidates.get(summary.candidate_hash)
        if candidate is None or summary.median_time_us is None or summary.median_time_us <= 0.0:
            continue
        shape = allowed_shapes.get(summary.shape_id)
        if shapes is not None and shape is None:
            continue
        if shape is None:
            with db.connection() as con:
                row = con.execute(
                    "SELECT m, n, batch, k FROM shapes WHERE shape_id = ?",
                    (summary.shape_id,),
                ).fetchone()
            if row is None:
                continue
            shape = Shape(int(row["m"]), int(row["n"]), int(row["batch"]), int(row["k"]))
        rows.append(candidate_shape_features(candidate, shape))
        targets.append(math.log(summary.median_time_us))
    return rows, targets


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


def select_surrogate_pool(
    candidates: Sequence[Candidate],
    *,
    db: EvoTensileDB,
    problem_type_hash: str | None,
    benchmark_protocol_hash: str | None,
    shapes: Sequence[Shape],
    count: int,
    seed: int,
    min_evidence: int = DEFAULT_SURROGATE_MIN_EVIDENCE,
    covering_cold_start: bool = False,
    cold_start_precovered_tokens: set[str] | None = None,
) -> list[Candidate]:
    deduped = list({candidate.hash: candidate for candidate in candidates}.values())
    if count <= 0:
        return []
    if len(deduped) <= count:
        return deduped
    if not shapes:
        return _diverse_fallback(deduped, count=count, seed=seed)
    training_rows, targets = _training_rows(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        shapes=shapes,
    )
    if len(training_rows) < min_evidence:
        if covering_cold_start and len(shapes) == 1:
            return select_covering_cold_pool(
                deduped,
                shape=shapes[0],
                count=count,
                seed=seed,
                precovered_tokens=cold_start_precovered_tokens,
            )
        return _diverse_fallback(deduped, count=count, seed=seed)

    model = ExtraTreesSurrogate(seed=seed)
    model.fit(training_rows, targets)
    candidate_rows = [candidate_shape_features(candidate, shape) for candidate in deduped for shape in shapes]
    means, standard_deviations = model.predict(candidate_rows)
    predictions = []
    shape_count = len(shapes)
    for index, candidate in enumerate(deduped):
        start = index * shape_count
        candidate_means = means[start : start + shape_count]
        candidate_stds = standard_deviations[start : start + shape_count]
        mean = sum(candidate_means) / shape_count
        within_variance = sum(value * value for value in candidate_stds) / shape_count
        between_variance = sum((value - mean) ** 2 for value in candidate_means) / shape_count
        predictions.append(
            CandidatePrediction(
                candidate=candidate,
                mean_log_time=mean,
                std_log_time=math.sqrt(within_variance + between_variance),
            )
        )

    exploit_count = max(1, int(count * 0.55))
    uncertainty_count = max(1, int(count * 0.20))
    diversity_count = max(1, int(count * 0.15))
    selected: list[Candidate] = []
    _append_unique(
        selected,
        [item.candidate for item in sorted(predictions, key=lambda item: (item.acquisition, item.candidate.hash))],
        exploit_count,
    )
    _append_unique(
        selected,
        [
            item.candidate
            for item in sorted(
                predictions, key=lambda item: (-item.std_log_time, item.mean_log_time, item.candidate.hash)
            )
        ],
        exploit_count + uncertainty_count,
    )
    remaining = [candidate for candidate in deduped if candidate.hash not in {item.hash for item in selected}]
    _append_unique(
        selected,
        _diverse_fallback(remaining, count=min(diversity_count, len(remaining)), seed=seed + 1),
        exploit_count + uncertainty_count + diversity_count,
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
