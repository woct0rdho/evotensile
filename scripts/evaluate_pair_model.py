#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from evotensile.campaign.evaluator import PairEvaluationOutcome
from evotensile.profile import DEFAULT_PROFILE
from evotensile.scheduling.models import PairRequest
from evotensile.search.pair_model import (
    ContextualPairModel,
    PairModelConfiguration,
    PairPrediction,
    evaluate_pair_predictions,
)
from evotensile.search.replay import load_db_oracle_matrix
from evotensile.search.shape_clustering import (
    ShapeClusteringConfiguration,
    cluster_shapes,
    shape_descriptor_distances,
)
from evotensile.search.surrogate import ExtraTreesSurrogate, candidate_shape_features
from evotensile.shapes import pilot_100_shapes


def _fold(value: str, folds: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % folds


def _outcomes(oracle):
    return tuple(
        PairEvaluationOutcome(
            request=PairRequest(record.candidate, shape_by_id[shape_id]),
            provenance="replay",
            source_ref="retained-grid100",
            status=record.status,
            known=True,
            disclosed=True,
            samples=1 if record.screening_gflops is not None else 0,
            performance=record.screening_gflops,
        )
        for (shape_id, _), record in sorted(oracle.items())
    )


def _shared_metrics(train, test, *, seed, estimators):
    model = ContextualPairModel(
        workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
        configuration=PairModelConfiguration(
            n_estimators=estimators,
            min_performance_rows=24,
            seed=seed,
            jobs=DEFAULT_PROFILE.default_surrogate_jobs,
        ),
    )
    summary = model.fit(train)
    predictions = model.predict([(outcome.request.candidate, outcome.request.shape) for outcome in test])
    return summary.to_dict(), evaluate_pair_predictions(predictions, test).to_dict()


def _shape_local_predictions(train, test, *, seed, estimators):
    positive_by_shape = defaultdict(list)
    reference_by_shape = {}
    for outcome in train:
        if outcome.performance is None or outcome.performance <= 0.0:
            continue
        shape_id = outcome.request.shape.id
        positive_by_shape[shape_id].append(outcome)
        reference_by_shape[shape_id] = max(reference_by_shape.get(shape_id, 0.0), outcome.performance)
    predictions = []
    for shape_index, shape in enumerate(shapes):
        rows = positive_by_shape.get(shape.id, [])
        target = [outcome for outcome in test if outcome.request.shape.id == shape.id]
        if len(rows) < 8 or not target:
            continue
        model = ExtraTreesSurrogate(
            seed=seed + shape_index,
            jobs=DEFAULT_PROFILE.default_surrogate_jobs,
            n_estimators=estimators,
        )
        model.fit(
            [
                candidate_shape_features(
                    outcome.request.candidate,
                    shape,
                    workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
                )
                for outcome in rows
            ],
            [math.log((outcome.performance or 0.0) / reference_by_shape[shape.id]) for outcome in rows],
        )
        means, standard_deviations = model.predict(
            [
                candidate_shape_features(
                    outcome.request.candidate,
                    shape,
                    workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
                )
                for outcome in target
            ]
        )
        predictions.extend(
            PairPrediction(
                shape_id=shape.id,
                candidate_hash=outcome.request.candidate.hash,
                mean_normalized_log_performance=mean,
                epistemic_std_log_performance=max(standard_deviation, 1e-6),
                validity_probability=1.0,
                posterior_samples=(mean - standard_deviation, mean, mean + standard_deviation),
                reference_performance=reference_by_shape[shape.id],
            )
            for outcome, mean, standard_deviation in zip(target, means, standard_deviations, strict=True)
        )
    keys = {(prediction.shape_id, prediction.candidate_hash) for prediction in predictions}
    filtered_test = [outcome for outcome in test if outcome.key in keys]
    return evaluate_pair_predictions(predictions, filtered_test).to_dict()


def _nearest_predictions(train, test):
    positive = {
        outcome.key: outcome for outcome in train if outcome.performance is not None and outcome.performance > 0.0
    }
    best_by_shape = {}
    for outcome in positive.values():
        shape_id = outcome.request.shape.id
        best_by_shape[shape_id] = max(best_by_shape.get(shape_id, 0.0), outcome.performance or 0.0)
    predictions = []
    filtered_test = []
    for outcome in test:
        candidate_hash = outcome.request.candidate.hash
        sources = [
            source
            for (shape_id, source_hash), source in positive.items()
            if source_hash == candidate_hash and shape_id != outcome.request.shape.id
        ]
        if not sources:
            continue
        source = min(
            sources,
            key=lambda item: (
                shape_distances[(outcome.request.shape.id, item.request.shape.id)],
                item.request.shape.id,
            ),
        )
        normalized = math.log((source.performance or 0.0) / best_by_shape[source.request.shape.id])
        distance = shape_distances[(outcome.request.shape.id, source.request.shape.id)]
        uncertainty = 0.05 + 0.03 * distance
        predictions.append(
            PairPrediction(
                shape_id=outcome.request.shape.id,
                candidate_hash=candidate_hash,
                mean_normalized_log_performance=normalized,
                epistemic_std_log_performance=uncertainty,
                validity_probability=1.0,
                posterior_samples=(normalized - uncertainty, normalized, normalized + uncertainty),
                reference_performance=None,
            )
        )
        filtered_test.append(outcome)
    return evaluate_pair_predictions(predictions, filtered_test).to_dict()


def _low_rank_predictions(train, test, *, rank=8, iterations=8):
    import numpy as np

    candidate_ids = sorted({outcome.request.candidate.hash for outcome in (*train, *test)})
    shape_ids = sorted({outcome.request.shape.id for outcome in (*train, *test)})
    candidate_index = {candidate_hash: index for index, candidate_hash in enumerate(candidate_ids)}
    shape_index = {shape_id: index for index, shape_id in enumerate(shape_ids)}
    references = {}
    for outcome in train:
        if outcome.performance is not None and outcome.performance > 0.0:
            shape_id = outcome.request.shape.id
            references[shape_id] = max(references.get(shape_id, 0.0), outcome.performance)
    matrix = np.full((len(candidate_ids), len(shape_ids)), np.nan)
    for outcome in train:
        if outcome.performance is None or outcome.performance <= 0.0:
            continue
        shape_id = outcome.request.shape.id
        matrix[candidate_index[outcome.request.candidate.hash], shape_index[shape_id]] = math.log(
            outcome.performance / references[shape_id]
        )
    observed = ~np.isnan(matrix)
    if not observed.any():
        return None
    column_means = np.nanmean(matrix, axis=0)
    global_mean = float(np.nanmean(matrix))
    column_means = np.where(np.isnan(column_means), global_mean, column_means)
    filled = np.where(observed, matrix, column_means[None, :])
    reconstructed = filled
    for _ in range(iterations):
        row_means = filled.mean(axis=1, keepdims=True)
        centered = filled - row_means
        left, singular, right = np.linalg.svd(centered, full_matrices=False)
        active_rank = min(rank, len(singular))
        reconstructed = (left[:, :active_rank] * singular[:active_rank]) @ right[:active_rank, :] + row_means
        filled = np.where(observed, matrix, reconstructed)
    residual = matrix[observed] - reconstructed[observed]
    uncertainty = max(float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0, 0.05)
    predictions = []
    filtered_test = []
    for outcome in test:
        candidate_hash = outcome.request.candidate.hash
        shape_id = outcome.request.shape.id
        if candidate_hash not in candidate_index or shape_id not in shape_index or shape_id not in references:
            continue
        mean = float(filled[candidate_index[candidate_hash], shape_index[shape_id]])
        predictions.append(
            PairPrediction(
                shape_id=shape_id,
                candidate_hash=candidate_hash,
                mean_normalized_log_performance=mean,
                epistemic_std_log_performance=uncertainty,
                validity_probability=1.0,
                posterior_samples=(mean - uncertainty, mean, mean + uncertainty),
                reference_performance=references[shape_id],
            )
        )
        filtered_test.append(outcome)
    return evaluate_pair_predictions(predictions, filtered_test).to_dict()


def _split(outcomes, mode, folds):
    if mode == "candidate":
        return (
            [outcome for outcome in outcomes if _fold(outcome.request.candidate.hash, folds) != 0],
            [outcome for outcome in outcomes if _fold(outcome.request.candidate.hash, folds) == 0],
        )
    if mode == "shape":
        shape_fold = {shape.id: index % folds for index, shape in enumerate(sorted(shapes, key=lambda item: item.id))}
        return (
            [outcome for outcome in outcomes if shape_fold[outcome.request.shape.id] != 0],
            [outcome for outcome in outcomes if shape_fold[outcome.request.shape.id] == 0],
        )
    return (
        [
            outcome
            for outcome in outcomes
            if _fold(f"{outcome.request.shape.id}/{outcome.request.candidate.hash}", folds) != 0
        ],
        [
            outcome
            for outcome in outcomes
            if _fold(f"{outcome.request.shape.id}/{outcome.request.candidate.hash}", folds) == 0
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("out/grid100_full_20260618_repaired.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("out/grid100_pair_model_20260712.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--estimators", type=int, default=96)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    oracle = load_db_oracle_matrix(args.db, shapes=shapes)
    outcomes = _outcomes(oracle)
    results = {}
    for offset, mode in enumerate(("candidate", "shape", "pair")):
        train, test = _split(outcomes, mode, args.folds)
        summary, shared = _shared_metrics(train, test, seed=args.seed + offset, estimators=args.estimators)
        result = {
            "train_pairs": len(train),
            "test_pairs": len(test),
            "shared_contextual": shared,
            "fit": summary,
        }
        if mode in {"candidate", "pair"}:
            result["shape_local_extratrees"] = _shape_local_predictions(
                train,
                test,
                seed=args.seed + 3 + offset,
                estimators=args.estimators,
            )
        if mode in {"shape", "pair"}:
            result["nearest_shape"] = _nearest_predictions(train, test)
        if mode == "pair":
            result["low_rank"] = _low_rank_predictions(train, test)
        results[mode] = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "source_db": str(args.db),
                "shapes": len(shapes),
                "oracle_pairs": len(outcomes),
                "folds": args.folds,
                "estimators": args.estimators,
                "seed": args.seed,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)


shapes = pilot_100_shapes()
shape_by_id = {shape.id: shape for shape in shapes}
clustering = cluster_shapes(
    shapes,
    ShapeClusteringConfiguration(
        workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
        cluster_count=16,
    ),
)
shape_distances = shape_descriptor_distances(clustering)

if __name__ == "__main__":
    main()
