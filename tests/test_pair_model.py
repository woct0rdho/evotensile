import math

from evotensile.campaign.evaluator import PairEvaluationOutcome
from evotensile.profile import DEFAULT_PROFILE
from evotensile.scheduling.models import PairRequest
from evotensile.search.pair_model import (
    ContextualPairModel,
    PairModelConfiguration,
    evaluate_pair_predictions,
)
from evotensile.shapes import pilot_100_shapes
from tests.helpers import sample_candidates


def _performance(candidate, shape) -> float:
    params = candidate.canonical_params()
    shape_factor = 1.0 + math.log2(shape.k) / 20.0
    score = (
        100.0
        + 1.5 * params["DepthU"]
        + 12.0 * params["ScheduleIterAlg"]
        + 4.0 * params["GlobalSplitU"]
        + 3.0 * params["VectorWidthA"]
    )
    return score * shape_factor


def _outcome(candidate, shape, *, performance, status="ok", disclosed=True):
    return PairEvaluationOutcome(
        request=PairRequest(candidate, shape),
        provenance="replay",
        source_ref="pair-model-test",
        status=status,
        known=True,
        disclosed=disclosed,
        samples=3 if performance is not None else 0,
        performance=performance,
    )


def test_contextual_pair_model_uses_only_disclosed_pairs_and_supports_sparse_shapes():
    shapes = pilot_100_shapes()[:6]
    candidates = sample_candidates(30, seed=12345)
    train_candidates = candidates[:20]
    invalid_hashes = {candidate.hash for candidate in train_candidates[:3]}
    outcomes = []
    for shape in shapes[:5]:
        for candidate in train_candidates:
            outcomes.append(
                _outcome(
                    candidate,
                    shape,
                    performance=(None if candidate.hash in invalid_hashes else _performance(candidate, shape)),
                    status=("validation_failed" if candidate.hash in invalid_hashes else "ok"),
                )
            )
    outcomes.append(
        _outcome(
            candidates[20],
            shapes[0],
            performance=1_000_000.0,
            disclosed=False,
        )
    )
    model = ContextualPairModel(
        workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
        configuration=PairModelConfiguration(n_estimators=48, min_performance_rows=24, seed=12346, jobs=1),
    )

    summary = model.fit(outcomes)
    predictions = model.predict(
        [
            (train_candidates[0], shapes[5]),
            (train_candidates[4], shapes[5]),
            (candidates[21], shapes[0]),
        ]
    )

    assert summary.validity_rows == len(shapes[:5]) * len(train_candidates)
    assert summary.performance_rows == len(shapes[:5]) * (len(train_candidates) - len(invalid_hashes))
    assert summary.feature_count > 40
    assert len(summary.feature_contract_hash) == 64
    assert predictions[0].reference_performance is None
    assert predictions[1].reference_performance is None
    assert predictions[2].reference_performance is not None
    assert len(predictions[0].posterior_samples) == 48
    assert predictions[0].validity_probability < predictions[1].validity_probability
    assert predictions[2].epistemic_std_log_performance >= 0.0


def test_contextual_pair_model_uses_the_same_path_for_one_shape():
    shape = pilot_100_shapes()[0]
    candidates = sample_candidates(28, seed=12345)
    outcomes = [_outcome(candidate, shape, performance=_performance(candidate, shape)) for candidate in candidates]
    model = ContextualPairModel(
        workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
        configuration=PairModelConfiguration(n_estimators=32, min_performance_rows=24, seed=12346, jobs=1),
    )

    model.fit(outcomes)
    predictions = model.predict([(candidate, shape) for candidate in candidates[:3]])

    assert len(predictions) == 3
    assert all(prediction.shape_id == shape.id for prediction in predictions)
    assert all(prediction.reference_performance is not None for prediction in predictions)


def test_pair_model_reports_rank_calibration_topk_and_improvement_metrics():
    shapes = pilot_100_shapes()[:5]
    candidates = sample_candidates(36, seed=12345)
    train_candidates = candidates[:24]
    test_candidates = candidates[24:]
    train = [
        _outcome(candidate, shape, performance=_performance(candidate, shape))
        for shape in shapes
        for candidate in train_candidates
    ]
    model = ContextualPairModel(
        workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
        configuration=PairModelConfiguration(n_estimators=64, min_performance_rows=24, seed=12346, jobs=1),
    )
    model.fit(train)
    test = [
        _outcome(candidate, shape, performance=_performance(candidate, shape))
        for shape in shapes
        for candidate in test_candidates
    ]
    predictions = model.predict([(outcome.request.candidate, outcome.request.shape) for outcome in test])

    metrics = evaluate_pair_predictions(predictions, test, top_k=3)

    assert metrics.pairs == len(test)
    assert metrics.shapes == len(shapes)
    assert metrics.candidates == len(test_candidates)
    assert metrics.mean_shape_rank_correlation is not None
    assert metrics.mean_shape_rank_correlation > 0.5
    assert metrics.mean_shape_top_k_recall is not None
    assert metrics.mean_shape_top_k_recall >= 0.4
    assert metrics.interval_90_coverage is not None
    assert 0.0 <= metrics.interval_90_coverage <= 1.0
    assert metrics.validity_brier is not None
    assert metrics.probability_improvement_brier is not None
    assert metrics.probability_improvement_calibration_error is not None
    assert all(0.0 <= prediction.probability_of_improvement() <= 1.0 for prediction in predictions)
