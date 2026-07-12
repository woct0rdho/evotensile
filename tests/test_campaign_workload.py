import json
import math
from pathlib import Path

import pytest

from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.workload import ResolvedWorkloadWeights, ShapeWorkload, load_workload_weights
from evotensile.database import BenchmarkSummary
from evotensile.search.grid_evidence import candidate_grid_scores
from evotensile.shapes import pilot_100_shapes


def _provenance():
    return {
        "call_count_source": "test-workload",
        "baseline_label": "test-baseline",
        "baseline_source": "test.sqlite",
        "benchmark_protocol_hash": "bproto_test",
        "environment_compatibility_tag": "test-environment",
    }


def test_workload_weights_use_call_count_times_baseline_latency_and_normalize():
    shapes = pilot_100_shapes()[:2]

    workload = ResolvedWorkloadWeights.workload(
        [shape.id for shape in shapes],
        (
            ShapeWorkload(shapes[0].id, call_count=2.0, baseline_latency_us=10.0),
            ShapeWorkload(shapes[1].id, call_count=1.0, baseline_latency_us=80.0),
        ),
        provenance=_provenance(),
    )

    assert workload.weights[shapes[0].id] == pytest.approx(0.4)
    assert workload.weights[shapes[1].id] == pytest.approx(1.6)
    assert sum(workload.weights.values()) == pytest.approx(2.0)
    assert workload.total_call_count == 3.0
    assert workload.total_baseline_time_us == 100.0


def test_workload_file_requires_exact_shape_coverage(tmp_path: Path):
    shapes = pilot_100_shapes()[:2]
    path = tmp_path / "workload.json"
    path.write_text(
        json.dumps(
            {
                "provenance": _provenance(),
                "shapes": [
                    {
                        "shape_id": shape.id,
                        "call_count": index + 1,
                        "baseline_latency_us": 10.0,
                    }
                    for index, shape in enumerate(shapes)
                ],
            }
        ),
        encoding="utf-8",
    )

    workload = load_workload_weights(path, shape_ids=[shape.id for shape in shapes])

    assert workload.mode == "workload"
    assert workload.weights[shapes[1].id] == pytest.approx(4.0 / 3.0)


def test_singleton_workload_weight_is_exactly_one():
    shape = pilot_100_shapes()[0]

    workload = ResolvedWorkloadWeights.workload(
        (shape.id,),
        (ShapeWorkload(shape.id, call_count=123.0, baseline_latency_us=456.0),),
        provenance=_provenance(),
    )

    assert workload.weights == {shape.id: 1.0}


def test_controller_checkpoints_workload_and_reports_weighted_and_uniform_regret():
    shapes = pilot_100_shapes()[:2]
    controller = CampaignControllerState(
        shape_ids=tuple(shape.id for shape in shapes),
        time_budget_s=10.0,
        session_started_at=0.0,
    )
    workload = ResolvedWorkloadWeights.workload(
        controller.shape_ids,
        (
            ShapeWorkload(shapes[0].id, call_count=1.0, baseline_latency_us=10.0),
            ShapeWorkload(shapes[1].id, call_count=1.0, baseline_latency_us=90.0),
        ),
        provenance=_provenance(),
    )
    controller.set_workload(workload)
    controller.record_query(shapes[0].id, "slow", known=True)
    controller.disclose(shapes[0].id, "slow", performance=50.0)
    controller.record_query(shapes[1].id, "fast", known=True)
    controller.disclose(shapes[1].id, "fast", performance=100.0)

    metrics = controller.grid_metrics({shape.id: 100.0 for shape in shapes})
    checkpoint = controller.to_checkpoint(now=0.0)
    restored = CampaignControllerState.from_checkpoint(checkpoint, session_started_at=20.0)

    assert metrics.mean_log_regret == pytest.approx(math.log(2.0) / 2.0)
    assert metrics.weighted_mean_log_regret == pytest.approx(math.log(2.0) * 0.1)
    assert metrics.worst_log_regret == pytest.approx(math.log(2.0))
    assert restored.resolved_workload == workload
    assert restored.to_checkpoint(now=20.0) == checkpoint


def test_weighted_grid_scores_prioritize_coverage_of_high_contribution_shapes():
    shapes = pilot_100_shapes()[:2]
    first, second = shapes
    summaries = {
        first.id: [
            BenchmarkSummary(first.id, "candidate-a", 2, 100.0, 100.0, 10.0, 10.0),
            BenchmarkSummary(first.id, "candidate-b", 2, 90.0, 90.0, 11.0, 11.0),
        ],
        second.id: [
            BenchmarkSummary(second.id, "candidate-b", 2, 90.0, 90.0, 11.0, 11.0),
        ],
    }

    uniform = candidate_grid_scores(
        summaries,
        target_shape_ids=[first.id, second.id],
    )
    weighted = candidate_grid_scores(
        summaries,
        target_shape_ids=[first.id, second.id],
        shape_weights={first.id: 1.8, second.id: 0.2},
    )

    assert uniform["candidate-a"].coverage_fraction == 0.5
    assert weighted["candidate-a"].coverage_fraction == pytest.approx(0.9)
    assert weighted["candidate-a"].generalist_score < uniform["candidate-a"].generalist_score
