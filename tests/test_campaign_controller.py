import math

import pytest

from evotensile.campaign.controller import (
    CampaignControllerState,
    SoftAdmissionBudget,
    estimate_admission_duration_s,
)
from evotensile.shapes import pilot_100_shapes


def test_soft_admission_budget_checks_only_before_work_and_records_overrun():
    budget = SoftAdmissionBudget(time_budget_s=10.0, session_started_at=100.0)

    admitted = budget.decide(predicted_duration_s=7.0, reserve_s=2.0, now=101.0)
    rejected = budget.decide(predicted_duration_s=7.1, reserve_s=2.0, now=101.0)

    assert admitted.admitted is True
    assert admitted.reason == "admitted"
    assert rejected.admitted is False
    assert rejected.reason == "insufficient_predicted_budget"
    assert budget.elapsed_s(now=112.0) == 12.0
    assert budget.overrun_s(now=112.0) == 2.0
    assert budget.decide(now=112.0).reason == "soft_deadline"


def test_controller_state_tracks_query_causality_metrics_costs_and_checkpoint():
    shapes = pilot_100_shapes()[:2]
    state = CampaignControllerState(
        shape_ids=tuple(shape.id for shape in shapes),
        time_budget_s=10.0,
        session_started_at=100.0,
    )
    first_hash = "cand_first"
    second_hash = "cand_second"

    assert state.record_query(shapes[0].id, first_hash, known=True) is True
    assert state.record_query(shapes[0].id, first_hash, known=True) is False
    assert state.disclose(shapes[0].id, first_hash, performance=10.0) is True
    assert state.disclose(shapes[0].id, first_hash, performance=10.0) is False
    assert state.record_query(shapes[1].id, second_hash, known=False) is True
    with pytest.raises(ValueError, match="queried and known"):
        state.disclose(shapes[1].id, second_hash, performance=20.0)

    assert state.record_prepared(first_hash, [shape.id for shape in shapes]) == {shape.id for shape in shapes}
    assert state.record_prepared(first_hash, [shapes[0].id]) == set()
    state.record_phase_time("preparation", 2.5)
    state.record_phase_time("screening", 0.5)
    state.set_reserve("confirmation", 1.0)
    decision = state.decide_admission(predicted_duration_s=5.0, reserve_s=1.0, now=101.0)
    assert decision.admitted is True

    metrics = state.grid_metrics(
        {shapes[0].id: 20.0, shapes[1].id: 30.0},
        weights={shapes[0].id: 1.0, shapes[1].id: 3.0},
    )
    assert metrics.resolved_shapes == 1
    assert metrics.unresolved_shapes == 1
    assert metrics.mean_log_regret == pytest.approx(math.log(2.0))
    assert metrics.weighted_mean_log_regret == pytest.approx(math.log(2.0))
    assert metrics.per_shape_log_regret[shapes[1].id] is None

    summary = state.summary(
        oracle_best_by_shape={shapes[0].id: 20.0, shapes[1].id: 30.0},
        now=105.0,
    )
    assert summary["queried_pairs"] == 2
    assert summary["known_pairs"] == 1
    assert summary["unknown_pairs"] == 1
    assert summary["candidate_coverage"] == {first_hash: 1, second_hash: 1}
    assert summary["prepared_artifact_coverage"] == {first_hash: 2}
    assert summary["phase_time_s"]["preparation"] == 2.5

    checkpoint = state.to_checkpoint(now=105.0)
    restored = CampaignControllerState.from_checkpoint(checkpoint, session_started_at=200.0)

    assert restored.to_checkpoint(now=200.0) == checkpoint
    assert restored.admission_deadline == 205.0
    assert restored.overrun_s(now=206.0) == 1.0


def test_disclosed_pair_can_receive_later_positive_performance():
    shape = pilot_100_shapes()[0]
    state = CampaignControllerState(
        shape_ids=(shape.id,),
        time_budget_s=5.0,
        session_started_at=10.0,
    )

    state.record_query(shape.id, "cand_retry", known=True)
    assert state.disclose(shape.id, "cand_retry") is True
    assert state.disclose(shape.id, "cand_retry", performance=12.0) is False
    assert state.incumbents[shape.id].performance == 12.0


def test_singleton_controller_has_one_incumbent_and_no_cross_shape_state():
    shape = pilot_100_shapes()[0]
    state = CampaignControllerState(
        shape_ids=(shape.id,),
        time_budget_s=5.0,
        session_started_at=10.0,
    )

    state.record_query(shape.id, "cand_single", known=True)
    state.disclose(shape.id, "cand_single", performance=12.0)

    summary = state.summary(oracle_best_by_shape={shape.id: 12.0}, now=16.0)
    assert summary["shape_ids"] == [shape.id]
    assert summary["resolved_shapes"] == 1
    assert summary["unresolved_shape_ids"] == []
    assert summary["budget_overrun_s"] == 1.0
    assert summary["grid_metrics"]["mean_log_regret"] == 0.0


def test_admission_duration_estimate_uses_recent_robust_per_unit_cost():
    observations = [(24.0, 24), (30.0, 24), (27.0, 24)]

    estimate = estimate_admission_duration_s(observations, expected_units=24)

    assert estimate == pytest.approx(39.5)
    assert estimate_admission_duration_s([], expected_units=24) == 30.0
