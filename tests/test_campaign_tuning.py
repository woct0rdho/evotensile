from evotensile.campaign.policy import (
    InitializationRegime,
    selected_campaign_policy,
    selected_campaign_round_schedule,
)
from evotensile.campaign.tuning import (
    PolicyTrialObservation,
    aggregate_policy_trials,
    fold_regret_metrics,
    mechanically_stratified_folds,
    select_robust_default,
)
from evotensile.search.shape_clustering import ShapeClusteringConfiguration, cluster_shapes
from evotensile.shapes import pilot_100_shapes
from scripts.tune_campaign_policy import _select_multi_round_schedule, _stable_candidate_order
from tests.helpers import sample_candidates


def _trial(configuration_id, seed, mean, p95, worst, unresolved, prepared):
    return PolicyTrialObservation(
        configuration_id=configuration_id,
        seed=seed,
        ordering_id=f"ordering-{seed}",
        fold_id="fold-0",
        mean_log_regret=mean,
        p95_log_regret=p95,
        worst_log_regret=worst,
        unresolved_shapes=unresolved,
        queried_pairs=100,
        unknown_pairs=0,
        prepared_candidates=prepared,
    )


def test_selected_profile_defaults_match_tuning_artifact_identities():
    expected: dict[InitializationRegime, str] = {
        "blind": "campaign_policy_46baa1a9",
        "anchored-untuned": "campaign_policy_89ea03a4",
        "anchored-tuned": "campaign_policy_e7961c9f",
    }

    assert {regime: selected_campaign_policy(regime).identity_hash for regime in expected} == expected
    assert selected_campaign_round_schedule("blind").to_dict() == {
        "name": "fixed",
        "pair_budget_fractions": [0.5, 0.5],
        "repair_enabled": [True, True],
    }
    assert selected_campaign_round_schedule("anchored-untuned").name == "role-specialized"
    assert selected_campaign_round_schedule("anchored-tuned").repair_enabled == (False, True)


def test_mechanical_folds_cover_shapes_once_and_mix_clusters():
    shapes = pilot_100_shapes()[:12]
    clustering = cluster_shapes(
        shapes,
        ShapeClusteringConfiguration(workgroup_processor_count=20, cluster_count=3),
    )

    folds = mechanically_stratified_folds(clustering, fold_count=4)

    flattened = [shape_id for shape_ids in folds.values() for shape_id in shape_ids]
    assert sorted(flattened) == sorted(shape.id for shape in shapes)
    assert len(flattened) == len(set(flattened))
    assert all(shape_ids for shape_ids in folds.values())


def test_seeded_candidate_order_is_stable_when_catalog_grows():
    candidates = sample_candidates(9, seed=20260712)

    original = _stable_candidate_order(candidates[:8], seed=17)
    expanded = _stable_candidate_order(candidates, seed=17)

    original_hashes = [candidate.hash for candidate in original]
    assert [candidate.hash for candidate in expanded if candidate.hash in original_hashes] == original_hashes


def test_multi_round_selection_balances_mean_tail_and_unknown_cost():
    rows = [
        {
            "schedule": "fixed",
            "summary": {
                "mean_log_regret": 0.83,
                "p95_log_regret": 1.91,
                "worst_log_regret": 2.18,
                "unresolved_shapes": 0,
                "prepared_candidates": 91,
                "unknown_pairs": 561,
            },
        },
        {
            "schedule": "role_specialized",
            "summary": {
                "mean_log_regret": 0.81,
                "p95_log_regret": 2.00,
                "worst_log_regret": 2.20,
                "unresolved_shapes": 0,
                "prepared_candidates": 91,
                "unknown_pairs": 619,
            },
        },
    ]

    selected, selection = _select_multi_round_schedule(rows)

    assert selected["schedule"] == "fixed"
    assert selection["pareto_schedules"] == ["fixed", "role_specialized"]
    assert selection["robust_scores"]["fixed"] < selection["robust_scores"]["role_specialized"]


def test_fold_metrics_penalize_unresolved_without_hiding_resolved_regret():
    shape_ids = ("a", "b", "c")
    mean, p95, worst, unresolved = fold_regret_metrics(
        shape_ids=shape_ids,
        oracle_best={"a": 100.0, "b": 100.0, "c": 100.0},
        incumbent_performance={"a": 100.0, "b": 50.0},
    )

    assert mean > 0.0
    assert p95 > mean
    assert worst > p95
    assert unresolved == 1


def test_aggregate_marks_pareto_front_and_selects_stable_compromise():
    observations = [
        _trial("balanced", 1, 0.10, 0.20, 0.40, 2, 20),
        _trial("balanced", 2, 0.11, 0.21, 0.41, 2, 20),
        _trial("tail", 1, 0.12, 0.16, 0.25, 3, 22),
        _trial("tail", 2, 0.13, 0.17, 0.26, 3, 22),
        _trial("dominated", 1, 0.20, 0.30, 0.50, 4, 30),
        _trial("dominated", 2, 0.21, 0.31, 0.51, 4, 30),
    ]

    aggregates = aggregate_policy_trials(observations)
    selected = select_robust_default(aggregates)

    by_id = {aggregate.configuration_id: aggregate for aggregate in aggregates}
    assert by_id["balanced"].pareto_optimal
    assert by_id["tail"].pareto_optimal
    assert not by_id["dominated"].pareto_optimal
    assert selected.configuration_id in {"balanced", "tail"}
    assert selected.robust_score is not None
