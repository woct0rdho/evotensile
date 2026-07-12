#!/usr/bin/env python3

import argparse
import hashlib
import json
import statistics
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from evotensile.campaign.acquisition import BundleAcquisitionPolicy, BundleCostModel, plan_candidate_bundles
from evotensile.campaign.baselines import evaluate_representative_first_baseline
from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.evaluator import PairEvaluationOutcome, ReplayEvaluator
from evotensile.campaign.policy import CampaignPolicyConfiguration
from evotensile.campaign.repair import (
    RepairPolicy,
    assess_repair_deficits,
    build_repair_candidate_pool,
    plan_repair_acquisition,
)
from evotensile.campaign.round_controller import StagedRoundConfiguration
from evotensile.campaign.tuning import (
    PolicyAggregate,
    PolicyTrialObservation,
    aggregate_policy_trials,
    fold_regret_metrics,
    mechanically_stratified_folds,
    select_robust_default,
)
from evotensile.candidate import Candidate, Shape
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.scheduling.models import EvidenceStage, PairRequest
from evotensile.search.pair_model import ContextualPairModel, PairModelConfiguration
from evotensile.search.replay import ExactOracleReplayState, OracleRecord, load_db_oracle_matrix
from evotensile.search.shape_clustering import ShapeClusteringConfiguration, cluster_shapes
from evotensile.shapes import pilot_100_shapes


@dataclass(frozen=True)
class ProfileInput:
    oracle: dict[tuple[str, str], OracleRecord]
    baseline_pairs: list[tuple[Shape, Candidate]] | None
    references: dict[str, float | None] | None
    metadata: dict[str, object]


@dataclass(frozen=True)
class SelectedProfile:
    aggregate: PolicyAggregate
    configuration: CampaignPolicyConfiguration


def _stable_candidate_order(candidates, *, seed):
    return sorted(
        candidates,
        key=lambda candidate: (
            hashlib.sha256(f"{seed}:{candidate.hash}".encode()).digest(),
            candidate.hash,
        ),
    )


def _select_multi_round_schedule(rows):
    objective_names = (
        "mean_log_regret",
        "p95_log_regret",
        "worst_log_regret",
        "unresolved_shapes",
        "prepared_candidates",
        "unknown_pairs",
    )
    objectives = {row["schedule"]: tuple(float(row["summary"][name]) for name in objective_names) for row in rows}
    tolerance = 1e-12
    pareto_schedules = {
        schedule
        for schedule, values in objectives.items()
        if not any(
            all(left <= right + tolerance for left, right in zip(other, values, strict=True))
            and any(left < right - tolerance for left, right in zip(other, values, strict=True))
            for other_schedule, other in objectives.items()
            if other_schedule != schedule
        )
    }
    minima = tuple(min(values[index] for values in objectives.values()) for index in range(len(objective_names)))
    maxima = tuple(max(values[index] for values in objectives.values()) for index in range(len(objective_names)))
    scores = {}
    for schedule, values in objectives.items():
        normalized = tuple(
            0.0 if maximum == minimum else (value - minimum) / (maximum - minimum)
            for value, minimum, maximum in zip(values, minima, maxima, strict=True)
        )
        scores[schedule] = max(normalized) + 0.1 * statistics.fmean(normalized)
    selected = min(
        (row for row in rows if row["schedule"] in pareto_schedules),
        key=lambda row: (
            scores[row["schedule"]],
            row["summary"]["worst_log_regret"],
            row["summary"]["p95_log_regret"],
            row["summary"]["unknown_pairs"],
            row["schedule"],
        ),
    )
    return selected, {
        "objective_names": list(objective_names),
        "pareto_schedules": sorted(pareto_schedules),
        "robust_scores": dict(sorted(scores.items())),
    }


def _phase_configuration(repair_pairs, total_pairs, *, guard_s):
    repair_fraction = repair_pairs / total_pairs
    return StagedRoundConfiguration(
        phase_fractions=(
            ("broad", 0.35),
            ("promotion", 0.45 - repair_fraction),
            ("repair", repair_fraction),
            ("stabilization", 0.10),
            ("confirmation", 0.10),
        ),
        no_new_preparation_guard_s=guard_s,
    )


def _tuning_acquisition_policy(
    *,
    coverage_weight: float,
    information_weight: float,
    total_pairs: int,
) -> BundleAcquisitionPolicy:
    return BundleAcquisitionPolicy(
        improvement_weight=1.0,
        coverage_weight=coverage_weight,
        information_weight=information_weight,
        bundle_sizes=(1, 2, 4, 8, 16),
        max_pairs=total_pairs,
        max_bundles=96,
        max_predicted_cost_s=300.0,
        evidence_stage=EvidenceStage.PROBE,
    )


def _configurations(total_pairs, *, initialization_profile, initialization_label=None):
    return (
        CampaignPolicyConfiguration(
            name="balanced-16-requested",
            initialization_profile=initialization_profile,
            initialization_label=initialization_label,
            cluster_count=16,
            calibration_candidate_count=8 if initialization_profile == "anchored" else 0,
            artifact_scope="requested",
            round=_phase_configuration(12, total_pairs, guard_s=30.0),
            acquisition=_tuning_acquisition_policy(
                coverage_weight=0.50,
                information_weight=0.10,
                total_pairs=total_pairs,
            ),
            repair=RepairPolicy(uncertainty_weight=0.0, mutation_candidates_per_shape=4),
        ),
        CampaignPolicyConfiguration(
            name="coverage-12-cluster",
            initialization_profile=initialization_profile,
            initialization_label=initialization_label,
            cluster_count=12,
            calibration_candidate_count=12 if initialization_profile == "anchored" else 0,
            artifact_scope="cluster",
            round=_phase_configuration(8, total_pairs, guard_s=30.0),
            acquisition=_tuning_acquisition_policy(
                coverage_weight=0.80,
                information_weight=0.10,
                total_pairs=total_pairs,
            ),
            repair=RepairPolicy(uncertainty_weight=0.0, mutation_candidates_per_shape=4),
        ),
        CampaignPolicyConfiguration(
            name="information-20-requested",
            initialization_profile=initialization_profile,
            initialization_label=initialization_label,
            cluster_count=20,
            calibration_candidate_count=8 if initialization_profile == "anchored" else 0,
            artifact_scope="requested",
            round=_phase_configuration(8, total_pairs, guard_s=20.0),
            acquisition=_tuning_acquisition_policy(
                coverage_weight=0.50,
                information_weight=0.25,
                total_pairs=total_pairs,
            ),
            repair=RepairPolicy(uncertainty_weight=0.10, mutation_candidates_per_shape=4),
        ),
        CampaignPolicyConfiguration(
            name="tail-16-cluster",
            initialization_profile=initialization_profile,
            initialization_label=initialization_label,
            cluster_count=16,
            calibration_candidate_count=4 if initialization_profile == "anchored" else 0,
            artifact_scope="cluster",
            round=_phase_configuration(16, total_pairs, guard_s=20.0),
            acquisition=_tuning_acquisition_policy(
                coverage_weight=0.35,
                information_weight=0.05,
                total_pairs=total_pairs,
            ),
            repair=RepairPolicy(
                uncertainty_weight=0.0,
                maximum_deficit_fraction=0.20,
                minimum_close_probability=0.20,
                mutation_candidates_per_shape=4,
            ),
        ),
    )


def _state(path, *, oracle, shapes, source_ref):
    db = EvoTensileDB.connect(
        path,
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    evaluator = ReplayEvaluator(
        ExactOracleReplayState(
            db=db,
            shapes=shapes,
            oracle=oracle,
            profile=DEFAULT_PROFILE,
            source_ref=source_ref,
        ),
        prepare_workers=4,
        prepare_seconds_per_candidate=0.1,
    )
    controller = CampaignControllerState(
        shape_ids=tuple(shape.id for shape in shapes),
        time_budget_s=300.0,
        session_started_at=0.0,
    )
    return evaluator, controller


def _seed_state(path, *, oracle, shapes, seed_candidates, seed_clustering, source_ref):
    evaluator, controller = _state(path, oracle=oracle, shapes=shapes, source_ref=source_ref)
    controller.set_clustering(seed_clustering.to_dict())
    result = evaluate_representative_first_baseline(
        evaluator,
        controller,
        candidates=seed_candidates,
        shapes=shapes,
        clustering=seed_clustering,
    ).result
    return evaluator, controller, list(result.outcomes)


def _anchored_state(
    path,
    *,
    oracle,
    shapes,
    baseline_pairs,
    baseline_label,
):
    evaluator, controller = _state(
        path,
        oracle=oracle,
        shapes=shapes,
        source_ref=baseline_label,
    )
    outcomes = []
    references = {}
    for shape, candidate in baseline_pairs:
        record = oracle.get((shape.id, candidate.hash))
        performance = None if record is None else record.screening_gflops
        if performance is None or performance <= 0.0:
            raise ValueError(f"anchored baseline pair lacks exact timing evidence: {shape.id}/{candidate.hash}")
        controller.record_query(shape.id, candidate.hash, known=True)
        controller.disclose(shape.id, candidate.hash, performance=performance)
        references[shape.id] = performance
        outcomes.append(
            PairEvaluationOutcome(
                request=PairRequest(candidate, shape, evidence_stage=EvidenceStage.SCREENING),
                provenance="anchored-baseline",
                source_ref=baseline_label,
                status="ok",
                known=True,
                disclosed=True,
                samples=10,
                performance=performance,
            )
        )
    if set(references) != {shape.id for shape in shapes}:
        raise ValueError("anchored baseline must cover every campaign shape exactly")
    return evaluator, controller, outcomes, references


def _model(outcomes, *, estimators, seed):
    model = ContextualPairModel(
        workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
        configuration=PairModelConfiguration(
            n_estimators=estimators,
            min_performance_rows=24,
            seed=seed,
            jobs=DEFAULT_PROFILE.default_surrogate_jobs,
        ),
    )
    model.fit(outcomes)
    return model


def _cost_model(seed):
    return BundleCostModel(
        workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
        fallback_preparation_s=0.1,
        fallback_validation_s=0.0,
        fallback_timing_s=0.001,
        seed=seed,
        jobs=DEFAULT_PROFILE.default_surrogate_jobs,
    )


def _artifact_scopes(shapes, clustering, mode):
    shape_by_id = {shape.id: shape for shape in shapes}
    if mode == "requested":
        return {shape.id: (shape,) for shape in shapes}
    return {
        shape_id: tuple(shape_by_id[member_id] for member_id in cluster.shape_ids)
        for cluster in clustering.clusters
        for shape_id in cluster.shape_ids
    }


def _needs_anchored_calibration(observations):
    positive_by_shape = {}
    for outcome in observations:
        if outcome.performance is not None and outcome.performance > 0.0:
            shape_id = outcome.request.shape.id
            positive_by_shape[shape_id] = positive_by_shape.get(shape_id, 0) + 1
    return bool(positive_by_shape) and max(positive_by_shape.values()) <= 1


def _repair_pair_budget(configuration, pair_budget):
    fraction = configuration.round.fraction_by_phase["repair"]
    return max(0, min(pair_budget, round(pair_budget * fraction)))


def _run_increment(
    evaluator,
    controller,
    observations,
    *,
    candidates,
    shapes,
    configuration,
    pair_budget,
    estimators,
    seed,
    allow_repair,
    reference_performance=None,
):
    clustering = cluster_shapes(
        shapes,
        ShapeClusteringConfiguration(
            workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
            cluster_count=configuration.cluster_count,
        ),
    )
    controller.set_clustering(clustering.to_dict())
    repair_pairs = _repair_pair_budget(configuration, pair_budget) if allow_repair else 0
    calibration_requests = []
    if configuration.calibration_candidate_count > 0 and _needs_anchored_calibration(observations):
        shape_by_id = {shape.id: shape for shape in shapes}
        representatives = [shape_by_id[shape_id] for shape_id in clustering.medoid_shape_ids]
        for candidate in candidates[: configuration.calibration_candidate_count]:
            for shape in representatives:
                if (shape.id, candidate.hash) in controller.queried_pairs:
                    continue
                calibration_requests.append(PairRequest(candidate, shape, evidence_stage=EvidenceStage.PROBE))
                if len(calibration_requests) >= pair_budget - repair_pairs:
                    break
            if len(calibration_requests) >= pair_budget - repair_pairs:
                break
    calibration_result = evaluator.evaluate(calibration_requests)
    calibration_result.apply(controller)
    observations.extend(calibration_result.outcomes)
    broad_pairs = pair_budget - repair_pairs - len(calibration_requests)
    model = _model(observations, estimators=estimators, seed=seed)
    predictions = model.predict([(candidate, shape) for candidate in candidates for shape in shapes])
    broad_policy = replace(configuration.acquisition, max_pairs=broad_pairs, repair_weight=0.0)
    broad_plan = plan_candidate_bundles(
        controller,
        candidates=candidates,
        shapes=shapes,
        predictions=predictions,
        cost_model=_cost_model(seed + 1),
        policy=broad_policy,
        artifact_shapes_by_target=_artifact_scopes(shapes, clustering, configuration.artifact_scope),
    )
    broad_result = evaluator.evaluate(
        broad_plan.timing_requests,
        artifact_shapes_by_candidate=broad_plan.artifact_shapes_by_candidate,
    )
    broad_result.apply(controller)
    observations.extend(broad_result.outcomes)
    report = {
        "calibration_pairs": len(calibration_requests),
        "broad_pairs": len(broad_plan.timing_requests),
        "repair_pairs": 0,
        "unknown_pairs": calibration_result.unknown_pairs + broad_result.unknown_pairs,
        "unknown_pair_keys": [
            list(outcome.key) for outcome in (*calibration_result.outcomes, *broad_result.outcomes) if not outcome.known
        ],
    }
    if repair_pairs <= 0:
        return report
    repair_model = _model(observations, estimators=estimators, seed=seed + 2)
    catalog_predictions = repair_model.predict([(candidate, shape) for candidate in candidates for shape in shapes])
    replay_repair_policy = replace(configuration.repair, mutation_candidates_per_shape=0, seed=seed + 3)
    deficits = assess_repair_deficits(
        controller,
        shapes=shapes,
        clustering=clustering,
        predictions=catalog_predictions,
        reference_performance=reference_performance,
        policy=replay_repair_policy,
    )
    pool = build_repair_candidate_pool(
        controller,
        shapes=shapes,
        clustering=clustering,
        deficits=deficits,
        observations=observations,
        candidate_catalog={candidate.hash: candidate for candidate in candidates},
        broad_candidates=[score.bundle.candidate for score in broad_plan.selected],
        policy=replay_repair_policy,
    )
    repair_predictions = repair_model.predict([(candidate, shape) for candidate in pool.candidates for shape in shapes])
    repair = plan_repair_acquisition(
        controller,
        candidates=pool.candidates,
        shapes=shapes,
        deficits=deficits,
        predictions=repair_predictions,
        cost_model=_cost_model(seed + 3),
        acquisition_policy=BundleAcquisitionPolicy(
            improvement_weight=0.0,
            coverage_weight=0.0,
            information_weight=0.0,
            repair_weight=1.0,
            bundle_sizes=(1, 2, 4),
            max_pairs=repair_pairs,
            max_bundles=repair_pairs,
            max_predicted_cost_s=300.0,
            evidence_stage=EvidenceStage.PROBE,
        ),
        repair_policy=replay_repair_policy,
    )
    repair_result = evaluator.evaluate(
        repair.plan.timing_requests,
        artifact_shapes_by_candidate=repair.plan.artifact_shapes_by_candidate,
    )
    repair_result.apply(controller)
    observations.extend(repair_result.outcomes)
    report.update(
        {
            "repair_pairs": len(repair.plan.timing_requests),
            "repair_deficits": len(deficits),
            "unknown_pairs": report["unknown_pairs"] + repair_result.unknown_pairs,
            "unknown_pair_keys": [
                *report["unknown_pair_keys"],
                *[list(outcome.key) for outcome in repair_result.outcomes if not outcome.known],
            ],
        }
    )
    return report


def _controller_summary(controller, oracle_best):
    metrics = controller.grid_metrics(oracle_best)
    return {
        "queried_pairs": len(controller.queried_pairs),
        "known_pairs": len(controller.known_pairs),
        "unknown_pairs": len(controller.unknown_pairs),
        "resolved_shapes": metrics.resolved_shapes,
        "unresolved_shapes": metrics.unresolved_shapes,
        "mean_log_regret": metrics.mean_log_regret,
        "p95_log_regret": metrics.p95_log_regret,
        "worst_log_regret": metrics.worst_log_regret,
        "prepared_candidates": len(controller.prepared_artifact_shapes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("out/grid100_full_20260618_repaired.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("out/grid100_policy_tuning_20260712.json"))
    parser.add_argument("--pair-budget", type=int, default=385)
    parser.add_argument("--seed-candidates", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--estimators", type=int, default=64)
    parser.add_argument(
        "--untuned-baseline-db",
        type=Path,
        default=Path("out/grid100_untuned_hipblaslt_baseline_20260712.sqlite"),
    )
    parser.add_argument(
        "--tuned-baseline-db",
        type=Path,
        default=Path("out/grid100_tuned_hipblaslt_baseline_20260712.sqlite"),
    )
    parser.add_argument("--overlay-db", type=Path, default=None)
    parser.add_argument(
        "--singleton-tuning",
        type=Path,
        default=Path("out/grid100_singleton_policy_tuning_20260712.json"),
    )
    args = parser.parse_args()
    if args.pair_budget <= 0 or args.seed_candidates <= 0 or args.seeds <= 0 or args.folds <= 0:
        raise ValueError("tuning budgets, seeds, and folds must be positive")
    shapes = pilot_100_shapes()
    retained_oracle = load_db_oracle_matrix(args.db, shapes=shapes)
    if args.overlay_db is not None and args.overlay_db.exists():
        retained_oracle.update(load_db_oracle_matrix(args.overlay_db, shapes=shapes))
    seed_clustering = cluster_shapes(
        shapes,
        ShapeClusteringConfiguration(
            workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
            cluster_count=16,
        ),
    )
    folds = mechanically_stratified_folds(seed_clustering, fold_count=args.folds)
    profile_inputs = {
        "blind": ProfileInput(
            oracle=retained_oracle,
            baseline_pairs=None,
            references=None,
            metadata={"source_db": str(args.db)},
        )
    }
    for label, path in (
        ("anchored-untuned", args.untuned_baseline_db),
        ("anchored-tuned", args.tuned_baseline_db),
    ):
        if not path.exists():
            continue
        baseline_db = EvoTensileDB.connect(
            path,
            environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
        )
        discoveries = baseline_db.baseline_discoveries(baseline_label=label)
        if not discoveries:
            raise ValueError(f"baseline DB lacks labeled discovery {label}: {path}")
        baseline_pairs = baseline_db.baseline_selection_pairs(discoveries[-1].discovery_id)
        baseline_oracle = load_db_oracle_matrix(path, shapes=shapes)
        combined_oracle = dict(retained_oracle)
        combined_oracle.update(baseline_oracle)
        references = {
            shape.id: baseline_oracle[(shape.id, candidate.hash)].screening_gflops
            for shape, candidate in baseline_pairs
        }
        profile_inputs[label] = ProfileInput(
            oracle=combined_oracle,
            baseline_pairs=baseline_pairs,
            references=references,
            metadata={
                "source_db": str(path),
                "discovery": discoveries[-1].context,
            },
        )
    configurations_by_profile = {
        profile_name: _configurations(
            args.pair_budget,
            initialization_profile="blind" if profile_name == "blind" else "anchored",
            initialization_label=None if profile_name == "blind" else profile_name,
        )
        for profile_name in profile_inputs
    }
    configurations = tuple(
        configuration
        for profile_configurations in configurations_by_profile.values()
        for configuration in profile_configurations
    )
    observations = []
    run_reports = []
    selected_unknown_pairs: dict[str, set[tuple[str, str]]] = {
        configuration.identity_hash: set() for configuration in configurations
    }
    oracle_best_by_profile = {}
    selected_by_profile: dict[str, SelectedProfile] = {}
    multi_round = []
    with tempfile.TemporaryDirectory(prefix="evotensile-policy-tuning-") as directory:
        for profile_name, profile_input in profile_inputs.items():
            oracle = profile_input.oracle
            candidates = sorted(
                {record.candidate.hash: record.candidate for record in oracle.values()}.values(),
                key=lambda candidate: candidate.hash,
            )
            oracle_best = {
                shape.id: max(
                    record.screening_gflops or 0.0 for (shape_id, _), record in oracle.items() if shape_id == shape.id
                )
                for shape in shapes
            }
            oracle_best_by_profile[profile_name] = oracle_best
            for seed_index in range(args.seeds):
                seed = 12345 + seed_index
                ordered_candidates = _stable_candidate_order(candidates, seed=seed)
                if profile_name == "blind":
                    seed_candidates = ordered_candidates[: args.seed_candidates]
                    pool = ordered_candidates[args.seed_candidates :]
                else:
                    seed_candidates = []
                    pool = ordered_candidates
                for configuration in configurations_by_profile[profile_name]:
                    path = Path(directory) / f"{profile_name}-{configuration.name}-{seed}.sqlite"
                    if profile_name == "blind":
                        evaluator, controller, visible = _seed_state(
                            path,
                            oracle=oracle,
                            shapes=shapes,
                            seed_candidates=seed_candidates,
                            seed_clustering=seed_clustering,
                            source_ref=f"{profile_name}:{configuration.name}:{seed}",
                        )
                        references = None
                    else:
                        evaluator, controller, visible, references = _anchored_state(
                            path,
                            oracle=oracle,
                            shapes=shapes,
                            baseline_pairs=profile_input.baseline_pairs,
                            baseline_label=profile_name,
                        )
                    before_pairs = len(controller.queried_pairs)
                    increment = _run_increment(
                        evaluator,
                        controller,
                        visible,
                        candidates=pool,
                        shapes=shapes,
                        configuration=configuration,
                        pair_budget=args.pair_budget,
                        estimators=args.estimators,
                        seed=seed,
                        allow_repair=True,
                        reference_performance=references,
                    )
                    added_pairs = len(controller.queried_pairs) - before_pairs
                    incumbent_performance = {
                        shape_id: incumbent.performance for shape_id, incumbent in controller.incumbents.items()
                    }
                    for fold_id, shape_ids in folds.items():
                        mean, p95, worst, unresolved = fold_regret_metrics(
                            shape_ids=shape_ids,
                            oracle_best=oracle_best,
                            incumbent_performance=incumbent_performance,
                        )
                        observations.append(
                            PolicyTrialObservation(
                                configuration_id=configuration.identity_hash,
                                seed=seed,
                                ordering_id=f"shuffle:{seed}",
                                fold_id=fold_id,
                                mean_log_regret=mean,
                                p95_log_regret=p95,
                                worst_log_regret=worst,
                                unresolved_shapes=unresolved,
                                queried_pairs=added_pairs,
                                unknown_pairs=len(controller.unknown_pairs),
                                prepared_candidates=len(controller.prepared_artifact_shapes),
                            )
                        )
                    selected_unknown_pairs[configuration.identity_hash].update(controller.unknown_pairs)
                    run_reports.append(
                        {
                            "initialization": profile_name,
                            "configuration_id": configuration.identity_hash,
                            "seed": seed,
                            "ordering_id": f"shuffle:{seed}",
                            "added_pairs": added_pairs,
                            "increment": increment,
                            "summary": _controller_summary(controller, oracle_best),
                        }
                    )
        aggregates_by_profile = {
            profile_name: aggregate_policy_trials(
                [
                    observation
                    for observation in observations
                    if observation.configuration_id
                    in {configuration.identity_hash for configuration in profile_configurations}
                ]
            )
            for profile_name, profile_configurations in configurations_by_profile.items()
        }
        aggregates = tuple(
            aggregate for profile_aggregates in aggregates_by_profile.values() for aggregate in profile_aggregates
        )
        for profile_name, profile_configurations in configurations_by_profile.items():
            profile_aggregates = list(aggregates_by_profile[profile_name])
            selected = select_robust_default(profile_aggregates)
            selected_configuration = next(
                configuration
                for configuration in profile_configurations
                if configuration.identity_hash == selected.configuration_id
            )
            selected_by_profile[profile_name] = SelectedProfile(
                aggregate=selected,
                configuration=selected_configuration,
            )
            profile_input = profile_inputs[profile_name]
            oracle = profile_input.oracle
            candidates = sorted(
                {record.candidate.hash: record.candidate for record in oracle.values()}.values(),
                key=lambda candidate: candidate.hash,
            )
            candidates = _stable_candidate_order(candidates, seed=12345)
            if profile_name == "blind":
                seed_candidates = candidates[: args.seed_candidates]
                pool = candidates[args.seed_candidates :]
            else:
                seed_candidates = []
                pool = candidates
            for schedule_name, role_repairs, budget_fractions in (
                ("fixed", (True, True), (0.5, 0.5)),
                ("role_specialized", (False, True), (0.6, 0.4)),
            ):
                path = Path(directory) / f"multi-{profile_name}-{schedule_name}.sqlite"
                if profile_name == "blind":
                    evaluator, controller, visible = _seed_state(
                        path,
                        oracle=oracle,
                        shapes=shapes,
                        seed_candidates=seed_candidates,
                        seed_clustering=seed_clustering,
                        source_ref=f"multi:{profile_name}:{schedule_name}",
                    )
                    references = None
                else:
                    evaluator, controller, visible, references = _anchored_state(
                        path,
                        oracle=oracle,
                        shapes=shapes,
                        baseline_pairs=profile_input.baseline_pairs,
                        baseline_label=profile_name,
                    )
                before_pairs = len(controller.queried_pairs)
                reports = []
                allocated_budgets = [round(args.pair_budget * fraction) for fraction in budget_fractions]
                allocated_budgets[-1] += args.pair_budget - sum(allocated_budgets)
                for round_index, (allow_repair, round_budget) in enumerate(
                    zip(role_repairs, allocated_budgets, strict=True)
                ):
                    reports.append(
                        _run_increment(
                            evaluator,
                            controller,
                            visible,
                            candidates=pool,
                            shapes=shapes,
                            configuration=selected_configuration,
                            pair_budget=round_budget,
                            estimators=args.estimators,
                            seed=12345 + round_index,
                            allow_repair=allow_repair,
                            reference_performance=references,
                        )
                    )
                multi_round.append(
                    {
                        "initialization": profile_name,
                        "schedule": schedule_name,
                        "pair_budget_fractions": list(budget_fractions),
                        "rounds": reports,
                        "added_pairs": len(controller.queried_pairs) - before_pairs,
                        "summary": _controller_summary(
                            controller,
                            oracle_best_by_profile[profile_name],
                        ),
                    }
                )
    overlay_pairs = set()
    if args.overlay_db is not None and args.overlay_db.exists():
        overlay_pairs = set(load_db_oracle_matrix(args.overlay_db, shapes=shapes))
    hybrid_finalists = {}
    for profile_name, selected in selected_by_profile.items():
        configuration_id = selected.configuration.identity_hash
        across_seed_unknown = selected_unknown_pairs[configuration_id]
        canonical_run = next(
            report
            for report in run_reports
            if report["initialization"] == profile_name
            and report["configuration_id"] == configuration_id
            and report["seed"] == 12345
        )
        canonical_increment = cast(dict[str, object], canonical_run["increment"])
        unknown_pair_keys = cast(list[list[str]], canonical_increment["unknown_pair_keys"])
        canonical_unknown = {tuple(pair) for pair in unknown_pair_keys}
        hybrid_finalists[profile_name] = {
            "canonical_seed": 12345,
            "canonical_unknown_pairs": len(canonical_unknown),
            "across_seed_unknown_pairs": len(across_seed_unknown),
            "overlay_pairs_available": len(canonical_unknown & overlay_pairs),
            "remaining_native_pairs": [list(pair) for pair in sorted(canonical_unknown - overlay_pairs)],
            "overlay_required_for_default": bool(canonical_unknown - overlay_pairs),
        }
    selected_multi_round = {}
    for profile_name in profile_inputs:
        profile_rows = [row for row in multi_round if row["initialization"] == profile_name]
        selected_row, selection = _select_multi_round_schedule(profile_rows)
        objective = "pareto_normalized_regret_tail_coverage_cost"
        selected_multi_round[profile_name] = {
            "schedule": selected_row["schedule"],
            "objective": objective,
            "pair_budget_fractions": selected_row["pair_budget_fractions"],
            "selection": selection,
            "summary": selected_row["summary"],
        }
    singleton_tuning = json.loads(args.singleton_tuning.read_text(encoding="utf-8"))
    payload = {
        "source_db": str(args.db),
        "initialization_sources": {
            profile_name: profile_input.metadata for profile_name, profile_input in profile_inputs.items()
        },
        "overlay_db": None if args.overlay_db is None else str(args.overlay_db),
        "shape_count": len(shapes),
        "seed_candidate_count": args.seed_candidates,
        "pair_budget": args.pair_budget,
        "seeds": args.seeds,
        "mechanical_folds": {fold_id: list(shape_ids) for fold_id, shape_ids in folds.items()},
        "configurations": {configuration.identity_hash: configuration.to_dict() for configuration in configurations},
        "trials": [observation.to_dict() for observation in observations],
        "run_reports": run_reports,
        "aggregates": [aggregate.to_dict() for aggregate in aggregates],
        "selected_by_initialization": {
            profile_name: {
                "configuration_id": selected.configuration.identity_hash,
                "configuration": selected.configuration.to_dict(),
                "aggregate": selected.aggregate.to_dict(),
            }
            for profile_name, selected in selected_by_profile.items()
        },
        "multi_round_comparison": multi_round,
        "selected_multi_round_by_initialization": selected_multi_round,
        "hybrid_finalists": hybrid_finalists,
        "singleton": {
            "shared_configuration_schema": True,
            "default_changed": singleton_tuning["default_changed"],
            "selected_default": singleton_tuning["selected_default"],
            "best_bundle_policy": singleton_tuning["best_bundle_policy"],
            "aggregate": singleton_tuning["aggregate"],
            "evidence": str(args.singleton_tuning),
        },
        "selection_summary": {
            "mean_seed_regret_variance": statistics.fmean(
                aggregate.seed_mean_regret_variance for aggregate in aggregates
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
