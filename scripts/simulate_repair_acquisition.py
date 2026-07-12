#!/usr/bin/env python3

import argparse
import json
import tempfile
from pathlib import Path

from evotensile.campaign.acquisition import BundleAcquisitionPolicy, BundleCostModel, plan_candidate_bundles
from evotensile.campaign.baselines import evaluate_representative_first_baseline
from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.evaluator import ReplayEvaluator
from evotensile.campaign.repair import (
    RepairPolicy,
    assess_repair_deficits,
    build_repair_candidate_pool,
    plan_repair_acquisition,
    summarize_repair,
)
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.scheduling.models import EvidenceStage
from evotensile.search.pair_model import ContextualPairModel, PairModelConfiguration
from evotensile.search.replay import ExactOracleReplayState, load_db_oracle_matrix
from evotensile.search.shape_clustering import ShapeClusteringConfiguration, cluster_shapes
from evotensile.shapes import pilot_100_shapes


def _seed(path, *, oracle, shapes, candidates, clustering):
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
            source_ref=path.stem,
        ),
        prepare_workers=4,
        prepare_seconds_per_candidate=0.1,
    )
    controller = CampaignControllerState(
        shape_ids=tuple(shape.id for shape in shapes),
        time_budget_s=300.0,
        session_started_at=0.0,
    )
    controller.set_clustering(clustering.to_dict())
    seed = evaluate_representative_first_baseline(
        evaluator,
        controller,
        candidates=candidates,
        shapes=shapes,
        clustering=clustering,
    )
    return evaluator, controller, seed.result


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


def _broad_policy(pair_budget):
    return BundleAcquisitionPolicy(
        coverage_weight=0.5,
        information_weight=0.1,
        bundle_sizes=(1, 2, 4, 8, 16),
        max_pairs=pair_budget,
        max_bundles=64,
        max_predicted_cost_s=300.0,
        evidence_stage=EvidenceStage.PROBE,
    )


def _summary(controller, *, oracle_best, added_from):
    metrics = controller.grid_metrics(oracle_best)
    return {
        "added_pairs": len(controller.queried_pairs) - added_from,
        "total_pairs": len(controller.queried_pairs),
        "known_pairs": len(controller.known_pairs),
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
    parser.add_argument("--output", type=Path, default=Path("out/grid100_repair_acquisition_20260712.json"))
    parser.add_argument("--seed-candidates", type=int, default=80)
    parser.add_argument("--pair-budget", type=int, default=385)
    parser.add_argument("--repair-pairs", type=int, default=12)
    parser.add_argument("--estimators", type=int, default=96)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    if args.repair_pairs <= 0 or args.repair_pairs >= args.pair_budget:
        raise ValueError("repair pair reserve must be positive and smaller than total pair budget")
    shapes = pilot_100_shapes()
    oracle = load_db_oracle_matrix(args.db, shapes=shapes)
    candidate_by_hash = {record.candidate.hash: record.candidate for record in oracle.values()}
    candidates = sorted(candidate_by_hash.values(), key=lambda candidate: candidate.hash)
    seed_candidates = candidates[: args.seed_candidates]
    clustering = cluster_shapes(
        shapes,
        ShapeClusteringConfiguration(
            workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
            cluster_count=16,
        ),
    )
    oracle_best = {
        shape.id: max(
            record.screening_gflops or 0.0 for (shape_id, _), record in oracle.items() if shape_id == shape.id
        )
        for shape in shapes
    }
    rows = []
    with tempfile.TemporaryDirectory(prefix="evotensile-repair-") as directory:
        broad_pairs = args.pair_budget - args.repair_pairs
        for name in ("broad_continuation", "broad_plus_repair"):
            evaluator, controller, seed_result = _seed(
                Path(directory) / f"{name}.sqlite",
                oracle=oracle,
                shapes=shapes,
                candidates=seed_candidates,
                clustering=clustering,
            )
            initial_pairs = len(controller.queried_pairs)
            model = _model(seed_result.outcomes, estimators=args.estimators, seed=args.seed)
            predictions = model.predict([(candidate, shape) for candidate in candidates for shape in shapes])
            broad_plan = plan_candidate_bundles(
                controller,
                candidates=candidates,
                shapes=shapes,
                predictions=predictions,
                cost_model=_cost_model(args.seed + 1),
                policy=_broad_policy(broad_pairs),
            )
            broad_result = evaluator.evaluate(
                broad_plan.timing_requests,
                artifact_shapes_by_candidate=broad_plan.artifact_shapes_by_candidate,
            )
            broad_result.apply(controller)
            observations = (*seed_result.outcomes, *broad_result.outcomes)
            continuation_model = _model(observations, estimators=args.estimators, seed=args.seed + 2)
            extra = {
                "broad_pairs": len(broad_plan.timing_requests),
                "reserve_pairs": args.repair_pairs,
            }
            if name == "broad_continuation":
                continuation_predictions = continuation_model.predict(
                    [(candidate, shape) for candidate in candidates for shape in shapes]
                )
                continuation_plan = plan_candidate_bundles(
                    controller,
                    candidates=candidates,
                    shapes=shapes,
                    predictions=continuation_predictions,
                    cost_model=_cost_model(args.seed + 3),
                    policy=_broad_policy(args.repair_pairs),
                )
                continuation_result = evaluator.evaluate(
                    continuation_plan.timing_requests,
                    artifact_shapes_by_candidate=continuation_plan.artifact_shapes_by_candidate,
                )
                continuation_result.apply(controller)
                extra["continuation_pairs"] = len(continuation_plan.timing_requests)
            else:
                repair_model = continuation_model
                catalog_predictions = repair_model.predict(
                    [(candidate, shape) for candidate in candidates for shape in shapes]
                )
                repair_policy = RepairPolicy(
                    uncertainty_weight=0.0,
                    mutation_candidates_per_shape=0,
                    seed=args.seed + 4,
                )
                deficits = assess_repair_deficits(
                    controller,
                    shapes=shapes,
                    clustering=clustering,
                    predictions=catalog_predictions,
                    policy=repair_policy,
                )
                broad_candidates = [score.bundle.candidate for score in broad_plan.selected]
                pool = build_repair_candidate_pool(
                    controller,
                    shapes=shapes,
                    clustering=clustering,
                    deficits=deficits,
                    observations=observations,
                    candidate_catalog=candidate_by_hash,
                    broad_candidates=broad_candidates,
                    policy=repair_policy,
                )
                repair_predictions = repair_model.predict(
                    [(candidate, shape) for candidate in pool.candidates for shape in shapes]
                )
                prepared_before = {
                    candidate_hash: set(shape_ids)
                    for candidate_hash, shape_ids in controller.prepared_artifact_shapes.items()
                }
                repair = plan_repair_acquisition(
                    controller,
                    candidates=pool.candidates,
                    shapes=shapes,
                    deficits=deficits,
                    predictions=repair_predictions,
                    cost_model=_cost_model(args.seed + 3),
                    acquisition_policy=BundleAcquisitionPolicy(
                        improvement_weight=0.0,
                        coverage_weight=0.0,
                        information_weight=0.0,
                        repair_weight=1.0,
                        bundle_sizes=(1, 2, 4),
                        max_pairs=args.repair_pairs,
                        max_bundles=args.repair_pairs,
                        max_predicted_cost_s=300.0,
                        evidence_stage=EvidenceStage.PROBE,
                    ),
                    repair_policy=repair_policy,
                )
                repair_result = evaluator.evaluate(
                    repair.plan.timing_requests,
                    artifact_shapes_by_candidate=repair.plan.artifact_shapes_by_candidate,
                )
                repair_result.apply(controller)
                report = summarize_repair(
                    repair,
                    controller_after=controller,
                    prepared_artifact_shapes_before=prepared_before,
                    useful_close_fraction=repair_policy.useful_close_fraction,
                )
                extra.update(
                    {
                        "repair_pairs": len(repair.plan.timing_requests),
                        "repair_candidate_pool": pool.to_dict(),
                        "repair_acquisition": repair.to_dict(),
                        "repair_report": report.to_dict(),
                    }
                )
            rows.append(
                {
                    "policy": name,
                    **_summary(controller, oracle_best=oracle_best, added_from=initial_pairs),
                    **extra,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "source_db": str(args.db),
                "shapes": len(shapes),
                "candidate_count": len(candidates),
                "seed_candidate_count": len(seed_candidates),
                "pair_budget": args.pair_budget,
                "repair_pair_reserve": args.repair_pairs,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
