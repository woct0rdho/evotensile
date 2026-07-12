#!/usr/bin/env python3

import argparse
import json
import tempfile
from pathlib import Path

from evotensile.campaign.acquisition import (
    BundleAcquisitionPolicy,
    BundleCostModel,
    plan_candidate_bundles,
)
from evotensile.campaign.baselines import evaluate_representative_first_baseline
from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.evaluator import ReplayEvaluator
from evotensile.campaign.promotion import PromotionPolicy, execute_promotion_race
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.scheduling.models import EvidenceStage, PairRequest
from evotensile.search.pair_model import ContextualPairModel, PairModelConfiguration
from evotensile.search.replay import ExactOracleReplayState, load_db_oracle_matrix
from evotensile.search.shape_clustering import ShapeClusteringConfiguration, cluster_shapes
from evotensile.shapes import pilot_100_shapes


def _seed_campaign(path, name, *, oracle, shapes, candidates, clustering):
    db = EvoTensileDB.connect(
        path,
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    state = ExactOracleReplayState(
        db=db,
        shapes=shapes,
        oracle=oracle,
        profile=DEFAULT_PROFILE,
        source_ref=name,
    )
    evaluator = ReplayEvaluator(state, prepare_workers=4, prepare_seconds_per_candidate=0.1)
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


def _apply_requests(evaluator, controller, requests, *, artifact_shapes=None):
    if not requests:
        return None
    result = evaluator.evaluate(requests, artifact_shapes_by_candidate=artifact_shapes)
    result.apply(controller)
    return result


def _summary(name, controller, *, before_pairs, oracle_best, extra=None):
    metrics = controller.grid_metrics(oracle_best)
    return {
        "policy": name,
        "added_pairs": len(controller.queried_pairs) - before_pairs,
        "total_pairs": len(controller.queried_pairs),
        "known_pairs": len(controller.known_pairs),
        "resolved_shapes": metrics.resolved_shapes,
        "unresolved_shapes": metrics.unresolved_shapes,
        "mean_log_regret": metrics.mean_log_regret,
        "p95_log_regret": metrics.p95_log_regret,
        "worst_log_regret": metrics.worst_log_regret,
        "prepared_candidates": len(controller.prepared_artifact_shapes),
        "phase_time_s": controller.phase_time_s,
        **(extra or {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("out/grid100_full_20260618_repaired.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("out/grid100_bundle_acquisition_20260712.json"))
    parser.add_argument("--seed-candidates", type=int, default=80)
    parser.add_argument("--pair-budget", type=int, default=385)
    parser.add_argument("--estimators", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
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
    with tempfile.TemporaryDirectory(prefix="evotensile-acquisition-") as directory:
        _, _, seed_result = _seed_campaign(
            Path(directory) / "model.sqlite",
            "model-seed",
            oracle=oracle,
            shapes=shapes,
            candidates=seed_candidates,
            clustering=clustering,
        )
        model = ContextualPairModel(
            workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
            configuration=PairModelConfiguration(
                n_estimators=args.estimators,
                min_performance_rows=24,
                seed=args.seed,
                jobs=DEFAULT_PROFILE.default_surrogate_jobs,
            ),
        )
        fit_summary = model.fit(seed_result.outcomes)
        predictions = model.predict([(candidate, shape) for candidate in candidates for shape in shapes])
        prediction_by_key = {(prediction.shape_id, prediction.candidate_hash): prediction for prediction in predictions}
        rows = []

        evaluator, controller, seed = _seed_campaign(
            Path(directory) / "representative.sqlite",
            "representative",
            oracle=oracle,
            shapes=shapes,
            candidates=seed_candidates,
            clustering=clustering,
        )
        before_pairs = len(controller.queried_pairs)
        rows.append(_summary("representative_only", controller, before_pairs=before_pairs, oracle_best=oracle_best))

        evaluator, controller, seed = _seed_campaign(
            Path(directory) / "transfer.sqlite",
            "transfer",
            oracle=oracle,
            shapes=shapes,
            candidates=seed_candidates,
            clustering=clustering,
        )
        before_pairs = len(controller.queried_pairs)
        race = execute_promotion_race(
            evaluator,
            controller,
            shapes=shapes,
            clustering=clustering,
            observations=seed.outcomes,
            policy=PromotionPolicy(),
        )
        rows.append(
            _summary(
                "transfer",
                controller,
                before_pairs=before_pairs,
                oracle_best=oracle_best,
                extra={"probe_pairs": race.probe_pairs, "main_pairs": race.main_pairs},
            )
        )

        evaluator, controller, seed = _seed_campaign(
            Path(directory) / "dense.sqlite",
            "dense",
            oracle=oracle,
            shapes=shapes,
            candidates=seed_candidates,
            clustering=clustering,
        )
        before_pairs = len(controller.queried_pairs)
        dense_candidates = sorted(
            candidates,
            key=lambda candidate: (
                -sum(
                    prediction_by_key[(shape.id, candidate.hash)].mean_normalized_log_performance
                    * prediction_by_key[(shape.id, candidate.hash)].validity_probability
                    for shape in shapes
                ),
                candidate.hash,
            ),
        )
        dense_requests = []
        for candidate in dense_candidates:
            candidate_requests = [
                PairRequest(candidate, shape, evidence_stage=EvidenceStage.PROBE)
                for shape in shapes
                if (shape.id, candidate.hash) not in controller.queried_pairs
            ]
            if len(dense_requests) + len(candidate_requests) > args.pair_budget:
                continue
            dense_requests.extend(candidate_requests)
            if len(dense_requests) >= args.pair_budget:
                break
        _apply_requests(evaluator, controller, dense_requests)
        rows.append(_summary("global_dense", controller, before_pairs=before_pairs, oracle_best=oracle_best))

        evaluator, controller, seed = _seed_campaign(
            Path(directory) / "independent.sqlite",
            "independent",
            oracle=oracle,
            shapes=shapes,
            candidates=seed_candidates,
            clustering=clustering,
        )
        before_pairs = len(controller.queried_pairs)
        queues = {
            shape.id: sorted(
                (candidate for candidate in candidates if (shape.id, candidate.hash) not in controller.queried_pairs),
                key=lambda candidate: (
                    -prediction_by_key[(shape.id, candidate.hash)].mean_normalized_log_performance,
                    candidate.hash,
                ),
            )
            for shape in shapes
        }
        independent_requests = []
        rank = 0
        while len(independent_requests) < args.pair_budget:
            added = False
            for shape in shapes:
                if rank < len(queues[shape.id]) and len(independent_requests) < args.pair_budget:
                    independent_requests.append(
                        PairRequest(queues[shape.id][rank], shape, evidence_stage=EvidenceStage.PROBE)
                    )
                    added = True
            if not added:
                break
            rank += 1
        _apply_requests(evaluator, controller, independent_requests)
        rows.append(_summary("independent_model_rank", controller, before_pairs=before_pairs, oracle_best=oracle_best))

        joint_policies = {
            "joint_quality": BundleAcquisitionPolicy(
                coverage_weight=0.2,
                information_weight=0.1,
                bundle_sizes=(1, 2, 4, 8, 16),
                max_pairs=args.pair_budget,
                max_bundles=64,
                max_predicted_cost_s=300.0,
                evidence_stage=EvidenceStage.PROBE,
            ),
            "joint_balanced": BundleAcquisitionPolicy(
                coverage_weight=0.5,
                information_weight=0.1,
                bundle_sizes=(1, 2, 4, 8, 16),
                max_pairs=args.pair_budget,
                max_bundles=64,
                max_predicted_cost_s=300.0,
                evidence_stage=EvidenceStage.PROBE,
            ),
            "joint_coverage": BundleAcquisitionPolicy(
                coverage_weight=1.0,
                information_weight=0.1,
                bundle_sizes=(1, 2, 4, 8, 16),
                max_pairs=args.pair_budget,
                max_bundles=64,
                max_predicted_cost_s=300.0,
                evidence_stage=EvidenceStage.PROBE,
            ),
            "joint_information": BundleAcquisitionPolicy(
                coverage_weight=0.5,
                information_weight=0.3,
                bundle_sizes=(1, 2, 4, 8, 16),
                max_pairs=args.pair_budget,
                max_bundles=64,
                max_predicted_cost_s=300.0,
                evidence_stage=EvidenceStage.PROBE,
            ),
        }
        for joint_index, (joint_name, joint_policy) in enumerate(joint_policies.items()):
            evaluator, controller, seed = _seed_campaign(
                Path(directory) / f"joint_{joint_index}.sqlite",
                joint_name,
                oracle=oracle,
                shapes=shapes,
                candidates=seed_candidates,
                clustering=clustering,
            )
            before_pairs = len(controller.queried_pairs)
            cost_model = BundleCostModel(
                workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
                fallback_preparation_s=0.1,
                fallback_validation_s=0.0,
                fallback_timing_s=0.001,
                seed=args.seed,
                jobs=DEFAULT_PROFILE.default_surrogate_jobs,
            )
            plan = plan_candidate_bundles(
                controller,
                candidates=candidates,
                shapes=shapes,
                predictions=predictions,
                cost_model=cost_model,
                policy=joint_policy,
            )
            _apply_requests(
                evaluator,
                controller,
                plan.timing_requests,
                artifact_shapes=plan.artifact_shapes_by_candidate,
            )
            rows.append(
                _summary(
                    joint_name,
                    controller,
                    before_pairs=before_pairs,
                    oracle_best=oracle_best,
                    extra={
                        "selected_bundles": len(plan.selected),
                        "predicted_cost_s": plan.predicted_cost_s,
                        "preparation_order": list(plan.preparation_order),
                        "acquisition_plan": plan.to_dict(),
                    },
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "source_db": str(args.db),
                "shapes": len(shapes),
                "oracle_pairs": len(oracle),
                "candidate_count": len(candidates),
                "seed_candidate_count": len(seed_candidates),
                "clusters": len(clustering.clusters),
                "added_pair_budget": args.pair_budget,
                "model_fit": fit_summary.to_dict(),
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
