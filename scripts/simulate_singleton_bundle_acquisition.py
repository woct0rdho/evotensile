#!/usr/bin/env python3

import argparse
import json
import math
import random
import tempfile
from pathlib import Path

from evotensile.campaign.acquisition import BundleAcquisitionPolicy, BundleCostModel, plan_candidate_bundles
from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.evaluator import ReplayEvaluator
from evotensile.campaign.protocols import CAMPAIGN_SCREENING_PROTOCOL
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.scheduling.models import EvidenceStage, PairRequest
from evotensile.search.evidence import load_proposal_evidence_snapshot
from evotensile.search.pair_model import ContextualPairModel, PairModelConfiguration
from evotensile.search.replay import ExactOracleReplayState, load_db_oracle_matrix
from evotensile.search.surrogate import select_surrogate_pool
from evotensile.shapes import pilot_100_shapes


def _state(path, shape, oracle, name):
    db = EvoTensileDB.connect(
        path,
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    evaluator = ReplayEvaluator(
        ExactOracleReplayState(
            db=db,
            shapes=[shape],
            oracle=oracle,
            profile=DEFAULT_PROFILE,
            source_ref=name,
        ),
        prepare_workers=4,
        prepare_seconds_per_candidate=0.1,
    )
    controller = CampaignControllerState(
        shape_ids=(shape.id,),
        time_budget_s=300.0,
        session_started_at=0.0,
    )
    return evaluator, controller


def _evaluate(evaluator, controller, candidates, shape):
    result = evaluator.evaluate(
        [PairRequest(candidate, shape, evidence_stage=EvidenceStage.PROBE) for candidate in candidates]
    )
    result.apply(controller)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("out/grid100_full_20260618_repaired.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("out/grid100_singleton_bundle_20260712.json"))
    parser.add_argument("--seed-evidence", type=int, default=32)
    parser.add_argument("--shortlist", type=int, default=16)
    parser.add_argument("--estimators", type=int, default=96)
    args = parser.parse_args()
    shapes = pilot_100_shapes()
    selected_shapes = [shapes[index] for index in (0, 24, 49, 74, 99)]
    full_oracle = load_db_oracle_matrix(args.db, shapes=selected_shapes)
    rows = []
    with tempfile.TemporaryDirectory(prefix="evotensile-singleton-acquisition-") as directory:
        for shape_index, shape in enumerate(selected_shapes):
            shape_seed = 12345 + shape_index * 3
            shape_oracle = {key: record for key, record in full_oracle.items() if key[0] == shape.id}
            candidates = sorted(
                {
                    record.candidate.hash: record.candidate
                    for record in shape_oracle.values()
                    if record.screening_gflops is not None and record.screening_gflops > 0.0
                }.values(),
                key=lambda candidate: candidate.hash,
            )
            random.Random(shape_seed).shuffle(candidates)
            seed_candidates = candidates[: args.seed_evidence]
            pool = candidates[args.seed_evidence :]
            oracle_best = max(record.screening_gflops or 0.0 for record in shape_oracle.values())

            evaluator, controller = _state(
                Path(directory) / f"surrogate_{shape_index}.sqlite",
                shape,
                shape_oracle,
                f"surrogate-{shape.id}",
            )
            _evaluate(evaluator, controller, seed_candidates, shape)
            seed_performance = controller.incumbents[shape.id].performance
            snapshot = load_proposal_evidence_snapshot(
                evaluator.state.db,
                problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
                benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(CAMPAIGN_SCREENING_PROTOCOL),
                shapes=[shape],
            )
            surrogate_selected = select_surrogate_pool(
                pool,
                evidence=snapshot,
                shapes=[shape],
                count=min(args.shortlist, len(pool)),
                seed=shape_seed + 1,
                min_evidence=24,
                surrogate_jobs=DEFAULT_PROFILE.default_surrogate_jobs,
                workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
            )
            _evaluate(evaluator, controller, surrogate_selected, shape)
            surrogate_performance = controller.incumbents[shape.id].performance

            evaluator, controller = _state(
                Path(directory) / f"bundle_{shape_index}.sqlite",
                shape,
                shape_oracle,
                f"bundle-{shape.id}",
            )
            seed_result = _evaluate(evaluator, controller, seed_candidates, shape)
            model = ContextualPairModel(
                workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
                configuration=PairModelConfiguration(
                    n_estimators=args.estimators,
                    min_performance_rows=24,
                    seed=shape_seed + 2,
                    jobs=DEFAULT_PROFILE.default_surrogate_jobs,
                ),
            )
            model.fit(seed_result.outcomes)
            predictions = model.predict([(candidate, shape) for candidate in pool])
            plan = plan_candidate_bundles(
                controller,
                candidates=pool,
                shapes=[shape],
                predictions=predictions,
                cost_model=BundleCostModel(
                    workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
                    fallback_preparation_s=0.1,
                    fallback_validation_s=0.0,
                    fallback_timing_s=0.001,
                ),
                policy=BundleAcquisitionPolicy(
                    coverage_weight=0.0,
                    information_weight=0.1,
                    bundle_sizes=(1,),
                    max_pairs=min(args.shortlist, len(pool)),
                    max_bundles=min(args.shortlist, len(pool)),
                    max_predicted_cost_s=300.0,
                    evidence_stage=EvidenceStage.PROBE,
                ),
            )
            result = evaluator.evaluate(
                plan.timing_requests,
                artifact_shapes_by_candidate=plan.artifact_shapes_by_candidate,
            )
            result.apply(controller)
            bundle_performance = controller.incumbents[shape.id].performance
            rows.append(
                {
                    "shape_id": shape.id,
                    "candidate_count": len(candidates),
                    "seed_performance": seed_performance,
                    "surrogate_performance": surrogate_performance,
                    "bundle_performance": bundle_performance,
                    "surrogate_log_regret": math.log(oracle_best / surrogate_performance),
                    "bundle_log_regret": math.log(oracle_best / bundle_performance),
                    "surrogate_improved_seed": surrogate_performance > seed_performance,
                    "bundle_improved_seed": bundle_performance > seed_performance,
                    "winner": (
                        "bundle"
                        if bundle_performance > surrogate_performance
                        else "surrogate"
                        if surrogate_performance > bundle_performance
                        else "tie"
                    ),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "source_db": str(args.db),
                "seed_evidence": args.seed_evidence,
                "shortlist": args.shortlist,
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
