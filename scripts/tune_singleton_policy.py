#!/usr/bin/env python3

import argparse
import json
import math
import random
import statistics
import tempfile
from pathlib import Path
from typing import TypedDict

from evotensile.campaign.acquisition import BundleAcquisitionPolicy, BundleCostModel, plan_candidate_bundles
from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.evaluator import ReplayEvaluator
from evotensile.campaign.policy import CampaignPolicyConfiguration
from evotensile.campaign.protocols import CAMPAIGN_SCREENING_PROTOCOL
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.scheduling.models import EvidenceStage, PairRequest
from evotensile.search.evidence import load_proposal_evidence_snapshot
from evotensile.search.pair_model import ContextualPairModel, PairModelConfiguration
from evotensile.search.replay import ExactOracleReplayState, load_db_oracle_matrix
from evotensile.search.surrogate import select_surrogate_pool
from evotensile.shapes import pilot_100_shapes


class SingletonTrial(TypedDict):
    policy: str
    configuration_id: str | None
    seed: int
    shape_id: str
    log_regret: float


def _state(path, shape, oracle, source_ref):
    db = EvoTensileDB.connect(
        path,
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    return (
        ReplayEvaluator(
            ExactOracleReplayState(
                db=db,
                shapes=[shape],
                oracle=oracle,
                profile=DEFAULT_PROFILE,
                source_ref=source_ref,
            ),
            prepare_workers=4,
            prepare_seconds_per_candidate=0.1,
        ),
        CampaignControllerState(
            shape_ids=(shape.id,),
            time_budget_s=300.0,
            session_started_at=0.0,
        ),
    )


def _evaluate(evaluator, controller, candidates, shape):
    result = evaluator.evaluate(
        [PairRequest(candidate, shape, evidence_stage=EvidenceStage.PROBE) for candidate in candidates]
    )
    result.apply(controller)
    return result


def _bundle_configurations(shortlist):
    return tuple(
        CampaignPolicyConfiguration(
            name=f"singleton-information-{information_weight:g}",
            initialization_profile="blind",
            cluster_count=1,
            acquisition=BundleAcquisitionPolicy(
                coverage_weight=0.0,
                information_weight=information_weight,
                bundle_sizes=(1,),
                max_pairs=shortlist,
                max_bundles=shortlist,
                max_predicted_cost_s=300.0,
                evidence_stage=EvidenceStage.PROBE,
            ),
            singleton_acquisition_enabled=True,
        )
        for information_weight in (0.05, 0.10, 0.25)
    )


def _percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("out/grid100_full_20260618_repaired.sqlite"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/grid100_singleton_policy_tuning_20260712.json"),
    )
    parser.add_argument("--seed-evidence", type=int, default=32)
    parser.add_argument("--shortlist", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--estimators", type=int, default=64)
    args = parser.parse_args()
    shapes = pilot_100_shapes()
    selected_shapes = [shapes[index] for index in (0, 24, 49, 74, 99)]
    full_oracle = load_db_oracle_matrix(args.db, shapes=selected_shapes)
    configurations = _bundle_configurations(args.shortlist)
    rows: list[SingletonTrial] = []
    with tempfile.TemporaryDirectory(prefix="evotensile-singleton-tuning-") as directory:
        for seed_index in range(args.seeds):
            seed = 12345 + seed_index
            for shape_index, shape in enumerate(selected_shapes):
                shape_seed = seed + shape_index * 3
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
                    Path(directory) / f"surrogate-{seed}-{shape_index}.sqlite",
                    shape,
                    shape_oracle,
                    f"surrogate:{seed}:{shape.id}",
                )
                _evaluate(evaluator, controller, seed_candidates, shape)
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
                surrogate_regret = math.log(oracle_best / controller.incumbents[shape.id].performance)
                rows.append(
                    {
                        "policy": "existing-surrogate",
                        "configuration_id": None,
                        "seed": seed,
                        "shape_id": shape.id,
                        "log_regret": surrogate_regret,
                    }
                )

                for configuration in configurations:
                    evaluator, controller = _state(
                        Path(directory) / f"{configuration.name}-{seed}-{shape_index}.sqlite",
                        shape,
                        shape_oracle,
                        f"{configuration.name}:{seed}:{shape.id}",
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
                        policy=configuration.acquisition,
                    )
                    result = evaluator.evaluate(
                        plan.timing_requests,
                        artifact_shapes_by_candidate=plan.artifact_shapes_by_candidate,
                    )
                    result.apply(controller)
                    rows.append(
                        {
                            "policy": configuration.name,
                            "configuration_id": configuration.identity_hash,
                            "seed": seed,
                            "shape_id": shape.id,
                            "log_regret": math.log(oracle_best / controller.incumbents[shape.id].performance),
                        }
                    )
    aggregate = {}
    for policy in sorted({row["policy"] for row in rows}):
        regrets = [row["log_regret"] for row in rows if row["policy"] == policy]
        aggregate[policy] = {
            "trials": len(regrets),
            "mean_log_regret": statistics.fmean(regrets),
            "p95_log_regret": _percentile(regrets, 0.95),
            "worst_log_regret": max(regrets),
        }
    best_bundle = min(
        (configuration.name for configuration in configurations),
        key=lambda name: (
            aggregate[name]["mean_log_regret"],
            aggregate[name]["p95_log_regret"],
            aggregate[name]["worst_log_regret"],
            name,
        ),
    )
    surrogate = aggregate["existing-surrogate"]
    bundle = aggregate[best_bundle]
    bundle_dominates = (
        bundle["mean_log_regret"] < surrogate["mean_log_regret"]
        and bundle["p95_log_regret"] <= surrogate["p95_log_regret"]
        and bundle["worst_log_regret"] <= surrogate["worst_log_regret"]
    )
    output = {
        "source_db": str(args.db),
        "seed_evidence": args.seed_evidence,
        "shortlist": args.shortlist,
        "seeds": args.seeds,
        "shapes": [shape.id for shape in selected_shapes],
        "configurations": {configuration.identity_hash: configuration.to_dict() for configuration in configurations},
        "rows": rows,
        "aggregate": aggregate,
        "best_bundle_policy": best_bundle,
        "bundle_dominates_surrogate": bundle_dominates,
        "selected_default": best_bundle if bundle_dominates else "existing-surrogate",
        "default_changed": bundle_dominates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
