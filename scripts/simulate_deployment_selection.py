#!/usr/bin/env python3

import argparse
import json
import statistics
import tempfile
from pathlib import Path

from tune_campaign_policy import (
    _anchored_state,
    _model,
    _run_increment,
    _stable_candidate_order,
)

from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.deployment import (
    FinalistConfirmationPolicy,
    plan_confirmation_finalists,
    plan_stabilization_finalists,
    run_final_confirmation,
    select_deployment_solution_bank,
)
from evotensile.campaign.policy import selected_campaign_policy
from evotensile.campaign.workload import ResolvedWorkloadWeights
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.search.replay import load_db_oracle_matrix
from evotensile.shapes import pilot_100_shapes


def _load_workload(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ResolvedWorkloadWeights.from_dict(payload["workload"])


def _aggregate(rows, tolerances):
    output = {}
    for tolerance in tolerances:
        key = f"{tolerance:.3f}"
        selections = [row["selections"][key] for row in rows]
        output[key] = {
            "trials": len(selections),
            "mean_solution_count": statistics.fmean(selection["solution_count"] for selection in selections),
            "minimum_solution_count": min(selection["solution_count"] for selection in selections),
            "maximum_solution_count": max(selection["solution_count"] for selection in selections),
            "mean_code_object_count": statistics.fmean(selection["code_object_count"] for selection in selections),
            "mean_generalist_count": statistics.fmean(
                len(selection["generalist_coverage"]) for selection in selections
            ),
            "mean_specialist_shape_count": statistics.fmean(
                len(selection["specialist_shape_ids"]) for selection in selections
            ),
            "mean_uniform_loss_fraction": statistics.fmean(
                selection["uniform_mean_loss_fraction"] for selection in selections
            ),
            "mean_workload_weighted_loss_fraction": statistics.fmean(
                selection["workload_weighted_mean_loss_fraction"] for selection in selections
            ),
            "worst_shape_loss_fraction": max(selection["worst_shape_loss_fraction"] for selection in selections),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("out/grid100_compatible_20260712.sqlite"),
    )
    parser.add_argument(
        "--baseline-db",
        type=Path,
        default=Path("out/grid100_untuned_hipblaslt_baseline_20260712.sqlite"),
    )
    parser.add_argument(
        "--workload-artifact",
        type=Path,
        default=Path("out/grid100_workload_weighting_20260712.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/grid100_deployment_selection_20260712.json"),
    )
    parser.add_argument("--pair-budget", type=int, default=385)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--estimators", type=int, default=64)
    args = parser.parse_args()
    if args.pair_budget <= 0 or args.seeds <= 0 or args.estimators <= 0:
        raise ValueError("deployment replay budgets, seeds, and estimators must be positive")

    shapes = pilot_100_shapes()
    shape_by_id = {shape.id: shape for shape in shapes}
    oracle = load_db_oracle_matrix(
        args.db,
        shapes=shapes,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
    )
    baseline_db = EvoTensileDB.connect(
        args.baseline_db,
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    discoveries = baseline_db.baseline_discoveries(baseline_label="anchored-untuned")
    if not discoveries:
        raise ValueError("deployment replay requires anchored-untuned baseline discovery")
    baseline_pairs = baseline_db.baseline_selection_pairs(discoveries[-1].discovery_id)
    workload = _load_workload(args.workload_artifact)
    if workload.shape_ids != tuple(shape.id for shape in shapes):
        raise ValueError("deployment workload must match the pilot shape set")
    catalog = sorted(
        {record.candidate.hash: record.candidate for record in oracle.values()}.values(),
        key=lambda candidate: candidate.hash,
    )
    candidate_by_hash = {candidate.hash: candidate for candidate in catalog}
    campaign_policy = selected_campaign_policy(
        "anchored-untuned",
        pair_budget=args.pair_budget,
    )
    finalist_policy = FinalistConfirmationPolicy(
        relative_tolerance=0.02,
        minimum_close_probability=0.10,
        maximum_finalists_per_shape=3,
        stabilization_samples=4,
        min_samples=10,
        fallback_group_cost_s=1.0,
    )
    tolerances = (0.0, 0.01, 0.02, 0.05)
    rows = []

    with tempfile.TemporaryDirectory(prefix="evotensile-deployment-replay-") as directory:
        for seed_index in range(args.seeds):
            seed = 20260712 + seed_index
            candidates = _stable_candidate_order(catalog, seed=seed)
            evaluator, campaign_controller, observations, references = _anchored_state(
                Path(directory) / f"campaign-{seed}.sqlite",
                oracle=oracle,
                shapes=shapes,
                baseline_pairs=baseline_pairs,
                baseline_label=f"anchored-untuned:deployment:{seed}",
            )
            campaign_controller.set_workload(workload)
            increment = _run_increment(
                evaluator,
                campaign_controller,
                observations,
                candidates=candidates,
                shapes=shapes,
                configuration=campaign_policy,
                pair_budget=args.pair_budget,
                estimators=args.estimators,
                seed=seed,
                allow_repair=True,
                reference_performance=references,
            )
            model = _model(observations, estimators=args.estimators, seed=seed + 101)
            predictions = model.predict([(candidate, shape) for candidate in candidates for shape in shapes])
            incumbents = {
                shape_id: incumbent.candidate_hash for shape_id, incumbent in campaign_controller.incumbents.items()
            }
            stabilization_plan = plan_stabilization_finalists(
                predictions,
                candidates=candidate_by_hash,
                shapes=shape_by_id,
                incumbent_candidates_by_shape=incumbents,
                shape_weights=workload.weights,
                policy=finalist_policy,
            )
            stabilization_result = evaluator.evaluate(stabilization_plan.requests)
            stabilization_result.apply(campaign_controller)
            observations.extend(stabilization_result.outcomes)

            confirmation_model = _model(
                observations,
                estimators=args.estimators,
                seed=seed + 202,
            )
            confirmation_predictions = confirmation_model.predict(
                [(candidate, shape) for candidate in candidates for shape in shapes]
            )
            stabilized_incumbents = {
                shape_id: incumbent.candidate_hash for shape_id, incumbent in campaign_controller.incumbents.items()
            }
            confirmation_plan = plan_confirmation_finalists(
                confirmation_predictions,
                candidates=candidate_by_hash,
                shapes=shape_by_id,
                incumbent_candidates_by_shape=stabilized_incumbents,
                shape_weights=workload.weights,
                policy=finalist_policy,
            )
            confirmation_controller = CampaignControllerState(
                shape_ids=tuple(shape.id for shape in shapes),
                time_budget_s=300.0,
                session_started_at=0.0,
            )
            confirmation_controller.set_workload(workload)
            clock = [0.0]
            confirmation = run_final_confirmation(
                evaluator,
                confirmation_controller,
                confirmation_plan,
                now=lambda: clock[0],
                charge_result_time=lambda result: clock.__setitem__(
                    0,
                    clock[0] + sum(result.phase_time_s.values()),
                ),
            )
            selections = {
                f"{tolerance:.3f}": select_deployment_solution_bank(
                    confirmation.outcomes,
                    shape_ids=[shape.id for shape in shapes],
                    tolerance_fraction=tolerance,
                    shape_weights=workload.weights,
                ).to_dict()
                for tolerance in tolerances
            }
            rows.append(
                {
                    "seed": seed,
                    "campaign_increment": increment,
                    "stabilization": {
                        "planned_pairs": len(stabilization_plan.requests),
                        "known_pairs": stabilization_result.known_pairs,
                        "unknown_pairs": stabilization_result.unknown_pairs,
                    },
                    "confirmation": {
                        **confirmation.to_dict(),
                        "simulated_time_s": clock[0],
                        "budget_overrun_s": confirmation_controller.overrun_s(now=clock[0]),
                    },
                    "selections": selections,
                }
            )

    output = {
        "source_db": str(args.db),
        "baseline_db": str(args.baseline_db),
        "workload_artifact": str(args.workload_artifact),
        "initialization": "anchored-untuned",
        "campaign_policy": campaign_policy.to_dict(),
        "finalist_policy": finalist_policy.to_dict(),
        "pair_budget": args.pair_budget,
        "confirmation_soft_budget_s": 300.0,
        "tolerances": list(tolerances),
        "code_object_counting": "conservative one logical code object per selected candidate",
        "rows": rows,
        "aggregate": _aggregate(rows, tolerances),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
