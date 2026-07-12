#!/usr/bin/env python3

import argparse
import json
import statistics
import tempfile
from pathlib import Path

from tune_campaign_policy import _anchored_state, _run_increment, _stable_candidate_order

from evotensile.campaign.policy import selected_campaign_policy
from evotensile.campaign.workload import ResolvedWorkloadWeights, ShapeWorkload
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.search.replay import load_db_oracle_matrix
from evotensile.shapes import pilot_100_shapes


def _time_us(shape, performance_gflops):
    return 2.0 * shape.m * shape.n * shape.batch * shape.k / (performance_gflops * 1e3)


def _oracle_best(oracle, shapes):
    return {
        shape.id: max(
            record.screening_gflops or 0.0 for (shape_id, _), record in oracle.items() if shape_id == shape.id
        )
        for shape in shapes
    }


def _metrics(controller, *, oracle_best, workload):
    uniform = controller.grid_metrics(
        oracle_best,
        weights={shape_id: 1.0 for shape_id in controller.shape_ids},
    )
    weighted = controller.grid_metrics(oracle_best, weights=workload.weights)
    unresolved_ids = [shape_id for shape_id in controller.shape_ids if shape_id not in controller.incumbents]
    return {
        "unweighted": uniform.to_dict(),
        "workload_weighted": weighted.to_dict(),
        "unresolved_workload_weight": sum(workload.weights[shape_id] for shape_id in unresolved_ids),
    }


def _aggregate(rows, mode):
    selected = [row for row in rows if row["mode"] == mode]
    return {
        "trials": len(selected),
        "mean_unweighted_log_regret": statistics.fmean(
            row["metrics"]["unweighted"]["mean_log_regret"] for row in selected
        ),
        "mean_workload_weighted_log_regret": statistics.fmean(
            row["metrics"]["workload_weighted"]["weighted_mean_log_regret"] for row in selected
        ),
        "mean_p95_log_regret": statistics.fmean(row["metrics"]["unweighted"]["p95_log_regret"] for row in selected),
        "worst_log_regret": max(row["metrics"]["unweighted"]["worst_log_regret"] for row in selected),
        "mean_unresolved_shapes": statistics.fmean(
            row["metrics"]["unweighted"]["unresolved_shapes"] for row in selected
        ),
        "mean_unresolved_workload_weight": statistics.fmean(
            row["metrics"]["unresolved_workload_weight"] for row in selected
        ),
        "mean_added_pairs": statistics.fmean(row["added_pairs"] for row in selected),
        "mean_unknown_pairs": statistics.fmean(row["unknown_pairs"] for row in selected),
        "mean_prepared_candidates": statistics.fmean(row["prepared_candidates"] for row in selected),
        "mean_simulated_time_s": statistics.fmean(row["simulated_time_s"] for row in selected),
        "mean_high_weight_pair_fraction": statistics.fmean(row["high_weight_pair_fraction"] for row in selected),
    }


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
        "--output",
        type=Path,
        default=Path("out/grid100_workload_weighting_20260712.json"),
    )
    parser.add_argument("--pair-budget", type=int, default=385)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--estimators", type=int, default=64)
    args = parser.parse_args()
    if args.pair_budget <= 0 or args.seeds <= 0 or args.estimators <= 0:
        raise ValueError("workload replay budgets, seeds, and estimators must be positive")

    shapes = pilot_100_shapes()
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
        raise ValueError("workload replay requires the anchored-untuned baseline discovery")
    baseline_pairs = baseline_db.baseline_selection_pairs(discoveries[-1].discovery_id)
    baseline_performance = {
        shape.id: oracle[(shape.id, candidate.hash)].screening_gflops for shape, candidate in baseline_pairs
    }
    if any(value is None or value <= 0.0 for value in baseline_performance.values()):
        raise ValueError("workload replay baseline pairs require positive exact timing")
    workload = ResolvedWorkloadWeights.workload(
        [shape.id for shape in shapes],
        [
            ShapeWorkload(
                shape.id,
                call_count=1.0,
                baseline_latency_us=_time_us(shape, baseline_performance[shape.id]),
            )
            for shape in shapes
        ],
        provenance={
            "call_count_source": "synthetic-one-call-per-shape",
            "baseline_label": "anchored-untuned",
            "baseline_source": str(args.baseline_db),
            "benchmark_protocol_hash": DEFAULT_PROFILE.benchmark_protocol_hash(),
            "environment_compatibility_tag": DEFAULT_PROFILE.environment_compatibility_tag,
        },
    )
    uniform = ResolvedWorkloadWeights.uniform([shape.id for shape in shapes])
    oracle_best = _oracle_best(oracle, shapes)
    catalog = sorted(
        {record.candidate.hash: record.candidate for record in oracle.values()}.values(),
        key=lambda candidate: candidate.hash,
    )
    policy = selected_campaign_policy("anchored-untuned", pair_budget=args.pair_budget)
    sorted_weights = sorted(workload.weights.values())
    high_weight_threshold = sorted_weights[int(0.75 * (len(sorted_weights) - 1))]
    high_weight_shape_ids = {
        shape_id for shape_id, weight in workload.weights.items() if weight >= high_weight_threshold
    }

    rows = []
    with tempfile.TemporaryDirectory(prefix="evotensile-workload-replay-") as directory:
        for seed_index in range(args.seeds):
            seed = 20260712 + seed_index
            candidates = _stable_candidate_order(catalog, seed=seed)
            for mode, active_weights in (("uniform", uniform), ("workload", workload)):
                evaluator, controller, observations, references = _anchored_state(
                    Path(directory) / f"{mode}-{seed}.sqlite",
                    oracle=oracle,
                    shapes=shapes,
                    baseline_pairs=baseline_pairs,
                    baseline_label=f"anchored-untuned:{mode}:{seed}",
                )
                controller.set_workload(active_weights)
                before_pairs = set(controller.queried_pairs)
                increment = _run_increment(
                    evaluator,
                    controller,
                    observations,
                    candidates=candidates,
                    shapes=shapes,
                    configuration=policy,
                    pair_budget=args.pair_budget,
                    estimators=args.estimators,
                    seed=seed,
                    allow_repair=True,
                    reference_performance=references,
                )
                added_pair_keys = set(controller.queried_pairs) - before_pairs
                high_weight_pairs = sum(shape_id in high_weight_shape_ids for shape_id, _ in added_pair_keys)
                rows.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "increment": increment,
                        "added_pairs": len(added_pair_keys),
                        "unknown_pairs": len(controller.unknown_pairs),
                        "prepared_candidates": len(controller.prepared_artifact_shapes),
                        "simulated_time_s": sum(controller.phase_time_s.values()),
                        "high_weight_pairs": high_weight_pairs,
                        "high_weight_pair_fraction": high_weight_pairs / max(len(added_pair_keys), 1),
                        "metrics": _metrics(
                            controller,
                            oracle_best=oracle_best,
                            workload=workload,
                        ),
                    }
                )
    aggregate = {mode: _aggregate(rows, mode) for mode in ("uniform", "workload")}
    output = {
        "source_db": str(args.db),
        "baseline_db": str(args.baseline_db),
        "initialization": "anchored-untuned",
        "policy": policy.to_dict(),
        "pair_budget": args.pair_budget,
        "predicted_cost_cap_s": policy.acquisition.max_predicted_cost_s,
        "seeds": args.seeds,
        "workload": workload.to_dict(),
        "workload_definition": "one call per shape weighted by exact untuned baseline latency",
        "high_weight_threshold": high_weight_threshold,
        "high_weight_shape_ids": sorted(high_weight_shape_ids),
        "rows": rows,
        "aggregate": aggregate,
        "delta_workload_minus_uniform": {
            key: aggregate["workload"][key] - aggregate["uniform"][key]
            for key in aggregate["uniform"]
            if key != "trials"
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
