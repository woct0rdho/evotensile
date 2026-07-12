#!/usr/bin/env python3

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from evotensile.profile import PROFILES, get_profile
from evotensile.search.replay import (
    ReplayCostModel,
    load_csv_oracle,
    load_db_oracle,
    load_hot_summary,
    merge_oracle_records,
    simulate_candidate_stream,
)
from evotensile.search.surrogate import DEFAULT_SURROGATE_MIN_EVIDENCE
from evotensile.shapes import parse_shape


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay blind candidate streams against exact historical measurements")
    parser.add_argument("--shape", default="8192,8192,1,8192")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None)
    parser.add_argument("--oracle-db", action="append", default=[])
    parser.add_argument("--oracle-csv", action="append", default=[])
    parser.add_argument("--stream-db", action="append", default=[])
    parser.add_argument("--stream-csv", action="append", default=[])
    parser.add_argument("--hot-summary", action="append", default=[])
    parser.add_argument("--protocol-hash", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--time-budget", type=float, default=1200.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pool-window", type=int, default=128)
    parser.add_argument(
        "--surrogate-min-evidence",
        type=int,
        default=DEFAULT_SURROGATE_MIN_EVIDENCE,
    )
    parser.add_argument("--prepare-workers", type=int, default=4)
    parser.add_argument("--prepare-seconds-per-candidate", type=float, default=8.0)
    parser.add_argument("--hot-reserve", type=float, default=60.0)
    parser.add_argument("--target-hot-gflops", type=float, default=None)
    parser.add_argument("--covering-cold-start", action="store_true")
    parser.add_argument("--island-count", type=int, default=1)
    parser.add_argument("--island-isolation-rounds", type=int, default=0)
    parser.add_argument("--leader-stabilization", action="store_true")
    parser.add_argument("--early-stop-on-convergence", action="store_true")
    parser.add_argument(
        "--diagnostic-pool",
        action="store_true",
        help="Acknowledge that directed/control candidates appear in the visible stream; results are not blind proof",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    shape = parse_shape(args.shape)
    profile = get_profile(args.profile)
    oracle_groups = [
        load_db_oracle(path, shape=shape, benchmark_protocol_hash=args.protocol_hash) for path in args.oracle_db
    ]
    oracle_groups.extend(
        load_csv_oracle(path, order_offset=1e12 * (index + 1)) for index, path in enumerate(args.oracle_csv)
    )
    hot = {}
    for path in args.hot_summary:
        hot.update(load_hot_summary(path))
    oracle = merge_oracle_records(oracle_groups, hot_measurements=hot)

    stream_records = [
        record
        for path in args.stream_db
        for record in load_db_oracle(path, shape=shape, benchmark_protocol_hash=args.protocol_hash)
    ]
    for index, path in enumerate(args.stream_csv):
        stream_records.extend(load_csv_oracle(path, order_offset=1e12 * (index + 1)))
    if args.stream_csv and not args.diagnostic_pool:
        raise SystemExit(
            "CSV candidate streams require --diagnostic-pool because they may contain directed/control candidates"
        )
    stream_records.sort(key=lambda record: (record.order, record.candidate.hash))
    stream = list({record.candidate.hash: record.candidate for record in stream_records}.values())
    if not stream:
        raise SystemExit("candidate stream is empty")

    cost = ReplayCostModel(
        time_budget_s=args.time_budget,
        prepare_workers=args.prepare_workers,
        prepare_seconds_per_candidate=args.prepare_seconds_per_candidate,
        hot_reserve_s=args.hot_reserve,
    )
    seeds = args.seed or [20260710]
    results = [
        simulate_candidate_stream(
            stream,
            oracle=oracle,
            shape=shape,
            profile=profile,
            cost=cost,
            seed=seed,
            batch_size=args.batch_size,
            pool_window=args.pool_window,
            surrogate_min_evidence=args.surrogate_min_evidence,
            target_hot_gflops=args.target_hot_gflops,
            covering_cold_start=args.covering_cold_start,
            island_count=args.island_count,
            island_isolation_rounds=args.island_isolation_rounds,
            leader_stabilization=args.leader_stabilization,
            early_stop_on_convergence=args.early_stop_on_convergence,
        )
        for seed in seeds
    ]
    payload = {
        "shape": shape.id,
        "proof_eligible": not args.diagnostic_pool,
        "oracle_candidates": len(oracle),
        "stream_candidates": len(stream),
        "cost_model": {
            **asdict(cost),
            "screening_launches": cost.screening_launches,
            "hot_launches": cost.hot_launches,
        },
        "results": [result.summary() for result in results],
        "successes": sum(result.reached_target for result in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
