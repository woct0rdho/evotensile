#!/usr/bin/env python3

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from evotensile.adaptive_retime import (
    AdaptivePolicy,
    candidate_map,
    decide_retime_by_shape,
    load_timing_stats,
    write_decisions_csv,
    write_pair_decisions_csv,
    write_timing_stats_csv,
)
from evotensile.database import EvoTensileDB
from evotensile.profile import PROFILES, get_profile
from evotensile.protocol import BenchmarkProtocol
from evotensile.runner import DEFAULT_TENSILELITE_BIN
from evotensile.scheduler import execute_schedule
from evotensile.shapes import shape_from_id


def _parse_shape_ids(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {shape_from_id(value).id if not value.startswith("m") else value for value in values}


def _group_retime_jobs(decisions):
    grouped_shape_ids: dict[tuple[int, tuple[str, ...]], list[str]] = defaultdict(list)
    rank_order_by_key: dict[tuple[int, tuple[str, ...]], dict[str, int]] = {}
    for decision in decisions:
        if not decision.needs_retime:
            continue
        candidate_hashes = tuple(decision.retime_candidate_hashes)
        key = (decision.target_samples, tuple(sorted(candidate_hashes)))
        grouped_shape_ids[key].append(decision.shape_id)
        ranks = rank_order_by_key.setdefault(key, {})
        for rank, candidate_hash in enumerate(candidate_hashes):
            ranks.setdefault(candidate_hash, rank)

    groups = []
    for (target_samples, candidate_hashes), shape_ids in sorted(
        grouped_shape_ids.items(), key=lambda item: (item[0][0], len(item[1]), item[0][1]), reverse=True
    ):
        ranks = rank_order_by_key[(target_samples, candidate_hashes)]
        ordered_hashes = sorted(candidate_hashes, key=lambda candidate_hash: (ranks[candidate_hash], candidate_hash))
        groups.append((target_samples, sorted(shape_ids), ordered_hashes))
    return groups


def _policy_from_args(args: argparse.Namespace) -> AdaptivePolicy:
    return AdaptivePolicy(
        epsilon_pct=args.epsilon_pct,
        confidence=args.confidence,
        min_retime_samples=args.min_retime_samples,
        max_retime_samples=args.max_retime_samples,
        sample_step=args.sample_step,
        max_k=args.max_k,
        min_effect_pct=args.min_effect_pct,
    )


def _protocol_from_args(args: argparse.Namespace, profile, *, num_benchmarks: int) -> BenchmarkProtocol:
    return profile.default_protocol.with_overrides(
        num_warmups=args.num_warmups,
        num_benchmarks=num_benchmarks,
        enqueues_per_sync=args.enqueues_per_sync,
        syncs_per_benchmark=args.syncs_per_benchmark,
        num_elements_to_validate=args.num_elements_to_validate,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive statistical retime of unresolved EvoTensile candidates")
    parser.add_argument("--db", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None)
    parser.add_argument("--source-protocol-hash", action="append", default=[])
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--shape-id", action="append", default=[])
    parser.add_argument("--limit-shapes", type=int, default=None)
    parser.add_argument("--epsilon-pct", type=float, default=2.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--min-retime-samples", type=int, default=20)
    parser.add_argument("--max-retime-samples", type=int, default=80)
    parser.add_argument("--sample-step", type=int, default=10)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--min-effect-pct", type=float, default=0.5)
    parser.add_argument("--num-warmups", type=int, default=None)
    parser.add_argument("--enqueues-per-sync", type=int, default=None)
    parser.add_argument("--syncs-per-benchmark", type=int, default=None)
    parser.add_argument("--num-elements-to-validate", type=int, default=None)
    parser.add_argument("--tensilelite-bin", default=DEFAULT_TENSILELITE_BIN)
    parser.add_argument("--compile-threads", type=int, default=4)
    parser.add_argument(
        "--runner-bin", default=None, help="Structured runner executable; defaults to the target profile"
    )
    parser.add_argument("--build-timeout", type=float, default=None)
    parser.add_argument("--runner-timeout", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    if args.min_samples <= 0:
        raise SystemExit("--min-samples must be positive")
    if args.min_retime_samples <= 0 or args.max_retime_samples < args.min_retime_samples:
        raise SystemExit("invalid retime sample bounds")
    if args.max_k < 2:
        raise SystemExit("--max-k must be at least 2")
    if not 0.0 < args.confidence < 1.0:
        raise SystemExit("--confidence must be between 0 and 1")

    profile = get_profile(args.profile)
    source_protocol_hashes = args.source_protocol_hash or [profile.benchmark_protocol_hash()]
    policy = _policy_from_args(args)
    runner_bin = args.runner_bin or profile.default_runner_bin
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db = EvoTensileDB.connect(args.db)
    db.init()
    shape_filter = _parse_shape_ids(args.shape_id)
    stats_by_shape = load_timing_stats(
        db,
        problem_type_hash=profile.problem_type_hash,
        benchmark_protocol_hashes=source_protocol_hashes,
        min_samples=args.min_samples,
        shape_ids=shape_filter,
    )
    decisions = list(decide_retime_by_shape(stats_by_shape, policy=policy).values())
    if args.limit_shapes is not None:
        decisions = decisions[: args.limit_shapes]
    if not decisions:
        raise SystemExit("no source timing statistics found for requested filters")

    write_timing_stats_csv(output_dir / "source_timing_stats.csv", stats_by_shape)
    write_decisions_csv(output_dir / "retime_decisions.csv", decisions)
    write_pair_decisions_csv(output_dir / "retime_pair_decisions.csv", decisions)

    groups = _group_retime_jobs(decisions)
    candidate_hashes = [candidate_hash for _, _, hashes in groups for candidate_hash in hashes]
    candidates_by_hash = candidate_map(db, candidate_hashes) if candidate_hashes else {}
    status_counts = Counter(decision.status for decision in decisions)
    k_counts = Counter(len(decision.retime_candidate_hashes) for decision in decisions if decision.needs_retime)
    sample_counts = Counter(decision.target_samples for decision in decisions if decision.needs_retime)
    planned_pairs = sum(len(candidate_hashes) * len(shape_ids) for _, shape_ids, candidate_hashes in groups)
    planned_samples = sum(
        target_samples * len(candidate_hashes) * len(shape_ids)
        for target_samples, shape_ids, candidate_hashes in groups
    )
    summary = {
        "db": args.db,
        "output_dir": str(output_dir),
        "profile": profile.name,
        "source_problem_type_hash": profile.problem_type_hash,
        "source_benchmark_protocol_hashes": source_protocol_hashes,
        "policy": asdict(policy),
        "shape_count": len(decisions),
        "status_counts": dict(sorted(status_counts.items())),
        "retime_shape_count": sum(decision.needs_retime for decision in decisions),
        "retime_pair_count": planned_pairs,
        "retime_sample_count": planned_samples,
        "retime_k_counts": {str(key): value for key, value in sorted(k_counts.items())},
        "retime_sample_counts": {str(key): value for key, value in sorted(sample_counts.items())},
        "group_count": len(groups),
        "runner_bin": runner_bin,
        "build_timeout_s": args.build_timeout,
        "runner_timeout_s": args.runner_timeout,
        "dry_run": args.dry_run,
        "generate_only": args.generate_only,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    executed_groups = []
    if not args.dry_run:
        for group_index, (target_samples, shape_ids, hashes) in enumerate(groups):
            protocol = _protocol_from_args(args, profile, num_benchmarks=target_samples)
            shapes = [shape_from_id(shape_id) for shape_id in shape_ids]
            candidates = [candidates_by_hash[candidate_hash] for candidate_hash in hashes]
            group_dir = output_dir / f"group_{group_index:04d}_n{target_samples}"
            result = execute_schedule(
                db,
                shapes=shapes,
                candidates=candidates,
                output_root=group_dir,
                target_profile=profile,
                protocol=protocol,
                min_samples=target_samples,
                candidate_batch_size=max(1, len(candidates)),
                shape_batch_size=max(1, len(shapes)),
                dry_run=False,
                generate_only=args.generate_only,
                tensilelite_bin=args.tensilelite_bin,
                compile_threads=args.compile_threads,
                keep_going=args.keep_going,
                runner_bin=runner_bin,
                build_timeout_s=args.build_timeout,
                runner_timeout_s=args.runner_timeout,
            )
            inserted = sum(batch.ingest.inserted for batch in result.executed_batches if batch.ingest is not None)
            rejected = sum(batch.ingest.rejected for batch in result.executed_batches if batch.ingest is not None)
            unmapped = sum(batch.ingest.unmapped for batch in result.executed_batches if batch.ingest is not None)
            status_counts = Counter()
            for batch in result.executed_batches:
                if batch.ingest is None:
                    continue
                status_counts.update(batch.ingest.status_counts)
            group_summary = {
                "group_index": group_index,
                "target_samples": target_samples,
                "shape_count": len(shapes),
                "candidate_count": len(candidates),
                "planned_batches": len(result.planned_batches),
                "executed_batches": len(result.executed_batches),
                "planned_pairs": result.missing_pairs,
                "nominal_pairs": result.nominal_pairs,
                "planned_samples": sum(batch.missing_samples for batch in result.planned_batches),
                "nominal_samples": sum(batch.nominal_samples for batch in result.planned_batches),
                "inserted": inserted,
                "rejected": rejected,
                "unmapped": unmapped,
                "status_counts": dict(sorted(status_counts.items())),
                "benchmark_protocol_hash": profile.benchmark_protocol_hash(protocol),
            }
            executed_groups.append(group_summary)
            print(json.dumps(group_summary, sort_keys=True))

    metadata = {**summary, "executed_groups": executed_groups}
    (output_dir / "adaptive_retime_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
