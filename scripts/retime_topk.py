#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evotensile.candidate import Candidate
from evotensile.database import EvaluationSummary, EvoTensileDB
from evotensile.profile import PROFILES, TargetProfile, get_profile
from evotensile.protocol import BenchmarkProtocol
from evotensile.runner import DEFAULT_TENSILELITE_BIN
from evotensile.scheduler import execute_schedule
from evotensile.shapes import Shape, shape_from_id


def _collect_topk(
    db: EvoTensileDB,
    *,
    problem_hash: str,
    protocol_hash: str,
    top_k: int,
    min_samples: int,
    shape_ids: set[str] | None,
) -> dict[str, list[EvaluationSummary]]:
    by_shape: dict[str, list[EvaluationSummary]] = defaultdict(list)
    summaries = db.rank_evaluations(
        problem_type_hash=problem_hash,
        benchmark_protocol_hash=protocol_hash,
        min_samples=min_samples,
    )
    for summary in summaries:
        if shape_ids is not None and summary.shape_id not in shape_ids:
            continue
        if len(by_shape[summary.shape_id]) < top_k:
            by_shape[summary.shape_id].append(summary)
    return dict(sorted(by_shape.items()))


def _candidate_map(db: EvoTensileDB, hashes: list[str]) -> dict[str, Candidate]:
    candidates = db.get_candidates(list(dict.fromkeys(hashes)))
    out = {candidate.hash: candidate for candidate in candidates}
    missing = sorted(set(hashes) - set(out))
    if missing:
        raise SystemExit(f"missing candidate records in DB: {', '.join(missing[:8])}")
    return out


def _groups(
    topk_by_shape: dict[str, list[EvaluationSummary]],
    candidates_by_hash: dict[str, Candidate],
) -> list[tuple[list[Shape], list[Candidate]]]:
    grouped_shape_ids: dict[tuple[str, ...], list[str]] = defaultdict(list)
    rank_order_by_key: dict[tuple[str, ...], dict[str, int]] = {}
    for shape_id, summaries in topk_by_shape.items():
        ranked_hashes = [summary.candidate_hash for summary in summaries]
        key = tuple(sorted(ranked_hashes))
        grouped_shape_ids[key].append(shape_id)
        ranks = rank_order_by_key.setdefault(key, {})
        for rank, candidate_hash in enumerate(ranked_hashes):
            ranks.setdefault(candidate_hash, rank)

    groups: list[tuple[list[Shape], list[Candidate]]] = []
    for candidate_hashes, shape_ids in sorted(
        grouped_shape_ids.items(), key=lambda item: (len(item[1]), item[0]), reverse=True
    ):
        shapes = [shape_from_id(shape_id) for shape_id in sorted(shape_ids)]
        ranks = rank_order_by_key[candidate_hashes]
        ordered_hashes = sorted(candidate_hashes, key=lambda candidate_hash: (ranks[candidate_hash], candidate_hash))
        candidates = [candidates_by_hash[candidate_hash] for candidate_hash in ordered_hashes]
        groups.append((shapes, candidates))
    return groups


def _parse_shape_ids(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {shape_from_id(value).id if not value.startswith("m") else value for value in values}


def _protocol_from_args(args: argparse.Namespace, profile: TargetProfile) -> BenchmarkProtocol:
    return profile.default_protocol.with_overrides(
        num_warmups=args.num_warmups,
        num_benchmarks=args.num_benchmarks,
        enqueues_per_sync=args.enqueues_per_sync,
        syncs_per_benchmark=args.syncs_per_benchmark,
        num_elements_to_validate=args.num_elements_to_validate,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Retime top-K per-shape EvoTensile candidates exactly")
    parser.add_argument("--db", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--shape-id", action="append", default=[])
    parser.add_argument("--limit-shapes", type=int, default=None)
    parser.add_argument("--num-warmups", type=int, default=None)
    parser.add_argument("--num-benchmarks", type=int, default=None)
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

    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if args.min_samples <= 0:
        raise SystemExit("--min-samples must be positive")

    profile = get_profile(args.profile)
    source_protocol = profile.default_protocol
    target_protocol = _protocol_from_args(args, profile)
    runner_bin = args.runner_bin or profile.default_runner_bin
    db = EvoTensileDB.connect(args.db)
    db.init()
    source_problem_hash = profile.problem_type_hash
    source_protocol_hash = profile.benchmark_protocol_hash(source_protocol)
    target_problem_hash = profile.problem_type_hash
    target_protocol_hash = profile.benchmark_protocol_hash(target_protocol)

    shape_filter = _parse_shape_ids(args.shape_id)
    topk_by_shape = _collect_topk(
        db,
        problem_hash=source_problem_hash,
        protocol_hash=source_protocol_hash,
        top_k=args.top_k,
        min_samples=args.min_samples,
        shape_ids=shape_filter,
    )
    if args.limit_shapes is not None:
        topk_by_shape = dict(list(topk_by_shape.items())[: args.limit_shapes])
    if not topk_by_shape:
        raise SystemExit("no source winners found for requested filters")

    hashes = [summary.candidate_hash for summaries in topk_by_shape.values() for summary in summaries]
    candidates_by_hash = _candidate_map(db, hashes)
    groups = _groups(topk_by_shape, candidates_by_hash)
    intended_pairs = sum(len(summaries) for summaries in topk_by_shape.values())
    nominal_pairs = sum(len(shapes) * len(candidates) for shapes, candidates in groups)

    summary = {
        "db": args.db,
        "output_dir": args.output_dir,
        "profile": profile.name,
        "source_problem_type_hash": source_problem_hash,
        "source_benchmark_protocol_hash": source_protocol_hash,
        "target_problem_type_hash": target_problem_hash,
        "target_benchmark_protocol_hash": target_protocol_hash,
        "top_k": args.top_k,
        "min_samples": args.min_samples,
        "shape_count": len(topk_by_shape),
        "unique_candidate_count": len(set(hashes)),
        "group_count": len(groups),
        "intended_pairs": intended_pairs,
        "nominal_pairs": nominal_pairs,
        "target_protocol": target_protocol.global_parameters(),
        "runner_bin": runner_bin,
        "build_timeout_s": args.build_timeout,
        "runner_timeout_s": args.runner_timeout,
        "dry_run": args.dry_run,
        "generate_only": args.generate_only,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    executed_groups = []
    if not args.dry_run:
        output_root = Path(args.output_dir)
        for group_index, (shapes, candidates) in enumerate(groups):
            group_dir = output_root / f"group_{group_index:04d}"
            result = execute_schedule(
                db,
                shapes=shapes,
                candidates=candidates,
                output_root=group_dir,
                target_profile=profile,
                protocol=target_protocol,
                min_samples=args.min_samples,
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
            group_summary = {
                "group_index": group_index,
                "shape_count": len(shapes),
                "candidate_count": len(candidates),
                "planned_batches": len(result.planned_batches),
                "executed_batches": len(result.executed_batches),
                "inserted": inserted,
                "rejected": rejected,
                "unmapped": unmapped,
            }
            executed_groups.append(group_summary)
            print(json.dumps(group_summary, sort_keys=True))

    metadata = {**summary, "executed_groups": executed_groups}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retime_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
