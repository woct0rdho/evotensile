#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evotensile.cache import normalize_version_name, problem_type_hash
from evotensile.candidate import Candidate
from evotensile.database import EvaluationSummary, EvoTensileDB
from evotensile.runner import DEFAULT_TENSILELITE_BIN, serial_benchmark_protocol_hash
from evotensile.scheduler import execute_schedule
from evotensile.shapes import Shape, shape_from_id


def _collect_topk(
    db: EvoTensileDB,
    *,
    version_name: str,
    problem_hash: str,
    protocol_hash: str,
    top_k: int,
    min_samples: int,
    shape_ids: set[str] | None,
) -> dict[str, list[EvaluationSummary]]:
    by_shape: dict[str, list[EvaluationSummary]] = defaultdict(list)
    summaries = db.rank_evaluations(
        version_name=version_name,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Retime top-K per-shape EvoTensile candidates exactly")
    parser.add_argument("--db", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-version-name", required=True)
    parser.add_argument("--target-version-name", required=True)
    parser.add_argument("--source-problem-type-hash", default=None)
    parser.add_argument("--source-benchmark-protocol-hash", default=None)
    parser.add_argument("--target-problem-type-hash", default=None)
    parser.add_argument("--target-benchmark-protocol-hash", default=None)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--shape-id", action="append", default=[])
    parser.add_argument("--limit-shapes", type=int, default=None)
    parser.add_argument("--global-parameter", action="append", default=[])
    parser.add_argument("--tensilelite-bin", default=DEFAULT_TENSILELITE_BIN)
    parser.add_argument("--compile-threads", type=int, default=4)
    parser.add_argument("--benchmark-threads", type=int, default=1)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if args.min_samples <= 0:
        raise SystemExit("--min-samples must be positive")

    db = EvoTensileDB.connect(args.db)
    db.init()
    source_version = normalize_version_name(args.source_version_name)
    target_version = normalize_version_name(args.target_version_name)
    source_problem_hash = args.source_problem_type_hash or problem_type_hash()
    source_protocol_hash = serial_benchmark_protocol_hash(
        [], benchmark_protocol_hash=args.source_benchmark_protocol_hash
    )
    target_problem_hash = args.target_problem_type_hash or source_problem_hash
    target_protocol_hash = serial_benchmark_protocol_hash(
        args.global_parameter, benchmark_protocol_hash=args.target_benchmark_protocol_hash
    )

    shape_filter = _parse_shape_ids(args.shape_id)
    topk_by_shape = _collect_topk(
        db,
        version_name=source_version,
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
        "source_version_name": source_version,
        "target_version_name": target_version,
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
        "global_parameter": args.global_parameter,
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
                version_name=target_version,
                problem_type_hash=target_problem_hash,
                benchmark_protocol_hash=target_protocol_hash,
                min_samples=args.min_samples,
                candidate_batch_size=max(1, len(candidates)),
                shape_batch_size=max(1, len(shapes)),
                dry_run=False,
                generate_only=args.generate_only,
                tensilelite_bin=args.tensilelite_bin,
                compile_threads=args.compile_threads,
                benchmark_threads=args.benchmark_threads,
                global_parameters=args.global_parameter,
                extra_args=args.extra_arg,
                keep_going=args.keep_going,
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
