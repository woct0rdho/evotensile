#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from evotensile.adaptive_retime import AdaptivePolicy, decide_retime_by_shape, load_timing_stats, winner_by_shape
from evotensile.database import EvoTensileDB
from evotensile.profile import PROFILES, get_profile


def _quantiles(values: list[float], probabilities: list[float]) -> dict[float, float | None]:
    ordered = sorted(values)
    if not ordered:
        return {probability: None for probability in probabilities}
    out = {}
    for probability in probabilities:
        position = (len(ordered) - 1) * probability
        lo = math.floor(position)
        hi = math.ceil(position)
        if lo == hi:
            out[probability] = ordered[lo]
        else:
            out[probability] = ordered[lo] * (hi - position) + ordered[hi] * (position - lo)
    return out


def _protocol_hashes_for_db(db_path: str | Path) -> list[str]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT DISTINCT benchmark_protocol_hash
            FROM evaluations
            ORDER BY benchmark_protocol_hash
            """
        ).fetchall()
    finally:
        con.close()
    return [str(row[0]) for row in rows]


def _load_stats_and_times(
    db: EvoTensileDB,
    *,
    db_path: str | Path,
    problem_type_hash: str,
    protocol_hashes: list[str],
):
    stats_by_shape = load_timing_stats(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hashes=protocol_hashes,
        min_samples=1,
    )
    clauses = ["problem_type_hash = ?", "status = 'ok'", "time_us IS NOT NULL"]
    params: list[str] = [problem_type_hash]
    if protocol_hashes:
        clauses.append(f"benchmark_protocol_hash IN ({','.join('?' for _ in protocol_hashes)})")
        params.extend(protocol_hashes)
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            f"""
            SELECT shape_id, candidate_hash, time_us
            FROM evaluations
            WHERE {" AND ".join(clauses)}
            ORDER BY eval_id
            """,
            params,
        ).fetchall()
    finally:
        con.close()
    times_by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for shape_id, candidate_hash, time_us in rows:
        times_by_pair[(str(shape_id), str(candidate_hash))].append(float(time_us))
    return stats_by_shape, times_by_pair


def _bootstrap_winner_probs(
    times_by_pair: dict[tuple[str, str], list[float]],
    shapes: list[str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    probs = {}
    for shape_id in shapes:
        candidate_times = {
            candidate_hash: samples
            for (pair_shape_id, candidate_hash), samples in times_by_pair.items()
            if pair_shape_id == shape_id and samples
        }
        if not candidate_times:
            continue
        counts: Counter[str] = Counter()
        for _ in range(iterations):
            best_hash = None
            best_median = None
            for candidate_hash, samples in candidate_times.items():
                resampled = [samples[rng.randrange(len(samples))] for _ in range(len(samples))]
                median = statistics.median(resampled)
                if (
                    best_median is None
                    or median < best_median
                    or (median == best_median and best_hash is not None and candidate_hash < best_hash)
                ):
                    best_hash = candidate_hash
                    best_median = median
            if best_hash is not None:
                counts[best_hash] += 1
        probs[shape_id] = {candidate_hash: count / iterations for candidate_hash, count in counts.items()}
    return probs


def _selected_probabilities(winners, probabilities: dict[str, dict[str, float]]) -> list[float]:
    selected = []
    for shape_id, winner in winners.items():
        probability = probabilities.get(shape_id, {}).get(winner.candidate_hash)
        if probability is not None:
            selected.append(probability)
    return selected


def _probability_summary(values: list[float]) -> dict[str, float | int | None]:
    quantiles = _quantiles(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "n": len(values),
        "q10": quantiles[0.10],
        "q25": quantiles[0.25],
        "median": quantiles[0.50],
        "q75": quantiles[0.75],
        "q90": quantiles[0.90],
        "below_0.5": sum(value < 0.5 for value in values),
        "below_0.8": sum(value < 0.8 for value in values),
        "below_0.9": sum(value < 0.9 for value in values),
        "below_0.95": sum(value < 0.95 for value in values),
    }


def _gap_values(stats_by_shape) -> list[float]:
    values = []
    for stats in stats_by_shape.values():
        ranked = sorted(stats, key=lambda item: (item.score_log_time, item.candidate_hash))
        if len(ranked) >= 2:
            values.append((math.exp(ranked[1].score_log_time - ranked[0].score_log_time) - 1.0) * 100.0)
    return values


def _gap_summary(stats_by_shape) -> dict[str, object]:
    values = _gap_values(stats_by_shape)
    return {
        "quantiles": {
            str(key): value for key, value in _quantiles(values, [0, 0.10, 0.25, 0.50, 0.75, 0.90, 1]).items()
        },
        "under": {str(threshold): sum(value < threshold for value in values) for threshold in [0.5, 1, 2, 5]},
    }


def _winner_changes(left, right) -> list[int]:
    common = sorted(set(left) & set(right))
    changed = sum(left[shape_id].candidate_hash != right[shape_id].candidate_hash for shape_id in common)
    return [changed, len(common)]


def _runtime_summary(db_path: str | Path) -> dict[str, float | int | None]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT metadata_json
            FROM runs
            WHERE status = 'ok'
            """
        ).fetchall()
    finally:
        con.close()
    build = []
    runner = []
    for (metadata_json,) in rows:
        metadata = json.loads(metadata_json)
        duration = metadata.get("duration_s")
        if duration is None:
            continue
        if "runnable_pairs" in metadata:
            runner.append(float(duration))
        else:
            build.append(float(duration))
    return {
        "build_runs": len(build),
        "runner_runs": len(runner),
        "adaptive_build_sum": sum(build),
        "adaptive_runner_sum": sum(runner),
        "adaptive_build_median": statistics.median(build) if build else None,
        "adaptive_runner_median": statistics.median(runner) if runner else None,
        "adaptive_wall_proxy_sum": sum(build) + sum(runner),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare first-pass, old retime, and adaptive retime reliability")
    parser.add_argument("--db", default=None, help="Use one DB for all inputs")
    parser.add_argument("--first-db", default=None)
    parser.add_argument("--old-retime-db", default=None)
    parser.add_argument("--adaptive-db", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None)
    parser.add_argument("--first-protocol-hash", action="append", default=[])
    parser.add_argument("--old-retime-protocol-hash", action="append", default=[])
    parser.add_argument("--adaptive-protocol-hash", action="append", default=[])
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--epsilon-pct", type=float, default=2.0)
    parser.add_argument("--confidence", type=float, default=0.90)
    args = parser.parse_args()

    profile = get_profile(args.profile)
    first_db_path = args.first_db or args.db
    old_db_path = args.old_retime_db or args.db
    adaptive_db_path = args.adaptive_db or args.db
    if first_db_path is None or old_db_path is None or adaptive_db_path is None:
        raise SystemExit("provide --db or all of --first-db, --old-retime-db, and --adaptive-db")
    first_db = EvoTensileDB.connect(first_db_path)
    old_db = EvoTensileDB.connect(old_db_path)
    adaptive_db = EvoTensileDB.connect(adaptive_db_path)
    default_protocol_hash = profile.benchmark_protocol_hash()
    first_protocol_hashes = args.first_protocol_hash or [default_protocol_hash]
    old_protocol_hashes = args.old_retime_protocol_hash or [default_protocol_hash]
    adaptive_protocol_hashes = args.adaptive_protocol_hash or _protocol_hashes_for_db(adaptive_db_path)

    first_stats, first_times = _load_stats_and_times(
        first_db,
        db_path=first_db_path,
        problem_type_hash=profile.problem_type_hash,
        protocol_hashes=first_protocol_hashes,
    )
    old_stats, old_times = _load_stats_and_times(
        old_db,
        db_path=old_db_path,
        problem_type_hash=profile.problem_type_hash,
        protocol_hashes=old_protocol_hashes,
    )
    adaptive_stats, adaptive_times = _load_stats_and_times(
        adaptive_db,
        db_path=adaptive_db_path,
        problem_type_hash=profile.problem_type_hash,
        protocol_hashes=adaptive_protocol_hashes,
    )

    first_winners = winner_by_shape(first_stats)
    old_winners = winner_by_shape(old_stats)
    adaptive_winners = {**first_winners, **winner_by_shape(adaptive_stats)}
    all_shapes = sorted(first_winners)

    source_rank_by_shape = {}
    for shape_id, stats in first_stats.items():
        ranked = sorted(stats, key=lambda item: (item.score_log_time, item.candidate_hash))
        source_rank_by_shape[shape_id] = {item.candidate_hash: index + 1 for index, item in enumerate(ranked)}

    policy = AdaptivePolicy(epsilon_pct=args.epsilon_pct, confidence=args.confidence)
    status_counts = {
        "first_pass_policy": dict(
            Counter(decision.status for decision in decide_retime_by_shape(first_stats, policy=policy).values())
        ),
        "old_top4_policy_on_old_retime": dict(
            Counter(decision.status for decision in decide_retime_by_shape(old_stats, policy=policy).values())
        ),
        "adaptive_policy_on_adaptive_retime": dict(
            Counter(decision.status for decision in decide_retime_by_shape(adaptive_stats, policy=policy).values())
        ),
    }

    first_probs = _bootstrap_winner_probs(first_times, all_shapes, iterations=args.bootstrap_iterations, seed=1)
    old_probs = _bootstrap_winner_probs(old_times, sorted(old_winners), iterations=args.bootstrap_iterations, seed=2)
    adaptive_probs = _bootstrap_winner_probs(
        adaptive_times, sorted(winner_by_shape(adaptive_stats)), iterations=args.bootstrap_iterations, seed=3
    )

    adaptive_rank_counts = Counter(
        source_rank_by_shape[shape_id].get(winner.candidate_hash, "missing")
        for shape_id, winner in adaptive_winners.items()
    )
    old_rank_counts = Counter(
        source_rank_by_shape[shape_id].get(winner.candidate_hash, "missing") for shape_id, winner in old_winners.items()
    )
    adaptive_rank_gt4_shapes = [
        shape_id
        for shape_id, winner in adaptive_winners.items()
        if isinstance(source_rank_by_shape[shape_id].get(winner.candidate_hash), int)
        and source_rank_by_shape[shape_id][winner.candidate_hash] > 4
    ]
    adaptive_rank_gt8_shapes = [
        shape_id
        for shape_id, winner in adaptive_winners.items()
        if isinstance(source_rank_by_shape[shape_id].get(winner.candidate_hash), int)
        and source_rank_by_shape[shape_id][winner.candidate_hash] > 8
    ]
    adaptive_sample_counts = Counter(len(samples) for samples in adaptive_times.values())

    summary = {
        "dbs": {
            "first_pass": str(first_db_path),
            "old_top4_retime": str(old_db_path),
            "adaptive_top8": str(adaptive_db_path),
        },
        "protocols": {
            "first_pass": first_protocol_hashes,
            "old_top4": old_protocol_hashes,
            "adaptive": adaptive_protocol_hashes,
        },
        "policy": {"epsilon_pct": args.epsilon_pct, "confidence": args.confidence},
        "status_counts": status_counts,
        "candidate_pair_counts": {
            "first_pass": len(first_times),
            "old_top4_retime": len(old_times),
            "adaptive_retime": len(adaptive_times),
        },
        "sample_counts": {
            "first_pass_ok_samples": sum(len(samples) for samples in first_times.values()),
            "old_top4_ok_samples": sum(len(samples) for samples in old_times.values()),
            "adaptive_ok_samples": sum(len(samples) for samples in adaptive_times.values()),
            "adaptive_pair_sample_count_distribution": dict(sorted(adaptive_sample_counts.items())),
        },
        "runtime_seconds": _runtime_summary(adaptive_db_path),
        "winner_changes": {
            "old_top4_vs_first_pass": _winner_changes(first_winners, old_winners),
            "adaptive_vs_first_pass": _winner_changes(first_winners, adaptive_winners),
            "adaptive_vs_old_top4": _winner_changes(old_winners, adaptive_winners),
        },
        "source_rank_counts": {
            "old_top4_winners": dict(
                sorted(old_rank_counts.items(), key=lambda item: item[0] if isinstance(item[0], int) else 999)
            ),
            "adaptive_winners": dict(
                sorted(adaptive_rank_counts.items(), key=lambda item: item[0] if isinstance(item[0], int) else 999)
            ),
            "adaptive_winners_rank_gt4_count": len(adaptive_rank_gt4_shapes),
            "adaptive_winners_rank_gt4_shapes": adaptive_rank_gt4_shapes,
            "adaptive_winners_rank_gt8_count": len(adaptive_rank_gt8_shapes),
            "adaptive_winners_rank_gt8_shapes": adaptive_rank_gt8_shapes,
        },
        "bootstrap_selected_winner_probability": {
            "first_pass": _probability_summary(_selected_probabilities(first_winners, first_probs)),
            "old_top4_retime": _probability_summary(_selected_probabilities(old_winners, old_probs)),
            "adaptive_retime_only": _probability_summary(
                _selected_probabilities(winner_by_shape(adaptive_stats), adaptive_probs)
            ),
        },
        "top1_top2_gap_pct": {
            "first_pass": _gap_summary(first_stats),
            "old_top4_retime": _gap_summary(old_stats),
            "adaptive_retime": _gap_summary(adaptive_stats),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (output_dir / "winner_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "shape_id",
                "first_winner",
                "old_top4_winner",
                "adaptive_winner",
                "adaptive_first_rank",
                "old_first_rank",
                "adaptive_vs_first_changed",
                "adaptive_vs_old_changed",
                "adaptive_rank_gt4",
                "adaptive_rank_gt8",
                "adaptive_bootstrap_p",
                "old_bootstrap_p",
                "first_bootstrap_p",
                "adaptive_samples_for_winner",
            ],
        )
        writer.writeheader()
        for shape_id in all_shapes:
            first_winner = first_winners[shape_id].candidate_hash
            old_winner_stats = old_winners.get(shape_id)
            old_winner = old_winner_stats.candidate_hash if old_winner_stats is not None else ""
            adaptive_winner = adaptive_winners[shape_id].candidate_hash
            adaptive_rank = source_rank_by_shape[shape_id].get(adaptive_winner)
            old_rank = source_rank_by_shape[shape_id].get(old_winner) if old_winner else ""
            writer.writerow(
                {
                    "shape_id": shape_id,
                    "first_winner": first_winner,
                    "old_top4_winner": old_winner,
                    "adaptive_winner": adaptive_winner,
                    "adaptive_first_rank": adaptive_rank or "",
                    "old_first_rank": old_rank or "",
                    "adaptive_vs_first_changed": int(adaptive_winner != first_winner),
                    "adaptive_vs_old_changed": int(bool(old_winner) and adaptive_winner != old_winner),
                    "adaptive_rank_gt4": int(isinstance(adaptive_rank, int) and adaptive_rank > 4),
                    "adaptive_rank_gt8": int(isinstance(adaptive_rank, int) and adaptive_rank > 8),
                    "adaptive_bootstrap_p": adaptive_probs.get(shape_id, {}).get(adaptive_winner, ""),
                    "old_bootstrap_p": old_probs.get(shape_id, {}).get(old_winner, "") if old_winner else "",
                    "first_bootstrap_p": first_probs.get(shape_id, {}).get(first_winner, ""),
                    "adaptive_samples_for_winner": len(adaptive_times.get((shape_id, adaptive_winner), [])),
                }
            )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
