#!/usr/bin/env python3
import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import TypedDict

from evotensile.adaptive_retime import AdaptivePolicy, ProbePolicy
from evotensile.database import EvoTensileDB
from evotensile.profile import get_profile
from evotensile.protocol import BenchmarkProtocol
from evotensile.runner import DEFAULT_TENSILELITE_BIN
from evotensile.scheduler import execute_schedule, propose_candidates
from evotensile.search.hot_confirm import hot_confirm_topk
from evotensile.shapes import parse_shape


class CampaignPolicy(TypedDict):
    cold_candidates: int
    cold_pool_multiplier: int
    feedback_candidates: int
    feedback_random: int
    feedback_semantic_mutation: int
    feedback_de: int
    feedback_gomea: int
    feedback_pool_multiplier: int
    surrogate_min_evidence: int
    elite_count: int
    candidate_batch_size: int
    prepare_workers: int
    hot_reserve_s: float
    max_feedback_rounds_safety: int


class ProposalArgs(TypedDict):
    num_random: int
    elite_count: int
    local_count: int
    de_count: int
    gomea_count: int
    adaptive_operators: bool
    surrogate_pool_multiplier: int


def _proposal_args(policy: CampaignPolicy, *, cold: bool) -> ProposalArgs:
    if cold:
        return {
            "num_random": policy["cold_candidates"],
            "elite_count": 0,
            "local_count": 0,
            "de_count": 0,
            "gomea_count": 0,
            "adaptive_operators": False,
            "surrogate_pool_multiplier": policy["cold_pool_multiplier"],
        }
    return {
        "num_random": policy["feedback_random"],
        "elite_count": policy["elite_count"],
        "local_count": policy["feedback_semantic_mutation"],
        "de_count": policy["feedback_de"],
        "gomea_count": policy["feedback_gomea"],
        "adaptive_operators": True,
        "surrogate_pool_multiplier": policy["feedback_pool_multiplier"],
    }


FROZEN_POLICY: CampaignPolicy = {
    "cold_candidates": 48,
    "cold_pool_multiplier": 4,
    "feedback_candidates": 24,
    "feedback_random": 4,
    "feedback_semantic_mutation": 6,
    "feedback_de": 4,
    "feedback_gomea": 10,
    "feedback_pool_multiplier": 8,
    "surrogate_min_evidence": 24,
    "elite_count": 32,
    "candidate_batch_size": 8,
    "prepare_workers": 8,
    "hot_reserve_s": 60.0,
    "max_feedback_rounds_safety": 100,
}


def _round_summary(result) -> dict[str, object]:
    status_counts: Counter[str] = Counter()
    errors = []
    for batch in result.executed_batches:
        if batch.ingest is not None:
            status_counts.update(batch.ingest.status_counts)
            errors.extend(batch.ingest.errors)
    return {
        "planned_batches": len(result.planned_batches),
        "executed_batches": len(result.executed_batches),
        "missing_pairs": result.missing_pairs,
        "probe_survivor_pairs": result.probe_survivor_pairs,
        "probe_screened_pairs": result.probe_screened_pairs,
        "status_counts": dict(sorted(status_counts.items())),
        "errors": errors,
    }


def _leader(db: EvoTensileDB, *, shape_id: str, problem_type_hash: str, protocol_hash: str):
    rows = db.rank_evaluations(
        shape_id=shape_id,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=protocol_hash,
        min_samples=2,
        limit=1,
    )
    if not rows:
        return None
    row = rows[0]
    return {
        "candidate_hash": row.candidate_hash,
        "median_gflops": row.median_gflops,
        "samples": row.samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen blind one-shape 20-minute search policy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", default="8192,8192,1,8192")
    parser.add_argument("--profile", default="gfx1151-nt-hhs")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--time-budget", type=float, default=1200.0)
    parser.add_argument("--hot-reserve", type=float, default=FROZEN_POLICY["hot_reserve_s"])
    parser.add_argument(
        "--max-feedback-rounds",
        type=int,
        default=FROZEN_POLICY["max_feedback_rounds_safety"],
    )
    parser.add_argument("--runner-bin", type=Path, default=Path("build/evotensile-structured-runner"))
    parser.add_argument("--tensilelite-bin", type=Path, default=Path(DEFAULT_TENSILELITE_BIN))
    parser.add_argument("--build-timeout", type=float, default=300.0)
    parser.add_argument("--runner-timeout", type=float, default=300.0)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    profile = get_profile(args.profile)
    shape = parse_shape(args.shape)
    protocol = BenchmarkProtocol(
        num_warmups=1,
        num_benchmarks=2,
        enqueues_per_sync=1,
        syncs_per_benchmark=1,
    )
    protocol_hash = profile.benchmark_protocol_hash(protocol)
    db_path = args.output / "campaign.sqlite"
    db = EvoTensileDB.connect(db_path)
    db.init()
    db.register_shapes([shape])

    start = time.monotonic()
    hard_deadline = start + args.time_budget
    search_deadline = hard_deadline - args.hot_reserve
    policy: CampaignPolicy = {
        **FROZEN_POLICY,
        "hot_reserve_s": args.hot_reserve,
        "max_feedback_rounds_safety": args.max_feedback_rounds,
    }
    record = {
        "blind": True,
        "seed": args.seed,
        "shape": shape.id,
        "profile": profile.name,
        "problem_type_hash": profile.problem_type_hash,
        "screening_protocol_hash": protocol_hash,
        "time_budget_s": args.time_budget,
        "policy": policy,
        "rounds": [],
    }
    (args.output / "frozen_policy.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    adaptive_policy = AdaptivePolicy()
    probe_policy = ProbePolicy()
    compile_cache = args.output / "compile_cache"
    round_index = 0
    while round_index <= args.max_feedback_rounds:
        now = time.monotonic()
        if now >= search_deadline:
            break
        if round_index > 0:
            recent_durations = [float(item["duration_s"]) for item in record["rounds"][-5:]]
            next_round_guard_s = max(30.0, max(recent_durations, default=0.0) + 10.0)
            if search_deadline - now < next_round_guard_s:
                break
        proposal_args = _proposal_args(policy, cold=round_index == 0)
        round_seed = args.seed + round_index * 10007
        candidates = propose_candidates(
            db,
            proposal="family-qd",
            seed=round_seed,
            problem_type_hash=profile.problem_type_hash,
            benchmark_protocol_hash=protocol_hash,
            shape_id=shape.id,
            target_shapes=[shape],
            transfer_shape_count=0,
            transfer_per_shape=0,
            mutation_rate=profile.default_mutation_rate,
            crossover_rate=profile.default_crossover_rate,
            random_gene_rate=profile.default_random_gene_rate,
            learned_linkage=True,
            surrogate_min_evidence=policy["surrogate_min_evidence"],
            **proposal_args,
        )
        round_dir = args.output / f"round_{round_index:02d}"
        round_dir.mkdir()
        (round_dir / "proposals.json").write_text(
            json.dumps(
                {
                    "round": round_index,
                    "seed": round_seed,
                    "proposal_args": proposal_args,
                    "candidates": [
                        {
                            "candidate_hash": candidate.hash,
                            "source": candidate.source,
                            "parent_hashes": list(candidate.parent_hashes),
                            "params": candidate.canonical_params(),
                        }
                        for candidate in candidates
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        round_start = time.monotonic()
        schedule = execute_schedule(
            db,
            shapes=[shape],
            candidates=candidates,
            output_root=round_dir,
            target_profile=profile,
            protocol=protocol,
            min_samples=2,
            candidate_batch_size=policy["candidate_batch_size"],
            shape_batch_size=1,
            tensilelite_bin=args.tensilelite_bin,
            compile_threads=1,
            keep_going=True,
            runner_bin=args.runner_bin,
            build_timeout_s=args.build_timeout,
            runner_timeout_s=args.runner_timeout,
            adaptive_policy=adaptive_policy,
            probe_policy=probe_policy,
            adaptive_max_rounds=0,
            prepare_workers=policy["prepare_workers"],
            compile_cache_root=compile_cache,
        )
        round_record = {
            "round": round_index,
            "seed": round_seed,
            "candidate_count": len(candidates),
            "duration_s": time.monotonic() - round_start,
            "elapsed_s": time.monotonic() - start,
            "leader": _leader(
                db,
                shape_id=shape.id,
                problem_type_hash=profile.problem_type_hash,
                protocol_hash=protocol_hash,
            ),
            "schedule": _round_summary(schedule),
        }
        record["rounds"].append(round_record)
        (args.output / "campaign_progress.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(round_record, sort_keys=True), flush=True)
        round_index += 1

    hot_records = []
    if time.monotonic() < hard_deadline:
        hot_records = hot_confirm_topk(
            db_path=db_path,
            output_dir=args.output / "hot_loop_top8",
            runner_bin=args.runner_bin,
            shape_id=shape.id,
            problem_type_hash=profile.problem_type_hash,
            screening_protocol_hash=protocol_hash,
            validation_protocol_hash=protocol.validation_protocol_hash(),
            top_k=8,
            deadline=hard_deadline,
            runner_timeout_s=args.runner_timeout,
        )
    record.update(
        {
            "elapsed_s": time.monotonic() - start,
            "budget_overrun_s": max(0.0, time.monotonic() - hard_deadline),
            "screening_leader": _leader(
                db,
                shape_id=shape.id,
                problem_type_hash=profile.problem_type_hash,
                protocol_hash=protocol_hash,
            ),
            "hot_leader": hot_records[0] if hot_records else None,
            "hot_confirmed": len(hot_records),
        }
    )
    (args.output / "campaign_summary.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
