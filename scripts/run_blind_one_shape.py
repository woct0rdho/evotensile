#!/usr/bin/env python3
import argparse
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

from evotensile.adaptive_retime import AdaptivePolicy, ProbePolicy
from evotensile.candidate import Candidate, Shape
from evotensile.database import EvoTensileDB
from evotensile.profile import TargetProfile, get_profile
from evotensile.protocol import BenchmarkProtocol
from evotensile.runner import DEFAULT_TENSILELITE_BIN
from evotensile.scheduler import ScheduleResult, execute_schedule, propose_candidates
from evotensile.search.campaign_control import (
    convergence_detected,
    estimate_next_round_duration_s,
    load_island_elites,
    plateau_detected,
    population_diagnostics,
    split_budget,
    tag_proposals,
)
from evotensile.search.hot_confirm import hot_confirm_topk
from evotensile.search.mechanics import mechanical_coverage_tokens
from evotensile.search.screening_stabilize import (
    ScreeningStabilizationPolicy,
    stabilize_screening_leaders,
)
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
    validation_workers: int
    hot_reserve_s: float
    max_feedback_rounds_safety: int
    leader_stabilization: bool
    leader_top_k: int
    leader_min_samples: int
    leader_max_samples: int
    leader_min_timed_duration_us: float
    island_count: int
    island_isolation_rounds: int
    island_elites: int
    plateau_patience: int
    plateau_min_improvement_fraction: float
    restart_max_mean_hamming: float


class ProposalArgs(TypedDict):
    num_random: int
    elite_count: int
    local_count: int
    de_count: int
    gomea_count: int
    adaptive_operators: bool
    surrogate_pool_multiplier: int
    covering_cold_start: bool
    adaptive_group_credit: bool
    micro_exhaustive_neighborhoods: bool
    adaptive_donor_selection: bool
    cost_aware_operator_credit: bool
    surrogate_min_evidence: int


SEARCH_POLICY: CampaignPolicy = {
    "cold_candidates": 48,
    "cold_pool_multiplier": 8,
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
    "validation_workers": 1,
    "hot_reserve_s": 60.0,
    "max_feedback_rounds_safety": 100,
    "leader_stabilization": True,
    "leader_top_k": 4,
    "leader_min_samples": 6,
    "leader_max_samples": 10,
    "leader_min_timed_duration_us": 100_000.0,
    "island_count": 2,
    "island_isolation_rounds": 6,
    "island_elites": 16,
    "plateau_patience": 3,
    "plateau_min_improvement_fraction": 0.005,
    "restart_max_mean_hamming": 5.0,
}


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _round_summary(result: ScheduleResult) -> dict[str, object]:
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


def _leader(
    db: EvoTensileDB,
    *,
    shape_id: str,
    problem_type_hash: str,
    protocol_hash: str,
    island_id: str | None = None,
) -> dict[str, object] | None:
    rows = db.rank_evaluations(
        shape_id=shape_id,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=protocol_hash,
        min_samples=2,
        limit=None if island_id is not None else 1,
    )
    if not rows:
        return None
    if island_id is not None:
        candidates = {
            candidate.hash: candidate for candidate in db.get_candidates([row.candidate_hash for row in rows])
        }
        rows = [
            row
            for row in rows
            if str(candidates[row.candidate_hash].proposal_metadata.get("island_id", "")) == island_id
        ]
        if not rows:
            return None
    row = rows[0]
    return {
        "candidate_hash": row.candidate_hash,
        "median_gflops": row.median_gflops,
        "samples": row.samples,
    }


def _candidate_payload(candidate: Candidate) -> dict[str, object]:
    return {
        "candidate_hash": candidate.hash,
        "source": candidate.source,
        "parent_hashes": list(candidate.parent_hashes),
        "proposal_metadata": dict(candidate.proposal_metadata),
        "params": candidate.canonical_params(),
    }


def _candidate_from_payload(payload: Mapping[str, object]) -> Candidate:
    params = payload.get("params")
    parent_hashes = payload.get("parent_hashes", [])
    proposal_metadata = payload.get("proposal_metadata", {})
    if not isinstance(params, Mapping):
        raise ValueError("checkpoint candidate params must be a mapping")
    if not isinstance(parent_hashes, Sequence) or isinstance(parent_hashes, (str, bytes)):
        raise ValueError("checkpoint parent hashes must be a sequence")
    if not isinstance(proposal_metadata, Mapping):
        raise ValueError("checkpoint proposal metadata must be a mapping")
    candidate = Candidate(
        params={str(key): value for key, value in params.items()},
        source=str(payload["source"]),
        parent_hashes=tuple(str(value) for value in parent_hashes),
        proposal_metadata={str(key): value for key, value in proposal_metadata.items()},
    )
    expected_hash = str(payload["candidate_hash"])
    if candidate.hash != expected_hash:
        raise ValueError(f"checkpoint candidate hash mismatch: expected {expected_hash}, got {candidate.hash}")
    return candidate


def _cold_args(policy: CampaignPolicy, *, count: int) -> ProposalArgs:
    return {
        "num_random": count,
        "elite_count": 0,
        "local_count": 0,
        "de_count": 0,
        "gomea_count": 0,
        "adaptive_operators": False,
        "surrogate_pool_multiplier": policy["cold_pool_multiplier"],
        "covering_cold_start": True,
        "adaptive_group_credit": False,
        "micro_exhaustive_neighborhoods": False,
        "adaptive_donor_selection": False,
        "cost_aware_operator_credit": False,
        "surrogate_min_evidence": policy["surrogate_min_evidence"],
    }


def _feedback_args(
    policy: CampaignPolicy,
    *,
    part_index: int = 0,
    parts: int = 1,
) -> ProposalArgs:
    return {
        "num_random": split_budget(policy["feedback_random"], parts)[part_index],
        "elite_count": policy["elite_count"] if parts == 1 else policy["island_elites"],
        "local_count": split_budget(policy["feedback_semantic_mutation"], parts)[part_index],
        "de_count": split_budget(policy["feedback_de"], parts)[part_index],
        "gomea_count": split_budget(policy["feedback_gomea"], parts)[part_index],
        "adaptive_operators": True,
        "surrogate_pool_multiplier": policy["feedback_pool_multiplier"],
        "covering_cold_start": False,
        "adaptive_group_credit": True,
        "micro_exhaustive_neighborhoods": True,
        "adaptive_donor_selection": True,
        "cost_aware_operator_credit": True,
        "surrogate_min_evidence": policy["surrogate_min_evidence"],
    }


def _proposal_call(
    db: EvoTensileDB,
    *,
    shape: Shape,
    profile: TargetProfile,
    protocol_hash: str,
    seed: int,
    proposal_args: ProposalArgs,
    island_id: str,
    parents: Sequence[Candidate] | None,
    learned_linkage: bool,
    restart_index: int,
    cold_start_precovered_tokens: set[str] | None = None,
) -> tuple[list[Candidate], dict[str, object]]:
    parent_hashes = {candidate.hash for candidate in parents or ()}
    started = time.perf_counter()
    candidates = propose_candidates(
        db,
        proposal="family-qd",
        seed=seed,
        problem_type_hash=profile.problem_type_hash,
        benchmark_protocol_hash=protocol_hash,
        shape_id=shape.id,
        target_shapes=[shape],
        transfer_shape_count=0,
        transfer_per_shape=0,
        mutation_rate=profile.default_mutation_rate,
        crossover_rate=profile.default_crossover_rate,
        random_gene_rate=profile.default_random_gene_rate,
        learned_linkage=learned_linkage,
        parent_candidates=parents,
        cold_start_precovered_tokens=cold_start_precovered_tokens,
        **proposal_args,
    )
    duration = time.perf_counter() - started
    tagged = tag_proposals(
        candidates,
        island_id=island_id,
        parent_hashes=parent_hashes,
        proposal_duration_s=duration,
        restart_index=restart_index,
    )
    return tagged, {
        "island_id": island_id,
        "seed": seed,
        "restart_index": restart_index,
        "learned_linkage": learned_linkage,
        "parent_hashes": sorted(parent_hashes),
        "proposal_args": proposal_args,
        "duration_s": duration,
        "candidate_count": len(tagged),
        "generated_count": sum(candidate.hash not in parent_hashes for candidate in tagged),
    }


def _island_ids(policy: CampaignPolicy) -> tuple[str, ...]:
    return tuple(f"island-{index}" for index in range(policy["island_count"]))


def _leader_history(record: Mapping[str, object], *, island_id: str | None = None) -> list[float]:
    history = []
    rounds = record.get("rounds", [])
    if not isinstance(rounds, Sequence) or isinstance(rounds, (str, bytes)):
        return history
    for item in rounds:
        if not isinstance(item, Mapping):
            continue
        if island_id is None:
            leader = item.get("leader")
        else:
            leaders = item.get("island_leaders")
            leader = leaders.get(island_id) if isinstance(leaders, Mapping) else None
        if isinstance(leader, Mapping):
            median_gflops = leader.get("median_gflops")
            if isinstance(median_gflops, (int, float, str)):
                history.append(float(median_gflops))
    return history


def _restart_due(
    record: Mapping[str, object],
    *,
    island_id: str | None,
    policy: CampaignPolicy,
) -> bool:
    history = _leader_history(record, island_id=island_id)
    if not plateau_detected(
        history,
        patience=policy["plateau_patience"],
        minimum_improvement_fraction=policy["plateau_min_improvement_fraction"],
    ):
        return False
    rounds = record.get("rounds", [])
    if (
        not isinstance(rounds, Sequence)
        or isinstance(rounds, (str, bytes))
        or not rounds
        or not isinstance(rounds[-1], Mapping)
    ):
        return False
    diagnostics = rounds[-1].get("population_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return False
    mean_hamming = diagnostics.get("mean_pairwise_hamming")
    return isinstance(mean_hamming, (int, float, str)) and float(mean_hamming) <= policy["restart_max_mean_hamming"]


def _propose_round(
    db: EvoTensileDB,
    *,
    record: dict[str, Any],
    round_index: int,
    seed: int,
    shape: Shape,
    profile: TargetProfile,
    protocol_hash: str,
    policy: CampaignPolicy,
) -> tuple[list[Candidate], list[dict[str, object]]]:
    calls: list[dict[str, object]] = []
    combined: list[Candidate] = []
    islands = _island_ids(policy)
    if round_index == 0:
        precovered_tokens: set[str] = set()
        for island_index, (island_id, count) in enumerate(
            zip(islands, split_budget(policy["cold_candidates"], len(islands)), strict=True)
        ):
            candidates, call = _proposal_call(
                db,
                shape=shape,
                profile=profile,
                protocol_hash=protocol_hash,
                seed=seed + island_index * 1_000_003,
                proposal_args=_cold_args(policy, count=count),
                island_id=island_id,
                parents=None,
                learned_linkage=False,
                restart_index=0,
                cold_start_precovered_tokens=precovered_tokens,
            )
            combined.extend(candidates)
            precovered_tokens.update(
                token for candidate in candidates for token in mechanical_coverage_tokens(candidate, shape)
            )
            calls.append(call)
        return list({candidate.hash: candidate for candidate in combined}.values()), calls

    if round_index <= policy["island_isolation_rounds"]:
        for island_index, island_id in enumerate(islands):
            parents = load_island_elites(
                db,
                island_id=island_id,
                shape_id=shape.id,
                problem_type_hash=profile.problem_type_hash,
                benchmark_protocol_hash=protocol_hash,
                limit=policy["island_elites"],
            )
            restart_due = _restart_due(record, island_id=island_id, policy=policy)
            restart_index = sum(
                1
                for item in record["rounds"]
                if isinstance(item, Mapping)
                and any(
                    isinstance(call, Mapping)
                    and call.get("island_id") == island_id
                    and int(call.get("restart_index", 0)) > 0
                    for call in item.get("proposal_calls", [])
                )
            )
            proposal_args = (
                _cold_args(policy, count=split_budget(policy["feedback_candidates"], len(islands))[island_index])
                if restart_due or not parents
                else _feedback_args(policy, part_index=island_index, parts=len(islands))
            )
            if restart_due:
                restart_index += 1
                parents = []
            candidates, call = _proposal_call(
                db,
                shape=shape,
                profile=profile,
                protocol_hash=protocol_hash,
                seed=seed + island_index * 1_000_003,
                proposal_args=proposal_args,
                island_id=island_id,
                parents=parents or None,
                learned_linkage=False,
                restart_index=restart_index,
            )
            combined.extend(candidates)
            calls.append(call)
        return list({candidate.hash: candidate for candidate in combined}.values()), calls

    global_restart = _restart_due(record, island_id=None, policy=policy)
    if global_restart:
        feedback_args = _feedback_args(policy, part_index=0, parts=2)
        feedback, feedback_call = _proposal_call(
            db,
            shape=shape,
            profile=profile,
            protocol_hash=protocol_hash,
            seed=seed,
            proposal_args=feedback_args,
            island_id="merged",
            parents=None,
            learned_linkage=True,
            restart_index=0,
        )
        restart_count = sum(
            1
            for item in record["rounds"]
            if isinstance(item, Mapping)
            and any(
                isinstance(call, Mapping) and str(call.get("island_id", "")).startswith("restart-")
                for call in item.get("proposal_calls", [])
            )
        )
        restart_id = f"restart-{restart_count + 1}"
        restart, restart_call = _proposal_call(
            db,
            shape=shape,
            profile=profile,
            protocol_hash=protocol_hash,
            seed=seed + 1_000_003,
            proposal_args=_cold_args(policy, count=split_budget(policy["feedback_candidates"], 2)[1]),
            island_id=restart_id,
            parents=None,
            learned_linkage=False,
            restart_index=restart_count + 1,
        )
        combined.extend(feedback)
        combined.extend(restart)
        calls.extend([feedback_call, restart_call])
    else:
        candidates, call = _proposal_call(
            db,
            shape=shape,
            profile=profile,
            protocol_hash=protocol_hash,
            seed=seed,
            proposal_args=_feedback_args(policy),
            island_id="merged",
            parents=None,
            learned_linkage=True,
            restart_index=0,
        )
        combined.extend(candidates)
        calls.append(call)
    return list({candidate.hash: candidate for candidate in combined}.values()), calls


def _load_pending_proposals(path: Path) -> tuple[list[Candidate], list[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = [_candidate_from_payload(item) for item in payload["candidates"]]
    return candidates, list(payload.get("proposal_calls", []))


def _checkpoint(
    output: Path,
    *,
    record: Mapping[str, object],
    phase: str,
    round_index: int,
    round_seed: int | None,
    candidate_hashes: Sequence[str],
) -> None:
    _write_json_atomic(
        output / "campaign_checkpoint.json",
        {
            "phase": phase,
            "round": round_index,
            "round_seed": round_seed,
            "candidate_hashes": list(candidate_hashes),
            "search_elapsed_s": record.get("search_elapsed_s", 0.0),
            "active_elapsed_s": record.get("active_elapsed_s", 0.0),
            "policy": record["policy"],
            "deterministic_rng": "round and proposal-call seeds fully determine generator and surrogate RNG state",
            "operator_credit_state": "derived from the checkpointed campaign DB",
            "surrogate_state": "refit deterministically from the checkpointed campaign DB and stored proposals",
        },
    )


def _load_or_create_campaign(
    args: argparse.Namespace,
    *,
    profile: TargetProfile,
    shape: Shape,
    protocol_hash: str,
    policy: CampaignPolicy,
) -> tuple[dict[str, Any], bool]:
    progress_path = args.output / "campaign_progress.json"
    frozen_path = args.output / "frozen_policy.json"
    if args.output.exists():
        if not args.resume:
            raise SystemExit(f"output already exists: {args.output}")
        if not frozen_path.exists():
            raise SystemExit(f"cannot resume without {frozen_path}")
        record = json.loads((progress_path if progress_path.exists() else frozen_path).read_text(encoding="utf-8"))
        expected = (args.seed, shape.id, profile.name)
        actual = (int(record["seed"]), str(record["shape"]), str(record["profile"]))
        if actual != expected:
            raise SystemExit(f"resume identity mismatch: expected {expected}, found {actual}")
        return record, True

    args.output.mkdir(parents=True)
    record: dict[str, Any] = {
        "blind": True,
        "seed": args.seed,
        "shape": shape.id,
        "profile": profile.name,
        "problem_type_hash": profile.problem_type_hash,
        "screening_protocol_hash": protocol_hash,
        "time_budget_s": args.time_budget,
        "policy": policy,
        "rounds": [],
        "search_elapsed_s": 0.0,
        "active_elapsed_s": 0.0,
        "stop_reason": None,
    }
    _write_json_atomic(frozen_path, record)
    return record, False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the blind one-shape 20-minute search policy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", default="8192,8192,1,8192")
    parser.add_argument("--profile", default="gfx1151-nt-hhs")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--time-budget", type=float, default=1200.0)
    parser.add_argument("--hot-reserve", type=float, default=SEARCH_POLICY["hot_reserve_s"])
    parser.add_argument(
        "--max-feedback-rounds",
        type=int,
        default=SEARCH_POLICY["max_feedback_rounds_safety"],
    )
    parser.add_argument("--no-leader-stabilization", action="store_true")
    parser.add_argument("--early-stop-on-convergence", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--runner-bin", type=Path, default=Path("build/evotensile-structured-runner"))
    parser.add_argument("--tensilelite-bin", type=Path, default=Path(DEFAULT_TENSILELITE_BIN))
    parser.add_argument("--build-timeout", type=float, default=300.0)
    parser.add_argument("--runner-timeout", type=float, default=300.0)
    args = parser.parse_args()

    profile = get_profile(args.profile)
    shape = parse_shape(args.shape)
    protocol = BenchmarkProtocol(
        num_warmups=1,
        num_benchmarks=2,
        enqueues_per_sync=1,
        syncs_per_benchmark=1,
    )
    protocol_hash = profile.benchmark_protocol_hash(protocol)
    policy: CampaignPolicy = {
        **SEARCH_POLICY,
        "hot_reserve_s": args.hot_reserve,
        "max_feedback_rounds_safety": args.max_feedback_rounds,
        "leader_stabilization": not args.no_leader_stabilization,
    }
    record, resumed = _load_or_create_campaign(
        args,
        profile=profile,
        shape=shape,
        protocol_hash=protocol_hash,
        policy=policy,
    )
    policy = record["policy"]
    db_path = args.output / "campaign.sqlite"
    db = EvoTensileDB.connect(db_path)
    db.init()
    db.register_shapes([shape])

    session_start = time.monotonic()
    prior_search_elapsed = float(record.get("search_elapsed_s", 0.0))
    prior_active_elapsed = float(record.get("active_elapsed_s", prior_search_elapsed))
    remaining_active = max(0.0, args.time_budget - prior_active_elapsed)
    remaining_search = max(0.0, args.time_budget - policy["hot_reserve_s"] - prior_search_elapsed)
    hard_deadline = session_start + remaining_active
    search_deadline = session_start + min(remaining_active, remaining_search)
    adaptive_policy = AdaptivePolicy()
    probe_policy = ProbePolicy()
    stabilization_policy = ScreeningStabilizationPolicy(
        top_k=policy["leader_top_k"],
        min_samples=policy["leader_min_samples"],
        max_samples=policy["leader_max_samples"],
        min_timed_duration_us=policy["leader_min_timed_duration_us"],
    )
    compile_cache = args.output / "compile_cache"

    checkpoint_path = args.output / "campaign_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    record["search_elapsed_s"] = max(
        float(record.get("search_elapsed_s", 0.0)),
        float(checkpoint.get("search_elapsed_s", 0.0)),
    )
    record["active_elapsed_s"] = max(
        float(record.get("active_elapsed_s", 0.0)),
        float(checkpoint.get("active_elapsed_s", 0.0)),
    )
    if checkpoint.get("phase") == "finished" and (args.output / "campaign_summary.json").exists():
        print((args.output / "campaign_summary.json").read_text(encoding="utf-8"), end="")
        return 0
    prior_search_elapsed = float(record.get("search_elapsed_s", 0.0))
    prior_active_elapsed = float(record.get("active_elapsed_s", prior_search_elapsed))
    remaining_active = max(0.0, args.time_budget - prior_active_elapsed)
    remaining_search = max(0.0, args.time_budget - policy["hot_reserve_s"] - prior_search_elapsed)
    hard_deadline = session_start + remaining_active
    search_deadline = session_start + min(remaining_active, remaining_search)
    round_index = int(checkpoint.get("round", len(record["rounds"])))
    if checkpoint.get("phase") == "completed":
        round_index = max(round_index, len(record["rounds"]))

    while round_index <= args.max_feedback_rounds:
        now = time.monotonic()
        if now >= search_deadline:
            record["stop_reason"] = "search_deadline"
            break
        pending = checkpoint.get("phase") == "proposed" and int(checkpoint.get("round", -1)) == round_index
        if not pending and round_index > 0:
            next_round_guard_s = estimate_next_round_duration_s(
                record["rounds"],
                expected_missing_pairs=policy["feedback_candidates"],
            )
            if search_deadline - now < next_round_guard_s:
                record["stop_reason"] = "insufficient_predicted_round_budget"
                break

        round_seed = args.seed + round_index * 10007
        round_dir = args.output / f"round_{round_index:02d}"
        round_dir.mkdir(exist_ok=True)
        proposals_path = round_dir / "proposals.json"
        if pending:
            candidates, proposal_calls = _load_pending_proposals(proposals_path)
        else:
            candidates, proposal_calls = _propose_round(
                db,
                record=record,
                round_index=round_index,
                seed=round_seed,
                shape=shape,
                profile=profile,
                protocol_hash=protocol_hash,
                policy=policy,
            )
            _write_json_atomic(
                proposals_path,
                {
                    "round": round_index,
                    "seed": round_seed,
                    "proposal_calls": proposal_calls,
                    "candidates": [_candidate_payload(candidate) for candidate in candidates],
                },
            )
            record["search_elapsed_s"] = prior_search_elapsed + time.monotonic() - session_start
            record["active_elapsed_s"] = prior_active_elapsed + time.monotonic() - session_start
            _write_json_atomic(args.output / "campaign_progress.json", record)
            _checkpoint(
                args.output,
                record=record,
                phase="proposed",
                round_index=round_index,
                round_seed=round_seed,
                candidate_hashes=[candidate.hash for candidate in candidates],
            )
            checkpoint = {
                "phase": "proposed",
                "round": round_index,
                "round_seed": round_seed,
            }

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
            cost_aware_scheduling=True,
            validation_workers=policy.get("validation_workers", 1),
        )
        stabilization = None
        if policy["leader_stabilization"] and time.monotonic() < search_deadline:
            stabilization = stabilize_screening_leaders(
                db,
                shape=shape,
                problem_type_hash=profile.problem_type_hash,
                screening_protocol=protocol,
                validation_protocol_hash=protocol.validation_protocol_hash(),
                output_dir=round_dir / "leader_stabilization",
                runner_bin=args.runner_bin,
                policy=stabilization_policy,
                architecture=str(profile.library_logic["ArchitectureName"]),
                deadline=search_deadline,
                runner_timeout_s=args.runner_timeout,
            )
        diagnostics = population_diagnostics(candidates, shape)
        island_leaders = {
            island_id: _leader(
                db,
                shape_id=shape.id,
                problem_type_hash=profile.problem_type_hash,
                protocol_hash=protocol_hash,
                island_id=island_id,
            )
            for island_id in _island_ids(policy)
        }
        round_record = {
            "round": round_index,
            "seed": round_seed,
            "resumed_pending_proposals": bool(pending and resumed),
            "candidate_count": len(candidates),
            "duration_s": time.monotonic()
            - round_start
            + sum(
                float(value)
                for call in proposal_calls
                if isinstance((value := call.get("duration_s")), (int, float, str))
            ),
            "elapsed_s": prior_search_elapsed + time.monotonic() - session_start,
            "leader": _leader(
                db,
                shape_id=shape.id,
                problem_type_hash=profile.problem_type_hash,
                protocol_hash=protocol_hash,
            ),
            "island_leaders": island_leaders,
            "population_diagnostics": diagnostics.to_dict(),
            "proposal_calls": proposal_calls,
            "schedule": _round_summary(schedule),
            "leader_stabilization": None if stabilization is None else stabilization.to_dict(),
        }
        record["rounds"].append(round_record)
        record["search_elapsed_s"] = prior_search_elapsed + time.monotonic() - session_start
        record["active_elapsed_s"] = prior_active_elapsed + time.monotonic() - session_start
        _write_json_atomic(args.output / "campaign_progress.json", record)
        _checkpoint(
            args.output,
            record=record,
            phase="completed",
            round_index=round_index + 1,
            round_seed=None,
            candidate_hashes=(),
        )
        checkpoint = {"phase": "completed", "round": round_index + 1}
        print(json.dumps(round_record, sort_keys=True), flush=True)
        round_index += 1

        if args.early_stop_on_convergence:
            history = _leader_history(record)
            if convergence_detected(history, diagnostics):
                record["stop_reason"] = "converged"
                break

    search_session_elapsed = time.monotonic() - session_start
    record["search_elapsed_s"] = max(
        float(record.get("search_elapsed_s", 0.0)),
        prior_search_elapsed + min(search_session_elapsed, max(0.0, search_deadline - session_start)),
    )
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
            "active_elapsed_s": prior_active_elapsed + time.monotonic() - session_start,
            "elapsed_s": prior_active_elapsed + time.monotonic() - session_start,
            "budget_overrun_s": max(0.0, prior_active_elapsed + time.monotonic() - session_start - args.time_budget),
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
    _write_json_atomic(args.output / "campaign_summary.json", record)
    _write_json_atomic(args.output / "campaign_progress.json", record)
    _checkpoint(
        args.output,
        record=record,
        phase="finished",
        round_index=round_index,
        round_seed=None,
        candidate_hashes=(),
    )
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
