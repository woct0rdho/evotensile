import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from evotensile.campaign.models import CampaignConfiguration
from evotensile.campaign.proposal_policy import island_ids, leader_history, propose_round
from evotensile.campaign.store import CampaignStore
from evotensile.candidate import Shape
from evotensile.database import EvoTensileDB
from evotensile.profile import TargetProfile
from evotensile.scheduler import execute_schedule
from evotensile.scheduling.models import ScheduleResult
from evotensile.search.campaign_control import (
    convergence_detected,
    estimate_confirmation_reserve_s,
    estimate_next_round_duration_s,
    population_diagnostics,
)
from evotensile.search.hot_confirm import hot_confirm_topk
from evotensile.search.screening_stabilize import stabilize_screening_leaders


@dataclass(frozen=True)
class OneShapeCampaign:
    configuration: CampaignConfiguration
    profile: TargetProfile
    shape: Shape
    store: CampaignStore
    resume: bool = False


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
        "probe_policy_hash": result.probe_policy_hash,
        "probe_survivor_pairs": result.probe_survivor_pairs,
        "probe_screened_pairs": result.probe_screened_pairs,
        "probe_preprepare_screened_pairs": result.probe_preprepare_screened_pairs,
        "status_counts": dict(sorted(status_counts.items())),
        "errors": errors,
    }


def _confirmation_reserve_s(
    db: EvoTensileDB,
    *,
    shape_id: str,
    problem_type_hash: str,
    protocol_hash: str,
    configuration: CampaignConfiguration,
) -> float:
    finalists = db.rank_evaluations(
        shape_id=shape_id,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=protocol_hash,
        min_samples=configuration.min_samples,
        limit=configuration.hot_top_k,
    )
    return estimate_confirmation_reserve_s(
        [row.median_time_us for row in finalists if row.median_time_us is not None],
        protocol=configuration.hot_protocol,
        top_k=configuration.hot_top_k,
        minimum_reserve_s=configuration.hot_reserve_s,
    )


def _leader(
    db: EvoTensileDB,
    *,
    shape_id: str,
    problem_type_hash: str,
    protocol_hash: str,
    min_samples: int,
    island_id: str | None = None,
) -> dict[str, object] | None:
    rows = db.rank_evaluations(
        shape_id=shape_id,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=protocol_hash,
        min_samples=min_samples,
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


def run_one_shape_campaign(campaign: OneShapeCampaign) -> int:
    profile = campaign.profile
    shape = campaign.shape
    configuration = campaign.configuration
    protocol = configuration.screening_protocol
    protocol_hash = protocol.protocol_hash()
    store = campaign.store
    record, resumed = store.load_or_create(
        configuration,
        resume=campaign.resume,
        island_ids=island_ids(configuration),
    )
    db_path = store.db_path
    db = EvoTensileDB.connect(db_path)
    db.init()
    db.register_shapes([shape])

    session_start = time.monotonic()
    adaptive_policy = configuration.adaptive_policy
    probe_policy = configuration.probe_policy
    stabilization_policy = configuration.stabilization_policy
    compile_cache = store.compile_cache_path if configuration.compile_cache else None

    checkpoint = store.load_checkpoint()
    record["search_elapsed_s"] = max(
        float(record.get("search_elapsed_s", 0.0)),
        float(checkpoint.get("search_elapsed_s", 0.0)),
    )
    record["active_elapsed_s"] = max(
        float(record.get("active_elapsed_s", 0.0)),
        float(checkpoint.get("active_elapsed_s", 0.0)),
    )
    if "restart_counters" in checkpoint:
        record["restart_counters"] = checkpoint["restart_counters"]
    if checkpoint.get("phase") == "finished" and store.summary_path.exists():
        print(store.summary_path.read_text(encoding="utf-8"), end="")
        return 0
    prior_search_elapsed = float(record.get("search_elapsed_s", 0.0))
    prior_active_elapsed = float(record.get("active_elapsed_s", prior_search_elapsed))
    remaining_campaign_s = max(0.0, configuration.time_budget_s - prior_active_elapsed)
    confirmation_reserve_s = _confirmation_reserve_s(
        db,
        shape_id=shape.id,
        problem_type_hash=profile.problem_type_hash,
        protocol_hash=protocol_hash,
        configuration=configuration,
    )
    record["confirmation_reserve_s"] = confirmation_reserve_s
    remaining_search = max(
        0.0,
        configuration.time_budget_s - confirmation_reserve_s - prior_search_elapsed,
    )
    campaign_admission_deadline = session_start + remaining_campaign_s
    search_admission_deadline = session_start + min(remaining_campaign_s, remaining_search)
    round_index = int(checkpoint.get("round", len(record["rounds"])))
    if checkpoint.get("phase") == "completed":
        round_index = max(round_index, len(record["rounds"]))

    while round_index <= configuration.max_feedback_rounds:
        confirmation_reserve_s = _confirmation_reserve_s(
            db,
            shape_id=shape.id,
            problem_type_hash=profile.problem_type_hash,
            protocol_hash=protocol_hash,
            configuration=configuration,
        )
        record["confirmation_reserve_s"] = confirmation_reserve_s
        remaining_search = max(
            0.0,
            configuration.time_budget_s - confirmation_reserve_s - prior_search_elapsed,
        )
        search_admission_deadline = session_start + min(remaining_campaign_s, remaining_search)
        now = time.monotonic()
        if now >= search_admission_deadline:
            record["stop_reason"] = "search_soft_deadline"
            break
        pending = checkpoint.get("phase") == "proposed" and int(checkpoint.get("round", -1)) == round_index
        if not pending and round_index > 0:
            next_round_guard_s = estimate_next_round_duration_s(
                record["rounds"],
                expected_missing_pairs=configuration.feedback_candidates,
            )
            if search_admission_deadline - now < next_round_guard_s:
                record["stop_reason"] = "insufficient_predicted_round_budget"
                break

        round_seed = configuration.seed + round_index * 10007
        round_dir = store.round_dir(round_index)
        if pending:
            round_proposal = store.load_proposal(round_index)
        else:
            round_proposal = propose_round(
                db,
                record=record,
                round_index=round_index,
                seed=round_seed,
                shape=shape,
                profile=profile,
                protocol_hash=protocol_hash,
                configuration=configuration,
            )
            store.write_proposal(round_index, round_seed, round_proposal)
            record["search_elapsed_s"] = prior_search_elapsed + time.monotonic() - session_start
            record["active_elapsed_s"] = prior_active_elapsed + time.monotonic() - session_start
            store.write_progress(record)
            store.write_checkpoint(
                record=record,
                phase="proposed",
                round_index=round_index,
                round_seed=round_seed,
                candidate_hashes=[candidate.hash for candidate in round_proposal.selected],
            )
            checkpoint = {
                "phase": "proposed",
                "round": round_index,
                "round_seed": round_seed,
            }

        candidates = list(round_proposal.selected)
        proposal_events = list(round_proposal.events)
        round_start = time.monotonic()
        schedule = execute_schedule(
            db,
            shapes=[shape],
            candidates=candidates,
            output_root=round_dir,
            target_profile=profile,
            protocol=protocol,
            min_samples=configuration.min_samples,
            candidate_batch_size=configuration.candidate_batch_size,
            shape_batch_size=configuration.shape_batch_size,
            tensilelite_bin=Path(configuration.tensilelite_bin),
            compile_threads=configuration.compile_threads,
            keep_going=configuration.keep_going,
            runner_bin=Path(configuration.runner_bin),
            build_timeout_s=configuration.build_timeout_s,
            runner_timeout_s=configuration.runner_timeout_s,
            adaptive_policy=adaptive_policy,
            probe_policy=probe_policy,
            adaptive_max_rounds=configuration.adaptive_max_rounds,
            prepare_workers=configuration.prepare_workers,
            prepare_wave_batches=configuration.prepare_wave_batches,
            compile_cache_root=compile_cache,
            cost_aware_scheduling=configuration.cost_aware_scheduling,
            validation_workers=configuration.validation_workers,
        )
        stabilization = None
        if configuration.leader_stabilization and time.monotonic() < search_admission_deadline:
            stabilization = stabilize_screening_leaders(
                db,
                shapes=[shape],
                problem_type_hash=profile.problem_type_hash,
                screening_protocol=protocol,
                validation_protocol_hash=protocol.validation_protocol_hash(),
                output_dir=round_dir / "leader_stabilization",
                runner_bin=Path(configuration.runner_bin),
                policy=stabilization_policy,
                admission_deadline=search_admission_deadline,
                runner_timeout_s=configuration.runner_timeout_s,
            )
        active_diagnostics = population_diagnostics(
            round_proposal.active,
            shape,
            workgroup_processor_count=configuration.workgroup_processor_count,
        )
        archive_diagnostics = population_diagnostics(
            round_proposal.archive,
            shape,
            workgroup_processor_count=configuration.workgroup_processor_count,
        )
        measured_new = {
            candidate.hash: candidate for batch in schedule.planned_batches for candidate in batch.candidates
        }
        measured_new_diagnostics = population_diagnostics(
            tuple(measured_new.values()),
            shape,
            workgroup_processor_count=configuration.workgroup_processor_count,
        )
        island_leaders = {
            island_id: _leader(
                db,
                shape_id=shape.id,
                problem_type_hash=profile.problem_type_hash,
                protocol_hash=protocol_hash,
                min_samples=configuration.min_samples,
                island_id=island_id,
            )
            for island_id in island_ids(configuration)
        }
        round_record = {
            "round": round_index,
            "seed": round_seed,
            "resumed_pending_proposals": bool(pending and resumed),
            "selected_candidate_count": len(candidates),
            "active_candidate_count": len(round_proposal.active),
            "archive_candidate_count": len(round_proposal.archive),
            "measured_new_candidate_count": len(measured_new),
            "duration_s": time.monotonic() - round_start + sum(event.duration_s for event in proposal_events),
            "elapsed_s": prior_search_elapsed + time.monotonic() - session_start,
            "confirmation_reserve_s": confirmation_reserve_s,
            "leader": _leader(
                db,
                shape_id=shape.id,
                problem_type_hash=profile.problem_type_hash,
                protocol_hash=protocol_hash,
                min_samples=configuration.min_samples,
            ),
            "island_leaders": island_leaders,
            "active_population_diagnostics": active_diagnostics.to_dict(),
            "measured_new_population_diagnostics": measured_new_diagnostics.to_dict(),
            "archive_diagnostics": archive_diagnostics.to_dict(),
            "proposal_events": [event.to_dict() for event in proposal_events],
            "schedule": _round_summary(schedule),
            "leader_stabilization": None if stabilization is None else stabilization.to_dict(),
        }
        record["rounds"].append(round_record)
        record["search_elapsed_s"] = prior_search_elapsed + time.monotonic() - session_start
        record["active_elapsed_s"] = prior_active_elapsed + time.monotonic() - session_start
        store.write_progress(record)
        store.write_checkpoint(
            record=record,
            phase="completed",
            round_index=round_index + 1,
            round_seed=None,
            candidate_hashes=(),
        )
        checkpoint = {"phase": "completed", "round": round_index + 1}
        print(json.dumps(round_record, sort_keys=True), flush=True)
        round_index += 1

        if configuration.early_stop_on_convergence:
            history = leader_history(record)
            if convergence_detected(
                history,
                active_diagnostics,
                patience=configuration.convergence_patience,
                minimum_improvement_fraction=configuration.convergence_minimum_improvement_fraction,
                maximum_mean_hamming=configuration.convergence_maximum_mean_hamming,
            ):
                record["stop_reason"] = "converged"
                break

    search_session_elapsed = time.monotonic() - session_start
    record["search_elapsed_s"] = max(
        float(record.get("search_elapsed_s", 0.0)),
        prior_search_elapsed + search_session_elapsed,
    )
    hot_records = hot_confirm_topk(
        db_path=db_path,
        output_dir=store.root / "hot_loop_top8",
        runner_bin=Path(configuration.runner_bin),
        shape_id=shape.id,
        problem_type_hash=profile.problem_type_hash,
        screening_protocol_hash=protocol_hash,
        validation_protocol_hash=protocol.validation_protocol_hash(),
        hot_protocol=configuration.hot_protocol,
        top_k=configuration.hot_top_k,
        admission_deadline=campaign_admission_deadline,
        runner_timeout_s=configuration.runner_timeout_s,
    )
    record.update(
        {
            "active_elapsed_s": prior_active_elapsed + time.monotonic() - session_start,
            "elapsed_s": prior_active_elapsed + time.monotonic() - session_start,
            "budget_overrun_s": max(
                0.0,
                prior_active_elapsed + time.monotonic() - session_start - configuration.time_budget_s,
            ),
            "screening_leader": _leader(
                db,
                shape_id=shape.id,
                problem_type_hash=profile.problem_type_hash,
                protocol_hash=protocol_hash,
                min_samples=configuration.min_samples,
            ),
            "hot_leader": hot_records[0] if hot_records else None,
            "hot_confirmed": len(hot_records),
        }
    )
    store.write_summary(record)
    store.write_progress(record)
    store.write_checkpoint(
        record=record,
        phase="finished",
        round_index=round_index,
        round_seed=None,
        candidate_hashes=(),
    )
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0
