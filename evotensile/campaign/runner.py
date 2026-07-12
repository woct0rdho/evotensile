import json
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from evotensile.campaign.controller import (
    CampaignControllerState,
    estimate_admission_duration_s,
)
from evotensile.campaign.evaluator import RealEvaluator, RealEvaluatorContext
from evotensile.campaign.models import CampaignConfiguration
from evotensile.campaign.proposal_policy import island_ids, leader_history, propose_round
from evotensile.campaign.store import CampaignRecord, CampaignStore, StoredCampaignCheckpoint
from evotensile.candidate import Shape
from evotensile.database import EvoTensileDB
from evotensile.profile import TargetProfile
from evotensile.scheduling.models import EvidenceStage, PairRequest, ScheduleResult
from evotensile.search.campaign_control import (
    convergence_detected,
    estimate_confirmation_reserve_s,
    population_diagnostics,
)
from evotensile.search.hot_confirm import hot_confirm_topk
from evotensile.search.screening_stabilize import stabilize_screening_leaders
from evotensile.search.shape_clustering import ShapeClusteringConfiguration, cluster_shapes


@dataclass(frozen=True)
class CampaignRun:
    configuration: CampaignConfiguration
    profile: TargetProfile
    shapes: tuple[Shape, ...]
    store: CampaignStore
    resume: bool = False

    def __post_init__(self) -> None:
        if not self.shapes:
            raise ValueError("campaign requires at least one shape")
        if len({shape.id for shape in self.shapes}) != len(self.shapes):
            raise ValueError("campaign shapes must be unique")


def _round_cost_observations(rounds: list[Mapping[str, object]]) -> list[tuple[float, int]]:
    observations = []
    for item in rounds[-6:]:
        schedule = item.get("schedule")
        if not isinstance(schedule, dict):
            continue
        requested_value = schedule.get("requested_pairs", 0)
        duration_value = item.get("duration_s", 0.0)
        requested = int(requested_value) if isinstance(requested_value, (int, float, str)) else 0
        duration = float(duration_value) if isinstance(duration_value, (int, float, str)) else 0.0
        if requested > 0 and duration > 0.0:
            observations.append((duration, requested))
    return observations


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
        "requested_pairs": result.requested_pairs,
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
    finalists = db.rank_benchmarks(
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
    rows = db.rank_benchmarks(
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


def _restore_controller(
    checkpoint: StoredCampaignCheckpoint,
    *,
    shape_ids: tuple[str, ...],
    time_budget_s: float,
    session_started_at: float,
    resume: bool,
) -> CampaignControllerState:
    payload = checkpoint.get("controller")
    if payload is None:
        if resume and checkpoint:
            raise ValueError("campaign checkpoint does not contain controller state")
        return CampaignControllerState(
            shape_ids=shape_ids,
            time_budget_s=time_budget_s,
            session_started_at=session_started_at,
        )
    controller = CampaignControllerState.from_checkpoint(
        payload,
        session_started_at=session_started_at,
    )
    if controller.shape_ids != shape_ids or controller.time_budget_s != time_budget_s:
        raise ValueError("campaign checkpoint controller identity mismatch")
    return controller


def _write_checkpoint(
    store: CampaignStore,
    *,
    record: CampaignRecord,
    controller: CampaignControllerState,
    round_seed: int | None,
    candidate_hashes: tuple[str, ...] | list[str],
) -> None:
    record["controller"] = controller.summary()
    store.write_progress(record)
    store.write_checkpoint(
        record=record,
        controller=controller,
        round_seed=round_seed,
        candidate_hashes=candidate_hashes,
    )


def run_campaign(campaign: CampaignRun) -> int:
    if len(campaign.shapes) != 1:
        raise ValueError("the current proposal policy profile supports exactly one campaign shape")
    profile = campaign.profile
    shape = campaign.shapes[0]
    configuration = campaign.configuration
    if configuration.shape_id != shape.id:
        raise ValueError("campaign configuration shape does not match controller shape")
    protocol = configuration.screening_protocol
    protocol_hash = protocol.protocol_hash()
    store = campaign.store
    record, resumed = store.load_or_create(
        configuration,
        resume=campaign.resume,
        island_ids=island_ids(configuration),
    )
    db = EvoTensileDB.connect(
        store.db_path,
        environment_compatibility_tag=profile.environment_compatibility_tag,
    )
    db.init()
    db.register_shapes([shape])

    session_start = time.monotonic()
    checkpoint = store.load_checkpoint()
    controller = _restore_controller(
        checkpoint,
        shape_ids=(shape.id,),
        time_budget_s=configuration.time_budget_s,
        session_started_at=session_start,
        resume=campaign.resume,
    )
    if controller.clustering is None:
        controller.set_clustering(
            cluster_shapes(
                campaign.shapes,
                ShapeClusteringConfiguration(
                    workgroup_processor_count=profile.workgroup_processor_count,
                    cluster_count=1,
                ),
            ).to_dict()
        )
    if "restart_counters" in checkpoint:
        record["restart_counters"] = checkpoint["restart_counters"]
    if controller.phase == "finished" and store.summary_path.exists():
        print(store.summary_path.read_text(encoding="utf-8"), end="")
        return 0

    adaptive_policy = configuration.adaptive_policy
    probe_policy = configuration.probe_policy
    stabilization_policy = configuration.stabilization_policy
    compile_cache = store.compile_cache_path if configuration.compile_cache else None

    while controller.round_index <= configuration.max_feedback_rounds:
        round_index = controller.round_index
        confirmation_reserve_s = _confirmation_reserve_s(
            db,
            shape_id=shape.id,
            problem_type_hash=profile.problem_type_hash,
            protocol_hash=protocol_hash,
            configuration=configuration,
        )
        controller.set_reserve("confirmation", confirmation_reserve_s)
        record["confirmation_reserve_s"] = confirmation_reserve_s
        pending = controller.phase == "proposed"
        if not pending:
            predicted_duration_s = 0.0
            if round_index > 0:
                predicted_duration_s = estimate_admission_duration_s(
                    _round_cost_observations(record["rounds"]),
                    expected_units=configuration.feedback_candidates,
                )
            decision = controller.decide_admission(
                predicted_duration_s=predicted_duration_s,
                reserve_s=confirmation_reserve_s,
            )
            if not decision.admitted:
                record["stop_reason"] = (
                    "search_soft_deadline" if decision.reason == "soft_deadline" else decision.reason
                )
                break
        else:
            controller.append_trace("resume_pending", {"round_index": round_index})

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
            controller.record_phase_time("proposal", sum(event.duration_s for event in round_proposal.events))
            store.write_proposal(round_index, round_seed, round_proposal)
            controller.transition("proposed", round_index=round_index)
            _write_checkpoint(
                store,
                record=record,
                controller=controller,
                round_seed=round_seed,
                candidate_hashes=[candidate.hash for candidate in round_proposal.selected],
            )

        candidates = list(round_proposal.selected)
        proposal_events = list(round_proposal.events)
        round_start = time.monotonic()
        evaluation = RealEvaluator(
            RealEvaluatorContext(
                db=db,
                output_root=round_dir,
                target_profile=profile,
                protocol=protocol,
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
                prepare_workers=configuration.prepare_workers,
                prepare_wave_batches=configuration.prepare_wave_batches,
                compile_cache_root=compile_cache,
                cost_aware_scheduling=configuration.cost_aware_scheduling,
                validation_workers=configuration.validation_workers,
            )
        ).evaluate(
            [
                PairRequest(
                    candidate=candidate,
                    shape=shape,
                    evidence_stage=EvidenceStage.SCREENING,
                    min_samples=configuration.min_samples,
                )
                for candidate in candidates
            ]
        )
        evaluation.apply(controller)
        schedule = evaluation.schedules[0]

        stabilization = None
        search_admission_deadline = controller.admission_deadline - confirmation_reserve_s
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
            controller.record_phase_time("stabilization", stabilization.duration_s)
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
            pair.request.candidate.hash: pair.request.candidate
            for batch in schedule.planned_batches
            for pair in batch.pairs
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
        round_record: dict[str, object] = {
            "round": round_index,
            "seed": round_seed,
            "resumed_pending_proposals": bool(pending and resumed),
            "selected_candidate_count": len(candidates),
            "active_candidate_count": len(round_proposal.active),
            "archive_candidate_count": len(round_proposal.archive),
            "measured_new_candidate_count": len(measured_new),
            "duration_s": time.monotonic() - round_start + sum(event.duration_s for event in proposal_events),
            "elapsed_s": controller.elapsed_s(),
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
        controller.transition("completed", round_index=round_index + 1)
        _write_checkpoint(
            store,
            record=record,
            controller=controller,
            round_seed=None,
            candidate_hashes=(),
        )
        print(json.dumps(round_record, sort_keys=True), flush=True)

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

    controller.transition("confirmation")
    confirmation_start = time.monotonic()
    hot_records = hot_confirm_topk(
        db_path=store.db_path,
        environment_compatibility_tag=profile.environment_compatibility_tag,
        output_dir=store.root / "hot_loop_top8",
        runner_bin=Path(configuration.runner_bin),
        shape_id=shape.id,
        problem_type_hash=profile.problem_type_hash,
        screening_protocol=protocol,
        hot_protocol=configuration.hot_protocol,
        top_k=configuration.hot_top_k,
        admission_deadline=controller.admission_deadline,
        runner_timeout_s=configuration.runner_timeout_s,
    )
    controller.record_phase_time("confirmation", time.monotonic() - confirmation_start)
    controller.transition("finished")
    record.update(
        {
            "elapsed_s": controller.elapsed_s(),
            "budget_overrun_s": controller.overrun_s(),
            "screening_leader": _leader(
                db,
                shape_id=shape.id,
                problem_type_hash=profile.problem_type_hash,
                protocol_hash=protocol_hash,
                min_samples=configuration.min_samples,
            ),
            "hot_leader": hot_records[0] if hot_records else None,
            "hot_confirmed": len(hot_records),
            "controller": controller.summary(),
        }
    )
    store.write_summary(record)
    _write_checkpoint(
        store,
        record=record,
        controller=controller,
        round_seed=None,
        candidate_hashes=(),
    )
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0
