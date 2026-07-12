import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, TypedDict, cast

from evotensile.campaign.acquisition import AcquisitionPlan
from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.evaluator import EvaluationResult, PairEvaluator
from evotensile.candidate import Candidate, Shape
from evotensile.scheduling.models import EvidenceStage, PairRequest

ROUND_PHASES = ("broad", "promotion", "repair", "stabilization", "confirmation")


class PendingRequestPayload(TypedDict):
    shape_id: str
    candidate_hash: str
    evidence_stage: str
    min_samples: int
    priority: float


class PendingRoundWavePayload(TypedDict):
    phase: str
    predicted_cost_s: float
    requests: list[PendingRequestPayload]
    artifact_shapes_by_candidate: dict[str, list[str]]
    plan_report: dict[str, object]


class StagedRoundStatePayload(TypedDict):
    round_id: str
    configuration_hash: str
    model_identity: str
    phase_index: int
    phase: str | None
    wave_index: int
    pending: PendingRoundWavePayload | None
    admissions: list[dict[str, object]]
    completed_waves: list[dict[str, object]]
    stop_reason: str | None
    completed: bool


@dataclass(frozen=True)
class StagedRoundConfiguration:
    phase_fractions: tuple[tuple[str, float], ...] = (
        ("broad", 0.25),
        ("promotion", 0.35),
        ("repair", 0.15),
        ("stabilization", 0.10),
        ("confirmation", 0.15),
    )
    no_new_preparation_guard_s: float = 30.0

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.phase_fractions)
        if names != ROUND_PHASES:
            raise ValueError(f"staged round phases must be exactly {ROUND_PHASES}")
        fractions = tuple(float(value) for _, value in self.phase_fractions)
        if any(not math.isfinite(value) or value < 0.0 for value in fractions):
            raise ValueError("staged round phase fractions must be finite and nonnegative")
        if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("staged round phase fractions must sum to one")
        if not math.isfinite(self.no_new_preparation_guard_s) or self.no_new_preparation_guard_s < 0.0:
            raise ValueError("staged round preparation guard must be finite and nonnegative")

    @property
    def fraction_by_phase(self) -> dict[str, float]:
        return dict(self.phase_fractions)

    def cumulative_fraction(self, phase: str) -> float:
        cumulative = 0.0
        for name, fraction in self.phase_fractions:
            cumulative += fraction
            if name == phase:
                return cumulative
        raise ValueError(f"unknown staged round phase: {phase}")

    def to_dict(self) -> dict[str, object]:
        return {
            "phase_fractions": [[name, value] for name, value in self.phase_fractions],
            "no_new_preparation_guard_s": self.no_new_preparation_guard_s,
        }

    @property
    def identity_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class PendingRoundWave:
    phase: str
    predicted_cost_s: float
    requests: tuple[PendingRequestPayload, ...]
    artifact_shapes_by_candidate: dict[str, tuple[str, ...]]
    plan_report: dict[str, object]

    @classmethod
    def from_plan(cls, phase: str, plan: AcquisitionPlan) -> "PendingRoundWave":
        return cls(
            phase=phase,
            predicted_cost_s=plan.predicted_cost_s,
            requests=tuple(
                {
                    "shape_id": request.shape.id,
                    "candidate_hash": request.candidate.hash,
                    "evidence_stage": request.evidence_stage.value,
                    "min_samples": request.min_samples,
                    "priority": request.priority,
                }
                for request in plan.timing_requests
            ),
            artifact_shapes_by_candidate={
                candidate_hash: tuple(shape.id for shape in shapes)
                for candidate_hash, shapes in plan.artifact_shapes_by_candidate.items()
            },
            plan_report=plan.to_dict(),
        )

    def restore(
        self,
        *,
        candidates: Mapping[str, Candidate],
        shapes: Mapping[str, Shape],
    ) -> tuple[tuple[PairRequest, ...], dict[str, tuple[Shape, ...]]]:
        requests = []
        for payload in self.requests:
            shape_id = str(payload["shape_id"])
            candidate_hash = str(payload["candidate_hash"])
            requests.append(
                PairRequest(
                    candidates[candidate_hash],
                    shapes[shape_id],
                    evidence_stage=EvidenceStage(str(payload["evidence_stage"])),
                    min_samples=int(payload["min_samples"]),
                    priority=float(payload["priority"]),
                )
            )
        artifact_shapes = {
            candidate_hash: tuple(shapes[shape_id] for shape_id in shape_ids)
            for candidate_hash, shape_ids in self.artifact_shapes_by_candidate.items()
        }
        return tuple(requests), artifact_shapes

    def to_dict(self) -> PendingRoundWavePayload:
        return {
            "phase": self.phase,
            "predicted_cost_s": self.predicted_cost_s,
            "requests": list(self.requests),
            "artifact_shapes_by_candidate": {
                candidate_hash: list(shape_ids)
                for candidate_hash, shape_ids in sorted(self.artifact_shapes_by_candidate.items())
            },
            "plan_report": self.plan_report,
        }

    @classmethod
    def from_dict(cls, payload: PendingRoundWavePayload) -> "PendingRoundWave":
        return cls(
            phase=payload["phase"],
            predicted_cost_s=float(payload["predicted_cost_s"]),
            requests=tuple(payload["requests"]),
            artifact_shapes_by_candidate={
                str(candidate_hash): tuple(str(shape_id) for shape_id in shape_ids)
                for candidate_hash, shape_ids in payload["artifact_shapes_by_candidate"].items()
            },
            plan_report=dict(payload["plan_report"]),
        )


@dataclass
class StagedRoundState:
    round_id: str
    configuration_hash: str
    model_identity: str
    phase_index: int = 0
    wave_index: int = 0
    pending: PendingRoundWave | None = None
    admissions: list[dict[str, object]] = field(default_factory=list)
    completed_waves: list[dict[str, object]] = field(default_factory=list)
    stop_reason: str | None = None

    @property
    def completed(self) -> bool:
        return self.phase_index >= len(ROUND_PHASES) or self.stop_reason in {"soft_deadline", "overrun"}

    @property
    def phase(self) -> str | None:
        return None if self.phase_index >= len(ROUND_PHASES) else ROUND_PHASES[self.phase_index]

    def to_dict(self) -> StagedRoundStatePayload:
        return {
            "round_id": self.round_id,
            "configuration_hash": self.configuration_hash,
            "model_identity": self.model_identity,
            "phase_index": self.phase_index,
            "phase": self.phase,
            "wave_index": self.wave_index,
            "pending": None if self.pending is None else self.pending.to_dict(),
            "admissions": list(self.admissions),
            "completed_waves": list(self.completed_waves),
            "stop_reason": self.stop_reason,
            "completed": self.completed,
        }

    @classmethod
    def from_dict(cls, payload: StagedRoundStatePayload) -> "StagedRoundState":
        pending_value = payload["pending"]
        return cls(
            round_id=str(payload["round_id"]),
            configuration_hash=str(payload["configuration_hash"]),
            model_identity=str(payload["model_identity"]),
            phase_index=int(payload.get("phase_index", 0)),
            wave_index=int(payload.get("wave_index", 0)),
            pending=None if pending_value is None else PendingRoundWave.from_dict(pending_value),
            admissions=list(payload["admissions"]),
            completed_waves=list(payload["completed_waves"]),
            stop_reason=None if payload.get("stop_reason") is None else str(payload["stop_reason"]),
        )


class StagedRoundPlanner(Protocol):
    def plan_wave(
        self,
        phase: str,
        controller: CampaignControllerState,
    ) -> AcquisitionPlan | None: ...

    def observe(self, phase: str, result: EvaluationResult) -> None: ...


@dataclass(frozen=True)
class StagedRoundResult:
    state: StagedRoundState
    evaluation_results: tuple[EvaluationResult, ...]


CheckpointCallback = Callable[[CampaignControllerState], None]
NowFunction = Callable[[], float]
ResultTimeCallback = Callable[[EvaluationResult], None]


def run_staged_round(
    controller: CampaignControllerState,
    *,
    round_id: str,
    configuration: StagedRoundConfiguration,
    model_identity: str,
    candidates: Mapping[str, Candidate],
    shapes: Mapping[str, Shape],
    planner: StagedRoundPlanner,
    evaluator: PairEvaluator,
    checkpoint: CheckpointCallback | None = None,
    now: NowFunction = time.monotonic,
    charge_result_time: ResultTimeCallback | None = None,
) -> StagedRoundResult:
    if set(shapes) != set(controller.shape_ids):
        raise ValueError("staged round shapes must match controller identity")
    state = _restore_or_create_state(
        controller,
        round_id=round_id,
        configuration=configuration,
        model_identity=model_identity,
    )
    if state.completed and state.pending is None:
        return StagedRoundResult(state, ())
    _set_round_reserves(controller, configuration)
    results = []
    while not state.completed:
        phase = state.phase
        assert phase is not None
        controller.transition(phase)
        if state.pending is not None:
            result = _execute_pending(
                controller,
                state,
                candidates=candidates,
                shapes=shapes,
                planner=planner,
                evaluator=evaluator,
                charge_result_time=charge_result_time,
            )
            results.append(result)
            _persist(controller, state, checkpoint)
            if controller.overrun_s(now=now()) > 0.0:
                state.stop_reason = "overrun"
                _persist(controller, state, checkpoint)
                break
            continue
        plan = planner.plan_wave(phase, controller)
        if plan is None or not plan.timing_requests:
            state.phase_index += 1
            _persist(controller, state, checkpoint)
            continue
        decision = _admission_decision(
            controller,
            state,
            phase=phase,
            plan=plan,
            configuration=configuration,
            now=now(),
        )
        state.admissions.append(decision)
        controller.append_trace("round_admission", decision)
        if not bool(decision["admitted"]):
            if decision["reason"] == "soft_deadline":
                state.stop_reason = "soft_deadline"
                _persist(controller, state, checkpoint)
                break
            state.phase_index += 1
            _persist(controller, state, checkpoint)
            continue
        state.pending = PendingRoundWave.from_plan(phase, plan)
        _persist(controller, state, checkpoint)
        result = _execute_pending(
            controller,
            state,
            candidates=candidates,
            shapes=shapes,
            planner=planner,
            evaluator=evaluator,
            charge_result_time=charge_result_time,
        )
        results.append(result)
        _persist(controller, state, checkpoint)
        if controller.overrun_s(now=now()) > 0.0:
            state.stop_reason = "overrun"
            _persist(controller, state, checkpoint)
            break
    if state.phase_index >= len(ROUND_PHASES) and state.stop_reason is None:
        state.stop_reason = "completed"
    _persist(controller, state, checkpoint)
    return StagedRoundResult(state, tuple(results))


def _restore_or_create_state(
    controller: CampaignControllerState,
    *,
    round_id: str,
    configuration: StagedRoundConfiguration,
    model_identity: str,
) -> StagedRoundState:
    if controller.active_round is None:
        state = StagedRoundState(round_id, configuration.identity_hash, model_identity)
        controller.set_active_round(state.to_dict())
        return state
    state = StagedRoundState.from_dict(cast(StagedRoundStatePayload, controller.active_round))
    if (
        state.round_id != round_id
        or state.configuration_hash != configuration.identity_hash
        or state.model_identity != model_identity
    ):
        raise ValueError("staged round checkpoint identity mismatch")
    return state


def _set_round_reserves(
    controller: CampaignControllerState,
    configuration: StagedRoundConfiguration,
) -> None:
    fractions = configuration.fraction_by_phase
    for phase in ("repair", "stabilization", "confirmation"):
        controller.set_reserve(phase, controller.time_budget_s * fractions[phase])


def _admission_decision(
    controller: CampaignControllerState,
    state: StagedRoundState,
    *,
    phase: str,
    plan: AcquisitionPlan,
    configuration: StagedRoundConfiguration,
    now: float,
) -> dict[str, object]:
    total_remaining = controller.budget.remaining_s(now=now)
    phase_deadline = controller.session_started_at + (
        controller.time_budget_s * configuration.cumulative_fraction(phase)
    )
    phase_remaining = max(0.0, phase_deadline - now)
    reason = "admitted"
    admitted = True
    if total_remaining <= 0.0:
        admitted = False
        reason = "soft_deadline"
    elif plan.preparation_order and total_remaining <= configuration.no_new_preparation_guard_s:
        admitted = False
        reason = "no_new_preparation_guard"
    elif plan.predicted_cost_s > phase_remaining:
        admitted = False
        reason = "phase_deadline"
    else:
        total_decision = controller.decide_admission(
            predicted_duration_s=plan.predicted_cost_s,
            now=now,
        )
        admitted = total_decision.admitted
        reason = total_decision.reason
    return {
        "phase": phase,
        "wave_index": state.wave_index,
        "admitted": admitted,
        "reason": reason,
        "predicted_cost_s": plan.predicted_cost_s,
        "phase_remaining_s": phase_remaining,
        "total_remaining_s": total_remaining,
        "preparation_candidates": list(plan.preparation_order),
        "request_pairs": [list(request.key) for request in plan.timing_requests],
    }


def _execute_pending(
    controller: CampaignControllerState,
    state: StagedRoundState,
    *,
    candidates: Mapping[str, Candidate],
    shapes: Mapping[str, Shape],
    planner: StagedRoundPlanner,
    evaluator: PairEvaluator,
    charge_result_time: ResultTimeCallback | None,
) -> EvaluationResult:
    pending = state.pending
    if pending is None:
        raise ValueError("staged round has no pending wave")
    requests, artifact_shapes = pending.restore(candidates=candidates, shapes=shapes)
    result = evaluator.evaluate(requests, artifact_shapes_by_candidate=artifact_shapes)
    result.apply(controller)
    if charge_result_time is not None:
        charge_result_time(result)
    planner.observe(pending.phase, result)
    state.completed_waves.append(
        {
            "phase": pending.phase,
            "wave_index": state.wave_index,
            "predicted_cost_s": pending.predicted_cost_s,
            "request_pairs": [list(request.key) for request in requests],
            "result_mode": result.mode,
            "known_pairs": result.known_pairs,
            "unknown_pairs": result.unknown_pairs,
        }
    )
    state.wave_index += 1
    state.pending = None
    return result


def _persist(
    controller: CampaignControllerState,
    state: StagedRoundState,
    checkpoint: CheckpointCallback | None,
) -> None:
    controller.set_active_round(state.to_dict())
    if checkpoint is not None:
        checkpoint(controller)
