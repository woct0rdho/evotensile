import math
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypedDict, cast

from evotensile.campaign.workload import ResolvedWorkloadPayload, ResolvedWorkloadWeights

COST_PHASES = (
    "proposal",
    "preparation",
    "validation",
    "probe",
    "screening",
    "repair",
    "stabilization",
    "confirmation",
)

PairKey = tuple[str, str]


class IncumbentPayload(TypedDict):
    candidate_hash: str
    performance: float


class GridMetricsPayload(TypedDict):
    per_shape_log_regret: dict[str, float | None]
    resolved_shapes: int
    unresolved_shapes: int
    mean_log_regret: float | None
    weighted_mean_log_regret: float | None
    median_log_regret: float | None
    p90_log_regret: float | None
    p95_log_regret: float | None
    worst_log_regret: float | None


class CampaignSummaryPayload(TypedDict, total=False):
    shape_ids: list[str]
    phase: str
    round_index: int
    time_budget_s: float
    elapsed_s: float
    budget_overrun_s: float
    reserves_s: dict[str, float]
    queried_pairs: int
    known_pairs: int
    unknown_pairs: int
    disclosed_pairs: int
    resolved_shapes: int
    unresolved_shape_ids: list[str]
    prepared_candidates: int
    candidate_coverage: dict[str, int]
    prepared_artifact_coverage: dict[str, int]
    clustering: dict[str, object] | None
    workload: ResolvedWorkloadPayload
    active_round: dict[str, object] | None
    phase_time_s: dict[str, float]
    grid_metrics: GridMetricsPayload


class ControllerCheckpointPayload(TypedDict):
    shape_ids: list[str]
    time_budget_s: float
    elapsed_s: float
    phase: str
    round_index: int
    reserves_s: dict[str, float]
    queried_pairs: list[list[str]]
    known_pairs: list[list[str]]
    unknown_pairs: list[list[str]]
    disclosed_pairs: list[list[str]]
    incumbents: dict[str, IncumbentPayload]
    prepared_artifact_shapes: dict[str, list[str]]
    clustering: dict[str, object] | None
    workload: ResolvedWorkloadPayload | None
    active_round: dict[str, object] | None
    phase_time_s: dict[str, float]
    trace: list[dict[str, object]]


@dataclass(frozen=True)
class Incumbent:
    candidate_hash: str
    performance: float

    def __post_init__(self) -> None:
        if not self.candidate_hash:
            raise ValueError("incumbent candidate hash must be non-empty")
        if not math.isfinite(self.performance) or self.performance <= 0.0:
            raise ValueError("incumbent performance must be finite and positive")

    def to_dict(self) -> IncumbentPayload:
        return {
            "candidate_hash": self.candidate_hash,
            "performance": self.performance,
        }


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    reason: str
    remaining_s: float
    predicted_duration_s: float
    reserve_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "admitted": self.admitted,
            "reason": self.reason,
            "remaining_s": self.remaining_s,
            "predicted_duration_s": self.predicted_duration_s,
            "reserve_s": self.reserve_s,
        }


@dataclass(frozen=True)
class SoftAdmissionBudget:
    time_budget_s: float
    session_started_at: float
    prior_elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_budget_s) or self.time_budget_s <= 0.0:
            raise ValueError("soft time budget must be finite and positive")
        if not math.isfinite(self.session_started_at):
            raise ValueError("session start must be finite")
        if not math.isfinite(self.prior_elapsed_s) or self.prior_elapsed_s < 0.0:
            raise ValueError("prior elapsed time must be finite and nonnegative")

    @property
    def admission_deadline(self) -> float:
        return self.session_started_at + max(0.0, self.time_budget_s - self.prior_elapsed_s)

    def elapsed_s(self, *, now: float | None = None) -> float:
        current = time.monotonic() if now is None else now
        if not math.isfinite(current):
            raise ValueError("current time must be finite")
        return self.prior_elapsed_s + max(0.0, current - self.session_started_at)

    def remaining_s(self, *, now: float | None = None) -> float:
        current = time.monotonic() if now is None else now
        if not math.isfinite(current):
            raise ValueError("current time must be finite")
        return max(0.0, self.admission_deadline - current)

    def overrun_s(self, *, now: float | None = None) -> float:
        return max(0.0, self.elapsed_s(now=now) - self.time_budget_s)

    def decide(
        self,
        *,
        predicted_duration_s: float = 0.0,
        reserve_s: float = 0.0,
        now: float | None = None,
    ) -> AdmissionDecision:
        if not math.isfinite(predicted_duration_s) or predicted_duration_s < 0.0:
            raise ValueError("predicted duration must be finite and nonnegative")
        if not math.isfinite(reserve_s) or reserve_s < 0.0:
            raise ValueError("admission reserve must be finite and nonnegative")
        remaining = self.remaining_s(now=now)
        if remaining <= 0.0:
            return AdmissionDecision(False, "soft_deadline", remaining, predicted_duration_s, reserve_s)
        if predicted_duration_s + reserve_s > remaining:
            return AdmissionDecision(
                False,
                "insufficient_predicted_budget",
                remaining,
                predicted_duration_s,
                reserve_s,
            )
        return AdmissionDecision(True, "admitted", remaining, predicted_duration_s, reserve_s)


@dataclass(frozen=True)
class GridMetrics:
    per_shape_log_regret: dict[str, float | None]
    resolved_shapes: int
    unresolved_shapes: int
    mean_log_regret: float | None
    weighted_mean_log_regret: float | None
    median_log_regret: float | None
    p90_log_regret: float | None
    p95_log_regret: float | None
    worst_log_regret: float | None

    def to_dict(self) -> GridMetricsPayload:
        return {
            "per_shape_log_regret": self.per_shape_log_regret,
            "resolved_shapes": self.resolved_shapes,
            "unresolved_shapes": self.unresolved_shapes,
            "mean_log_regret": self.mean_log_regret,
            "weighted_mean_log_regret": self.weighted_mean_log_regret,
            "median_log_regret": self.median_log_regret,
            "p90_log_regret": self.p90_log_regret,
            "p95_log_regret": self.p95_log_regret,
            "worst_log_regret": self.worst_log_regret,
        }


def _pair_from_value(value: object, *, field_name: str) -> PairKey:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{field_name} entries must be shape/candidate pairs")
    shape_id, candidate_hash = (str(item) for item in value)
    if not shape_id or not candidate_hash:
        raise ValueError(f"{field_name} entries must be non-empty")
    return shape_id, candidate_hash


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def estimate_admission_duration_s(
    observations: Sequence[tuple[float, int]],
    *,
    expected_units: int,
    minimum_s: float = 20.0,
    multiplicative_margin: float = 1.15,
    fixed_overhead_s: float = 5.0,
    history_limit: int = 6,
    default_s: float = 30.0,
) -> float:
    if expected_units < 0:
        raise ValueError("expected admission units must be nonnegative")
    if minimum_s < 0.0 or multiplicative_margin < 0.0 or fixed_overhead_s < 0.0 or default_s < 0.0:
        raise ValueError("admission estimate parameters must be nonnegative")
    usable = [
        duration_s / units for duration_s, units in observations[-history_limit:] if duration_s > 0.0 and units > 0
    ]
    if not usable:
        return max(minimum_s, default_s)
    median_per_unit = statistics.median(usable)
    robust_margin = statistics.median(abs(value - median_per_unit) for value in usable)
    estimate = (median_per_unit + robust_margin) * max(1, expected_units)
    return max(minimum_s, estimate * multiplicative_margin + fixed_overhead_s)


@dataclass
class CampaignControllerState:
    shape_ids: tuple[str, ...]
    time_budget_s: float
    session_started_at: float
    prior_elapsed_s: float = 0.0
    phase: str = "search"
    round_index: int = 0
    reserves_s: dict[str, float] = field(default_factory=dict)
    queried_pairs: list[PairKey] = field(default_factory=list)
    known_pairs: set[PairKey] = field(default_factory=set)
    unknown_pairs: set[PairKey] = field(default_factory=set)
    disclosed_pairs: set[PairKey] = field(default_factory=set)
    incumbents: dict[str, Incumbent] = field(default_factory=dict)
    prepared_artifact_shapes: dict[str, set[str]] = field(default_factory=dict)
    clustering: dict[str, object] | None = None
    workload: ResolvedWorkloadPayload | None = None
    active_round: dict[str, object] | None = None
    phase_time_s: dict[str, float] = field(default_factory=dict)
    trace: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.shape_ids or len(set(self.shape_ids)) != len(self.shape_ids):
            raise ValueError("controller shapes must be non-empty and unique")
        if any(not shape_id for shape_id in self.shape_ids):
            raise ValueError("controller shape IDs must be non-empty")
        SoftAdmissionBudget(self.time_budget_s, self.session_started_at, self.prior_elapsed_s)
        if self.round_index < 0:
            raise ValueError("controller round index must be nonnegative")
        shape_id_set = set(self.shape_ids)
        if not set(self.incumbents).issubset(shape_id_set):
            raise ValueError("controller incumbent references an unregistered shape")
        for pair in [*self.queried_pairs, *self.known_pairs, *self.unknown_pairs, *self.disclosed_pairs]:
            if pair[0] not in shape_id_set:
                raise ValueError("controller pair references an unregistered shape")
        if set(self.known_pairs) & set(self.unknown_pairs):
            raise ValueError("controller pairs cannot be both known and unknown")
        if set(self.disclosed_pairs) - set(self.known_pairs):
            raise ValueError("only known queried pairs can be disclosed")
        if set(self.known_pairs | self.unknown_pairs) - set(self.queried_pairs):
            raise ValueError("known and unknown pairs must be queried")
        if self.clustering is not None:
            self._validate_clustering(self.clustering)
        if self.workload is not None:
            resolved_workload = ResolvedWorkloadWeights.from_dict(self.workload)
            if resolved_workload.shape_ids != self.shape_ids:
                raise ValueError("controller workload must match the ordered registered shape set")
        if self.active_round is not None and not isinstance(self.active_round, dict):
            raise ValueError("controller active round must be a dictionary")

    @property
    def resolved_workload(self) -> ResolvedWorkloadWeights:
        if self.workload is None:
            return ResolvedWorkloadWeights.uniform(self.shape_ids)
        return ResolvedWorkloadWeights.from_dict(self.workload)

    @property
    def shape_weights(self) -> dict[str, float]:
        return dict(self.resolved_workload.weights)

    @property
    def budget(self) -> SoftAdmissionBudget:
        return SoftAdmissionBudget(self.time_budget_s, self.session_started_at, self.prior_elapsed_s)

    @property
    def admission_deadline(self) -> float:
        return self.budget.admission_deadline

    def elapsed_s(self, *, now: float | None = None) -> float:
        return self.budget.elapsed_s(now=now)

    def overrun_s(self, *, now: float | None = None) -> float:
        return self.budget.overrun_s(now=now)

    def decide_admission(
        self,
        *,
        predicted_duration_s: float = 0.0,
        reserve_s: float = 0.0,
        now: float | None = None,
    ) -> AdmissionDecision:
        decision = self.budget.decide(
            predicted_duration_s=predicted_duration_s,
            reserve_s=reserve_s,
            now=now,
        )
        self.append_trace("admission", decision.to_dict())
        return decision

    def transition(self, phase: str, *, round_index: int | None = None) -> None:
        if not phase:
            raise ValueError("controller phase must be non-empty")
        if round_index is not None:
            if round_index < 0:
                raise ValueError("controller round index must be nonnegative")
            self.round_index = round_index
        self.phase = phase
        self.append_trace("transition", {"phase": phase, "round_index": self.round_index})

    def set_reserve(self, name: str, duration_s: float) -> None:
        if not name:
            raise ValueError("reserve name must be non-empty")
        if not math.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError("reserve duration must be finite and nonnegative")
        self.reserves_s[name] = duration_s

    def record_query(self, shape_id: str, candidate_hash: str, *, known: bool) -> bool:
        pair = self._pair(shape_id, candidate_hash)
        if pair in self.known_pairs:
            if not known:
                raise ValueError("known pair cannot later become unknown")
            return False
        if pair in self.unknown_pairs:
            if known:
                raise ValueError("unknown pair cannot become known without a new evidence source")
            return False
        self.queried_pairs.append(pair)
        (self.known_pairs if known else self.unknown_pairs).add(pair)
        return True

    def disclose(self, shape_id: str, candidate_hash: str, *, performance: float | None = None) -> bool:
        pair = self._pair(shape_id, candidate_hash)
        if pair not in self.known_pairs:
            raise ValueError("pair must be queried and known before disclosure")
        if performance is not None and (not math.isfinite(performance) or performance <= 0.0):
            raise ValueError("disclosed performance must be finite and positive")
        already_disclosed = pair in self.disclosed_pairs
        self.disclosed_pairs.add(pair)
        if performance is not None:
            incumbent = self.incumbents.get(shape_id)
            if incumbent is None or performance > incumbent.performance:
                self.incumbents[shape_id] = Incumbent(candidate_hash, performance)
        return not already_disclosed

    def set_clustering(self, clustering: Mapping[str, object]) -> None:
        payload = {str(key): value for key, value in clustering.items()}
        self._validate_clustering(payload)
        self.clustering = payload
        self.append_trace(
            "clustering",
            {
                "clusters": len(cast(list[object], payload["clusters"])),
                "shape_ids": list(self.shape_ids),
            },
        )

    def set_workload(self, workload: ResolvedWorkloadWeights) -> None:
        if workload.shape_ids != self.shape_ids:
            raise ValueError("controller workload must match the ordered registered shape set")
        self.workload = workload.to_dict()
        self.append_trace(
            "workload",
            {
                "mode": workload.mode,
                "weights": dict(sorted(workload.weights.items())),
            },
        )

    def set_active_round(self, payload: Mapping[str, object]) -> None:
        self.active_round = {str(key): value for key, value in payload.items()}

    def clear_active_round(self) -> None:
        self.active_round = None

    def record_prepared(self, candidate_hash: str, shape_ids: Sequence[str]) -> set[str]:
        if not candidate_hash:
            raise ValueError("prepared candidate hash must be non-empty")
        unique_shape_ids = set(shape_ids)
        if not unique_shape_ids.issubset(set(self.shape_ids)):
            raise ValueError("prepared artifact scope references an unregistered shape")
        prepared = self.prepared_artifact_shapes.setdefault(candidate_hash, set())
        new_shape_ids = unique_shape_ids - prepared
        prepared.update(new_shape_ids)
        return new_shape_ids

    def record_phase_time(self, phase: str, duration_s: float) -> None:
        if phase not in COST_PHASES:
            raise ValueError(f"unsupported controller cost phase: {phase}")
        if not math.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError("phase duration must be finite and nonnegative")
        self.phase_time_s[phase] = self.phase_time_s.get(phase, 0.0) + duration_s

    def append_trace(self, event: str, payload: Mapping[str, object] | None = None) -> None:
        if not event:
            raise ValueError("trace event must be non-empty")
        self.trace.append({"event": event, **dict(payload or {})})

    def grid_metrics(
        self,
        oracle_best_by_shape: Mapping[str, float],
        *,
        weights: Mapping[str, float] | None = None,
    ) -> GridMetrics:
        shape_id_set = set(self.shape_ids)
        if set(oracle_best_by_shape) != shape_id_set:
            raise ValueError("oracle best values must cover the exact controller shape set")
        active_weights = self.shape_weights
        if weights is not None:
            if set(weights) != shape_id_set:
                raise ValueError("grid weights must cover the exact controller shape set")
            active_weights = {shape_id: float(weights[shape_id]) for shape_id in self.shape_ids}
            if any(not math.isfinite(value) or value < 0.0 for value in active_weights.values()):
                raise ValueError("grid weights must be finite and nonnegative")
        regrets: dict[str, float | None] = {}
        resolved_regrets: list[float] = []
        weighted_terms: list[tuple[float, float]] = []
        for shape_id in self.shape_ids:
            oracle_best = float(oracle_best_by_shape[shape_id])
            if not math.isfinite(oracle_best) or oracle_best <= 0.0:
                raise ValueError("oracle best performance must be finite and positive")
            incumbent = self.incumbents.get(shape_id)
            if incumbent is None:
                regrets[shape_id] = None
                continue
            regret = max(0.0, math.log(oracle_best / incumbent.performance))
            regrets[shape_id] = regret
            resolved_regrets.append(regret)
            weighted_terms.append((regret, active_weights[shape_id]))
        total_weight = sum(weight for _, weight in weighted_terms)
        weighted_mean = (
            sum(regret * weight for regret, weight in weighted_terms) / total_weight if total_weight > 0.0 else None
        )
        return GridMetrics(
            per_shape_log_regret=regrets,
            resolved_shapes=len(resolved_regrets),
            unresolved_shapes=len(self.shape_ids) - len(resolved_regrets),
            mean_log_regret=statistics.fmean(resolved_regrets) if resolved_regrets else None,
            weighted_mean_log_regret=weighted_mean,
            median_log_regret=statistics.median(resolved_regrets) if resolved_regrets else None,
            p90_log_regret=_percentile(resolved_regrets, 0.90),
            p95_log_regret=_percentile(resolved_regrets, 0.95),
            worst_log_regret=max(resolved_regrets) if resolved_regrets else None,
        )

    def summary(
        self,
        *,
        oracle_best_by_shape: Mapping[str, float] | None = None,
        weights: Mapping[str, float] | None = None,
        now: float | None = None,
    ) -> CampaignSummaryPayload:
        queried_shapes_by_candidate: dict[str, set[str]] = {}
        for shape_id, candidate_hash in self.queried_pairs:
            queried_shapes_by_candidate.setdefault(candidate_hash, set()).add(shape_id)
        payload: CampaignSummaryPayload = {
            "shape_ids": list(self.shape_ids),
            "phase": self.phase,
            "round_index": self.round_index,
            "time_budget_s": self.time_budget_s,
            "elapsed_s": self.elapsed_s(now=now),
            "budget_overrun_s": self.overrun_s(now=now),
            "reserves_s": dict(sorted(self.reserves_s.items())),
            "queried_pairs": len(self.queried_pairs),
            "known_pairs": len(self.known_pairs),
            "unknown_pairs": len(self.unknown_pairs),
            "disclosed_pairs": len(self.disclosed_pairs),
            "resolved_shapes": len(self.incumbents),
            "unresolved_shape_ids": [shape_id for shape_id in self.shape_ids if shape_id not in self.incumbents],
            "prepared_candidates": len(self.prepared_artifact_shapes),
            "candidate_coverage": {
                candidate_hash: len(shape_ids)
                for candidate_hash, shape_ids in sorted(queried_shapes_by_candidate.items())
            },
            "prepared_artifact_coverage": {
                candidate_hash: len(shape_ids)
                for candidate_hash, shape_ids in sorted(self.prepared_artifact_shapes.items())
            },
            "clustering": self.clustering,
            "workload": self.resolved_workload.to_dict(),
            "active_round": self.active_round,
            "phase_time_s": {phase: self.phase_time_s.get(phase, 0.0) for phase in COST_PHASES},
        }
        if oracle_best_by_shape is not None:
            payload["grid_metrics"] = self.grid_metrics(oracle_best_by_shape, weights=weights).to_dict()
        return payload

    def to_checkpoint(self, *, now: float | None = None) -> ControllerCheckpointPayload:
        return {
            "shape_ids": list(self.shape_ids),
            "time_budget_s": self.time_budget_s,
            "elapsed_s": self.elapsed_s(now=now),
            "phase": self.phase,
            "round_index": self.round_index,
            "reserves_s": dict(sorted(self.reserves_s.items())),
            "queried_pairs": [list(pair) for pair in self.queried_pairs],
            "known_pairs": [list(pair) for pair in sorted(self.known_pairs)],
            "unknown_pairs": [list(pair) for pair in sorted(self.unknown_pairs)],
            "disclosed_pairs": [list(pair) for pair in sorted(self.disclosed_pairs)],
            "incumbents": {shape_id: incumbent.to_dict() for shape_id, incumbent in sorted(self.incumbents.items())},
            "prepared_artifact_shapes": {
                candidate_hash: sorted(shape_ids)
                for candidate_hash, shape_ids in sorted(self.prepared_artifact_shapes.items())
            },
            "clustering": self.clustering,
            "workload": self.workload,
            "active_round": self.active_round,
            "phase_time_s": dict(sorted(self.phase_time_s.items())),
            "trace": list(self.trace),
        }

    @classmethod
    def from_checkpoint(
        cls,
        payload: ControllerCheckpointPayload,
        *,
        session_started_at: float,
    ) -> "CampaignControllerState":
        incumbents = {
            shape_id: Incumbent(
                candidate_hash=value["candidate_hash"],
                performance=float(value["performance"]),
            )
            for shape_id, value in payload["incumbents"].items()
        }
        return cls(
            shape_ids=tuple(payload["shape_ids"]),
            time_budget_s=float(payload["time_budget_s"]),
            session_started_at=session_started_at,
            prior_elapsed_s=float(payload.get("elapsed_s", 0.0)),
            phase=str(payload.get("phase", "search")),
            round_index=int(payload.get("round_index", 0)),
            reserves_s=dict(payload["reserves_s"]),
            queried_pairs=[_pair_from_value(value, field_name="queried_pairs") for value in payload["queried_pairs"]],
            known_pairs={_pair_from_value(value, field_name="known_pairs") for value in payload["known_pairs"]},
            unknown_pairs={_pair_from_value(value, field_name="unknown_pairs") for value in payload["unknown_pairs"]},
            disclosed_pairs={
                _pair_from_value(value, field_name="disclosed_pairs") for value in payload["disclosed_pairs"]
            },
            incumbents=incumbents,
            prepared_artifact_shapes={
                str(candidate_hash): {str(shape_id) for shape_id in shape_ids}
                for candidate_hash, shape_ids in payload["prepared_artifact_shapes"].items()
            },
            clustering=payload["clustering"],
            workload=payload["workload"],
            active_round=payload["active_round"],
            phase_time_s=dict(payload["phase_time_s"]),
            trace=list(payload["trace"]),
        )

    def _validate_clustering(self, clustering: Mapping[str, object]) -> None:
        clustering_shape_ids = tuple(cast(list[str], clustering["shape_ids"]))
        if set(clustering_shape_ids) != set(self.shape_ids) or len(clustering_shape_ids) != len(self.shape_ids):
            raise ValueError("controller clustering must cover the exact registered shape set")
        clusters = cast(list[dict[str, object]], clustering["clusters"])
        represented: set[str] = set()
        medoids: set[str] = set()
        for cluster in clusters:
            medoid = str(cluster.get("medoid_shape_id", ""))
            members = set(cast(list[str], cluster["shape_ids"]))
            if not medoid or medoid not in members:
                raise ValueError("controller cluster medoid must be one of its members")
            if represented & members:
                raise ValueError("controller clusters must not overlap")
            represented.update(members)
            medoids.add(medoid)
        if represented != set(self.shape_ids) or len(medoids) != len(clusters):
            raise ValueError("controller clusters must partition the registered shapes")

    def _pair(self, shape_id: str, candidate_hash: str) -> PairKey:
        if shape_id not in self.shape_ids:
            raise ValueError(f"shape is not registered in controller state: {shape_id}")
        if not candidate_hash:
            raise ValueError("candidate hash must be non-empty")
        return shape_id, candidate_hash
