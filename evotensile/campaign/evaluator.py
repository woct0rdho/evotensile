import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from evotensile.adaptive_retime import AdaptivePolicy, ProbePolicy
from evotensile.artifacts import load_artifact_mappings
from evotensile.campaign.controller import COST_PHASES, CampaignControllerState
from evotensile.candidate import Shape
from evotensile.database import EvoTensileDB
from evotensile.profile import TargetProfile
from evotensile.protocol import BenchmarkProtocol
from evotensile.runner import DEFAULT_TENSILELITE_BIN
from evotensile.scheduler import DEFAULT_COMPILE_THREADS, execute_schedule
from evotensile.scheduling.models import PairRequest, ScheduleResult
from evotensile.scheduling.planning import normalize_pair_requests, requested_candidates, requested_shapes
from evotensile.search.replay import ExactOracleReplayState


@dataclass(frozen=True)
class PairEvaluationOutcome:
    request: PairRequest
    provenance: str
    source_ref: str
    status: str
    known: bool
    disclosed: bool
    samples: int = 0
    performance: float | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.request.key


@dataclass(frozen=True)
class EvaluationResult:
    mode: str
    outcomes: tuple[PairEvaluationOutcome, ...]
    prepared_artifact_shapes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    phase_time_s: dict[str, float] = field(default_factory=dict)
    schedules: tuple[ScheduleResult, ...] = ()

    def __post_init__(self) -> None:
        keys = [outcome.key for outcome in self.outcomes]
        if len(keys) != len(set(keys)):
            raise ValueError("evaluation outcomes must contain unique exact pairs")
        if any(phase not in COST_PHASES for phase in self.phase_time_s):
            raise ValueError("evaluation result contains an unsupported cost phase")

    @property
    def known_pairs(self) -> int:
        return sum(outcome.known for outcome in self.outcomes)

    @property
    def unknown_pairs(self) -> int:
        return len(self.outcomes) - self.known_pairs

    def apply(self, controller: CampaignControllerState) -> None:
        for candidate_hash, shape_ids in self.prepared_artifact_shapes.items():
            controller.record_prepared(candidate_hash, shape_ids)
        for phase, duration_s in self.phase_time_s.items():
            controller.record_phase_time(phase, duration_s)
        for outcome in self.outcomes:
            controller.record_query(
                outcome.request.shape.id,
                outcome.request.candidate.hash,
                known=outcome.known,
            )
            if outcome.known and outcome.disclosed:
                controller.disclose(
                    outcome.request.shape.id,
                    outcome.request.candidate.hash,
                    performance=outcome.performance,
                )
        controller.append_trace(
            "evaluation",
            {
                "mode": self.mode,
                "pairs": len(self.outcomes),
                "known_pairs": self.known_pairs,
                "unknown_pairs": self.unknown_pairs,
            },
        )


class PairEvaluator(Protocol):
    def evaluate(
        self,
        requests: Sequence[PairRequest],
        *,
        artifact_shapes_by_candidate: Mapping[str, Sequence[Shape]] | None = None,
    ) -> EvaluationResult: ...


@dataclass(frozen=True)
class ReplayEvaluator:
    state: ExactOracleReplayState
    prepare_workers: int = 1
    prepare_seconds_per_candidate: float = 0.0

    def evaluate(
        self,
        requests: Sequence[PairRequest],
        *,
        artifact_shapes_by_candidate: Mapping[str, Sequence[Shape]] | None = None,
    ) -> EvaluationResult:
        normalized = normalize_pair_requests(requests)
        candidates = requested_candidates(normalized)
        preparation_s = self.state.prepare_candidates(
            candidates,
            workers=self.prepare_workers,
            seconds_per_candidate=self.prepare_seconds_per_candidate,
        )
        prepared = _resolved_artifact_scopes(normalized, artifact_shapes_by_candidate)
        outcomes = []
        phase_time_s = {"preparation": preparation_s} if preparation_s > 0.0 else {}
        phase_by_stage = {
            "probe": "probe",
            "screening": "screening",
            "stabilization": "stabilization",
            "confirmation": "confirmation",
        }
        protocol = self.state.screening_protocol
        for request in normalized:
            previous_samples = self.state.screening_samples(request.shape, request.candidate.hash)
            query = self.state.query_pair(
                request.shape,
                request.candidate,
                disclose=True,
                samples=request.min_samples,
            )
            record = query.record
            current_samples = self.state.screening_samples(request.shape, request.candidate.hash)
            added_samples = current_samples - previous_samples
            if (
                added_samples > 0
                and record is not None
                and record.screening_gflops is not None
                and record.screening_gflops > 0.0
            ):
                time_us = (
                    2.0
                    * request.shape.m
                    * request.shape.n
                    * request.shape.batch
                    * request.shape.k
                    / (record.screening_gflops * 1e3)
                )
                launches = protocol.num_warmups + (
                    added_samples * protocol.enqueues_per_sync * protocol.syncs_per_benchmark
                )
                duration_s = launches * time_us / 1e6
                self.state.record_pair_time(request.shape, request.candidate.hash, duration_s)
                phase = phase_by_stage[request.evidence_stage.value]
                phase_time_s[phase] = phase_time_s.get(phase, 0.0) + duration_s
            outcomes.append(
                PairEvaluationOutcome(
                    request=request,
                    provenance="replay",
                    source_ref=self.state.source_ref,
                    status="unknown" if record is None else record.status,
                    known=record is not None,
                    disclosed=record is not None,
                    samples=current_samples,
                    performance=None if record is None else record.screening_gflops,
                )
            )
        return EvaluationResult(
            mode="replay",
            outcomes=tuple(outcomes),
            prepared_artifact_shapes=prepared,
            phase_time_s=phase_time_s,
        )


@dataclass(frozen=True)
class RealEvaluatorContext:
    db: EvoTensileDB
    output_root: str | Path
    target_profile: TargetProfile
    protocol: BenchmarkProtocol
    runner_bin: str | Path
    tensilelite_bin: str | Path = DEFAULT_TENSILELITE_BIN
    compile_threads: int | None = DEFAULT_COMPILE_THREADS
    candidate_batch_size: int | None = None
    shape_batch_size: int | None = None
    build_timeout_s: float | None = None
    runner_timeout_s: float | None = None
    prepare_workers: int | None = None
    prepare_wave_batches: int | None = None
    validation_workers: int | None = None
    compile_cache_root: str | Path | None = None
    cost_aware_scheduling: bool = False
    ignore_cache: bool = False
    keep_going: bool = True
    adaptive_policy: AdaptivePolicy | None = None
    probe_policy: ProbePolicy | None = None


@dataclass(frozen=True)
class RealEvaluator:
    context: RealEvaluatorContext
    source_ref: str = "native_schedule"

    def evaluate(
        self,
        requests: Sequence[PairRequest],
        *,
        artifact_shapes_by_candidate: Mapping[str, Sequence[Shape]] | None = None,
    ) -> EvaluationResult:
        normalized = normalize_pair_requests(requests)
        self.context.db.init()
        before = self.context.db.native_run_phase_durations()
        started = time.monotonic()
        schedule = execute_schedule(
            self.context.db,
            requests=normalized,
            artifact_shapes_by_candidate=artifact_shapes_by_candidate,
            output_root=self.context.output_root,
            target_profile=self.context.target_profile,
            protocol=self.context.protocol,
            candidate_batch_size=self.context.candidate_batch_size,
            shape_batch_size=self.context.shape_batch_size,
            tensilelite_bin=self.context.tensilelite_bin,
            compile_threads=self.context.compile_threads,
            keep_going=self.context.keep_going,
            runner_bin=self.context.runner_bin,
            build_timeout_s=self.context.build_timeout_s,
            runner_timeout_s=self.context.runner_timeout_s,
            adaptive_policy=self.context.adaptive_policy,
            probe_policy=self.context.probe_policy,
            prepare_workers=self.context.prepare_workers,
            prepare_wave_batches=self.context.prepare_wave_batches,
            compile_cache_root=self.context.compile_cache_root,
            cost_aware_scheduling=self.context.cost_aware_scheduling,
            validation_workers=self.context.validation_workers,
            ignore_cache=self.context.ignore_cache,
        )
        wall_s = time.monotonic() - started
        after = self.context.db.native_run_phase_durations()
        phase_time_s = _native_phase_delta(before, after)
        evidence_stages = {request.evidence_stage for request in normalized}
        if len(evidence_stages) == 1:
            evidence_stage = next(iter(evidence_stages))
            if evidence_stage.value != "screening" and "screening" in phase_time_s:
                phase_time_s[evidence_stage.value] = phase_time_s.pop("screening")
        if not phase_time_s and wall_s > 0.0:
            phase = next(iter(evidence_stages)).value if len(evidence_stages) == 1 else "screening"
            phase_time_s[phase] = wall_s
        outcomes = _database_outcomes(
            self.context.db,
            normalized,
            profile=self.context.target_profile,
            protocol=self.context.protocol,
            provenance="native",
            source_ref=self.source_ref,
        )
        prepared: dict[str, set[str]] = {}
        for batch in schedule.planned_batches:
            for candidate in batch.artifact_candidates:
                prepared.setdefault(candidate.hash, set()).update(shape.id for shape in batch.artifact_shapes)
        artifacts = load_artifact_mappings(
            self.context.db,
            problem_type_hash=self.context.target_profile.problem_type_hash,
            shape_ids=[shape.id for shape in requested_shapes(normalized)],
            candidate_hashes=[candidate.hash for candidate in requested_candidates(normalized)],
        )
        for shape_id, candidate_hash in artifacts:
            if (shape_id, candidate_hash) in {request.key for request in normalized}:
                prepared.setdefault(candidate_hash, set()).add(shape_id)
        return EvaluationResult(
            mode="real",
            outcomes=tuple(outcomes),
            prepared_artifact_shapes={
                candidate_hash: tuple(sorted(shape_ids)) for candidate_hash, shape_ids in prepared.items()
            },
            phase_time_s=phase_time_s,
            schedules=(schedule,),
        )


@dataclass(frozen=True)
class HybridEvaluator:
    replay: ReplayEvaluator
    real: RealEvaluator

    def __post_init__(self) -> None:
        if self.replay.state.db.path != self.real.context.db.path:
            raise ValueError("hybrid replay and native evidence must share one explicit overlay database")

    def evaluate(
        self,
        requests: Sequence[PairRequest],
        *,
        artifact_shapes_by_candidate: Mapping[str, Sequence[Shape]] | None = None,
    ) -> EvaluationResult:
        normalized = normalize_pair_requests(requests)
        replay_requests = [
            request
            for request in normalized
            if self.replay.state.has_oracle_pair(request.shape, request.candidate.hash)
        ]
        real_requests = [request for request in normalized if request not in replay_requests]
        results = []
        if replay_requests:
            replay_scopes = _subset_artifact_scopes(artifact_shapes_by_candidate, replay_requests)
            results.append(
                self.replay.evaluate(
                    replay_requests,
                    artifact_shapes_by_candidate=replay_scopes,
                )
            )
        if real_requests:
            real_scopes = _subset_artifact_scopes(artifact_shapes_by_candidate, real_requests)
            results.append(
                self.real.evaluate(
                    real_requests,
                    artifact_shapes_by_candidate=real_scopes,
                )
            )
        outcomes_by_key = {outcome.key: outcome for result in results for outcome in result.outcomes}
        prepared: dict[str, set[str]] = {}
        phase_time_s: dict[str, float] = {}
        schedules = []
        for result in results:
            for candidate_hash, shape_ids in result.prepared_artifact_shapes.items():
                prepared.setdefault(candidate_hash, set()).update(shape_ids)
            for phase, duration_s in result.phase_time_s.items():
                phase_time_s[phase] = phase_time_s.get(phase, 0.0) + duration_s
            schedules.extend(result.schedules)
        return EvaluationResult(
            mode="hybrid",
            outcomes=tuple(outcomes_by_key[request.key] for request in normalized),
            prepared_artifact_shapes={
                candidate_hash: tuple(sorted(shape_ids)) for candidate_hash, shape_ids in prepared.items()
            },
            phase_time_s=phase_time_s,
            schedules=tuple(schedules),
        )


def _resolved_artifact_scopes(
    requests: Sequence[PairRequest],
    supplied: Mapping[str, Sequence[Shape]] | None,
) -> dict[str, tuple[str, ...]]:
    requested: dict[str, set[str]] = {}
    for request in requests:
        requested.setdefault(request.candidate.hash, set()).add(request.shape.id)
    scopes = {}
    for candidate_hash, shape_ids in requested.items():
        scope = shape_ids
        if supplied is not None and candidate_hash in supplied:
            scope = {shape.id for shape in supplied[candidate_hash]}
            if not shape_ids.issubset(scope):
                raise ValueError(f"artifact scope does not cover exact requests for {candidate_hash}")
        scopes[candidate_hash] = tuple(sorted(scope))
    return scopes


def _subset_artifact_scopes(
    supplied: Mapping[str, Sequence[Shape]] | None,
    requests: Sequence[PairRequest],
) -> dict[str, Sequence[Shape]] | None:
    if supplied is None:
        return None
    hashes = {request.candidate.hash for request in requests}
    return {candidate_hash: shapes for candidate_hash, shapes in supplied.items() if candidate_hash in hashes}


def _native_phase_delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    phase_names = {
        "prepare": "preparation",
        "validation": "validation",
        "probe": "probe",
        "screening": "screening",
    }
    totals: dict[str, float] = {}
    for native_phase, controller_phase in phase_names.items():
        duration_s = after.get(native_phase, 0.0) - before.get(native_phase, 0.0)
        if duration_s > 0.0:
            totals[controller_phase] = totals.get(controller_phase, 0.0) + duration_s
    return totals


def _database_outcomes(
    db: EvoTensileDB,
    requests: Sequence[PairRequest],
    *,
    profile: TargetProfile,
    protocol: BenchmarkProtocol,
    provenance: str,
    source_ref: str,
) -> list[PairEvaluationOutcome]:
    shapes = requested_shapes(requests)
    candidates = requested_candidates(requests)
    benchmark_states = db.benchmark_evidence_states(
        problem_type_hash=profile.problem_type_hash,
        benchmark_protocol_hash=profile.benchmark_protocol_hash(protocol),
        shape_ids=[shape.id for shape in shapes],
        candidate_hashes=[candidate.hash for candidate in candidates],
    )
    validation_states = db.validation_cache_states(
        problem_type_hash=profile.problem_type_hash,
        validation_protocol_hash=protocol.validation_protocol_hash(),
        shape_ids=[shape.id for shape in shapes],
        candidate_hashes=[candidate.hash for candidate in candidates],
    )
    rankings = {
        (row.shape_id, row.candidate_hash): row
        for row in db.rank_benchmarks(
            problem_type_hash=profile.problem_type_hash,
            benchmark_protocol_hash=profile.benchmark_protocol_hash(protocol),
            min_samples=1,
        )
    }
    outcomes = []
    for request in requests:
        benchmark_state = benchmark_states.get(request.key)
        validation_state = validation_states.get(request.key)
        ranking = rankings.get(request.key)
        if benchmark_state is not None and benchmark_state.resolved_status is not None:
            status = benchmark_state.resolved_status
            known = True
            samples = benchmark_state.ok_samples
        elif validation_state is not None:
            status = f"validation_{validation_state}"
            known = True
            samples = 0
        else:
            status = "unknown"
            known = False
            samples = 0
        outcomes.append(
            PairEvaluationOutcome(
                request=request,
                provenance=provenance,
                source_ref=source_ref,
                status=status,
                known=known,
                disclosed=known,
                samples=samples,
                performance=None if ranking is None else ranking.median_gflops,
            )
        )
    return outcomes
