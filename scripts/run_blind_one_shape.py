#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from evotensile.adaptive_retime import AdaptivePolicy, ProbePolicy
from evotensile.candidate import Candidate, Shape, stable_hash
from evotensile.database import EvoTensileDB
from evotensile.profile import TargetProfile, get_profile
from evotensile.protocol import BenchmarkProtocol
from evotensile.runner import DEFAULT_TENSILELITE_BIN
from evotensile.scheduler import (
    DEFAULT_LINKAGE_MAX_CLUSTERS,
    DEFAULT_LINKAGE_MIN_SAMPLES,
    DEFAULT_LINKAGE_ORDINAL_BINS,
    DEFAULT_LINKAGE_TRUNCATION_TAU,
    ScheduleResult,
    execute_schedule,
    propose_candidates,
)
from evotensile.search.campaign_control import (
    ProposalEvent,
    convergence_detected,
    estimate_confirmation_reserve_s,
    estimate_next_round_duration_s,
    load_island_elites,
    plateau_detected,
    population_diagnostics,
    restart_epoch,
    split_budget,
    tag_generated_proposals,
)
from evotensile.search.hot_confirm import hot_confirm_topk
from evotensile.search.mechanics import mechanical_coverage_tokens
from evotensile.search.screening_stabilize import (
    ScreeningStabilizationPolicy,
    stabilize_screening_leaders,
)
from evotensile.shapes import parse_shape

CAMPAIGN_CONFIGURATION_VERSION = 1
DEFAULT_HOT_RESERVE_S = 60.0
DEFAULT_MAX_FEEDBACK_ROUNDS = 100
CAMPAIGN_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "EVOTENSILE_APU_LOCK_PATH",
    "HIPBLASLT_TENSILE_LIBPATH",
    "HIP_PATH",
    "HIP_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "LD_LIBRARY_PATH",
    "OPENBLAS_NUM_THREADS",
    "PATH",
    "PYTHONHASHSEED",
    "ROCM_PATH",
    "ROCR_VISIBLE_DEVICES",
    "TENSILELITE_ROOT",
)


@dataclass(frozen=True)
class CampaignConfiguration:
    version: int
    seed: int
    shape_id: str
    profile_name: str
    problem_type_hash: str
    runner_bin: str
    runner_fingerprint: str
    tensilelite_bin: str
    tensilelite_fingerprint: str
    implementation_fingerprint: str
    environment: tuple[tuple[str, str], ...]
    time_budget_s: float
    hot_reserve_s: float
    max_feedback_rounds: int
    early_stop_on_convergence: bool
    build_timeout_s: float
    runner_timeout_s: float
    screening_protocol: BenchmarkProtocol
    hot_protocol: BenchmarkProtocol
    adaptive_policy: AdaptivePolicy = field(default_factory=AdaptivePolicy)
    probe_policy: ProbePolicy = field(default_factory=ProbePolicy)
    stabilization_policy: ScreeningStabilizationPolicy = field(default_factory=ScreeningStabilizationPolicy)
    cold_candidates: int = 48
    cold_pool_multiplier: int = 8
    feedback_candidates: int = 24
    feedback_random: int = 4
    feedback_semantic_mutation: int = 6
    feedback_de: int = 4
    feedback_gomea: int = 10
    feedback_pool_multiplier: int = 8
    surrogate_min_evidence: int = 24
    elite_count: int = 32
    candidate_batch_size: int = 1
    shape_batch_size: int = 1
    min_samples: int = 2
    adaptive_max_rounds: int = 0
    prepare_workers: int = 32
    prepare_wave_batches: int = 32
    validation_workers: int = 1
    surrogate_jobs: int = 1
    compute_unit_count: int = 40
    workgroup_processor_count: int = 20
    compute_units_per_workgroup_processor: int = 2
    compile_threads: int = 1
    keep_going: bool = True
    compile_cache: bool = True
    cost_aware_scheduling: bool = True
    leader_stabilization: bool = True
    island_count: int = 2
    island_isolation_rounds: int = 6
    island_elites: int = 16
    plateau_patience: int = 3
    plateau_min_improvement_fraction: float = 0.005
    restart_max_mean_hamming: float = 5.0
    convergence_patience: int = 8
    convergence_minimum_improvement_fraction: float = 0.0025
    convergence_maximum_mean_hamming: float = 4.0
    hot_top_k: int = 8
    proposal_mode: str = "family-qd"
    mutation_rate: float = 0.25
    crossover_rate: float = 0.8
    random_gene_rate: float = 0.1
    linkage_truncation_tau: float = DEFAULT_LINKAGE_TRUNCATION_TAU
    linkage_min_samples: int = DEFAULT_LINKAGE_MIN_SAMPLES
    linkage_max_clusters: int = DEFAULT_LINKAGE_MAX_CLUSTERS
    linkage_ordinal_bins: int = DEFAULT_LINKAGE_ORDINAL_BINS
    transfer_shape_count: int = 0
    transfer_per_shape: int = 0
    adaptive_operators: bool = True
    adaptive_group_credit: bool = True
    micro_exhaustive_neighborhoods: bool = True
    adaptive_donor_selection: bool = True
    cost_aware_operator_credit: bool = True
    covering_cold_start: bool = True

    def __post_init__(self) -> None:
        if self.version != CAMPAIGN_CONFIGURATION_VERSION:
            raise ValueError("unsupported campaign configuration version")
        if self.time_budget_s <= 0.0:
            raise ValueError("campaign time budget must be positive")
        if not 0.0 <= self.hot_reserve_s < self.time_budget_s:
            raise ValueError("hot reserve must be non-negative and smaller than the time budget")
        if self.max_feedback_rounds < 0:
            raise ValueError("maximum feedback rounds must be non-negative")
        if self.feedback_candidates != (
            self.feedback_random + self.feedback_semantic_mutation + self.feedback_de + self.feedback_gomea
        ):
            raise ValueError("feedback candidate count must equal the operator budget sum")
        if self.island_count <= 0:
            raise ValueError("island count must be positive")
        if not 0 < self.island_elites <= self.elite_count:
            raise ValueError("island elites must be positive and no greater than merged elites")
        if self.candidate_batch_size <= 0 or self.shape_batch_size <= 0:
            raise ValueError("campaign batch sizes must be positive")
        if (
            self.prepare_workers <= 0
            or self.prepare_wave_batches <= 0
            or self.validation_workers <= 0
            or self.surrogate_jobs <= 0
        ):
            raise ValueError("campaign worker counts must be positive")
        if self.compute_unit_count <= 0 or self.workgroup_processor_count <= 0:
            raise ValueError("campaign hardware execution-unit counts must be positive")
        if self.compute_units_per_workgroup_processor <= 0:
            raise ValueError("campaign compute units per work-group processor must be positive")
        if self.compute_unit_count != (self.workgroup_processor_count * self.compute_units_per_workgroup_processor):
            raise ValueError("campaign compute-unit and work-group-processor topology is inconsistent")
        if self.compile_threads <= 0:
            raise ValueError("campaign compile threads must be positive")
        if self.build_timeout_s <= 0.0 or self.runner_timeout_s <= 0.0:
            raise ValueError("campaign subprocess timeouts must be positive")
        if self.screening_protocol.role != "main" or self.hot_protocol.role != "main":
            raise ValueError("screening and hot protocols must use the main benchmark role")
        if self.screening_protocol.num_elements_to_validate == 0:
            raise ValueError("screening protocol must retain correctness validation")
        if self.hot_protocol.num_elements_to_validate != 0:
            raise ValueError("hot protocol must disable repeated validation")
        if self.hot_top_k <= 0:
            raise ValueError("hot finalist count must be positive")

    def to_dict(self) -> dict[str, object]:
        return json.loads(json.dumps(asdict(self), sort_keys=True))

    @property
    def identity_hash(self) -> str:
        return stable_hash(self.to_dict(), prefix="campaign_")[:25]


@dataclass(frozen=True)
class RoundProposal:
    selected: tuple[Candidate, ...]
    active: tuple[Candidate, ...]
    archive: tuple[Candidate, ...]
    events: tuple[ProposalEvent, ...]


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


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _content_fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve(strict=True) for item in paths}, key=str):
        if not path.is_file():
            continue
        digest.update(str(path).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _binary_identity(path: Path, *, include_python_tree: bool = False) -> tuple[str, str]:
    resolved = path.resolve(strict=True)
    files = [resolved]
    if include_python_tree and resolved.parent.name == "bin":
        source_root = resolved.parent.parent
        files.extend(source_root.rglob("*.py"))
    return str(resolved), _content_fingerprint(files)


def _implementation_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), *(root / "evotensile").rglob("*.py")]
    return _content_fingerprint(files)


def _campaign_configuration(
    args: argparse.Namespace,
    *,
    profile: TargetProfile,
    shape: Shape,
) -> CampaignConfiguration:
    screening_protocol = BenchmarkProtocol(
        num_warmups=1,
        num_benchmarks=2,
        enqueues_per_sync=1,
        syncs_per_benchmark=1,
    )
    hot_protocol = BenchmarkProtocol(
        num_warmups=20,
        num_benchmarks=10,
        enqueues_per_sync=10,
        syncs_per_benchmark=1,
        num_elements_to_validate=0,
        validation_backend=screening_protocol.validation_backend,
    )
    runner_bin, runner_fingerprint = _binary_identity(args.runner_bin)
    tensilelite_bin, tensilelite_fingerprint = _binary_identity(
        args.tensilelite_bin,
        include_python_tree=True,
    )
    return CampaignConfiguration(
        version=CAMPAIGN_CONFIGURATION_VERSION,
        seed=args.seed,
        shape_id=shape.id,
        profile_name=profile.name,
        problem_type_hash=profile.problem_type_hash,
        runner_bin=runner_bin,
        runner_fingerprint=runner_fingerprint,
        tensilelite_bin=tensilelite_bin,
        tensilelite_fingerprint=tensilelite_fingerprint,
        implementation_fingerprint=_implementation_fingerprint(),
        environment=tuple((key, os.environ.get(key, "")) for key in CAMPAIGN_ENVIRONMENT_KEYS),
        time_budget_s=args.time_budget,
        hot_reserve_s=args.hot_reserve,
        max_feedback_rounds=args.max_feedback_rounds,
        early_stop_on_convergence=args.early_stop_on_convergence,
        build_timeout_s=args.build_timeout,
        runner_timeout_s=args.runner_timeout,
        screening_protocol=screening_protocol,
        hot_protocol=hot_protocol,
        stabilization_policy=ScreeningStabilizationPolicy(),
        leader_stabilization=not args.no_leader_stabilization,
        prepare_workers=profile.default_prepare_workers,
        prepare_wave_batches=profile.default_prepare_wave_batches,
        validation_workers=profile.default_validation_workers,
        surrogate_jobs=profile.default_surrogate_jobs,
        compute_unit_count=profile.compute_unit_count,
        workgroup_processor_count=profile.workgroup_processor_count,
        compute_units_per_workgroup_processor=profile.compute_units_per_workgroup_processor,
        mutation_rate=profile.default_mutation_rate,
        crossover_rate=profile.default_crossover_rate,
        random_gene_rate=profile.default_random_gene_rate,
    )


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


def _cold_args(configuration: CampaignConfiguration, *, count: int) -> ProposalArgs:
    return {
        "num_random": count,
        "elite_count": 0,
        "local_count": 0,
        "de_count": 0,
        "gomea_count": 0,
        "adaptive_operators": False,
        "surrogate_pool_multiplier": configuration.cold_pool_multiplier,
        "covering_cold_start": True,
        "adaptive_group_credit": False,
        "micro_exhaustive_neighborhoods": False,
        "adaptive_donor_selection": False,
        "cost_aware_operator_credit": False,
        "surrogate_min_evidence": configuration.surrogate_min_evidence,
    }


def _feedback_args(
    configuration: CampaignConfiguration,
    *,
    part_index: int = 0,
    parts: int = 1,
) -> ProposalArgs:
    return {
        "num_random": split_budget(configuration.feedback_random, parts)[part_index],
        "elite_count": configuration.elite_count if parts == 1 else configuration.island_elites,
        "local_count": split_budget(configuration.feedback_semantic_mutation, parts)[part_index],
        "de_count": split_budget(configuration.feedback_de, parts)[part_index],
        "gomea_count": split_budget(configuration.feedback_gomea, parts)[part_index],
        "adaptive_operators": configuration.adaptive_operators,
        "surrogate_pool_multiplier": configuration.feedback_pool_multiplier,
        "covering_cold_start": False,
        "adaptive_group_credit": configuration.adaptive_group_credit,
        "micro_exhaustive_neighborhoods": configuration.micro_exhaustive_neighborhoods,
        "adaptive_donor_selection": configuration.adaptive_donor_selection,
        "cost_aware_operator_credit": configuration.cost_aware_operator_credit,
        "surrogate_min_evidence": configuration.surrogate_min_evidence,
    }


def _proposal_call(
    db: EvoTensileDB,
    *,
    shape: Shape,
    profile: TargetProfile,
    configuration: CampaignConfiguration,
    protocol_hash: str,
    seed: int,
    proposal_args: ProposalArgs,
    island_id: str,
    parents: Sequence[Candidate] | None,
    learned_linkage: bool,
    restart_index: int,
    cold_start_precovered_tokens: set[str] | None = None,
) -> RoundProposal:
    parent_hashes = tuple(sorted(candidate.hash for candidate in parents or ()))
    started = time.perf_counter()
    proposal = propose_candidates(
        db,
        target_profile=profile,
        proposal=configuration.proposal_mode,
        seed=seed,
        problem_type_hash=profile.problem_type_hash,
        benchmark_protocol_hash=protocol_hash,
        shape_id=shape.id,
        target_shapes=[shape],
        transfer_shape_count=configuration.transfer_shape_count,
        transfer_per_shape=configuration.transfer_per_shape,
        mutation_rate=configuration.mutation_rate,
        crossover_rate=configuration.crossover_rate,
        random_gene_rate=configuration.random_gene_rate,
        learned_linkage=learned_linkage,
        linkage_truncation_tau=configuration.linkage_truncation_tau,
        linkage_min_samples=configuration.linkage_min_samples,
        linkage_max_clusters=configuration.linkage_max_clusters,
        linkage_ordinal_bins=configuration.linkage_ordinal_bins,
        parent_candidates=parents,
        cold_start_precovered_tokens=cold_start_precovered_tokens,
        surrogate_jobs=configuration.surrogate_jobs,
        workgroup_processor_count=configuration.workgroup_processor_count,
        **proposal_args,
    )
    duration = time.perf_counter() - started
    generated_hashes = {candidate.hash for candidate in proposal.generated}
    proposal_cost_s = duration / max(len(generated_hashes), 1)
    selected = tag_generated_proposals(
        proposal.selected,
        generated_hashes=generated_hashes,
        island_id=island_id,
        proposal_cost_s=proposal_cost_s,
        restart_index=restart_index,
    )
    selected_by_hash = {candidate.hash: candidate for candidate in selected}
    active = tuple(
        selected_by_hash[candidate.hash] for candidate in proposal.generated if candidate.hash in selected_by_hash
    )
    archive = tuple(
        selected_by_hash[candidate.hash] for candidate in proposal.preserved if candidate.hash in selected_by_hash
    )
    event = ProposalEvent(
        island_id=island_id,
        seed=seed,
        restart_index=restart_index,
        learned_linkage=learned_linkage,
        scope_kind=proposal.scope.kind,
        scope_shape_ids=proposal.scope.shape_ids,
        parent_hashes=parent_hashes,
        preserved_hashes=tuple(candidate.hash for candidate in proposal.preserved),
        generated_hashes=tuple(candidate.hash for candidate in proposal.generated),
        selected_hashes=tuple(candidate.hash for candidate in selected),
        duration_s=duration,
        proposal_cost_s=proposal_cost_s,
        proposal_args=proposal_args,
    )
    return RoundProposal(
        selected=tuple(selected),
        active=active,
        archive=archive,
        events=(event,),
    )


def _merge_proposals(proposals: Sequence[RoundProposal]) -> RoundProposal:
    selected = {candidate.hash: candidate for proposal in proposals for candidate in proposal.selected}
    active = {candidate.hash: candidate for proposal in proposals for candidate in proposal.active}
    archive = {candidate.hash: candidate for proposal in proposals for candidate in proposal.archive}
    return RoundProposal(
        selected=tuple(selected.values()),
        active=tuple(active.values()),
        archive=tuple(archive.values()),
        events=tuple(event for proposal in proposals for event in proposal.events),
    )


def _island_ids(configuration: CampaignConfiguration) -> tuple[str, ...]:
    return tuple(f"island-{index}" for index in range(configuration.island_count))


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
    configuration: CampaignConfiguration,
) -> bool:
    history = _leader_history(record, island_id=island_id)
    if not plateau_detected(
        history,
        patience=configuration.plateau_patience,
        minimum_improvement_fraction=configuration.plateau_min_improvement_fraction,
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
    diagnostics = rounds[-1].get("active_population_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return False
    mean_hamming = diagnostics.get("mean_pairwise_hamming")
    return isinstance(mean_hamming, (int, float, str)) and float(mean_hamming) <= configuration.restart_max_mean_hamming


def _propose_round(
    db: EvoTensileDB,
    *,
    record: dict[str, Any],
    round_index: int,
    seed: int,
    shape: Shape,
    profile: TargetProfile,
    protocol_hash: str,
    configuration: CampaignConfiguration,
) -> RoundProposal:
    proposals: list[RoundProposal] = []
    islands = _island_ids(configuration)
    if round_index == 0:
        precovered_tokens: set[str] = set()
        for island_index, (island_id, count) in enumerate(
            zip(islands, split_budget(configuration.cold_candidates, len(islands)), strict=True)
        ):
            proposal = _proposal_call(
                db,
                shape=shape,
                profile=profile,
                configuration=configuration,
                protocol_hash=protocol_hash,
                seed=seed + island_index * 1_000_003,
                proposal_args=_cold_args(configuration, count=count),
                island_id=island_id,
                parents=None,
                learned_linkage=False,
                restart_index=0,
                cold_start_precovered_tokens=precovered_tokens,
            )
            proposals.append(proposal)
            precovered_tokens.update(
                token
                for candidate in proposal.selected
                for token in mechanical_coverage_tokens(
                    candidate,
                    shape,
                    workgroup_processor_count=configuration.workgroup_processor_count,
                )
            )
        return _merge_proposals(proposals)

    if round_index <= configuration.island_isolation_rounds:
        for island_index, island_id in enumerate(islands):
            parents = load_island_elites(
                db,
                island_id=island_id,
                shape_id=shape.id,
                problem_type_hash=profile.problem_type_hash,
                benchmark_protocol_hash=protocol_hash,
                limit=configuration.island_elites,
            )
            restart_due = _restart_due(record, island_id=island_id, configuration=configuration)
            restart_index = restart_epoch(
                record["restart_counters"],
                scope=island_id,
                transition=restart_due,
            )
            proposal_args = (
                _cold_args(
                    configuration,
                    count=split_budget(configuration.feedback_candidates, len(islands))[island_index],
                )
                if restart_due or not parents
                else _feedback_args(configuration, part_index=island_index, parts=len(islands))
            )
            if restart_due:
                parents = []
            proposals.append(
                _proposal_call(
                    db,
                    shape=shape,
                    profile=profile,
                    configuration=configuration,
                    protocol_hash=protocol_hash,
                    seed=seed + island_index * 1_000_003,
                    proposal_args=proposal_args,
                    island_id=island_id,
                    parents=parents or None,
                    learned_linkage=False,
                    restart_index=restart_index,
                )
            )
        return _merge_proposals(proposals)

    global_restart = _restart_due(record, island_id=None, configuration=configuration)
    merged_restart_index = restart_epoch(
        record["restart_counters"],
        scope="merged",
        transition=global_restart,
    )
    if global_restart:
        proposals.append(
            _proposal_call(
                db,
                shape=shape,
                profile=profile,
                configuration=configuration,
                protocol_hash=protocol_hash,
                seed=seed,
                proposal_args=_feedback_args(configuration, part_index=0, parts=2),
                island_id="merged",
                parents=None,
                learned_linkage=True,
                restart_index=merged_restart_index,
            )
        )
        proposals.append(
            _proposal_call(
                db,
                shape=shape,
                profile=profile,
                configuration=configuration,
                protocol_hash=protocol_hash,
                seed=seed + 1_000_003,
                proposal_args=_cold_args(
                    configuration,
                    count=split_budget(configuration.feedback_candidates, 2)[1],
                ),
                island_id=f"restart-{merged_restart_index}",
                parents=None,
                learned_linkage=False,
                restart_index=merged_restart_index,
            )
        )
    else:
        proposals.append(
            _proposal_call(
                db,
                shape=shape,
                profile=profile,
                configuration=configuration,
                protocol_hash=protocol_hash,
                seed=seed,
                proposal_args=_feedback_args(configuration),
                island_id="merged",
                parents=None,
                learned_linkage=True,
                restart_index=merged_restart_index,
            )
        )
    return _merge_proposals(proposals)


def _load_pending_proposals(path: Path) -> RoundProposal:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = [_candidate_from_payload(item) for item in payload["candidates"]]
    by_hash = {candidate.hash: candidate for candidate in candidates}
    active = tuple(by_hash[candidate_hash] for candidate_hash in payload["active_candidate_hashes"])
    archive = tuple(by_hash[candidate_hash] for candidate_hash in payload["archive_candidate_hashes"])
    return RoundProposal(
        selected=tuple(candidates),
        active=active,
        archive=archive,
        events=tuple(ProposalEvent.from_mapping(event) for event in payload["proposal_events"]),
    )


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
            "configuration_hash": record["configuration_hash"],
            "restart_counters": record["restart_counters"],
            "deterministic_rng": "round and proposal-event seeds fully determine generator and surrogate RNG state",
            "operator_credit_state": "derived from the checkpointed campaign DB",
            "surrogate_state": "refit deterministically from the checkpointed campaign DB and stored proposals",
        },
    )


def _load_or_create_campaign(
    args: argparse.Namespace,
    *,
    configuration: CampaignConfiguration,
) -> tuple[dict[str, Any], bool]:
    progress_path = args.output / "campaign_progress.json"
    configuration_path = args.output / "campaign_configuration.json"
    expected_configuration = configuration.to_dict()
    if args.output.exists():
        if not args.resume:
            raise SystemExit(f"output already exists: {args.output}")
        if not configuration_path.exists():
            raise SystemExit(f"cannot resume without {configuration_path}")
        frozen_configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
        if frozen_configuration != expected_configuration:
            raise SystemExit("resume configuration mismatch; start a new campaign root")
        record = json.loads(
            (progress_path if progress_path.exists() else configuration_path).read_text(encoding="utf-8")
        )
        if record.get("configuration_hash") != configuration.identity_hash:
            raise SystemExit("resume configuration hash mismatch; start a new campaign root")
        return record, True

    args.output.mkdir(parents=True)
    _write_json_atomic(configuration_path, expected_configuration)
    record: dict[str, Any] = {
        "blind": True,
        "configuration": expected_configuration,
        "configuration_hash": configuration.identity_hash,
        "screening_protocol_hash": configuration.screening_protocol.protocol_hash(),
        "validation_protocol_hash": configuration.screening_protocol.validation_protocol_hash(),
        "hot_protocol_hash": configuration.hot_protocol.protocol_hash(),
        "rounds": [],
        "restart_counters": {
            **{island_id: 0 for island_id in _island_ids(configuration)},
            "merged": 0,
        },
        "search_elapsed_s": 0.0,
        "active_elapsed_s": 0.0,
        "stop_reason": None,
    }
    _write_json_atomic(progress_path, record)
    return record, False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the blind one-shape 20-minute search policy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", default="8192,8192,1,8192")
    parser.add_argument("--profile", default="gfx1151-nt-hhs")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--time-budget", type=float, default=1200.0)
    parser.add_argument("--hot-reserve", type=float, default=DEFAULT_HOT_RESERVE_S)
    parser.add_argument("--max-feedback-rounds", type=int, default=DEFAULT_MAX_FEEDBACK_ROUNDS)
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
    configuration = _campaign_configuration(args, profile=profile, shape=shape)
    protocol = configuration.screening_protocol
    protocol_hash = protocol.protocol_hash()
    record, resumed = _load_or_create_campaign(args, configuration=configuration)
    db_path = args.output / "campaign.sqlite"
    db = EvoTensileDB.connect(db_path)
    db.init()
    db.register_shapes([shape])

    session_start = time.monotonic()
    adaptive_policy = configuration.adaptive_policy
    probe_policy = configuration.probe_policy
    stabilization_policy = configuration.stabilization_policy
    compile_cache = args.output / "compile_cache" if configuration.compile_cache else None

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
    if "restart_counters" in checkpoint:
        record["restart_counters"] = checkpoint["restart_counters"]
    if checkpoint.get("phase") == "finished" and (args.output / "campaign_summary.json").exists():
        print((args.output / "campaign_summary.json").read_text(encoding="utf-8"), end="")
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
        round_dir = args.output / f"round_{round_index:02d}"
        round_dir.mkdir(exist_ok=True)
        proposals_path = round_dir / "proposals.json"
        if pending:
            round_proposal = _load_pending_proposals(proposals_path)
        else:
            round_proposal = _propose_round(
                db,
                record=record,
                round_index=round_index,
                seed=round_seed,
                shape=shape,
                profile=profile,
                protocol_hash=protocol_hash,
                configuration=configuration,
            )
            _write_json_atomic(
                proposals_path,
                {
                    "round": round_index,
                    "seed": round_seed,
                    "proposal_events": [event.to_dict() for event in round_proposal.events],
                    "active_candidate_hashes": [candidate.hash for candidate in round_proposal.active],
                    "archive_candidate_hashes": [candidate.hash for candidate in round_proposal.archive],
                    "candidates": [_candidate_payload(candidate) for candidate in round_proposal.selected],
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
                shape=shape,
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
            for island_id in _island_ids(configuration)
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

        if configuration.early_stop_on_convergence:
            history = _leader_history(record)
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
        output_dir=args.output / "hot_loop_top8",
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
