import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .adaptive_retime import AdaptivePolicy, ProbePolicy
from .candidate import Candidate, Shape
from .database import EvoTensileDB
from .profile import PROFILES, TargetProfile, get_profile
from .proposal import (
    FamilyQDPolicy,
    ProposalResult,
    load_proposal_config,
    load_proposal_script,
)
from .protocol import BenchmarkProtocol, apply_benchmark_protocol_overrides
from .runner import DEFAULT_TENSILELITE_BIN
from .scheduler import DEFAULT_COMPILE_THREADS, execute_schedule
from .scheduling.models import EvidenceStage, PairRequest, ScheduleResult
from .search.acquisition import propose_candidates
from .search.coverage import candidate_coverage
from .search.evidence import load_proposal_evidence_snapshot
from .search.family import family_descriptor_counts, load_family_archive
from .search.grid_evidence import GRID_OBJECTIVES, GridObjective
from .search.learned_linkage import learn_linkage_models_from_snapshot
from .search_space import DOMAINS, MATRIX_INSTRUCTIONS, macro_tile, random_candidates
from .shapes import parse_shape


def _profile(args: argparse.Namespace) -> TargetProfile:
    return get_profile(getattr(args, "profile", None))


def _protocol(args: argparse.Namespace, profile: TargetProfile) -> BenchmarkProtocol:
    return apply_benchmark_protocol_overrides(profile.default_protocol, vars(args))


def _parse_shapes(args: argparse.Namespace, profile: TargetProfile) -> list[Shape]:
    if getattr(args, "shapes", None):
        return [parse_shape(s) for s in args.shapes]
    shapes = profile.shapes()
    if getattr(args, "limit_shapes", None):
        return shapes[: args.limit_shapes]
    return shapes


def _timing_policies(args: argparse.Namespace) -> tuple[AdaptivePolicy | None, ProbePolicy | None]:
    if getattr(args, "fixed_sampling", False):
        return None, None
    return (
        AdaptivePolicy(
            epsilon_pct=args.adaptive_epsilon_pct,
            confidence=args.adaptive_confidence,
            min_retime_samples=args.adaptive_min_samples,
            max_retime_samples=args.adaptive_max_samples,
            sample_step=args.adaptive_sample_step,
            max_k=args.adaptive_max_k,
            min_effect_pct=args.adaptive_min_effect_pct,
            max_rounds=args.adaptive_max_rounds,
        ),
        ProbePolicy(
            samples=args.adaptive_probe_samples,
            initial_samples=args.adaptive_probe_initial_samples,
            max_slowdown_factor=args.adaptive_probe_max_slowdown_factor,
            confidence=args.adaptive_probe_confidence,
            noise_floor_pct=args.adaptive_probe_noise_floor_pct,
            min_survivors=args.adaptive_probe_min_survivors,
        ),
    )


FAMILY_QD_ARGUMENTS = (
    "num_random",
    "transfer_shapes",
    "transfer_per_shape",
    "elite_count",
    "local_count",
    "de_count",
    "gomea_count",
    "mutation_rate",
    "crossover_rate",
    "random_gene_rate",
    "learned_linkage",
    "linkage_truncation_tau",
    "linkage_min_samples",
    "linkage_max_clusters",
    "linkage_ordinal_bins",
    "adaptive_operators",
    "surrogate_pool_multiplier",
    "surrogate_min_evidence",
    "covering_cold_start",
    "adaptive_group_credit",
    "micro_exhaustive_neighborhoods",
    "adaptive_donor_selection",
    "cost_aware_operator_credit",
)


def _resolve_profile_defaults(args: argparse.Namespace, profile: TargetProfile) -> None:
    policy = FamilyQDPolicy()
    defaults = {
        "num_random": policy.num_random,
        "transfer_shapes": policy.transfer_shape_count,
        "transfer_per_shape": policy.transfer_per_shape,
        "elite_count": policy.elite_count,
        "local_count": policy.local_count,
        "de_count": policy.de_count,
        "gomea_count": policy.gomea_count,
        "mutation_rate": policy.mutation_rate,
        "crossover_rate": policy.crossover_rate,
        "random_gene_rate": policy.random_gene_rate,
        "learned_linkage": policy.learned_linkage,
        "linkage_truncation_tau": policy.linkage_truncation_tau,
        "linkage_min_samples": policy.linkage_min_samples,
        "linkage_max_clusters": policy.linkage_max_clusters,
        "linkage_ordinal_bins": policy.linkage_ordinal_bins,
        "adaptive_operators": policy.adaptive_operators,
        "surrogate_pool_multiplier": policy.surrogate_pool_multiplier,
        "surrogate_min_evidence": policy.surrogate_min_evidence,
        "covering_cold_start": policy.covering_cold_start,
        "adaptive_group_credit": policy.adaptive_group_credit,
        "micro_exhaustive_neighborhoods": policy.micro_exhaustive_neighborhoods,
        "adaptive_donor_selection": policy.adaptive_donor_selection,
        "cost_aware_operator_credit": policy.cost_aware_operator_credit,
        "surrogate_jobs": profile.default_surrogate_jobs,
        "cost_aware_scheduling": True,
    }
    for name, value in defaults.items():
        if hasattr(args, name) and getattr(args, name) is None:
            setattr(args, name, value)


def _resolved_profile(args: argparse.Namespace) -> TargetProfile:
    profile = _profile(args)
    for name in FAMILY_QD_ARGUMENTS:
        if hasattr(args, name):
            setattr(args, f"_{name}_supplied", getattr(args, name) is not None)
    _resolve_profile_defaults(args, profile)
    return profile


def _candidates(args: argparse.Namespace):
    _resolve_profile_defaults(args, _profile(args))
    return random_candidates(args.num_random, seed=args.seed)


def _add_profile_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None, help="Target profile")


def _add_candidate_shape_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-random", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--limit-shapes", type=int, default=None)
    parser.add_argument("--shapes", nargs="*")


def _add_protocol_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-warmups", type=int, default=None, help="Main timing warmup launches")
    parser.add_argument(
        "--num-benchmarks",
        type=int,
        default=None,
        help="Initial main timing samples per probe survivor",
    )
    parser.add_argument(
        "--enqueues-per-sync",
        type=int,
        default=None,
        help="Main timing launches per sample; probes always use one",
    )
    parser.add_argument("--syncs-per-benchmark", type=int, default=None)
    parser.add_argument("--num-elements-to-validate", type=int, default=None)
    parser.add_argument(
        "--validation-backend",
        choices=("cpu", "hipblaslt"),
        default=None,
        help="Structured-runner validation backend; defaults to hipblaslt GPU oracle",
    )


def _add_timeout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--build-timeout",
        type=float,
        default=None,
        help="TensileLite build timeout in seconds; defaults to the target profile, 0 disables",
    )
    parser.add_argument(
        "--runner-timeout",
        type=float,
        default=None,
        help="Structured runner timeout in seconds; defaults to the target profile, 0 disables",
    )


@dataclass(frozen=True)
class ScheduleCliContext:
    profile: TargetProfile
    protocol: BenchmarkProtocol
    db: EvoTensileDB
    problem_hash: str
    protocol_hash: str
    shapes: list[Shape]
    runner_bin: str | None
    build_timeout: float | None
    runner_timeout: float | None
    adaptive_policy: AdaptivePolicy | None
    probe_policy: ProbePolicy | None


def _validate_schedule_args(args: argparse.Namespace) -> TargetProfile:
    profile = _resolved_profile(args)
    if args.candidate_batch_size is not None and args.candidate_batch_size <= 0:
        raise ValueError("--candidate-batch-size must be positive")
    positive_ints = (
        "shape_batch_size",
        "min_samples",
        "prepare_workers",
        "prepare_wave_batches",
        "validation_workers",
        "surrogate_jobs",
    )
    for name in positive_ints:
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    nonnegative_ints = (
        "num_random",
        "elite_count",
        "local_count",
        "de_count",
        "gomea_count",
        "transfer_shapes",
        "transfer_per_shape",
        "adaptive_max_rounds",
        "adaptive_min_samples",
        "adaptive_max_samples",
        "adaptive_sample_step",
        "adaptive_max_k",
        "linkage_min_samples",
        "linkage_max_clusters",
        "linkage_ordinal_bins",
    )
    for name in nonnegative_ints:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.adaptive_max_samples < args.adaptive_min_samples:
        raise ValueError("--adaptive-max-samples must be >= --adaptive-min-samples")
    if args.linkage_truncation_tau <= 0.0 or args.linkage_truncation_tau > 1.0:
        raise ValueError("--linkage-truncation-tau must be in (0, 1]")
    return profile


def _schedule_context(args: argparse.Namespace) -> ScheduleCliContext:
    profile = _validate_schedule_args(args)
    protocol = _protocol(args, profile)
    db = EvoTensileDB.connect(
        args.db,
        environment_compatibility_tag=profile.environment_compatibility_tag,
    )
    db.init()
    adaptive_policy, probe_policy = _timing_policies(args)
    return ScheduleCliContext(
        profile=profile,
        protocol=protocol,
        db=db,
        problem_hash=profile.problem_type_hash,
        protocol_hash=profile.benchmark_protocol_hash(protocol),
        shapes=_parse_shapes(args, profile),
        runner_bin=args.runner_bin,
        build_timeout=args.build_timeout,
        runner_timeout=args.runner_timeout,
        adaptive_policy=adaptive_policy,
        probe_policy=probe_policy,
    )


def _family_qd_policy(args: argparse.Namespace) -> FamilyQDPolicy:
    return FamilyQDPolicy(
        num_random=args.num_random,
        elite_count=args.elite_count,
        local_count=args.local_count,
        de_count=args.de_count,
        gomea_count=args.gomea_count,
        transfer_shape_count=args.transfer_shapes,
        transfer_per_shape=args.transfer_per_shape,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        random_gene_rate=args.random_gene_rate,
        learned_linkage=args.learned_linkage,
        linkage_truncation_tau=args.linkage_truncation_tau,
        linkage_min_samples=args.linkage_min_samples,
        linkage_max_clusters=args.linkage_max_clusters,
        linkage_ordinal_bins=args.linkage_ordinal_bins,
        adaptive_operators=args.adaptive_operators,
        surrogate_pool_multiplier=args.surrogate_pool_multiplier,
        surrogate_min_evidence=args.surrogate_min_evidence,
        covering_cold_start=args.covering_cold_start,
        adaptive_group_credit=args.adaptive_group_credit,
        micro_exhaustive_neighborhoods=args.micro_exhaustive_neighborhoods,
        adaptive_donor_selection=args.adaptive_donor_selection,
        cost_aware_operator_credit=args.cost_aware_operator_credit,
    )


def _propose_candidates_for_shapes(
    db: EvoTensileDB,
    args: argparse.Namespace,
    *,
    problem_hash: str,
    protocol_hash: str,
    shapes: list[Shape],
    proposal_shape_id: str | None = None,
) -> ProposalResult:
    provider = None
    provenance = None
    config = None
    policy = _family_qd_policy(args)
    if args.proposal_config is not None and args.proposal_script is None:
        raise ValueError("--proposal-config requires --proposal-script")
    if args.proposal_script is not None:
        overridden = [name for name in FAMILY_QD_ARGUMENTS if getattr(args, f"_{name}_supplied", False)]
        if overridden:
            flags = ", ".join(f"--{name.replace('_', '-')}" for name in overridden)
            raise ValueError(f"custom proposal scripts cannot use family-QD options: {flags}")
        provider, provenance = load_proposal_script(args.proposal_script)
        config = load_proposal_config(args.proposal_config)
        policy = None
        print(
            "warning: custom Python proposal providers are not automatically reproducible; "
            "update environment_compatibility_tag when provider or dependency changes require a new evidence namespace",
            file=sys.stderr,
        )
    return propose_candidates(
        db,
        target_profile=_profile(args),
        policy=policy,
        provider=provider,
        provider_provenance=provenance,
        provider_config=config,
        seed=args.seed,
        problem_type_hash=problem_hash,
        benchmark_protocol_hash=protocol_hash,
        shape_id=proposal_shape_id,
        target_shapes=shapes,
        scope_kind=args.proposal_scope,
    )


def _execute_schedule_from_args(
    args: argparse.Namespace,
    context: ScheduleCliContext,
    *,
    requests: list[PairRequest],
    dry_run: bool | None = None,
) -> ScheduleResult:
    return execute_schedule(
        context.db,
        requests=requests,
        output_root=args.output_dir,
        target_profile=context.profile,
        protocol=context.protocol,
        candidate_batch_size=args.candidate_batch_size,
        shape_batch_size=args.shape_batch_size,
        ignore_cache=args.ignore_cache,
        max_batches=args.max_batches,
        dry_run=args.dry_run if dry_run is None else dry_run,
        generate_only=args.generate_only,
        tensilelite_bin=args.tensilelite_bin,
        compile_threads=args.compile_threads,
        keep_going=not args.stop_on_error,
        runner_bin=context.runner_bin,
        build_timeout_s=context.build_timeout,
        runner_timeout_s=context.runner_timeout,
        adaptive_policy=context.adaptive_policy,
        probe_policy=context.probe_policy,
        prepare_workers=args.prepare_workers,
        compile_cache_root=None
        if args.no_compile_cache
        else args.compile_cache_dir or Path(args.output_dir) / "compile_cache",
        cost_aware_scheduling=args.cost_aware_scheduling,
        validation_workers=args.validation_workers,
        prepare_wave_batches=args.prepare_wave_batches,
    )


def _learned_linkage_metadata(
    args: argparse.Namespace, context: ScheduleCliContext, shapes: list[Shape]
) -> dict[str, object]:
    if not args.learned_linkage:
        return {
            "learned_linkage_requested": False,
            "learned_linkage_enabled": False,
            "linkage_model_count": 0,
            "linkage_fallback_reason": "disabled",
            "linkage_models": [],
        }
    snapshot = load_proposal_evidence_snapshot(
        context.db,
        problem_type_hash=context.problem_hash,
        benchmark_protocol_hash=context.protocol_hash,
        shapes=shapes,
    )
    models, summary = learn_linkage_models_from_snapshot(
        snapshot,
        shapes=shapes,
        truncation_tau=args.linkage_truncation_tau,
        min_samples=args.linkage_min_samples,
        max_clusters=args.linkage_max_clusters,
        ordinal_bins=args.linkage_ordinal_bins,
    )
    return {
        "learned_linkage_requested": True,
        "learned_linkage_enabled": summary.enabled,
        "linkage_model_count": summary.model_count,
        "linkage_evidence_count": summary.evidence_count,
        "linkage_selected_count": summary.selected_count,
        "linkage_fallback_reason": summary.fallback_reason,
        "linkage_truncation_tau": args.linkage_truncation_tau,
        "linkage_min_samples": args.linkage_min_samples,
        "linkage_max_clusters": args.linkage_max_clusters,
        "linkage_ordinal_bins": args.linkage_ordinal_bins,
        "linkage_models": [model.summary() for model in models],
    }


def _family_metadata(
    context: ScheduleCliContext,
    *,
    candidates: list[Candidate],
    shapes: list[Shape],
) -> dict[str, object]:
    descriptor_counts = family_descriptor_counts(candidates)
    snapshot = load_proposal_evidence_snapshot(
        context.db,
        problem_type_hash=context.problem_hash,
        benchmark_protocol_hash=context.protocol_hash,
        shapes=shapes,
    )
    archive = load_family_archive(
        snapshot,
        shapes=shapes,
        min_samples=1,
        objective=GridObjective.GENERALIST if len(shapes) > 1 else GridObjective.SPECIALIST,
        limit=16,
    )
    return {
        "candidate_family_count": len(descriptor_counts),
        "candidate_family_counts": dict(sorted(descriptor_counts.items())),
        "archive_family_count": len(archive),
        "archive_families": [entry.summary() for entry in archive],
    }


def _schedule_metadata_common(
    args: argparse.Namespace,
    context: ScheduleCliContext,
    *,
    result: ScheduleResult,
    proposal: ProposalResult,
    candidates: list[Candidate],
    shapes: list[Shape],
) -> dict[str, object]:
    return {
        "db": args.db,
        "output_dir": args.output_dir,
        "profile": context.profile.name,
        "problem_type_hash": context.problem_hash,
        "benchmark_protocol_hash": context.protocol_hash,
        "validation_protocol_hash": context.protocol.validation_protocol_hash(),
        "protocol": context.protocol.global_parameters(),
        "runner_protocol": context.protocol.runner_parameters(),
        "validation_backend": context.protocol.validation_backend,
        "proposal_provider": dict(proposal.provider),
        "proposal_metadata": dict(proposal.metadata),
        "proposal_scope": args.proposal_scope or ("shape" if len(shapes) == 1 else "shape-set"),
        "proposal_scope_shape_ids": [shape.id for shape in shapes],
        "adaptive_operators": args.adaptive_operators,
        "surrogate_pool_multiplier": args.surrogate_pool_multiplier,
        "surrogate_min_evidence": args.surrogate_min_evidence,
        "covering_cold_start": args.covering_cold_start,
        "adaptive_group_credit": args.adaptive_group_credit,
        "micro_exhaustive_neighborhoods": args.micro_exhaustive_neighborhoods,
        "adaptive_donor_selection": args.adaptive_donor_selection,
        "cost_aware_operator_credit": args.cost_aware_operator_credit,
        "cost_aware_scheduling": args.cost_aware_scheduling,
        "candidates": len(candidates),
        "shapes": len(shapes),
        "candidate_batch_size": result.candidate_batch_size,
        "shape_batch_size": result.shape_batch_size,
        "prepare_workers": result.prepare_workers,
        "prepare_wave_batches": result.prepare_wave_batches,
        "validation_workers": result.validation_workers,
        "surrogate_jobs": args.surrogate_jobs,
        "compute_unit_count": context.profile.compute_unit_count,
        "workgroup_processor_count": context.profile.workgroup_processor_count,
        "compute_units_per_workgroup_processor": context.profile.compute_units_per_workgroup_processor,
        "compile_cache_root": None
        if args.no_compile_cache
        else str(args.compile_cache_dir or Path(args.output_dir) / "compile_cache"),
        "compile_cache_enabled": not args.no_compile_cache,
        "min_samples": args.min_samples,
        "ignore_cache": args.ignore_cache,
        "dry_run": args.dry_run,
        "generate_only": args.generate_only,
        "stop_on_error": args.stop_on_error,
        "runner_bin": result.runner_bin,
        "build_timeout_s": result.build_timeout_s,
        "runner_timeout_s": result.runner_timeout_s,
        "adaptive_sampling": context.adaptive_policy is not None,
        "adaptive_max_rounds": None if context.adaptive_policy is None else context.adaptive_policy.max_rounds,
        "completed_waves": result.completed_waves,
        "adaptive_rounds": result.adaptive_rounds,
        "adaptive_policy": None if context.adaptive_policy is None else context.adaptive_policy.__dict__,
        "probe_policy": None if context.probe_policy is None else context.probe_policy.__dict__,
        "probe_protocol_hash": result.probe_protocol_hash,
        "probe_policy_hash": result.probe_policy_hash,
        "probe_survivor_pairs": result.probe_survivor_pairs,
        "probe_screened_pairs": result.probe_screened_pairs,
        "probe_preprepare_screened_pairs": result.probe_preprepare_screened_pairs,
        **_learned_linkage_metadata(args, context, shapes),
        "planned_batches": len(result.planned_batches),
        "planned_requested_pairs": result.requested_pairs,
        "planned_requested_samples": result.requested_samples,
        **_family_metadata(context, candidates=candidates, shapes=shapes),
    }


def _add_proposal_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--proposal-script",
        type=Path,
        default=None,
        help="Trusted Python file exporting propose(context); defaults to built-in family-QD",
    )
    parser.add_argument(
        "--proposal-config",
        type=Path,
        default=None,
        help="JSON object passed to a custom proposal script",
    )
    parser.add_argument(
        "--proposal-scope",
        choices=("shape", "cluster", "shape-set"),
        default=None,
        help="Label the declared proposal shape scope; inferred from shape count when omitted",
    )
    parser.add_argument("--proposal-shape-id", default=None, help="Limit cached elite selection to one shape id")
    parser.add_argument(
        "--transfer-shapes",
        type=int,
        default=None,
        help="Seed normal proposal generation from this many nearest already-tuned shapes; 0 disables transfer",
    )
    parser.add_argument(
        "--transfer-per-shape",
        type=int,
        default=None,
        help="Seed normal proposal generation with this many top candidates per nearest shape",
    )
    parser.add_argument("--elite-count", type=int, default=None)
    parser.add_argument("--local-count", type=int, default=None)
    parser.add_argument("--de-count", type=int, default=None)
    parser.add_argument("--gomea-count", type=int, default=None)
    parser.add_argument("--mutation-rate", type=float, default=None)
    parser.add_argument(
        "--adaptive-operators",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allocate family-QD variation budget from queried child-versus-parent evidence",
    )
    parser.add_argument(
        "--surrogate-pool-multiplier",
        type=int,
        default=None,
        help="Generate this multiple of the requested proposal budget, then shortlist blindly from DB evidence",
    )
    parser.add_argument(
        "--surrogate-min-evidence",
        type=int,
        default=None,
        help="Unique varied candidates required per shape before fitting its proposal surrogate",
    )
    parser.add_argument(
        "--covering-cold-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Shortlist evidence-free one-shape pools by mechanical and parameter coverage",
    )
    parser.add_argument(
        "--adaptive-group-credit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Bias semantic and neighborhood groups from queried group-level rewards",
    )
    parser.add_argument(
        "--micro-exhaustive-neighborhoods",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enumerate bounded valid alternatives inside selected GOMEA neighborhoods",
    )
    parser.add_argument(
        "--adaptive-donor-selection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Mix quality, diverse, and random GOMEA donors using queried donor-mode rewards",
    )
    parser.add_argument(
        "--cost-aware-operator-credit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Scale queried operator, group, and donor rewards by measured evaluation cost",
    )
    parser.add_argument("--crossover-rate", type=float, default=None)
    parser.add_argument("--random-gene-rate", type=float, default=None)
    parser.add_argument(
        "--learned-linkage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use validated DB evidence to learn basin-aware GOMEA linkage models",
    )
    parser.add_argument("--linkage-truncation-tau", type=float, default=None)
    parser.add_argument("--linkage-min-samples", type=int, default=None)
    parser.add_argument("--linkage-max-clusters", type=int, default=None)
    parser.add_argument("--linkage-ordinal-bins", type=int, default=None)


def _add_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=None,
        help=(
            "Candidates per TensileLite config; compile caching forces 1, otherwise defaults to a "
            "profile-bounded throughput heuristic"
        ),
    )
    parser.add_argument("--shape-batch-size", type=int, default=None)
    parser.add_argument(
        "--prepare-workers",
        type=int,
        default=None,
        help="Parallel build/map/diagnostic/validation workers; defaults to the target profile",
    )
    parser.add_argument(
        "--prepare-wave-batches",
        type=int,
        default=None,
        help="Maximum prepared batches before serial timing and coordinator feedback; defaults to the target profile",
    )
    parser.add_argument(
        "--validation-workers",
        type=int,
        default=None,
        help="Concurrent validation runners; defaults to the target profile",
    )
    parser.add_argument(
        "--surrogate-jobs",
        type=int,
        default=None,
        help="CPU jobs per ExtraTrees fit; defaults to the target profile",
    )
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=None,
        help="Stable TensileLite build-cache directory; defaults to OUTPUT_DIR/compile_cache",
    )
    parser.add_argument("--no-compile-cache", action="store_true", help="Disable stable TensileLite build-cache reuse")
    parser.add_argument(
        "--cost-aware-scheduling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Order parallel preparation longest-predicted-work first",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--tensilelite-bin", default=DEFAULT_TENSILELITE_BIN)
    parser.add_argument(
        "--compile-threads",
        type=int,
        default=DEFAULT_COMPILE_THREADS,
        help="TensileLite CpuThreads per batch; defaults to 1",
    )
    parser.add_argument(
        "--runner-bin", default=None, help="Structured runner executable; defaults to the target profile"
    )
    _add_timeout_args(parser)
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after the first failed batch")


def _add_adaptive_args(parser: argparse.ArgumentParser) -> None:
    adaptive = AdaptivePolicy()
    probe = ProbePolicy()
    parser.add_argument("--fixed-sampling", action="store_true", help="Disable probing and adaptive top-ups")
    parser.add_argument("--adaptive-probe-samples", type=int, default=probe.samples)
    parser.add_argument("--adaptive-probe-initial-samples", type=int, default=probe.initial_samples)
    parser.add_argument(
        "--adaptive-probe-max-slowdown-factor",
        type=float,
        default=probe.max_slowdown_factor,
    )
    parser.add_argument("--adaptive-probe-confidence", type=float, default=probe.confidence)
    parser.add_argument("--adaptive-probe-noise-floor-pct", type=float, default=probe.noise_floor_pct)
    parser.add_argument("--adaptive-probe-min-survivors", type=int, default=probe.min_survivors)
    parser.add_argument("--adaptive-max-rounds", type=int, default=adaptive.max_rounds)
    parser.add_argument("--adaptive-epsilon-pct", type=float, default=adaptive.epsilon_pct)
    parser.add_argument("--adaptive-confidence", type=float, default=adaptive.confidence)
    parser.add_argument("--adaptive-min-samples", type=int, default=adaptive.min_retime_samples)
    parser.add_argument("--adaptive-max-samples", type=int, default=adaptive.max_retime_samples)
    parser.add_argument("--adaptive-sample-step", type=int, default=adaptive.sample_step)
    parser.add_argument("--adaptive-max-k", type=int, default=adaptive.max_k)
    parser.add_argument("--adaptive-min-effect-pct", type=float, default=adaptive.min_effect_pct)


def _add_schedule_args(parser: argparse.ArgumentParser) -> None:
    _add_candidate_shape_args(parser)
    _add_profile_arg(parser)
    _add_protocol_args(parser)
    _add_proposal_args(parser)
    _add_execution_args(parser)
    _add_adaptive_args(parser)


def cmd_proposal_coverage(args: argparse.Namespace) -> int:
    profile = _resolved_profile(args)
    db = EvoTensileDB.connect(
        args.db,
        environment_compatibility_tag=profile.environment_compatibility_tag,
    )
    db.init()
    shapes = _parse_shapes(args, profile)
    protocol = _protocol(args, profile)
    proposal = _propose_candidates_for_shapes(
        db,
        args,
        problem_hash=profile.problem_type_hash,
        protocol_hash=profile.benchmark_protocol_hash(protocol),
        shapes=shapes,
        proposal_shape_id=args.proposal_shape_id,
    )
    candidates = list(proposal.selected)
    coverage = candidate_coverage(candidates)
    descriptor_counts = family_descriptor_counts(candidates)
    payload = {
        **coverage,
        "profile": profile.name,
        "proposal_provider": dict(proposal.provider),
        "proposal_metadata": dict(proposal.metadata),
        "num_random": args.num_random,
        "gomea_count": args.gomea_count,
        "de_count": args.de_count,
        "local_count": args.local_count,
        "adaptive_operators": args.adaptive_operators,
        "surrogate_pool_multiplier": args.surrogate_pool_multiplier,
        "surrogate_min_evidence": args.surrogate_min_evidence,
        "covering_cold_start": args.covering_cold_start,
        "adaptive_group_credit": args.adaptive_group_credit,
        "micro_exhaustive_neighborhoods": args.micro_exhaustive_neighborhoods,
        "adaptive_donor_selection": args.adaptive_donor_selection,
        "cost_aware_operator_credit": args.cost_aware_operator_credit,
        "candidate_family_count": len(descriptor_counts),
        "candidate_family_counts": dict(sorted(descriptor_counts.items())),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_summarize_families(args: argparse.Namespace) -> int:
    profile = _profile(args)
    protocol = _protocol(args, profile)
    db = EvoTensileDB.connect(
        args.db,
        environment_compatibility_tag=profile.environment_compatibility_tag,
    )
    db.init()
    shapes = (
        _parse_shapes(args, profile) if getattr(args, "shapes", None) or getattr(args, "limit_shapes", None) else None
    )
    snapshot = load_proposal_evidence_snapshot(
        db,
        problem_type_hash=profile.problem_type_hash,
        benchmark_protocol_hash=profile.benchmark_protocol_hash(protocol),
        shapes=shapes,
    )
    archive = load_family_archive(
        snapshot,
        shapes=shapes,
        min_samples=args.min_samples,
        objective=args.archive_objective,
        limit=args.limit,
    )
    payload = {
        "profile": profile.name,
        "problem_type_hash": profile.problem_type_hash,
        "benchmark_protocol_hash": profile.benchmark_protocol_hash(protocol),
        "families": len(archive),
        "entries": [entry.summary() for entry in archive],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_summarize_space(args: argparse.Namespace) -> int:
    profile = _profile(args)
    candidates = _candidates(args)
    print("EvoTensile search-space summary")
    print(f"  profile: {profile.name}")
    print(f"  problem_type_hash: {profile.problem_type_hash}")
    print(f"  benchmark_protocol_hash: {profile.benchmark_protocol_hash()}")
    print(f"  MatrixInstruction choices: {len(MATRIX_INSTRUCTIONS)}")
    for mi in MATRIX_INSTRUCTIONS:
        mt0, mt1 = macro_tile(mi)
        print(f"    {mi} -> MT{mt0}x{mt1}")
    print("  Domain sizes:")
    product = 1
    for name, values in DOMAINS.items():
        product *= len(values)
        print(f"    {name}: {len(values)}")
    print(f"  Raw listed product before cheap constraints: {product:,}")
    print(f"  Generated candidates: {len(candidates)} ({args.num_random} requested random, deduped)")
    print(f"  Profile shapes: {len(profile.shapes())}")
    return 0


def cmd_summarize_cache(args: argparse.Namespace) -> int:
    profile = _profile(args)
    protocol = _protocol(args, profile)
    db = EvoTensileDB.connect(
        args.db,
        environment_compatibility_tag=profile.environment_compatibility_tag,
    )
    db.init()
    summary = db.benchmark_status_summary(
        problem_type_hash=profile.problem_type_hash,
        benchmark_protocol_hash=profile.benchmark_protocol_hash(protocol),
    )
    print(f"db: {args.db}")
    print(f"profile: {profile.name}")
    print(f"problem_type_hash: {profile.problem_type_hash}")
    print(f"benchmark_protocol_hash: {profile.benchmark_protocol_hash(protocol)}")
    print("status counts:")
    if summary:
        for status, count in summary.items():
            print(f"  {status}: {count}")
    else:
        print("  <none>")
    return 0


def cmd_rank_benchmarks(args: argparse.Namespace) -> int:
    profile = _profile(args)
    protocol = _protocol(args, profile)
    db = EvoTensileDB.connect(
        args.db,
        environment_compatibility_tag=profile.environment_compatibility_tag,
    )
    summaries = db.rank_benchmarks(
        problem_type_hash=profile.problem_type_hash,
        benchmark_protocol_hash=profile.benchmark_protocol_hash(protocol),
        shape_id=args.shape_id,
        min_samples=args.min_samples,
        limit=args.limit,
    )
    print("shape_id,candidate_hash,samples,median_gflops,best_gflops,median_time_us,best_time_us")
    for summary in summaries:
        print(
            f"{summary.shape_id},{summary.candidate_hash},{summary.samples},"
            f"{summary.median_gflops if summary.median_gflops is not None else ''},"
            f"{summary.best_gflops if summary.best_gflops is not None else ''},"
            f"{summary.median_time_us if summary.median_time_us is not None else ''},"
            f"{summary.best_time_us if summary.best_time_us is not None else ''}"
        )
    return 0


def _executed_batch_summaries(result: ScheduleResult) -> tuple[list[dict[str, object]], dict[str, int]]:
    summaries: list[dict[str, object]] = []
    total_status_counts: dict[str, int] = {}
    for executed in result.executed_batches:
        ingest = executed.ingest
        status_counts = ingest.status_counts if ingest is not None else {}
        for status, count in status_counts.items():
            total_status_counts[status] = total_status_counts.get(status, 0) + count
        summaries.append(
            {
                "batch_index": executed.planned.batch_index,
                "build_returncode": executed.build_returncode,
                "validation_returncode": executed.validation_returncode,
                "runner_returncode": executed.runner_returncode,
                "phase": executed.phase,
                "requires_validation": executed.planned.requires_validation,
                "yaml_path": str(executed.yaml_path),
                "manifest_path": str(executed.manifest_path),
                "output_dir": str(executed.output_dir),
                "build_output_dir": str(executed.build_output_dir) if executed.build_output_dir is not None else None,
                "ingest": {
                    "inserted": ingest.inserted if ingest is not None else 0,
                    "rejected": ingest.rejected if ingest is not None else 0,
                    "unmapped": ingest.unmapped if ingest is not None else 0,
                    "status_counts": status_counts,
                    "errors": ingest.errors if ingest is not None else [],
                },
            }
        )
    return summaries, total_status_counts


def cmd_schedule_batches(args: argparse.Namespace) -> int:
    context = _schedule_context(args)
    proposal = _propose_candidates_for_shapes(
        context.db,
        args,
        problem_hash=context.problem_hash,
        protocol_hash=context.protocol_hash,
        shapes=context.shapes,
        proposal_shape_id=args.proposal_shape_id,
    )
    candidates = list(proposal.selected)
    requests = [
        PairRequest(
            candidate=candidate,
            shape=shape,
            evidence_stage=EvidenceStage.SCREENING,
            min_samples=args.min_samples,
        )
        for shape in context.shapes
        for candidate in candidates
    ]
    result = _execute_schedule_from_args(args, context, requests=requests)
    print(f"db: {args.db}")
    print(f"output_dir: {args.output_dir}")
    print(f"profile: {context.profile.name}")
    print(f"problem_type_hash: {context.problem_hash}")
    print(f"benchmark_protocol_hash: {context.protocol_hash}")
    print(f"proposal_provider: {proposal.provider['identity']}")
    print(f"candidates: {len(candidates)}")
    print(f"candidate_batch_size: {result.candidate_batch_size}")
    print(f"shape_batch_size: {result.shape_batch_size}")
    print(f"prepare_workers: {result.prepare_workers}")
    if result.runner_bin:
        print(f"runner_bin: {result.runner_bin}")
    print(f"planned batches: {len(result.planned_batches)}")
    print(f"planned requested pairs: {result.requested_pairs}")
    print(f"planned requested samples: {result.requested_samples}")
    for batch in result.planned_batches:
        print(
            f"batch {batch.batch_index:04d}: artifact_candidates={len(batch.artifact_candidates)} "
            f"artifact_shapes={len(batch.artifact_shapes)} requested_pairs={batch.requested_pairs} "
            f"requested_samples={batch.requested_samples} evidence_stage={batch.evidence_stage.value} "
            f"requires_validation={batch.requires_validation}"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    executed_batches, status_counts_total = _executed_batch_summaries(result)
    metadata = _schedule_metadata_common(
        args,
        context,
        result=result,
        proposal=proposal,
        candidates=candidates,
        shapes=context.shapes,
    )
    metadata.update(
        {
            "batches": [
                {
                    "batch_index": batch.batch_index,
                    "artifact_candidates": len(batch.artifact_candidates),
                    "artifact_shapes": len(batch.artifact_shapes),
                    "requested_pairs": batch.requested_pairs,
                    "requested_samples": batch.requested_samples,
                    "evidence_stage": batch.evidence_stage.value,
                    "requires_validation": batch.requires_validation,
                }
                for batch in result.planned_batches
            ],
            "executed_batches": executed_batches,
            "status_counts": status_counts_total,
        }
    )
    metadata_path = output_dir / "schedule_metadata.json"
    metadata["metadata_path"] = str(metadata_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"metadata: {metadata_path}")
    if args.dry_run:
        return 0
    print(f"executed batches: {len(result.executed_batches)}")
    for executed in result.executed_batches:
        ingest = executed.ingest
        inserted = ingest.inserted if ingest is not None else 0
        rejected = ingest.rejected if ingest is not None else 0
        unmapped = ingest.unmapped if ingest is not None else 0
        print(
            f"executed {executed.planned.batch_index:04d}: phase={executed.phase} build={executed.build_returncode} "
            f"validation={executed.validation_returncode} runner={executed.runner_returncode} "
            f"inserted={inserted} rejected={rejected} unmapped={unmapped} "
            f"yaml={executed.yaml_path}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evotensile")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cmd = sub.add_parser("summarize-space", help="Print search-space and generated-candidate summary")
    _add_profile_arg(cmd)
    cmd.add_argument("--num-random", type=int, default=None)
    cmd.add_argument("--seed", type=int, default=12345)
    cmd.set_defaults(func=cmd_summarize_space)

    cmd = sub.add_parser("proposal-coverage", help="Summarize generated proposal coverage without executing")
    cmd.add_argument("--db", required=True)
    _add_candidate_shape_args(cmd)
    _add_profile_arg(cmd)
    _add_protocol_args(cmd)
    _add_proposal_args(cmd)
    cmd.set_defaults(func=cmd_proposal_coverage)

    cmd = sub.add_parser("schedule-batches", help="Cache-aware batch scheduling, build, runner, and ingestion")
    cmd.add_argument("--db", required=True)
    cmd.add_argument("--output-dir", required=True)
    _add_schedule_args(cmd)
    cmd.set_defaults(func=cmd_schedule_batches)

    cmd = sub.add_parser("summarize-cache", help="Summarize benchmark evidence statuses")
    cmd.add_argument("--db", required=True)
    _add_profile_arg(cmd)
    _add_protocol_args(cmd)
    cmd.set_defaults(func=cmd_summarize_cache)

    cmd = sub.add_parser("summarize-families", help="Summarize family archive cells from benchmark evidence")
    cmd.add_argument("--db", required=True)
    _add_candidate_shape_args(cmd)
    _add_profile_arg(cmd)
    _add_protocol_args(cmd)
    cmd.add_argument("--min-samples", type=int, default=1)
    cmd.add_argument("--limit", type=int, default=20)
    cmd.add_argument("--archive-objective", choices=GRID_OBJECTIVES, default=GridObjective.SPECIALIST)
    cmd.set_defaults(func=cmd_summarize_families)

    cmd = sub.add_parser("rank-benchmarks", help="Rank validation-passed benchmark evidence")
    cmd.add_argument("--db", required=True)
    _add_profile_arg(cmd)
    _add_protocol_args(cmd)
    cmd.add_argument("--shape-id", default=None)
    cmd.add_argument("--min-samples", type=int, default=1)
    cmd.add_argument("--limit", type=int, default=20)
    cmd.set_defaults(func=cmd_rank_benchmarks)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
