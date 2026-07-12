import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

from evotensile.campaign.acquisition import select_singleton_bundle_pool
from evotensile.candidate import Candidate, Shape
from evotensile.database import BenchmarkSummary, EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE, TargetProfile
from evotensile.proposal import (
    BUILTIN_PROPOSAL_IDENTITY,
    FamilyQDPolicy,
    ProposalContext,
    ProposalOutput,
    ProposalProvider,
    ProposalResult,
    ProviderProvenance,
    execute_proposal_provider,
    proposal_scope,
)
from evotensile.search.differential_evolution import differential_evolution_candidates
from evotensile.search.evidence import ProposalEvidenceSnapshot, load_proposal_evidence_snapshot
from evotensile.search.family import (
    DEFAULT_FAMILY_ELITES_PER_CELL,
    family_stratified_random_candidates,
    load_family_archive,
)
from evotensile.search.gomea import gomea_candidates, gomea_neighborhood_candidates
from evotensile.search.grid_evidence import GridObjective
from evotensile.search.learned_linkage import LinkageLearningSummary, LinkageModel, learn_linkage_models_from_snapshot
from evotensile.search.local_search import mutate_elites, semantic_mutation_candidates
from evotensile.search.operator_credit import allocate_operator_budget, credit_ucb_scores, load_operator_credit_views
from evotensile.search.shape_neighborhoods import representative_shape_order, shape_distance
from evotensile.search.surrogate import select_surrogate_pool
from evotensile.search_space import eligible_for_shape_scope
from evotensile.shapes import shape_from_id
from evotensile.utils import dedupe_candidates


class GridCandidateEvidence:
    def __init__(
        self,
        candidate_hash: str,
        shape_regrets: tuple[tuple[str, float, float], ...],
        samples: int,
    ):
        self.candidate_hash = candidate_hash
        self.shape_regrets = shape_regrets
        self.samples = samples

    @property
    def coverage(self) -> int:
        return len(self.shape_regrets)

    @property
    def weighted_coverage(self) -> float:
        return sum(weight for _, _, weight in self.shape_regrets)

    @property
    def mean_regret(self) -> float:
        total_weight = self.weighted_coverage
        if total_weight <= 0.0:
            return math.inf
        return sum(regret * weight for _, regret, weight in self.shape_regrets) / total_weight

    @property
    def specialist_regret(self) -> float:
        return min((regret for _, regret, _ in self.shape_regrets), default=math.inf)


def grid_candidate_evidence(
    summaries: Sequence[BenchmarkSummary],
    *,
    shape_weights: Mapping[str, float] | None = None,
) -> list[GridCandidateEvidence]:
    incumbent_time_by_shape: dict[str, float] = {}
    for summary in summaries:
        if summary.median_time_us is None or summary.median_time_us <= 0.0:
            continue
        incumbent_time_by_shape[summary.shape_id] = min(
            incumbent_time_by_shape.get(summary.shape_id, math.inf), summary.median_time_us
        )
    regrets_by_candidate: dict[str, list[tuple[str, float, float]]] = {}
    samples_by_candidate: dict[str, int] = {}
    for summary in summaries:
        incumbent = incumbent_time_by_shape.get(summary.shape_id)
        if incumbent is None or summary.median_time_us is None or summary.median_time_us <= 0.0:
            continue
        regrets_by_candidate.setdefault(summary.candidate_hash, []).append(
            (
                summary.shape_id,
                summary.median_time_us / incumbent - 1.0,
                float((shape_weights or {}).get(summary.shape_id, 1.0)),
            )
        )
        samples_by_candidate[summary.candidate_hash] = (
            samples_by_candidate.get(summary.candidate_hash, 0) + summary.samples
        )
    return [
        GridCandidateEvidence(
            candidate_hash,
            tuple(sorted(shape_regrets)),
            samples_by_candidate[candidate_hash],
        )
        for candidate_hash, shape_regrets in regrets_by_candidate.items()
    ]


def ranked_elites(
    evidence: ProposalEvidenceSnapshot,
    *,
    shape_id: str | None,
    target_shapes: Sequence[Shape],
    elite_count: int,
    shape_weights: Mapping[str, float] | None = None,
) -> list[Candidate]:
    summaries = list(evidence.shape_summaries(shape_id))
    if shape_id is not None:
        return [
            evidence.candidates[summary.candidate_hash]
            for summary in summaries[:elite_count]
            if summary.candidate_hash in evidence.candidates
        ]
    target_shape_ids = {shape.id for shape in target_shapes}
    if target_shape_ids:
        summaries = [summary for summary in summaries if summary.shape_id in target_shape_ids]
    candidate_evidence = grid_candidate_evidence(
        summaries,
        shape_weights=shape_weights,
    )
    specialist_count = min(len(candidate_evidence), (elite_count + 1) // 2)
    summaries_by_shape: dict[str, list[BenchmarkSummary]] = {}
    for summary in summaries:
        summaries_by_shape.setdefault(summary.shape_id, []).append(summary)
    for shape_summaries in summaries_by_shape.values():
        shape_summaries.sort(key=lambda item: (item.median_time_us or math.inf, item.candidate_hash))
    representative_rank = {shape.id: index for index, shape in enumerate(representative_shape_order(target_shapes))}
    ranked_shapes = sorted(
        (shape for shape in target_shapes if shape.id in summaries_by_shape),
        key=lambda shape: (
            -float((shape_weights or {}).get(shape.id, 1.0)),
            representative_rank[shape.id],
        ),
    )
    if not ranked_shapes:
        ranked_shapes = representative_shape_order([shape_from_id(item) for item in summaries_by_shape])
    selected_hashes: list[str] = []
    rank_index = 0
    while len(selected_hashes) < specialist_count:
        added = False
        for shape in ranked_shapes:
            shape_summaries = summaries_by_shape[shape.id]
            if rank_index >= len(shape_summaries):
                continue
            candidate_hash = shape_summaries[rank_index].candidate_hash
            if candidate_hash not in selected_hashes:
                selected_hashes.append(candidate_hash)
                added = True
                if len(selected_hashes) >= specialist_count:
                    break
        if not added:
            break
        rank_index += 1
    generalist_order = sorted(
        (item for item in candidate_evidence if item.candidate_hash not in selected_hashes),
        key=lambda item: (
            -item.weighted_coverage,
            item.mean_regret,
            item.specialist_regret,
            item.candidate_hash,
        ),
    )
    selected_hashes.extend(
        item.candidate_hash for item in generalist_order[: max(0, elite_count - len(selected_hashes))]
    )
    return [
        evidence.candidates[candidate_hash]
        for candidate_hash in selected_hashes
        if candidate_hash in evidence.candidates
    ]


def learned_linkage_models(
    evidence: ProposalEvidenceSnapshot,
    *,
    enabled: bool,
    target_shapes: Sequence[Shape],
    policy: FamilyQDPolicy,
) -> tuple[list[LinkageModel], LinkageLearningSummary]:
    if not enabled:
        return [], LinkageLearningSummary(False, 0, 0, 0, "disabled")
    return learn_linkage_models_from_snapshot(
        evidence,
        shapes=list(target_shapes),
        truncation_tau=policy.linkage_truncation_tau,
        min_samples=policy.linkage_min_samples,
        max_clusters=policy.linkage_max_clusters,
        ordinal_bins=policy.linkage_ordinal_bins,
    )


def nearest_source_shape_ids(target: Shape, source_shape_ids: set[str], *, limit: int) -> list[str]:
    ranked: list[tuple[float, str]] = []
    for shape_id in source_shape_ids:
        try:
            source = shape_from_id(shape_id)
        except ValueError:
            continue
        ranked.append((shape_distance(source, target), source.id))
    return [shape_id for _, shape_id in sorted(ranked)[:limit]]


def transfer_elites(
    evidence: ProposalEvidenceSnapshot,
    *,
    target_shapes: Sequence[Shape],
    nearest_shape_count: int,
    per_shape: int,
) -> list[Candidate]:
    if nearest_shape_count <= 0 or per_shape <= 0 or not target_shapes:
        return []
    source_shape_ids = {summary.shape_id for summary in evidence.summaries}
    queues: list[list[tuple[str, str, str]]] = []
    for target in representative_shape_order(target_shapes):
        queue: list[tuple[str, str, str]] = []
        for source_shape_id in nearest_source_shape_ids(target, source_shape_ids, limit=nearest_shape_count):
            for summary in evidence.shape_summaries(source_shape_id)[:per_shape]:
                queue.append((summary.candidate_hash, target.id, source_shape_id))
        queues.append(queue)
    selected_causes: list[tuple[str, str, str]] = []
    seen_hashes: set[str] = set()
    global_cap = nearest_shape_count * per_shape
    while queues and len(selected_causes) < global_cap:
        remaining = []
        for queue in queues:
            while queue and queue[0][0] in seen_hashes:
                queue.pop(0)
            if queue and len(selected_causes) < global_cap:
                cause = queue.pop(0)
                selected_causes.append(cause)
                seen_hashes.add(cause[0])
            if queue:
                remaining.append(queue)
        queues = remaining
    return [
        Candidate(
            params=evidence.candidates[candidate_hash].canonical_params(),
            source="transfer",
            parent_hashes=(candidate_hash,),
            proposal_metadata={
                "transfer_target_shape_ids": [target_shape_id],
                "transfer_source_shape_ids": [source_shape_id],
            },
        )
        for candidate_hash, target_shape_id, source_shape_id in selected_causes
        if candidate_hash in evidence.candidates
    ]


def family_archive_leaders(
    evidence: ProposalEvidenceSnapshot,
    *,
    shape_id: str | None,
    target_shapes: Sequence[Shape],
    elite_count: int,
    shape_weights: Mapping[str, float] | None = None,
) -> list[Candidate]:
    if elite_count <= 0:
        return []
    archive_shapes = list(target_shapes)
    if shape_id is not None:
        archive_shapes = [shape for shape in archive_shapes if shape.id == shape_id]
    objectives = (
        (GridObjective.SPECIALIST, GridObjective.GENERALIST, GridObjective.COVERAGE, GridObjective.UNCERTAINTY)
        if len(archive_shapes) > 1
        else (GridObjective.SPECIALIST,)
    )
    entries_by_objective = [
        load_family_archive(
            evidence,
            shapes=archive_shapes,
            min_samples=1,
            objective=objective,
            limit=elite_count,
            elites_per_family=min(DEFAULT_FAMILY_ELITES_PER_CELL, elite_count),
            shape_weights=shape_weights,
        )
        for objective in objectives
    ]
    leaders: list[Candidate] = []
    rank = 0
    while len(leaders) < elite_count and any(rank < len(entries) for entries in entries_by_objective):
        leaders.extend(entries[rank].leader for entries in entries_by_objective if rank < len(entries))
        leaders = dedupe_candidates(leaders)
        rank += 1
    return leaders[:elite_count]


def family_qd_provider(context: ProposalContext) -> ProposalOutput:
    policy = context.family_qd_policy
    if policy is None:
        raise ValueError("built-in family-QD requires a FamilyQDPolicy")
    shapes = list(context.shapes)
    evidence = context.evidence
    multiplier = max(1, policy.surrogate_pool_multiplier)
    pool_random = policy.num_random * multiplier
    pool_local = policy.local_count * multiplier
    pool_de = policy.de_count * multiplier
    pool_gomea = policy.gomea_count * multiplier

    supplied_parents = dedupe_candidates(list(context.parent_candidates or ()))
    if context.parent_candidates is None:
        elites = ranked_elites(
            evidence,
            shape_id=context.shape_id,
            target_shapes=shapes,
            elite_count=policy.elite_count,
            shape_weights=context.shape_weights or None,
        )
        transferred = (
            transfer_elites(
                evidence,
                target_shapes=shapes,
                nearest_shape_count=policy.transfer_shape_count,
                per_shape=policy.transfer_per_shape,
            )
            if context.shape_id is None
            else []
        )
        archive = family_archive_leaders(
            evidence,
            shape_id=context.shape_id,
            target_shapes=shapes,
            elite_count=policy.elite_count,
            shape_weights=context.shape_weights or None,
        )
    else:
        elites = supplied_parents
        transferred = []
        archive = []

    preserved = [
        candidate
        for candidate in dedupe_candidates([*transferred, *archive, *supplied_parents])
        if eligible_for_shape_scope(candidate.canonical_params(), shapes)
    ]
    elites = [
        candidate
        for candidate in dedupe_candidates([*archive, *elites, *transferred])
        if eligible_for_shape_scope(candidate.canonical_params(), shapes)
    ]
    candidates = list(preserved)
    credits = (
        load_operator_credit_views(
            evidence,
            shapes=shapes,
            shape_weights=context.shape_weights or None,
        )
        if policy.adaptive_operators
        else None
    )
    allocation = (
        allocate_operator_budget(
            pool_local + pool_de + pool_gomea,
            credits.operator,
            cost_aware=policy.cost_aware_operator_credit,
        )
        if credits is not None
        else None
    )
    group_weights = (
        credit_ucb_scores(dict(credits.semantic_group), cost_aware=policy.cost_aware_operator_credit)
        if credits is not None and policy.adaptive_group_credit
        else None
    )
    donor_weights = (
        credit_ucb_scores(dict(credits.donor_mode), cost_aware=policy.cost_aware_operator_credit)
        if credits is not None and policy.adaptive_donor_selection
        else None
    )

    candidates.extend(
        family_stratified_random_candidates(
            evidence,
            pool_random,
            seed=context.seed,
            target_shapes=shapes,
        )
    )
    mutation_budget = allocation["semantic-mutation"] if allocation is not None else pool_local
    if mutation_budget > 0:
        if allocation is None:
            candidates.extend(
                mutate_elites(
                    elites,
                    count=mutation_budget,
                    seed=context.seed + 1009,
                    mutation_rate=policy.mutation_rate,
                    target_shapes=shapes,
                )
            )
        else:
            candidates.extend(
                semantic_mutation_candidates(
                    elites,
                    count=mutation_budget,
                    seed=context.seed + 1009,
                    target_shapes=shapes,
                    exclude={candidate.hash for candidate in candidates},
                    group_weights=group_weights,
                )
            )
    de_budget = allocation["de"] if allocation is not None else pool_de
    candidates.extend(
        differential_evolution_candidates(
            elites,
            count=de_budget,
            seed=context.seed + 2003,
            crossover_rate=policy.crossover_rate,
            random_gene_rate=policy.random_gene_rate,
            exclude={candidate.hash for candidate in candidates},
            target_shapes=shapes,
        )
    )
    linkage, linkage_summary = learned_linkage_models(
        evidence,
        enabled=policy.learned_linkage,
        target_shapes=shapes,
        policy=policy,
    )
    if allocation is None:
        neighborhood_budget = pool_gomea // 2
        mixing_budget = pool_gomea - neighborhood_budget
    else:
        neighborhood_budget = allocation["gomea-neighborhood"]
        mixing_budget = allocation["gomea-mixing"]
    candidates.extend(
        gomea_neighborhood_candidates(
            elites,
            count=neighborhood_budget,
            max_elites=None,
            exclude={candidate.hash for candidate in candidates},
            seed=context.seed + 2903,
            source="gomea-neighborhood" if allocation is not None else "gomea",
            target_shapes=shapes,
            group_weights=group_weights,
            micro_exhaustive=policy.micro_exhaustive_neighborhoods,
        )
    )
    candidates.extend(
        gomea_candidates(
            elites,
            count=mixing_budget,
            seed=context.seed + 3001,
            elite_count=policy.elite_count,
            exclude={candidate.hash for candidate in candidates},
            target_shapes=shapes,
            linkage_models=linkage,
            family_local_probability=0.8 if allocation is not None else 0.0,
            source="gomea-mixing" if allocation is not None else "gomea",
            donor_mode_weights=donor_weights,
            adaptive_donor_selection=policy.adaptive_donor_selection,
        )
    )

    pool = dedupe_candidates(candidates)
    preserved_hashes = {candidate.hash for candidate in preserved}
    generated = [candidate for candidate in pool if candidate.hash not in preserved_hashes]
    selection_method = "unfiltered"
    if multiplier <= 1:
        selected = pool
    else:
        variation_budget = policy.local_count + policy.de_count + policy.gomea_count if elites else 0
        selected_count = policy.num_random + variation_budget
        selected_generated = None
        if policy.singleton_acquisition_enabled and len(shapes) == 1:
            selected_generated = select_singleton_bundle_pool(
                generated,
                evidence=evidence,
                shape=shapes[0],
                count=selected_count,
                seed=context.seed + 4001,
                workgroup_processor_count=context.target_profile.workgroup_processor_count,
                jobs=context.target_profile.default_surrogate_jobs,
                min_evidence=policy.surrogate_min_evidence,
                information_weight=policy.singleton_information_weight,
            )
            if selected_generated is not None:
                selection_method = "singleton-bundle-acquisition"
        if selected_generated is None:
            selected_generated = select_surrogate_pool(
                generated,
                evidence=evidence,
                shapes=shapes,
                count=selected_count,
                seed=context.seed + 4001,
                min_evidence=policy.surrogate_min_evidence,
                covering_cold_start=policy.covering_cold_start,
                cold_start_precovered_tokens=set(context.cold_start_precovered_tokens),
                surrogate_jobs=context.target_profile.default_surrogate_jobs,
                workgroup_processor_count=context.target_profile.workgroup_processor_count,
            )
            selection_method = "surrogate"
        selected = dedupe_candidates([*preserved, *selected_generated])
    return ProposalOutput(
        candidates=tuple(pool),
        selected_candidate_hashes=tuple(candidate.hash for candidate in selected),
        metadata={
            "policy": policy.to_dict(),
            "shape_weighted": bool(context.shape_weights),
            "shape_weights": dict(sorted(context.shape_weights.items())),
            "selection_method": selection_method,
            "linkage": {
                "enabled": linkage_summary.enabled,
                "evidence_count": linkage_summary.evidence_count,
                "selected_count": linkage_summary.selected_count,
                "model_count": linkage_summary.model_count,
                "fallback_reason": linkage_summary.fallback_reason,
            },
        },
    )


def propose_candidates(
    db: EvoTensileDB,
    *,
    target_profile: TargetProfile = DEFAULT_PROFILE,
    policy: FamilyQDPolicy | None = None,
    provider: ProposalProvider | None = None,
    provider_provenance: ProviderProvenance | None = None,
    provider_config: Mapping[str, object] | None = None,
    seed: int = 1,
    problem_type_hash: str | None = None,
    benchmark_protocol_hash: str | None = None,
    shape_id: str | None = None,
    target_shapes: Sequence[Shape] = (),
    shape_weights: Mapping[str, float] | None = None,
    scope_kind: str | None = None,
    parent_candidates: Sequence[Candidate] | None = None,
    cold_start_precovered_tokens: set[str] | None = None,
    proposal_island_id: str | None = None,
    proposal_restart_index: int = 0,
) -> ProposalResult:
    if provider is not None and policy is not None:
        raise ValueError("custom proposal providers cannot use family-QD policy settings")
    if provider is None:
        policy = policy or FamilyQDPolicy()
        provider = family_qd_provider
        provider_provenance = ProviderProvenance(identity=BUILTIN_PROPOSAL_IDENTITY)
    elif provider_provenance is None:
        raise ValueError("custom proposal provider provenance is required")
    shapes = tuple(target_shapes)
    resolved_shape_weights = {}
    if shape_weights is not None:
        if set(shape_weights) != {shape.id for shape in shapes}:
            raise ValueError("proposal shape weights must cover the exact target shape set")
        resolved_shape_weights = {shape.id: float(shape_weights[shape.id]) for shape in shapes}
        if any(not math.isfinite(value) or value < 0.0 for value in resolved_shape_weights.values()):
            raise ValueError("proposal shape weights must be finite and nonnegative")
    resolved_problem_hash = problem_type_hash or target_profile.problem_type_hash
    resolved_protocol_hash = benchmark_protocol_hash or target_profile.benchmark_protocol_hash()
    evidence = load_proposal_evidence_snapshot(
        db,
        problem_type_hash=resolved_problem_hash,
        benchmark_protocol_hash=resolved_protocol_hash,
        shapes=shapes,
    )
    context = ProposalContext(
        target_profile=target_profile,
        shapes=shapes,
        scope=proposal_scope(shapes, scope_kind),
        seed=seed,
        evidence=evidence,
        shape_weights=resolved_shape_weights,
        config=provider_config or {},
        family_qd_policy=policy,
        shape_id=shape_id,
        parent_candidates=None if parent_candidates is None else tuple(parent_candidates),
        cold_start_precovered_tokens=frozenset(cold_start_precovered_tokens or ()),
        island_id=proposal_island_id,
        restart_index=proposal_restart_index,
    )
    assert provider_provenance is not None
    return execute_proposal_provider(
        db,
        context=context,
        provider=provider,
        provenance=provider_provenance,
    )


def policy_with_overrides(policy: FamilyQDPolicy, **overrides: object) -> FamilyQDPolicy:
    return replace(policy, **overrides)
