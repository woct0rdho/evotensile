import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from evotensile.candidate import Candidate, Shape
from evotensile.database import EvaluationSummary, EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE, TargetProfile
from evotensile.search.differential_evolution import differential_evolution_candidates
from evotensile.search.evidence import ProposalEvidenceSnapshot, load_proposal_evidence_snapshot
from evotensile.search.family import (
    DEFAULT_FAMILY_ELITES_PER_CELL,
    family_stratified_random_candidates,
    load_family_archive,
)
from evotensile.search.gomea import gomea_candidates, gomea_neighborhood_candidates
from evotensile.search.grid_evidence import GridObjective
from evotensile.search.learned_linkage import (
    DEFAULT_MAX_CLUSTERS,
    DEFAULT_MIN_LINKAGE_SAMPLES,
    DEFAULT_ORDINAL_BINS,
    DEFAULT_TRUNCATION_TAU,
    LinkageLearningSummary,
    LinkageModel,
    learn_linkage_models_from_snapshot,
)
from evotensile.search.local_search import mutate_elites, semantic_mutation_candidates
from evotensile.search.operator_credit import (
    allocate_operator_budget,
    credit_ucb_scores,
    load_operator_credit_views,
)
from evotensile.search.random_search import initial_random_batch
from evotensile.search.shape_neighborhoods import representative_shape_order, shape_distance
from evotensile.search.surrogate import DEFAULT_SURROGATE_MIN_EVIDENCE, select_surrogate_pool
from evotensile.search_space import random_candidate
from evotensile.shapes import shape_from_id
from evotensile.utils import dedupe_candidates


@dataclass(frozen=True)
class ProposalScope:
    kind: str
    shape_ids: tuple[str, ...]


@dataclass(frozen=True)
class ProposalResult:
    scope: ProposalScope
    preserved: tuple[Candidate, ...]
    generated: tuple[Candidate, ...]
    selected: tuple[Candidate, ...]


@dataclass(frozen=True)
class GridCandidateEvidence:
    candidate_hash: str
    shape_regrets: tuple[tuple[str, float], ...]
    samples: int

    @property
    def coverage(self) -> int:
        return len(self.shape_regrets)

    @property
    def mean_regret(self) -> float:
        return sum(regret for _, regret in self.shape_regrets) / max(self.coverage, 1)

    @property
    def specialist_regret(self) -> float:
        return min((regret for _, regret in self.shape_regrets), default=math.inf)


PROPOSAL_MODES = (
    "random",
    "seed-random",
    "local",
    "seed-random-local",
    "de",
    "seed-random-de",
    "gomea",
    "seed-random-gomea",
    "evolutionary",
    "family-qd",
)

DEFAULT_LEARNED_LINKAGE_ENABLED = True

DEFAULT_LINKAGE_TRUNCATION_TAU = DEFAULT_TRUNCATION_TAU

DEFAULT_LINKAGE_MIN_SAMPLES = DEFAULT_MIN_LINKAGE_SAMPLES

DEFAULT_LINKAGE_MAX_CLUSTERS = DEFAULT_MAX_CLUSTERS

DEFAULT_LINKAGE_ORDINAL_BINS = DEFAULT_ORDINAL_BINS


def _grid_candidate_evidence(summaries: Sequence[EvaluationSummary]) -> list[GridCandidateEvidence]:
    incumbent_time_by_shape: dict[str, float] = {}
    for summary in summaries:
        if summary.median_time_us is None or summary.median_time_us <= 0.0:
            continue
        incumbent_time_by_shape[summary.shape_id] = min(
            incumbent_time_by_shape.get(summary.shape_id, math.inf), summary.median_time_us
        )
    regrets_by_candidate: dict[str, list[tuple[str, float]]] = {}
    samples_by_candidate: dict[str, int] = {}
    for summary in summaries:
        incumbent = incumbent_time_by_shape.get(summary.shape_id)
        if incumbent is None or summary.median_time_us is None or summary.median_time_us <= 0.0:
            continue
        regrets_by_candidate.setdefault(summary.candidate_hash, []).append(
            (summary.shape_id, summary.median_time_us / incumbent - 1.0)
        )
        samples_by_candidate[summary.candidate_hash] = (
            samples_by_candidate.get(summary.candidate_hash, 0) + summary.samples
        )
    return [
        GridCandidateEvidence(
            candidate_hash=candidate_hash,
            shape_regrets=tuple(sorted(shape_regrets)),
            samples=samples_by_candidate[candidate_hash],
        )
        for candidate_hash, shape_regrets in regrets_by_candidate.items()
    ]


def _ranked_elites(
    evidence: ProposalEvidenceSnapshot,
    *,
    shape_id: str | None,
    target_shapes: Sequence[Shape] | None,
    elite_count: int,
) -> list[Candidate]:
    summaries = list(evidence.shape_summaries(shape_id))
    if shape_id is not None:
        return [
            evidence.candidates[summary.candidate_hash]
            for summary in summaries[:elite_count]
            if summary.candidate_hash in evidence.candidates
        ]
    target_shape_ids = {shape.id for shape in target_shapes or ()}
    if target_shape_ids:
        summaries = [summary for summary in summaries if summary.shape_id in target_shape_ids]
    grid_evidence = _grid_candidate_evidence(summaries)
    specialist_count = min(len(grid_evidence), (elite_count + 1) // 2)
    summaries_by_shape: dict[str, list[EvaluationSummary]] = {}
    for summary in summaries:
        summaries_by_shape.setdefault(summary.shape_id, []).append(summary)
    for shape_summaries in summaries_by_shape.values():
        shape_summaries.sort(key=lambda item: (item.median_time_us or math.inf, item.candidate_hash))
    ranked_shapes = [
        shape for shape in representative_shape_order(target_shapes or ()) if shape.id in summaries_by_shape
    ]
    if not ranked_shapes:
        ranked_shapes = representative_shape_order([shape_from_id(shape_id) for shape_id in summaries_by_shape])
    selected_hashes: list[str] = []
    rank_index = 0
    while len(selected_hashes) < specialist_count:
        added = False
        for shape in ranked_shapes:
            shape_id = shape.id
            shape_summaries = summaries_by_shape[shape_id]
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
        (item for item in grid_evidence if item.candidate_hash not in selected_hashes),
        key=lambda item: (-item.coverage, item.mean_regret, item.specialist_regret, item.candidate_hash),
    )
    selected_hashes.extend(
        item.candidate_hash for item in generalist_order[: max(0, elite_count - len(selected_hashes))]
    )
    return [
        evidence.candidates[candidate_hash]
        for candidate_hash in selected_hashes
        if candidate_hash in evidence.candidates
    ]


def _learned_linkage_models_for_proposal(
    evidence: ProposalEvidenceSnapshot,
    *,
    enabled: bool,
    target_shapes: list[Shape] | None,
    min_samples: int,
    truncation_tau: float,
    max_clusters: int,
    ordinal_bins: int,
) -> tuple[list[LinkageModel], LinkageLearningSummary]:
    if not enabled:
        return [], LinkageLearningSummary(False, 0, 0, 0, "disabled")
    return learn_linkage_models_from_snapshot(
        evidence,
        shapes=target_shapes,
        truncation_tau=truncation_tau,
        min_samples=min_samples,
        max_clusters=max_clusters,
        ordinal_bins=ordinal_bins,
    )


def _nearest_source_shape_ids(target: Shape, source_shape_ids: set[str], *, limit: int) -> list[str]:
    ranked: list[tuple[float, str]] = []
    for shape_id in source_shape_ids:
        try:
            source = shape_from_id(shape_id)
        except ValueError:
            continue
        ranked.append((shape_distance(source, target), source.id))
    return [shape_id for _, shape_id in sorted(ranked)[:limit]]


def _transfer_elites(
    evidence: ProposalEvidenceSnapshot,
    *,
    target_shapes: list[Shape],
    nearest_shape_count: int,
    per_shape: int,
) -> list[Candidate]:
    if nearest_shape_count <= 0 or per_shape <= 0 or not target_shapes:
        return []
    summaries = evidence.summaries
    source_shape_ids = {summary.shape_id for summary in summaries}
    queues: list[list[tuple[str, str, str]]] = []
    for target in representative_shape_order(target_shapes):
        queue: list[tuple[str, str, str]] = []
        for source_shape_id in _nearest_source_shape_ids(target, source_shape_ids, limit=nearest_shape_count):
            source_summaries = evidence.shape_summaries(source_shape_id)[:per_shape]
            for summary in source_summaries:
                queue.append((summary.candidate_hash, target.id, source_shape_id))
        queues.append(queue)
    selected_causes: list[tuple[str, str, str]] = []
    seen_hashes: set[str] = set()
    global_cap = nearest_shape_count * per_shape
    while queues and len(selected_causes) < global_cap:
        remaining: list[list[tuple[str, str, str]]] = []
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
    candidates_by_hash = {
        candidate.hash: candidate
        for candidate_hash, _, _ in selected_causes
        if (candidate := evidence.candidates.get(candidate_hash)) is not None
    }
    return [
        Candidate(
            params=candidates_by_hash[candidate_hash].canonical_params(),
            source="transfer",
            parent_hashes=(candidate_hash,),
            proposal_metadata={
                "transfer_target_shape_ids": [target_shape_id],
                "transfer_source_shape_ids": [source_shape_id],
            },
        )
        for candidate_hash, target_shape_id, source_shape_id in selected_causes
    ]


def _scoped_random_batch(num_random: int, *, seed: int, target_shapes: list[Shape] | None) -> list[Candidate]:
    if not target_shapes:
        return initial_random_batch(num_random, seed=seed)
    rng = random.Random(seed)
    out: dict[str, Candidate] = {}
    while len(out) < num_random:
        candidate = random_candidate(rng, target_shapes=target_shapes)
        out[candidate.hash] = candidate
    return list(out.values())


def _proposal_scope(target_shapes: Sequence[Shape] | None, scope_kind: str | None) -> ProposalScope:
    shape_ids = tuple(shape.id for shape in target_shapes or ())
    inferred = "global" if not shape_ids else ("shape" if len(shape_ids) == 1 else "shape-set")
    kind = scope_kind or inferred
    if kind not in {"global", "shape", "cluster", "shape-set"}:
        raise ValueError(f"unknown proposal scope kind: {kind}")
    if kind == "global" and shape_ids:
        raise ValueError("global proposal scope cannot contain shapes")
    if kind != "global" and not shape_ids:
        raise ValueError(f"{kind} proposal scope requires at least one shape")
    if kind == "shape" and len(shape_ids) != 1:
        raise ValueError("shape proposal scope requires exactly one shape")
    return ProposalScope(kind=kind, shape_ids=shape_ids)


def _family_archive_leaders(
    evidence: ProposalEvidenceSnapshot,
    *,
    shape_id: str | None,
    target_shapes: list[Shape] | None,
    elite_count: int,
) -> list[Candidate]:
    if elite_count <= 0:
        return []
    archive_shapes = target_shapes
    if shape_id is not None and target_shapes:
        archive_shapes = [shape for shape in target_shapes if shape.id == shape_id]
    objectives = (
        (GridObjective.SPECIALIST, GridObjective.GENERALIST, GridObjective.COVERAGE, GridObjective.UNCERTAINTY)
        if archive_shapes and len(archive_shapes) > 1
        else (GridObjective.SPECIALIST,)
    )
    objective_entries = [
        load_family_archive(
            evidence,
            shapes=archive_shapes,
            min_samples=1,
            objective=objective,
            limit=elite_count,
            elites_per_family=min(DEFAULT_FAMILY_ELITES_PER_CELL, elite_count),
        )
        for objective in objectives
    ]
    leaders: list[Candidate] = []
    rank = 0
    while len(leaders) < elite_count and any(rank < len(entries) for entries in objective_entries):
        for entries in objective_entries:
            if rank < len(entries):
                leaders.append(entries[rank].leader)
        leaders = dedupe_candidates(leaders)
        rank += 1
    return leaders[:elite_count]


def propose_candidates(
    db: EvoTensileDB,
    *,
    target_profile: TargetProfile = DEFAULT_PROFILE,
    proposal: str | None = None,
    num_random: int | None = None,
    seed: int = 1,
    problem_type_hash: str | None = None,
    benchmark_protocol_hash: str | None = None,
    shape_id: str | None = None,
    target_shapes: list[Shape] | None = None,
    scope_kind: str | None = None,
    transfer_shape_count: int | None = None,
    transfer_per_shape: int | None = None,
    elite_count: int | None = None,
    local_count: int | None = None,
    de_count: int | None = None,
    gomea_count: int | None = None,
    mutation_rate: float | None = None,
    crossover_rate: float | None = None,
    random_gene_rate: float | None = None,
    learned_linkage: bool = DEFAULT_LEARNED_LINKAGE_ENABLED,
    linkage_truncation_tau: float = DEFAULT_LINKAGE_TRUNCATION_TAU,
    linkage_min_samples: int = DEFAULT_LINKAGE_MIN_SAMPLES,
    linkage_max_clusters: int = DEFAULT_LINKAGE_MAX_CLUSTERS,
    linkage_ordinal_bins: int = DEFAULT_LINKAGE_ORDINAL_BINS,
    adaptive_operators: bool = False,
    surrogate_pool_multiplier: int = 1,
    surrogate_min_evidence: int = DEFAULT_SURROGATE_MIN_EVIDENCE,
    covering_cold_start: bool = False,
    adaptive_group_credit: bool = False,
    micro_exhaustive_neighborhoods: bool = False,
    adaptive_donor_selection: bool = False,
    cost_aware_operator_credit: bool = False,
    surrogate_jobs: int | None = None,
    workgroup_processor_count: int | None = None,
    parent_candidates: Sequence[Candidate] | None = None,
    cold_start_precovered_tokens: set[str] | None = None,
) -> ProposalResult:
    """Build and classify preserved, generated, and selected candidates."""
    proposal = target_profile.default_proposal if proposal is None else proposal
    num_random = target_profile.default_num_random if num_random is None else num_random
    transfer_shape_count = (
        target_profile.default_transfer_shapes if transfer_shape_count is None else transfer_shape_count
    )
    transfer_per_shape = target_profile.default_transfer_per_shape if transfer_per_shape is None else transfer_per_shape
    elite_count = target_profile.default_elite_count if elite_count is None else elite_count
    local_count = target_profile.default_local_count if local_count is None else local_count
    de_count = target_profile.default_de_count if de_count is None else de_count
    gomea_count = target_profile.default_gomea_count if gomea_count is None else gomea_count
    mutation_rate = target_profile.default_mutation_rate if mutation_rate is None else mutation_rate
    crossover_rate = target_profile.default_crossover_rate if crossover_rate is None else crossover_rate
    random_gene_rate = target_profile.default_random_gene_rate if random_gene_rate is None else random_gene_rate
    surrogate_jobs = target_profile.default_surrogate_jobs if surrogate_jobs is None else surrogate_jobs
    workgroup_processor_count = (
        target_profile.workgroup_processor_count if workgroup_processor_count is None else workgroup_processor_count
    )
    if proposal not in PROPOSAL_MODES:
        raise ValueError(f"unknown proposal mode: {proposal}")
    evidence = load_proposal_evidence_snapshot(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        shapes=target_shapes,
    )
    scope = _proposal_scope(target_shapes, scope_kind)
    pool_multiplier = max(1, surrogate_pool_multiplier)
    pool_num_random = num_random * pool_multiplier
    pool_local_count = local_count * pool_multiplier
    pool_de_count = de_count * pool_multiplier
    pool_gomea_count = gomea_count * pool_multiplier

    candidates: list[Candidate] = []
    uses_random = proposal in {
        "random",
        "seed-random",
        "seed-random-local",
        "seed-random-de",
        "seed-random-gomea",
        "evolutionary",
        "family-qd",
    }
    needs_elites = proposal in {
        "local",
        "seed-random-local",
        "de",
        "seed-random-de",
        "gomea",
        "seed-random-gomea",
        "evolutionary",
        "family-qd",
    }
    supplied_parents = dedupe_candidates(list(parent_candidates or ()))
    elites = (
        supplied_parents
        if needs_elites and parent_candidates is not None
        else _ranked_elites(
            evidence,
            shape_id=shape_id,
            target_shapes=target_shapes,
            elite_count=elite_count,
        )
        if needs_elites
        else []
    )
    transfer_elites = (
        _transfer_elites(
            evidence,
            target_shapes=target_shapes or [],
            nearest_shape_count=transfer_shape_count,
            per_shape=transfer_per_shape,
        )
        if needs_elites and shape_id is None and parent_candidates is None
        else []
    )
    if transfer_elites:
        # Nearby winners should be evaluated before random restarts, especially when candidate batches are truncated.
        candidates.extend(transfer_elites)
        elites = dedupe_candidates([*elites, *transfer_elites])

    family_leaders: list[Candidate] = []
    if proposal == "family-qd" and parent_candidates is None:
        family_leaders = _family_archive_leaders(
            evidence,
            shape_id=shape_id,
            target_shapes=target_shapes,
            elite_count=elite_count,
        )
        candidates.extend(family_leaders)
        elites = dedupe_candidates([*family_leaders, *elites])
    elif supplied_parents:
        candidates.extend(supplied_parents)

    operator_allocation: dict[str, int] | None = None
    semantic_group_weights: dict[str, float] | None = None
    donor_mode_weights: dict[str, float] | None = None
    if adaptive_operators and proposal == "family-qd":
        credit_views = load_operator_credit_views(evidence, shapes=target_shapes)
        operator_credits = credit_views.operator
        operator_allocation = allocate_operator_budget(
            pool_local_count + pool_de_count + pool_gomea_count,
            operator_credits,
            cost_aware=cost_aware_operator_credit,
        )
        if adaptive_group_credit:
            semantic_group_weights = credit_ucb_scores(
                dict(credit_views.semantic_group),
                cost_aware=cost_aware_operator_credit,
            )
        if adaptive_donor_selection:
            donor_mode_weights = credit_ucb_scores(
                dict(credit_views.donor_mode),
                cost_aware=cost_aware_operator_credit,
            )

    if uses_random:
        random_batch = (
            family_stratified_random_candidates(
                evidence,
                pool_num_random,
                seed=seed,
                target_shapes=target_shapes,
            )
            if proposal == "family-qd"
            else _scoped_random_batch(pool_num_random, seed=seed, target_shapes=target_shapes)
        )
        candidates.extend(random_batch)

    mutation_budget = operator_allocation["semantic-mutation"] if operator_allocation is not None else pool_local_count
    if proposal in {"local", "seed-random-local", "evolutionary", "family-qd"} and mutation_budget > 0:
        if operator_allocation is not None:
            candidates.extend(
                semantic_mutation_candidates(
                    elites,
                    count=mutation_budget,
                    seed=seed + 1009,
                    target_shapes=target_shapes,
                    exclude={candidate.hash for candidate in candidates},
                    group_weights=semantic_group_weights,
                )
            )
        else:
            candidates.extend(
                mutate_elites(
                    elites,
                    count=mutation_budget,
                    seed=seed + 1009,
                    mutation_rate=mutation_rate,
                    target_shapes=target_shapes,
                )
            )

    de_budget = operator_allocation["de"] if operator_allocation is not None else pool_de_count
    if proposal in {"de", "seed-random-de", "evolutionary", "family-qd"} and de_budget > 0:
        parents = dedupe_candidates(elites)
        candidates.extend(
            differential_evolution_candidates(
                parents,
                count=de_budget,
                seed=seed + 2003,
                crossover_rate=crossover_rate,
                random_gene_rate=random_gene_rate,
                exclude={candidate.hash for candidate in candidates},
                target_shapes=target_shapes,
            )
        )

    adaptive_gomea_budget = (
        0
        if operator_allocation is None
        else operator_allocation["gomea-neighborhood"] + operator_allocation["gomea-mixing"]
    )
    if proposal in {"gomea", "seed-random-gomea", "evolutionary", "family-qd"} and (
        gomea_count > 0 or adaptive_gomea_budget > 0
    ):
        parents = dedupe_candidates(elites)
        linkage_models, _ = _learned_linkage_models_for_proposal(
            evidence,
            enabled=learned_linkage,
            target_shapes=target_shapes,
            min_samples=linkage_min_samples,
            truncation_tau=linkage_truncation_tau,
            max_clusters=linkage_max_clusters,
            ordinal_bins=linkage_ordinal_bins,
        )
        neighborhood_parents = parents
        if operator_allocation is None:
            gomea_budget = max(0, pool_gomea_count)
            neighborhood_budget = gomea_budget // 2
            mixing_budget = gomea_budget - neighborhood_budget
        else:
            neighborhood_budget = operator_allocation["gomea-neighborhood"]
            mixing_budget = operator_allocation["gomea-mixing"]
        candidates.extend(
            gomea_neighborhood_candidates(
                neighborhood_parents,
                count=neighborhood_budget,
                max_elites=None,
                exclude={candidate.hash for candidate in candidates},
                seed=seed + 2903,
                source="gomea-neighborhood" if operator_allocation is not None else "gomea",
                target_shapes=target_shapes,
                group_weights=semantic_group_weights,
                micro_exhaustive=micro_exhaustive_neighborhoods,
            )
        )
        candidates.extend(
            gomea_candidates(
                parents,
                count=mixing_budget,
                seed=seed + 3001,
                elite_count=elite_count,
                exclude={candidate.hash for candidate in candidates},
                target_shapes=target_shapes,
                linkage_models=linkage_models,
                family_local_probability=0.8 if operator_allocation is not None else 0.0,
                source="gomea-mixing" if operator_allocation is not None else "gomea",
                donor_mode_weights=donor_mode_weights,
                adaptive_donor_selection=adaptive_donor_selection,
            )
        )

    deduped = dedupe_candidates(candidates)
    intentional_preserved_hashes = (
        {candidate.hash for candidate in [*transfer_elites, *family_leaders, *supplied_parents]}
        if proposal == "family-qd"
        else {candidate.hash for candidate in [*transfer_elites, *supplied_parents]}
    )
    known_hashes = {candidate.hash for candidate in db.get_candidates([candidate.hash for candidate in deduped])}
    preserved_hashes = intentional_preserved_hashes | known_hashes
    preserved = [candidate for candidate in deduped if candidate.hash in preserved_hashes]
    generated = [candidate for candidate in deduped if candidate.hash not in preserved_hashes]
    scoped_generated = {
        candidate.hash: Candidate(
            params=candidate.canonical_params(),
            source=candidate.source,
            parent_hashes=candidate.parent_hashes,
            proposal_metadata={
                **candidate.proposal_metadata,
                "proposal_scope_kind": scope.kind,
                "proposal_scope_shape_ids": list(scope.shape_ids),
            },
        )
        for candidate in generated
    }
    generated = list(scoped_generated.values())
    if pool_multiplier <= 1:
        selected = [scoped_generated.get(candidate.hash, candidate) for candidate in deduped]
    else:
        variation_budget = local_count + de_count + gomea_count if elites else 0
        selection_count = num_random + variation_budget
        selected_generated = select_surrogate_pool(
            generated,
            evidence=evidence,
            shapes=target_shapes or [],
            count=selection_count,
            seed=seed + 4001,
            min_evidence=surrogate_min_evidence,
            covering_cold_start=covering_cold_start,
            cold_start_precovered_tokens=cold_start_precovered_tokens,
            surrogate_jobs=surrogate_jobs,
            workgroup_processor_count=workgroup_processor_count,
        )
        selected = dedupe_candidates(
            [*preserved, *(scoped_generated[candidate.hash] for candidate in selected_generated)]
        )
    db.record_proposal_occurrences(
        candidates,
        problem_type_hash=problem_type_hash or "",
        benchmark_protocol_hash=benchmark_protocol_hash or "",
        scope_kind=scope.kind,
        scope_shape_ids=scope.shape_ids,
        selected_candidates=selected,
    )
    return ProposalResult(
        scope=scope,
        preserved=tuple(preserved),
        generated=tuple(generated),
        selected=tuple(selected),
    )
