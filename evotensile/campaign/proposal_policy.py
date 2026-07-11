import time
from collections.abc import Mapping, Sequence
from typing import Any

from evotensile.campaign.models import CampaignConfiguration, ProposalArgs, RoundProposal
from evotensile.candidate import Candidate, Shape
from evotensile.database import EvoTensileDB
from evotensile.profile import TargetProfile
from evotensile.search.acquisition import propose_candidates
from evotensile.search.campaign_control import (
    ProposalEvent,
    load_island_elites,
    plateau_detected,
    restart_epoch,
    split_budget,
    tag_generated_proposals,
)
from evotensile.search.mechanics import mechanical_coverage_tokens


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


def propose_campaign_candidates(
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


def island_ids(configuration: CampaignConfiguration) -> tuple[str, ...]:
    return tuple(f"island-{index}" for index in range(configuration.island_count))


def leader_history(record: Mapping[str, object], *, island_id: str | None = None) -> list[float]:
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
    history = leader_history(record, island_id=island_id)
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


def propose_round(
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
    islands = island_ids(configuration)
    if round_index == 0:
        precovered_tokens: set[str] = set()
        for island_index, (island_id, count) in enumerate(
            zip(islands, split_budget(configuration.cold_candidates, len(islands)), strict=True)
        ):
            proposal = propose_campaign_candidates(
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
                propose_campaign_candidates(
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
            propose_campaign_candidates(
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
            propose_campaign_candidates(
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
            propose_campaign_candidates(
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
