from dataclasses import replace
from pathlib import Path

import pytest

from evotensile.campaign.configuration import CampaignConfigurationRequest, build_campaign_configuration
from evotensile.campaign.proposal_policy import propose_campaign_candidates
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.search.acquisition import propose_candidates
from evotensile.shapes import pilot_100_shapes
from tests.helpers import sample_candidates


def test_merged_proposal_separates_archive_and_novel_candidates(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "merged.sqlite")
    db.init()
    shape = pilot_100_shapes()[0]
    archive_candidates = sample_candidates(12, seed=20260712)
    db.register_candidates(archive_candidates)
    db.register_shapes([shape])
    for index, candidate in enumerate(archive_candidates):
        db.insert_evaluation(
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="queried",
            status="ok",
            problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
            benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
            time_us=100.0 + index,
            validation="PASSED",
        )

    tool = tmp_path / "tool"
    tool.write_text("tool\n", encoding="utf-8")
    configuration = build_campaign_configuration(
        CampaignConfigurationRequest(
            runner_bin=tool,
            tensilelite_bin=tool,
            seed=20260713,
            time_budget_s=1200.0,
            hot_reserve_s=60.0,
            max_feedback_rounds=100,
            early_stop_on_convergence=False,
            build_timeout_s=300.0,
            runner_timeout_s=300.0,
            leader_stabilization=True,
        ),
        profile=DEFAULT_PROFILE,
        shape=shape,
    )
    with pytest.raises(ValueError, match="operator budget sum"):
        replace(configuration, feedback_candidates=25)

    proposal = propose_campaign_candidates(
        db,
        shape=shape,
        profile=DEFAULT_PROFILE,
        configuration=configuration,
        protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        seed=20260713,
        proposal_args={
            "num_random": 2,
            "elite_count": 2,
            "local_count": 0,
            "de_count": 0,
            "gomea_count": 0,
            "adaptive_operators": False,
            "surrogate_pool_multiplier": 1,
            "covering_cold_start": False,
            "adaptive_group_credit": False,
            "micro_exhaustive_neighborhoods": False,
            "adaptive_donor_selection": False,
            "cost_aware_operator_credit": False,
            "surrogate_min_evidence": 24,
        },
        island_id="merged",
        parents=None,
        learned_linkage=True,
        restart_index=0,
    )

    assert len(proposal.archive) == 2
    assert len(proposal.active) == 2
    assert all("island_id" not in candidate.proposal_metadata for candidate in proposal.archive)
    assert all(candidate.proposal_metadata["island_id"] == "merged" for candidate in proposal.active)
    event = proposal.events[0]
    assert set(event.preserved_hashes) == {candidate.hash for candidate in proposal.archive}
    assert set(event.selected_generated_hashes) == {candidate.hash for candidate in proposal.active}
    assert event.scope_kind == "shape"
    assert event.scope_shape_ids == (shape.id,)
    assert len(event.generated_hashes) == 2
    assert event.proposal_cost_s > 0.0


def test_parent_override_prevents_cross_island_parent_selection(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "parents.sqlite")
    db.init()
    shape = pilot_100_shapes()[0]
    all_candidates = sample_candidates(12, seed=20260710)
    island_parents = all_candidates[:8]
    db.register_candidates(all_candidates)
    db.register_shapes([shape])
    for index, candidate in enumerate(all_candidates):
        db.insert_evaluation(
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="queried",
            status="ok",
            problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
            benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
            time_us=100.0 + index,
            validation="PASSED",
        )

    proposed = propose_candidates(
        db,
        proposal="family-qd",
        num_random=0,
        local_count=8,
        de_count=0,
        gomea_count=8,
        elite_count=8,
        target_shapes=[shape],
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        parent_candidates=island_parents,
        seed=20260711,
    ).selected

    parent_hashes = {candidate.hash for candidate in island_parents}
    generated = [candidate for candidate in proposed if candidate.hash not in parent_hashes]
    assert generated
    assert all(set(candidate.parent_hashes) <= parent_hashes for candidate in generated)
    assert not ({candidate.hash for candidate in all_candidates[8:]} & {candidate.hash for candidate in proposed})
