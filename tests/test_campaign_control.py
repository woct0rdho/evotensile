from pathlib import Path

import pytest

from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.protocol import DEFAULT_BENCHMARK_PROTOCOL
from evotensile.search.campaign_control import (
    convergence_detected,
    estimate_confirmation_reserve_s,
    estimate_next_round_duration_s,
    load_island_elites,
    plateau_detected,
    population_diagnostics,
    restart_epoch,
    tag_generated_proposals,
)
from evotensile.search.operator_credit import OperatorCredit, allocate_operator_budget
from evotensile.shapes import pilot_100_shapes
from tests.helpers import sample_candidates


def test_island_metadata_and_elite_loading_are_query_local(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "islands.sqlite")
    db.init()
    shape = pilot_100_shapes()[0]
    candidates = sample_candidates(6)
    first = tag_generated_proposals(
        candidates[:3],
        generated_hashes={candidate.hash for candidate in candidates[:3]},
        island_id="island-0",
        proposal_cost_s=0.1,
    )
    second = tag_generated_proposals(
        candidates[3:],
        generated_hashes={candidate.hash for candidate in candidates[3:]},
        island_id="island-1",
        proposal_cost_s=0.2,
    )
    db.register_candidates([*first, *second])
    db.register_shapes([shape])
    for index, candidate in enumerate([*first, *second]):
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

    elites = load_island_elites(
        db,
        island_id="island-1",
        shape_id=shape.id,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        limit=2,
    )

    assert len(elites) == 2
    assert all(candidate.proposal_metadata["island_id"] == "island-1" for candidate in elites)
    assert all(abs(float(candidate.proposal_metadata["proposal_cost_s"]) - 0.2) < 1e-12 for candidate in elites)


def test_restart_epoch_increments_only_on_restart_transition():
    counters = {"island-0": 0}

    assert restart_epoch(counters, scope="island-0", transition=False) == 0
    assert restart_epoch(counters, scope="island-0", transition=True) == 1
    assert restart_epoch(counters, scope="island-0", transition=False) == 1
    assert restart_epoch(counters, scope="island-0", transition=False) == 1
    assert restart_epoch(counters, scope="island-0", transition=True) == 2
    assert counters == {"island-0": 2}


def test_confirmation_reserve_scales_with_finalist_launch_cost():
    protocol = DEFAULT_BENCHMARK_PROTOCOL.with_overrides(
        num_warmups=20,
        num_benchmarks=10,
        enqueues_per_sync=10,
        num_elements_to_validate=0,
    )

    assert estimate_confirmation_reserve_s(
        [25_000.0] * 8,
        protocol=protocol,
        top_k=8,
        minimum_reserve_s=10.0,
    ) == pytest.approx(44.0)
    assert (
        estimate_confirmation_reserve_s(
            [25_000.0] * 8,
            protocol=protocol,
            top_k=8,
            minimum_reserve_s=60.0,
        )
        == 60.0
    )


def test_population_plateau_cost_guard_and_convergence_are_deterministic():
    shape = pilot_100_shapes()[0]
    diagnostics = population_diagnostics(
        sample_candidates(8),
        shape,
        workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
    )
    rounds = [
        {"duration_s": 24.0, "schedule": {"missing_pairs": 24}},
        {"duration_s": 30.0, "schedule": {"missing_pairs": 24}},
        {"duration_s": 27.0, "schedule": {"missing_pairs": 24}},
    ]

    assert diagnostics.candidates == 8
    assert diagnostics.matrix_instructions >= 1
    assert diagnostics.mechanical_tokens > diagnostics.candidates
    assert plateau_detected([100.0, 110.0, 110.1, 110.2, 110.15], patience=3, minimum_improvement_fraction=0.005)
    assert estimate_next_round_duration_s(rounds, expected_missing_pairs=24) >= 30.0
    low_diversity = diagnostics.__class__(8, 1, 1, 20, 3.0, 1)
    assert convergence_detected([100.0, 110.0, *([110.1] * 8)], low_diversity)


def test_cost_aware_credit_prefers_equal_success_at_lower_cost():
    credits = {
        "semantic-mutation": OperatorCredit(
            arm="semantic-mutation",
            successes=4,
            failures=1,
            cumulative_cost_s=10.0,
        ),
        "de": OperatorCredit(arm="de", successes=4, failures=1, cumulative_cost_s=80.0),
        "gomea-neighborhood": OperatorCredit(arm="gomea-neighborhood"),
        "gomea-mixing": OperatorCredit(arm="gomea-mixing"),
    }

    allocation = allocate_operator_budget(40, credits, cost_aware=True)

    assert allocation["semantic-mutation"] > allocation["de"]
    assert sum(allocation.values()) == 40
