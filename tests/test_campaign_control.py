import json
import sys
from pathlib import Path

from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.protocol import DEFAULT_BENCHMARK_PROTOCOL
from evotensile.scheduler import execute_schedule, propose_candidates
from evotensile.search.campaign_control import (
    convergence_detected,
    estimate_next_round_duration_s,
    load_island_elites,
    plateau_detected,
    population_diagnostics,
    tag_proposals,
)
from evotensile.search.cost_model import load_candidate_evaluation_costs
from evotensile.search.operator_credit import OperatorCredit, allocate_operator_budget
from evotensile.shapes import pilot_100_shapes
from scripts.run_blind_one_shape_20min import _candidate_from_payload, _candidate_payload, main
from tests.helpers import sample_candidates
from tests.test_structured_runner import _fake_build_tensile, _fake_structured_runner


def test_island_metadata_and_elite_loading_are_query_local(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "islands.sqlite")
    db.init()
    shape = pilot_100_shapes()[0]
    candidates = sample_candidates(6)
    first = tag_proposals(
        candidates[:3],
        island_id="island-0",
        parent_hashes=set(),
        proposal_duration_s=0.3,
    )
    second = tag_proposals(
        candidates[3:],
        island_id="island-1",
        parent_hashes=set(),
        proposal_duration_s=0.6,
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
    )

    parent_hashes = {candidate.hash for candidate in island_parents}
    generated = [candidate for candidate in proposed if candidate.hash not in parent_hashes]
    assert generated
    assert all(set(candidate.parent_hashes) <= parent_hashes for candidate in generated)
    assert not ({candidate.hash for candidate in all_candidates[8:]} & {candidate.hash for candidate in proposed})


def test_population_plateau_cost_guard_and_convergence_are_deterministic():
    shape = pilot_100_shapes()[0]
    diagnostics = population_diagnostics(sample_candidates(8), shape)
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


def test_candidate_checkpoint_payload_preserves_exact_hash_and_metadata():
    candidate = tag_proposals(
        sample_candidates(1),
        island_id="island-0",
        parent_hashes=set(),
        proposal_duration_s=0.25,
    )[0]

    restored = _candidate_from_payload(_candidate_payload(candidate))

    assert restored.hash == candidate.hash
    assert restored.source == candidate.source
    assert restored.parent_hashes == candidate.parent_hashes
    assert restored.proposal_metadata == candidate.proposal_metadata


def test_campaign_driver_checkpoints_two_islands_and_resumes_finished_run(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    fake_tensile = _fake_build_tensile(tmp_path)
    fake_runner = _fake_structured_runner(tmp_path)
    output = tmp_path / "campaign"
    arguments = [
        "run_blind_one_shape_20min.py",
        "--output",
        str(output),
        "--shape",
        "512,128,1,256",
        "--seed",
        "20260710",
        "--time-budget",
        "30",
        "--hot-reserve",
        "5",
        "--max-feedback-rounds",
        "0",
        "--runner-bin",
        str(fake_runner),
        "--tensilelite-bin",
        str(fake_tensile),
    ]
    monkeypatch.setattr(sys, "argv", arguments)

    assert main() == 0
    capsys.readouterr()
    summary = json.loads((output / "campaign_summary.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((output / "campaign_checkpoint.json").read_text(encoding="utf-8"))
    proposals = json.loads((output / "round_00" / "proposals.json").read_text(encoding="utf-8"))

    assert checkpoint["phase"] == "finished"
    assert {call["island_id"] for call in proposals["proposal_calls"]} == {"island-0", "island-1"}
    assert all("island_id" in candidate["proposal_metadata"] for candidate in proposals["candidates"])
    assert summary["rounds"][0]["schedule"]["missing_pairs"] == 48

    monkeypatch.setattr(sys, "argv", [*arguments, "--resume"])
    assert main() == 0


def test_recorded_run_costs_cover_prepare_validation_and_screening(tmp_path: Path):
    fake_tensile = _fake_build_tensile(tmp_path)
    fake_runner = _fake_structured_runner(tmp_path)
    db = EvoTensileDB.connect(tmp_path / "costs.sqlite")
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    protocol = DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=2)

    execute_schedule(
        db,
        shapes=[shape],
        candidates=[candidate],
        output_root=tmp_path / "round",
        protocol=protocol,
        candidate_batch_size=1,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=fake_runner,
        cost_aware_scheduling=True,
    )
    costs = load_candidate_evaluation_costs(db)

    assert costs[candidate.hash].prepare_s > 0.0
    assert costs[candidate.hash].validation_s > 0.0
    assert costs[candidate.hash].screening_s > 0.0
    assert costs[candidate.hash].total_s >= costs[candidate.hash].prepare_s
