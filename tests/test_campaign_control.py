import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.protocol import DEFAULT_BENCHMARK_PROTOCOL
from evotensile.scheduler import execute_schedule, propose_candidates
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
from evotensile.search.evaluation_cost import load_candidate_evaluation_costs
from evotensile.search.operator_credit import OperatorCredit, allocate_operator_budget
from evotensile.shapes import pilot_100_shapes
from scripts.run_blind_one_shape import (
    _campaign_configuration,
    _candidate_from_payload,
    _candidate_payload,
    _proposal_call,
    main,
)
from tests.helpers import sample_candidates
from tests.test_structured_runner import fake_build_tensile, fake_structured_runner


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
    configuration = _campaign_configuration(
        argparse.Namespace(
            runner_bin=tool,
            tensilelite_bin=tool,
            seed=20260713,
            time_budget=1200.0,
            hot_reserve=60.0,
            max_feedback_rounds=100,
            early_stop_on_convergence=False,
            build_timeout=300.0,
            runner_timeout=300.0,
            no_leader_stabilization=False,
        ),
        profile=DEFAULT_PROFILE,
        shape=shape,
    )
    with pytest.raises(ValueError, match="operator budget sum"):
        replace(configuration, feedback_candidates=25)

    proposal = _proposal_call(
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
        effective_cu_count=DEFAULT_PROFILE.effective_cu_count,
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


def test_candidate_checkpoint_payload_preserves_exact_hash_and_metadata():
    source = sample_candidates(1)
    candidate = tag_generated_proposals(
        source,
        generated_hashes={source[0].hash},
        island_id="island-0",
        proposal_cost_s=0.25,
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
    fake_tensile = fake_build_tensile(tmp_path)
    fake_runner = fake_structured_runner(tmp_path)
    output = tmp_path / "campaign"
    arguments = [
        "run_blind_one_shape.py",
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
    configuration = json.loads((output / "campaign_configuration.json").read_text(encoding="utf-8"))

    assert checkpoint["phase"] == "finished"
    assert {event["island_id"] for event in proposals["proposal_events"]} == {"island-0", "island-1"}
    assert all("island_id" in candidate["proposal_metadata"] for candidate in proposals["candidates"])
    assert summary["rounds"][0]["schedule"]["missing_pairs"] == 48
    assert summary["rounds"][0]["active_candidate_count"] == 48
    assert summary["rounds"][0]["archive_candidate_count"] == 0
    assert summary["rounds"][0]["active_population_diagnostics"]["candidates"] == 48
    assert summary["rounds"][0]["archive_diagnostics"]["candidates"] == 0
    assert configuration["version"] == 1
    assert configuration["adaptive_policy"]["confidence"] == 0.90
    assert configuration["screening_protocol"]["num_benchmarks"] == 2
    assert configuration["hot_protocol"]["num_warmups"] == 20
    assert configuration["candidate_batch_size"] == 1
    assert configuration["prepare_workers"] == 32
    assert configuration["validation_workers"] == 1
    assert Path(configuration["runner_bin"]).is_absolute()
    assert len(configuration["runner_fingerprint"]) == 64
    assert len(configuration["tensilelite_fingerprint"]) == 64
    assert len(configuration["implementation_fingerprint"]) == 64
    assert configuration["environment"]

    monkeypatch.setattr(sys, "argv", [*arguments, "--resume"])
    assert main() == 0

    monkeypatch.setattr(sys, "argv", [*arguments, "--resume", "--runner-timeout", "301"])
    with pytest.raises(SystemExit, match="resume configuration mismatch"):
        main()

    fake_runner.write_text(fake_runner.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [*arguments, "--resume"])
    with pytest.raises(SystemExit, match="resume configuration mismatch"):
        main()


def test_campaign_soft_budget_does_not_clamp_admitted_job_timeout(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    fake_tensile = fake_build_tensile(tmp_path)
    fake_runner = fake_structured_runner(tmp_path)
    output = tmp_path / "soft_budget"
    monkeypatch.setenv("EVOTENSILE_TEST_BUILD_SLEEP_S", "0.1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_blind_one_shape.py",
            "--output",
            str(output),
            "--shape",
            "512,128,1,256",
            "--time-budget",
            "0.05",
            "--hot-reserve",
            "0",
            "--max-feedback-rounds",
            "0",
            "--runner-bin",
            str(fake_runner),
            "--tensilelite-bin",
            str(fake_tensile),
            "--build-timeout",
            "10",
            "--runner-timeout",
            "10",
        ],
    )

    assert main() == 0
    capsys.readouterr()
    summary = json.loads((output / "campaign_summary.json").read_text(encoding="utf-8"))

    assert len(summary["rounds"]) == 1
    assert summary["rounds"][0]["schedule"]["status_counts"]["ok"] > 0
    assert summary["elapsed_s"] > summary["configuration"]["time_budget_s"]
    assert summary["budget_overrun_s"] > 0.0


def test_indexed_run_cost_divides_shared_duration_once(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "costs.sqlite")
    db.init()
    candidates = sample_candidates(2)
    db.register_candidates(candidates)

    db.insert_run(
        "shared",
        yaml_path=None,
        output_dir=None,
        status="ok",
        candidate_hashes=[candidates[0].hash, candidates[1].hash, candidates[0].hash],
        cost_phase="prepare",
        duration_s=6.0,
    )
    costs = load_candidate_evaluation_costs(db)

    assert costs[candidates[0].hash].prepare_s == 3.0
    assert costs[candidates[1].hash].prepare_s == 3.0

    db.insert_run(
        "shared",
        yaml_path=None,
        output_dir=None,
        status="ok",
        candidate_hashes=[candidates[1].hash],
        cost_phase="prepare",
        duration_s=2.0,
    )
    replaced_costs = load_candidate_evaluation_costs(db)
    assert replaced_costs[candidates[0].hash].prepare_s == 0.0
    assert replaced_costs[candidates[1].hash].prepare_s == 2.0


def test_recorded_run_costs_cover_prepare_validation_and_screening(tmp_path: Path):
    fake_tensile = fake_build_tensile(tmp_path)
    fake_runner = fake_structured_runner(tmp_path)
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
