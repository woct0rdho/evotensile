import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from evotensile.campaign.configuration import CampaignConfigurationRequest, build_campaign_configuration
from evotensile.campaign.runner import CampaignRun, run_campaign
from evotensile.campaign.store import CampaignStore
from evotensile.profile import DEFAULT_PROFILE, PROFILES
from evotensile.shapes import parse_shape
from scripts import run_blind_one_shape
from tests.helpers import fake_build_tensile, fake_structured_runner


def _small_campaign_configuration(request: CampaignConfigurationRequest, *, shape):
    return replace(
        build_campaign_configuration(request, profile=DEFAULT_PROFILE, shape=shape),
        cold_candidates=2,
        cold_pool_multiplier=2,
        hot_top_k=1,
    )


def test_campaign_driver_checkpoints_two_islands_and_resumes_finished_run(
    tmp_path: Path,
    capsys,
):
    fake_tensile = fake_build_tensile(tmp_path)
    fake_runner = fake_structured_runner(tmp_path)
    output = tmp_path / "campaign"
    shape = parse_shape("512,128,1,256")
    request = CampaignConfigurationRequest(
        runner_bin=fake_runner,
        tensilelite_bin=fake_tensile,
        seed=20260710,
        time_budget_s=30.0,
        hot_reserve_s=5.0,
        max_feedback_rounds=0,
        early_stop_on_convergence=False,
        build_timeout_s=300.0,
        runner_timeout_s=300.0,
        leader_stabilization=True,
    )
    configuration = _small_campaign_configuration(request, shape=shape)
    campaign = CampaignRun(
        configuration=configuration,
        profile=DEFAULT_PROFILE,
        shapes=(shape,),
        store=CampaignStore(output),
    )

    assert run_campaign(campaign) == 0
    capsys.readouterr()
    summary = json.loads((output / "campaign_summary.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((output / "campaign_checkpoint.json").read_text(encoding="utf-8"))
    proposals = json.loads((output / "round_00" / "proposals.json").read_text(encoding="utf-8"))
    configuration = json.loads((output / "campaign_configuration.json").read_text(encoding="utf-8"))

    assert checkpoint["controller"]["phase"] == "finished"
    assert {event["island_id"] for event in proposals["proposal_events"]} == {"island-0", "island-1"}
    assert all("island_id" in candidate["proposal_metadata"] for candidate in proposals["candidates"])
    assert summary["rounds"][0]["schedule"]["requested_pairs"] == 2
    assert summary["rounds"][0]["active_candidate_count"] == 2
    assert summary["rounds"][0]["archive_candidate_count"] == 0
    assert summary["rounds"][0]["active_population_diagnostics"]["candidates"] == 2
    assert summary["rounds"][0]["archive_diagnostics"]["candidates"] == 0
    assert summary["controller"]["shape_ids"] == [shape.id]
    assert summary["controller"]["queried_pairs"] == 2
    assert summary["controller"]["known_pairs"] == 2
    assert summary["controller"]["unknown_pairs"] == 0
    assert summary["controller"]["prepared_candidates"] == 2
    assert summary["controller"]["phase_time_s"]["proposal"] >= 0.0
    assert summary["controller"]["phase_time_s"]["screening"] > 0.0
    assert "version" not in configuration
    assert summary["controller"]["clustering"]["clusters"][0]["medoid_shape_id"] == shape.id
    assert configuration["adaptive_policy"]["confidence"] == 0.90
    assert configuration["adaptive_policy"]["max_rounds"] == 0
    assert configuration["screening_protocol"]["num_benchmarks"] == 2
    assert configuration["hot_protocol"]["num_warmups"] == 20
    assert configuration["candidate_batch_size"] == 1
    assert configuration["prepare_workers"] == 32
    assert configuration["validation_workers"] == 1
    assert configuration["compute_unit_count"] == 40
    assert configuration["workgroup_processor_count"] == 20
    assert configuration["compute_units_per_workgroup_processor"] == 2
    assert Path(configuration["runner_bin"]).is_absolute()
    assert len(configuration["runner_fingerprint"]) == 64
    assert len(configuration["tensilelite_fingerprint"]) == 64
    assert len(configuration["implementation_fingerprint"]) == 64
    assert configuration["environment"]

    assert run_campaign(replace(campaign, resume=True)) == 0

    mismatched = CampaignRun(
        configuration=_small_campaign_configuration(replace(request, runner_timeout_s=301.0), shape=shape),
        profile=DEFAULT_PROFILE,
        shapes=(shape,),
        store=CampaignStore(output),
        resume=True,
    )
    with pytest.raises(SystemExit, match="resume configuration mismatch"):
        run_campaign(mismatched)

    fake_runner.write_text(fake_runner.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    changed_binary = CampaignRun(
        configuration=_small_campaign_configuration(request, shape=shape),
        profile=DEFAULT_PROFILE,
        shapes=(shape,),
        store=CampaignStore(output),
        resume=True,
    )
    with pytest.raises(SystemExit, match="resume configuration mismatch"):
        run_campaign(changed_binary)


def test_campaign_soft_budget_does_not_clamp_admitted_job_timeout(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    fake_tensile = fake_build_tensile(tmp_path)
    fake_runner = fake_structured_runner(tmp_path)
    output = tmp_path / "soft_budget"
    monkeypatch.setenv("EVOTENSILE_TEST_BUILD_SLEEP_S", "0.1")
    full_configuration_builder = run_blind_one_shape.build_campaign_configuration
    monkeypatch.setattr(
        run_blind_one_shape,
        "build_campaign_configuration",
        lambda *args, **kwargs: replace(
            full_configuration_builder(*args, **kwargs),
            cold_candidates=2,
            cold_pool_multiplier=1,
            hot_top_k=1,
        ),
    )
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

    assert run_blind_one_shape.main() == 0
    capsys.readouterr()
    summary = json.loads((output / "campaign_summary.json").read_text(encoding="utf-8"))

    assert len(summary["rounds"]) == 1
    assert summary["rounds"][0]["schedule"]["status_counts"]["ok"] > 0
    assert summary["elapsed_s"] > summary["configuration"]["time_budget_s"]
    assert summary["budget_overrun_s"] > 0.0


def test_campaign_script_resolves_selected_profile_execution_defaults(tmp_path: Path, monkeypatch):
    profile = replace(
        DEFAULT_PROFILE,
        name="script-profile",
        default_runner_bin="profile-runner",
        default_build_timeout_s=123.0,
        default_runner_timeout_s=45.0,
    )
    requests: list[CampaignConfigurationRequest] = []
    selected_profiles = []

    def build_configuration(request, *, profile, shape):
        requests.append(request)
        selected_profiles.append(profile)
        return object()

    monkeypatch.setitem(PROFILES, profile.name, profile)
    monkeypatch.setattr(run_blind_one_shape, "build_campaign_configuration", build_configuration)
    monkeypatch.setattr(run_blind_one_shape, "run_campaign", lambda campaign: 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_blind_one_shape.py",
            "--output",
            str(tmp_path / "campaign"),
            "--profile",
            profile.name,
        ],
    )

    assert run_blind_one_shape.main() == 0
    request = requests[0]
    assert selected_profiles == [profile]
    assert request.runner_bin == Path(profile.default_runner_bin)
    assert request.build_timeout_s == profile.default_build_timeout_s
    assert request.runner_timeout_s == profile.default_runner_timeout_s
