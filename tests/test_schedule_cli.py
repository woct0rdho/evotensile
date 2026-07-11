import json
from dataclasses import replace
from pathlib import Path

import pytest

from evotensile.cli import main as cli_main
from evotensile.profile import DEFAULT_PROFILE, PROFILES
from evotensile.scheduling.planning import production_candidate_batch_size


def test_schedule_cli_metadata_records_operational_modes(tmp_path: Path):
    def run_cli(output_dir: Path, *extra_args: str) -> dict:
        rc = cli_main(
            [
                "schedule-batches",
                "--db",
                str(tmp_path / "sched.sqlite"),
                "--output-dir",
                str(output_dir),
                "--num-random",
                "1",
                "--limit-shapes",
                "1",
                "--shape-batch-size",
                "1",
                "--dry-run",
                *extra_args,
            ]
        )
        assert rc == 0
        return json.loads((output_dir / "schedule_metadata.json").read_text(encoding="utf-8"))

    default_metadata = run_cli(tmp_path / "default")
    assert default_metadata["profile"] == DEFAULT_PROFILE.name
    assert default_metadata["planned_batches"] >= 1
    assert default_metadata["executed_batches"] == []
    assert default_metadata["runner_bin"] == DEFAULT_PROFILE.default_runner_bin
    assert default_metadata["candidate_batch_size"] == production_candidate_batch_size(
        candidate_count=default_metadata["candidates"],
        shape_count=default_metadata["shapes"],
        shape_batch_size=default_metadata["shape_batch_size"],
        prepare_workers=default_metadata["prepare_workers"],
        max_candidate_batch_size=DEFAULT_PROFILE.default_candidate_batch_size,
    )
    assert default_metadata["prepare_workers"] == DEFAULT_PROFILE.default_prepare_workers == 32
    assert default_metadata["validation_workers"] == DEFAULT_PROFILE.default_validation_workers == 1
    assert default_metadata["surrogate_jobs"] == DEFAULT_PROFILE.default_surrogate_jobs
    assert default_metadata["compute_unit_count"] == DEFAULT_PROFILE.compute_unit_count == 40
    assert default_metadata["workgroup_processor_count"] == DEFAULT_PROFILE.workgroup_processor_count == 20
    assert default_metadata["compute_units_per_workgroup_processor"] == 2
    assert default_metadata["adaptive_sampling"] is True
    assert default_metadata["stop_on_error"] is False
    assert default_metadata["learned_linkage_requested"] is True
    assert default_metadata["learned_linkage_enabled"] is False
    assert default_metadata["linkage_fallback_reason"] == "insufficient_validated_evidence"
    assert default_metadata["candidate_family_count"] >= 1
    assert sum(default_metadata["candidate_family_counts"].values()) == default_metadata["candidates"]
    assert default_metadata["archive_family_count"] == 0

    assert default_metadata["compile_cache_enabled"] is True
    assert default_metadata["compile_cache_root"] == str(tmp_path / "default" / "compile_cache")

    cached_batch_metadata = run_cli(
        tmp_path / "cached_batch",
        "--num-random",
        "16",
        "--prepare-workers",
        "8",
    )
    assert cached_batch_metadata["candidate_batch_size"] == 1
    assert cached_batch_metadata["planned_batches"] >= cached_batch_metadata["prepare_workers"]

    large_batch_metadata = run_cli(
        tmp_path / "large_batch",
        "--num-random",
        "16",
        "--prepare-workers",
        "8",
        "--no-compile-cache",
    )
    assert large_batch_metadata["candidate_batch_size"] > 1

    debug_singleton_metadata = run_cli(tmp_path / "debug_singleton", "--candidate-batch-size", "1")
    assert debug_singleton_metadata["candidate_batch_size"] == 1

    production_policy_metadata = run_cli(
        tmp_path / "production_policy",
        "--search-policy",
        "gfx1151-grid-v1",
        "--num-random",
        "0",
        "--no-adaptive-donor-selection",
    )
    assert production_policy_metadata["search_policy"] == "gfx1151-grid-v1"
    assert production_policy_metadata["search_policy_settings"]["proposal"] == "family-qd"
    assert production_policy_metadata["proposal"] == "family-qd"
    assert production_policy_metadata["surrogate_pool_multiplier"] == 8
    assert production_policy_metadata["adaptive_operators"] is True
    assert production_policy_metadata["adaptive_group_credit"] is True
    assert production_policy_metadata["adaptive_donor_selection"] is False
    assert production_policy_metadata["cost_aware_operator_credit"] is True
    assert production_policy_metadata["cost_aware_scheduling"] is True

    no_learned_metadata = run_cli(tmp_path / "no_learned", "--no-learned-linkage")
    assert no_learned_metadata["learned_linkage_requested"] is False
    assert no_learned_metadata["linkage_fallback_reason"] == "disabled"

    no_compile_cache_metadata = run_cli(tmp_path / "no_compile_cache", "--no-compile-cache")
    assert no_compile_cache_metadata["compile_cache_enabled"] is False
    assert no_compile_cache_metadata["compile_cache_root"] is None

    fail_fast_metadata = run_cli(tmp_path / "fail_fast", "--stop-on-error")
    fixed_sampling_metadata = run_cli(tmp_path / "fixed", "--fixed-sampling")
    assert fail_fast_metadata["stop_on_error"] is True
    assert fixed_sampling_metadata["adaptive_sampling"] is False


def test_schedule_cli_resolves_selected_profile_defaults(tmp_path: Path):
    profile = replace(
        DEFAULT_PROFILE,
        name="test-profile",
        default_proposal="random",
        default_num_random=3,
        default_elite_count=5,
        default_local_count=7,
        default_de_count=9,
        default_gomea_count=11,
        default_transfer_shapes=2,
        default_transfer_per_shape=3,
        default_mutation_rate=0.15,
        default_crossover_rate=0.65,
        default_random_gene_rate=0.05,
        default_candidate_batch_size=4,
        default_shape_batch_size=2,
        default_prepare_workers=6,
        default_validation_workers=1,
        default_surrogate_jobs=2,
        compute_unit_count=24,
        workgroup_processor_count=12,
    )
    PROFILES[profile.name] = profile
    try:
        output_dir = tmp_path / "selected_profile"
        assert (
            cli_main(
                [
                    "schedule-batches",
                    "--db",
                    str(tmp_path / "sched.sqlite"),
                    "--output-dir",
                    str(output_dir),
                    "--profile",
                    profile.name,
                    "--limit-shapes",
                    "1",
                    "--dry-run",
                ]
            )
            == 0
        )
    finally:
        del PROFILES[profile.name]

    metadata = json.loads((output_dir / "schedule_metadata.json").read_text(encoding="utf-8"))
    assert metadata["profile"] == profile.name
    assert metadata["proposal"] == profile.default_proposal
    assert metadata["candidates"] == profile.default_num_random
    assert metadata["shape_batch_size"] == profile.default_shape_batch_size
    assert metadata["prepare_workers"] == profile.default_prepare_workers
    assert metadata["validation_workers"] == profile.default_validation_workers
    assert metadata["surrogate_jobs"] == profile.default_surrogate_jobs
    assert metadata["compute_unit_count"] == profile.compute_unit_count
    assert metadata["workgroup_processor_count"] == profile.workgroup_processor_count
    assert metadata["compute_units_per_workgroup_processor"] == profile.compute_units_per_workgroup_processor


def test_search_policy_rejects_incompatible_profile(tmp_path: Path):
    profile = replace(DEFAULT_PROFILE, name="test-profile")
    PROFILES[profile.name] = profile
    try:
        with pytest.raises(ValueError, match="requires profile gfx1151-nt-hhs"):
            cli_main(
                [
                    "schedule-batches",
                    "--db",
                    str(tmp_path / "sched.sqlite"),
                    "--output-dir",
                    str(tmp_path / "output"),
                    "--profile",
                    profile.name,
                    "--search-policy",
                    "gfx1151-grid-v1",
                    "--dry-run",
                ]
            )
    finally:
        del PROFILES[profile.name]
