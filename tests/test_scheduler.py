from dataclasses import replace
from pathlib import Path
from textwrap import dedent

from evotensile.candidate import Shape
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.scheduler import execute_schedule
from evotensile.scheduling.planning import plan_pair_requests
from evotensile.search_space import make_candidate
from evotensile.shapes import pilot_100_shapes
from evotensile.tensilelite_diagnostics import DiagnosticRecord, DiagnosticRunResult
from tests.helpers import REFERENCE_CANDIDATE, pair_requests, sample_candidates


def test_execute_schedule_resolves_profile_and_explicit_timeouts(tmp_path: Path):
    profile = replace(
        DEFAULT_PROFILE,
        max_no_cache_candidate_batch_size=3,
        default_shape_batch_size=7,
        default_prepare_workers=5,
        default_prepare_wave_batches=4,
        default_validation_workers=2,
        default_runner_bin="profile-runner",
        default_build_timeout_s=41.0,
        default_runner_timeout_s=17.0,
    )
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    shapes = pilot_100_shapes()[:1]
    candidates = sample_candidates(1)

    defaults = execute_schedule(
        db,
        requests=pair_requests(candidates, shapes),
        output_root=tmp_path / "defaults",
        target_profile=profile,
        dry_run=True,
    )
    assert defaults.build_timeout_s == 41.0
    assert defaults.runner_timeout_s == 17.0
    assert defaults.candidate_batch_size == 1
    assert defaults.shape_batch_size == 7
    assert defaults.prepare_workers == 5
    assert defaults.prepare_wave_batches == 4
    assert defaults.validation_workers == 2
    assert defaults.runner_bin == "profile-runner"

    explicit = execute_schedule(
        db,
        requests=pair_requests(candidates, shapes),
        output_root=tmp_path / "explicit",
        target_profile=profile,
        dry_run=True,
        build_timeout_s=9.0,
        runner_timeout_s=5.0,
    )
    assert explicit.build_timeout_s == 9.0
    assert explicit.runner_timeout_s == 5.0

    disabled = execute_schedule(
        db,
        requests=pair_requests(candidates, shapes),
        output_root=tmp_path / "disabled",
        target_profile=profile,
        dry_run=True,
        build_timeout_s=0.0,
        runner_timeout_s=-1.0,
    )
    assert disabled.build_timeout_s is None
    assert disabled.runner_timeout_s is None


def test_execute_schedule_records_shape_rule_rejection_without_build(tmp_path: Path):
    fake_tensilelite = tmp_path / "fail_if_called.py"
    fake_tensilelite.write_text("#!/usr/bin/env python3\nraise SystemExit(99)\n", encoding="utf-8")
    fake_tensilelite.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidate = make_candidate(
        {**REFERENCE_CANDIDATE.canonical_params(), "GlobalSplitU": 2, "DepthU": 32},
        source="shape_rule",
    )
    shape = Shape(m=8192, n=8192, batch=1, k=8192)

    result = execute_schedule(
        db,
        requests=pair_requests([candidate], [shape]),
        output_root=tmp_path / "batches",
        candidate_batch_size=1,
        shape_batch_size=1,
        tensilelite_bin=fake_tensilelite,
        runner_bin=tmp_path / "unused_runner",
    )

    assert result.executed_batches == []
    assert db.benchmark_status_summary() == {"rejected": 1}


def test_execute_schedule_records_single_candidate_build_timeout(tmp_path: Path):
    fake_tensilelite = tmp_path / "slow_tensile.py"
    fake_tensilelite.write_text(
        "#!/usr/bin/env python3\nimport time\ntime.sleep(10)\n",
        encoding="utf-8",
    )
    fake_tensilelite.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()

    result = execute_schedule(
        db,
        requests=pair_requests([candidate], [shape]),
        output_root=tmp_path / "batches",
        candidate_batch_size=1,
        shape_batch_size=1,
        tensilelite_bin=fake_tensilelite,
        runner_bin=tmp_path / "unused_runner",
        build_timeout_s=0.1,
    )

    assert len(result.executed_batches) == 1
    assert result.executed_batches[0].build_returncode == 124
    assert db.benchmark_status_summary() == {"build_timeout": 1}
    assert (
        len(
            plan_pair_requests(
                db,
                requests=pair_requests([candidate], [shape]),
                problem_type_hash=p_hash,
                benchmark_protocol_hash=b_hash,
                validation_protocol_hash=DEFAULT_PROFILE.default_protocol.validation_protocol_hash(),
                candidate_batch_size=1,
                shape_batch_size=1,
            )
        )
        == 1
    )


def test_execute_schedule_salvages_final_yaml_and_uses_diagnostics_for_nonzero_build(tmp_path: Path, monkeypatch):
    fake_tensilelite = tmp_path / "fake_tensile.py"
    fake_tensilelite.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import sys
            from pathlib import Path
            import yaml

            config_path, out = Path(sys.argv[1]), Path(sys.argv[2])
            out.mkdir(parents=True, exist_ok=True)
            config = yaml.safe_load(config_path.read_text())
            problem = config["BenchmarkProblems"][0][1]
            problem_sizes = problem["BenchmarkFinalParameters"][0]["ProblemSizes"]
            item = problem["ForkParameters"][0]["Groups"][0][1]
            sol = dict(item)
            mi = sol["MatrixInstruction"]
            sol["MatrixInstruction"] = mi[:4]
            sol["MIWaveTile"] = [mi[5], mi[6]]
            sol["MIWaveGroup"] = [mi[7], mi[8]]
            sol["SolutionIndex"] = 0
            sol["KernelNameMin"] = "Kernel0"
            (out / "1_BenchmarkProblems" / "Cijk_Ailk_Bjlk_HHS_BH_Bias_H_HA_S_SAV_UserArgs_00" / "Data").mkdir(parents=True, exist_ok=True)
            (out / "1_BenchmarkProblems" / "Cijk_Ailk_Bjlk_HHS_BH_Bias_H_HA_S_SAV_UserArgs_00" / "Data" / "00_Final.yaml").write_text(
                yaml.safe_dump(
                    [{"MinimumRequiredVersion": "5.0.0"}, {"ProblemSizes": problem_sizes}, sol],
                    sort_keys=False,
                )
            )
            lib = out / "4_LibraryClient" / "library" / "gfx1151"
            lib.mkdir(parents=True, exist_ok=True)
            (lib / "TensileLibrary_gfx1151.yaml").write_text("---\\nsolutions: []\\n")
            (lib / "Kernels.so-000-gfx1151.hsaco").write_bytes(b"fake")
            sys.exit(2)
            """
        ),
        encoding="utf-8",
    )
    fake_tensilelite.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidates = [sample_candidates(1)[0], REFERENCE_CANDIDATE]
    shape = pilot_100_shapes()[0]

    fake_runner = tmp_path / "fake_runner.py"
    fake_runner.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import argparse
            import json

            parser = argparse.ArgumentParser()
            parser.add_argument("--mode", required=True)
            parser.add_argument("--pairs")
            parser.add_argument("--output")
            args, _ = parser.parse_known_args()
            with open(args.pairs) as src, open(args.output, "w") as out:
                for line in src:
                    pair = json.loads(line)
                    if args.mode == "validate":
                        out.write(json.dumps({
                            "shape_id": pair["shape_id"],
                            "candidate_hash": pair["candidate_hash"],
                            "status": "ok",
                            "sample_index": 0,
                            "time_us": None,
                            "validation": "PASSED",
                            "solution_index": pair["library_solution_index"],
                        }) + "\\n")
                    else:
                        for sample_index in range(pair.get("num_benchmarks", 1)):
                            out.write(json.dumps({
                                "shape_id": pair["shape_id"],
                                "candidate_hash": pair["candidate_hash"],
                                "status": "ok",
                                "sample_index": sample_index,
                                "time_us": 1.0 + sample_index * 0.001,
                                "validation": "NO_CHECK",
                                "solution_index": pair["library_solution_index"],
                            }) + "\\n")
            """
        ),
        encoding="utf-8",
    )
    fake_runner.chmod(0o755)

    def fake_diagnostics(*args, **kwargs):
        diagnostics_path = tmp_path / "diagnostics.jsonl"
        diagnostics_path.write_text("", encoding="utf-8")
        db.insert_run(
            "diagnostics_run",
            phase="prepare",
            status="ok",
            duration_s=0.0,
            candidate_hashes=[candidate.hash for candidate in candidates],
        )
        return DiagnosticRunResult(
            run_id="diagnostics_run",
            returncode=0,
            records=[
                DiagnosticRecord(
                    candidate_hash=candidates[0].hash,
                    candidate_index=0,
                    status="kernelwriter_failed",
                    phase="kernelwriter",
                    reason="KernelWriter returned errcode -2",
                    shape_ids=(shape.id,),
                )
            ],
            results_path=diagnostics_path,
            stdout_path=diagnostics_path,
            stderr_path=diagnostics_path,
            command=["diagnostics"],
            duration_s=0.0,
        )

    monkeypatch.setattr("evotensile.scheduling.preparation.run_tensilelite_diagnostics", fake_diagnostics)

    result = execute_schedule(
        db,
        requests=pair_requests(candidates, [shape]),
        output_root=tmp_path / "batches",
        candidate_batch_size=2,
        shape_batch_size=1,
        tensilelite_bin=fake_tensilelite,
        runner_bin=fake_runner,
        keep_going=True,
    )

    assert len(result.executed_batches) == 1
    assert result.executed_batches[0].ingest is not None
    assert result.executed_batches[0].ingest.status_counts == {"ok": 10, "build_failed": 1}
    assert db.benchmark_status_summary() == {"build_failed": 1, "ok": 10}


def test_multi_candidate_build_failure_unattributed_is_not_reusable_cache(tmp_path: Path, monkeypatch):
    fake_tensilelite = tmp_path / "fake_tensile.py"
    fake_tensilelite.write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(2)\n", encoding="utf-8")
    fake_tensilelite.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidates = sample_candidates(2)
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()

    def fake_diagnostics(*args, **kwargs):
        diagnostics_path = tmp_path / "empty_diagnostics.jsonl"
        diagnostics_path.write_text("", encoding="utf-8")
        db.insert_run(
            "diagnostics_unattributed",
            phase="prepare",
            status="ok",
            duration_s=0.0,
            candidate_hashes=[candidate.hash for candidate in candidates],
        )
        return DiagnosticRunResult(
            run_id="diagnostics_unattributed",
            returncode=0,
            records=[],
            results_path=diagnostics_path,
            stdout_path=diagnostics_path,
            stderr_path=diagnostics_path,
            command=["diagnostics"],
            duration_s=0.0,
        )

    monkeypatch.setattr("evotensile.scheduling.preparation.run_tensilelite_diagnostics", fake_diagnostics)

    result = execute_schedule(
        db,
        requests=pair_requests(candidates, [shape]),
        output_root=tmp_path / "batches",
        candidate_batch_size=2,
        shape_batch_size=1,
        tensilelite_bin=fake_tensilelite,
        runner_bin=tmp_path / "unused_runner",
        keep_going=True,
    )

    assert len(result.executed_batches) == 1
    assert result.executed_batches[0].ingest is not None
    assert result.executed_batches[0].ingest.status_counts == {"build_failed_unattributed": 2}
    assert db.benchmark_status_summary() == {"build_failed_unattributed": 2}
    assert (
        len(
            plan_pair_requests(
                db,
                requests=pair_requests(candidates, [shape]),
                problem_type_hash=p_hash,
                benchmark_protocol_hash=b_hash,
                validation_protocol_hash=DEFAULT_PROFILE.default_protocol.validation_protocol_hash(),
                candidate_batch_size=2,
                shape_batch_size=1,
            )
        )
        == 1
    )


def test_execute_schedule_records_single_candidate_build_failure(tmp_path: Path):
    fake_tensilelite = tmp_path / "fake_tensile.py"
    fake_tensilelite.write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(2)\n", encoding="utf-8")
    fake_tensilelite.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()

    result = execute_schedule(
        db,
        requests=pair_requests([candidate], [shape]),
        output_root=tmp_path / "batches",
        candidate_batch_size=1,
        shape_batch_size=1,
        tensilelite_bin=fake_tensilelite,
        runner_bin=tmp_path / "unused_runner",
    )

    assert len(result.executed_batches) == 1
    assert db.benchmark_status_summary() == {"build_failed": 1}
    assert (
        plan_pair_requests(
            db,
            requests=pair_requests([candidate], [shape]),
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            validation_protocol_hash=DEFAULT_PROFILE.default_protocol.validation_protocol_hash(),
            candidate_batch_size=1,
            shape_batch_size=1,
        )
        == []
    )


def test_execute_schedule_generate_only_writes_batch_inputs(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidates = sample_candidates(2)
    shapes = pilot_100_shapes()[:1]

    result = execute_schedule(
        db,
        requests=pair_requests(candidates, shapes),
        output_root=tmp_path / "batches",
        candidate_batch_size=1,
        shape_batch_size=1,
        generate_only=True,
    )

    assert len(result.planned_batches) == 2
    assert len(result.executed_batches) == 2
    for executed in result.executed_batches:
        assert executed.yaml_path.exists()
        assert executed.manifest_path.exists()
        assert executed.output_dir.name == "run"
