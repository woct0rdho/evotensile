import json
import sqlite3
import time
from pathlib import Path
from textwrap import dedent

from evotensile.adaptive_retime import AdaptivePolicy, ProbePolicy
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.protocol import DEFAULT_BENCHMARK_PROTOCOL
from evotensile.scheduler import execute_schedule
from evotensile.shapes import pilot_100_shapes
from evotensile.structured_runner import (
    RunnablePair,
    StructuredSample,
    library_dir_from_build,
    read_structured_results,
    run_structured_phase,
    validate_benchmark_samples,
    validate_validation_samples,
)
from evotensile.subprocess_utils import run_logged_process
from tests.helpers import sample_candidates


def test_timed_out_process_kills_descendants(tmp_path: Path):
    marker = tmp_path / "orphan-marker"
    script = tmp_path / "spawn_child.py"
    script.write_text(
        dedent(
            f"""\
            #!/usr/bin/env python3
            import subprocess
            import time

            subprocess.Popen([
                "python3",
                "-c",
                "import pathlib,time; time.sleep(0.3); pathlib.Path({str(marker)!r}).write_text('alive')",
            ])
            time.sleep(10)
            """
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    with (tmp_path / "stdout.log").open("w") as stdout, (tmp_path / "stderr.log").open("w") as stderr:
        returncode, timed_out = run_logged_process(
            [str(script)],
            stdout=stdout,
            stderr=stderr,
            env=None,
            timeout_s=0.1,
        )
    time.sleep(0.4)

    assert returncode == 124
    assert timed_out
    assert not marker.exists()


def _fake_structured_runner(path: Path) -> Path:
    script = path / "fake_structured_runner.py"
    script.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import argparse
            import json
            import os
            import time
            from pathlib import Path

            p = argparse.ArgumentParser()
            p.add_argument("--mode", choices=("validate", "benchmark"), required=True)
            p.add_argument("--pairs")
            p.add_argument("--output")
            p.add_argument("--library-dir", default=None)
            args, _ = p.parse_known_args()

            events_path = os.environ.get("EVOTENSILE_TEST_PHASE_EVENTS")
            if events_path:
                with Path(events_path).open("a") as events:
                    events.write(f"{args.mode}_start\\n")
            active_dir = os.environ.get("EVOTENSILE_TEST_GPU_ACTIVE_DIR") if args.mode == "benchmark" else None
            active_path = Path(active_dir) if active_dir else None
            if active_path is not None:
                try:
                    active_path.mkdir()
                except FileExistsError:
                    active_path.with_suffix(".overlap").write_text("overlap\\n")
                time.sleep(0.2)

            time_multipliers = json.loads(os.environ.get("EVOTENSILE_TEST_TIME_MULTIPLIERS", "{}"))
            with open(args.pairs) as src, open(args.output, "w") as out:
                for line in src:
                    pair = json.loads(line)
                    flops = 2.0 * pair["m"] * pair["n"] * pair["batch"] * pair["k"]
                    if args.mode == "validate":
                        out.write(
                            json.dumps(
                                {
                                    "shape_id": pair["shape_id"],
                                    "candidate_hash": pair["candidate_hash"],
                                    "status": "ok",
                                    "sample_index": 0,
                                    "time_us": None,
                                    "validation": "PASSED",
                                    "solution_index": pair["library_solution_index"],
                                },
                                sort_keys=True,
                            )
                            + "\\n"
                        )
                    else:
                        for sample_index in range(pair.get("num_benchmarks", 1)):
                            multiplier = float(time_multipliers.get(pair["candidate_hash"], 1.0))
                            time_us = max(1.0, flops / 1.0e9) * multiplier * (1.0 + sample_index * 0.001)
                            out.write(
                                json.dumps(
                                    {
                                        "shape_id": pair["shape_id"],
                                        "candidate_hash": pair["candidate_hash"],
                                        "status": "ok",
                                        "sample_index": sample_index,
                                        "time_us": time_us,
                                        "validation": "NO_CHECK",
                                        "solution_index": pair["library_solution_index"],
                                    },
                                    sort_keys=True,
                                )
                                + "\\n"
                            )
            if active_path is not None and active_path.exists():
                active_path.rmdir()
            if events_path:
                with Path(events_path).open("a") as events:
                    events.write(f"{args.mode}_end\\n")
            """
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _fake_build_tensile(path: Path) -> Path:
    script = path / "fake_tensile_build.py"
    script.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import os
            import sys
            from pathlib import Path

            import yaml

            events_path = os.environ.get("EVOTENSILE_TEST_PHASE_EVENTS")
            if events_path:
                with Path(events_path).open("a") as events:
                    events.write("compile_start\\n")
            config_path, out = Path(sys.argv[1]), Path(sys.argv[2])
            out.mkdir(parents=True, exist_ok=True)
            if "--build-only" not in sys.argv:
                sys.exit(9)
            config = yaml.safe_load(config_path.read_text())
            problem = config["BenchmarkProblems"][0][1]
            problem_sizes = problem["BenchmarkFinalParameters"][0]["ProblemSizes"]
            solutions = []
            for i, item in enumerate(problem["ForkParameters"][0]["Groups"][0]):
                sol = dict(item)
                mi = sol["MatrixInstruction"]
                sol["MatrixInstruction"] = mi[:4]
                sol["MIWaveTile"] = [mi[5], mi[6]]
                sol["MIWaveGroup"] = [mi[7], mi[8]]
                sol["SolutionIndex"] = i
                sol["KernelNameMin"] = f"Kernel{i}"
                solutions.append(sol)
            final = [{"MinimumRequiredVersion": "5.0.0"}, {"ProblemSizes": problem_sizes}, *solutions]
            (out / "00_Final.yaml").write_text(yaml.safe_dump(final, sort_keys=False))
            lib = out / "4_LibraryClient" / "library" / "gfx1151"
            lib.mkdir(parents=True, exist_ok=True)
            (lib / "TensileLibrary_gfx1151.yaml").write_text("---\\nsolutions: []\\n")
            (lib / "Kernels.so-000-gfx1151.hsaco").write_bytes(b"fake")
            if events_path:
                with Path(events_path).open("a") as events:
                    events.write("compile_end\\n")
            sys.exit(int(os.environ.get("EVOTENSILE_TEST_BUILD_RETURNCODE", "0")))
            """
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _pair() -> RunnablePair:
    return RunnablePair(
        shape_id="m1_n1_b1_k1",
        candidate_hash="cand_1",
        problem_index=0,
        requested_solution_index=0,
        library_solution_index=0,
        manifest_solution_index=0,
    )


def test_validation_samples_create_separate_correctness_evidence():
    pair = _pair()
    outcome = validate_validation_samples(
        [
            StructuredSample(
                shape_id=pair.shape_id,
                candidate_hash=pair.candidate_hash,
                status="ok",
                sample_index=0,
                time_us=None,
                validation="PASSED checked=1 backend=hipblaslt_gpu_compare",
                solution_index=0,
            )
        ],
        runnable_pairs=[pair],
        problem_type_hash="ptype",
        validation_protocol_hash="vproto",
        benchmark_protocol_hash="bproto",
        run_id="validation_run",
    )

    assert outcome.passed_pairs == [pair]
    assert outcome.validations[0].status == "passed"
    assert outcome.validations[0].detail == "PASSED checked=1 backend=hipblaslt_gpu_compare"
    assert outcome.negative_evaluations == []


def test_validation_failure_creates_reusable_negative_evaluation():
    pair = _pair()
    outcome = validate_validation_samples(
        [
            StructuredSample(
                shape_id=pair.shape_id,
                candidate_hash=pair.candidate_hash,
                status="validation_fail",
                validation="FAILED mismatch",
                solution_index=0,
            )
        ],
        runnable_pairs=[pair],
        problem_type_hash="ptype",
        validation_protocol_hash="vproto",
        benchmark_protocol_hash="bproto",
        run_id="validation_run",
    )

    assert outcome.passed_pairs == []
    assert outcome.validations[0].status == "failed"
    assert outcome.negative_evaluations[0].status == "validation_fail"


def test_benchmark_samples_must_be_timing_only():
    pair = _pair()
    inserts = validate_benchmark_samples(
        [
            StructuredSample(
                shape_id=pair.shape_id,
                candidate_hash=pair.candidate_hash,
                status="ok",
                sample_index=0,
                time_us=1.0,
                validation="NO_CHECK",
                solution_index=0,
            )
        ],
        runnable_pairs=[pair],
        protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1, num_elements_to_validate=0),
        problem_type_hash="ptype",
        benchmark_protocol_hash="bproto",
        run_id="benchmark_run",
    )

    assert inserts[0].status == "ok"
    assert inserts[0].time_us == 1.0
    assert inserts[0].validation == "PASSED prior_validation"


def test_library_dir_from_build_accepts_tensilelite_build_only_cache_layout(tmp_path: Path):
    build_dir = tmp_path / "build"
    cache_lib = (
        build_dir
        / "1_BenchmarkProblems"
        / "Cijk_Ailk_Bjlk_HHS_BH_Bias_H_HA_S_SAV_UserArgs_00"
        / "00_Final"
        / "caches"
        / "abc123"
        / "source"
        / "library"
        / "gfx1151"
    )
    cache_lib.mkdir(parents=True)

    assert library_dir_from_build(build_dir) == cache_lib


def test_run_structured_phase_passes_explicit_mode_and_backend(tmp_path: Path):
    runner = tmp_path / "runner.py"
    runner.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import json
            import sys
            from pathlib import Path

            Path(sys.argv[sys.argv.index("--output") + 1]).write_text("")
            Path(sys.argv[sys.argv.index("--output") + 1]).with_suffix(".argv.json").write_text(json.dumps(sys.argv[1:]))
            """
        ),
        encoding="utf-8",
    )
    runner.chmod(0o755)
    pair = RunnablePair(
        shape_id="m1_n1_b1_k1",
        candidate_hash="cand_1",
        problem_index=0,
        requested_solution_index=0,
        library_solution_index=0,
        manifest_solution_index=0,
    )

    output = run_structured_phase(
        mode="validate",
        run_dir=tmp_path,
        pairs=[pair],
        shapes=[pilot_100_shapes()[0].__class__(m=1, n=1, batch=1, k=1)],
        protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1),
        runner_bin=runner,
        library_dir=tmp_path,
    )

    assert output.returncode == 0
    assert output.command[output.command.index("--mode") + 1] == "validate"
    assert "--validation-backend" in output.command
    assert output.command[output.command.index("--validation-backend") + 1] == "hipblaslt"


def test_parallel_prepare_finishes_before_serial_benchmark_queue(tmp_path: Path, monkeypatch):
    fake_tensile = _fake_build_tensile(tmp_path)
    fake_runner = _fake_structured_runner(tmp_path)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidates = sample_candidates(2)
    shape = pilot_100_shapes()[0]
    events_path = tmp_path / "phase-events.log"
    monkeypatch.setenv("EVOTENSILE_APU_LOCK_PATH", str(tmp_path / "apu.lock"))
    monkeypatch.setenv("EVOTENSILE_TEST_GPU_ACTIVE_DIR", str(tmp_path / "gpu-active"))
    monkeypatch.setenv("EVOTENSILE_TEST_PHASE_EVENTS", str(events_path))

    result = execute_schedule(
        db,
        shapes=[shape],
        candidates=candidates,
        output_root=tmp_path / "batches",
        protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1),
        candidate_batch_size=1,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=fake_runner,
        keep_going=True,
        prepare_workers=2,
    )

    assert len(result.executed_batches) == 2
    events = events_path.read_text(encoding="utf-8").splitlines()
    assert events.count("compile_end") == 2
    assert events.count("validate_end") == 2
    prepare_end_indices = [i for i, event in enumerate(events) if event in {"compile_end", "validate_end"}]
    assert events.index("benchmark_start") > max(prepare_end_indices)
    assert not (tmp_path / "gpu-active.overlap").exists()
    assert not (tmp_path / "gpu-active").exists()
    assert db.cache_summary() == {"ok": 2}


def test_adaptive_topup_reuses_prepared_artifacts(tmp_path: Path, monkeypatch):
    fake_tensile = _fake_build_tensile(tmp_path)
    fake_runner = _fake_structured_runner(tmp_path)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidates = sample_candidates(2)
    shape = pilot_100_shapes()[0]
    events_path = tmp_path / "adaptive-events.log"
    monkeypatch.setenv("EVOTENSILE_APU_LOCK_PATH", str(tmp_path / "apu.lock"))
    monkeypatch.setenv("EVOTENSILE_TEST_GPU_ACTIVE_DIR", str(tmp_path / "gpu-active"))
    monkeypatch.setenv("EVOTENSILE_TEST_PHASE_EVENTS", str(events_path))
    decision_calls = 0

    def forced_topup(*args, **kwargs):
        nonlocal decision_calls
        decision_calls += 1
        return [(3, [shape], candidates)] if decision_calls == 1 else []

    monkeypatch.setattr("evotensile.scheduler._adaptive_topup_groups", forced_topup)

    result = execute_schedule(
        db,
        shapes=[shape],
        candidates=candidates,
        output_root=tmp_path / "batches",
        protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1),
        candidate_batch_size=2,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=fake_runner,
        keep_going=True,
        prepare_workers=2,
        adaptive_policy=AdaptivePolicy(),
        probe_policy=ProbePolicy(samples=2, min_survivors=2),
        adaptive_max_rounds=2,
    )

    events = events_path.read_text(encoding="utf-8").splitlines()
    assert events.count("compile_start") == 1
    assert events.count("validate_start") == 1
    assert events.count("benchmark_start") == 3
    assert result.adaptive_rounds == 1
    assert [batch.phase for batch in result.executed_batches] == ["probe", "initial", "adaptive"]
    assert not (tmp_path / "gpu-active.overlap").exists()
    assert not (tmp_path / "gpu-active").exists()
    assert db.cache_summary() == {"ok": 10}


def test_adaptive_probe_limits_slow_candidates_to_three_launches(tmp_path: Path, monkeypatch):
    fake_tensile = _fake_build_tensile(tmp_path)
    fake_runner = _fake_structured_runner(tmp_path)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidates = sample_candidates(2)
    shape = pilot_100_shapes()[0]
    main_protocol = DEFAULT_BENCHMARK_PROTOCOL.with_overrides(
        num_warmups=1,
        num_benchmarks=2,
        enqueues_per_sync=10,
    )
    probe_policy = ProbePolicy(samples=3, max_slowdown_factor=4.0, min_survivors=1)
    monkeypatch.setenv(
        "EVOTENSILE_TEST_TIME_MULTIPLIERS",
        json.dumps({candidates[1].hash: 10.0}),
    )

    result = execute_schedule(
        db,
        shapes=[shape],
        candidates=candidates,
        output_root=tmp_path / "batches",
        protocol=main_protocol,
        candidate_batch_size=2,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=fake_runner,
        keep_going=True,
        adaptive_policy=AdaptivePolicy(),
        probe_policy=probe_policy,
        adaptive_max_rounds=0,
    )

    probe_protocol = main_protocol.with_overrides(
        role="probe",
        num_warmups=0,
        num_benchmarks=3,
        enqueues_per_sync=1,
        syncs_per_benchmark=1,
        num_elements_to_validate=0,
    )
    probe_rank = db.rank_evaluations(
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=probe_protocol.protocol_hash(),
        shape_id=shape.id,
    )
    main_rank = db.rank_evaluations(
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=main_protocol.protocol_hash(),
        shape_id=shape.id,
    )

    assert result.probe_survivor_pairs == 1
    assert result.probe_screened_pairs == 1
    assert [summary.samples for summary in probe_rank] == [3, 3]
    assert [(summary.candidate_hash, summary.samples) for summary in main_rank] == [(candidates[0].hash, 2)]
    assert [batch.phase for batch in result.executed_batches] == ["probe", "initial"]

    pair_files = list((tmp_path / "batches").rglob("benchmark_*.pairs.jsonl"))
    pair_groups = [[json.loads(line) for line in path.read_text().splitlines()] for path in pair_files]
    probe_pairs = next(rows for rows in pair_groups if rows[0]["enqueues_per_sync"] == 1)
    main_pairs = next(rows for rows in pair_groups if rows[0]["enqueues_per_sync"] == 10)
    assert len(probe_pairs) == 2
    assert all(row["num_benchmarks"] == 3 and row["num_warmups"] == 0 for row in probe_pairs)
    assert len(main_pairs) == 1
    assert main_pairs[0]["candidate_hash"] == candidates[0].hash
    assert main_pairs[0]["num_benchmarks"] == 2


def test_adaptive_probe_uses_compatible_db_incumbent(tmp_path: Path, monkeypatch):
    fake_tensile = _fake_build_tensile(tmp_path)
    fake_runner = _fake_structured_runner(tmp_path)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(2)
    shape = pilot_100_shapes()[0]
    main_protocol = DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=2)
    db.register_shapes([shape])
    db.insert_evaluation(
        shape_id=shape.id,
        candidate_hash="incumbent",
        run_id="prior",
        status="ok",
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=main_protocol.protocol_hash(),
        time_us=max(1.0, 2.0 * shape.m * shape.n * shape.batch * shape.k / 1.0e9),
        validation="PASSED prior_validation",
    )
    monkeypatch.setenv(
        "EVOTENSILE_TEST_TIME_MULTIPLIERS",
        json.dumps({candidates[0].hash: 4.5, candidates[1].hash: 6.0}),
    )

    result = execute_schedule(
        db,
        shapes=[shape],
        candidates=candidates,
        output_root=tmp_path / "batches",
        protocol=main_protocol,
        candidate_batch_size=2,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=fake_runner,
        keep_going=True,
        adaptive_policy=AdaptivePolicy(),
        probe_policy=ProbePolicy(samples=3, max_slowdown_factor=4.0, min_survivors=1),
        adaptive_max_rounds=0,
    )

    main_rank = db.rank_evaluations(
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=main_protocol.protocol_hash(),
        shape_id=shape.id,
    )
    by_hash = {summary.candidate_hash: summary.samples for summary in main_rank}
    assert result.probe_survivor_pairs == 1
    assert result.probe_screened_pairs == 1
    assert by_hash == {"incumbent": 1, candidates[0].hash: 2}


def test_singleton_nonzero_build_salvages_runnable_artifact(tmp_path: Path, monkeypatch):
    fake_tensile = _fake_build_tensile(tmp_path)
    fake_runner = _fake_structured_runner(tmp_path)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    monkeypatch.setenv("EVOTENSILE_TEST_BUILD_RETURNCODE", "2")

    result = execute_schedule(
        db,
        shapes=pilot_100_shapes()[:1],
        candidates=sample_candidates(1),
        output_root=tmp_path / "batches",
        protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1),
        candidate_batch_size=1,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=fake_runner,
        keep_going=True,
    )

    executed = result.executed_batches[0]
    assert executed.build_returncode == 2
    assert executed.validation_returncode == 0
    assert executed.runner_returncode == 0
    assert executed.ingest is not None
    assert executed.ingest.status_counts == {"ok": 1}
    assert db.cache_summary() == {"ok": 1}


def test_structured_external_runner_ingests_exact_shape_candidate_rows(tmp_path: Path):
    fake_tensile = _fake_build_tensile(tmp_path)
    fake_runner = _fake_structured_runner(tmp_path)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidates = sample_candidates(2)
    shapes = pilot_100_shapes()[:2]
    protocol = DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=3)

    result = execute_schedule(
        db,
        shapes=shapes,
        candidates=candidates,
        output_root=tmp_path / "batches",
        protocol=protocol,
        candidate_batch_size=2,
        shape_batch_size=2,
        tensilelite_bin=fake_tensile,
        runner_bin=fake_runner,
        keep_going=True,
    )

    assert len(result.executed_batches) == 1
    executed = result.executed_batches[0]
    assert executed.build_returncode == 0
    assert executed.runner_returncode == 0
    assert executed.ingest is not None
    assert executed.ingest.status_counts == {"ok": 12}
    assert db.cache_summary() == {"ok": 12}

    result_files = list(executed.output_dir.glob("benchmark_*.results.jsonl"))
    assert len(result_files) == 1
    samples = read_structured_results(result_files[0])
    assert {(s.shape_id, s.candidate_hash) for s in samples} == {
        (shape.id, candidate.hash) for shape in shapes for candidate in candidates
    }
    assert all(sample.validation == "NO_CHECK" for sample in samples)

    with sqlite3.connect(tmp_path / "sched.sqlite") as con:
        rows = con.execute(
            "SELECT shape_id, candidate_hash, COUNT(*) FROM evaluations "
            "WHERE status='ok' GROUP BY shape_id, candidate_hash ORDER BY shape_id, candidate_hash"
        ).fetchall()
    assert rows == sorted((shape.id, candidate.hash, 3) for shape in shapes for candidate in candidates)


def test_structured_external_runner_topup_reuses_prior_validation(tmp_path: Path):
    fake_tensile = _fake_build_tensile(tmp_path)
    fake_runner = _fake_structured_runner(tmp_path)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]

    first = execute_schedule(
        db,
        shapes=[shape],
        candidates=[candidate],
        output_root=tmp_path / "batches",
        protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1),
        candidate_batch_size=1,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=fake_runner,
        keep_going=True,
    )
    second = execute_schedule(
        db,
        shapes=[shape],
        candidates=[candidate],
        output_root=tmp_path / "batches",
        protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=2),
        min_samples=2,
        candidate_batch_size=1,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=fake_runner,
        keep_going=True,
    )

    assert first.executed_batches[0].planned.requires_validation
    assert not second.executed_batches[0].planned.requires_validation
    with sqlite3.connect(tmp_path / "sched.sqlite") as con:
        timing_rows = con.execute(
            "SELECT status, validation, COUNT(*) FROM evaluations GROUP BY status, validation ORDER BY validation"
        ).fetchall()
        validation_rows = con.execute("SELECT status, COUNT(*) FROM validations GROUP BY status").fetchall()
    assert timing_rows == [("ok", "PASSED prior_validation", 2)]
    assert validation_rows == [("passed", 1)]


def test_compile_cache_reuses_tensilelite_build_dir_across_runs(tmp_path: Path):
    fake_tensile = tmp_path / "fake_tensile_cache.py"
    fake_tensile.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import json
            import sys
            from pathlib import Path

            import yaml

            config_path, out = Path(sys.argv[1]), Path(sys.argv[2])
            out.mkdir(parents=True, exist_ok=True)
            calls = out / "calls.jsonl"
            with calls.open("a") as handle:
                handle.write(json.dumps(sys.argv[1:]) + "\\n")
            if "--build-only" not in sys.argv:
                sys.exit(9)
            config = yaml.safe_load(config_path.read_text())
            problem = config["BenchmarkProblems"][0][1]
            problem_sizes = problem["BenchmarkFinalParameters"][0]["ProblemSizes"]
            solutions = []
            for i, item in enumerate(problem["ForkParameters"][0]["Groups"][0]):
                sol = dict(item)
                mi = sol["MatrixInstruction"]
                sol["MatrixInstruction"] = mi[:4]
                sol["MIWaveTile"] = [mi[5], mi[6]]
                sol["MIWaveGroup"] = [mi[7], mi[8]]
                sol["SolutionIndex"] = i
                sol["KernelNameMin"] = f"Kernel{i}"
                solutions.append(sol)
            final = [{"MinimumRequiredVersion": "5.0.0"}, {"ProblemSizes": problem_sizes}, *solutions]
            (out / "00_Final.yaml").write_text(yaml.safe_dump(final, sort_keys=False))
            cache = out / "1_BenchmarkProblems" / "ptype" / "00_Final" / "caches" / "abc123"
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "cache.yaml").write_text("CodeObjectFiles: []\\nLibraryFile: library.yaml\\n")
            lib = cache / "source" / "library" / "gfx1151"
            lib.mkdir(parents=True, exist_ok=True)
            (lib / "TensileLibrary_gfx1151.yaml").write_text("---\\nsolutions: []\\n")
            (lib / "Kernels.so-000-gfx1151.hsaco").write_bytes(b"fake")
            sys.exit(0)
            """
        ),
        encoding="utf-8",
    )
    fake_tensile.chmod(0o755)
    fake_runner = _fake_structured_runner(tmp_path)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    compile_cache_root = tmp_path / "compile_cache"

    first = execute_schedule(
        db,
        shapes=[shape],
        candidates=[candidate],
        output_root=tmp_path / "batches",
        protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1),
        candidate_batch_size=1,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=fake_runner,
        keep_going=True,
        compile_cache_root=compile_cache_root,
    )
    second = execute_schedule(
        db,
        shapes=[shape],
        candidates=[candidate],
        output_root=tmp_path / "batches",
        protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=2),
        min_samples=2,
        candidate_batch_size=1,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=fake_runner,
        keep_going=True,
        compile_cache_root=compile_cache_root,
    )

    assert first.executed_batches[0].build_output_dir == second.executed_batches[0].build_output_dir
    build_dir = first.executed_batches[0].build_output_dir
    assert build_dir is not None
    calls = [json.loads(line) for line in (build_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "--use-cache" not in calls[0]
    assert "--use-cache" in calls[1]
    assert (build_dir / ".evotensile_compile_cache_ok").exists()
    assert first.executed_batches[0].output_dir != build_dir
    assert second.executed_batches[0].output_dir != build_dir


def test_structured_external_backend_rejects_unexpected_pair(tmp_path: Path):
    fake_tensile = _fake_build_tensile(tmp_path)
    runner = tmp_path / "bad_runner.py"
    runner.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import argparse
            import json

            p = argparse.ArgumentParser()
            p.add_argument("--pairs")
            p.add_argument("--output")
            p.add_argument("--library-dir")
            args, _ = p.parse_known_args()

            with open(args.output, "w") as out:
                out.write(
                    json.dumps(
                        {
                            "shape_id": "m1_n1_b1_k1",
                            "candidate_hash": "cand_bad",
                            "status": "ok",
                            "time_us": 1,
                            "validation": "PASSED",
                        }
                    )
                    + "\\n"
                )
            """
        ),
        encoding="utf-8",
    )
    runner.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")

    result = execute_schedule(
        db,
        shapes=pilot_100_shapes()[:1],
        candidates=sample_candidates(1),
        output_root=tmp_path / "batches",
        tensilelite_bin=fake_tensile,
        runner_bin=runner,
        keep_going=True,
    )

    assert len(result.executed_batches) == 1
    assert result.executed_batches[0].ingest is not None
    assert result.executed_batches[0].ingest.ok is False
    assert "unexpected pair" in result.executed_batches[0].ingest.errors[0]
    assert db.cache_summary() == {}


def test_structured_external_backend_rejects_wrong_solution_index(tmp_path: Path):
    fake_tensile = _fake_build_tensile(tmp_path)
    runner = tmp_path / "wrong_solution_runner.py"
    runner.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import argparse
            import json

            p = argparse.ArgumentParser()
            p.add_argument("--pairs")
            p.add_argument("--output")
            p.add_argument("--library-dir")
            args, _ = p.parse_known_args()

            with open(args.pairs) as src, open(args.output, "w") as out:
                pair = json.loads(next(src))
                out.write(
                    json.dumps(
                        {
                            "shape_id": pair["shape_id"],
                            "candidate_hash": pair["candidate_hash"],
                            "status": "ok",
                            "sample_index": 0,
                            "time_us": 1,
                            "validation": "PASSED",
                            "solution_index": pair["library_solution_index"] + 1,
                        }
                    )
                    + "\\n"
                )
            """
        ),
        encoding="utf-8",
    )
    runner.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")

    result = execute_schedule(
        db,
        shapes=pilot_100_shapes()[:1],
        candidates=sample_candidates(1),
        output_root=tmp_path / "batches",
        protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1),
        tensilelite_bin=fake_tensile,
        runner_bin=runner,
        keep_going=True,
    )

    ingest = result.executed_batches[0].ingest
    assert ingest is not None
    assert ingest.ok is False
    assert "wrong solution_index" in ingest.errors[0]
    assert db.cache_summary() == {}


def test_structured_external_backend_rejects_incomplete_samples(tmp_path: Path):
    fake_tensile = _fake_build_tensile(tmp_path)
    runner = tmp_path / "incomplete_runner.py"
    runner.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import argparse
            import json

            p = argparse.ArgumentParser()
            p.add_argument("--mode", required=True)
            p.add_argument("--pairs")
            p.add_argument("--output")
            p.add_argument("--library-dir")
            args, _ = p.parse_known_args()

            with open(args.pairs) as src, open(args.output, "w") as out:
                pair = json.loads(next(src))
                out.write(
                    json.dumps(
                        {
                            "shape_id": pair["shape_id"],
                            "candidate_hash": pair["candidate_hash"],
                            "status": "ok",
                            "sample_index": 0,
                            "time_us": None if args.mode == "validate" else 1,
                            "validation": "PASSED" if args.mode == "validate" else "NO_CHECK",
                            "solution_index": pair["library_solution_index"],
                        }
                    )
                    + "\\n"
                )
            """
        ),
        encoding="utf-8",
    )
    runner.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")

    result = execute_schedule(
        db,
        shapes=pilot_100_shapes()[:1],
        candidates=sample_candidates(1),
        output_root=tmp_path / "batches",
        protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=2),
        tensilelite_bin=fake_tensile,
        runner_bin=runner,
        keep_going=True,
    )

    ingest = result.executed_batches[0].ingest
    assert ingest is not None
    assert ingest.ok is False
    assert "incomplete sample set" in ingest.errors[0]
    assert db.cache_summary() == {}


def test_structured_external_backend_rejects_duplicate_sample_indices(tmp_path: Path):
    fake_tensile = _fake_build_tensile(tmp_path)
    runner = tmp_path / "duplicate_runner.py"
    runner.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import argparse
            import json

            p = argparse.ArgumentParser()
            p.add_argument("--mode", required=True)
            p.add_argument("--pairs")
            p.add_argument("--output")
            p.add_argument("--library-dir")
            args, _ = p.parse_known_args()

            with open(args.pairs) as src, open(args.output, "w") as out:
                pair = json.loads(next(src))
                count = 1 if args.mode == "validate" else 2
                for _ in range(count):
                    out.write(
                        json.dumps(
                            {
                                "shape_id": pair["shape_id"],
                                "candidate_hash": pair["candidate_hash"],
                                "status": "ok",
                                "sample_index": 0,
                                "time_us": None if args.mode == "validate" else 1,
                                "validation": "PASSED" if args.mode == "validate" else "NO_CHECK",
                                "solution_index": pair["library_solution_index"],
                            }
                        )
                        + "\\n"
                    )
            """
        ),
        encoding="utf-8",
    )
    runner.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")

    result = execute_schedule(
        db,
        shapes=pilot_100_shapes()[:1],
        candidates=sample_candidates(1),
        output_root=tmp_path / "batches",
        protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=2),
        tensilelite_bin=fake_tensile,
        runner_bin=runner,
        keep_going=True,
    )

    ingest = result.executed_batches[0].ingest
    assert ingest is not None
    assert ingest.ok is False
    assert "duplicate sample_index" in ingest.errors[0]
    assert db.cache_summary() == {}


def test_structured_external_backend_rejects_nonzero_return_with_positive_rows(tmp_path: Path):
    fake_tensile = _fake_build_tensile(tmp_path)
    runner = tmp_path / "nonzero_runner.py"
    runner.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import argparse
            import json
            import sys

            p = argparse.ArgumentParser()
            p.add_argument("--mode", required=True)
            p.add_argument("--pairs")
            p.add_argument("--output")
            p.add_argument("--library-dir")
            args, _ = p.parse_known_args()

            with open(args.pairs) as src, open(args.output, "w") as out:
                pair = json.loads(next(src))
                out.write(
                    json.dumps(
                        {
                            "shape_id": pair["shape_id"],
                            "candidate_hash": pair["candidate_hash"],
                            "status": "ok",
                            "sample_index": 0,
                            "time_us": None if args.mode == "validate" else 1,
                            "validation": "PASSED" if args.mode == "validate" else "NO_CHECK",
                            "solution_index": pair["library_solution_index"],
                        }
                    )
                    + "\\n"
                )
            sys.exit(3 if args.mode == "benchmark" else 0)
            """
        ),
        encoding="utf-8",
    )
    runner.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")

    result = execute_schedule(
        db,
        shapes=pilot_100_shapes()[:1],
        candidates=sample_candidates(1),
        output_root=tmp_path / "batches",
        protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1),
        tensilelite_bin=fake_tensile,
        runner_bin=runner,
        keep_going=True,
    )

    ingest = result.executed_batches[0].ingest
    assert result.executed_batches[0].runner_returncode == 3
    assert ingest is not None
    assert ingest.ok is False
    assert "positive result rows" in ingest.errors[0]
    assert db.cache_summary() == {}


def test_structured_external_backend_records_runner_timeout(tmp_path: Path):
    fake_tensile = _fake_build_tensile(tmp_path)
    runner = tmp_path / "slow_runner.py"
    runner.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import argparse
            import json
            import time

            parser = argparse.ArgumentParser()
            parser.add_argument("--mode", required=True)
            parser.add_argument("--pairs")
            parser.add_argument("--output")
            args, _ = parser.parse_known_args()
            if args.mode == "benchmark":
                time.sleep(10)
            with open(args.pairs) as src, open(args.output, "w") as out:
                pair = json.loads(next(src))
                out.write(json.dumps({
                    "shape_id": pair["shape_id"],
                    "candidate_hash": pair["candidate_hash"],
                    "status": "ok",
                    "sample_index": 0,
                    "time_us": None,
                    "validation": "PASSED",
                    "solution_index": pair["library_solution_index"],
                }) + "\\n")
            """
        ),
        encoding="utf-8",
    )
    runner.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")

    result = execute_schedule(
        db,
        shapes=pilot_100_shapes()[:1],
        candidates=sample_candidates(1),
        output_root=tmp_path / "batches",
        protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1),
        tensilelite_bin=fake_tensile,
        runner_bin=runner,
        runner_timeout_s=0.1,
        keep_going=True,
    )

    executed = result.executed_batches[0]
    assert executed.runner_returncode == 124
    assert executed.ingest is not None
    assert executed.ingest.ok is False
    assert "benchmark phase timed out" in executed.ingest.errors[0]
    assert db.cache_summary() == {"runner_timeout": 1}


def test_structured_maps_renumbered_normalized_final_yaml_solution(tmp_path: Path):
    script = tmp_path / "fake_normalizing_tensile.py"
    script.write_text(
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
            sol["GlobalReadVectorWidthA"] = 1
            sol["GlobalReadVectorWidthB"] = 1
            sol["StaggerUStride"] = 256.0
            sol["StoreVectorWidth"] = 1
            (out / "00_Final.yaml").write_text(
                yaml.safe_dump(
                    [{"MinimumRequiredVersion": "5.0.0"}, {"ProblemSizes": problem_sizes}, sol],
                    sort_keys=False,
                )
            )
            lib = out / "4_LibraryClient" / "library" / "gfx1151"
            lib.mkdir(parents=True, exist_ok=True)
            (lib / "TensileLibrary_gfx1151.yaml").write_text("---\\nsolutions: []\\n")
            (lib / "Kernels.so-000-gfx1151.hsaco").write_bytes(b"fake")
            sys.exit(0)
            """
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidates = sample_candidates(2)

    result = execute_schedule(
        db,
        shapes=pilot_100_shapes()[:1],
        candidates=candidates,
        output_root=tmp_path / "batches",
        protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1),
        tensilelite_bin=script,
        runner_bin=_fake_structured_runner(tmp_path),
        keep_going=True,
    )

    assert len(result.executed_batches) == 1
    assert result.executed_batches[0].ingest is not None
    assert result.executed_batches[0].ingest.status_counts == {"ok": 1, "rejected": 1}
    with sqlite3.connect(tmp_path / "sched.sqlite") as con:
        rows = con.execute(
            "SELECT candidate_hash, status, solution_index FROM evaluations ORDER BY status, candidate_hash"
        ).fetchall()
    assert (candidates[1].hash, "ok", 0) in rows
    assert (candidates[0].hash, "rejected", None) in rows


def test_structured_records_rejected_candidate_from_final_yaml(tmp_path: Path):
    script = tmp_path / "fake_rejecting_tensile.py"
    script.write_text(
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
            item = problem["ForkParameters"][0]["Groups"][0][0]
            sol = dict(item)
            mi = sol["MatrixInstruction"]
            sol["MatrixInstruction"] = mi[:4]
            sol["MIWaveTile"] = [mi[5], mi[6]]
            sol["MIWaveGroup"] = [mi[7], mi[8]]
            sol["SolutionIndex"] = 0
            sol["KernelNameMin"] = "Kernel0"
            (out / "00_Final.yaml").write_text(
                yaml.safe_dump(
                    [{"MinimumRequiredVersion": "5.0.0"}, {"ProblemSizes": problem_sizes}, sol],
                    sort_keys=False,
                )
            )
            lib = out / "4_LibraryClient" / "library" / "gfx1151"
            lib.mkdir(parents=True, exist_ok=True)
            (lib / "TensileLibrary_gfx1151.yaml").write_text("---\\nsolutions: []\\n")
            (lib / "Kernels.so-000-gfx1151.hsaco").write_bytes(b"fake")
            sys.exit(0)
            """
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidates = sample_candidates(2)

    result = execute_schedule(
        db,
        shapes=pilot_100_shapes()[:1],
        candidates=candidates,
        output_root=tmp_path / "batches",
        protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=2),
        tensilelite_bin=script,
        runner_bin=_fake_structured_runner(tmp_path),
        keep_going=True,
    )

    assert len(result.executed_batches) == 1
    assert result.executed_batches[0].ingest is not None
    assert result.executed_batches[0].ingest.status_counts == {"ok": 2, "rejected": 1}
    assert db.cache_summary() == {"ok": 2, "rejected": 1}
