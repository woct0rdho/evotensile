import sqlite3
from pathlib import Path
from textwrap import dedent

from evotensile.database import EvoTensileDB
from evotensile.protocol import DEFAULT_BENCHMARK_PROTOCOL
from evotensile.scheduler import execute_schedule
from evotensile.shapes import pilot_100_shapes
from evotensile.structured_runner import (
    RunnablePair,
    StructuredSample,
    _library_dir_from_run,
    read_structured_results,
    validate_structured_samples,
)
from tests.helpers import sample_candidates


def _fake_structured_runner(path: Path) -> Path:
    script = path / "fake_structured_runner.py"
    script.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import argparse
            import json

            p = argparse.ArgumentParser()
            p.add_argument("--pairs")
            p.add_argument("--output")
            p.add_argument("--library-dir", default=None)
            args = p.parse_args()

            with open(args.pairs) as src, open(args.output, "w") as out:
                for line in src:
                    pair = json.loads(line)
                    flops = 2.0 * pair["m"] * pair["n"] * pair["batch"] * pair["k"]
                    for sample_index in range(pair.get("num_benchmarks", 1)):
                        time_us = max(1.0, flops / 1.0e9) * (1.0 + sample_index * 0.001)
                        out.write(
                            json.dumps(
                                {
                                    "shape_id": pair["shape_id"],
                                    "candidate_hash": pair["candidate_hash"],
                                    "status": "ok",
                                    "sample_index": sample_index,
                                    "time_us": time_us,
                                    "validation": "NO_CHECK" if pair.get("num_elements_to_validate") == 0 else "PASSED",
                                    "solution_index": pair["library_solution_index"],
                                },
                                sort_keys=True,
                            )
                            + "\\n"
                        )
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
            import sys
            from pathlib import Path

            import yaml

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
            sys.exit(0)
            """
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def test_validate_structured_samples_requires_prior_validation_for_no_check_rows():
    pair = RunnablePair(
        shape_id="m1_n1_b1_k1",
        candidate_hash="cand_1",
        problem_index=0,
        requested_solution_index=0,
        library_solution_index=0,
        manifest_solution_index=0,
    )
    sample = StructuredSample(
        shape_id=pair.shape_id,
        candidate_hash=pair.candidate_hash,
        status="ok",
        sample_index=0,
        time_us=1.0,
        validation="NO_CHECK",
        solution_index=0,
    )

    inserts = validate_structured_samples(
        [sample],
        runnable_pairs=[pair],
        protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1, num_elements_to_validate=0),
    )

    assert inserts[0].status == "validation_unknown"


def test_validate_structured_samples_accepts_negative_rows_without_sample_index():
    pair = RunnablePair(
        shape_id="m1_n1_b1_k1",
        candidate_hash="cand_1",
        problem_index=0,
        requested_solution_index=0,
        library_solution_index=0,
        manifest_solution_index=0,
    )
    sample = StructuredSample(
        shape_id=pair.shape_id,
        candidate_hash=pair.candidate_hash,
        status="rejected",
        sample_index=None,
        time_us=None,
        validation="DID_NOT_SATISFY_ASSERTS",
        solution_index=0,
    )

    inserts = validate_structured_samples(
        [sample],
        runnable_pairs=[pair],
        protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1),
    )

    assert inserts[0].status == "rejected"
    assert inserts[0].validation == "DID_NOT_SATISFY_ASSERTS"


def test_validate_structured_samples_accepts_trusted_no_check_rows():
    pair = RunnablePair(
        shape_id="m1_n1_b1_k1",
        candidate_hash="cand_1",
        problem_index=0,
        requested_solution_index=0,
        library_solution_index=0,
        manifest_solution_index=0,
    )
    sample = StructuredSample(
        shape_id=pair.shape_id,
        candidate_hash=pair.candidate_hash,
        status="ok",
        sample_index=0,
        time_us=1.0,
        validation="NO_CHECK",
        solution_index=0,
    )

    inserts = validate_structured_samples(
        [sample],
        runnable_pairs=[pair],
        protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1, num_elements_to_validate=0),
        allow_no_check=True,
    )

    assert inserts[0].status == "ok"


def test_library_dir_from_run_accepts_tensilelite_build_only_cache_layout(tmp_path: Path):
    run_dir = tmp_path / "run"
    cache_lib = (
        run_dir
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

    assert _library_dir_from_run(run_dir) == cache_lib


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

    result_files = list(executed.output_dir.glob("*.results.jsonl"))
    assert len(result_files) == 1
    samples = read_structured_results(result_files[0])
    assert {(s.shape_id, s.candidate_hash) for s in samples} == {
        (shape.id, candidate.hash) for shape in shapes for candidate in candidates
    }
    assert all(sample.validation == "PASSED" for sample in samples)

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
        rows = con.execute(
            "SELECT status, validation, COUNT(*) FROM evaluations GROUP BY status, validation ORDER BY validation"
        ).fetchall()
    assert rows == [("ok", "NO_CHECK", 1), ("ok", "PASSED", 1)]


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
            args = p.parse_args()

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
            args = p.parse_args()

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
            p.add_argument("--pairs")
            p.add_argument("--output")
            p.add_argument("--library-dir")
            args = p.parse_args()

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
            p.add_argument("--pairs")
            p.add_argument("--output")
            p.add_argument("--library-dir")
            args = p.parse_args()

            with open(args.pairs) as src, open(args.output, "w") as out:
                pair = json.loads(next(src))
                for _ in range(2):
                    out.write(
                        json.dumps(
                            {
                                "shape_id": pair["shape_id"],
                                "candidate_hash": pair["candidate_hash"],
                                "status": "ok",
                                "sample_index": 0,
                                "time_us": 1,
                                    "validation": "PASSED",
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
            p.add_argument("--pairs")
            p.add_argument("--output")
            p.add_argument("--library-dir")
            args = p.parse_args()

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
                            "solution_index": pair["library_solution_index"],
                        }
                    )
                    + "\\n"
                )
            sys.exit(3)
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
            import time

            time.sleep(10)
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
    assert "incomplete sample set" in executed.ingest.errors[0]
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
