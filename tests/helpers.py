from pathlib import Path
from textwrap import dedent

from evotensile.candidate import Candidate
from evotensile.database import BenchmarkEventInsert, EvoTensileDB, ValidationInsert
from evotensile.profile import DEFAULT_PROFILE
from evotensile.search_space import make_candidate, random_candidates, repair_linked_overrides


def sample_candidates(count: int, *, seed: int = 1151) -> list[Candidate]:
    return random_candidates(count, seed=seed)


def sample_candidate(*, seed: int = 1151) -> Candidate:
    return sample_candidates(1, seed=seed)[0]


REFERENCE_CANDIDATE = make_candidate(repair_linked_overrides({}), source="reference")


def insert_test_benchmark_event(
    db: EvoTensileDB,
    *,
    shape_id: str,
    candidate_hash: str,
    run_id: str | None,
    status: str,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    source_kind: str = "replay",
    time_us: float | None = None,
    samples_us: tuple[float, ...] = (),
    solution_index: int | None = None,
) -> None:
    validation_protocol_hash = None
    if status == "ok":
        validation_protocol_hash = DEFAULT_PROFILE.default_protocol.validation_protocol_hash()
        db.insert_validations(
            [
                ValidationInsert(
                    shape_id=shape_id,
                    candidate_hash=candidate_hash,
                    run_id=f"{run_id}:validation",
                    status="passed",
                    problem_type_hash=problem_type_hash,
                    validation_protocol_hash=validation_protocol_hash,
                    source_kind=source_kind,
                )
            ]
        )
    if time_us is not None:
        if samples_us:
            raise ValueError("provide either time_us or samples_us, not both")
        samples_us = (time_us,)
    db.insert_benchmark_events(
        [
            BenchmarkEventInsert(
                shape_id=shape_id,
                candidate_hash=candidate_hash,
                run_id=run_id,
                status=status,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                source_kind=source_kind,
                samples_us=samples_us,
                validation_protocol_hash=validation_protocol_hash,
                solution_index=solution_index,
            )
        ]
    )


def fake_structured_runner(path: Path) -> Path:
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
            active_dir = (
                os.environ.get("EVOTENSILE_TEST_GPU_ACTIVE_DIR")
                if args.mode == "benchmark"
                else os.environ.get("EVOTENSILE_TEST_VALIDATION_ACTIVE_DIR")
            )
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


def fake_build_tensile(path: Path) -> Path:
    script = path / "fake_tensile_build.py"
    script.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import os
            import sys
            import time
            from pathlib import Path

            import yaml

            events_path = os.environ.get("EVOTENSILE_TEST_PHASE_EVENTS")
            if events_path:
                with Path(events_path).open("a") as events:
                    events.write("compile_start\\n")
            config_path, out = Path(sys.argv[1]), Path(sys.argv[2])
            out.mkdir(parents=True, exist_ok=True)
            time.sleep(float(os.environ.get("EVOTENSILE_TEST_BUILD_SLEEP_S", "0")))
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
