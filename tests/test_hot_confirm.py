import json
import time
from pathlib import Path
from textwrap import dedent

import pytest

from evotensile.artifacts import register_artifact_bundle
from evotensile.campaign.protocols import CAMPAIGN_SCREENING_PROTOCOL
from evotensile.database import EvoTensileDB, ValidationInsert
from evotensile.profile import DEFAULT_PROFILE
from evotensile.protocol import DEFAULT_BENCHMARK_PROTOCOL
from evotensile.search.hot_confirm import hot_confirm_topk
from evotensile.structured_runner import RunnablePair
from tests.helpers import insert_test_benchmark_event, sample_candidates


def _prepare_hot_db(tmp_path: Path, *, candidate_count: int = 1):
    db_path = tmp_path / "hot.sqlite"
    db = EvoTensileDB.connect(db_path)
    db.init()
    candidates = sample_candidates(candidate_count)
    shape = DEFAULT_PROFILE.shapes()[0]
    db.register_candidates(candidates)
    db.register_shapes([shape])
    runnable_pairs = []
    validations = []
    for index, candidate in enumerate(candidates):
        for time_us in (10.0 + index, 10.1 + index):
            insert_test_benchmark_event(
                db,
                shape_id=shape.id,
                candidate_hash=candidate.hash,
                run_id="screening",
                status="ok",
                problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
                benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(CAMPAIGN_SCREENING_PROTOCOL),
                time_us=time_us,
                solution_index=index,
            )
        validations.append(
            ValidationInsert(
                shape_id=shape.id,
                candidate_hash=candidate.hash,
                run_id="validation",
                status="passed",
                problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
                validation_protocol_hash=DEFAULT_PROFILE.default_protocol.validation_protocol_hash(),
                detail="PASSED",
                solution_index=index,
                source_kind="replay",
            )
        )
        runnable_pairs.append(
            RunnablePair(
                shape_id=shape.id,
                candidate_hash=candidate.hash,
                problem_index=index,
                requested_solution_index=index,
                library_solution_index=index,
                manifest_solution_index=index,
            )
        )
    db.insert_validations(validations)
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    solution_yaml = build_dir / "00_Final.yaml"
    solution_yaml.write_text("[]\n", encoding="utf-8")
    library_dir = build_dir / "library" / str(DEFAULT_PROFILE.library_logic["ArchitectureName"])
    library_dir.mkdir(parents=True)
    (library_dir / "TensileLibrary.yaml").write_text("solutions: []\n", encoding="utf-8")
    (library_dir / "Kernels.hsaco").write_bytes(b"fake code object")
    register_artifact_bundle(
        db,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        runnable_pairs=runnable_pairs,
        build_run_id="build",
        build_output_dir=build_dir,
        library_dir=library_dir,
        solution_yaml_paths=[solution_yaml],
        manifest_path=None,
    )
    return db_path, shape, candidates


def _runner(path: Path, behavior: str) -> Path:
    script = path / f"runner_{behavior}.py"
    script.write_text(
        dedent(
            f"""\
            #!/usr/bin/env python3
            import argparse
            import json
            import math
            import time

            parser = argparse.ArgumentParser()
            parser.add_argument("--mode")
            parser.add_argument("--pairs")
            parser.add_argument("--output")
            args, _ = parser.parse_known_args()
            behavior = {behavior!r}
            with open(args.pairs) as source:
                pair = json.loads(next(source))
            if behavior == "timeout-first" and pair["problem_index"] == 0:
                time.sleep(10)
            if behavior == "slow":
                time.sleep(0.1)
            with open(args.output, "w") as output:
                count = 1 if behavior == "missing" else 2
                for sample_index in range(count):
                    row = {{
                        "shape_id": pair["shape_id"],
                        "candidate_hash": pair["candidate_hash"],
                        "status": "ok",
                        "sample_index": sample_index,
                        "time_us": 1.0 + sample_index,
                        "validation": "NO_CHECK",
                        "solution_index": pair["library_solution_index"],
                    }}
                    if behavior == "wrong-pair":
                        row["candidate_hash"] = "cand_wrong"
                    elif behavior == "wrong-solution":
                        row["solution_index"] += 1
                    elif behavior == "duplicate":
                        row["sample_index"] = 0
                    elif behavior == "nan":
                        row["time_us"] = math.nan
                    elif behavior == "validated":
                        row["validation"] = "PASSED"
                    output.write(json.dumps(row) + "\\n")
            """
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _confirm(
    tmp_path: Path,
    runner: Path,
    *,
    candidate_count: int = 1,
    timeout: float = 5.0,
    admission_deadline: float | None = None,
):
    db_path, shape, _ = _prepare_hot_db(tmp_path, candidate_count=candidate_count)
    protocol = DEFAULT_BENCHMARK_PROTOCOL.with_overrides(
        num_warmups=0,
        num_benchmarks=2,
        enqueues_per_sync=1,
        num_elements_to_validate=0,
    )
    records = hot_confirm_topk(
        db_path=db_path,
        output_dir=tmp_path / "hot",
        runner_bin=runner,
        shape_id=shape.id,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        screening_protocol=CAMPAIGN_SCREENING_PROTOCOL,
        hot_protocol=protocol,
        top_k=candidate_count,
        admission_deadline=admission_deadline,
        runner_timeout_s=timeout,
    )
    summary = json.loads((tmp_path / "hot" / "summary.json").read_text(encoding="utf-8"))
    return records, summary


def test_hot_confirmation_uses_strict_structured_result_validation(tmp_path: Path):
    records, summary = _confirm(tmp_path, _runner(tmp_path, "ok"))

    assert len(records) == 1
    assert records[0]["samples"] == 2
    assert summary["failures"] == []


@pytest.mark.parametrize(
    ("behavior", "reason"),
    [
        ("wrong-pair", "unexpected pair"),
        ("wrong-solution", "wrong solution_index"),
        ("duplicate", "duplicate sample_index"),
        ("missing", "incomplete sample set"),
        ("nan", "invalid time"),
        ("validated", "performed validation"),
    ],
)
def test_hot_confirmation_rejects_malformed_results(tmp_path: Path, behavior: str, reason: str):
    records, summary = _confirm(tmp_path, _runner(tmp_path, behavior))

    assert records == []
    assert reason in summary["failures"][0]["reason"]


def test_hot_confirmation_soft_deadline_does_not_clamp_admitted_finalist(tmp_path: Path):
    records, summary = _confirm(
        tmp_path,
        _runner(tmp_path, "slow"),
        candidate_count=2,
        timeout=1.0,
        admission_deadline=time.monotonic() + 0.05,
    )

    assert len(records) == 1
    assert records[0]["screen_rank"] == 1
    assert summary["failures"] == []


def test_hot_confirmation_continues_after_timed_out_finalist(tmp_path: Path):
    records, summary = _confirm(
        tmp_path,
        _runner(tmp_path, "timeout-first"),
        candidate_count=2,
        timeout=0.05,
    )

    assert len(records) == 1
    assert records[0]["screen_rank"] == 2
    assert summary["failures"][0]["screen_rank"] == 1
    assert summary["failures"][0]["timed_out"] is True
