import json
from pathlib import Path

from evotensile.cli import main as cli_main
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.search.coverage import candidate_coverage
from evotensile.search.random_search import initial_random_batch
from evotensile.shapes import pilot_100_shapes
from tests.helpers import insert_test_benchmark_event


def test_candidate_coverage_counts_unique_values():
    candidates = initial_random_batch(16, seed=1151)

    summary = candidate_coverage(candidates)

    assert summary["candidates"] == 16
    assert summary["unique_candidate_hashes"] == 16
    assert summary["invalid_reason_counts"] == {}
    assert summary["unique_values"]["MatrixInstruction"] > 1


def test_proposal_coverage_cli(tmp_path: Path, capsys):
    rc = cli_main(
        [
            "proposal-coverage",
            "--db",
            str(tmp_path / "coverage.sqlite"),
            "--num-random",
            "8",
            "--gomea-count",
            "0",
            "--de-count",
            "0",
            "--local-count",
            "0",
            "--limit-shapes",
            "1",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidates"] == 8
    assert payload["proposal"] == "seed-random-gomea"
    assert payload["candidate_family_count"] >= 1


def test_summarize_families_cli(tmp_path: Path, capsys):
    db_path = tmp_path / "families.sqlite"
    db = EvoTensileDB.connect(
        db_path,
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    db.init()
    candidates = initial_random_batch(2, seed=1151)
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([shape])
    for idx, candidate in enumerate(candidates):
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=1.0 + idx,
        )

    rc = cli_main(["summarize-families", "--db", str(db_path), "--limit-shapes", "1"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"] == DEFAULT_PROFILE.name
    assert payload["families"] >= 1
    assert payload["entries"][0]["leader_candidate_hash"] == candidates[0].hash
