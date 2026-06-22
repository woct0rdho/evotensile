import json
from pathlib import Path

from evotensile.cli import main as cli_main
from evotensile.search.coverage import candidate_coverage
from evotensile.search.random_search import initial_random_batch


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
