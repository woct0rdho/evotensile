import json
from pathlib import Path

from evotensile.cli import main as cli_main
from evotensile.rejection_mining import classification_counts, classify_tensilelite_log, summarize_rejection_logs


def test_classify_schema_error():
    summary = classify_tensilelite_log(
        "Traceback (most recent call last):\nConfigTypeError: config.yaml: SourceSwap = 1 (int); expected bool\n"
    )

    assert summary.classification == "schema"
    assert "expected bool" in summary.messages[0]


def test_classify_solutionstructs_zero():
    summary = classify_tensilelite_log(
        "# Actual Solutions: 0 / 1 after SolutionStructs\nTensile::FATAL: Your parameters resulted in 0 valid solutions.\n"
    )

    assert summary.classification == "solutionstructs_zero"
    assert summary.actual_solutions == 0
    assert summary.total_solutions == 1
    assert summary.solution_stage == "SolutionStructs"


def test_summarize_rejection_logs_scans_directories(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "a.stdout.log").write_text("# Actual Solutions: 1 / 2 after SolutionStructs\n", encoding="utf-8")
    (run_dir / "b.stderr.log").write_text("total vgpr: 455 not in [0, 256]\n", encoding="utf-8")

    summaries = summarize_rejection_logs([run_dir])

    assert classification_counts(summaries) == {"solutionstructs_partial": 1, "kernelwriter_resource": 1}


def test_summarize_rejections_cli_json(tmp_path: Path, capsys):
    log = tmp_path / "run.stdout.log"
    log.write_text("# Actual Solutions: 0 / 1 after SolutionStructs\n", encoding="utf-8")

    rc = cli_main(["summarize-rejections", str(log), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"] == {"solutionstructs_zero": 1}
    assert payload["logs"][0]["actual_solutions"] == 0
