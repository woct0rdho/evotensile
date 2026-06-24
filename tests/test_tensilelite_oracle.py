import json
from textwrap import dedent

from evotensile.candidate import Candidate
from evotensile.database import EvoTensileDB
from scripts.tensilelite_oracle import _candidate_params_from_db, _candidate_params_from_file, _interesting_log_lines
from tests.helpers import DOCUMENTED_WINNER_CANDIDATE


def test_oracle_loads_candidate_params_from_nested_json(tmp_path):
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps({"params": DOCUMENTED_WINNER_CANDIDATE.canonical_params()}), encoding="utf-8")

    params = _candidate_params_from_file(path)

    assert params["MatrixInstruction"] == DOCUMENTED_WINNER_CANDIDATE.canonical_params()["MatrixInstruction"]
    assert params["KernelLanguage"] == "Assembly"


def test_oracle_loads_candidate_params_from_db(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "search.sqlite")
    db.init()
    candidate = Candidate(DOCUMENTED_WINNER_CANDIDATE.canonical_params(), source="oracle_test")
    db.upsert_candidate(candidate)

    params = _candidate_params_from_db(tmp_path / "search.sqlite", candidate.hash)

    assert params["DepthU"] == DOCUMENTED_WINNER_CANDIDATE.canonical_params()["DepthU"]
    assert params["WavefrontSize"] == 32


def test_oracle_extracts_interesting_log_lines(tmp_path):
    path = tmp_path / "run.stdout.log"
    path.write_text(
        dedent(
            """
            ordinary progress line
            Tensile::WARNING: Failed to generate assembly source code for kernel
            reject: totalVectorsB 576 % NumThreads 128 != 0
            RuntimeError: No valid solutions found
            """
        ),
        encoding="utf-8",
    )

    lines = _interesting_log_lines([path])

    assert any("Failed to generate assembly" in line for line in lines)
    assert any("totalVectorsB" in line for line in lines)
    assert any("No valid solutions" in line for line in lines)
