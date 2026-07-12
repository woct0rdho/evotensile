from scripts.confirm_grid100_round_winners import RoundReportPayload, _screening_contenders


def test_screening_contenders_keeps_all_successful_candidates_above_threshold():
    report: RoundReportPayload = {
        "round_id": "round-test",
        "incumbent_improvements": [],
        "outcomes": [
            {
                "candidate_hash": "cand_best",
                "improvement_fraction": 0.08,
                "shape_id": "shape-a",
                "status": "ok",
            },
            {
                "candidate_hash": "cand_runner_up",
                "improvement_fraction": 0.03,
                "shape_id": "shape-a",
                "status": "ok",
            },
            {
                "candidate_hash": "cand_small",
                "improvement_fraction": 0.005,
                "shape_id": "shape-a",
                "status": "ok",
            },
            {
                "candidate_hash": "cand_failed",
                "improvement_fraction": 0.10,
                "shape_id": "shape-b",
                "status": "validation_failed",
            },
        ],
    }

    contenders = _screening_contenders(report, minimum_gain=0.01)

    assert contenders == {"shape-a": {"cand_best", "cand_runner_up"}}
