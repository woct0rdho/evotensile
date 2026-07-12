from scripts.finalize_grid100_production_search import TimingSummary, _select_contenders


def _summary(candidate_hash, performance):
    return TimingSummary(
        shape_id="m512_n128_b1_k256",
        candidate_hash=candidate_hash,
        samples=10,
        median_time_us=1.0 / performance,
        median_gflops=performance,
        relative_mad=0.0,
    )


def test_select_contenders_keeps_close_candidates_and_mandatory_baseline():
    shape_id = "m512_n128_b1_k256"
    rankings = {
        shape_id: (
            _summary("cand_new", 100.0),
            _summary("cand_close", 99.0),
            _summary("cand_far", 90.0),
            _summary("cand_baseline", 80.0),
        )
    }

    selected = _select_contenders(
        rankings,
        maximum_contenders=3,
        relative_tolerance=0.02,
        mandatory_candidate_by_shape={shape_id: "cand_baseline"},
    )

    assert selected[shape_id] == ("cand_new", "cand_close", "cand_baseline")


def test_select_contenders_does_not_duplicate_mandatory_candidate():
    shape_id = "m512_n128_b1_k256"
    rankings = {
        shape_id: (
            _summary("cand_new", 100.0),
            _summary("cand_baseline", 99.0),
        )
    }

    selected = _select_contenders(
        rankings,
        maximum_contenders=3,
        relative_tolerance=0.02,
        mandatory_candidate_by_shape={shape_id: "cand_baseline"},
    )

    assert selected[shape_id] == ("cand_new", "cand_baseline")
