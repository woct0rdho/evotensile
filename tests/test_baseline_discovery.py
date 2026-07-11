from evotensile.database import BaselineSelectionInsert, EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.shapes import pilot_100_shapes
from tests.helpers import sample_candidates


def test_baseline_discovery_is_zero_evidence_planning_data(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "baseline.sqlite")
    db.init()
    shape = pilot_100_shapes()[0]
    candidate = sample_candidates(1)[0]

    discovery_id = db.record_baseline_discovery(
        [
            BaselineSelectionInsert(
                shape=shape,
                candidate=candidate,
                hipblaslt_solution_index=17,
                hipblaslt_solution_name="installed-solution",
                logic_solution_index=3,
                query_gflops=12_345.0,
                query_time_us=4.5,
            )
        ],
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        context={"logic": "installed"},
        duration_s=0.25,
    )

    assert discovery_id is not None
    pairs = db.baseline_selection_pairs(discovery_id)
    assert pairs[0][0] == shape
    assert pairs[0][1].hash == candidate.hash
    counts = db.counts()
    assert counts["evidence_sources"] == 0
    assert counts["native_runs"] == 0
    assert counts["benchmark_events"] == 0
    assert counts["benchmark_samples"] == 0
    assert counts["validations"] == 0
    assert (
        db.rank_benchmarks(
            problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
            benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        )
        == []
    )
