from pathlib import Path

from evotensile.database import EvoTensileDB
from evotensile.protocol import DEFAULT_BENCHMARK_PROTOCOL
from evotensile.scheduler import execute_schedule
from evotensile.search.measured_cost import load_candidate_measured_costs
from evotensile.shapes import pilot_100_shapes
from tests.helpers import fake_build_tensile, fake_structured_runner, sample_candidates


def test_recorded_run_costs_cover_prepare_validation_and_screening(tmp_path: Path):
    fake_tensile = fake_build_tensile(tmp_path)
    fake_runner = fake_structured_runner(tmp_path)
    db = EvoTensileDB.connect(tmp_path / "costs.sqlite")
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    protocol = DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=2)

    execute_schedule(
        db,
        shapes=[shape],
        candidates=[candidate],
        output_root=tmp_path / "round",
        protocol=protocol,
        candidate_batch_size=1,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=fake_runner,
        cost_aware_scheduling=True,
    )
    costs = load_candidate_measured_costs(db)

    assert costs[candidate.hash].prepare_s > 0.0
    assert costs[candidate.hash].validation_s > 0.0
    assert costs[candidate.hash].screening_s > 0.0
    assert costs[candidate.hash].total_s >= costs[candidate.hash].prepare_s


def test_indexed_run_cost_divides_shared_duration_once(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "costs.sqlite")
    db.init()
    candidates = sample_candidates(2)
    db.register_candidates(candidates)

    db.insert_run(
        "shared",
        phase="prepare",
        status="ok",
        duration_s=6.0,
        candidate_hashes=[candidates[0].hash, candidates[1].hash, candidates[0].hash],
    )
    costs = load_candidate_measured_costs(db)

    assert costs[candidates[0].hash].prepare_s == 3.0
    assert costs[candidates[1].hash].prepare_s == 3.0

    db.insert_run(
        "shared",
        phase="prepare",
        status="ok",
        duration_s=2.0,
        candidate_hashes=[candidates[1].hash],
    )
    replaced_costs = load_candidate_measured_costs(db)
    assert replaced_costs[candidates[0].hash].prepare_s == 0.0
    assert replaced_costs[candidates[1].hash].prepare_s == 2.0
