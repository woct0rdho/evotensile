from pathlib import Path

from evotensile.cache import benchmark_protocol_hash_from_items, normalize_version_name, problem_type_hash
from evotensile.database import EvoTensileDB
from evotensile.scheduler import execute_schedule, plan_batches
from evotensile.search_space import known_seed_candidates
from evotensile.shapes import pilot_100_shapes


def test_plan_batches_skips_cached_ok_pairs(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = known_seed_candidates()[:2]
    shapes = pilot_100_shapes()[:2]
    p_hash = problem_type_hash()
    b_hash = benchmark_protocol_hash_from_items([])
    db.insert_evaluation(
        shape_id=shapes[0].id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="ok",
        version_name="vtest",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )

    batches = plan_batches(
        db,
        shapes=shapes,
        candidates=candidates,
        version_name="vtest",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        candidate_batch_size=2,
        shape_batch_size=2,
    )

    assert len(batches) == 1
    assert batches[0].missing_pairs == 3
    assert batches[0].nominal_pairs == 4
    assert batches[0].extra_pairs == 1


def test_execute_schedule_generate_only_writes_batch_inputs(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidates = known_seed_candidates()[:2]
    shapes = pilot_100_shapes()[:1]
    p_hash = problem_type_hash()
    b_hash = benchmark_protocol_hash_from_items([])

    result = execute_schedule(
        db,
        shapes=shapes,
        candidates=candidates,
        output_root=tmp_path / "batches",
        version_name=normalize_version_name("vtest"),
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        candidate_batch_size=1,
        shape_batch_size=1,
        generate_only=True,
    )

    assert len(result.planned_batches) == 2
    assert len(result.executed_batches) == 2
    for executed in result.executed_batches:
        assert executed.yaml_path.exists()
        assert executed.manifest_path.exists()
        assert executed.output_dir.name == "run"
