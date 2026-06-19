import json
from pathlib import Path

from evotensile.cache import normalize_version_name
from evotensile.cli import build_parser
from evotensile.cli import main as cli_main
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.scheduler import (
    DEFAULT_GOMEA_COUNT,
    DEFAULT_NUM_RANDOM,
    DEFAULT_PROPOSAL,
    execute_schedule,
    plan_batches,
    propose_candidates,
)
from evotensile.search_space import documented_winner_candidate, known_seed_candidates
from evotensile.shapes import pilot_100_shapes


def test_plan_batches_skips_cached_ok_pairs(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = known_seed_candidates()[:2]
    shapes = pilot_100_shapes()[:2]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
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

    assert len(batches) == 2
    assert sum(batch.missing_pairs for batch in batches) == 3
    assert sum(batch.nominal_pairs for batch in batches) == 3
    assert all(batch.extra_pairs == 0 for batch in batches)
    assert [candidate.hash for candidate in batches[0].candidates] == [candidates[1].hash]
    assert [shape.id for shape in batches[0].shapes] == [shapes[0].id]
    assert [candidate.hash for candidate in batches[1].candidates] == [candidate.hash for candidate in candidates]
    assert [shape.id for shape in batches[1].shapes] == [shapes[1].id]


def test_plan_batches_skips_reusable_negative_cache_entries(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = known_seed_candidates()[:2]
    shapes = pilot_100_shapes()[:1]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.insert_evaluation(
        shape_id=shapes[0].id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="rejected",
        version_name="vtest",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )
    db.insert_evaluation(
        shape_id=shapes[0].id,
        candidate_hash=candidates[1].hash,
        run_id="cached",
        status="build_failed",
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
        shape_batch_size=1,
    )

    assert batches == []


def test_local_proposal_mutates_cached_elites(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = known_seed_candidates()[:2]
    shapes = pilot_100_shapes()[:1]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.insert_evaluation(
        shape_id=shapes[0].id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="ok",
        version_name="vtest",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        gflops=10.0,
    )

    proposed = propose_candidates(
        db,
        proposal="local",
        version_name="vtest",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        local_count=1,
        elite_count=1,
        seed=7,
    )

    assert len(proposed) == 1
    assert proposed[0].source == "mutation"
    assert proposed[0].parent_hashes == (candidates[0].hash,)


def test_nearest_shape_transfer_seeds_cached_winners(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = known_seed_candidates()[:3]
    target = pilot_100_shapes()[0]
    near_shape = pilot_100_shapes()[1]
    far_shape = pilot_100_shapes()[-1]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    for shape, candidate, gflops in ((near_shape, candidates[1], 200.0), (far_shape, candidates[2], 300.0)):
        db.insert_evaluation(
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            version_name="vtest",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            gflops=gflops,
        )

    proposed = propose_candidates(
        db,
        proposal="seed-random-gomea",
        num_random=0,
        gomea_count=0,
        target_shapes=[target],
        transfer_shape_count=1,
        transfer_per_shape=1,
        version_name="vtest",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )

    transfer = [candidate for candidate in proposed if candidate.source == "transfer"]
    assert [candidate.hash for candidate in transfer] == [candidates[1].hash]
    assert transfer[0].parent_hashes == (candidates[1].hash,)


def test_exact_shape_elites_disable_nearest_shape_transfer(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = known_seed_candidates()[:2]
    target = pilot_100_shapes()[0]
    exact_shape = pilot_100_shapes()[1]
    transfer_shape = pilot_100_shapes()[2]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.insert_evaluation(
        shape_id=exact_shape.id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="ok",
        version_name="vtest",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        gflops=100.0,
    )
    db.insert_evaluation(
        shape_id=transfer_shape.id,
        candidate_hash=candidates[1].hash,
        run_id="cached",
        status="ok",
        version_name="vtest",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        gflops=200.0,
    )

    proposed = propose_candidates(
        db,
        proposal="gomea",
        gomea_count=4,
        target_shapes=[target],
        shape_id=exact_shape.id,
        transfer_shape_count=4,
        transfer_per_shape=1,
        version_name="vtest",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )

    assert all(candidate.source != "transfer" for candidate in proposed)
    assert {candidates[0].hash} & {parent for candidate in proposed for parent in candidate.parent_hashes}
    assert candidates[1].hash not in {parent for candidate in proposed for parent in candidate.parent_hashes}


def test_evolutionary_proposal_uses_cached_elites(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = known_seed_candidates()[:3]
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    for idx, candidate in enumerate(candidates):
        db.insert_evaluation(
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            version_name="vtest",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            gflops=100.0 - idx,
        )

    proposed = propose_candidates(
        db,
        proposal="evolutionary",
        num_random=2,
        local_count=2,
        de_count=2,
        gomea_count=2,
        elite_count=3,
        version_name="vtest",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        seed=7,
    )

    sources = {candidate.source for candidate in proposed}
    assert {"seed", "random", "mutation", "de", "gomea"} & sources == {
        "seed",
        "random",
        "mutation",
        "de",
        "gomea",
    }


def test_schedule_cli_uses_grid100_evolutionary_defaults():
    args = build_parser().parse_args(["schedule-batches", "--db", "db.sqlite", "--output-dir", "out"])

    assert args.proposal == DEFAULT_PROPOSAL == "seed-random-gomea"
    assert args.num_random == DEFAULT_NUM_RANDOM == 64
    assert args.gomea_count == DEFAULT_GOMEA_COUNT == 64


def test_seed_random_gomea_reproduces_documented_winner_without_hindsight(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()

    proposed = propose_candidates(
        db,
        proposal="seed-random-gomea",
        num_random=12,
        gomea_count=64,
        seed=1151,
    )

    hashes = [candidate.hash for candidate in proposed]
    winner_index = hashes.index(documented_winner_candidate().hash)
    assert winner_index + 1 <= 32
    assert proposed[winner_index].source == "gomea"


def test_execute_schedule_records_single_candidate_build_timeout(tmp_path: Path):
    fake_tensile = tmp_path / "slow_tensile.py"
    fake_tensile.write_text(
        "#!/usr/bin/env python3\nimport time\ntime.sleep(10)\n",
        encoding="utf-8",
    )
    fake_tensile.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidate = known_seed_candidates()[0]
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()

    result = execute_schedule(
        db,
        shapes=[shape],
        candidates=[candidate],
        output_root=tmp_path / "batches",
        version_name="vtimeout",
        candidate_batch_size=1,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=tmp_path / "unused_runner",
        build_timeout_s=0.1,
    )

    assert len(result.executed_batches) == 1
    assert result.executed_batches[0].build_returncode == 124
    assert db.cache_summary(version_name="vtimeout") == {"build_timeout": 1}
    assert (
        len(
            plan_batches(
                db,
                shapes=[shape],
                candidates=[candidate],
                version_name="vtimeout",
                problem_type_hash=p_hash,
                benchmark_protocol_hash=b_hash,
                candidate_batch_size=1,
                shape_batch_size=1,
            )
        )
        == 1
    )


def test_execute_schedule_records_single_candidate_build_failure(tmp_path: Path):
    fake_tensile = tmp_path / "fake_tensile.py"
    fake_tensile.write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(2)\n", encoding="utf-8")
    fake_tensile.chmod(0o755)
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidate = known_seed_candidates()[0]
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()

    result = execute_schedule(
        db,
        shapes=[shape],
        candidates=[candidate],
        output_root=tmp_path / "batches",
        version_name="vtest",
        candidate_batch_size=1,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=tmp_path / "unused_runner",
    )

    assert len(result.executed_batches) == 1
    assert db.cache_summary(version_name="vtest") == {"build_failed": 1}
    assert (
        plan_batches(
            db,
            shapes=[shape],
            candidates=[candidate],
            version_name="vtest",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            candidate_batch_size=1,
            shape_batch_size=1,
        )
        == []
    )


def test_schedule_cli_writes_machine_readable_metadata(tmp_path: Path):
    db_path = tmp_path / "sched.sqlite"
    output_dir = tmp_path / "out"

    rc = cli_main(
        [
            "schedule-batches",
            "--db",
            str(db_path),
            "--output-dir",
            str(output_dir),
            "--version-name",
            "vmetadata",
            "--num-random",
            "1",
            "--limit-shapes",
            "1",
            "--candidate-batch-size",
            "1",
            "--shape-batch-size",
            "1",
            "--dry-run",
        ]
    )

    assert rc == 0
    metadata = json.loads((output_dir / "schedule_metadata.json").read_text(encoding="utf-8"))
    assert metadata["profile"] == DEFAULT_PROFILE.name
    assert metadata["version_name"] == "vmetadata"
    assert metadata["planned_batches"] >= 1
    assert metadata["executed_batches"] == []
    assert metadata["runner_bin"] == DEFAULT_PROFILE.default_runner_bin


def test_execute_schedule_generate_only_writes_batch_inputs(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    candidates = known_seed_candidates()[:2]
    shapes = pilot_100_shapes()[:1]

    result = execute_schedule(
        db,
        shapes=shapes,
        candidates=candidates,
        output_root=tmp_path / "batches",
        version_name=normalize_version_name("vtest"),
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
