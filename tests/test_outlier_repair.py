import json
from pathlib import Path

from evotensile.candidate import Shape
from evotensile.cli import main as cli_main
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.search.outlier_repair import detect_underperforming_shapes, repair_seed_candidates
from tests.helpers import insert_test_benchmark_event, sample_candidates


def _time_us_for_gflops(shape: Shape, gflops: float) -> float:
    return 2.0 * shape.m * shape.n * shape.batch * shape.k / (gflops * 1e9) * 1e6


def test_detect_underperforming_shapes_flags_local_envelope_gap(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(4)
    shapes = [
        Shape(m=512, n=512, batch=1, k=512),
        Shape(m=640, n=512, batch=1, k=512),
        Shape(m=768, n=512, batch=1, k=512),
        Shape(m=896, n=512, batch=1, k=512),
        Shape(m=1024, n=512, batch=1, k=512),
    ]
    target = shapes[2]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes(shapes)
    for shape in shapes:
        candidate = candidates[0] if shape == target else candidates[1]
        gflops = 800.0 if shape == target else 1000.0
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=_time_us_for_gflops(shape, gflops),
        )

    outliers = detect_underperforming_shapes(
        db,
        shapes=shapes,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        min_samples=1,
        neighbor_count=4,
        envelope_quantile=0.75,
        threshold_pct=10.0,
    )

    assert [outlier.shape.id for outlier in outliers] == [target.id]
    assert outliers[0].candidate_hash == candidates[0].hash
    assert outliers[0].predicted_neighbor_gflops > 990.0
    assert outliers[0].residual_pct > 20.0


def test_repair_seed_candidates_include_neighbor_top_candidates(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(4)
    target = Shape(m=768, n=512, batch=1, k=512)
    neighbor = Shape(m=896, n=512, batch=1, k=512)
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([target, neighbor])
    insert_test_benchmark_event(
        db,
        shape_id=target.id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=_time_us_for_gflops(target, 800.0),
    )
    for candidate, gflops in ((candidates[1], 1000.0), (candidates[2], 950.0), (candidates[3], 900.0)):
        insert_test_benchmark_event(
            db,
            shape_id=neighbor.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=_time_us_for_gflops(neighbor, gflops),
        )
    outlier = detect_underperforming_shapes(
        db,
        shapes=[target],
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        min_samples=1,
        neighbor_count=1,
        envelope_quantile=0.75,
        threshold_pct=10.0,
    )[0]

    seeds = repair_seed_candidates(
        db,
        outliers=[outlier],
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        min_samples=1,
        neighbor_per_shape=2,
    )

    assert [seed.source for seed in seeds] == ["repair-transfer", "repair-transfer", "repair-transfer"]
    assert [seed.parent_hashes[0] for seed in seeds] == [
        candidates[0].hash,
        candidates[1].hash,
        candidates[2].hash,
    ]


def test_repair_outliers_cli_writes_metadata(tmp_path: Path):
    db_path = tmp_path / "sched.sqlite"
    output_dir = tmp_path / "repair"
    db = EvoTensileDB.connect(
        db_path,
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    db.init()
    candidates = sample_candidates(3)
    shapes = [
        Shape(m=512, n=512, batch=1, k=512),
        Shape(m=768, n=512, batch=1, k=512),
        Shape(m=1024, n=512, batch=1, k=512),
    ]
    target = shapes[1]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes(shapes)
    for shape in shapes:
        candidate = candidates[0] if shape == target else candidates[1]
        gflops = 700.0 if shape == target else 1000.0
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=_time_us_for_gflops(shape, gflops),
        )

    rc = cli_main(
        [
            "repair-outliers",
            "--db",
            str(db_path),
            "--output-dir",
            str(output_dir),
            "--shapes",
            target.id.replace("m", "", 1).replace("_n", ",").replace("_b", ",").replace("_k", ","),
            "--outlier-min-samples",
            "1",
            "--neighbor-count",
            "2",
            "--neighbor-per-shape",
            "1",
            "--num-random",
            "0",
            "--gomea-count",
            "0",
            "--local-count",
            "0",
            "--de-count",
            "0",
            "--generate-only",
        ]
    )

    assert rc == 0
    metadata = json.loads((output_dir / "repair_metadata.json").read_text(encoding="utf-8"))
    assert metadata["outliers"][0]["shape_id"] == target.id
    assert metadata["repair_seed_candidates"] == 2
    assert metadata["planned_missing_pairs"] >= 1
    assert metadata["executed_batches"]
    assert metadata["executed_batches"][0]["phase"] == "generated"
    assert metadata["executed_batches"][0]["ingest"] == {
        "errors": [],
        "inserted": 0,
        "rejected": 0,
        "status_counts": {},
        "unmapped": 0,
    }
    assert metadata["status_counts"] == {}
