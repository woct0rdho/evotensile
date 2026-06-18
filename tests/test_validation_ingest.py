from pathlib import Path

from evotensile.cache import benchmark_protocol_hash_from_items, problem_type_hash
from evotensile.database import EvoTensileDB
from evotensile.manifest import manifest_by_problem_solution, read_manifest, write_manifest
from evotensile.parser import evaluation_status, parse_tensilelite_csv
from evotensile.search_space import known_seed_candidates
from evotensile.shapes import pilot_100_shapes


def test_parse_csvwinner_validation_pass(tmp_path: Path):
    path = tmp_path / "winner.csv"
    path.write_text(
        "GFlops, SizeI, SizeJ, SizeK, SizeL, TotalFlops, WinnerGFlops, WinnerTimeUS, WinnerIdx, WinnerName, KernelA\n"
        "0, 8192, 8192, 1, 8192, 1099511627776, 46698.1, 23545.0, 0, KernelA, 46698.1\n",
        encoding="utf-8",
    )
    rows = parse_tensilelite_csv(path)
    assert len(rows) == 1
    row = rows[0]
    assert row.shape_id == "m8192_n8192_b1_k8192"
    assert row.solution_index == 0
    assert row.gflops == 46698.1
    assert row.time_us == 23545.0
    assert evaluation_status(row, require_validation=False) == "ok"
    assert evaluation_status(row, require_validation=True) == "validation_unknown"


def test_parse_stdout_log_validation_fail(tmp_path: Path):
    path = tmp_path / "client.log"
    path.write_text(
        "noise\n"
        "run,problem-progress,solution-progress,operation,problem-sizes,solution,validation,time-us,gflops\n"
        '0,0/0,1/3,Contraction,"(512,256,1,1024)",KernelB,FAILED,10.5,1234.5\n',
        encoding="utf-8",
    )
    rows = parse_tensilelite_csv(path)
    assert len(rows) == 1
    row = rows[0]
    assert row.shape_id == "m512_n256_b1_k1024"
    assert row.solution_index == 1
    assert row.validation == "FAILED"
    assert evaluation_status(row) == "validation_fail"


def test_manifest_and_validation_gated_db(tmp_path: Path):
    candidates = known_seed_candidates()[:2]
    shapes = pilot_100_shapes()[:1]
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, candidates, shapes)
    entries = read_manifest(manifest)
    by_key = manifest_by_problem_solution(entries)
    assert by_key[(0, 0)].candidate_hash == candidates[0].hash
    assert by_key[(0, 1)].candidate_hash == candidates[1].hash

    csv_path = tmp_path / "log.csv"
    csv_path.write_text(
        "run,problem-progress,solution-progress,operation,problem-sizes,solution,validation,time-us,gflops\n"
        '0,0/0,0/1,Contraction,"(512,128,1,256)",Kernel0,PASSED,10.0,1000.0\n'
        '1,0/0,0/1,Contraction,"(512,128,1,256)",Kernel0,PASSED,9.5,1050.0\n'
        '0,0/0,1/1,Contraction,"(512,128,1,256)",Kernel1,FAILED,11.0,900.0\n',
        encoding="utf-8",
    )
    rows = parse_tensilelite_csv(csv_path)
    db = EvoTensileDB.connect(tmp_path / "evals.sqlite")
    db.init()
    p_hash = problem_type_hash()
    b_hash = benchmark_protocol_hash_from_items([])
    for row in rows:
        entry = by_key[(row.problem_index or 0, row.solution_index or 0)]
        db.insert_evaluation(
            shape_id=entry.shape_id,
            candidate_hash=entry.candidate_hash,
            run_id="run_validation_test",
            status=evaluation_status(row),
            version_name="vtest",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=row.time_us,
            gflops=row.gflops,
            validation=row.validation,
            solution_index=row.solution_index,
            raw_csv_row=str(row.raw),
        )

    assert db.cache_summary(version_name="vtest") == {"ok": 2, "validation_fail": 1}
    assert (
        db.cached_evaluation_count(
            version_name="vtest",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            shape_id=shapes[0].id,
            candidate_hash=candidates[0].hash,
        )
        == 2
    )
    assert (
        db.cached_evaluation_count(
            version_name="vtest",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            shape_id=shapes[0].id,
            candidate_hash=candidates[1].hash,
        )
        == 0
    )
    ranked = db.rank_evaluations(
        version_name="vtest",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        shape_id=shapes[0].id,
        min_samples=2,
    )
    assert len(ranked) == 1
    assert ranked[0].candidate_hash == candidates[0].hash
    assert ranked[0].samples == 2
    assert ranked[0].median_gflops == 1025.0
