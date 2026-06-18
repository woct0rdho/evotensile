from evotensile.cache import CacheKey, benchmark_protocol_hash_from_items, normalize_version_name, problem_type_hash
from evotensile.database import EvoTensileDB
from evotensile.runner import serial_benchmark_global_parameters, serial_benchmark_protocol_hash
from evotensile.search_space import known_seed_candidates
from evotensile.shapes import pilot_100_shapes


def test_version_name_is_manual_namespace():
    assert normalize_version_name(None) == "unversioned"
    assert normalize_version_name("  local_patch_a ") == "local_patch_a"


def test_protocol_hash_changes_with_timing_params_not_cpu_threads():
    base = benchmark_protocol_hash_from_items([])
    changed = benchmark_protocol_hash_from_items(["NumWarmups=5"])
    cpu_only = benchmark_protocol_hash_from_items(["CpuThreads=16"])
    assert base != changed
    assert base == cpu_only


def test_serial_benchmark_globals_override_parallel_gpu_execution():
    params = serial_benchmark_global_parameters(["ParallelGpuExecution=0", "NumWarmups=5"])

    assert params == ["NumWarmups=5", "ParallelGpuExecution=1"]
    assert serial_benchmark_protocol_hash(["ParallelGpuExecution=0"]) == serial_benchmark_protocol_hash([])


def test_db_cache_key_lookup(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "cache.sqlite")
    db.init()
    candidate = known_seed_candidates()[0]
    shape = pilot_100_shapes()[0]
    p_hash = problem_type_hash()
    b_hash = benchmark_protocol_hash_from_items([])
    key = CacheKey(
        version_name="v0",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        shape_id=shape.id,
        candidate_hash=candidate.hash,
    )

    assert not db.has_cached_evaluation(key)
    db.insert_evaluation(
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="run_test",
        status="ok",
        version_name="v0",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=123.0,
    )
    assert db.has_cached_evaluation(key)
    assert db.has_reusable_cache_entry(key)
    assert db.cache_summary(version_name="v0") == {"ok": 1}


def test_negative_cache_statuses_are_reusable_but_not_rankable(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "cache.sqlite")
    db.init()
    candidate = known_seed_candidates()[0]
    shape = pilot_100_shapes()[0]
    p_hash = problem_type_hash()
    b_hash = benchmark_protocol_hash_from_items([])
    key = CacheKey(
        version_name="v0",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        shape_id=shape.id,
        candidate_hash=candidate.hash,
    )

    db.insert_evaluation(
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="run_test",
        status="rejected",
        version_name="v0",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )

    assert not db.has_cached_evaluation(key)
    assert db.has_reusable_cache_entry(key)
    assert db.rank_evaluations(version_name="v0") == []
    assert db.cache_summary(version_name="v0") == {"rejected": 1}
