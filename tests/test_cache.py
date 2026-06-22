from evotensile.cache import CacheKey, benchmark_protocol_hash, problem_type_hash
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.protocol import DEFAULT_BENCHMARK_PROTOCOL, global_parameter_items
from evotensile.search_space import known_seed_candidates
from evotensile.shapes import pilot_100_shapes


def test_default_protocol_uses_full_validation():
    assert DEFAULT_BENCHMARK_PROTOCOL.num_elements_to_validate == -1
    assert DEFAULT_PROFILE.benchmark_protocol_hash() == DEFAULT_BENCHMARK_PROTOCOL.protocol_hash()


def test_string_global_parameters_are_quoted_for_tensilelite_cli():
    assert global_parameter_items({"RuntimeLanguage": "HIP", "MinimumRequiredVersion": "5.0.0"}) == [
        "RuntimeLanguage='HIP'",
        "MinimumRequiredVersion='5.0.0'",
    ]


def test_protocol_hash_ignores_sample_budget_but_tracks_timing_protocol():
    base = benchmark_protocol_hash(DEFAULT_BENCHMARK_PROTOCOL)
    more_samples = benchmark_protocol_hash(DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=120))
    changed_warmups = benchmark_protocol_hash(DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_warmups=5))
    assert base == more_samples
    assert base != changed_warmups


def test_profile_derives_cache_identity():
    assert problem_type_hash() == DEFAULT_PROFILE.problem_type_hash
    assert benchmark_protocol_hash(DEFAULT_PROFILE.default_protocol) == DEFAULT_PROFILE.benchmark_protocol_hash()


def test_db_cache_key_lookup(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "cache.sqlite")
    db.init()
    candidate = known_seed_candidates()[0]
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    key = CacheKey(
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
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=123.0,
    )
    assert db.has_cached_evaluation(key)
    assert db.has_reusable_cache_entry(key)
    assert db.cache_summary() == {"ok": 1}


def test_negative_cache_statuses_are_reusable_but_not_rankable(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "cache.sqlite")
    db.init()
    candidate = known_seed_candidates()[0]
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    key = CacheKey(
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
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )

    assert not db.has_cached_evaluation(key)
    assert db.has_reusable_cache_entry(key)
    assert db.rank_evaluations() == []
    assert db.cache_summary() == {"rejected": 1}
