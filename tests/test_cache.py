from textwrap import dedent

from evotensile.cache import CacheKey
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.protocol import DEFAULT_BENCHMARK_PROTOCOL, global_parameter_items
from evotensile.runner import run_tensilelite
from evotensile.shapes import pilot_100_shapes
from tests.helpers import sample_candidates


def test_default_protocol_uses_full_hipblaslt_validation():
    assert DEFAULT_BENCHMARK_PROTOCOL.num_elements_to_validate == -1
    assert DEFAULT_BENCHMARK_PROTOCOL.validation_backend == "hipblaslt"
    assert DEFAULT_PROFILE.benchmark_protocol_hash() == DEFAULT_BENCHMARK_PROTOCOL.protocol_hash()


def test_string_global_parameters_are_quoted_for_tensilelite_cli():
    assert global_parameter_items({"RuntimeLanguage": "HIP", "MinimumRequiredVersion": "5.0.0"}) == [
        "RuntimeLanguage='HIP'",
        "MinimumRequiredVersion='5.0.0'",
    ]


def test_protocol_hash_ignores_sampling_budget_and_validation_execution():
    base = DEFAULT_BENCHMARK_PROTOCOL.protocol_hash()
    more_samples = DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=120).protocol_hash()
    gpu_only_topup = DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_elements_to_validate=0).protocol_hash()
    cpu_validation = DEFAULT_BENCHMARK_PROTOCOL.with_overrides(validation_backend="cpu").protocol_hash()
    changed_warmups = DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_warmups=5).protocol_hash()
    probe = DEFAULT_BENCHMARK_PROTOCOL.with_overrides(role="probe").protocol_hash()
    assert base == more_samples
    assert base == gpu_only_topup
    assert base == cpu_validation
    assert base != changed_warmups
    assert base != probe


def test_validation_protocol_hash_tracks_correctness_compatibility():
    base = DEFAULT_BENCHMARK_PROTOCOL.validation_protocol_hash()
    assert base == DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=120).validation_protocol_hash()
    assert base == DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_warmups=1).validation_protocol_hash()
    assert base == DEFAULT_BENCHMARK_PROTOCOL.with_overrides(role="probe").validation_protocol_hash()
    assert base != DEFAULT_BENCHMARK_PROTOCOL.with_overrides(validation_backend="cpu").validation_protocol_hash()
    assert base != DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_elements_to_validate=128).validation_protocol_hash()
    assert base != DEFAULT_BENCHMARK_PROTOCOL.with_overrides(data_init_type_a=2).validation_protocol_hash()


def test_db_cache_key_lookup(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "cache.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
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
        validation="PASSED prior_validation",
    )
    assert db.has_cached_evaluation(key)
    assert db.has_reusable_cache_entry(key)
    assert db.cache_summary() == {"ok": 1}


def test_positive_benchmark_evidence_supersedes_reusable_negatives(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "cache.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    key = CacheKey(p_hash, b_hash, shape.id, candidate.hash)
    db.register_candidates([candidate])
    db.register_shapes([shape])

    db.insert_evaluation(
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="rejected",
        status="rejected",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )
    db.insert_evaluation(
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="timed",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=123.0,
        validation="PASSED prior_validation",
    )
    db.insert_evaluation(
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="later_build",
        status="build_failed",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )

    state = db.benchmark_evidence_states(
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        shape_ids=[shape.id],
        candidate_hashes=[candidate.hash],
    )[(shape.id, candidate.hash)]
    assert state.ok_samples == 1
    assert state.resolved_status == "ok"
    assert not state.reusable_negative
    assert db.has_reusable_cache_entry(key)
    assert not db.has_reusable_cache_entry(key, min_ok_samples=2)
    assert len(db.rank_evaluations()) == 1


def test_latest_reusable_negative_resolves_negative_only_state(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "cache.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    for run_id, status in (("first", "rejected"), ("second", "build_failed")):
        db.insert_evaluation(
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id=run_id,
            status=status,
            problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
            benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        )

    state = db.benchmark_evidence_states(
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        shape_ids=[shape.id],
        candidate_hashes=[candidate.hash],
    )[(shape.id, candidate.hash)]
    assert state.ok_samples == 0
    assert state.resolved_status == "build_failed"
    assert state.reusable_negative
    assert state.latest_negative_eval_id is not None


def test_reusable_negative_insertion_is_idempotent(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "cache.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    for _ in range(2):
        db.insert_evaluation(
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="same_run",
            status="rejected",
            problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
            benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        )

    assert db.cache_summary() == {"rejected": 1}


def test_run_tensilelite_use_cache_emits_cli_flag(tmp_path):
    fake_tensile = tmp_path / "fake_tensile.py"
    fake_tensile.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import json
            import sys
            from pathlib import Path

            out = Path(sys.argv[2])
            out.mkdir(parents=True, exist_ok=True)
            (out / "argv.json").write_text(json.dumps(sys.argv[1:]))
            """
        ),
        encoding="utf-8",
    )
    fake_tensile.chmod(0o755)
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("{}\n", encoding="utf-8")

    result = run_tensilelite(yaml_path, tmp_path / "out", tensilelite_bin=fake_tensile, build_only=True, use_cache=True)

    assert result.ok
    assert result.command[:4] == [str(fake_tensile), str(yaml_path), str(tmp_path / "out"), "--use-cache"]
    assert "--build-only" in result.command


def test_negative_cache_statuses_are_reusable_but_not_rankable(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "cache.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
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
