from textwrap import dedent

import pytest

from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.protocol import DEFAULT_BENCHMARK_PROTOCOL, global_parameter_items
from evotensile.runner import run_tensilelite
from evotensile.shapes import pilot_100_shapes
from tests.helpers import insert_test_benchmark_event, sample_candidates


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


def test_database_environment_compatibility_tag_guards_open(tmp_path, monkeypatch):
    monkeypatch.delenv("EVOTENSILE_ENVIRONMENT_COMPATIBILITY_TAG", raising=False)
    path = tmp_path / "cache.sqlite"

    with pytest.raises(ValueError, match="environment compatibility tag is required"):
        EvoTensileDB.connect(path)

    db = EvoTensileDB.connect(path, environment_compatibility_tag="test-a")
    db.init()
    reopened = EvoTensileDB.connect(path, environment_compatibility_tag="test-a")
    with reopened.connection() as con:
        assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    with pytest.raises(ValueError, match="environment compatibility tag mismatch"):
        EvoTensileDB.connect(path, environment_compatibility_tag="test-b")


def test_historical_provenance_is_read_only(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "cache.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    db.register_candidates([candidate])
    db.register_shapes([shape])

    with pytest.raises(ValueError, match="unsupported evidence source kind"):
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="retired_import",
            source_kind="historical_migration",
            status="rejected",
            problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
            benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        )


def test_db_cache_key_lookup(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "cache.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates([candidate])
    db.register_shapes([shape])

    assert (
        db.reusable_cache_entries(
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            shape_ids=[shape.id],
            candidate_hashes=[candidate.hash],
        )
        == set()
    )
    insert_test_benchmark_event(
        db,
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="run_test",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=123.0,
    )
    assert db.reusable_cache_entries(
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        shape_ids=[shape.id],
        candidate_hashes=[candidate.hash],
    ) == {(shape.id, candidate.hash)}
    assert db.benchmark_status_summary() == {"ok": 1}


def test_evidence_provenance_rejects_dangling_native_runs(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "cache.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    db.register_candidates([candidate])
    db.register_shapes([shape])

    with pytest.raises(ValueError, match="not registered"):
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="missing_native_run",
            source_kind="native_run",
            status="ok",
            problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
            benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
            time_us=123.0,
        )

    with db.connection() as con:
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []
        assert con.execute("SELECT COUNT(*) FROM benchmark_events").fetchone()[0] == 0


def test_benchmark_event_preserves_equal_ordered_samples(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "cache.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    db.register_candidates([candidate])
    db.register_shapes([shape])

    insert_test_benchmark_event(
        db,
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="repeated_samples",
        status="ok",
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        samples_us=(12.5, 12.5, 13.0),
    )

    with db.connection() as con:
        assert con.execute("SELECT COUNT(*) FROM benchmark_events").fetchone()[0] == 1
        samples = con.execute("SELECT sample_index, time_us FROM benchmark_samples ORDER BY sample_index").fetchall()
    assert [tuple(row) for row in samples] == [(0, 12.5), (1, 12.5), (2, 13.0)]


def test_positive_benchmark_evidence_supersedes_reusable_negatives(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "cache.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates([candidate])
    db.register_shapes([shape])

    insert_test_benchmark_event(
        db,
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="rejected",
        status="rejected",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )
    insert_test_benchmark_event(
        db,
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="timed",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=123.0,
    )
    insert_test_benchmark_event(
        db,
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
    assert db.reusable_cache_entries(
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        shape_ids=[shape.id],
        candidate_hashes=[candidate.hash],
    ) == {(shape.id, candidate.hash)}
    assert (
        db.reusable_cache_entries(
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            shape_ids=[shape.id],
            candidate_hashes=[candidate.hash],
            min_ok_samples=2,
        )
        == set()
    )
    assert len(db.rank_benchmarks()) == 1


def test_latest_reusable_negative_resolves_negative_only_state(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "cache.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    db.register_candidates([candidate])
    db.register_shapes([shape])
    for run_id, status in (("first", "rejected"), ("second", "build_failed")):
        insert_test_benchmark_event(
            db,
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
    assert state.latest_negative_event_id is not None


def test_reusable_negative_insertion_is_idempotent(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "cache.sqlite")
    db.init()
    candidate = sample_candidates(1)[0]
    shape = pilot_100_shapes()[0]
    db.register_candidates([candidate])
    db.register_shapes([shape])
    for _ in range(2):
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="same_run",
            status="rejected",
            problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
            benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        )

    assert db.benchmark_status_summary() == {"rejected": 1}


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
    db.register_candidates([candidate])
    db.register_shapes([shape])

    insert_test_benchmark_event(
        db,
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id="run_test",
        status="rejected",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )

    assert db.reusable_cache_entries(
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        shape_ids=[shape.id],
        candidate_hashes=[candidate.hash],
    ) == {(shape.id, candidate.hash)}
    assert db.rank_benchmarks() == []
    assert db.benchmark_status_summary() == {"rejected": 1}
