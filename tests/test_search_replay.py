import json
import math
from dataclasses import asdict, replace

import pytest

from evotensile.campaign.protocols import CAMPAIGN_HOT_PROTOCOL, CAMPAIGN_SCREENING_PROTOCOL
from evotensile.candidate import Shape
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.search import replay as replay_module
from evotensile.search.hot_confirm import hot_confirm_topk
from evotensile.search.replay import (
    ExactOracleReplayState,
    OracleRecord,
    ReplayCostModel,
    load_db_oracle_matrix,
    merge_oracle_records,
    simulate_candidate_stream,
)
from tests.helpers import insert_test_benchmark_event, sample_candidates


def test_replay_cost_model_recursively_serializes_nested_protocols():
    payload = json.loads(json.dumps(asdict(ReplayCostModel())))

    assert payload["screening_protocol"]["role"] == "main"
    assert payload["stabilization_policy"]["min_samples"] == 8
    assert ReplayCostModel().screening_launches == 3
    assert ReplayCostModel().hot_launches == 120


def test_load_db_oracle_matrix_indexes_exact_shape_candidate_pairs(tmp_path):
    shapes = [Shape(512, 128, 1, 256), Shape(640, 256, 1, 512)]
    candidates = sample_candidates(2, seed=20260730)
    db_path = tmp_path / "oracle.sqlite"
    db = EvoTensileDB.connect(db_path)
    db.init()
    db.register_shapes(shapes)
    db.register_candidates(candidates)
    protocol_hash = DEFAULT_PROFILE.benchmark_protocol_hash(CAMPAIGN_SCREENING_PROTOCOL)
    insert_test_benchmark_event(
        db,
        shape_id=shapes[0].id,
        candidate_hash=candidates[0].hash,
        run_id="shape-0-ok",
        status="ok",
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=protocol_hash,
        time_us=100.0,
    )
    insert_test_benchmark_event(
        db,
        shape_id=shapes[1].id,
        candidate_hash=candidates[0].hash,
        run_id="shape-1-ok",
        status="ok",
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=protocol_hash,
        time_us=200.0,
    )
    insert_test_benchmark_event(
        db,
        shape_id=shapes[1].id,
        candidate_hash=candidates[1].hash,
        run_id="shape-1-rejected",
        status="rejected",
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=protocol_hash,
    )

    matrix = load_db_oracle_matrix(
        db_path,
        shapes=shapes,
        benchmark_protocol_hash=protocol_hash,
    )

    assert set(matrix) == {
        (shapes[0].id, candidates[0].hash),
        (shapes[1].id, candidates[0].hash),
        (shapes[1].id, candidates[1].hash),
    }
    assert (
        matrix[(shapes[0].id, candidates[0].hash)].screening_gflops
        != matrix[(shapes[1].id, candidates[0].hash)].screening_gflops
    )
    assert matrix[(shapes[1].id, candidates[1].hash)].status == "rejected"
    assert matrix[(shapes[1].id, candidates[1].hash)].screening_gflops is None


def test_multi_shape_replay_state_is_query_causal_and_reuses_preparation(tmp_path):
    shapes = [Shape(512, 128, 1, 256), Shape(640, 256, 1, 512)]
    candidates = sample_candidates(3, seed=20260731)
    oracle = {
        (shapes[0].id, candidates[0].hash): OracleRecord(
            candidate=candidates[0],
            status="ok",
            screening_gflops=10_000.0,
        ),
        (shapes[1].id, candidates[0].hash): OracleRecord(
            candidate=candidates[0],
            status="ok",
            screening_gflops=12_000.0,
        ),
        (shapes[0].id, candidates[1].hash): OracleRecord(
            candidate=candidates[1],
            status="validation_fail",
        ),
        (shapes[1].id, candidates[1].hash): OracleRecord(
            candidate=candidates[1],
            status="rejected",
        ),
    }
    state = ExactOracleReplayState(
        db=EvoTensileDB.connect(tmp_path / "replay.sqlite"),
        shapes=shapes,
        oracle=oracle,
        profile=DEFAULT_PROFILE,
        source_ref="multi-shape-test",
    )

    assert state.evidence_snapshot().summaries == ()
    assert state.prepare_candidates(candidates[:2], workers=2, seconds_per_candidate=8.0) == 8.0
    assert state.prepare_candidates([candidates[0]], workers=2, seconds_per_candidate=8.0) == 0.0

    first = state.query_pair(shapes[0], candidates[0])
    assert first.known is True
    assert first.first_query is True
    assert [summary.shape_id for summary in state.evidence_snapshot().summaries] == [shapes[0].id]

    state.query_pair(shapes[1], candidates[0], disclose=False)
    assert [summary.shape_id for summary in state.evidence_snapshot().summaries] == [shapes[0].id]
    assert state.summary()["unresolved_shape_ids"] == [shapes[1].id]
    assert state.queried_shape_ids(candidates[0].hash, successful_only=True) == (shapes[0].id,)
    state.disclose_pair(shapes[1], candidates[0].hash)
    summaries = state.evidence_snapshot().summaries
    assert {summary.shape_id for summary in summaries} == {shape.id for shape in shapes}
    duplicate = state.query_pair(shapes[1], candidates[0])
    assert duplicate.first_query is False
    assert state.evidence_snapshot().summaries == summaries

    failed = state.query_pair(shapes[0], candidates[1])
    assert failed.known is True
    counts = state.evidence_snapshot(shapes=[shapes[0]]).evidence_status_counts
    assert counts[candidates[1].hash]["validation_failed"] == 1

    unknown = state.query_pair(shapes[1], candidates[2])
    assert unknown.known is False
    assert (shapes[1].id, candidates[2].hash) in state.unknown_pairs
    assert candidates[2].hash not in state.evidence_snapshot().evidence_status_counts

    with pytest.raises(ValueError, match="before exact query"):
        state.disclose_pair(shapes[1], candidates[1].hash)
    state.query_pair(shapes[1], candidates[1])
    rejected_counts = state.evidence_snapshot(shapes=[shapes[1]]).evidence_status_counts
    assert rejected_counts[candidates[1].hash]["rejected"] == 1

    state.record_pair_time(shapes[0], candidates[0].hash, 0.2)
    state.record_pair_time(shapes[1], candidates[0].hash, 0.3)
    summary = state.summary()
    assert summary["prepared_candidates"] == 2
    assert summary["preparation_time_s"] == 8.0
    assert summary["pair_time_s"] == pytest.approx(0.5)
    assert summary["simulated_time_s"] == pytest.approx(8.5)
    assert summary["unresolved_shape_ids"] == []
    assert state.queried_shape_ids(candidates[0].hash, successful_only=True) == tuple(shape.id for shape in shapes)
    assert state.shape_state(shapes[1]).incumbent_hash == candidates[0].hash


def test_merge_oracle_records_keeps_exact_hash_measurements_and_hot_results():
    candidates = sample_candidates(2, seed=20260710)
    records = [
        OracleRecord(candidate=candidates[0], status="ok", screening_gflops=10_000.0, order=2.0),
        OracleRecord(candidate=candidates[0], status="ok", screening_gflops=11_000.0, order=1.0),
        OracleRecord(candidate=candidates[1], status="validation_fail", order=3.0),
    ]

    merged = merge_oracle_records([records], hot_measurements={candidates[0].hash: 12_000.0})

    assert merged[candidates[0].hash].screening_gflops == 11_000.0
    assert merged[candidates[0].hash].hot_gflops == 12_000.0
    assert merged[candidates[0].hash].order == 1.0
    assert merged[candidates[1].hash].screening_gflops is None


def test_replay_uses_explicit_profile_identity_and_mechanics(tmp_path, monkeypatch):
    shape = Shape(512, 128, 1, 256)
    candidate = sample_candidates(1)[0]
    profile = replace(
        DEFAULT_PROFILE,
        name="replacement-profile",
        environment_compatibility_tag="replacement-environment",
        workgroup_processor_count=10,
        compute_unit_count=20,
    )
    observed: dict[str, object] = {}
    original_connect = EvoTensileDB.connect
    original_select = replay_module.select_surrogate_pool

    def connect(path, *, environment_compatibility_tag=None):
        observed["environment"] = environment_compatibility_tag
        return original_connect(path, environment_compatibility_tag=environment_compatibility_tag)

    def select(candidates, **kwargs):
        observed["problem_type_hash"] = kwargs["evidence"].problem_type_hash
        observed["benchmark_protocol_hash"] = kwargs["evidence"].benchmark_protocol_hash
        observed["workgroup_processor_count"] = kwargs["workgroup_processor_count"]
        return original_select(candidates, **kwargs)

    monkeypatch.setattr(replay_module.EvoTensileDB, "connect", connect)
    monkeypatch.setattr(replay_module, "select_surrogate_pool", select)
    simulate_candidate_stream(
        [candidate],
        oracle={},
        shape=shape,
        profile=profile,
        cost=ReplayCostModel(time_budget_s=1.0, prepare_seconds_per_candidate=0.0, hot_reserve_s=0.0),
        seed=1,
        surrogate_min_evidence=24,
        batch_size=1,
        pool_window=1,
    )

    assert observed == {
        "environment": "replacement-environment",
        "problem_type_hash": profile.problem_type_hash,
        "benchmark_protocol_hash": profile.benchmark_protocol_hash(CAMPAIGN_SCREENING_PROTOCOL),
        "workgroup_processor_count": 10,
    }


def test_replay_discloses_only_queried_exact_hashes_and_confirms_hot_target():
    shape = Shape(8192, 8192, 1, 8192)
    candidates = sample_candidates(3, seed=20260711)
    oracle = {
        candidates[0].hash: OracleRecord(
            candidate=candidates[0],
            status="ok",
            screening_gflops=20_000.0,
            hot_gflops=25_000.0,
        ),
        candidates[1].hash: OracleRecord(
            candidate=candidates[1],
            status="ok",
            screening_gflops=10_000.0,
            hot_gflops=12_000.0,
        ),
    }
    cost = ReplayCostModel(
        time_budget_s=1200.0,
        prepare_seconds_per_candidate=0.0,
        probe_launches=0,
        screening_protocol=CAMPAIGN_SCREENING_PROTOCOL.with_overrides(num_warmups=0),
        hot_protocol=CAMPAIGN_HOT_PROTOCOL.with_overrides(num_warmups=0, num_benchmarks=1, enqueues_per_sync=1),
        probe_min_survivors=1,
    )

    result = simulate_candidate_stream(
        candidates,
        oracle=oracle,
        shape=shape,
        profile=DEFAULT_PROFILE,
        cost=cost,
        seed=20260710,
        batch_size=3,
        pool_window=3,
        surrogate_min_evidence=24,
        hot_finalists=2,
        target_hot_gflops=24_000.0,
    )

    assert set(result.queried) == {candidate.hash for candidate in candidates}
    assert result.unknown == [candidates[2].hash]
    assert result.best_screening_hash == candidates[0].hash
    assert result.best_hot_hash == candidates[0].hash
    assert result.best_hot_gflops == 25_000.0
    assert result.reached_target is True


def test_replay_models_covering_islands_and_leader_stabilization():
    shape = Shape(8192, 8192, 1, 8192)
    candidates = sample_candidates(16, seed=20260720)
    oracle = {
        candidate.hash: OracleRecord(
            candidate=candidate,
            status="ok",
            screening_gflops=20_000.0 + index * 100.0,
        )
        for index, candidate in enumerate(candidates)
    }
    cost = ReplayCostModel(
        time_budget_s=1200.0,
        prepare_seconds_per_candidate=0.0,
        probe_launches=0,
        screening_protocol=CAMPAIGN_SCREENING_PROTOCOL.with_overrides(num_warmups=0),
        hot_protocol=CAMPAIGN_HOT_PROTOCOL.with_overrides(num_warmups=0, num_benchmarks=1, enqueues_per_sync=1),
        probe_min_survivors=1,
    )

    result = simulate_candidate_stream(
        candidates,
        oracle=oracle,
        shape=shape,
        profile=DEFAULT_PROFILE,
        cost=cost,
        seed=20260710,
        surrogate_min_evidence=24,
        batch_size=8,
        pool_window=16,
        covering_cold_start=True,
        island_count=2,
        island_isolation_rounds=2,
        leader_stabilization=True,
    )

    assert len(result.queried) == 16
    assert result.simulated_time_s <= cost.time_budget_s
    stabilization_samples = [item["stabilization_samples"] for item in result.trace]
    diagnostics = [item["population_diagnostics"] for item in result.trace]
    assert all(isinstance(value, int) for value in stabilization_samples)
    assert sum(value for value in stabilization_samples if isinstance(value, int)) > 0
    assert all(isinstance(value, dict) and value.get("candidates") == 8 for value in diagnostics)


def test_replay_staged_probe_charges_one_launch_for_catastrophic_rows():
    shape = Shape(8192, 8192, 1, 8192)
    candidates = sample_candidates(3, seed=20260721)
    gflops = [20_000.0, 10_000.0, 1_000.0]
    oracle = {
        candidate.hash: OracleRecord(candidate=candidate, status="ok", screening_gflops=value)
        for candidate, value in zip(candidates, gflops, strict=True)
    }
    cost = ReplayCostModel(
        time_budget_s=1200.0,
        prepare_seconds_per_candidate=0.0,
        probe_launches=3,
        initial_probe_launches=1,
        screening_protocol=CAMPAIGN_SCREENING_PROTOCOL.with_overrides(num_warmups=0),
        hot_protocol=CAMPAIGN_HOT_PROTOCOL.with_overrides(num_warmups=0, num_benchmarks=1, enqueues_per_sync=1),
        probe_min_survivors=1,
        probe_max_slowdown_factor=4.0,
    )

    result = simulate_candidate_stream(
        candidates,
        oracle=oracle,
        shape=shape,
        profile=DEFAULT_PROFILE,
        cost=cost,
        seed=20260710,
        surrogate_min_evidence=24,
        batch_size=3,
        pool_window=3,
    )

    flops = 2.0 * shape.m * shape.n * shape.batch * shape.k
    launch_seconds = [flops / (value * 1.0e9) for value in gflops]
    expected = (
        (3.0 + cost.screening_launches) * launch_seconds[0]
        + (3.0 + cost.screening_launches) * launch_seconds[1]
        + launch_seconds[2]
    )
    assert math.isclose(result.simulated_time_s, expected)
    assert result.screened == [candidates[2].hash]


def test_hot_confirmation_handles_an_empty_validated_ranking(tmp_path):
    db_path = tmp_path / "campaign.sqlite"
    db = EvoTensileDB.connect(db_path)
    db.init()

    records = hot_confirm_topk(
        db_path=db_path,
        output_dir=tmp_path / "hot",
        runner_bin=tmp_path / "runner",
        shape_id="m8192_n8192_b1_k8192",
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        screening_protocol=DEFAULT_PROFILE.default_protocol,
        hot_protocol=DEFAULT_PROFILE.default_protocol.with_overrides(num_elements_to_validate=0),
        top_k=8,
        runner_timeout_s=300.0,
    )

    assert records == []
    assert (tmp_path / "hot" / "summary.json").exists()


def test_replay_probe_screens_catastrophic_exact_rows_before_main_evidence():
    shape = Shape(8192, 8192, 1, 8192)
    candidates = sample_candidates(3, seed=20260712)
    oracle = {
        candidates[0].hash: OracleRecord(candidate=candidates[0], status="ok", screening_gflops=20_000.0),
        candidates[1].hash: OracleRecord(candidate=candidates[1], status="ok", screening_gflops=10_000.0),
        candidates[2].hash: OracleRecord(candidate=candidates[2], status="ok", screening_gflops=1_000.0),
    }
    cost = ReplayCostModel(
        time_budget_s=1200.0,
        prepare_seconds_per_candidate=0.0,
        probe_launches=0,
        screening_protocol=CAMPAIGN_SCREENING_PROTOCOL.with_overrides(num_warmups=0),
        hot_protocol=CAMPAIGN_HOT_PROTOCOL.with_overrides(num_warmups=0, num_benchmarks=1, enqueues_per_sync=1),
        probe_min_survivors=1,
        probe_max_slowdown_factor=4.0,
    )

    result = simulate_candidate_stream(
        candidates,
        oracle=oracle,
        shape=shape,
        profile=DEFAULT_PROFILE,
        cost=cost,
        seed=20260710,
        surrogate_min_evidence=24,
        batch_size=3,
        pool_window=3,
    )

    assert result.screened == [candidates[2].hash]
    assert set(result.screening_survivors) == {candidates[0].hash, candidates[1].hash}
