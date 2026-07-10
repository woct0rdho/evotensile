import math

from evotensile.candidate import Shape
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.search.hot_confirm import hot_confirm_topk
from evotensile.search.replay import OracleRecord, ReplayCostModel, merge_oracle_records, simulate_candidate_stream
from tests.helpers import sample_candidates


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
        screening_launches=0,
        hot_launches=0,
        probe_min_survivors=1,
    )

    result = simulate_candidate_stream(
        candidates,
        oracle=oracle,
        shape=shape,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
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
        screening_launches=0,
        hot_launches=0,
        probe_min_survivors=1,
    )

    result = simulate_candidate_stream(
        candidates,
        oracle=oracle,
        shape=shape,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        cost=cost,
        seed=20260710,
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
        screening_launches=0,
        hot_launches=0,
        probe_min_survivors=1,
        probe_max_slowdown_factor=4.0,
    )

    result = simulate_candidate_stream(
        candidates,
        oracle=oracle,
        shape=shape,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        cost=cost,
        seed=20260710,
        batch_size=3,
        pool_window=3,
    )

    flops = 2.0 * shape.m * shape.n * shape.batch * shape.k
    launch_seconds = [flops / (value * 1.0e9) for value in gflops]
    expected = 3.0 * launch_seconds[0] + 3.0 * launch_seconds[1] + launch_seconds[2]
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
        screening_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        validation_protocol_hash=DEFAULT_PROFILE.default_protocol.validation_protocol_hash(),
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
        screening_launches=0,
        hot_launches=0,
        probe_min_survivors=1,
        probe_max_slowdown_factor=4.0,
    )

    result = simulate_candidate_stream(
        candidates,
        oracle=oracle,
        shape=shape,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        cost=cost,
        seed=20260710,
        batch_size=3,
        pool_window=3,
    )

    assert result.screened == [candidates[2].hash]
    assert set(result.screening_survivors) == {candidates[0].hash, candidates[1].hash}
