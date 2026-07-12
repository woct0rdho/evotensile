import sqlite3
from contextlib import closing
from pathlib import Path

from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.evaluator import (
    HybridEvaluator,
    RealEvaluator,
    RealEvaluatorContext,
    ReplayEvaluator,
)
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.protocol import DEFAULT_BENCHMARK_PROTOCOL
from evotensile.scheduling.models import EvidenceStage, PairRequest
from evotensile.search.replay import ExactOracleReplayState, OracleRecord
from evotensile.shapes import pilot_100_shapes
from tests.helpers import fake_build_tensile, fake_structured_runner, sample_candidates


def _replay_state(
    db: EvoTensileDB,
    *,
    shapes,
    oracle,
) -> ExactOracleReplayState:
    return ExactOracleReplayState(
        db=db,
        shapes=shapes,
        oracle=oracle,
        profile=DEFAULT_PROFILE,
        screening_protocol=DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1),
        source_ref="retained_exact_oracle",
    )


def test_replay_evaluator_leaves_missing_exact_pair_unknown(tmp_path: Path):
    shapes = pilot_100_shapes()[:2]
    candidate = sample_candidates(1)[0]
    db = EvoTensileDB.connect(
        tmp_path / "overlay.sqlite",
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    state = _replay_state(
        db,
        shapes=shapes,
        oracle={
            (shapes[0].id, candidate.hash): OracleRecord(
                candidate=candidate,
                status="ok",
                screening_gflops=10.0,
            )
        },
    )
    evaluator = ReplayEvaluator(state)
    requests = [PairRequest(candidate, shape) for shape in shapes]
    controller = CampaignControllerState(
        shape_ids=tuple(shape.id for shape in shapes),
        time_budget_s=10.0,
        session_started_at=0.0,
    )

    result = evaluator.evaluate(requests)
    result.apply(controller)

    assert [outcome.status for outcome in result.outcomes] == ["ok", "unknown"]
    assert result.known_pairs == 1
    assert result.unknown_pairs == 1
    assert controller.known_pairs == {requests[0].key}
    assert controller.unknown_pairs == {requests[1].key}
    assert state.disclosed_pairs == {requests[0].key}


def test_real_evaluator_records_native_exact_result_and_measured_costs(tmp_path: Path):
    shape = pilot_100_shapes()[0]
    candidate = sample_candidates(1)[0]
    protocol = DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1)
    db = EvoTensileDB.connect(
        tmp_path / "real.sqlite",
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    evaluator = RealEvaluator(
        RealEvaluatorContext(
            db=db,
            output_root=tmp_path / "real",
            target_profile=DEFAULT_PROFILE,
            protocol=protocol,
            runner_bin=fake_structured_runner(tmp_path),
            tensilelite_bin=fake_build_tensile(tmp_path),
            candidate_batch_size=1,
            shape_batch_size=1,
        )
    )
    controller = CampaignControllerState(
        shape_ids=(shape.id,),
        time_budget_s=10.0,
        session_started_at=0.0,
    )

    result = evaluator.evaluate([PairRequest(candidate, shape)])
    result.apply(controller)

    outcome = result.outcomes[0]
    assert outcome.provenance == "native"
    assert outcome.status == "ok"
    assert outcome.samples == 1
    assert outcome.performance is not None
    assert result.prepared_artifact_shapes == {candidate.hash: (shape.id,)}
    assert result.phase_time_s["preparation"] >= 0.0
    assert controller.known_pairs == {(shape.id, candidate.hash)}
    assert controller.incumbents[shape.id].candidate_hash == candidate.hash


def test_real_confirmation_revalidates_and_remeasures_with_cache_ignored(tmp_path: Path):
    shape = pilot_100_shapes()[0]
    candidate = sample_candidates(1)[0]
    protocol = DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1)
    db = EvoTensileDB.connect(
        tmp_path / "confirmation.sqlite",
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    runner_bin = fake_structured_runner(tmp_path)
    tensilelite_bin = fake_build_tensile(tmp_path)
    screening = RealEvaluator(
        RealEvaluatorContext(
            db=db,
            output_root=tmp_path / "confirmation",
            target_profile=DEFAULT_PROFILE,
            protocol=protocol,
            runner_bin=runner_bin,
            tensilelite_bin=tensilelite_bin,
            candidate_batch_size=1,
            shape_batch_size=1,
        )
    )
    confirmation = RealEvaluator(
        RealEvaluatorContext(
            db=db,
            output_root=tmp_path / "confirmation",
            target_profile=DEFAULT_PROFILE,
            protocol=protocol,
            runner_bin=runner_bin,
            tensilelite_bin=tensilelite_bin,
            candidate_batch_size=1,
            shape_batch_size=1,
            ignore_cache=True,
        )
    )
    request = PairRequest(
        candidate,
        shape,
        evidence_stage=EvidenceStage.CONFIRMATION,
    )

    screening.evaluate([request])
    result = confirmation.evaluate([request])

    planned = result.schedules[0].planned_batches[0].pairs[0]
    assert planned.requires_validation
    assert planned.samples_to_collect == 1
    assert "confirmation" in result.phase_time_s
    assert "screening" not in result.phase_time_s
    summary = db.rank_benchmarks(
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(protocol),
        shape_id=shape.id,
        min_samples=1,
    )[0]
    assert summary.samples == 2


def test_real_evaluator_measures_artifact_scope_expansion(tmp_path: Path):
    shapes = pilot_100_shapes()[:2]
    candidate = sample_candidates(1)[0]
    protocol = DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1)
    db = EvoTensileDB.connect(
        tmp_path / "expanded.sqlite",
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    compile_cache = tmp_path / "compile_cache"
    evaluator = RealEvaluator(
        RealEvaluatorContext(
            db=db,
            output_root=tmp_path / "expanded",
            target_profile=DEFAULT_PROFILE,
            protocol=protocol,
            runner_bin=fake_structured_runner(tmp_path),
            tensilelite_bin=fake_build_tensile(tmp_path),
            candidate_batch_size=1,
            shape_batch_size=2,
            compile_cache_root=compile_cache,
        )
    )
    controller = CampaignControllerState(
        shape_ids=tuple(shape.id for shape in shapes),
        time_budget_s=10.0,
        session_started_at=0.0,
    )

    first = evaluator.evaluate([PairRequest(candidate, shapes[0])])
    first.apply(controller)
    second = evaluator.evaluate(
        [PairRequest(candidate, shapes[1])],
        artifact_shapes_by_candidate={candidate.hash: shapes},
    )
    second.apply(controller)

    assert first.prepared_artifact_shapes == {candidate.hash: (shapes[0].id,)}
    assert second.prepared_artifact_shapes == {candidate.hash: tuple(sorted(shape.id for shape in shapes))}
    assert second.phase_time_s["preparation"] > 0.0
    assert controller.prepared_artifact_shapes[candidate.hash] == {shape.id for shape in shapes}
    assert len([path for path in compile_cache.iterdir() if path.is_dir()]) == 2


def test_hybrid_evaluator_replays_known_pair_and_routes_absent_pair_to_native_runner(tmp_path: Path):
    shapes = pilot_100_shapes()[:2]
    candidates = sample_candidates(2)
    candidate = candidates[0]
    neighbor = candidates[1]
    protocol = DEFAULT_BENCHMARK_PROTOCOL.with_overrides(num_benchmarks=1)
    overlay_path = tmp_path / "hybrid_overlay.sqlite"
    db = EvoTensileDB.connect(
        overlay_path,
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    oracle = {
        (shapes[0].id, candidate.hash): OracleRecord(
            candidate=candidate,
            status="ok",
            screening_gflops=10.0,
            source_artifact="retained.sqlite",
        ),
        (shapes[0].id, neighbor.hash): OracleRecord(
            candidate=neighbor,
            status="ok",
            screening_gflops=11.0,
            source_artifact="retained.sqlite",
        ),
    }
    retained_snapshot = dict(oracle)
    replay = ReplayEvaluator(_replay_state(db, shapes=shapes, oracle=oracle))
    real = RealEvaluator(
        RealEvaluatorContext(
            db=db,
            output_root=tmp_path / "native_fallback",
            target_profile=DEFAULT_PROFILE,
            protocol=protocol,
            runner_bin=fake_structured_runner(tmp_path),
            tensilelite_bin=fake_build_tensile(tmp_path),
            candidate_batch_size=1,
            shape_batch_size=1,
        ),
        source_ref="hybrid_native_fallback",
    )
    evaluator = HybridEvaluator(replay, real)
    requests = [PairRequest(candidate, shapes[0]), PairRequest(candidate, shapes[1])]
    controller = CampaignControllerState(
        shape_ids=tuple(shape.id for shape in shapes),
        time_budget_s=10.0,
        session_started_at=0.0,
    )

    result = evaluator.evaluate(requests)
    result.apply(controller)

    assert [outcome.provenance for outcome in result.outcomes] == ["replay", "native"]
    assert [outcome.status for outcome in result.outcomes] == ["ok", "ok"]
    assert result.known_pairs == 2
    assert result.unknown_pairs == 0
    assert result.prepared_artifact_shapes[candidate.hash] == tuple(sorted(shape.id for shape in shapes))
    assert controller.known_pairs == {request.key for request in requests}
    assert controller.unknown_pairs == set()
    assert replay.state.queried_pairs == {requests[0].key}
    assert oracle == retained_snapshot

    ranked_keys = {
        (row.shape_id, row.candidate_hash)
        for row in db.rank_benchmarks(
            problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
            benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(protocol),
            min_samples=1,
        )
    }
    assert ranked_keys == {request.key for request in requests}
    assert (shapes[0].id, neighbor.hash) not in ranked_keys
    with closing(sqlite3.connect(overlay_path)) as connection:
        source_kinds = {row[0] for row in connection.execute("SELECT DISTINCT source_kind FROM evidence_sources")}
    assert {"replay", "native_run"}.issubset(source_kinds)
