from pathlib import Path

from evotensile.adaptive_retime import timing_stats_from_times
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.protocol import DEFAULT_BENCHMARK_PROTOCOL
from evotensile.scheduler import execute_schedule
from evotensile.search.screening_stabilize import (
    ScreeningStabilizationPolicy,
    screening_topup_requests,
    stabilize_screening_leaders,
)
from evotensile.shapes import pilot_100_shapes
from tests.helpers import sample_candidates
from tests.test_structured_runner import fake_build_tensile, fake_structured_runner


def test_screening_topup_requests_require_duration_and_confidence():
    protocol = DEFAULT_BENCHMARK_PROTOCOL.with_overrides(
        num_warmups=1,
        num_benchmarks=2,
        enqueues_per_sync=1,
    )
    stats = [
        timing_stats_from_times("shape", "best", [25_000.0, 25_000.0]),
        timing_stats_from_times("shape", "near", [25_500.0, 25_500.0]),
        timing_stats_from_times("shape", "slow", [35_000.0, 35_000.0]),
    ]

    requests = screening_topup_requests(
        stats,
        protocol=protocol,
        policy=ScreeningStabilizationPolicy(
            top_k=3,
            contender_epsilon_pct=3.0,
            min_samples=6,
            max_samples=10,
            min_timed_duration_us=100_000.0,
        ),
    )

    assert [request.candidate_hash for request in requests] == ["best", "near"]
    assert all(request.current_samples == 2 for request in requests)
    assert all(request.target_samples == 6 for request in requests)


def test_screening_stabilization_reuses_prior_artifacts(tmp_path: Path, monkeypatch):
    fake_tensile = fake_build_tensile(tmp_path)
    fake_runner = fake_structured_runner(tmp_path)
    db = EvoTensileDB.connect(tmp_path / "campaign.sqlite")
    candidates = sample_candidates(2)
    shape = pilot_100_shapes()[0]
    protocol = DEFAULT_BENCHMARK_PROTOCOL.with_overrides(
        num_warmups=1,
        num_benchmarks=2,
        enqueues_per_sync=1,
    )
    events_path = tmp_path / "events.log"
    monkeypatch.setenv("EVOTENSILE_TEST_PHASE_EVENTS", str(events_path))

    execute_schedule(
        db,
        shapes=[shape],
        candidates=candidates,
        output_root=tmp_path / "round_00",
        protocol=protocol,
        candidate_batch_size=2,
        shape_batch_size=1,
        tensilelite_bin=fake_tensile,
        runner_bin=fake_runner,
        keep_going=True,
        prepare_workers=2,
    )
    result = stabilize_screening_leaders(
        db,
        shape=shape,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        screening_protocol=protocol,
        validation_protocol_hash=protocol.validation_protocol_hash(),
        output_dir=tmp_path / "leader_stabilization",
        runner_bin=fake_runner,
        policy=ScreeningStabilizationPolicy(
            top_k=2,
            contender_epsilon_pct=100.0,
            min_samples=6,
            max_samples=6,
            min_timed_duration_us=0.0,
        ),
    )

    ranked = db.rank_evaluations(
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=protocol.protocol_hash(),
        shape_id=shape.id,
    )
    events = events_path.read_text(encoding="utf-8").splitlines()
    assert result.runs == 1
    assert result.added_samples == 8
    assert not result.errors
    assert [summary.samples for summary in ranked] == [6, 6]
    assert events.count("compile_start") == 1
    assert events.count("validate_start") == 1
    assert events.count("benchmark_start") == 2
