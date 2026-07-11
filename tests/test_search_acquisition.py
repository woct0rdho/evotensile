from dataclasses import replace
from pathlib import Path

from evotensile.candidate import Shape
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.proposal import FamilyQDPolicy
from evotensile.search.acquisition import propose_candidates
from evotensile.search_space import cheap_constraints
from evotensile.shapes import pilot_100_shapes
from tests.helpers import insert_test_benchmark_event, sample_candidates

BASE_TEST_POLICY = replace(
    FamilyQDPolicy(),
    num_random=0,
    elite_count=8,
    local_count=0,
    de_count=0,
    gomea_count=0,
    transfer_shape_count=0,
    transfer_per_shape=0,
    adaptive_operators=False,
    surrogate_pool_multiplier=1,
    covering_cold_start=False,
    adaptive_group_credit=False,
    micro_exhaustive_neighborhoods=False,
    adaptive_donor_selection=False,
    cost_aware_operator_credit=False,
)


def _policy(**overrides) -> FamilyQDPolicy:
    return replace(BASE_TEST_POLICY, **overrides)


def _insert_ranked_candidates(db: EvoTensileDB, candidates, shapes: list[Shape]) -> None:
    db.register_candidates(candidates)
    db.register_shapes(shapes)
    for shape_index, shape in enumerate(shapes):
        for candidate_index, candidate in enumerate(candidates):
            insert_test_benchmark_event(
                db,
                shape_id=shape.id,
                candidate_hash=candidate.hash,
                run_id="cached",
                status="ok",
                problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
                benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
                time_us=1000.0 * shape_index + candidate_index + 1.0,
            )


def test_family_qd_builds_one_evidence_snapshot_per_proposal(tmp_path: Path, monkeypatch):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(8)
    shape = pilot_100_shapes()[0]
    _insert_ranked_candidates(db, candidates, [shape])
    rank_calls = 0
    original_rank = db.rank_benchmarks

    def counted_rank(*args, **kwargs):
        nonlocal rank_calls
        rank_calls += 1
        return original_rank(*args, **kwargs)

    monkeypatch.setattr(db, "rank_benchmarks", counted_rank)
    result = propose_candidates(
        db,
        policy=_policy(
            num_random=2,
            local_count=2,
            de_count=2,
            gomea_count=4,
            adaptive_operators=True,
            adaptive_group_credit=True,
            adaptive_donor_selection=True,
            surrogate_pool_multiplier=2,
            linkage_min_samples=4,
            linkage_truncation_tau=1.0,
        ),
        target_shapes=[shape],
    )

    assert result.selected
    assert rank_calls == 1
    assert result.provider["identity"] == "builtin:family-qd:gfx1151-grid-v1"


def test_family_qd_preserves_shape_normalized_specialists(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(4)
    small = Shape(512, 128, 1, 256)
    large = Shape(8192, 8192, 1, 8192)
    db.register_candidates(candidates)
    db.register_shapes([small, large])
    for shape, rows in (
        (small, ((candidates[0], 1.0), (candidates[1], 2.0))),
        (large, ((candidates[2], 1000.0), (candidates[3], 1200.0))),
    ):
        for candidate, time_us in rows:
            insert_test_benchmark_event(
                db,
                shape_id=shape.id,
                candidate_hash=candidate.hash,
                run_id="cached",
                status="ok",
                problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
                benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
                time_us=time_us,
            )

    result = propose_candidates(
        db,
        policy=_policy(local_count=8, elite_count=2),
        target_shapes=[small, large],
        seed=7,
    )

    parent_hashes = {parent for candidate in result.generated for parent in candidate.parent_hashes}
    assert candidates[0].hash in parent_hashes
    assert candidates[2].hash in parent_hashes


def test_family_qd_transfer_round_robins_target_neighborhoods(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(2)
    first = Shape(512, 128, 1, 256)
    second = Shape(8192, 8192, 1, 8192)
    for shape, candidate, time_us in ((first, candidates[0], 1.0), (second, candidates[1], 1000.0)):
        db.register_candidates([candidate])
        db.register_shapes([shape])
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
            benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
            time_us=time_us,
        )

    result = propose_candidates(
        db,
        policy=_policy(transfer_shape_count=2, transfer_per_shape=1),
        target_shapes=[first, second],
    )

    transfer = [candidate for candidate in result.preserved if candidate.source == "transfer"]
    assert {candidate.hash for candidate in transfer} == {candidate.hash for candidate in candidates}
    assert set().union(*(set(candidate.proposal_metadata["transfer_target_shape_ids"]) for candidate in transfer)) == {
        first.id,
        second.id,
    }


def test_family_qd_parent_override_prevents_global_parent_selection(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    shape = pilot_100_shapes()[0]
    candidates = sample_candidates(12, seed=20260710)
    _insert_ranked_candidates(db, candidates, [shape])
    island_parents = candidates[:8]

    result = propose_candidates(
        db,
        policy=_policy(local_count=8, gomea_count=8),
        target_shapes=[shape],
        shape_id=shape.id,
        parent_candidates=island_parents,
        seed=20260711,
    )

    parent_hashes = {candidate.hash for candidate in island_parents}
    assert result.generated
    assert all(set(candidate.parent_hashes) <= parent_hashes for candidate in result.generated)
    assert not ({candidate.hash for candidate in candidates[8:]} & {candidate.hash for candidate in result.selected})


def test_family_qd_cold_start_is_shape_aware_and_balanced(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    shape = Shape(1024, 1024, 1, 1024)

    result = propose_candidates(
        db,
        policy=_policy(num_random=3),
        target_shapes=[shape],
        seed=1151,
    )

    assert len(result.selected) == 3
    assert {candidate.params["TransposeLDS"] for candidate in result.selected} == {0, 2}
    assert all(cheap_constraints(candidate.canonical_params(), shape=shape) for candidate in result.selected)


def test_family_qd_surrogate_multiplier_preserves_measurement_budget(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    shape = Shape(1024, 1024, 1, 1024)

    result = propose_candidates(
        db,
        policy=_policy(num_random=3, surrogate_pool_multiplier=2, covering_cold_start=True),
        target_shapes=[shape],
        seed=20260710,
    )

    assert len(result.generated) == 6
    assert len(result.selected) == 3


def test_family_qd_adaptive_operators_keep_distinct_sources(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(12, seed=20260710)
    shape = Shape(8192, 8192, 1, 8192)
    _insert_ranked_candidates(db, candidates, [shape])

    result = propose_candidates(
        db,
        policy=_policy(
            num_random=0,
            local_count=4,
            de_count=4,
            gomea_count=8,
            elite_count=12,
            adaptive_operators=True,
            adaptive_group_credit=True,
            micro_exhaustive_neighborhoods=True,
            adaptive_donor_selection=True,
        ),
        target_shapes=[shape],
        seed=20260711,
    )

    sources = {candidate.source for candidate in result.generated}
    assert {"semantic-mutation", "de", "gomea-neighborhood", "gomea-mixing"} <= sources
