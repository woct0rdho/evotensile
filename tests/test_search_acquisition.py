from pathlib import Path

from evotensile.candidate import Shape
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.search.acquisition import propose_candidates
from evotensile.search_space import cheap_constraints
from evotensile.shapes import pilot_100_shapes
from tests.helpers import REFERENCE_CANDIDATE, insert_test_benchmark_event, sample_candidates


def test_local_proposal_mutates_cached_elites(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(2)
    shapes = pilot_100_shapes()[:1]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes(shapes)
    insert_test_benchmark_event(
        db,
        shape_id=shapes[0].id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=10.0,
    )

    proposed = propose_candidates(
        db,
        proposal="local",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        local_count=1,
        elite_count=1,
        seed=7,
    ).selected

    mutations = [candidate for candidate in proposed if candidate.source == "mutation"]
    assert len(mutations) == 1
    assert mutations[0].parent_hashes == (candidates[0].hash,)
    assert {candidate.source for candidate in proposed} == {"mutation"}


def test_multi_shape_elites_include_shape_normalized_specialists(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(4)
    small = Shape(512, 128, 1, 256)
    large = Shape(8192, 8192, 1, 8192)
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
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
                problem_type_hash=p_hash,
                benchmark_protocol_hash=b_hash,
                time_us=time_us,
            )

    proposed = propose_candidates(
        db,
        proposal="local",
        local_count=8,
        elite_count=2,
        target_shapes=[small, large],
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        seed=7,
    ).selected

    parent_hashes = {parent_hash for candidate in proposed for parent_hash in candidate.parent_hashes}
    assert candidates[0].hash in parent_hashes
    assert candidates[2].hash in parent_hashes


def test_exact_shape_transfer_seeds_cached_winner(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(2)
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([shape])
    insert_test_benchmark_event(
        db,
        shape_id=shape.id,
        candidate_hash=candidates[1].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=1.0,
    )

    proposed = propose_candidates(
        db,
        proposal="seed-random-gomea",
        num_random=0,
        gomea_count=0,
        target_shapes=[shape],
        transfer_shape_count=1,
        transfer_per_shape=1,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    ).selected

    transfer = [candidate for candidate in proposed if candidate.source == "transfer"]
    assert [candidate.hash for candidate in transfer] == [candidates[1].hash]


def test_multi_target_transfer_round_robins_target_neighborhoods(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(2)
    first = Shape(512, 128, 1, 256)
    second = Shape(8192, 8192, 1, 8192)
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([first, second])
    for shape, candidate, time_us in ((first, candidates[0], 1.0), (second, candidates[1], 1000.0)):
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=time_us,
        )

    proposed = propose_candidates(
        db,
        proposal="seed-random-gomea",
        num_random=0,
        gomea_count=0,
        target_shapes=[first, second],
        transfer_shape_count=2,
        transfer_per_shape=1,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    ).selected

    transfer = [candidate for candidate in proposed if candidate.source == "transfer"]
    assert {candidate.hash for candidate in transfer} == {candidate.hash for candidate in candidates}
    assert set().union(*(set(candidate.proposal_metadata["transfer_target_shape_ids"]) for candidate in transfer)) == {
        first.id,
        second.id,
    }
    assert all(candidate.proposal_metadata["transfer_source_shape_ids"] for candidate in transfer)


def test_nearest_shape_transfer_seeds_cached_winners(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(3)
    target = pilot_100_shapes()[0]
    near_shape = pilot_100_shapes()[1]
    far_shape = pilot_100_shapes()[-1]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([target, near_shape, far_shape])
    for shape, candidate, time_us in ((near_shape, candidates[1], 5.0), (far_shape, candidates[2], 3.0)):
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=time_us,
        )

    proposed = propose_candidates(
        db,
        proposal="seed-random-gomea",
        num_random=0,
        gomea_count=0,
        target_shapes=[target],
        transfer_shape_count=1,
        transfer_per_shape=1,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    ).selected

    transfer = [candidate for candidate in proposed if candidate.source == "transfer"]
    assert [candidate.hash for candidate in transfer] == [candidates[1].hash]
    assert transfer[0].parent_hashes == (candidates[1].hash,)


def test_exact_shape_elites_disable_nearest_shape_transfer(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(2)
    target = pilot_100_shapes()[0]
    exact_shape = pilot_100_shapes()[1]
    transfer_shape = pilot_100_shapes()[2]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([target, exact_shape, transfer_shape])
    insert_test_benchmark_event(
        db,
        shape_id=exact_shape.id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=1.0,
    )
    insert_test_benchmark_event(
        db,
        shape_id=transfer_shape.id,
        candidate_hash=candidates[1].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=0.5,
    )

    proposed = propose_candidates(
        db,
        proposal="gomea",
        gomea_count=4,
        target_shapes=[target],
        shape_id=exact_shape.id,
        transfer_shape_count=4,
        transfer_per_shape=1,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    ).selected

    assert all(candidate.source != "transfer" for candidate in proposed)
    assert {candidates[0].hash} & {parent for candidate in proposed for parent in candidate.parent_hashes}
    assert candidates[1].hash not in {parent for candidate in proposed for parent in candidate.parent_hashes}


def test_family_qd_builds_one_evidence_snapshot_per_proposal(tmp_path: Path, monkeypatch):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(8)
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([shape])
    for index, candidate in enumerate(candidates):
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=1.0 + index,
        )

    rank_calls = 0
    original_rank = db.rank_benchmarks

    def counted_rank(*args, **kwargs):
        nonlocal rank_calls
        rank_calls += 1
        return original_rank(*args, **kwargs)

    monkeypatch.setattr(db, "rank_benchmarks", counted_rank)
    proposed = propose_candidates(
        db,
        proposal="family-qd",
        num_random=2,
        local_count=2,
        de_count=2,
        gomea_count=4,
        elite_count=8,
        target_shapes=[shape],
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        adaptive_operators=True,
        adaptive_group_credit=True,
        adaptive_donor_selection=True,
        surrogate_pool_multiplier=2,
        linkage_min_samples=4,
        linkage_truncation_tau=1.0,
    ).selected

    assert proposed
    assert rank_calls == 1


def test_gomea_proposal_uses_learned_linkage_by_default_when_evidence_exists(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(6)
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([shape])
    for idx, candidate in enumerate(candidates):
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=1.0 + idx,
        )

    proposed = propose_candidates(
        db,
        proposal="gomea",
        gomea_count=4,
        elite_count=6,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        linkage_min_samples=3,
        linkage_truncation_tau=1.0,
    ).selected

    assert proposed
    assert all(candidate.source == "gomea" for candidate in proposed)
    assert all(cheap_constraints(candidate.canonical_params()) for candidate in proposed)


def test_gomea_proposal_accepts_learned_linkage_from_db_evidence(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(6)
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([shape])
    for idx, candidate in enumerate(candidates):
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=1.0 + idx,
        )

    proposed = propose_candidates(
        db,
        proposal="gomea",
        gomea_count=4,
        elite_count=6,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        learned_linkage=True,
        linkage_min_samples=3,
        linkage_truncation_tau=1.0,
    ).selected

    assert proposed
    assert all(candidate.source == "gomea" for candidate in proposed)
    assert all(cheap_constraints(candidate.canonical_params()) for candidate in proposed)


def test_evolutionary_proposal_uses_cached_elites(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(4)
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([shape])
    for idx, candidate in enumerate(candidates):
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=1.0 + idx,
        )

    proposed = propose_candidates(
        db,
        proposal="evolutionary",
        num_random=2,
        local_count=2,
        de_count=2,
        gomea_count=2,
        elite_count=4,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        seed=7,
    ).selected

    sources = {candidate.source for candidate in proposed}
    assert {"random", "mutation", "de", "gomea"} & sources == {"random", "mutation", "de", "gomea"}
    assert "seed" not in sources


def test_random_proposal_does_not_include_fixed_controls(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()

    proposed = propose_candidates(db, proposal="seed-random", num_random=12, seed=1151).selected

    assert {candidate.source for candidate in proposed} == {"random"}
    assert REFERENCE_CANDIDATE.hash not in {candidate.hash for candidate in proposed}


def test_random_proposals_respect_target_shape_rules(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    shape = Shape(m=8192, n=8192, batch=1, k=8192)

    proposed = propose_candidates(db, proposal="random", num_random=16, seed=20260701, target_shapes=[shape]).selected

    assert len(proposed) == 16
    assert all(cheap_constraints(candidate.canonical_params(), shape=shape) for candidate in proposed)


def test_multi_shape_random_proposal_keeps_scoped_specialists(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    small = Shape(m=512, n=128, batch=1, k=256)
    large = Shape(m=8192, n=8192, batch=1, k=8192)

    result = propose_candidates(
        db,
        proposal="random",
        num_random=64,
        seed=20260701,
        target_shapes=[small, large],
        scope_kind="cluster",
    )

    specialists = [
        candidate
        for candidate in result.selected
        if cheap_constraints(candidate.canonical_params(), shape=small)
        and not cheap_constraints(candidate.canonical_params(), shape=large)
    ]
    assert specialists
    assert result.scope.kind == "cluster"
    assert result.scope.shape_ids == (small.id, large.id)
    assert all(
        candidate.proposal_metadata["proposal_scope_kind"] == "cluster"
        and candidate.proposal_metadata["proposal_scope_shape_ids"] == [small.id, large.id]
        for candidate in result.generated
    )


def test_non_random_proposals_return_empty_without_evidence(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()

    for proposal in ("local", "de", "gomea"):
        proposed = propose_candidates(
            db,
            proposal=proposal,
            local_count=64,
            de_count=64,
            gomea_count=64,
            seed=1151,
        ).selected

        assert proposed == ()


def test_mixed_random_proposals_do_not_use_random_as_evolution_parents(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()

    for proposal in ("seed-random-local", "seed-random-de", "seed-random-gomea", "evolutionary"):
        proposed = propose_candidates(
            db,
            proposal=proposal,
            num_random=4,
            local_count=64,
            de_count=64,
            gomea_count=64,
            seed=1151,
        ).selected

        assert {candidate.source for candidate in proposed} == {"random"}


def test_family_qd_cold_start_uses_balanced_random_family_coverage(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    shape = Shape(m=8192, n=8192, batch=1, k=8192)

    proposed = propose_candidates(
        db,
        proposal="family-qd",
        num_random=16,
        local_count=8,
        de_count=4,
        gomea_count=12,
        target_shapes=[shape],
        seed=819200,
    ).selected

    assert {candidate.source for candidate in proposed} == {"random"}
    assert len(proposed) == 16
    assert {candidate.params["TransposeLDS"] for candidate in proposed} == {0, 2}


def test_family_qd_proposal_preserves_family_archive_leaders(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(4, seed=1151)
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([shape])
    for idx, candidate in enumerate(candidates):
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=1.0 + idx,
        )

    proposed = propose_candidates(
        db,
        proposal="family-qd",
        num_random=3,
        local_count=2,
        de_count=0,
        gomea_count=2,
        elite_count=4,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        target_shapes=[shape],
        seed=1151,
    ).selected

    proposed_hashes = {candidate.hash for candidate in proposed}
    assert candidates[0].hash in proposed_hashes
    assert "random" in {candidate.source for candidate in proposed}
    assert len(proposed_hashes) == len(proposed)


def test_family_qd_adaptive_operators_use_separate_semantic_arms(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    candidates = sample_candidates(12, seed=20260710)
    shape = Shape(m=8192, n=8192, batch=1, k=8192)
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([shape])
    for index, candidate in enumerate(candidates):
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="cached",
            status="ok",
            problem_type_hash=p_hash,
            benchmark_protocol_hash=b_hash,
            time_us=100.0 + index,
        )

    proposed = propose_candidates(
        db,
        proposal="family-qd",
        num_random=0,
        local_count=4,
        de_count=4,
        gomea_count=8,
        elite_count=12,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        target_shapes=[shape],
        adaptive_operators=True,
        adaptive_group_credit=True,
        micro_exhaustive_neighborhoods=True,
        adaptive_donor_selection=True,
        seed=20260711,
    ).selected

    parent_hashes = {candidate.hash for candidate in candidates}
    generated_sources = {candidate.source for candidate in proposed if candidate.hash not in parent_hashes}
    assert "mutation" not in generated_sources
    assert "gomea" not in generated_sources
    assert {"semantic-mutation", "de", "gomea-neighborhood", "gomea-mixing"} <= generated_sources
    metadata_by_source = {
        candidate.source: candidate.proposal_metadata for candidate in proposed if candidate.hash not in parent_hashes
    }
    assert "semantic_group" in metadata_by_source["semantic-mutation"]
    assert metadata_by_source["gomea-neighborhood"]["enumerated_neighborhood"] is True
    assert metadata_by_source["gomea-mixing"]["donor_mode"] in {"quality", "diverse", "random"}
