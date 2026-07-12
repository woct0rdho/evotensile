from evotensile.database import EvoTensileDB, ValidationInsert
from evotensile.profile import DEFAULT_PROFILE
from evotensile.search.encoding import candidate_to_genome, hamming_distance
from evotensile.search.evidence import load_proposal_evidence_snapshot
from evotensile.search.family import (
    family_descriptor,
    family_descriptor_counts,
    family_stratified_random_candidates,
    load_family_archive,
)
from evotensile.search.grid_evidence import GRID_OBJECTIVES, GridObjective
from evotensile.search_space import DOMAINS, make_candidate
from evotensile.shapes import pilot_100_shapes
from tests.helpers import REFERENCE_CANDIDATE, insert_test_benchmark_event, sample_candidates


def _evidence(db: EvoTensileDB, shapes):
    return load_proposal_evidence_snapshot(
        db,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        shapes=shapes,
    )


def test_nt_hhs_family_descriptor_is_stable_for_reference_candidate():
    descriptor = family_descriptor(REFERENCE_CANDIDATE)

    assert descriptor.profile == "gfx1151-nt-hhs"
    assert descriptor.key.startswith("gfx1151-nt-hhs:TileAreaLog2=")
    assert descriptor.as_dict()["fields"] == {
        "TileAreaLog2": 10,
        "TileAspect": "balanced",
        "TransposeLDS": 0,
        "GlobalSplitU": 1,
    }


def test_family_descriptor_accepts_candidate_params_mapping():
    from_candidate = family_descriptor(REFERENCE_CANDIDATE)
    from_params = family_descriptor(REFERENCE_CANDIDATE.canonical_params())

    assert from_params == from_candidate


def test_family_descriptor_counts_candidates_by_key():
    candidates = sample_candidates(5, seed=1151)
    counts = family_descriptor_counts([*candidates, candidates[0]])

    assert sum(counts.values()) == 6
    assert counts[family_descriptor(candidates[0]).key] >= 2


def test_family_stratified_random_candidates_balance_target_aspect_and_broad_cells(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "families.sqlite")
    db.init()
    shape = pilot_100_shapes()[0]

    candidates = family_stratified_random_candidates(
        _evidence(db, [shape]),
        16,
        seed=1151,
        target_shapes=[shape],
    )
    fields = [dict(family_descriptor(candidate).fields) for candidate in candidates]

    assert len(candidates) == 16
    assert sum(field["TileAspect"] == "m_major" for field in fields) >= 8
    assert {field["TransposeLDS"] for field in fields} == {0, 2}
    family_counts = family_descriptor_counts(candidates)
    assert len(family_counts) == 8
    assert set(family_counts.values()) == {2}


def test_family_stratified_random_candidates_retry_failed_family(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "families.sqlite")
    db.init()
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    initial = family_stratified_random_candidates(
        _evidence(db, [shape]),
        8,
        seed=1151,
        target_shapes=[shape],
    )
    failed_family = family_descriptor(initial[0])
    db.register_candidates(initial)
    db.register_shapes([shape])
    db.insert_validations(
        [
            ValidationInsert(
                shape_id=shape.id,
                candidate_hash=initial[0].hash,
                run_id="failed",
                status="failed",
                problem_type_hash=p_hash,
                validation_protocol_hash=DEFAULT_PROFILE.default_protocol.validation_protocol_hash(),
                detail="FAILED",
                source_kind="replay",
            )
        ]
    )

    followup = family_stratified_random_candidates(
        _evidence(db, [shape]),
        32,
        seed=1152,
        target_shapes=[shape],
    )

    assert any(family_descriptor(candidate) == failed_family for candidate in followup)
    assert all(candidate.hash != initial[0].hash for candidate in followup)


def test_load_family_archive_keeps_best_leader_per_family(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "families.sqlite")
    db.init()
    candidates = sample_candidates(3, seed=1151)
    shapes = pilot_100_shapes()[:2]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes(shapes)
    for shape in shapes:
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
    db.insert_validations(
        [
            ValidationInsert(
                shape_id=shapes[0].id,
                candidate_hash=candidates[0].hash,
                run_id="cached",
                status="failed",
                problem_type_hash=p_hash,
                validation_protocol_hash=DEFAULT_PROFILE.default_protocol.validation_protocol_hash(),
                detail="FAILED",
                source_kind="replay",
            )
        ]
    )

    archive = load_family_archive(
        _evidence(db, shapes),
        shapes=shapes,
        objective=GridObjective.GENERALIST,
    )

    assert archive
    assert archive[0].leader_candidate_hash == candidates[1].hash
    assert archive[0].aggregate_score == 0.25
    assert archive[0].samples == 2
    assert archive[0].shape_count == 2
    assert archive[0].status_counts["ok"] >= 2


def test_family_archive_objectives_distinguish_sparse_specialists_and_broad_generalists(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "families.sqlite")
    db.init()
    shapes = pilot_100_shapes()[:2]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    specialist_params = REFERENCE_CANDIDATE.canonical_params()
    specialist_params["StaggerU"] = 0
    generalist_params = REFERENCE_CANDIDATE.canonical_params()
    generalist_params["StaggerU"] = 8
    specialist = make_candidate(specialist_params, source="archive-test")
    generalist = make_candidate(generalist_params, source="archive-test")
    db.register_candidates([specialist, generalist])
    db.register_shapes(shapes)
    for shape, candidate, time_us in (
        (shapes[0], specialist, 1.0),
        (shapes[0], generalist, 2.0),
        (shapes[1], generalist, 1.0),
    ):
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

    snapshot = _evidence(db, shapes)
    leaders = {
        objective: load_family_archive(
            snapshot,
            shapes=shapes,
            objective=objective,
        )[0]
        for objective in GRID_OBJECTIVES
    }
    weighted_generalist = load_family_archive(
        snapshot,
        shapes=shapes,
        objective=GridObjective.GENERALIST,
        shape_weights={shapes[0].id: 1.8, shapes[1].id: 0.2},
    )[0]

    assert leaders[GridObjective.SPECIALIST].leader_candidate_hash == specialist.hash
    assert leaders[GridObjective.GENERALIST].leader_candidate_hash == generalist.hash
    assert leaders[GridObjective.COVERAGE].leader_candidate_hash == generalist.hash
    assert leaders[GridObjective.UNCERTAINTY].leader_candidate_hash == specialist.hash
    assert leaders[GridObjective.GENERALIST].coverage_fraction == 1.0
    assert leaders[GridObjective.UNCERTAINTY].unresolved_shape_count == 1
    assert weighted_generalist.leader_candidate_hash == specialist.hash
    assert weighted_generalist.shape_weighted


def test_load_family_archive_keeps_diverse_quality_bounded_elites_per_family(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "families.sqlite")
    db.init()
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    candidates = []
    for index, stagger_u in enumerate(DOMAINS["StaggerU"][:5]):
        params = REFERENCE_CANDIDATE.canonical_params()
        params["StaggerU"] = stagger_u
        params["StaggerUMapping"] = index % 2
        candidates.append(make_candidate(params, source="archive-test"))
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

    archive = load_family_archive(
        _evidence(db, [shape]),
        shapes=[shape],
        objective=GridObjective.SPECIALIST,
        elites_per_family=3,
    )

    assert len(archive) == 3
    assert archive[0].leader_candidate_hash == candidates[0].hash
    assert [entry.family_rank for entry in archive] == [1, 2, 3]
    assert all(entry.descriptor == archive[0].descriptor for entry in archive)
    assert all(entry.observed_candidate_count == 5 for entry in archive)
    assert archive[1].novelty_distance > 0
    assert len({entry.leader_candidate_hash for entry in archive}) == 3
    selected_genomes = [candidate_to_genome(entry.leader) for entry in archive]
    assert min(hamming_distance(selected_genomes[0], genome) for genome in selected_genomes[1:]) > 0


def test_load_family_archive_filters_protocol_and_min_samples(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "families.sqlite")
    db.init()
    candidates = sample_candidates(2, seed=1152)
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([shape])
    insert_test_benchmark_event(
        db,
        shape_id=shape.id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=1.0,
    )
    insert_test_benchmark_event(
        db,
        shape_id=shape.id,
        candidate_hash=candidates[1].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash="other",
        time_us=0.5,
    )

    archive = load_family_archive(
        _evidence(db, [shape]),
        shapes=[shape],
        min_samples=2,
        objective=GridObjective.SPECIALIST,
    )

    assert archive == []
