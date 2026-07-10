import pytest

from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.search.family import (
    family_descriptor,
    family_descriptor_counts,
    family_stratified_random_candidates,
    load_family_archive,
)
from evotensile.shapes import pilot_100_shapes
from tests.helpers import REFERENCE_CANDIDATE, sample_candidates


def test_nt_hhs_family_descriptor_is_stable_for_reference_candidate():
    descriptor = family_descriptor(REFERENCE_CANDIDATE)

    assert descriptor.profile == "gfx1151-nt-hhs"
    assert descriptor.version == "nt_hhs_v2"
    assert descriptor.key.startswith("gfx1151-nt-hhs:nt_hhs_v2:")
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


def test_family_descriptor_rejects_unknown_profile():
    with pytest.raises(ValueError, match="unsupported family descriptor profile"):
        family_descriptor(REFERENCE_CANDIDATE, profile="unknown")


def test_family_stratified_random_candidates_balance_target_aspect_and_broad_cells(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "families.sqlite")
    db.init()
    shape = pilot_100_shapes()[0]

    candidates = family_stratified_random_candidates(
        db,
        16,
        seed=1151,
        target_shapes=[shape],
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
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
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    initial = family_stratified_random_candidates(
        db,
        8,
        seed=1151,
        target_shapes=[shape],
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )
    failed_family = family_descriptor(initial[0])
    db.register_candidates(initial)
    db.register_shapes([shape])
    db.insert_evaluation(
        shape_id=shape.id,
        candidate_hash=initial[0].hash,
        run_id="failed",
        status="validation_fail",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )

    followup = family_stratified_random_candidates(
        db,
        32,
        seed=1152,
        target_shapes=[shape],
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
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
            db.insert_evaluation(
                shape_id=shape.id,
                candidate_hash=candidate.hash,
                run_id="cached",
                status="ok",
                problem_type_hash=p_hash,
                benchmark_protocol_hash=b_hash,
                time_us=1.0 + idx,
                validation="PASSED",
            )
    db.insert_evaluation(
        shape_id=shapes[0].id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="validation_fail",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
    )

    archive = load_family_archive(
        db,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        shapes=shapes,
    )

    assert archive
    assert archive[0].leader_candidate_hash == candidates[0].hash
    assert archive[0].aggregate_score == 0.0
    assert archive[0].samples == 2
    assert archive[0].shape_count == 2
    assert archive[0].status_counts["ok"] >= 2
    assert archive[0].status_counts["validation_fail"] == 1


def test_load_family_archive_filters_protocol_and_min_samples(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "families.sqlite")
    db.init()
    candidates = sample_candidates(2, seed=1152)
    shape = pilot_100_shapes()[0]
    p_hash = DEFAULT_PROFILE.problem_type_hash
    b_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    db.register_candidates(candidates)
    db.register_shapes([shape])
    db.insert_evaluation(
        shape_id=shape.id,
        candidate_hash=candidates[0].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        time_us=1.0,
        validation="PASSED",
    )
    db.insert_evaluation(
        shape_id=shape.id,
        candidate_hash=candidates[1].hash,
        run_id="cached",
        status="ok",
        problem_type_hash=p_hash,
        benchmark_protocol_hash="other",
        time_us=0.5,
        validation="PASSED",
    )

    archive = load_family_archive(
        db,
        problem_type_hash=p_hash,
        benchmark_protocol_hash=b_hash,
        shapes=[shape],
        min_samples=2,
    )

    assert archive == []
