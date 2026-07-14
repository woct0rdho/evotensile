from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from evotensile.artifacts import register_artifact_bundle
from evotensile.candidate import Shape
from evotensile.database import EvoTensileDB, ValidationInsert
from evotensile.profile import DEFAULT_PROFILE
from evotensile.structured_runner import RunnablePair
from scripts.update_hipblaslt_gridbased_logic import (
    REFERENCE_SCHEMA_FILES,
    VARIANTS,
    Winner,
    _load_winners_from_assignments,
    _validate_winner_shape_set,
    update_logic_files,
)
from tests.helpers import REFERENCE_CANDIDATE, insert_test_benchmark_event, sample_candidates


def _profile_with_shapes(*shapes: Shape):
    return replace(DEFAULT_PROFILE, shapes_fn=lambda: list(shapes))


def test_explicit_deployment_assignment_is_not_replaced_by_database_rank(tmp_path: Path):
    shape = Shape(128, 128, 1, 128)
    profile = _profile_with_shapes(shape)
    candidates = sample_candidates(2)
    db = EvoTensileDB.connect(
        tmp_path / "selection.sqlite",
        environment_compatibility_tag=profile.environment_compatibility_tag,
    )
    db.init()
    db.register_candidates(candidates)
    db.register_shapes([shape])
    for candidate, time_us in zip(candidates, (1.0, 2.0), strict=True):
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="confirmation",
            status="ok",
            problem_type_hash=profile.problem_type_hash,
            benchmark_protocol_hash=profile.benchmark_protocol_hash(),
            time_us=time_us,
        )
        db.insert_validations(
            [
                ValidationInsert(
                    shape_id=shape.id,
                    candidate_hash=candidate.hash,
                    run_id="confirmation-validation",
                    status="passed",
                    problem_type_hash=profile.problem_type_hash,
                    validation_protocol_hash=profile.default_protocol.validation_protocol_hash(),
                    detail="PASSED",
                    source_kind="replay",
                )
            ]
        )

    winners = _load_winners_from_assignments(
        db,
        assignments={shape.id: candidates[1].hash},
        profile=profile,
        protocol=profile.default_protocol,
        min_samples=1,
    )

    ranked_leader = db.rank_benchmarks(
        problem_type_hash=profile.problem_type_hash,
        benchmark_protocol_hash=profile.benchmark_protocol_hash(),
        shape_id=shape.id,
        min_samples=1,
    )[0]
    assert ranked_leader.median_gflops is not None
    assert winners[0].candidate_hash == candidates[1].hash
    assert winners[0].median_gflops < ranked_leader.median_gflops


def test_logic_update_requires_complete_profile_shape_set():
    first = Shape(128, 128, 1, 128)
    second = Shape(256, 256, 1, 256)
    profile = _profile_with_shapes(first, second)
    winner = Winner(first.id, "cand_first", 1.0)

    with pytest.raises(ValueError, match="without winners"):
        _validate_winner_shape_set([], profile=profile, allow_partial=False)
    with pytest.raises(ValueError, match="missing 1 of 2 shapes"):
        _validate_winner_shape_set([winner], profile=profile, allow_partial=False)

    shape_set = _validate_winner_shape_set([winner], profile=profile, allow_partial=True)
    assert shape_set["missing"] == [second.id]


def test_logic_update_rejects_duplicate_and_extra_shapes():
    expected = Shape(128, 128, 1, 128)
    extra = Shape(512, 512, 1, 512)
    profile = _profile_with_shapes(expected)
    winner = Winner(expected.id, "cand_first", 1.0)

    with pytest.raises(ValueError, match="duplicate winner shapes"):
        _validate_winner_shape_set([winner, winner], profile=profile, allow_partial=True)
    with pytest.raises(ValueError, match="outside profile"):
        _validate_winner_shape_set(
            [winner, Winner(extra.id, "cand_extra", 2.0)],
            profile=profile,
            allow_partial=True,
        )


def _artifact_solution():
    params = REFERENCE_CANDIDATE.canonical_params()
    matrix_instruction = params["MatrixInstruction"]
    solution = {
        "SolutionIndex": 0,
        "KernelNameMin": "Kernel0",
        "MatrixInstruction": matrix_instruction[:4],
        "MIWaveTile": [matrix_instruction[5], matrix_instruction[6]],
        "MIWaveGroup": [matrix_instruction[7], matrix_instruction[8]],
    }
    for key, value in params.items():
        if key != "MatrixInstruction":
            solution[key] = value
    solution["MIArchVgpr"] = bool(solution["MIArchVgpr"])
    return solution


def _logic_export_fixture(
    tmp_path: Path,
    *,
    validation: bool,
    artifact: bool,
    shape_count: int = 1,
):
    shapes = (
        Shape(128, 128, 1, 128),
        Shape(256, 256, 1, 256),
    )[:shape_count]
    profile = _profile_with_shapes(*shapes)
    db_path = tmp_path / "logic.sqlite"
    db = EvoTensileDB.connect(
        db_path,
        environment_compatibility_tag=profile.environment_compatibility_tag,
    )
    db.init()
    db.register_candidates([REFERENCE_CANDIDATE])
    db.register_shapes(list(shapes))
    for shape in shapes:
        insert_test_benchmark_event(
            db,
            shape_id=shape.id,
            candidate_hash=REFERENCE_CANDIDATE.hash,
            run_id="confirmation",
            status="ok",
            problem_type_hash=profile.problem_type_hash,
            benchmark_protocol_hash=profile.benchmark_protocol_hash(),
            time_us=1.0,
            solution_index=0,
        )
    db.insert_validations(
        [
            ValidationInsert(
                shape_id=shape.id,
                candidate_hash=REFERENCE_CANDIDATE.hash,
                run_id="confirmation-validation",
                status="passed" if validation else "failed",
                problem_type_hash=profile.problem_type_hash,
                validation_protocol_hash=profile.default_protocol.validation_protocol_hash(),
                detail="PASSED" if validation else "FAILED",
                solution_index=0,
                source_kind="replay",
            )
            for shape in shapes
        ]
    )

    solution = _artifact_solution()
    logic_dir = tmp_path / "logic"
    logic_dir.mkdir()
    template = [{}, {}, {}, {}, {"UseE": False}, [solution], {}, []]
    template_text = yaml.safe_dump(template, sort_keys=False)
    (logic_dir / VARIANTS["hhs"].filename).write_text(template_text, encoding="utf-8")
    (logic_dir / REFERENCE_SCHEMA_FILES[0]).write_text(template_text, encoding="utf-8")
    if artifact:
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        solution_yaml = build_dir / "00_Final.yaml"
        artifact_solution = {**solution, "StaggerUStride": float(solution["StaggerUStride"])}
        solution_yaml.write_text(yaml.safe_dump([{}, {}, artifact_solution], sort_keys=False), encoding="utf-8")
        library_dir = build_dir / "library" / str(profile.library_logic["ArchitectureName"])
        library_dir.mkdir(parents=True)
        (library_dir / "TensileLibrary.yaml").write_text("solutions: []\n", encoding="utf-8")
        (library_dir / "Kernels.hsaco").write_bytes(b"code")
        register_artifact_bundle(
            db,
            problem_type_hash=profile.problem_type_hash,
            runnable_pairs=[
                RunnablePair(
                    shape_id=shape.id,
                    candidate_hash=REFERENCE_CANDIDATE.hash,
                    problem_index=index,
                    requested_solution_index=0,
                    library_solution_index=0,
                    manifest_solution_index=0,
                )
                for index, shape in enumerate(shapes)
            ],
            build_run_id="build",
            build_output_dir=build_dir,
            library_dir=library_dir,
            solution_yaml_paths=[solution_yaml],
            manifest_path=None,
        )
    return db_path, profile, logic_dir


def test_logic_export_requires_current_validation(tmp_path: Path):
    db_path, profile, logic_dir = _logic_export_fixture(tmp_path, validation=False, artifact=False)

    with pytest.raises(ValueError, match="without winners"):
        update_logic_files(
            db_path=db_path,
            profile=profile,
            protocol=profile.default_protocol,
            min_samples=1,
            logic_dir=logic_dir,
            variant_names=["hhs"],
        )


def test_logic_export_requires_complete_registered_artifact(tmp_path: Path):
    db_path, profile, logic_dir = _logic_export_fixture(tmp_path, validation=True, artifact=False)

    with pytest.raises(ValueError, match="lack complete registered artifacts"):
        update_logic_files(
            db_path=db_path,
            profile=profile,
            protocol=profile.default_protocol,
            min_samples=1,
            logic_dir=logic_dir,
            variant_names=["hhs"],
        )


def test_logic_export_writes_complete_bank_coverage_from_explicit_assignments(tmp_path: Path):
    db_path, profile, logic_dir = _logic_export_fixture(
        tmp_path,
        validation=True,
        artifact=True,
        shape_count=2,
    )
    destination = tmp_path / "complete-bank"

    result = update_logic_files(
        db_path=db_path,
        profile=profile,
        protocol=profile.default_protocol,
        min_samples=1,
        logic_dir=logic_dir,
        variant_names=["hhs"],
        destination_dir=destination,
        winner_assignments={shape.id: REFERENCE_CANDIDATE.hash for shape in profile.shapes()},
    )

    assert result["missing_shape_ids"] == []
    assert result["shape_count"] == 2
    assert result["candidate_count"] == 1
    assert result["registered_artifact_count"] == 2
    assert result["files"]["hhs"]["solution_count"] == 1
    assert result["files"]["hhs"]["exact_mapping_count"] == 2


def test_logic_export_uses_registered_artifact_and_writes_output(tmp_path: Path):
    db_path, profile, logic_dir = _logic_export_fixture(tmp_path, validation=True, artifact=True)
    destination = tmp_path / "staged"

    result = update_logic_files(
        db_path=db_path,
        profile=profile,
        protocol=profile.default_protocol,
        min_samples=1,
        logic_dir=logic_dir,
        variant_names=["hhs"],
        destination_dir=destination,
        winner_assignments={profile.shapes()[0].id: REFERENCE_CANDIDATE.hash},
    )

    assert result["winner_source"] == "deployment-selection"
    assert result["registered_artifact_count"] == 1
    assert result["shape_count"] == 1
    output_path = destination / VARIANTS["hhs"].filename
    assert output_path.exists()
    output = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert type(output[5][0]["StaggerUStride"]) is int
