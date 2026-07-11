import os
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from evotensile.artifacts import register_candidate_artifacts
from evotensile.candidate import Shape
from evotensile.database import EvoTensileDB, ValidationInsert
from evotensile.profile import DEFAULT_PROFILE
from evotensile.structured_runner import RunnablePair
from scripts.update_hipblaslt_gridbased_logic import (
    VARIANTS,
    Winner,
    _validate_winner_shape_set,
    _write_files_transactionally,
    update_logic_files,
)
from tests.helpers import REFERENCE_CANDIDATE


def _profile_with_shapes(*shapes: Shape):
    return replace(DEFAULT_PROFILE, shapes_fn=lambda: list(shapes))


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


def test_transactional_logic_write_restores_all_files_on_failure(tmp_path: Path, monkeypatch):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("old first\n", encoding="utf-8")
    second.write_text("old second\n", encoding="utf-8")
    real_replace = os.replace

    def fail_second_commit(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.suffix == ".tmp" and destination_path == second:
            raise OSError("injected commit failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_commit)
    with pytest.raises(OSError, match="injected commit failure"):
        _write_files_transactionally({first: "new first\n", second: "new second\n"})

    assert first.read_text(encoding="utf-8") == "old first\n"
    assert second.read_text(encoding="utf-8") == "old second\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["first.yaml", "second.yaml"]


def test_transactional_logic_write_commits_complete_set(tmp_path: Path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "nested" / "second.yaml"

    _write_files_transactionally({first: "new first\n", second: "new second\n"})

    assert first.read_text(encoding="utf-8") == "new first\n"
    assert second.read_text(encoding="utf-8") == "new second\n"


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


def _logic_export_fixture(tmp_path: Path, *, validation: bool, artifact: bool):
    shape = Shape(128, 128, 1, 128)
    profile = _profile_with_shapes(shape)
    db_path = tmp_path / "logic.sqlite"
    db = EvoTensileDB.connect(db_path)
    db.init()
    db.register_candidates([REFERENCE_CANDIDATE])
    db.register_shapes([shape])
    db.insert_evaluation(
        shape_id=shape.id,
        candidate_hash=REFERENCE_CANDIDATE.hash,
        run_id="screening",
        status="ok",
        problem_type_hash=profile.problem_type_hash,
        benchmark_protocol_hash=profile.benchmark_protocol_hash(),
        time_us=1.0,
        validation="PASSED prior_validation",
        solution_index=0,
    )
    if validation:
        db.insert_validations(
            [
                ValidationInsert(
                    shape_id=shape.id,
                    candidate_hash=REFERENCE_CANDIDATE.hash,
                    run_id="validation",
                    status="passed",
                    problem_type_hash=profile.problem_type_hash,
                    validation_protocol_hash=profile.default_protocol.validation_protocol_hash(),
                    detail="PASSED",
                    solution_index=0,
                )
            ]
        )

    solution = _artifact_solution()
    logic_dir = tmp_path / "logic"
    logic_dir.mkdir()
    template = [{}, {}, {}, {}, {"UseE": False}, [solution], {}, []]
    (logic_dir / VARIANTS["hhs"].filename).write_text(
        yaml.safe_dump(template, sort_keys=False),
        encoding="utf-8",
    )
    if artifact:
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        solution_yaml = build_dir / "00_Final.yaml"
        solution_yaml.write_text(yaml.safe_dump([{}, {}, solution], sort_keys=False), encoding="utf-8")
        library_dir = build_dir / "library" / str(profile.library_logic["ArchitectureName"])
        library_dir.mkdir(parents=True)
        (library_dir / "TensileLibrary.yaml").write_text("solutions: []\n", encoding="utf-8")
        (library_dir / "Kernels.hsaco").write_bytes(b"code")
        register_candidate_artifacts(
            db,
            problem_type_hash=profile.problem_type_hash,
            runnable_pairs=[
                RunnablePair(
                    shape_id=shape.id,
                    candidate_hash=REFERENCE_CANDIDATE.hash,
                    problem_index=0,
                    requested_solution_index=0,
                    library_solution_index=0,
                    manifest_solution_index=0,
                )
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

    with pytest.raises(ValueError, match="lack current passed validation"):
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


def test_logic_export_uses_registered_artifact_and_stages_complete_output(tmp_path: Path):
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
    )

    assert result["registered_artifact_count"] == 1
    assert result["shape_count"] == 1
    assert (destination / VARIANTS["hhs"].filename).exists()
