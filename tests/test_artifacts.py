from pathlib import Path

import pytest

from evotensile.artifacts import load_candidate_artifacts, register_candidate_artifacts
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.structured_runner import RunnablePair


def _artifact_inputs(tmp_path: Path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    solution_yaml = build_dir / "00_Final.yaml"
    solution_yaml.write_text("[]\n", encoding="utf-8")
    manifest = build_dir / "config.manifest.csv"
    manifest.write_text("candidate_hash\n", encoding="utf-8")
    library_dir = build_dir / "library" / str(DEFAULT_PROFILE.library_logic["ArchitectureName"])
    library_dir.mkdir(parents=True)
    (library_dir / "TensileLibrary.yaml").write_text("solutions: []\n", encoding="utf-8")
    code_object = library_dir / "Kernels.hsaco"
    code_object.write_bytes(b"first code object")
    pair = RunnablePair(
        shape_id="m128_n128_b1_k128",
        candidate_hash="cand_test",
        problem_index=0,
        requested_solution_index=3,
        library_solution_index=2,
        manifest_solution_index=3,
    )
    return build_dir, solution_yaml, manifest, library_dir, code_object, pair


def test_artifact_registry_loads_only_content_verified_complete_entries(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "artifacts.sqlite")
    db.init()
    build_dir, solution_yaml, manifest, library_dir, code_object, pair = _artifact_inputs(tmp_path)

    inserts = register_candidate_artifacts(
        db,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        runnable_pairs=[pair],
        build_run_id="build_run",
        build_output_dir=build_dir,
        library_dir=library_dir,
        solution_yaml_paths=[solution_yaml],
        manifest_path=manifest,
    )
    loaded = load_candidate_artifacts(
        db,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        shape_ids=[pair.shape_id],
        candidate_hashes=[pair.candidate_hash],
    )

    assert len(inserts) == 1
    assert loaded[(pair.shape_id, pair.candidate_hash)].runnable_pair == pair
    assert loaded[(pair.shape_id, pair.candidate_hash)].solution_yaml_paths == (solution_yaml,)
    assert db.counts()["candidate_artifacts"] == 1

    code_object.write_bytes(b"changed code object")
    assert (
        load_candidate_artifacts(
            db,
            problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
            shape_ids=[pair.shape_id],
            candidate_hashes=[pair.candidate_hash],
        )
        == {}
    )


def test_artifact_registry_rejects_missing_solution_yaml(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "artifacts.sqlite")
    db.init()
    build_dir, solution_yaml, manifest, library_dir, _, pair = _artifact_inputs(tmp_path)
    solution_yaml.unlink()

    with pytest.raises(FileNotFoundError):
        register_candidate_artifacts(
            db,
            problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
            runnable_pairs=[pair],
            build_run_id="build_run",
            build_output_dir=build_dir,
            library_dir=library_dir,
            solution_yaml_paths=[solution_yaml],
            manifest_path=manifest,
        )
