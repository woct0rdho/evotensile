from pathlib import Path

import pytest

from evotensile.artifacts import load_artifact_mappings, register_artifact_bundle
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.shapes import shape_from_id
from evotensile.structured_runner import RunnablePair
from tests.helpers import REFERENCE_CANDIDATE


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
        candidate_hash=REFERENCE_CANDIDATE.hash,
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
    db.register_candidates([REFERENCE_CANDIDATE])
    db.register_shapes([shape_from_id(pair.shape_id)])

    bundle_id = register_artifact_bundle(
        db,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        runnable_pairs=[pair],
        build_run_id="build_run",
        build_output_dir=build_dir,
        library_dir=library_dir,
        solution_yaml_paths=[solution_yaml],
        manifest_path=manifest,
    )
    loaded = load_artifact_mappings(
        db,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        shape_ids=[pair.shape_id],
        candidate_hashes=[pair.candidate_hash],
    )

    assert bundle_id is not None
    assert loaded[(pair.shape_id, pair.candidate_hash)].runnable_pair == pair
    assert loaded[(pair.shape_id, pair.candidate_hash)].solution_yaml_paths == (solution_yaml,)
    assert db.counts()["artifact_bundles"] == 1
    assert db.counts()["artifact_solution_yamls"] == 1
    assert db.counts()["artifact_mappings"] == 1

    code_object.write_bytes(b"changed code object")
    assert (
        load_artifact_mappings(
            db,
            problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
            shape_ids=[pair.shape_id],
            candidate_hashes=[pair.candidate_hash],
        )
        == {}
    )


def test_artifact_registry_shares_one_bundle_across_pair_mappings(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "artifacts.sqlite")
    db.init()
    build_dir, solution_yaml, manifest, library_dir, _, pair = _artifact_inputs(tmp_path)
    second_pair = RunnablePair(
        shape_id="m256_n128_b1_k128",
        candidate_hash=REFERENCE_CANDIDATE.hash,
        problem_index=1,
        requested_solution_index=3,
        library_solution_index=2,
        manifest_solution_index=3,
    )
    db.register_candidates([REFERENCE_CANDIDATE])
    db.register_shapes([shape_from_id(pair.shape_id), shape_from_id(second_pair.shape_id)])

    register_artifact_bundle(
        db,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        runnable_pairs=[pair, second_pair],
        build_run_id="build_run",
        build_output_dir=build_dir,
        library_dir=library_dir,
        solution_yaml_paths=[solution_yaml],
        manifest_path=manifest,
    )

    assert db.counts()["artifact_bundles"] == 1
    assert db.counts()["artifact_solution_yamls"] == 1
    assert db.counts()["artifact_mappings"] == 2
    assert set(load_artifact_mappings(db, problem_type_hash=DEFAULT_PROFILE.problem_type_hash)) == {
        (pair.shape_id, pair.candidate_hash),
        (second_pair.shape_id, second_pair.candidate_hash),
    }


def test_artifact_registry_rejects_missing_solution_yaml(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "artifacts.sqlite")
    db.init()
    build_dir, solution_yaml, manifest, library_dir, _, pair = _artifact_inputs(tmp_path)
    solution_yaml.unlink()

    with pytest.raises(FileNotFoundError):
        register_artifact_bundle(
            db,
            problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
            runnable_pairs=[pair],
            build_run_id="build_run",
            build_output_dir=build_dir,
            library_dir=library_dir,
            solution_yaml_paths=[solution_yaml],
            manifest_path=manifest,
        )
