from pathlib import Path

import yaml

from evotensile.candidate import Candidate
from evotensile.manifest import write_manifest
from evotensile.search_space import documented_winner_candidate
from evotensile.shapes import Shape
from evotensile.solution_mapping import build_solution_candidate_mapper, solution_matches_candidate
from evotensile.tensilelite_keys import (
    EXACT_KEY,
    KERNEL_NAME_MIN_KEY,
    MATRIX_INSTRUCTION_KEY,
    MI_WAVE_GROUP_KEY,
    MI_WAVE_TILE_KEY,
    PROBLEM_SIZES_KEY,
    SOLUTION_INDEX_KEY,
    STORE_VECTOR_WIDTH_KEY,
    WORK_GROUP_KEY,
)


def _final_solution_from_candidate(candidate: Candidate, *, solution_index: int = 0) -> dict:
    params = candidate.canonical_params()
    mi = params[MATRIX_INSTRUCTION_KEY]
    solution = {
        SOLUTION_INDEX_KEY: solution_index,
        KERNEL_NAME_MIN_KEY: f"Kernel{solution_index}",
        MATRIX_INSTRUCTION_KEY: mi[:4],
        MI_WAVE_TILE_KEY: [mi[5], mi[6]],
        MI_WAVE_GROUP_KEY: [mi[7], mi[8]],
        WORK_GROUP_KEY: [32, 4, 1],
        STORE_VECTOR_WIDTH_KEY: 1,
    }
    for key, value in params.items():
        if key in {MATRIX_INSTRUCTION_KEY, WORK_GROUP_KEY, STORE_VECTOR_WIDTH_KEY}:
            continue
        solution[key] = value
    solution["MIArchVgpr"] = bool(solution["MIArchVgpr"])
    return solution


def _write_solution_yaml(path: Path, shape: Shape, solution: dict) -> None:
    data = [
        {"MinimumRequiredVersion": "5.0.0"},
        {
            PROBLEM_SIZES_KEY: [
                {EXACT_KEY: [shape.m, shape.n, shape.batch, shape.k, shape.m, shape.m, shape.k, shape.k]}
            ]
        },
        {"BiasTypeArgs": [[4]]},
        {"ActivationArgs": [[{"Enum": "None"}]]},
        solution,
    ]
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_solution_mapping_uses_final_yaml_not_group_order(tmp_path: Path):
    base = documented_winner_candidate()
    deduped = Candidate({**base.canonical_params(), STORE_VECTOR_WIDTH_KEY: 1}, source="dedup_equivalent")
    rejected = Candidate({**base.canonical_params(), "DepthU": 32}, source="rejected_or_different")
    shape = Shape(512, 128, 1, 256)
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [base, rejected, deduped], [shape])
    final_yaml = tmp_path / "00_Final.yaml"
    solution = _final_solution_from_candidate(base, solution_index=0)
    _write_solution_yaml(final_yaml, shape, solution)

    assert solution_matches_candidate(solution, base.canonical_params())
    assert solution_matches_candidate(solution, deduped.canonical_params())
    assert not solution_matches_candidate(solution, rejected.canonical_params())

    from evotensile.manifest import read_manifest

    mapper = build_solution_candidate_mapper(read_manifest(manifest), [final_yaml])
    entries = mapper.entries_for(shape_id=shape.id, solution_index=0)
    assert {entry.candidate_hash for entry in entries} == {base.hash, deduped.hash}


def test_solution_mapping_ignores_derived_expand_pointer_swap():
    candidate = documented_winner_candidate()
    solution = _final_solution_from_candidate(candidate)
    solution["ExpandPointerSwap"] = True

    assert candidate.canonical_params()["ExpandPointerSwap"] == 0
    assert solution_matches_candidate(solution, candidate.canonical_params())


def test_solution_mapping_ignores_tlds2_derived_buffer_and_local_read_fields():
    candidate = Candidate(
        {
            **documented_winner_candidate().canonical_params(),
            "1LDSBuffer": 0,
            "PrefetchGlobalRead": 2,
            "PrefetchLocalRead": 0,
            "TransposeLDS": 2,
            "VectorWidthB": 1,
            "LdsBlockSizePerPadA": 128,
            "LdsBlockSizePerPadB": 128,
            "LdsPadA": 8,
            "LdsPadB": 8,
        },
        source="tlds2",
    )
    solution = _final_solution_from_candidate(candidate)
    solution["1LDSBuffer"] = 1
    solution["PrefetchLocalRead"] = 1

    assert candidate.canonical_params()["1LDSBuffer"] == 0
    assert candidate.canonical_params()["PrefetchLocalRead"] == 0
    assert solution_matches_candidate(solution, candidate.canonical_params())


def test_solution_mapping_ignores_inactive_stagger_derived_fields():
    candidate = Candidate(
        {
            **documented_winner_candidate().canonical_params(),
            "StaggerU": 0,
            "StaggerUMapping": 1,
            "StaggerUStride": 256,
        },
        source="inactive_stagger",
    )
    solution = _final_solution_from_candidate(candidate)
    solution["StaggerUMapping"] = 0
    solution["StaggerUStride"] = 64.0

    assert solution_matches_candidate(solution, candidate.canonical_params())


def test_solution_mapping_keeps_active_stagger_fields_strict():
    candidate = Candidate(
        {
            **documented_winner_candidate().canonical_params(),
            "StaggerU": 8,
            "StaggerUMapping": 1,
            "StaggerUStride": 256,
        },
        source="active_stagger",
    )
    solution = _final_solution_from_candidate(candidate)
    solution["StaggerUMapping"] = 0

    assert not solution_matches_candidate(solution, candidate.canonical_params())
