import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from evotensile.database import BaselineSelectionInsert, EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.search.replay import load_db_oracle_matrix
from evotensile.shapes import pilot_100_shapes
from scripts.merge_compatible_databases import merge_compatible_databases
from tests.helpers import insert_test_benchmark_event, sample_candidates


def _evidence_db(
    path: Path,
    *,
    candidate,
    shape,
    protocol_hash: str,
    run_id: str,
    problem_type_hash: str = DEFAULT_PROFILE.problem_type_hash,
):
    db = EvoTensileDB.connect(path)
    db.init()
    db.register_candidates([candidate])
    db.register_shapes([shape])
    insert_test_benchmark_event(
        db,
        shape_id=shape.id,
        candidate_hash=candidate.hash,
        run_id=run_id,
        status="ok",
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=protocol_hash,
        samples_us=(10.0, 11.0),
    )
    return db


def test_merge_compatible_databases_preserves_exact_evidence_and_provenance(tmp_path: Path):
    shape = pilot_100_shapes()[0]
    base_candidate, overlay_candidate = sample_candidates(2)
    protocol_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    base = _evidence_db(
        tmp_path / "base.sqlite",
        candidate=base_candidate,
        shape=shape,
        protocol_hash=protocol_hash,
        run_id="base-run",
    )
    overlay = _evidence_db(
        tmp_path / "overlay.sqlite",
        candidate=overlay_candidate,
        shape=shape,
        protocol_hash=protocol_hash,
        run_id="overlay-run",
    )
    overlay.record_baseline_discovery(
        [
            BaselineSelectionInsert(
                shape=shape,
                candidate=overlay_candidate,
                hipblaslt_solution_index=7,
            )
        ],
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        context={"baseline_label": "anchored-test"},
        duration_s=0.1,
    )
    output = tmp_path / "merged.sqlite"

    manifest = merge_compatible_databases(
        output,
        [base.path, overlay.path],
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=protocol_hash,
    )

    oracle = load_db_oracle_matrix(
        output,
        shapes=[shape],
        benchmark_protocol_hash=protocol_hash,
    )
    merged = EvoTensileDB.connect(output)
    discoveries = merged.baseline_discoveries(baseline_label="anchored-test")
    with closing(sqlite3.connect(output)) as connection:
        stored_manifest = json.loads(
            connection.execute(
                "SELECT metadata_value FROM database_metadata WHERE metadata_key = 'merged_source_manifest'"
            ).fetchone()[0]
        )
        source_refs = {row[0] for row in connection.execute("SELECT source_ref FROM evidence_sources")}

    assert set(oracle) == {
        (shape.id, base_candidate.hash),
        (shape.id, overlay_candidate.hash),
    }
    assert merged.counts()["benchmark_samples"] == 4
    assert merged.counts()["validations"] == 2
    assert discoveries[0].context["merged_from_db"] == str(overlay.path)
    assert any(source_ref.startswith("merged:") for source_ref in source_refs)
    assert manifest == stored_manifest


def test_merge_rejects_incompatible_problem_type(tmp_path: Path):
    shape = pilot_100_shapes()[0]
    base_candidate, overlay_candidate = sample_candidates(2)
    protocol_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    base = _evidence_db(
        tmp_path / "base.sqlite",
        candidate=base_candidate,
        shape=shape,
        protocol_hash=protocol_hash,
        run_id="base-run",
    )
    overlay = _evidence_db(
        tmp_path / "overlay.sqlite",
        candidate=overlay_candidate,
        shape=shape,
        protocol_hash=protocol_hash,
        problem_type_hash="incompatible-problem",
        run_id="overlay-run",
    )
    output = tmp_path / "merged.sqlite"

    with pytest.raises(ValueError, match="incompatible problem types"):
        merge_compatible_databases(
            output,
            [base.path, overlay.path],
            problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
            benchmark_protocol_hash=protocol_hash,
        )

    assert not output.exists()


def test_merge_rejects_incompatible_benchmark_protocol(tmp_path: Path):
    shape = pilot_100_shapes()[0]
    base_candidate, overlay_candidate = sample_candidates(2)
    protocol_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    base = _evidence_db(
        tmp_path / "base.sqlite",
        candidate=base_candidate,
        shape=shape,
        protocol_hash=protocol_hash,
        run_id="base-run",
    )
    overlay = _evidence_db(
        tmp_path / "overlay.sqlite",
        candidate=overlay_candidate,
        shape=shape,
        protocol_hash="incompatible-protocol",
        run_id="overlay-run",
    )
    output = tmp_path / "merged.sqlite"

    with pytest.raises(ValueError, match="incompatible benchmark protocols"):
        merge_compatible_databases(
            output,
            [base.path, overlay.path],
            problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
            benchmark_protocol_hash=protocol_hash,
        )

    assert not output.exists()
