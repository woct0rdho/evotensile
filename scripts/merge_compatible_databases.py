#!/usr/bin/env python3

import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

from evotensile.profile import DEFAULT_PROFILE


def _row_id(connection, query, parameters):
    row = connection.execute(query, parameters).fetchone()
    if row is None:
        raise ValueError(f"merge identity lookup failed: {query}")
    return int(row[0])


def _verify_equal(existing, incoming, *, identity):
    if existing != incoming:
        raise ValueError(f"conflicting {identity} definitions")


def _intern_hash_table(
    destination,
    source,
    *,
    table,
    id_column,
    hash_column,
):
    mapping = {}
    rows = source.execute(f"SELECT {id_column}, {hash_column}, definition_json FROM {table}").fetchall()
    for row in rows:
        destination.execute(
            f"INSERT OR IGNORE INTO {table}({hash_column}, definition_json) VALUES (?, ?)",
            (row[hash_column], row["definition_json"]),
        )
        destination_row = destination.execute(
            f"SELECT {id_column}, definition_json FROM {table} WHERE {hash_column} = ?",
            (row[hash_column],),
        ).fetchone()
        assert destination_row is not None
        if destination_row["definition_json"] is not None and row["definition_json"] is not None:
            _verify_equal(
                destination_row["definition_json"],
                row["definition_json"],
                identity=f"{table} {row[hash_column]}",
            )
        mapping[int(row[id_column])] = int(destination_row[id_column])
    return mapping


def _intern_candidates(destination, source):
    mapping = {}
    for row in source.execute("SELECT * FROM candidates ORDER BY candidate_id"):
        destination.execute(
            "INSERT OR IGNORE INTO candidates(candidate_hash, params_json, created_at) VALUES (?, ?, ?)",
            (row["candidate_hash"], row["params_json"], row["created_at"]),
        )
        destination_row = destination.execute(
            "SELECT candidate_id, params_json FROM candidates WHERE candidate_hash = ?",
            (row["candidate_hash"],),
        ).fetchone()
        assert destination_row is not None
        _verify_equal(
            destination_row["params_json"],
            row["params_json"],
            identity=f"candidate {row['candidate_hash']}",
        )
        mapping[int(row["candidate_id"])] = int(destination_row["candidate_id"])
    return mapping


def _intern_shapes(destination, source):
    mapping = {}
    for row in source.execute("SELECT * FROM shapes ORDER BY shape_key"):
        destination.execute(
            "INSERT OR IGNORE INTO shapes(shape_id, m, n, batch, k, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (row["shape_id"], row["m"], row["n"], row["batch"], row["k"], row["created_at"]),
        )
        destination_row = destination.execute(
            "SELECT shape_key, m, n, batch, k FROM shapes WHERE shape_id = ?",
            (row["shape_id"],),
        ).fetchone()
        assert destination_row is not None
        _verify_equal(
            tuple(destination_row[key] for key in ("m", "n", "batch", "k")),
            tuple(row[key] for key in ("m", "n", "batch", "k")),
            identity=f"shape {row['shape_id']}",
        )
        mapping[int(row["shape_key"])] = int(destination_row["shape_key"])
    return mapping


def _intern_namespaces(
    destination,
    source,
    *,
    table,
    id_column,
    left_column,
    right_column,
    left_mapping,
    right_mapping,
):
    mapping = {}
    for row in source.execute(f"SELECT * FROM {table} ORDER BY {id_column}"):
        left_id = left_mapping[int(row[left_column])]
        right_id = right_mapping[int(row[right_column])]
        destination.execute(
            f"INSERT OR IGNORE INTO {table}({left_column}, {right_column}) VALUES (?, ?)",
            (left_id, right_id),
        )
        mapping[int(row[id_column])] = _row_id(
            destination,
            f"SELECT {id_column} FROM {table} WHERE {left_column} = ? AND {right_column} = ?",
            (left_id, right_id),
        )
    return mapping


def _import_source(
    destination,
    source_path,
    *,
    required_problem_type_hash,
    required_benchmark_protocol_hash,
    required_environment_compatibility_tag,
):
    source = sqlite3.connect(f"file:{source_path.resolve()}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    source_tag = source.execute(
        "SELECT metadata_value FROM database_metadata WHERE metadata_key = 'environment_compatibility_tag'"
    ).fetchone()
    if source_tag is None or source_tag[0] != required_environment_compatibility_tag:
        raise ValueError(f"incompatible environment tag in {source_path}")
    problem_type_hashes = {
        str(row[0])
        for row in source.execute(
            "SELECT DISTINCT problem.problem_type_hash "
            "FROM benchmark_events AS event "
            "JOIN benchmark_namespaces AS namespace USING (benchmark_namespace_id) "
            "JOIN problem_types AS problem USING (problem_type_id)"
        )
    }
    if problem_type_hashes - {required_problem_type_hash}:
        raise ValueError(f"incompatible problem types in {source_path}: {sorted(problem_type_hashes)}")
    protocol_hashes = {
        str(row[0])
        for row in source.execute(
            "SELECT DISTINCT bp.benchmark_protocol_hash "
            "FROM benchmark_events AS event "
            "JOIN benchmark_namespaces AS namespace USING (benchmark_namespace_id) "
            "JOIN benchmark_protocols AS bp USING (benchmark_protocol_id)"
        )
    }
    if protocol_hashes - {required_benchmark_protocol_hash}:
        raise ValueError(f"incompatible benchmark protocols in {source_path}: {sorted(protocol_hashes)}")

    problem_types = _intern_hash_table(
        destination,
        source,
        table="problem_types",
        id_column="problem_type_id",
        hash_column="problem_type_hash",
    )
    benchmark_protocols = _intern_hash_table(
        destination,
        source,
        table="benchmark_protocols",
        id_column="benchmark_protocol_id",
        hash_column="benchmark_protocol_hash",
    )
    validation_protocols = _intern_hash_table(
        destination,
        source,
        table="validation_protocols",
        id_column="validation_protocol_id",
        hash_column="validation_protocol_hash",
    )
    benchmark_namespaces = _intern_namespaces(
        destination,
        source,
        table="benchmark_namespaces",
        id_column="benchmark_namespace_id",
        left_column="problem_type_id",
        right_column="benchmark_protocol_id",
        left_mapping=problem_types,
        right_mapping=benchmark_protocols,
    )
    validation_namespaces = _intern_namespaces(
        destination,
        source,
        table="validation_namespaces",
        id_column="validation_namespace_id",
        left_column="problem_type_id",
        right_column="validation_protocol_id",
        left_mapping=problem_types,
        right_mapping=validation_protocols,
    )
    candidates = _intern_candidates(destination, source)
    shapes = _intern_shapes(destination, source)

    source_identity = hashlib.sha256(str(source_path.resolve()).encode()).hexdigest()[:12]
    evidence_sources = {}
    for row in source.execute("SELECT * FROM evidence_sources ORDER BY source_id"):
        source_ref = f"merged:{source_identity}:{row['source_ref']}"
        destination.execute(
            "INSERT INTO evidence_sources(source_kind, source_ref, created_at) VALUES (?, ?, ?)",
            (row["source_kind"], source_ref, row["created_at"]),
        )
        evidence_sources[int(row["source_id"])] = int(destination.execute("SELECT last_insert_rowid()").fetchone()[0])

    for row in source.execute("SELECT * FROM native_runs ORDER BY source_id"):
        destination.execute(
            "INSERT INTO native_runs(source_id, phase, status, duration_s, returncode) VALUES (?, ?, ?, ?, ?)",
            (
                evidence_sources[int(row["source_id"])],
                row["phase"],
                row["status"],
                row["duration_s"],
                row["returncode"],
            ),
        )
    for row in source.execute("SELECT * FROM run_candidate_costs ORDER BY source_id, candidate_id, phase"):
        destination.execute(
            "INSERT INTO run_candidate_costs(source_id, candidate_id, phase, duration_s) VALUES (?, ?, ?, ?)",
            (
                evidence_sources[int(row["source_id"])],
                candidates[int(row["candidate_id"])],
                row["phase"],
                row["duration_s"],
            ),
        )

    event_count = 0
    sample_count = 0
    for row in source.execute("SELECT * FROM benchmark_events ORDER BY event_id"):
        validation_namespace_id = row["validation_namespace_id"]
        destination.execute(
            "INSERT INTO benchmark_events("
            "benchmark_namespace_id, shape_key, candidate_id, source_id, status, "
            "validation_namespace_id, solution_index, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                benchmark_namespaces[int(row["benchmark_namespace_id"])],
                shapes[int(row["shape_key"])],
                candidates[int(row["candidate_id"])],
                evidence_sources[int(row["source_id"])],
                row["status"],
                None if validation_namespace_id is None else validation_namespaces[int(validation_namespace_id)],
                row["solution_index"],
                row["created_at"],
            ),
        )
        event_id = int(destination.execute("SELECT last_insert_rowid()").fetchone()[0])
        samples = source.execute(
            "SELECT sample_index, time_us FROM benchmark_samples WHERE event_id = ? ORDER BY sample_index",
            (row["event_id"],),
        ).fetchall()
        destination.executemany(
            "INSERT INTO benchmark_samples(event_id, sample_index, time_us) VALUES (?, ?, ?)",
            [(event_id, sample["sample_index"], sample["time_us"]) for sample in samples],
        )
        event_count += 1
        sample_count += len(samples)

    validation_count = 0
    for row in source.execute("SELECT * FROM validations ORDER BY validation_id"):
        destination.execute(
            "INSERT INTO validations("
            "validation_namespace_id, shape_key, candidate_id, source_id, status, detail, "
            "solution_index, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                validation_namespaces[int(row["validation_namespace_id"])],
                shapes[int(row["shape_key"])],
                candidates[int(row["candidate_id"])],
                evidence_sources[int(row["source_id"])],
                row["status"],
                row["detail"],
                row["solution_index"],
                row["created_at"],
            ),
        )
        validation_count += 1

    discovery_count = 0
    selection_count = 0
    for row in source.execute("SELECT * FROM baseline_discoveries ORDER BY discovery_id"):
        context = json.loads(row["context_json"])
        context["merged_from_db"] = str(source_path)
        destination.execute(
            "INSERT INTO baseline_discoveries(problem_type_id, context_json, duration_s, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                problem_types[int(row["problem_type_id"])],
                json.dumps(context, sort_keys=True, separators=(",", ":")),
                row["duration_s"],
                row["created_at"],
            ),
        )
        discovery_id = int(destination.execute("SELECT last_insert_rowid()").fetchone()[0])
        selections = source.execute(
            "SELECT * FROM baseline_selections WHERE discovery_id = ? ORDER BY shape_key",
            (row["discovery_id"],),
        ).fetchall()
        destination.executemany(
            "INSERT INTO baseline_selections("
            "discovery_id, shape_key, candidate_id, hipblaslt_solution_index, "
            "hipblaslt_solution_name, hipblaslt_kernel_name, logic_solution_index, "
            "logic_solution_name, query_gflops, query_time_us"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    discovery_id,
                    shapes[int(selection["shape_key"])],
                    candidates[int(selection["candidate_id"])],
                    selection["hipblaslt_solution_index"],
                    selection["hipblaslt_solution_name"],
                    selection["hipblaslt_kernel_name"],
                    selection["logic_solution_index"],
                    selection["logic_solution_name"],
                    selection["query_gflops"],
                    selection["query_time_us"],
                )
                for selection in selections
            ],
        )
        discovery_count += 1
        selection_count += len(selections)
    source.close()
    return {
        "path": str(source_path),
        "benchmark_events": event_count,
        "benchmark_samples": sample_count,
        "validations": validation_count,
        "baseline_discoveries": discovery_count,
        "baseline_selections": selection_count,
    }


def merge_compatible_databases(
    output_path,
    source_paths,
    *,
    problem_type_hash,
    benchmark_protocol_hash,
):
    output = Path(output_path)
    sources = [Path(path) for path in source_paths]
    if len(sources) < 2:
        raise ValueError("database merge requires a base DB and at least one overlay")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sources[0], output)
    with sqlite3.connect(output) as base_connection:
        base_tag_row = base_connection.execute(
            "SELECT metadata_value FROM database_metadata WHERE metadata_key = 'environment_compatibility_tag'"
        ).fetchone()
        base_problem_type_hashes = {
            str(row[0])
            for row in base_connection.execute(
                "SELECT DISTINCT problem.problem_type_hash "
                "FROM benchmark_events AS event "
                "JOIN benchmark_namespaces AS namespace USING (benchmark_namespace_id) "
                "JOIN problem_types AS problem USING (problem_type_id)"
            )
        }
        base_protocol_hashes = {
            str(row[0])
            for row in base_connection.execute(
                "SELECT DISTINCT protocol.benchmark_protocol_hash "
                "FROM benchmark_events AS event "
                "JOIN benchmark_namespaces AS namespace USING (benchmark_namespace_id) "
                "JOIN benchmark_protocols AS protocol USING (benchmark_protocol_id)"
            )
        }
    if base_tag_row is None:
        output.unlink(missing_ok=True)
        raise ValueError(f"base database lacks an environment compatibility tag: {sources[0]}")
    if base_problem_type_hashes - {problem_type_hash}:
        output.unlink(missing_ok=True)
        raise ValueError(f"incompatible problem types in base DB {sources[0]}: {sorted(base_problem_type_hashes)}")
    if base_protocol_hashes - {benchmark_protocol_hash}:
        output.unlink(missing_ok=True)
        raise ValueError(f"incompatible benchmark protocols in base DB {sources[0]}: {sorted(base_protocol_hashes)}")
    environment_compatibility_tag = str(base_tag_row[0])
    manifest = {
        "base": str(sources[0]),
        "problem_type_hash": problem_type_hash,
        "benchmark_protocol_hash": benchmark_protocol_hash,
        "environment_compatibility_tag": environment_compatibility_tag,
        "overlays": [],
        "overlay_excluded_tables": [
            "proposal_events",
            "proposal_candidates",
            "artifact_bundles",
            "artifact_solution_yamls",
            "artifact_mappings",
        ],
    }
    try:
        destination = sqlite3.connect(output)
        destination.row_factory = sqlite3.Row
        destination.execute("PRAGMA foreign_keys=ON")
        destination.execute("PRAGMA journal_mode=WAL")
        for source in sources[1:]:
            with destination:
                manifest["overlays"].append(
                    _import_source(
                        destination,
                        source,
                        required_problem_type_hash=problem_type_hash,
                        required_benchmark_protocol_hash=benchmark_protocol_hash,
                        required_environment_compatibility_tag=environment_compatibility_tag,
                    )
                )
        with destination:
            destination.execute(
                "INSERT OR REPLACE INTO database_metadata(metadata_key, metadata_value) VALUES (?, ?)",
                ("merged_source_manifest", json.dumps(manifest, sort_keys=True)),
            )
        foreign_key_errors = destination.execute("PRAGMA foreign_key_check").fetchall()
        destination.close()
        if foreign_key_errors:
            raise ValueError(f"merged database has foreign-key errors: {foreign_key_errors}")
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--problem-type-hash", default=DEFAULT_PROFILE.problem_type_hash)
    parser.add_argument(
        "--benchmark-protocol-hash",
        default=DEFAULT_PROFILE.benchmark_protocol_hash(),
    )
    args = parser.parse_args()
    manifest = merge_compatible_databases(
        args.output,
        args.source,
        problem_type_hash=args.problem_type_hash,
        benchmark_protocol_hash=args.benchmark_protocol_hash,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
