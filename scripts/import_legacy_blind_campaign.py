#!/usr/bin/env python3

import argparse
import hashlib
import json
import shutil
import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any, TypedDict, cast

from evotensile.campaign.protocols import CAMPAIGN_HOT_PROTOCOL, CAMPAIGN_SCREENING_PROTOCOL
from evotensile.candidate import Candidate, Shape
from evotensile.database import BenchmarkEventInsert, EvoTensileDB, ValidationInsert
from evotensile.profile import PROFILES, TargetProfile, get_profile


class LegacyCandidatePayload(TypedDict):
    params: dict[str, Any]
    proposal_metadata: dict[str, Any]


class HotSummaryRow(TypedDict):
    candidate_hash: str
    duration_s: float


class ImportReport(TypedDict):
    base_database: str
    source_database: str
    source_sha256: str
    profile: str
    problem_type_hash: str
    benchmark_protocol_hashes: list[str]
    validation_protocol_hashes: list[str]
    source_candidates: int
    registered_candidates_before: int
    registered_candidates_after: int
    imported_runs: int
    imported_validations: int
    imported_benchmark_events: int
    imported_benchmark_samples: int
    imported_hot_events: int
    imported_hot_samples: int
    benchmark_status_counts: dict[str, int]
    integrity_check: str
    foreign_key_violations: int
    output_database: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _legacy_candidate(row: sqlite3.Row) -> Candidate:
    payload = cast(LegacyCandidatePayload, json.loads(str(row["candidate_json"])))
    candidate = Candidate(
        params=payload["params"],
        source=str(row["source"]),
        parent_hashes=tuple(str(value) for value in json.loads(str(row["parent_hashes"]))),
        proposal_metadata=payload.get("proposal_metadata", {}),
    )
    if candidate.hash != str(row["candidate_hash"]):
        raise ValueError(
            f"legacy candidate hash mismatch: stored={row['candidate_hash']}, reconstructed={candidate.hash}"
        )
    return candidate


def _phase(run_id: str, metadata: Mapping[str, object]) -> str:
    mode = metadata.get("mode")
    if isinstance(mode, str) and mode:
        return mode
    command = metadata.get("command")
    if isinstance(command, list) and "--build-only" in command:
        return "prepare"
    if run_id.startswith("validate_"):
        return "validation"
    if run_id.startswith("benchmark_"):
        return "benchmark"
    return "legacy"


def _duration(metadata: Mapping[str, object]) -> float:
    value = metadata.get("duration_s", 0.0)
    if not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, float(value))


def _destination_protocol_hashes(path: Path) -> set[str]:
    with closing(sqlite3.connect(path)) as connection:
        return {str(row[0]) for row in connection.execute("SELECT benchmark_protocol_hash FROM benchmark_protocols")}


def _source_protocols(connection: sqlite3.Connection, column: str, table: str) -> set[str]:
    return {str(row[0]) for row in connection.execute(f"SELECT DISTINCT {column} FROM {table}")}


def _validate_compatibility(
    source: sqlite3.Connection,
    *,
    profile: TargetProfile,
    destination_protocol_hashes: set[str],
) -> tuple[set[str], set[str]]:
    expected_tables = {"candidates", "evaluations", "runs", "shapes", "validations"}
    missing_tables = expected_tables - _table_names(source)
    if missing_tables:
        raise ValueError(f"legacy source is missing tables: {sorted(missing_tables)}")
    problem_type_hashes = _source_protocols(source, "problem_type_hash", "evaluations") | _source_protocols(
        source, "problem_type_hash", "validations"
    )
    if problem_type_hashes != {profile.problem_type_hash}:
        raise ValueError(f"incompatible legacy problem types: {sorted(problem_type_hashes)}")
    benchmark_protocol_hashes = _source_protocols(source, "benchmark_protocol_hash", "evaluations")
    allowed_benchmark_protocols = destination_protocol_hashes | {CAMPAIGN_SCREENING_PROTOCOL.protocol_hash()}
    unsupported_protocols = benchmark_protocol_hashes - allowed_benchmark_protocols
    if unsupported_protocols:
        raise ValueError(f"incompatible legacy benchmark protocols: {sorted(unsupported_protocols)}")
    validation_protocol_hashes = _source_protocols(source, "validation_protocol_hash", "validations")
    if len(validation_protocol_hashes) != 1:
        raise ValueError("legacy blind import requires exactly one validation protocol")
    legacy_shapes = [
        Shape(int(row["m"]), int(row["n"]), int(row["batch"]), int(row["k"]))
        for row in source.execute("SELECT * FROM shapes ORDER BY shape_id")
    ]
    if not legacy_shapes or any(shape not in profile.shapes() for shape in legacy_shapes):
        raise ValueError("legacy source contains a shape outside the target profile")
    return benchmark_protocol_hashes, validation_protocol_hashes


def _run_attributions(source: sqlite3.Connection) -> dict[str, set[str]]:
    attributions: dict[str, set[str]] = defaultdict(set)
    for table in ("evaluations", "validations"):
        for row in source.execute(f"SELECT run_id, candidate_hash FROM {table} WHERE run_id IS NOT NULL"):
            attributions[str(row["run_id"])].add(str(row["candidate_hash"]))
    return attributions


def _register_runs(
    database: EvoTensileDB,
    source: sqlite3.Connection,
    *,
    source_identity: str,
    attributions: Mapping[str, set[str]],
) -> tuple[dict[str, str], int]:
    run_refs: dict[str, str] = {}
    imported = 0
    source_run_ids: set[str] = set()
    for row in source.execute("SELECT * FROM runs ORDER BY timestamp, run_id"):
        run_id = str(row["run_id"])
        source_run_ids.add(run_id)
        run_ref = f"legacy-blind:{source_identity}:{run_id}"
        metadata_payload = json.loads(str(row["metadata_json"] or "{}"))
        metadata = cast(dict[str, object], metadata_payload if isinstance(metadata_payload, dict) else {})
        database.insert_run(
            run_ref,
            phase=_phase(run_id, metadata),
            status=str(row["status"]),
            duration_s=_duration(metadata),
            returncode=None if row["returncode"] is None else int(row["returncode"]),
            candidate_hashes=sorted(attributions.get(run_id, set())),
        )
        run_refs[run_id] = run_ref
        imported += 1
    missing_run_ids = sorted(set(attributions) - source_run_ids)
    for run_id in missing_run_ids:
        run_ref = f"legacy-blind:{source_identity}:{run_id}"
        database.insert_run(
            run_ref,
            phase=_phase(run_id, {}),
            status="ok",
            duration_s=0.0,
            candidate_hashes=sorted(attributions[run_id]),
        )
        run_refs[run_id] = run_ref
        imported += 1
    return run_refs, imported


def _validation_inserts(
    source: sqlite3.Connection,
    *,
    run_refs: Mapping[str, str],
    source_identity: str,
) -> list[ValidationInsert]:
    inserts = []
    for row in source.execute("SELECT * FROM validations ORDER BY created_at, validation_id"):
        run_id = None if row["run_id"] is None else str(row["run_id"])
        run_ref = (
            f"legacy-blind:{source_identity}:validation:{row['validation_id']}" if run_id is None else run_refs[run_id]
        )
        inserts.append(
            ValidationInsert(
                shape_id=str(row["shape_id"]),
                candidate_hash=str(row["candidate_hash"]),
                run_id=run_ref,
                status=str(row["status"]),
                problem_type_hash=str(row["problem_type_hash"]),
                validation_protocol_hash=str(row["validation_protocol_hash"]),
                source_kind="native_run",
                detail=None if row["detail"] is None else str(row["detail"]),
                solution_index=None if row["solution_index"] is None else int(row["solution_index"]),
            )
        )
    return inserts


def _register_synthetic_validation_runs(
    database: EvoTensileDB,
    source: sqlite3.Connection,
    *,
    source_identity: str,
) -> None:
    for row in source.execute("SELECT * FROM validations WHERE run_id IS NULL ORDER BY validation_id"):
        database.insert_run(
            f"legacy-blind:{source_identity}:validation:{row['validation_id']}",
            phase="validation",
            status="ok" if str(row["status"]) == "passed" else "failed",
            duration_s=0.0,
            candidate_hashes=[str(row["candidate_hash"])],
        )


def _benchmark_inserts(
    source: sqlite3.Connection,
    *,
    run_refs: Mapping[str, str],
    source_identity: str,
    validation_protocol_hash: str,
) -> tuple[list[BenchmarkEventInsert], int, dict[str, int]]:
    grouped: dict[tuple[object, ...], list[sqlite3.Row]] = defaultdict(list)
    for row in source.execute("SELECT * FROM evaluations ORDER BY created_at, eval_id"):
        run_id = None if row["run_id"] is None else str(row["run_id"])
        key = (
            str(row["problem_type_hash"]),
            str(row["benchmark_protocol_hash"]),
            str(row["shape_id"]),
            str(row["candidate_hash"]),
            run_id,
            str(row["status"]),
            None if row["solution_index"] is None else int(row["solution_index"]),
        )
        grouped[key].append(row)
    inserts = []
    sample_count = 0
    status_counts: dict[str, int] = defaultdict(int)
    for key, rows in grouped.items():
        problem_type_hash, benchmark_protocol_hash, shape_id, candidate_hash, run_id, status, solution_index = key
        if run_id is None:
            run_ref = f"legacy-blind:{source_identity}:evaluation:{rows[0]['eval_id']}"
        else:
            run_ref = run_refs[str(run_id)]
        samples = tuple(float(row["time_us"]) for row in rows if row["time_us"] is not None)
        sample_count += len(samples)
        status_counts[str(status)] += 1
        inserts.append(
            BenchmarkEventInsert(
                shape_id=str(shape_id),
                candidate_hash=str(candidate_hash),
                run_id=run_ref,
                status=str(status),
                problem_type_hash=str(problem_type_hash),
                benchmark_protocol_hash=str(benchmark_protocol_hash),
                source_kind="native_run",
                samples_us=samples,
                validation_protocol_hash=validation_protocol_hash if status == "ok" else None,
                solution_index=cast(int | None, solution_index),
            )
        )
    return inserts, sample_count, dict(status_counts)


def _register_synthetic_evaluation_runs(
    database: EvoTensileDB,
    source: sqlite3.Connection,
    *,
    source_identity: str,
) -> None:
    for row in source.execute("SELECT * FROM evaluations WHERE run_id IS NULL ORDER BY eval_id"):
        database.insert_run(
            f"legacy-blind:{source_identity}:evaluation:{row['eval_id']}",
            phase="benchmark",
            status="ok" if str(row["status"]) == "ok" else str(row["status"]),
            duration_s=0.0,
            candidate_hashes=[str(row["candidate_hash"])],
        )


def _hot_summary_rows(hot_dir: Path) -> dict[str, HotSummaryRow]:
    summary_path = hot_dir / "summary.json"
    if not summary_path.exists():
        return {}
    payload = cast(dict[str, object], json.loads(summary_path.read_text(encoding="utf-8")))
    ranked = payload.get("ranked", [])
    if not isinstance(ranked, list):
        raise ValueError("hot summary ranked rows must be a list")
    rows: dict[str, HotSummaryRow] = {}
    for raw_row_value in ranked:
        if not isinstance(raw_row_value, dict):
            raise ValueError("hot summary row must be an object")
        raw_row = cast(dict[str, object], raw_row_value)
        candidate_hash_value = raw_row.get("candidate_hash")
        duration_s_value = raw_row.get("duration_s", 0.0)
        if not isinstance(candidate_hash_value, str):
            raise ValueError("hot summary candidate_hash must be a string")
        if not isinstance(duration_s_value, (int, float)):
            raise ValueError("hot summary duration_s must be numeric")
        rows[candidate_hash_value] = HotSummaryRow(
            candidate_hash=candidate_hash_value,
            duration_s=float(duration_s_value),
        )
    return rows


def _import_hot_results(
    database: EvoTensileDB,
    hot_dir: Path,
    *,
    source_identity: str,
    problem_type_hash: str,
    validation_protocol_hash: str,
) -> tuple[int, int]:
    summary_rows = _hot_summary_rows(hot_dir)
    event_count = 0
    sample_count = 0
    for results_path in sorted(hot_dir.glob("rank_*/results.jsonl")):
        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            continue
        candidate_hash = str(rows[0]["candidate_hash"])
        if any(str(row["candidate_hash"]) != candidate_hash for row in rows):
            raise ValueError(f"mixed candidate hashes in {results_path}")
        ordered = sorted(rows, key=lambda row: int(row["sample_index"]))
        if any(str(row["status"]) != "ok" for row in ordered):
            raise ValueError(f"non-ok hot result in {results_path}")
        samples = tuple(float(row["time_us"]) for row in ordered)
        run_ref = f"legacy-blind:{source_identity}:hot:{candidate_hash}"
        database.insert_run(
            run_ref,
            phase="hot_confirmation",
            status="ok",
            duration_s=summary_rows.get(candidate_hash, HotSummaryRow(candidate_hash=candidate_hash, duration_s=0.0))[
                "duration_s"
            ],
            returncode=0,
            candidate_hashes=[candidate_hash],
        )
        database.insert_benchmark_events(
            [
                BenchmarkEventInsert(
                    shape_id=str(ordered[0]["shape_id"]),
                    candidate_hash=candidate_hash,
                    run_id=run_ref,
                    status="ok",
                    problem_type_hash=problem_type_hash,
                    benchmark_protocol_hash=CAMPAIGN_HOT_PROTOCOL.protocol_hash(),
                    source_kind="native_run",
                    samples_us=samples,
                    validation_protocol_hash=validation_protocol_hash,
                    solution_index=int(ordered[0]["solution_index"]),
                )
            ]
        )
        event_count += 1
        sample_count += len(samples)
    return event_count, sample_count


def import_legacy_blind_campaign(
    *,
    base_database: Path,
    source_database: Path,
    output_database: Path,
    profile: TargetProfile,
    hot_dir: Path | None = None,
) -> ImportReport:
    if output_database.exists():
        raise FileExistsError(output_database)
    if not base_database.exists() or not source_database.exists():
        raise FileNotFoundError("base and source databases must exist")
    output_database.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_database, output_database)
    source_sha256 = _sha256(source_database)
    source_identity = source_sha256[:12]
    database = EvoTensileDB.connect(
        output_database,
        environment_compatibility_tag=profile.environment_compatibility_tag,
    )
    database.init()
    try:
        with closing(sqlite3.connect(f"file:{source_database.resolve()}?mode=ro", uri=True)) as source:
            source.row_factory = sqlite3.Row
            benchmark_protocol_hashes, validation_protocol_hashes = _validate_compatibility(
                source,
                profile=profile,
                destination_protocol_hashes=_destination_protocol_hashes(output_database),
            )
            candidates = [_legacy_candidate(row) for row in source.execute("SELECT * FROM candidates")]
            source_candidates = len(candidates)
            shapes = [
                Shape(int(row["m"]), int(row["n"]), int(row["batch"]), int(row["k"]))
                for row in source.execute("SELECT * FROM shapes")
            ]
            with database.connection() as connection:
                registered_before = int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
            database.register_candidates(candidates)
            database.register_shapes(shapes)
            database.record_proposal_event(
                candidates,
                problem_type_hash=profile.problem_type_hash,
                benchmark_protocol_hash=CAMPAIGN_SCREENING_PROTOCOL.protocol_hash(),
                scope_kind="legacy_blind_campaign",
                scope_shape_ids=tuple(shape.id for shape in shapes),
                generated_hashes={candidate.hash for candidate in candidates},
                selected_candidates=candidates,
                proposal_args={"source_database": str(source_database), "source_sha256": source_sha256},
            )
            attributions = _run_attributions(source)
            run_refs, imported_runs = _register_runs(
                database,
                source,
                source_identity=source_identity,
                attributions=attributions,
            )
            _register_synthetic_validation_runs(database, source, source_identity=source_identity)
            validations = _validation_inserts(source, run_refs=run_refs, source_identity=source_identity)
            database.insert_validations(validations)
            _register_synthetic_evaluation_runs(database, source, source_identity=source_identity)
            validation_protocol_hash = next(iter(validation_protocol_hashes))
            benchmark_events, benchmark_samples, status_counts = _benchmark_inserts(
                source,
                run_refs=run_refs,
                source_identity=source_identity,
                validation_protocol_hash=validation_protocol_hash,
            )
            database.insert_benchmark_events(benchmark_events)
        hot_events = 0
        hot_samples = 0
        if hot_dir is not None:
            hot_events, hot_samples = _import_hot_results(
                database,
                hot_dir,
                source_identity=source_identity,
                problem_type_hash=profile.problem_type_hash,
                validation_protocol_hash=validation_protocol_hash,
            )
        with database.connection() as connection:
            registered_after = int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
            integrity_check = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            foreign_key_violations = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        if integrity_check != "ok" or foreign_key_violations:
            raise ValueError(
                f"imported database failed integrity checks: integrity={integrity_check}, fk={foreign_key_violations}"
            )
    except Exception:
        output_database.unlink(missing_ok=True)
        raise
    return ImportReport(
        base_database=str(base_database),
        source_database=str(source_database),
        source_sha256=source_sha256,
        profile=profile.name,
        problem_type_hash=profile.problem_type_hash,
        benchmark_protocol_hashes=sorted(benchmark_protocol_hashes),
        validation_protocol_hashes=sorted(validation_protocol_hashes),
        source_candidates=source_candidates,
        registered_candidates_before=registered_before,
        registered_candidates_after=registered_after,
        imported_runs=imported_runs,
        imported_validations=len(validations),
        imported_benchmark_events=len(benchmark_events),
        imported_benchmark_samples=benchmark_samples,
        imported_hot_events=hot_events,
        imported_hot_samples=hot_samples,
        benchmark_status_counts=status_counts,
        integrity_check=integrity_check,
        foreign_key_violations=foreign_key_violations,
        output_database=str(output_database),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="gfx1151-nt-hhs-comfy1135")
    parser.add_argument("--hot-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = import_legacy_blind_campaign(
        base_database=args.base,
        source_database=args.source,
        output_database=args.output,
        profile=get_profile(args.profile),
        hot_dir=args.hot_dir,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
