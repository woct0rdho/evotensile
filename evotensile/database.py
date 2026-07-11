import json
import math
import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .candidate import Candidate, Shape, canonical_json
from .metrics import gflops_from_us

NEGATIVE_CACHE_STATUSES = ("rejected", "build_failed")


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


@dataclass(frozen=True)
class BenchmarkEventInsert:
    shape_id: str
    candidate_hash: str
    run_id: str | None
    status: str
    problem_type_hash: str
    benchmark_protocol_hash: str
    source_kind: str
    samples_us: tuple[float, ...] = ()
    validation_protocol_hash: str | None = None
    solution_index: int | None = None


@dataclass(frozen=True)
class ValidationInsert:
    shape_id: str
    candidate_hash: str
    run_id: str | None
    status: str
    problem_type_hash: str
    validation_protocol_hash: str
    source_kind: str
    detail: str | None = None
    solution_index: int | None = None


@dataclass(frozen=True)
class BaselineSelectionInsert:
    shape: Shape
    candidate: Candidate
    hipblaslt_solution_index: int
    hipblaslt_solution_name: str | None = None
    hipblaslt_kernel_name: str | None = None
    logic_solution_index: int | None = None
    logic_solution_name: str | None = None
    query_gflops: float | None = None
    query_time_us: float | None = None


@dataclass(frozen=True)
class ArtifactBundleInsert:
    build_run_id: str
    build_output_dir: str
    library_dir: str
    solution_yaml_paths: tuple[str, ...]
    manifest_path: str | None
    code_object_identity: str


@dataclass(frozen=True)
class ArtifactMappingInsert:
    problem_type_hash: str
    shape_id: str
    candidate_hash: str
    problem_index: int
    requested_solution_index: int
    library_solution_index: int
    manifest_solution_index: int | None


@dataclass(frozen=True)
class CandidateArtifactRecord:
    mapping_id: int
    bundle_id: int
    problem_type_hash: str
    shape_id: str
    candidate_hash: str
    problem_index: int
    requested_solution_index: int
    library_solution_index: int
    manifest_solution_index: int | None
    build_run_id: str
    build_output_dir: str
    library_dir: str
    solution_yaml_paths: tuple[str, ...]
    manifest_path: str | None
    code_object_identity: str
    created_at: float


@dataclass
class _TimingBucket:
    shape: Shape
    time_us: list[float]


@dataclass(frozen=True)
class BenchmarkEvidenceState:
    ok_samples: int
    resolved_status: str | None
    latest_negative_event_id: int | None

    @property
    def reusable_negative(self) -> bool:
        return self.ok_samples == 0 and self.resolved_status in NEGATIVE_CACHE_STATUSES


@dataclass(frozen=True)
class ProposalOccurrence:
    occurrence_id: int
    proposal_event_id: int
    problem_type_hash: str
    benchmark_protocol_hash: str
    candidate_hash: str
    source: str
    parent_hashes: tuple[str, ...]
    proposal_metadata: dict[str, object]
    state: str
    scope_kind: str
    scope_shape_ids: tuple[str, ...]
    island_id: str | None
    restart_index: int
    duration_s: float
    selected: bool
    created_at: float


@dataclass(frozen=True)
class BenchmarkSummary:
    shape_id: str
    candidate_hash: str
    samples: int
    median_gflops: float | None
    best_gflops: float | None
    median_time_us: float | None
    best_time_us: float | None


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS database_metadata (
  metadata_key TEXT PRIMARY KEY,
  metadata_value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS problem_types (
  problem_type_id INTEGER PRIMARY KEY,
  problem_type_hash TEXT NOT NULL UNIQUE,
  definition_json TEXT
);

CREATE TABLE IF NOT EXISTS benchmark_protocols (
  benchmark_protocol_id INTEGER PRIMARY KEY,
  benchmark_protocol_hash TEXT NOT NULL UNIQUE,
  definition_json TEXT
);

CREATE TABLE IF NOT EXISTS validation_protocols (
  validation_protocol_id INTEGER PRIMARY KEY,
  validation_protocol_hash TEXT NOT NULL UNIQUE,
  definition_json TEXT
);

CREATE TABLE IF NOT EXISTS benchmark_namespaces (
  benchmark_namespace_id INTEGER PRIMARY KEY,
  problem_type_id INTEGER NOT NULL REFERENCES problem_types(problem_type_id),
  benchmark_protocol_id INTEGER NOT NULL REFERENCES benchmark_protocols(benchmark_protocol_id),
  UNIQUE(problem_type_id, benchmark_protocol_id)
);

CREATE TABLE IF NOT EXISTS validation_namespaces (
  validation_namespace_id INTEGER PRIMARY KEY,
  problem_type_id INTEGER NOT NULL REFERENCES problem_types(problem_type_id),
  validation_protocol_id INTEGER NOT NULL REFERENCES validation_protocols(validation_protocol_id),
  UNIQUE(problem_type_id, validation_protocol_id)
);

CREATE TABLE IF NOT EXISTS candidates (
  candidate_id INTEGER PRIMARY KEY,
  candidate_hash TEXT NOT NULL UNIQUE,
  params_json TEXT NOT NULL,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS shapes (
  shape_key INTEGER PRIMARY KEY,
  shape_id TEXT NOT NULL UNIQUE,
  m INTEGER NOT NULL,
  n INTEGER NOT NULL,
  batch INTEGER NOT NULL,
  k INTEGER NOT NULL,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS proposal_events (
  proposal_event_id INTEGER PRIMARY KEY,
  benchmark_namespace_id INTEGER NOT NULL REFERENCES benchmark_namespaces(benchmark_namespace_id),
  scope_kind TEXT NOT NULL,
  scope_shape_ids_json TEXT NOT NULL,
  proposal_args_json TEXT NOT NULL,
  island_id TEXT,
  restart_index INTEGER NOT NULL,
  duration_s REAL NOT NULL CHECK(duration_s >= 0),
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS proposal_candidates (
  occurrence_id INTEGER PRIMARY KEY,
  proposal_event_id INTEGER NOT NULL REFERENCES proposal_events(proposal_event_id) ON DELETE CASCADE,
  candidate_id INTEGER NOT NULL REFERENCES candidates(candidate_id),
  source TEXT NOT NULL,
  parent_candidate_ids_json TEXT NOT NULL,
  operator_metadata_json TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('generated', 'preserved')),
  selected INTEGER NOT NULL,
  UNIQUE(proposal_event_id, candidate_id, source, parent_candidate_ids_json, operator_metadata_json, state)
);

CREATE TABLE IF NOT EXISTS baseline_discoveries (
  discovery_id INTEGER PRIMARY KEY,
  problem_type_id INTEGER NOT NULL REFERENCES problem_types(problem_type_id),
  context_json TEXT NOT NULL,
  duration_s REAL NOT NULL CHECK(duration_s >= 0),
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS baseline_selections (
  discovery_id INTEGER NOT NULL REFERENCES baseline_discoveries(discovery_id) ON DELETE CASCADE,
  shape_key INTEGER NOT NULL REFERENCES shapes(shape_key),
  candidate_id INTEGER NOT NULL REFERENCES candidates(candidate_id),
  hipblaslt_solution_index INTEGER NOT NULL,
  hipblaslt_solution_name TEXT,
  hipblaslt_kernel_name TEXT,
  logic_solution_index INTEGER,
  logic_solution_name TEXT,
  query_gflops REAL,
  query_time_us REAL,
  PRIMARY KEY(discovery_id, shape_key)
);

CREATE TABLE IF NOT EXISTS evidence_sources (
  source_id INTEGER PRIMARY KEY,
  source_kind TEXT NOT NULL CHECK(source_kind IN (
    'native_run', 'historical_migration', 'static_rule', 'replay'
  )),
  source_ref TEXT NOT NULL UNIQUE,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS native_runs (
  source_id INTEGER PRIMARY KEY REFERENCES evidence_sources(source_id) ON DELETE CASCADE,
  phase TEXT NOT NULL,
  status TEXT NOT NULL,
  duration_s REAL NOT NULL CHECK(duration_s >= 0),
  returncode INTEGER
);

CREATE TABLE IF NOT EXISTS run_candidate_costs (
  source_id INTEGER NOT NULL REFERENCES native_runs(source_id) ON DELETE CASCADE,
  candidate_id INTEGER NOT NULL REFERENCES candidates(candidate_id),
  phase TEXT NOT NULL,
  duration_s REAL NOT NULL,
  PRIMARY KEY(source_id, candidate_id, phase)
);

CREATE INDEX IF NOT EXISTS idx_run_candidate_costs_candidate
  ON run_candidate_costs(candidate_id, phase);

CREATE TABLE IF NOT EXISTS benchmark_events (
  event_id INTEGER PRIMARY KEY,
  benchmark_namespace_id INTEGER NOT NULL REFERENCES benchmark_namespaces(benchmark_namespace_id),
  shape_key INTEGER NOT NULL REFERENCES shapes(shape_key),
  candidate_id INTEGER NOT NULL REFERENCES candidates(candidate_id),
  source_id INTEGER NOT NULL REFERENCES evidence_sources(source_id),
  status TEXT NOT NULL,
  validation_namespace_id INTEGER REFERENCES validation_namespaces(validation_namespace_id),
  solution_index INTEGER,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmark_samples (
  event_id INTEGER NOT NULL REFERENCES benchmark_events(event_id) ON DELETE CASCADE,
  sample_index INTEGER NOT NULL CHECK(sample_index >= 0),
  time_us REAL NOT NULL CHECK(time_us > 0),
  PRIMARY KEY(event_id, sample_index)
);

CREATE INDEX IF NOT EXISTS idx_benchmark_events_pair
  ON benchmark_events(benchmark_namespace_id, shape_key, candidate_id);

CREATE INDEX IF NOT EXISTS idx_benchmark_events_positive
  ON benchmark_events(benchmark_namespace_id, shape_key, candidate_id, created_at DESC, event_id DESC)
  WHERE status = 'ok';

CREATE INDEX IF NOT EXISTS idx_benchmark_events_negative
  ON benchmark_events(benchmark_namespace_id, shape_key, candidate_id, created_at DESC, event_id DESC)
  WHERE status IN ('rejected', 'build_failed');

CREATE TABLE IF NOT EXISTS validations (
  validation_id INTEGER PRIMARY KEY,
  validation_namespace_id INTEGER NOT NULL REFERENCES validation_namespaces(validation_namespace_id),
  shape_key INTEGER NOT NULL REFERENCES shapes(shape_key),
  candidate_id INTEGER NOT NULL REFERENCES candidates(candidate_id),
  source_id INTEGER NOT NULL REFERENCES evidence_sources(source_id),
  status TEXT NOT NULL,
  detail TEXT,
  solution_index INTEGER,
  created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_validations_latest
  ON validations(validation_namespace_id, shape_key, candidate_id, created_at DESC, validation_id DESC);

CREATE TABLE IF NOT EXISTS artifact_bundles (
  bundle_id INTEGER PRIMARY KEY,
  build_run_id TEXT NOT NULL,
  build_output_dir TEXT NOT NULL,
  library_dir TEXT NOT NULL,
  manifest_path TEXT,
  code_object_identity TEXT NOT NULL,
  created_at REAL NOT NULL,
  UNIQUE(library_dir, code_object_identity)
);

CREATE TABLE IF NOT EXISTS artifact_solution_yamls (
  bundle_id INTEGER NOT NULL REFERENCES artifact_bundles(bundle_id) ON DELETE CASCADE,
  solution_yaml_path TEXT NOT NULL,
  PRIMARY KEY(bundle_id, solution_yaml_path)
);

CREATE TABLE IF NOT EXISTS artifact_mappings (
  mapping_id INTEGER PRIMARY KEY,
  bundle_id INTEGER NOT NULL REFERENCES artifact_bundles(bundle_id) ON DELETE CASCADE,
  problem_type_id INTEGER NOT NULL REFERENCES problem_types(problem_type_id),
  shape_key INTEGER NOT NULL REFERENCES shapes(shape_key),
  candidate_id INTEGER NOT NULL REFERENCES candidates(candidate_id),
  problem_index INTEGER NOT NULL,
  requested_solution_index INTEGER NOT NULL,
  library_solution_index INTEGER NOT NULL,
  manifest_solution_index INTEGER,
  created_at REAL NOT NULL,
  UNIQUE(bundle_id, problem_type_id, shape_key, candidate_id, library_solution_index)
);

CREATE INDEX IF NOT EXISTS idx_artifact_mappings_pair
  ON artifact_mappings(problem_type_id, shape_key, candidate_id, created_at DESC);

"""


@dataclass
class EvoTensileDB:
    path: Path
    environment_compatibility_tag: str

    @classmethod
    def connect(
        cls,
        path: str | Path,
        *,
        environment_compatibility_tag: str | None = None,
    ) -> "EvoTensileDB":
        tag = (environment_compatibility_tag or os.environ.get("EVOTENSILE_ENVIRONMENT_COMPATIBILITY_TAG", "")).strip()
        if not tag:
            raise ValueError("environment compatibility tag is required")
        db = cls(Path(path), tag)
        db._verify_environment_compatibility()
        return db

    def _verify_environment_compatibility(self) -> None:
        if not self.path.exists():
            return
        with sqlite3.connect(self.path, timeout=60.0) as con:
            metadata_table = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'database_metadata'"
            ).fetchone()
            if metadata_table is None:
                has_tables = con.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
                ).fetchone()
                if has_tables is not None:
                    raise ValueError("database does not use the current schema")
                return
            row = con.execute(
                "SELECT metadata_value FROM database_metadata WHERE metadata_key = ?",
                ("environment_compatibility_tag",),
            ).fetchone()
            if row is None:
                raise ValueError("database has no environment compatibility tag")
            actual = str(row[0])
            if actual != self.environment_compatibility_tag:
                raise ValueError(
                    "environment compatibility tag mismatch: "
                    f"database={actual!r}, expected={self.environment_compatibility_tag!r}"
                )

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path, timeout=60.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=60000")
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def init(self) -> None:
        with self.connection() as con:
            con.executescript(SCHEMA)
            con.execute(
                "INSERT OR IGNORE INTO database_metadata(metadata_key, metadata_value) VALUES (?, ?)",
                ("environment_compatibility_tag", self.environment_compatibility_tag),
            )

    @staticmethod
    def _intern_hash(
        con: sqlite3.Connection,
        *,
        table: str,
        id_column: str,
        hash_column: str,
        value: str,
        definition_json: str | None = None,
    ) -> int:
        if not value:
            raise ValueError(f"{hash_column} is required")
        con.execute(
            f"INSERT OR IGNORE INTO {table}({hash_column}, definition_json) VALUES (?, ?)",
            (value, definition_json),
        )
        row = con.execute(f"SELECT {id_column} FROM {table} WHERE {hash_column} = ?", (value,)).fetchone()
        assert row is not None
        return int(row[0])

    @classmethod
    def _benchmark_namespace_id(
        cls,
        con: sqlite3.Connection,
        problem_type_hash: str,
        benchmark_protocol_hash: str,
    ) -> int:
        problem_type_id = cls._intern_hash(
            con,
            table="problem_types",
            id_column="problem_type_id",
            hash_column="problem_type_hash",
            value=problem_type_hash,
        )
        benchmark_protocol_id = cls._intern_hash(
            con,
            table="benchmark_protocols",
            id_column="benchmark_protocol_id",
            hash_column="benchmark_protocol_hash",
            value=benchmark_protocol_hash,
        )
        con.execute(
            "INSERT OR IGNORE INTO benchmark_namespaces(problem_type_id, benchmark_protocol_id) VALUES (?, ?)",
            (problem_type_id, benchmark_protocol_id),
        )
        row = con.execute(
            "SELECT benchmark_namespace_id FROM benchmark_namespaces "
            "WHERE problem_type_id = ? AND benchmark_protocol_id = ?",
            (problem_type_id, benchmark_protocol_id),
        ).fetchone()
        assert row is not None
        return int(row[0])

    @classmethod
    def _validation_namespace_id(
        cls,
        con: sqlite3.Connection,
        problem_type_hash: str,
        validation_protocol_hash: str,
    ) -> int:
        problem_type_id = cls._intern_hash(
            con,
            table="problem_types",
            id_column="problem_type_id",
            hash_column="problem_type_hash",
            value=problem_type_hash,
        )
        validation_protocol_id = cls._intern_hash(
            con,
            table="validation_protocols",
            id_column="validation_protocol_id",
            hash_column="validation_protocol_hash",
            value=validation_protocol_hash,
        )
        con.execute(
            "INSERT OR IGNORE INTO validation_namespaces(problem_type_id, validation_protocol_id) VALUES (?, ?)",
            (problem_type_id, validation_protocol_id),
        )
        row = con.execute(
            "SELECT validation_namespace_id FROM validation_namespaces "
            "WHERE problem_type_id = ? AND validation_protocol_id = ?",
            (problem_type_id, validation_protocol_id),
        ).fetchone()
        assert row is not None
        return int(row[0])

    @staticmethod
    def _candidate_id(con: sqlite3.Connection, candidate_hash: str) -> int:
        row = con.execute("SELECT candidate_id FROM candidates WHERE candidate_hash = ?", (candidate_hash,)).fetchone()
        if row is None:
            raise ValueError(f"candidate is not registered: {candidate_hash}")
        return int(row[0])

    @staticmethod
    def _shape_key(con: sqlite3.Connection, shape_id: str) -> int:
        row = con.execute("SELECT shape_key FROM shapes WHERE shape_id = ?", (shape_id,)).fetchone()
        if row is None:
            raise ValueError(f"shape is not registered: {shape_id}")
        return int(row[0])

    @staticmethod
    def _source_id(con: sqlite3.Connection, source_ref: str) -> int:
        row = con.execute("SELECT source_id FROM evidence_sources WHERE source_ref = ?", (source_ref,)).fetchone()
        if row is None:
            raise ValueError(f"evidence source is not registered: {source_ref}")
        return int(row[0])

    @classmethod
    def _evidence_source_id(
        cls,
        con: sqlite3.Connection,
        *,
        source_kind: str,
        source_ref: str,
        created_at: float,
    ) -> int:
        con.execute(
            """
            INSERT OR IGNORE INTO evidence_sources(source_kind, source_ref, created_at)
            VALUES (?, ?, ?)
            """,
            (source_kind, source_ref, created_at),
        )
        source_id = cls._source_id(con, source_ref)
        row = con.execute("SELECT source_kind FROM evidence_sources WHERE source_id = ?", (source_id,)).fetchone()
        assert row is not None
        if row[0] != source_kind:
            raise ValueError(f"evidence source kind mismatch for {source_ref}: {row[0]} != {source_kind}")
        return source_id

    @classmethod
    def _observation_source_id(
        cls,
        con: sqlite3.Connection,
        *,
        source_kind: str,
        source_ref: str | None,
        created_at: float,
    ) -> int:
        if source_kind == "native_run":
            if not source_ref:
                raise ValueError("native evidence requires a run reference")
            source_id = cls._source_id(con, source_ref)
            native = con.execute("SELECT 1 FROM native_runs WHERE source_id = ?", (source_id,)).fetchone()
            if native is None:
                raise ValueError(f"native run is not registered: {source_ref}")
            return source_id
        if source_kind == "static_rule":
            normalized_ref = source_ref or "static_rule:nt_hhs"
        elif source_kind == "replay":
            if not source_ref:
                raise ValueError("replay evidence requires a source reference")
            normalized_ref = f"replay:{source_ref}"
        else:
            raise ValueError(f"unsupported evidence source kind: {source_kind}")
        return cls._evidence_source_id(
            con,
            source_kind=source_kind,
            source_ref=normalized_ref,
            created_at=created_at,
        )

    @staticmethod
    def _candidate_row(candidate: Candidate, created_at: float) -> tuple[str, str, float]:
        return candidate.hash, canonical_json(candidate.canonical_params()), created_at

    def register_candidates(self, candidates: list[Candidate]) -> None:
        if not candidates:
            return
        now = time.time()
        with self.connection() as con:
            con.executemany(
                "INSERT OR IGNORE INTO candidates(candidate_hash, params_json, created_at) VALUES (?, ?, ?)",
                [self._candidate_row(candidate, now) for candidate in candidates],
            )

    def record_proposal_event(
        self,
        candidates: list[Candidate],
        *,
        problem_type_hash: str,
        benchmark_protocol_hash: str,
        scope_kind: str,
        scope_shape_ids: tuple[str, ...],
        generated_hashes: set[str],
        selected_candidates: list[Candidate],
        proposal_args: dict[str, object] | None = None,
        island_id: str | None = None,
        restart_index: int = 0,
        duration_s: float = 0.0,
    ) -> int | None:
        if not candidates:
            return None
        now = time.time()
        selected_by_hash = {candidate.hash: candidate for candidate in selected_candidates}
        selected_hashes: set[str] = set()
        with self.connection() as con:
            con.executemany(
                "INSERT OR IGNORE INTO candidates(candidate_hash, params_json, created_at) VALUES (?, ?, ?)",
                [self._candidate_row(candidate, now) for candidate in candidates],
            )
            benchmark_namespace_id = self._benchmark_namespace_id(con, problem_type_hash, benchmark_protocol_hash)
            cursor = con.execute(
                """
                INSERT INTO proposal_events
                  (benchmark_namespace_id, scope_kind, scope_shape_ids_json, proposal_args_json,
                   island_id, restart_index, duration_s, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    benchmark_namespace_id,
                    scope_kind,
                    canonical_json(list(scope_shape_ids)),
                    canonical_json(proposal_args or {}),
                    island_id,
                    restart_index,
                    max(0.0, duration_s),
                    now,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("proposal event insert returned no ID")
            proposal_event_id = int(cursor.lastrowid)
            rows = []
            for candidate in candidates:
                selected_candidate = selected_by_hash.get(candidate.hash)
                selected = bool(
                    selected_candidate is not None
                    and candidate.hash not in selected_hashes
                    and candidate.source == selected_candidate.source
                    and candidate.parent_hashes == selected_candidate.parent_hashes
                )
                if selected:
                    selected_hashes.add(candidate.hash)
                parent_ids = [self._candidate_id(con, parent_hash) for parent_hash in candidate.parent_hashes]
                metadata = dict(candidate.proposal_metadata)
                metadata.pop("proposal_scope_kind", None)
                metadata.pop("proposal_scope_shape_ids", None)
                metadata.pop("proposal_cost_s", None)
                metadata.pop("island_id", None)
                metadata.pop("restart_index", None)
                rows.append(
                    (
                        proposal_event_id,
                        self._candidate_id(con, candidate.hash),
                        candidate.source,
                        canonical_json(parent_ids),
                        canonical_json(metadata),
                        "generated" if candidate.hash in generated_hashes else "preserved",
                        int(selected),
                    )
                )
            con.executemany(
                """
                INSERT INTO proposal_candidates
                  (proposal_event_id, candidate_id, source, parent_candidate_ids_json,
                   operator_metadata_json, state, selected)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            return proposal_event_id

    def proposal_candidate_occurrences(
        self,
        *,
        problem_type_hash: str,
        benchmark_protocol_hash: str,
        selected_only: bool = True,
    ) -> list[ProposalOccurrence]:
        selected_clause = "AND pc.selected = 1" if selected_only else ""
        with self.connection() as con:
            rows = con.execute(
                f"""
                SELECT pc.occurrence_id, pe.proposal_event_id, pt.problem_type_hash,
                       bp.benchmark_protocol_hash, c.candidate_hash, pc.source,
                       pc.parent_candidate_ids_json, pc.operator_metadata_json, pc.state,
                       pe.scope_kind, pe.scope_shape_ids_json, pe.island_id, pe.restart_index,
                       pe.duration_s, pc.selected, pe.created_at
                FROM proposal_candidates AS pc
                JOIN proposal_events AS pe USING (proposal_event_id)
                JOIN benchmark_namespaces AS bn USING (benchmark_namespace_id)
                JOIN problem_types AS pt USING (problem_type_id)
                JOIN benchmark_protocols AS bp USING (benchmark_protocol_id)
                JOIN candidates AS c USING (candidate_id)
                WHERE pt.problem_type_hash = ? AND bp.benchmark_protocol_hash = ?
                {selected_clause}
                ORDER BY pc.occurrence_id
                """,
                (problem_type_hash, benchmark_protocol_hash),
            ).fetchall()
            parent_ids = sorted({int(value) for row in rows for value in json.loads(row["parent_candidate_ids_json"])})
            parent_hashes = {}
            if parent_ids:
                placeholders = ",".join("?" for _ in parent_ids)
                parent_hashes = {
                    int(row["candidate_id"]): str(row["candidate_hash"])
                    for row in con.execute(
                        f"SELECT candidate_id, candidate_hash FROM candidates WHERE candidate_id IN ({placeholders})",
                        parent_ids,
                    )
                }
        return [
            ProposalOccurrence(
                occurrence_id=int(row["occurrence_id"]),
                proposal_event_id=int(row["proposal_event_id"]),
                problem_type_hash=str(row["problem_type_hash"]),
                benchmark_protocol_hash=str(row["benchmark_protocol_hash"]),
                candidate_hash=str(row["candidate_hash"]),
                source=str(row["source"]),
                parent_hashes=tuple(
                    parent_hashes[int(value)] for value in json.loads(row["parent_candidate_ids_json"])
                ),
                proposal_metadata=dict(json.loads(row["operator_metadata_json"])),
                state=str(row["state"]),
                scope_kind=str(row["scope_kind"]),
                scope_shape_ids=tuple(json.loads(row["scope_shape_ids_json"])),
                island_id=None if row["island_id"] is None else str(row["island_id"]),
                restart_index=int(row["restart_index"]),
                duration_s=float(row["duration_s"]),
                selected=bool(row["selected"]),
                created_at=float(row["created_at"]),
            )
            for row in rows
        ]

    def record_baseline_discovery(
        self,
        selections: list[BaselineSelectionInsert],
        *,
        problem_type_hash: str,
        context: dict[str, object],
        duration_s: float,
    ) -> int | None:
        if not selections:
            return None
        now = time.time()
        self.register_candidates([selection.candidate for selection in selections])
        self.register_shapes([selection.shape for selection in selections])
        with self.connection() as con:
            problem_type_id = self._intern_hash(
                con,
                table="problem_types",
                id_column="problem_type_id",
                hash_column="problem_type_hash",
                value=problem_type_hash,
            )
            cursor = con.execute(
                "INSERT INTO baseline_discoveries(problem_type_id, context_json, duration_s, created_at) "
                "VALUES (?, ?, ?, ?)",
                (problem_type_id, canonical_json(context), max(0.0, duration_s), now),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("baseline discovery insert returned no ID")
            discovery_id = int(cursor.lastrowid)
            con.executemany(
                """
                INSERT INTO baseline_selections
                  (discovery_id, shape_key, candidate_id, hipblaslt_solution_index,
                   hipblaslt_solution_name, hipblaslt_kernel_name, logic_solution_index,
                   logic_solution_name, query_gflops, query_time_us)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        discovery_id,
                        self._shape_key(con, selection.shape.id),
                        self._candidate_id(con, selection.candidate.hash),
                        selection.hipblaslt_solution_index,
                        selection.hipblaslt_solution_name,
                        selection.hipblaslt_kernel_name,
                        selection.logic_solution_index,
                        selection.logic_solution_name,
                        selection.query_gflops,
                        selection.query_time_us,
                    )
                    for selection in selections
                ],
            )
            return discovery_id

    def baseline_selection_pairs(self, discovery_id: int) -> list[tuple[Shape, Candidate]]:
        with self.connection() as con:
            rows = con.execute(
                """
                SELECT s.m, s.n, s.batch, s.k, c.candidate_hash, c.params_json
                FROM baseline_selections AS bs
                JOIN shapes AS s USING (shape_key)
                JOIN candidates AS c USING (candidate_id)
                WHERE bs.discovery_id = ? ORDER BY s.shape_key
                """,
                (discovery_id,),
            ).fetchall()
        return [
            (
                Shape(m=int(row["m"]), n=int(row["n"]), batch=int(row["batch"]), k=int(row["k"])),
                Candidate(params=dict(json.loads(row["params_json"])), source="installed_hipblaslt_baseline"),
            )
            for row in rows
        ]

    def register_shapes(self, shapes: list[Shape]) -> None:
        if not shapes:
            return
        now = time.time()
        with self.connection() as con:
            con.executemany(
                """
                INSERT OR IGNORE INTO shapes
                  (shape_id, m, n, batch, k, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        shape.id,
                        shape.m,
                        shape.n,
                        shape.batch,
                        shape.k,
                        now,
                    )
                    for shape in shapes
                ],
            )

    def get_candidates(self, candidate_hashes: list[str]) -> list[Candidate]:
        if not candidate_hashes:
            return []
        placeholders = ",".join("?" for _ in candidate_hashes)
        with self.connection() as con:
            rows = con.execute(
                f"""
                SELECT candidate_hash, params_json
                FROM candidates
                WHERE candidate_hash IN ({placeholders})
                """,
                candidate_hashes,
            ).fetchall()
        by_hash: dict[str, Candidate] = {}
        for row in rows:
            by_hash[row["candidate_hash"]] = Candidate(params=dict(json.loads(row["params_json"])))
        return [by_hash[h] for h in candidate_hashes if h in by_hash]

    def insert_run(
        self,
        run_id: str,
        *,
        phase: str,
        status: str,
        duration_s: float,
        returncode: int | None = None,
        candidate_hashes: list[str] | None = None,
    ) -> None:
        if duration_s < 0.0:
            raise ValueError("native run duration must be non-negative")
        with self.connection() as con:
            source_id = self._evidence_source_id(
                con,
                source_kind="native_run",
                source_ref=run_id,
                created_at=time.time(),
            )
            con.execute(
                """
                INSERT OR REPLACE INTO native_runs(source_id, phase, status, duration_s, returncode)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_id, phase, status, duration_s, returncode),
            )
            con.execute("DELETE FROM run_candidate_costs WHERE source_id = ?", (source_id,))
            attributed_hashes = sorted(set(candidate_hashes or ()))
            if attributed_hashes:
                share = duration_s / len(attributed_hashes)
                con.executemany(
                    """
                    INSERT INTO run_candidate_costs(source_id, candidate_id, phase, duration_s)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (source_id, self._candidate_id(con, candidate_hash), phase, share)
                        for candidate_hash in attributed_hashes
                    ],
                )

    def insert_benchmark_events(self, events: list[BenchmarkEventInsert]) -> None:
        if not events:
            return
        now = time.time()
        with self.connection() as con:
            for event in events:
                if event.status == "ok":
                    if not event.samples_us or any(
                        not math.isfinite(time_us) or time_us <= 0.0 for time_us in event.samples_us
                    ):
                        raise ValueError("successful benchmark event requires finite positive samples")
                elif event.samples_us:
                    raise ValueError("negative benchmark event cannot contain timing samples")
                benchmark_namespace_id = self._benchmark_namespace_id(
                    con,
                    event.problem_type_hash,
                    event.benchmark_protocol_hash,
                )
                shape_key = self._shape_key(con, event.shape_id)
                candidate_id = self._candidate_id(con, event.candidate_hash)
                validation_namespace_id = None
                if event.status == "ok":
                    if not event.validation_protocol_hash:
                        raise ValueError("successful benchmark event requires a validation protocol")
                    validation_namespace_id = self._validation_namespace_id(
                        con,
                        event.problem_type_hash,
                        event.validation_protocol_hash,
                    )
                    latest_validation = con.execute(
                        """
                        SELECT status FROM validations
                        WHERE validation_namespace_id = ? AND shape_key = ? AND candidate_id = ?
                        ORDER BY created_at DESC, validation_id DESC LIMIT 1
                        """,
                        (validation_namespace_id, shape_key, candidate_id),
                    ).fetchone()
                    if latest_validation is None or latest_validation[0] != "passed":
                        raise ValueError("successful benchmark event requires latest compatible validation pass")
                source_id = self._observation_source_id(
                    con,
                    source_kind=event.source_kind,
                    source_ref=event.run_id,
                    created_at=now,
                )
                values = (
                    benchmark_namespace_id,
                    shape_key,
                    candidate_id,
                    source_id,
                    event.status,
                    validation_namespace_id,
                    event.solution_index,
                    now,
                )
                if event.status in NEGATIVE_CACHE_STATUSES:
                    cursor = con.execute(
                        """
                        INSERT INTO benchmark_events
                          (benchmark_namespace_id, shape_key, candidate_id, source_id, status,
                           validation_namespace_id, solution_index, created_at)
                        SELECT ?, ?, ?, ?, ?, ?, ?, ?
                        WHERE NOT EXISTS (
                          SELECT 1 FROM benchmark_events
                          WHERE benchmark_namespace_id = ?
                            AND shape_key = ?
                            AND candidate_id = ?
                            AND source_id = ?
                            AND status = ?
                            AND solution_index IS ?
                        )
                        """,
                        (
                            *values,
                            benchmark_namespace_id,
                            shape_key,
                            candidate_id,
                            source_id,
                            event.status,
                            event.solution_index,
                        ),
                    )
                    if cursor.rowcount == 0:
                        continue
                else:
                    cursor = con.execute(
                        """
                        INSERT INTO benchmark_events
                          (benchmark_namespace_id, shape_key, candidate_id, source_id, status,
                           validation_namespace_id, solution_index, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        values,
                    )
                if cursor.lastrowid is None:
                    raise RuntimeError("benchmark event insert did not return an event ID")
                event_id = cursor.lastrowid
                con.executemany(
                    "INSERT INTO benchmark_samples(event_id, sample_index, time_us) VALUES (?, ?, ?)",
                    [(event_id, sample_index, time_us) for sample_index, time_us in enumerate(event.samples_us)],
                )

    def insert_validations(self, validations: list[ValidationInsert]) -> None:
        if not validations:
            return
        now = time.time()
        with self.connection() as con:
            con.executemany(
                """
                INSERT INTO validations
                  (validation_namespace_id, shape_key, candidate_id, source_id, status,
                   detail, solution_index, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        self._validation_namespace_id(
                            con,
                            validation.problem_type_hash,
                            validation.validation_protocol_hash,
                        ),
                        self._shape_key(con, validation.shape_id),
                        self._candidate_id(con, validation.candidate_hash),
                        self._observation_source_id(
                            con,
                            source_kind=validation.source_kind,
                            source_ref=validation.run_id,
                            created_at=now,
                        ),
                        validation.status,
                        validation.detail,
                        validation.solution_index,
                        now,
                    )
                    for validation in validations
                ],
            )

    def insert_artifact_bundle(
        self,
        bundle: ArtifactBundleInsert,
        mappings: list[ArtifactMappingInsert],
    ) -> int | None:
        if not mappings:
            return None
        now = time.time()
        with self.connection() as con:
            con.execute(
                """
                INSERT INTO artifact_bundles
                  (build_run_id, build_output_dir, library_dir, manifest_path,
                   code_object_identity, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(library_dir, code_object_identity) DO UPDATE SET
                  build_run_id = excluded.build_run_id,
                  build_output_dir = excluded.build_output_dir,
                  manifest_path = excluded.manifest_path,
                  created_at = excluded.created_at
                """,
                (
                    bundle.build_run_id,
                    bundle.build_output_dir,
                    bundle.library_dir,
                    bundle.manifest_path,
                    bundle.code_object_identity,
                    now,
                ),
            )
            row = con.execute(
                "SELECT bundle_id FROM artifact_bundles WHERE library_dir = ? AND code_object_identity = ?",
                (bundle.library_dir, bundle.code_object_identity),
            ).fetchone()
            assert row is not None
            bundle_id = int(row["bundle_id"])
            con.execute("DELETE FROM artifact_solution_yamls WHERE bundle_id = ?", (bundle_id,))
            con.executemany(
                "INSERT INTO artifact_solution_yamls(bundle_id, solution_yaml_path) VALUES (?, ?)",
                [(bundle_id, path) for path in bundle.solution_yaml_paths],
            )
            for mapping in mappings:
                problem_type_id = self._intern_hash(
                    con,
                    table="problem_types",
                    id_column="problem_type_id",
                    hash_column="problem_type_hash",
                    value=mapping.problem_type_hash,
                )
                con.execute(
                    """
                    INSERT INTO artifact_mappings
                      (bundle_id, problem_type_id, shape_key, candidate_id, problem_index,
                       requested_solution_index, library_solution_index, manifest_solution_index, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(bundle_id, problem_type_id, shape_key, candidate_id, library_solution_index)
                    DO UPDATE SET
                      problem_index = excluded.problem_index,
                      requested_solution_index = excluded.requested_solution_index,
                      manifest_solution_index = excluded.manifest_solution_index,
                      created_at = excluded.created_at
                    """,
                    (
                        bundle_id,
                        problem_type_id,
                        self._shape_key(con, mapping.shape_id),
                        self._candidate_id(con, mapping.candidate_hash),
                        mapping.problem_index,
                        mapping.requested_solution_index,
                        mapping.library_solution_index,
                        mapping.manifest_solution_index,
                        now,
                    ),
                )
            return bundle_id

    def candidate_artifact_records(
        self,
        *,
        problem_type_hash: str,
        shape_ids: list[str] | None = None,
        candidate_hashes: list[str] | None = None,
    ) -> list[CandidateArtifactRecord]:
        clauses = ["pt.problem_type_hash = ?"]
        params: list[str | int] = [problem_type_hash]
        if shape_ids is not None:
            if not shape_ids:
                return []
            placeholders = ",".join("?" for _ in shape_ids)
            clauses.append(f"s.shape_id IN ({placeholders})")
            params.extend(shape_ids)
        if candidate_hashes is not None:
            if not candidate_hashes:
                return []
            placeholders = ",".join("?" for _ in candidate_hashes)
            clauses.append(f"c.candidate_hash IN ({placeholders})")
            params.extend(candidate_hashes)
        with self.connection() as con:
            rows = con.execute(
                f"""
                SELECT am.mapping_id, ab.bundle_id, pt.problem_type_hash, s.shape_id,
                       c.candidate_hash, am.problem_index, am.requested_solution_index,
                       am.library_solution_index, am.manifest_solution_index, ab.build_run_id,
                       ab.build_output_dir, ab.library_dir, ab.manifest_path,
                       ab.code_object_identity, am.created_at
                FROM artifact_mappings AS am
                JOIN artifact_bundles AS ab USING (bundle_id)
                JOIN problem_types AS pt USING (problem_type_id)
                JOIN shapes AS s USING (shape_key)
                JOIN candidates AS c USING (candidate_id)
                WHERE {" AND ".join(clauses)}
                ORDER BY am.created_at DESC, am.mapping_id DESC
                """,
                params,
            ).fetchall()
            bundle_ids = sorted({int(row["bundle_id"]) for row in rows})
            solution_paths: dict[int, tuple[str, ...]] = {}
            if bundle_ids:
                placeholders = ",".join("?" for _ in bundle_ids)
                path_rows = con.execute(
                    f"SELECT bundle_id, solution_yaml_path FROM artifact_solution_yamls "
                    f"WHERE bundle_id IN ({placeholders}) ORDER BY bundle_id, solution_yaml_path",
                    bundle_ids,
                ).fetchall()
                for bundle_id in bundle_ids:
                    solution_paths[bundle_id] = tuple(
                        str(path_row["solution_yaml_path"])
                        for path_row in path_rows
                        if int(path_row["bundle_id"]) == bundle_id
                    )
        return [
            CandidateArtifactRecord(
                **dict(row),
                solution_yaml_paths=solution_paths[int(row["bundle_id"])],
            )
            for row in rows
        ]

    def benchmark_evidence_states(
        self,
        *,
        problem_type_hash: str,
        benchmark_protocol_hash: str,
        shape_ids: list[str],
        candidate_hashes: list[str],
    ) -> dict[tuple[str, str], BenchmarkEvidenceState]:
        if not shape_ids or not candidate_hashes:
            return {}
        shape_placeholders = ",".join("?" for _ in shape_ids)
        candidate_placeholders = ",".join("?" for _ in candidate_hashes)
        negative_placeholders = ",".join("?" for _ in NEGATIVE_CACHE_STATUSES)
        common_params = (problem_type_hash, benchmark_protocol_hash, *shape_ids, *candidate_hashes)
        with self.connection() as con:
            rows = con.execute(
                f"""
                WITH compatible AS (
                  SELECT be.event_id, be.status, bs.time_us, be.validation_namespace_id, be.created_at,
                         be.shape_key, be.candidate_id, s.shape_id, c.candidate_hash
                  FROM benchmark_events AS be
                  LEFT JOIN benchmark_samples AS bs USING (event_id)
                  JOIN benchmark_namespaces AS bn USING (benchmark_namespace_id)
                  JOIN problem_types AS pt USING (problem_type_id)
                  JOIN benchmark_protocols AS bp USING (benchmark_protocol_id)
                  JOIN shapes AS s USING (shape_key)
                  JOIN candidates AS c USING (candidate_id)
                  WHERE pt.problem_type_hash = ?
                    AND bp.benchmark_protocol_hash = ?
                    AND s.shape_id IN ({shape_placeholders})
                    AND c.candidate_hash IN ({candidate_placeholders})
                ),
                positive AS (
                  SELECT shape_id, candidate_hash, COUNT(*) AS ok_samples
                  FROM compatible
                  WHERE status = 'ok'
                    AND time_us IS NOT NULL AND time_us > 0
                    AND EXISTS (
                      SELECT 1 FROM validations AS v
                      WHERE v.validation_namespace_id = compatible.validation_namespace_id
                        AND v.shape_key = compatible.shape_key
                        AND v.candidate_id = compatible.candidate_id
                        AND v.status = 'passed'
                        AND NOT EXISTS (
                          SELECT 1 FROM validations AS newer
                          WHERE newer.validation_namespace_id = v.validation_namespace_id
                            AND newer.shape_key = v.shape_key AND newer.candidate_id = v.candidate_id
                            AND (newer.created_at > v.created_at OR
                                 (newer.created_at = v.created_at AND newer.validation_id > v.validation_id))
                        )
                    )
                  GROUP BY shape_id, candidate_hash
                ),
                negative AS (
                  SELECT shape_id, candidate_hash, status, event_id
                  FROM (
                    SELECT shape_id, candidate_hash, status, event_id,
                           ROW_NUMBER() OVER (
                             PARTITION BY shape_id, candidate_hash
                             ORDER BY created_at DESC, event_id DESC
                           ) AS evidence_rank
                    FROM compatible
                    WHERE status IN ({negative_placeholders})
                  )
                  WHERE evidence_rank = 1
                ),
                evidence_keys AS (
                  SELECT shape_id, candidate_hash FROM positive
                  UNION
                  SELECT shape_id, candidate_hash FROM negative
                )
                SELECT evidence_keys.shape_id, evidence_keys.candidate_hash,
                       COALESCE(positive.ok_samples, 0) AS ok_samples,
                       negative.status AS latest_negative_status,
                       negative.event_id AS latest_negative_event_id
                FROM evidence_keys
                LEFT JOIN positive USING (shape_id, candidate_hash)
                LEFT JOIN negative USING (shape_id, candidate_hash)
                """,
                (*common_params, *NEGATIVE_CACHE_STATUSES),
            ).fetchall()

        states: dict[tuple[str, str], BenchmarkEvidenceState] = {}
        for row in rows:
            ok_samples = int(row["ok_samples"])
            latest_negative_status = row["latest_negative_status"]
            states[(row["shape_id"], row["candidate_hash"])] = BenchmarkEvidenceState(
                ok_samples=ok_samples,
                resolved_status="ok" if ok_samples > 0 else latest_negative_status,
                latest_negative_event_id=(
                    None if row["latest_negative_event_id"] is None else int(row["latest_negative_event_id"])
                ),
            )
        return states

    def validation_cache_states(
        self,
        *,
        problem_type_hash: str,
        validation_protocol_hash: str,
        shape_ids: list[str],
        candidate_hashes: list[str],
    ) -> dict[tuple[str, str], str]:
        if not shape_ids or not candidate_hashes:
            return {}
        shape_placeholders = ",".join("?" for _ in shape_ids)
        candidate_placeholders = ",".join("?" for _ in candidate_hashes)
        with self.connection() as con:
            rows = con.execute(
                f"""
                SELECT shape_id, candidate_hash, status
                FROM (
                  SELECT s.shape_id, c.candidate_hash, v.status,
                         ROW_NUMBER() OVER (
                           PARTITION BY v.shape_key, v.candidate_id
                           ORDER BY v.created_at DESC, v.validation_id DESC
                         ) AS evidence_rank
                  FROM validations AS v
                  JOIN validation_namespaces AS vn USING (validation_namespace_id)
                  JOIN problem_types AS pt USING (problem_type_id)
                  JOIN validation_protocols AS vp USING (validation_protocol_id)
                  JOIN shapes AS s USING (shape_key)
                  JOIN candidates AS c USING (candidate_id)
                  WHERE pt.problem_type_hash = ?
                    AND vp.validation_protocol_hash = ?
                    AND s.shape_id IN ({shape_placeholders})
                    AND c.candidate_hash IN ({candidate_placeholders})
                )
                WHERE evidence_rank = 1
                """,
                (problem_type_hash, validation_protocol_hash, *shape_ids, *candidate_hashes),
            ).fetchall()
        return {(row["shape_id"], row["candidate_hash"]): row["status"] for row in rows}

    def validated_cache_entries(
        self,
        *,
        problem_type_hash: str,
        validation_protocol_hash: str,
        shape_ids: list[str],
        candidate_hashes: list[str],
    ) -> set[tuple[str, str]]:
        states = self.validation_cache_states(
            problem_type_hash=problem_type_hash,
            validation_protocol_hash=validation_protocol_hash,
            shape_ids=shape_ids,
            candidate_hashes=candidate_hashes,
        )
        return {key for key, status in states.items() if status == "passed"}

    def reusable_cache_entries(
        self,
        *,
        problem_type_hash: str,
        benchmark_protocol_hash: str,
        shape_ids: list[str],
        candidate_hashes: list[str],
        min_ok_samples: int = 1,
    ) -> set[tuple[str, str]]:
        states = self.benchmark_evidence_states(
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            shape_ids=shape_ids,
            candidate_hashes=candidate_hashes,
        )
        return {key for key, state in states.items() if state.ok_samples >= min_ok_samples or state.reusable_negative}

    def evidence_status_counts(
        self,
        *,
        problem_type_hash: str,
        benchmark_protocol_hash: str,
        shape_ids: set[str] | None,
    ) -> dict[str, dict[str, int]]:
        clauses = ["pt.problem_type_hash = ?", "bp.benchmark_protocol_hash = ?"]
        params: list[str] = [problem_type_hash, benchmark_protocol_hash]
        if shape_ids is not None:
            if not shape_ids:
                return {}
            placeholders = ",".join("?" for _ in shape_ids)
            clauses.append(f"s.shape_id IN ({placeholders})")
            params.extend(sorted(shape_ids))
        counts: dict[str, dict[str, int]] = {}
        with self.connection() as con:
            benchmark_rows = con.execute(
                f"""
                SELECT c.candidate_hash, be.status, COUNT(*) AS n
                FROM benchmark_events AS be
                LEFT JOIN benchmark_samples AS bs USING (event_id)
                JOIN benchmark_namespaces AS bn USING (benchmark_namespace_id)
                JOIN problem_types AS pt USING (problem_type_id)
                JOIN benchmark_protocols AS bp USING (benchmark_protocol_id)
                JOIN shapes AS s USING (shape_key)
                JOIN candidates AS c USING (candidate_id)
                WHERE {" AND ".join(clauses)}
                GROUP BY be.candidate_id, be.status
                """,
                params,
            ).fetchall()
            validation_clauses = ["pt.problem_type_hash = ?"]
            validation_params: list[str] = [problem_type_hash]
            if shape_ids is not None:
                placeholders = ",".join("?" for _ in shape_ids)
                validation_clauses.append(f"s.shape_id IN ({placeholders})")
                validation_params.extend(sorted(shape_ids))
            validation_rows = con.execute(
                f"""
                SELECT c.candidate_hash, v.status, COUNT(*) AS n
                FROM validations AS v
                JOIN validation_namespaces AS vn USING (validation_namespace_id)
                JOIN problem_types AS pt USING (problem_type_id)
                JOIN shapes AS s USING (shape_key)
                JOIN candidates AS c USING (candidate_id)
                WHERE {" AND ".join(validation_clauses)}
                GROUP BY v.candidate_id, v.status
                """,
                validation_params,
            ).fetchall()
        for row in benchmark_rows:
            counts.setdefault(str(row["candidate_hash"]), {})[str(row["status"])] = int(row["n"])
        for row in validation_rows:
            counts.setdefault(str(row["candidate_hash"]), {})[f"validation_{row['status']}"] = int(row["n"])
        return counts

    def latest_positive_benchmark_times(
        self,
        *,
        problem_type_hash: str,
        benchmark_protocol_hash: str,
    ) -> dict[tuple[str, str], float]:
        with self.connection() as con:
            rows = con.execute(
                """
                SELECT s.shape_id, c.candidate_hash, MAX(be.created_at) AS latest_created_at
                FROM benchmark_events AS be
                JOIN benchmark_samples AS bs USING (event_id)
                JOIN benchmark_namespaces AS bn USING (benchmark_namespace_id)
                JOIN problem_types AS pt USING (problem_type_id)
                JOIN benchmark_protocols AS bp USING (benchmark_protocol_id)
                JOIN shapes AS s USING (shape_key)
                JOIN candidates AS c USING (candidate_id)
                WHERE pt.problem_type_hash = ?
                  AND bp.benchmark_protocol_hash = ?
                  AND be.status = 'ok'
                  AND EXISTS (
                    SELECT 1 FROM validations AS v
                    WHERE v.validation_namespace_id = be.validation_namespace_id
                      AND v.shape_key = be.shape_key AND v.candidate_id = be.candidate_id
                      AND v.status = 'passed'
                      AND NOT EXISTS (
                        SELECT 1 FROM validations AS newer
                        WHERE newer.validation_namespace_id = v.validation_namespace_id
                          AND newer.shape_key = v.shape_key AND newer.candidate_id = v.candidate_id
                          AND (newer.created_at > v.created_at OR
                               (newer.created_at = v.created_at AND newer.validation_id > v.validation_id))
                      )
                  )
                GROUP BY be.shape_key, be.candidate_id
                """,
                (problem_type_hash, benchmark_protocol_hash),
            ).fetchall()
        return {(str(row["shape_id"]), str(row["candidate_hash"])): float(row["latest_created_at"]) for row in rows}

    def rank_benchmarks(
        self,
        *,
        problem_type_hash: str | None = None,
        benchmark_protocol_hash: str | None = None,
        shape_id: str | None = None,
        min_samples: int = 1,
        limit: int | None = None,
    ) -> list[BenchmarkSummary]:
        clauses = [
            "be.status = 'ok'",
            "EXISTS (SELECT 1 FROM validations AS v "
            "WHERE v.validation_namespace_id = be.validation_namespace_id "
            "AND v.shape_key = be.shape_key AND v.candidate_id = be.candidate_id AND v.status = 'passed' "
            "AND NOT EXISTS (SELECT 1 FROM validations AS newer "
            "WHERE newer.validation_namespace_id = v.validation_namespace_id "
            "AND newer.shape_key = v.shape_key AND newer.candidate_id = v.candidate_id "
            "AND (newer.created_at > v.created_at OR "
            "(newer.created_at = v.created_at AND newer.validation_id > v.validation_id))))",
        ]
        params: list[str] = []
        if problem_type_hash is not None:
            clauses.append("pt.problem_type_hash = ?")
            params.append(problem_type_hash)
        if benchmark_protocol_hash is not None:
            clauses.append("bp.benchmark_protocol_hash = ?")
            params.append(benchmark_protocol_hash)
        if shape_id is not None:
            clauses.append("s.shape_id = ?")
            params.append(shape_id)
        where = "WHERE " + " AND ".join(clauses)
        with self.connection() as con:
            rows = con.execute(
                f"""
                SELECT s.shape_id, c.candidate_hash, bs.time_us, s.m, s.n, s.batch, s.k
                FROM benchmark_events AS be
                JOIN benchmark_samples AS bs USING (event_id)
                JOIN benchmark_namespaces AS bn USING (benchmark_namespace_id)
                JOIN problem_types AS pt USING (problem_type_id)
                JOIN benchmark_protocols AS bp USING (benchmark_protocol_id)
                JOIN shapes AS s USING (shape_key)
                JOIN candidates AS c USING (candidate_id)
                {where}
                """,
                params,
            ).fetchall()
        grouped: dict[tuple[str, str], _TimingBucket] = {}
        for row in rows:
            key = (row["shape_id"], row["candidate_hash"])
            bucket = grouped.setdefault(
                key,
                _TimingBucket(
                    shape=Shape(
                        m=int(row["m"]),
                        n=int(row["n"]),
                        batch=int(row["batch"]),
                        k=int(row["k"]),
                    ),
                    time_us=[],
                ),
            )
            if row["time_us"] is not None:
                bucket.time_us.append(float(row["time_us"]))

        summaries: list[BenchmarkSummary] = []
        for (sid, chash), bucket in grouped.items():
            samples = len(bucket.time_us)
            if samples < min_samples:
                continue
            gflops_values = [gflops_from_us(bucket.shape, time_us) for time_us in bucket.time_us]
            summaries.append(
                BenchmarkSummary(
                    shape_id=sid,
                    candidate_hash=chash,
                    samples=samples,
                    median_gflops=_median(gflops_values),
                    best_gflops=max(gflops_values) if gflops_values else None,
                    median_time_us=_median(bucket.time_us),
                    best_time_us=min(bucket.time_us) if bucket.time_us else None,
                )
            )

        def sort_key(summary: BenchmarkSummary) -> tuple[int, float, float]:
            if summary.median_time_us is not None:
                return (1, -summary.median_time_us, summary.median_gflops or 0.0)
            return (0, 0.0, 0.0)

        summaries.sort(key=sort_key, reverse=True)
        return summaries[:limit] if limit is not None else summaries

    def benchmark_status_summary(
        self,
        *,
        problem_type_hash: str | None = None,
        benchmark_protocol_hash: str | None = None,
    ) -> dict[str, int]:
        clauses = []
        params: list[str] = []
        if problem_type_hash is not None:
            clauses.append("pt.problem_type_hash = ?")
            params.append(problem_type_hash)
        if benchmark_protocol_hash is not None:
            clauses.append("bp.benchmark_protocol_hash = ?")
            params.append(benchmark_protocol_hash)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connection() as con:
            rows = con.execute(
                f"""
                SELECT be.status,
                       SUM(CASE WHEN be.status = 'ok' THEN 1 ELSE 1 END) AS n
                FROM benchmark_events AS be
                LEFT JOIN benchmark_samples AS bs USING (event_id)
                JOIN benchmark_namespaces AS bn USING (benchmark_namespace_id)
                JOIN problem_types AS pt USING (problem_type_id)
                JOIN benchmark_protocols AS bp USING (benchmark_protocol_id)
                {where}
                GROUP BY be.status
                ORDER BY be.status
                """,
                params,
            ).fetchall()
            return {row["status"]: int(row["n"]) for row in rows}

    def counts(self) -> dict[str, int]:
        with self.connection() as con:
            return {
                table: con.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
                for table in [
                    "candidates",
                    "shapes",
                    "proposal_events",
                    "proposal_candidates",
                    "baseline_discoveries",
                    "baseline_selections",
                    "evidence_sources",
                    "native_runs",
                    "benchmark_events",
                    "benchmark_samples",
                    "validations",
                    "artifact_bundles",
                    "artifact_solution_yamls",
                    "artifact_mappings",
                ]
            }
