import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .cache import DEFAULT_VERSION_NAME, CacheKey, normalize_version_name
from .candidate import Candidate, Shape


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


@dataclass(frozen=True)
class EvaluationSummary:
    shape_id: str
    candidate_hash: str
    samples: int
    median_gflops: float | None
    best_gflops: float | None
    median_time_us: float | None
    best_time_us: float | None


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS candidates (
  candidate_hash TEXT PRIMARY KEY,
  candidate_json TEXT NOT NULL,
  source TEXT NOT NULL,
  parent_hashes TEXT NOT NULL,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS shapes (
  shape_id TEXT PRIMARY KEY,
  m INTEGER NOT NULL,
  n INTEGER NOT NULL,
  batch INTEGER NOT NULL,
  k INTEGER NOT NULL,
  features_json TEXT NOT NULL,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  timestamp REAL NOT NULL,
  version_name TEXT NOT NULL DEFAULT 'unversioned',
  problem_type_hash TEXT,
  benchmark_protocol_hash TEXT,
  yaml_path TEXT,
  output_dir TEXT,
  tensile_bin TEXT,
  status TEXT NOT NULL,
  returncode INTEGER,
  stdout_path TEXT,
  stderr_path TEXT,
  metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS evaluations (
  eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
  version_name TEXT NOT NULL DEFAULT 'unversioned',
  problem_type_hash TEXT NOT NULL DEFAULT '',
  benchmark_protocol_hash TEXT NOT NULL DEFAULT '',
  shape_id TEXT NOT NULL,
  candidate_hash TEXT NOT NULL,
  run_id TEXT,
  status TEXT NOT NULL,
  time_us REAL,
  gflops REAL,
  validation TEXT,
  solution_index INTEGER,
  raw_csv_row TEXT,
  created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evaluations_cache_key
  ON evaluations(version_name, problem_type_hash, benchmark_protocol_hash, shape_id, candidate_hash);

CREATE INDEX IF NOT EXISTS idx_evaluations_shape_candidate
  ON evaluations(shape_id, candidate_hash);

CREATE INDEX IF NOT EXISTS idx_evaluations_shape_time
  ON evaluations(version_name, problem_type_hash, benchmark_protocol_hash, shape_id, time_us);
"""


@dataclass
class EvoTensileDB:
    path: Path

    @classmethod
    def connect(cls, path: str | Path) -> "EvoTensileDB":
        return cls(Path(path))

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def init(self) -> None:
        with self.connection() as con:
            con.executescript(SCHEMA)

    def upsert_candidate(self, candidate: Candidate) -> None:
        with self.connection() as con:
            con.execute(
                """
                INSERT OR IGNORE INTO candidates
                  (candidate_hash, candidate_json, source, parent_hashes, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    candidate.hash,
                    candidate.to_json(),
                    candidate.source,
                    json.dumps(list(candidate.parent_hashes)),
                    time.time(),
                ),
            )

    def upsert_shape(self, shape: Shape) -> None:
        with self.connection() as con:
            con.execute(
                """
                INSERT OR IGNORE INTO shapes
                  (shape_id, m, n, batch, k, features_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    shape.id,
                    shape.m,
                    shape.n,
                    shape.batch,
                    shape.k,
                    json.dumps(shape.features(), sort_keys=True),
                    time.time(),
                ),
            )

    def register_candidates(self, candidates: list[Candidate]) -> None:
        for candidate in candidates:
            self.upsert_candidate(candidate)

    def register_shapes(self, shapes: list[Shape]) -> None:
        for shape in shapes:
            self.upsert_shape(shape)

    def insert_run(
        self,
        run_id: str,
        *,
        yaml_path: str | None,
        output_dir: str | None,
        tensile_bin: str | None,
        status: str,
        version_name: str | None = None,
        problem_type_hash: str | None = None,
        benchmark_protocol_hash: str | None = None,
        returncode: int | None = None,
        stdout_path: str | None = None,
        stderr_path: str | None = None,
        metadata_json: str | None = None,
    ) -> None:
        with self.connection() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO runs
                  (run_id, timestamp, version_name, problem_type_hash, benchmark_protocol_hash,
                   yaml_path, output_dir, tensile_bin, status, returncode, stdout_path, stderr_path,
                   metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    time.time(),
                    normalize_version_name(version_name),
                    problem_type_hash,
                    benchmark_protocol_hash,
                    yaml_path,
                    output_dir,
                    tensile_bin,
                    status,
                    returncode,
                    stdout_path,
                    stderr_path,
                    metadata_json,
                ),
            )

    def insert_evaluation(
        self,
        *,
        shape_id: str,
        candidate_hash: str,
        run_id: str | None,
        status: str,
        version_name: str | None = None,
        problem_type_hash: str = "",
        benchmark_protocol_hash: str = "",
        time_us: float | None = None,
        gflops: float | None = None,
        validation: str | None = None,
        solution_index: int | None = None,
        raw_csv_row: str | None = None,
    ) -> None:
        with self.connection() as con:
            con.execute(
                """
                INSERT INTO evaluations
                  (version_name, problem_type_hash, benchmark_protocol_hash, shape_id, candidate_hash,
                   run_id, status, time_us, gflops, validation, solution_index, raw_csv_row, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalize_version_name(version_name),
                    problem_type_hash,
                    benchmark_protocol_hash,
                    shape_id,
                    candidate_hash,
                    run_id,
                    status,
                    time_us,
                    gflops,
                    validation,
                    solution_index,
                    raw_csv_row,
                    time.time(),
                ),
            )

    def cached_evaluation_count(
        self,
        *,
        version_name: str | None,
        problem_type_hash: str,
        benchmark_protocol_hash: str,
        shape_id: str,
        candidate_hash: str,
        statuses: tuple[str, ...] = ("ok",),
    ) -> int:
        placeholders = ",".join("?" for _ in statuses)
        with self.connection() as con:
            row = con.execute(
                f"""
                SELECT COUNT(*) AS n
                FROM evaluations
                WHERE version_name = ?
                  AND problem_type_hash = ?
                  AND benchmark_protocol_hash = ?
                  AND shape_id = ?
                  AND candidate_hash = ?
                  AND status IN ({placeholders})
                """,
                (
                    normalize_version_name(version_name),
                    problem_type_hash,
                    benchmark_protocol_hash,
                    shape_id,
                    candidate_hash,
                    *statuses,
                ),
            ).fetchone()
            return int(row["n"])

    def has_cached_evaluation(self, key: CacheKey, *, min_samples: int = 1) -> bool:
        return (
            self.cached_evaluation_count(
                version_name=key.version_name,
                problem_type_hash=key.problem_type_hash,
                benchmark_protocol_hash=key.benchmark_protocol_hash,
                shape_id=key.shape_id,
                candidate_hash=key.candidate_hash,
            )
            >= min_samples
        )

    def rank_evaluations(
        self,
        *,
        version_name: str | None = None,
        problem_type_hash: str | None = None,
        benchmark_protocol_hash: str | None = None,
        shape_id: str | None = None,
        min_samples: int = 1,
        limit: int | None = None,
    ) -> list[EvaluationSummary]:
        clauses = ["status = 'ok'"]
        params: list[str] = []
        if version_name is not None:
            clauses.append("version_name = ?")
            params.append(normalize_version_name(version_name))
        if problem_type_hash is not None:
            clauses.append("problem_type_hash = ?")
            params.append(problem_type_hash)
        if benchmark_protocol_hash is not None:
            clauses.append("benchmark_protocol_hash = ?")
            params.append(benchmark_protocol_hash)
        if shape_id is not None:
            clauses.append("shape_id = ?")
            params.append(shape_id)
        where = "WHERE " + " AND ".join(clauses)
        with self.connection() as con:
            rows = con.execute(
                f"""
                SELECT shape_id, candidate_hash, time_us, gflops
                FROM evaluations
                {where}
                """,
                params,
            ).fetchall()
        grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
        for row in rows:
            key = (row["shape_id"], row["candidate_hash"])
            bucket = grouped.setdefault(key, {"time_us": [], "gflops": []})
            if row["time_us"] is not None:
                bucket["time_us"].append(float(row["time_us"]))
            if row["gflops"] is not None:
                bucket["gflops"].append(float(row["gflops"]))

        summaries: list[EvaluationSummary] = []
        for (sid, chash), bucket in grouped.items():
            samples = max(len(bucket["time_us"]), len(bucket["gflops"]))
            if samples < min_samples:
                continue
            summaries.append(
                EvaluationSummary(
                    shape_id=sid,
                    candidate_hash=chash,
                    samples=samples,
                    median_gflops=_median(bucket["gflops"]),
                    best_gflops=max(bucket["gflops"]) if bucket["gflops"] else None,
                    median_time_us=_median(bucket["time_us"]),
                    best_time_us=min(bucket["time_us"]) if bucket["time_us"] else None,
                )
            )

        def sort_key(summary: EvaluationSummary) -> tuple[int, float, float]:
            if summary.median_gflops is not None:
                return (1, summary.median_gflops, -(summary.median_time_us or float("inf")))
            if summary.median_time_us is not None:
                return (0, -summary.median_time_us, 0.0)
            return (-1, 0.0, 0.0)

        summaries.sort(key=sort_key, reverse=True)
        return summaries[:limit] if limit is not None else summaries

    def cache_summary(
        self,
        *,
        version_name: str | None = None,
        problem_type_hash: str | None = None,
        benchmark_protocol_hash: str | None = None,
    ) -> dict[str, int]:
        clauses = []
        params: list[str] = []
        if version_name is not None:
            clauses.append("version_name = ?")
            params.append(normalize_version_name(version_name))
        if problem_type_hash is not None:
            clauses.append("problem_type_hash = ?")
            params.append(problem_type_hash)
        if benchmark_protocol_hash is not None:
            clauses.append("benchmark_protocol_hash = ?")
            params.append(benchmark_protocol_hash)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connection() as con:
            rows = con.execute(
                f"""
                SELECT status, COUNT(*) AS n
                FROM evaluations
                {where}
                GROUP BY status
                ORDER BY status
                """,
                params,
            ).fetchall()
            return {row["status"]: int(row["n"]) for row in rows}

    def counts(self) -> dict[str, int]:
        with self.connection() as con:
            return {
                table: con.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
                for table in ["candidates", "shapes", "runs", "evaluations"]
            }

    def distinct_versions(self) -> list[str]:
        with self.connection() as con:
            rows = con.execute(
                "SELECT DISTINCT version_name FROM runs UNION SELECT DISTINCT version_name FROM evaluations"
            ).fetchall()
            return sorted(row["version_name"] or DEFAULT_VERSION_NAME for row in rows)
