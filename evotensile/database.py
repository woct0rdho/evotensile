import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .cache import POSITIVE_CACHE_STATUSES, REUSABLE_CACHE_STATUSES, CacheKey
from .candidate import Candidate, Shape
from .metrics import gflops_from_us


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


@dataclass(frozen=True)
class EvaluationInsert:
    shape_id: str
    candidate_hash: str
    run_id: str | None
    status: str
    problem_type_hash: str = ""
    benchmark_protocol_hash: str = ""
    time_us: float | None = None
    validation: str | None = None
    solution_index: int | None = None


@dataclass
class _TimingBucket:
    shape: Shape
    time_us: list[float]


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
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  timestamp REAL NOT NULL,
  problem_type_hash TEXT,
  benchmark_protocol_hash TEXT,
  yaml_path TEXT,
  output_dir TEXT,
  tensilelite_bin TEXT,
  status TEXT NOT NULL,
  returncode INTEGER,
  stdout_path TEXT,
  stderr_path TEXT,
  metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS evaluations (
  eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
  problem_type_hash TEXT NOT NULL DEFAULT '',
  benchmark_protocol_hash TEXT NOT NULL DEFAULT '',
  shape_id TEXT NOT NULL,
  candidate_hash TEXT NOT NULL,
  run_id TEXT,
  status TEXT NOT NULL,
  time_us REAL,
  validation TEXT,
  solution_index INTEGER,
  created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evaluations_cache_key
  ON evaluations(problem_type_hash, benchmark_protocol_hash, shape_id, candidate_hash);

CREATE INDEX IF NOT EXISTS idx_evaluations_shape_candidate
  ON evaluations(shape_id, candidate_hash);

CREATE INDEX IF NOT EXISTS idx_evaluations_shape_time
  ON evaluations(problem_type_hash, benchmark_protocol_hash, shape_id, time_us);
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
                  (shape_id, m, n, batch, k, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    shape.id,
                    shape.m,
                    shape.n,
                    shape.batch,
                    shape.k,
                    time.time(),
                ),
            )

    def register_candidates(self, candidates: list[Candidate]) -> None:
        if not candidates:
            return
        now = time.time()
        with self.connection() as con:
            con.executemany(
                """
                INSERT OR IGNORE INTO candidates
                  (candidate_hash, candidate_json, source, parent_hashes, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        candidate.hash,
                        candidate.to_json(),
                        candidate.source,
                        json.dumps(list(candidate.parent_hashes)),
                        now,
                    )
                    for candidate in candidates
                ],
            )

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
                SELECT candidate_hash, candidate_json
                FROM candidates
                WHERE candidate_hash IN ({placeholders})
                """,
                candidate_hashes,
            ).fetchall()
        by_hash: dict[str, Candidate] = {}
        for row in rows:
            payload = json.loads(row["candidate_json"])
            by_hash[row["candidate_hash"]] = Candidate(
                params=payload["params"],
                source=payload.get("source", "db"),
                parent_hashes=tuple(payload.get("parent_hashes", [])),
            )
        return [by_hash[h] for h in candidate_hashes if h in by_hash]

    def insert_run(
        self,
        run_id: str,
        *,
        yaml_path: str | None,
        output_dir: str | None,
        tensilelite_bin: str | None,
        status: str,
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
                  (run_id, timestamp, problem_type_hash, benchmark_protocol_hash,
                   yaml_path, output_dir, tensilelite_bin, status, returncode, stdout_path, stderr_path,
                   metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    time.time(),
                    problem_type_hash,
                    benchmark_protocol_hash,
                    yaml_path,
                    output_dir,
                    tensilelite_bin,
                    status,
                    returncode,
                    stdout_path,
                    stderr_path,
                    metadata_json,
                ),
            )

    def update_run_status(self, run_id: str, *, status: str) -> None:
        with self.connection() as con:
            con.execute(
                """
                UPDATE runs
                SET status = ?
                WHERE run_id = ?
                """,
                (status, run_id),
            )

    def insert_evaluation(
        self,
        *,
        shape_id: str,
        candidate_hash: str,
        run_id: str | None,
        status: str,
        problem_type_hash: str = "",
        benchmark_protocol_hash: str = "",
        time_us: float | None = None,
        validation: str | None = None,
        solution_index: int | None = None,
    ) -> None:
        with self.connection() as con:
            con.execute(
                """
                INSERT INTO evaluations
                  (problem_type_hash, benchmark_protocol_hash, shape_id, candidate_hash,
                   run_id, status, time_us, validation, solution_index, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    problem_type_hash,
                    benchmark_protocol_hash,
                    shape_id,
                    candidate_hash,
                    run_id,
                    status,
                    time_us,
                    validation,
                    solution_index,
                    time.time(),
                ),
            )

    def insert_evaluations(self, evaluations: list[EvaluationInsert]) -> None:
        if not evaluations:
            return
        now = time.time()
        with self.connection() as con:
            con.executemany(
                """
                INSERT INTO evaluations
                  (problem_type_hash, benchmark_protocol_hash, shape_id, candidate_hash,
                   run_id, status, time_us, validation, solution_index, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        evaluation.problem_type_hash,
                        evaluation.benchmark_protocol_hash,
                        evaluation.shape_id,
                        evaluation.candidate_hash,
                        evaluation.run_id,
                        evaluation.status,
                        evaluation.time_us,
                        evaluation.validation,
                        evaluation.solution_index,
                        now,
                    )
                    for evaluation in evaluations
                ],
            )

    def cached_evaluation_count(
        self,
        *,
        problem_type_hash: str,
        benchmark_protocol_hash: str,
        shape_id: str,
        candidate_hash: str,
        statuses: tuple[str, ...] = POSITIVE_CACHE_STATUSES,
    ) -> int:
        placeholders = ",".join("?" for _ in statuses)
        with self.connection() as con:
            row = con.execute(
                f"""
                SELECT COUNT(*) AS n
                FROM evaluations
                WHERE problem_type_hash = ?
                  AND benchmark_protocol_hash = ?
                  AND shape_id = ?
                  AND candidate_hash = ?
                  AND status IN ({placeholders})
                """,
                (
                    problem_type_hash,
                    benchmark_protocol_hash,
                    shape_id,
                    candidate_hash,
                    *statuses,
                ),
            ).fetchone()
            return int(row["n"])

    def has_cached_evaluation(
        self,
        key: CacheKey,
        *,
        min_samples: int = 1,
        statuses: tuple[str, ...] = POSITIVE_CACHE_STATUSES,
    ) -> bool:
        return (
            self.cached_evaluation_count(
                problem_type_hash=key.problem_type_hash,
                benchmark_protocol_hash=key.benchmark_protocol_hash,
                shape_id=key.shape_id,
                candidate_hash=key.candidate_hash,
                statuses=statuses,
            )
            >= min_samples
        )

    def has_reusable_cache_entry(self, key: CacheKey, *, min_ok_samples: int = 1) -> bool:
        return (key.shape_id, key.candidate_hash) in self.reusable_cache_entries(
            problem_type_hash=key.problem_type_hash,
            benchmark_protocol_hash=key.benchmark_protocol_hash,
            shape_ids=[key.shape_id],
            candidate_hashes=[key.candidate_hash],
            min_ok_samples=min_ok_samples,
        )

    def reusable_cache_entry_counts(
        self,
        *,
        problem_type_hash: str,
        benchmark_protocol_hash: str,
        shape_ids: list[str],
        candidate_hashes: list[str],
    ) -> dict[tuple[str, str], dict[str, int]]:
        if not shape_ids or not candidate_hashes:
            return {}
        shape_placeholders = ",".join("?" for _ in shape_ids)
        candidate_placeholders = ",".join("?" for _ in candidate_hashes)
        status_placeholders = ",".join("?" for _ in REUSABLE_CACHE_STATUSES)
        with self.connection() as con:
            rows = con.execute(
                f"""
                SELECT shape_id, candidate_hash, status, COUNT(*) AS n
                FROM evaluations
                WHERE problem_type_hash = ?
                  AND benchmark_protocol_hash = ?
                  AND shape_id IN ({shape_placeholders})
                  AND candidate_hash IN ({candidate_placeholders})
                  AND status IN ({status_placeholders})
                GROUP BY shape_id, candidate_hash, status
                """,
                (
                    problem_type_hash,
                    benchmark_protocol_hash,
                    *shape_ids,
                    *candidate_hashes,
                    *REUSABLE_CACHE_STATUSES,
                ),
            ).fetchall()

        counts: dict[tuple[str, str], dict[str, int]] = {}
        for row in rows:
            key = (row["shape_id"], row["candidate_hash"])
            counts.setdefault(key, {})[row["status"]] = int(row["n"])
        return counts

    def reusable_cache_entries(
        self,
        *,
        problem_type_hash: str,
        benchmark_protocol_hash: str,
        shape_ids: list[str],
        candidate_hashes: list[str],
        min_ok_samples: int = 1,
    ) -> set[tuple[str, str]]:
        counts = self.reusable_cache_entry_counts(
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            shape_ids=shape_ids,
            candidate_hashes=candidate_hashes,
        )
        reusable: set[tuple[str, str]] = set()
        for key, status_counts in counts.items():
            ok_count = sum(status_counts.get(status, 0) for status in POSITIVE_CACHE_STATUSES)
            negative_count = sum(
                count for status, count in status_counts.items() if status not in POSITIVE_CACHE_STATUSES
            )
            if ok_count >= min_ok_samples or negative_count > 0:
                reusable.add(key)
        return reusable

    def rank_evaluations(
        self,
        *,
        problem_type_hash: str | None = None,
        benchmark_protocol_hash: str | None = None,
        shape_id: str | None = None,
        min_samples: int = 1,
        limit: int | None = None,
    ) -> list[EvaluationSummary]:
        clauses = ["e.status = 'ok'"]
        params: list[str] = []
        if problem_type_hash is not None:
            clauses.append("e.problem_type_hash = ?")
            params.append(problem_type_hash)
        if benchmark_protocol_hash is not None:
            clauses.append("e.benchmark_protocol_hash = ?")
            params.append(benchmark_protocol_hash)
        if shape_id is not None:
            clauses.append("e.shape_id = ?")
            params.append(shape_id)
        where = "WHERE " + " AND ".join(clauses)
        with self.connection() as con:
            rows = con.execute(
                f"""
                SELECT e.shape_id, e.candidate_hash, e.time_us, s.m, s.n, s.batch, s.k
                FROM evaluations AS e
                JOIN shapes AS s ON s.shape_id = e.shape_id
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

        summaries: list[EvaluationSummary] = []
        for (sid, chash), bucket in grouped.items():
            samples = len(bucket.time_us)
            if samples < min_samples:
                continue
            gflops_values = [gflops_from_us(bucket.shape, time_us) for time_us in bucket.time_us]
            summaries.append(
                EvaluationSummary(
                    shape_id=sid,
                    candidate_hash=chash,
                    samples=samples,
                    median_gflops=_median(gflops_values),
                    best_gflops=max(gflops_values) if gflops_values else None,
                    median_time_us=_median(bucket.time_us),
                    best_time_us=min(bucket.time_us) if bucket.time_us else None,
                )
            )

        def sort_key(summary: EvaluationSummary) -> tuple[int, float, float]:
            if summary.median_time_us is not None:
                return (1, -summary.median_time_us, summary.median_gflops or 0.0)
            return (0, 0.0, 0.0)

        summaries.sort(key=sort_key, reverse=True)
        return summaries[:limit] if limit is not None else summaries

    def cache_summary(
        self,
        *,
        problem_type_hash: str | None = None,
        benchmark_protocol_hash: str | None = None,
    ) -> dict[str, int]:
        clauses = []
        params: list[str] = []
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
