import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .candidate import Candidate, Shape

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
  shape_id TEXT NOT NULL,
  candidate_hash TEXT NOT NULL,
  run_id TEXT,
  status TEXT NOT NULL,
  time_us REAL,
  gflops REAL,
  validation TEXT,
  solution_index INTEGER,
  raw_csv_row TEXT,
  created_at REAL NOT NULL,
  UNIQUE(shape_id, candidate_hash, run_id)
);

CREATE INDEX IF NOT EXISTS idx_evaluations_shape_candidate
  ON evaluations(shape_id, candidate_hash);

CREATE INDEX IF NOT EXISTS idx_evaluations_shape_time
  ON evaluations(shape_id, time_us);
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
        import json

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
        import json

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
        returncode: int | None = None,
        stdout_path: str | None = None,
        stderr_path: str | None = None,
        metadata_json: str | None = None,
    ) -> None:
        with self.connection() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO runs
                  (run_id, timestamp, yaml_path, output_dir, tensile_bin, status, returncode,
                   stdout_path, stderr_path, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    time.time(),
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
        time_us: float | None = None,
        gflops: float | None = None,
        validation: str | None = None,
        solution_index: int | None = None,
        raw_csv_row: str | None = None,
    ) -> None:
        with self.connection() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO evaluations
                  (shape_id, candidate_hash, run_id, status, time_us, gflops, validation,
                   solution_index, raw_csv_row, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
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

    def counts(self) -> dict[str, int]:
        with self.connection() as con:
            return {
                table: con.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
                for table in ["candidates", "shapes", "runs", "evaluations"]
            }
