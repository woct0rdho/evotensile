import json
import sqlite3
from contextlib import closing

from evotensile.campaign.protocols import CAMPAIGN_HOT_PROTOCOL, CAMPAIGN_SCREENING_PROTOCOL
from evotensile.database import EvoTensileDB
from evotensile.profile import get_profile
from evotensile.search_space import make_candidate, repair_linked_overrides
from scripts.import_legacy_blind_campaign import import_legacy_blind_campaign

LEGACY_SCHEMA = """
CREATE TABLE candidates (
  candidate_hash TEXT PRIMARY KEY,
  candidate_json TEXT NOT NULL,
  source TEXT NOT NULL,
  parent_hashes TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE evaluations (
  eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
  problem_type_hash TEXT NOT NULL,
  benchmark_protocol_hash TEXT NOT NULL,
  shape_id TEXT NOT NULL,
  candidate_hash TEXT NOT NULL,
  run_id TEXT,
  status TEXT NOT NULL,
  time_us REAL,
  validation TEXT,
  solution_index INTEGER,
  created_at REAL NOT NULL
);
CREATE TABLE runs (
  run_id TEXT PRIMARY KEY,
  timestamp REAL NOT NULL,
  yaml_path TEXT,
  output_dir TEXT,
  status TEXT NOT NULL,
  returncode INTEGER,
  metadata_json TEXT
);
CREATE TABLE shapes (
  shape_id TEXT PRIMARY KEY,
  m INTEGER NOT NULL,
  n INTEGER NOT NULL,
  batch INTEGER NOT NULL,
  k INTEGER NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE validations (
  validation_id INTEGER PRIMARY KEY AUTOINCREMENT,
  problem_type_hash TEXT NOT NULL,
  validation_protocol_hash TEXT NOT NULL,
  shape_id TEXT NOT NULL,
  candidate_hash TEXT NOT NULL,
  run_id TEXT,
  status TEXT NOT NULL,
  detail TEXT,
  solution_index INTEGER,
  created_at REAL NOT NULL
);
"""


def test_import_legacy_blind_campaign_preserves_compatible_evidence(tmp_path):
    profile = get_profile("gfx1151-nt-hhs-comfy1135")
    shape = next(shape for shape in profile.shapes() if shape.m == shape.n == shape.k == 8192)
    candidate = make_candidate(repair_linked_overrides({}), source="legacy-random")
    base_path = tmp_path / "base.sqlite"
    source_path = tmp_path / "legacy.sqlite"
    output_path = tmp_path / "output.sqlite"
    hot_dir = tmp_path / "hot"
    rank_dir = hot_dir / f"rank_01_{candidate.hash}"
    rank_dir.mkdir(parents=True)

    base = EvoTensileDB.connect(base_path, environment_compatibility_tag=profile.environment_compatibility_tag)
    base.init()
    with closing(sqlite3.connect(source_path)) as source, source:
        source.executescript(LEGACY_SCHEMA)
        source.execute(
            "INSERT INTO candidates VALUES (?, ?, ?, ?, ?)",
            (candidate.hash, candidate.to_json(), candidate.source, json.dumps(candidate.parent_hashes), 1.0),
        )
        source.execute("INSERT INTO shapes VALUES (?, ?, ?, ?, ?, ?)", (*[shape.id, *shape.exact_list()], 1.0))
        for run_id, mode in (("validate_legacy", "validation"), ("benchmark_legacy", "benchmark")):
            source.execute(
                "INSERT INTO runs VALUES (?, ?, NULL, NULL, 'ok', 0, ?)",
                (run_id, 1.0, json.dumps({"duration_s": 0.5, "mode": mode})),
            )
        source.execute(
            "INSERT INTO validations VALUES (NULL, ?, ?, ?, ?, ?, 'passed', 'ok', 0, ?)",
            (
                profile.problem_type_hash,
                "vproto_54c03ca125088879",
                shape.id,
                candidate.hash,
                "validate_legacy",
                1.0,
            ),
        )
        source.executemany(
            "INSERT INTO evaluations VALUES (NULL, ?, ?, ?, ?, ?, 'ok', ?, 'PASSED', 0, ?)",
            [
                (
                    profile.problem_type_hash,
                    CAMPAIGN_SCREENING_PROTOCOL.protocol_hash(),
                    shape.id,
                    candidate.hash,
                    "benchmark_legacy",
                    time_us,
                    1.0,
                )
                for time_us in (25_000.0, 25_100.0)
            ],
        )

    (hot_dir / "summary.json").write_text(
        json.dumps({"ranked": [{"candidate_hash": candidate.hash, "duration_s": 1.25}]}),
        encoding="utf-8",
    )
    (rank_dir / "results.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "candidate_hash": candidate.hash,
                    "shape_id": shape.id,
                    "status": "ok",
                    "sample_index": index,
                    "time_us": time_us,
                    "solution_index": 0,
                }
            )
            for index, time_us in enumerate((24_000.0, 24_100.0))
        )
        + "\n",
        encoding="utf-8",
    )

    report = import_legacy_blind_campaign(
        base_database=base_path,
        source_database=source_path,
        output_database=output_path,
        profile=profile,
        hot_dir=hot_dir,
    )

    assert report["source_candidates"] == 1
    assert report["imported_validations"] == 1
    assert report["imported_benchmark_events"] == 1
    assert report["imported_benchmark_samples"] == 2
    assert report["imported_hot_events"] == 1
    assert report["imported_hot_samples"] == 2
    with closing(sqlite3.connect(output_path)) as output:
        protocols = dict(
            output.execute(
                """
                SELECT bp.benchmark_protocol_hash, COUNT(bs.time_us)
                FROM benchmark_protocols AS bp
                JOIN benchmark_namespaces AS bn USING (benchmark_protocol_id)
                JOIN benchmark_events AS be USING (benchmark_namespace_id)
                JOIN benchmark_samples AS bs USING (event_id)
                GROUP BY bp.benchmark_protocol_hash
                """
            ).fetchall()
        )
        assert protocols == {
            CAMPAIGN_SCREENING_PROTOCOL.protocol_hash(): 2,
            CAMPAIGN_HOT_PROTOCOL.protocol_hash(): 2,
        }
        assert output.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert output.execute("PRAGMA foreign_key_check").fetchall() == []
