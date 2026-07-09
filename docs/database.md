# Database Design

This document describes EvoTensile's SQLite result database and cache semantics.

## Evidence Namespace

Each SQLite file is one evidence namespace for a target hardware/environment/campaign. EvoTensile does not automatically invalidate cache entries by ROCm commit hash or source checkout. Store environment metadata in run records for audit, and use separate DB files when comparing incompatible campaigns.

Within one DB, reusable evaluation identity is:

```text
problem_type_hash
benchmark_protocol_hash
shape_id
candidate_hash
```

The scheduler and ranking tools always filter by the active profile's problem type hash and benchmark protocol hash when those values are supplied.

## Tables

`EvoTensileDB.init()` creates the schema and enables WAL mode.

### candidates

```text
candidate_hash TEXT PRIMARY KEY
candidate_json TEXT NOT NULL
source TEXT NOT NULL
parent_hashes TEXT NOT NULL
created_at REAL NOT NULL
```

`candidate_json` stores the canonical parameter dictionary, source label, and parent hashes. Candidate hashes are derived from canonical params, so the hash is stable across runs.

### shapes

```text
shape_id TEXT PRIMARY KEY
m INTEGER NOT NULL
n INTEGER NOT NULL
batch INTEGER NOT NULL
k INTEGER NOT NULL
created_at REAL NOT NULL
```

`shape_id` has the form `m{M}_n{N}_b{batch}_k{K}`. Shape FLOP calculations use `2 * M * N * K * batch`.

### runs

```text
run_id TEXT PRIMARY KEY
timestamp REAL NOT NULL
yaml_path TEXT
output_dir TEXT
status TEXT NOT NULL
returncode INTEGER
metadata_json TEXT
```

Run rows record build, structured-runner, and diagnostic invocations. Metadata stores commands, output paths, duration, timeout status, cache use, validation backend, and related counts.

### evaluations

```text
eval_id INTEGER PRIMARY KEY AUTOINCREMENT
problem_type_hash TEXT NOT NULL DEFAULT ''
benchmark_protocol_hash TEXT NOT NULL DEFAULT ''
shape_id TEXT NOT NULL
candidate_hash TEXT NOT NULL
run_id TEXT
status TEXT NOT NULL
time_us REAL
validation TEXT
solution_index INTEGER
created_at REAL NOT NULL
```

Each timing sample is one evaluation row. Negative rows such as rejected candidates or build failures also live here, usually without `time_us`.

Indexes support cache lookup and ranking:

```text
(problem_type_hash, benchmark_protocol_hash, shape_id, candidate_hash)
(shape_id, candidate_hash)
(problem_type_hash, benchmark_protocol_hash, shape_id, time_us)
```

## Candidate And Shape Registration

Before scheduling work, the scheduler inserts all proposed candidates and target shapes with `INSERT OR IGNORE`. This ensures later evaluation rows can always be joined to canonical candidate parameters and shape dimensions.

Imported hipBLASLt baselines, random candidates, mutations, DE children, GOMEA children, transfer candidates, and repair seeds all use the same `candidates` table. Their `source` and `parent_hashes` fields preserve provenance but do not affect cache identity.

## Evaluation Statuses

The cache module defines reusable status groups:

```text
POSITIVE_CACHE_STATUSES = ('ok',)
NEGATIVE_CACHE_STATUSES = ('rejected', 'validation_fail', 'build_failed')
REUSABLE_CACHE_STATUSES = positive + negative
```

Important statuses include:
- `ok`: finite positive timing sample with passing validation, or trusted timing-only top-up backed by prior validation.
- `rejected`: candidate/shape pair did not survive known rules or final TensileLite solution mapping.
- `validation_fail`: structured runner reported failed validation.
- `validation_unknown`: positive-looking row without accepted validation token. Not reusable positive evidence.
- `invalid`: malformed, missing, or non-positive timing row.
- `build_failed`: attributable build/codegen failure, reusable as negative evidence.
- `build_timeout`: singleton build timeout, recorded for audit.
- `runner_timeout`: structured runner timeout for a batch.
- `build_failed_unattributed` / `build_timeout_unattributed`: multi-candidate failure without trustworthy attribution. Audit only.
- `unmapped`: planned pair not present in manifest or mapping. Audit/debug signal.

Only statuses in `REUSABLE_CACHE_STATUSES` are used to skip planned work.

## Validation Semantics

Passing validation tokens are normalized by the first word of the validation string:

```text
PASSED
OK
VALID
```

`validated_cache_entries()` returns pairs with `status='ok'` and a passing validation token. Timing-only top-ups with `NO_CHECK` are allowed only when this prior validation evidence exists for the same pair.

Unknown validation is never ranked as positive. Failed validation is reusable negative evidence for the same cache key.

## Cache Lookup

`reusable_cache_entry_counts()` counts reusable statuses for candidate/shape sets under one problem/protocol hash pair. `has_reusable_cache_entry()` returns true when either:
- positive `ok` rows meet the requested `min_ok_samples`.
- any reusable negative status exists.

`_missing_candidate_indices_by_shape()` uses these counts to decide how many more samples are needed for each pair. It also checks shape-dependent invalid rules and skips pairs that are invalid for the target shape.

## Ranking

`rank_evaluations()` groups `status='ok'` rows by `(shape_id, candidate_hash)` and computes:
- sample count.
- median and best time in microseconds.
- median and best GFLOP/s from the joined shape dimensions.

Rows below `min_samples` are ignored. Results sort by best median time, with median GFLOP/s as a tie helper.

Ranking is used for:
- CLI `rank-evals` output.
- elite selection for local, DE, and GOMEA proposal modes.
- nearest-shape transfer seeds.
- learned-linkage evidence loading.
- outlier repair detection.
- final GridBased update selection.

## Batch And Run Metadata

`schedule-batches` writes `schedule_metadata.json` in the output directory. `repair-outliers` writes `repair_metadata.json`. These JSON files summarize proposal settings, protocol settings, learned-linkage status, adaptive policy, planned batches, executed batches, status counts, paths, and errors.

The SQLite `runs` table stores lower-level invocation metadata, while the output metadata files summarize one CLI-level operation.

## Concurrency

SQLite connections use:

```text
PRAGMA journal_mode=WAL
PRAGMA busy_timeout=60000
sqlite timeout=60s
```

This supports concurrent batch workers inserting results while avoiding short lock failures. The compile cache has its own filesystem lock and is separate from SQLite locking.

## Portability And Migration

The schema is intentionally small and append-only. There is no migration framework yet. If an incompatible schema change is needed, create a new DB or add backwards-compatible columns/tables with `CREATE TABLE IF NOT EXISTS` / `ALTER TABLE` guards.

Analytics that do not belong in the hot scheduler path can be performed by ad-hoc SQLite queries or by exporting to DuckDB later.
