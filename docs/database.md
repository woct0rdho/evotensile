# Database Design

This document describes EvoTensile's SQLite result database and cache semantics.

## Evidence Namespace

Each SQLite file is one evidence namespace for a target hardware/environment/campaign. Use separate DB files for incompatible hardware or software environments.

Timing identity is:

```text
problem_type_hash
benchmark_protocol_hash
shape_id
candidate_hash
```

Correctness identity is independent:

```text
problem_type_hash
validation_protocol_hash
shape_id
candidate_hash
```

This separation lets timing budgets change without repeating correctness while still invalidating correctness evidence when backend, extent, initialization, or validator version changes.

## Tables

`EvoTensileDB.init()` creates the schema and enables WAL mode.

### candidates

Stores canonical candidate JSON, source, parent hashes, proposal metadata, and creation time under a stable candidate hash. The hash depends only on canonical parameters. Island identity, semantic group, donor mode, requested transitions, changed genes, restart index, and proposal-cost metadata do not fragment cache identity.

Candidate registration uses `INSERT OR IGNORE`, so the first registered source/lineage/metadata for a parameter hash remains the DB-level candidate record. Later proposal appearances of the same hash remain auditable in per-round proposal artifacts but do not overwrite candidate-level credit identity.

### shapes

Stores exact `M`, `N`, batch, and `K` dimensions. Shape IDs use `m{M}_n{N}_b{batch}_k{K}`.

### runs

Stores every build, diagnostic, validation, and benchmark invocation. Metadata includes command, phase/mode, paths, duration, timeout state, and pair count. Cost-aware search reconstructs approximate candidate phase costs from these rows, pair files, and manifests rather than materializing a separate cost table.

### evaluations

```text
eval_id INTEGER PRIMARY KEY AUTOINCREMENT
problem_type_hash TEXT NOT NULL
benchmark_protocol_hash TEXT NOT NULL
shape_id TEXT NOT NULL
candidate_hash TEXT NOT NULL
run_id TEXT
status TEXT NOT NULL
time_us REAL
validation TEXT
solution_index INTEGER
created_at REAL NOT NULL
```

Each successful timing sample is one `status='ok'` row with finite positive `time_us`. Negative build, mapping, validation, or runner outcomes also live here, usually without timing.

Benchmark-only timing rows use `validation='PASSED prior_validation'`: the benchmark subprocess performed no validation, but the scheduler admitted the pair only after compatible correctness evidence.

### validations

```text
validation_id INTEGER PRIMARY KEY AUTOINCREMENT
problem_type_hash TEXT NOT NULL
validation_protocol_hash TEXT NOT NULL
shape_id TEXT NOT NULL
candidate_hash TEXT NOT NULL
run_id TEXT
status TEXT NOT NULL
detail TEXT
solution_index INTEGER
created_at REAL NOT NULL
```

Validation-only runs insert one row per pair. `status='passed'` is reusable correctness evidence. Failed validation also creates a `validation_fail` evaluation row so the pair remains reusable negative evidence under the active benchmark campaign.

Validation rows never count as timing samples.

## Candidate And Shape Registration

Before scheduling, candidates and shapes are inserted with `INSERT OR IGNORE`. Imported baselines, random candidates, mutations, DE/GOMEA children, transfer candidates, and repair seeds share the same tables and cache identities.

## Evaluation Statuses

Reusable groups are:

```text
POSITIVE_CACHE_STATUSES = ('ok',)
NEGATIVE_CACHE_STATUSES = ('rejected', 'validation_fail', 'build_failed')
```

Important statuses:
- `ok`: finite positive benchmark sample admitted after compatible validation.
- `rejected`: pair did not survive source-backed rules or final solution mapping.
- `validation_fail`: correctness verification failed.
- `build_failed`: attributable build/codegen failure.
- `build_timeout`: singleton build timeout. Audit-only.
- `runner_timeout`: benchmark timeout. Audit-only.
- `build_failed_unattributed` / `build_timeout_unattributed`: mixed-build failure without candidate attribution. Audit-only.
- `unmapped`: planned pair absent from manifest or mapping. Audit/debug evidence.

Only reusable statuses skip future work.

## Validation Semantics

`validated_cache_entries()` queries the `validations` table for `status='passed'` under the active validation-protocol hash. It no longer infers correctness from timing rows.

The validation-protocol identity includes validator version, backend, validation extent, input initialization settings, and relevant output behavior. A timing protocol change does not invalidate correctness unless one of those validation properties also changes.

There is no trusted no-validation path. Benchmark mode is allowed only for pairs already represented in the prepared validation-passed set or compatible cached validation evidence.

## Cache Lookup

`reusable_cache_entry_counts()` counts timing and reusable negative evaluations under one problem/benchmark protocol. `_missing_candidate_indices_by_shape()` combines:
- Existing timing sample count.
- Reusable negative evidence.
- Compatible correctness evidence from `validations`.
- Shape-dependent source-backed invalidity.

A pair may therefore require more timing while not requiring validation.

## Ranking

`rank_evaluations()` groups `status='ok'` timing rows by `(shape_id, candidate_hash)` and computes sample count, median/best time, and median/best GFLOP/s. Validation-only rows cannot enter ranking because they are stored separately and have no timing.

Ranking feeds CLI reports, proposal elites, transfer seeds, learned linkage, outlier repair, family archives, and final GridBased updates.

## Phase Metadata

`schedule_metadata.json` and `repair_metadata.json` record:
- benchmark and validation protocol hashes.
- prepare- and optional validation-worker counts.
- proposal, surrogate, group/donor-credit, and cost-aware flags.
- planned batches.
- staged probe, initial, stabilization, and adaptive execution phases when applicable.
- build, validation, and benchmark return codes.
- status counts and errors.

The `runs` table provides lower-level command and artifact provenance.

## Concurrency

SQLite uses WAL mode, a 60-second busy timeout, and short independent connections. Parallel prepare workers may insert build/diagnostic/validation evidence concurrently. An optional validation-worker semaphore can reduce validator concurrency without reducing compilation concurrency.

The prepare-worker pool is fully joined before serial benchmark insertion begins. Compile-cache population has a separate per-cache lock. A machine-wide shared/exclusive APU gate prevents timing from overlapping preparation activity across cooperating processes.

## Portability

The schema is additive and has no migration framework. New DBs are recommended for incompatible campaigns. Existing DBs receive the `validations` table through `CREATE TABLE IF NOT EXISTS`. Old timing rows are not automatically converted into validation evidence.
