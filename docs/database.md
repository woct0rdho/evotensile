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

This separation lets timing budgets change without repeating correctness while still invalidating correctness evidence when the declared validation backend, extent, initialization, or manually maintained validation-protocol schema version changes. The hash does not fingerprint the structured-runner binary, hipBLASLt or ROCm version, GPU identity, or generated-library contents. Incompatible hardware, software, or generated-code environments require a separate DB evidence namespace and fresh validation. An equivalent artifact rebuild does not. Frozen campaign configuration records binary and implementation fingerprints for campaign resume identity, while `candidate_artifacts` independently verifies the exact library contents used for later execution.

## Tables

`EvoTensileDB.init()` creates the schema and enables WAL mode.

### candidates

Stores canonical candidate JSON, source, parent hashes, proposal metadata, and creation time under a stable candidate hash. The hash depends only on canonical parameters. Island identity, semantic group, donor mode, requested transitions, changed genes, restart index, and proposal-cost metadata do not fragment cache identity.

Candidate registration uses `INSERT OR IGNORE`, so the first registered source/lineage/metadata for a parameter hash remains the DB-level candidate record. Adaptive credit does not use that first-registration lineage. Later proposal appearances are stored separately in `proposal_occurrences`.

### proposal_occurrences

Stores every proposal appearance independently of candidate identity:
- active problem and benchmark protocol hashes.
- candidate hash, occurrence source, parent hashes, and proposal metadata.
- explicit global/shape/cluster/shape-set scope and shape IDs.
- whether the occurrence survived shortlisting.
- occurrence timestamp.

Rows are append-only. The latest compatible selected occurrence becomes one operator-credit trial only when compatible child timing was queried after it. Repeated occurrences cannot claim the same current measurement. Unselected oversized-pool proposals and cache-only reproposals remain auditable but receive no reward. This lets a parameter hash first registered by one source later receive attribution under a different operator without overwriting the candidate row.

### shapes

Stores exact `M`, `N`, batch, and `K` dimensions. Shape IDs use `m{M}_n{N}_b{batch}_k{K}`.

### runs

Stores every build, diagnostic, validation, and benchmark invocation. Metadata includes command, phase/mode, paths, duration, timeout state, and pair count. Runtime artifact consumers do not infer libraries or final YAML from run metadata.

### run_candidate_costs

Indexes the shared duration attributed to each distinct candidate in a build, diagnostic, validation, probe, or screening run. Execution boundaries already know the exact candidate set, so `insert_run()` divides duration once and records `(run_id, candidate_hash, phase, duration_s)` without rereading manifests or runner pair files. Proposal cost remains candidate-origin metadata because it is measured before execution.

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

Each successful timing sample is one `status='ok'` row with finite positive `time_us`. Negative build, mapping, or runner outcomes also live here, usually without timing. Correctness outcomes live only in `validations`.

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

Validation-only runs insert one row per pair. Validation rows never count as timing samples.

### candidate_artifacts

Stores one exact mapped artifact record per `(problem type, shape, candidate, library solution, library path, content identity)`. Each record contains:
- runnable problem/requested/library/manifest solution indices.
- originating build run and build output directory.
- generated library directory.
- all generated solution-YAML paths used for mapping.
- optional manifest path.
- SHA-256-derived identity over the complete generated library contents.

Registration occurs immediately after authoritative final-YAML mapping and before validation/timing. Loading requires every recorded path to exist and the current library content identity to match. Stale, moved, incomplete, or modified artifacts are unavailable rather than inferred from surrounding directories.

## Candidate And Shape Registration

Before scheduling, candidates and shapes are inserted with `INSERT OR IGNORE`. Imported baselines, random candidates, mutations, DE/GOMEA children, transfer candidates, and repair seeds share the same tables and cache identities.

## Evaluation Statuses

Reusable benchmark-cache groups are:

```text
POSITIVE_CACHE_STATUSES = ('ok',)
NEGATIVE_CACHE_STATUSES = ('rejected', 'build_failed')
```

Important statuses:
- `ok`: finite positive benchmark sample admitted after compatible validation.
- `rejected`: pair did not survive source-backed rules or final solution mapping.
- `build_failed`: attributable build/codegen failure.
- `build_timeout`: singleton build timeout. Audit-only.
- `runner_timeout`: benchmark timeout. Audit-only.
- `build_failed_unattributed` / `build_timeout_unattributed`: mixed-build failure without candidate attribution. Audit-only.
- `unmapped`: planned pair absent from manifest or mapping. Audit/debug evidence.

Identical reusable negatives are idempotent by problem/protocol/pair/run/status/solution identity. Other audit rows remain append-only.

## Validation Semantics

`validation_cache_states()` selects the latest row by `(created_at, validation_id)` for each pair under the active validation-protocol hash. `validated_cache_entries()` returns only pairs whose latest compatible state is `passed`.

A latest `failed` state suppresses the pair only for that exact validation identity. Changing a hashed backend, extent, initialization, or validation-protocol schema field produces a different hash and requires fresh validation. Software, GPU, and generated-library content changes do not alter this hash automatically. Use a separate DB namespace and fresh validation when those changes can alter generated or executed code. A later pass supersedes an earlier failure, and a later failure supersedes an earlier pass.

There is no trusted no-validation path. Benchmark mode is allowed only for pairs already represented in the prepared validation-passed set or compatible cached validation evidence.

## Cache Lookup

`benchmark_evidence_states()` resolves one benchmark state per pair under the active problem and benchmark protocol:
- Valid finite `ok` samples have durable precedence over every reusable negative, regardless of insertion order.
- Without positive timing, the latest `rejected` or `build_failed` row by `(created_at, eval_id)` is the reusable negative state.
- Audit-only statuses do not control planning.

`_missing_candidate_indices_by_shape()` combines the resolved positive sample count or negative state with latest compatible correctness state and shape-dependent source-backed invalidity. A proven timed pair can therefore receive additional samples even when older or newer build/mapping failures remain in raw audit history.

## Ranking

`rank_evaluations()` groups `status='ok'` timing rows by `(shape_id, candidate_hash)` and computes sample count, median/best time, and median/best GFLOP/s. Validation-only rows cannot enter ranking because they are stored separately and have no timing.

Ranking feeds CLI reports, transfer seeds, learned linkage, outlier repair, family archives, and final GridBased updates. One proposal call builds one immutable `ProposalEvidenceSnapshot` containing compatible ranking summaries, candidates, selected occurrences, latest positive timestamps, indexed costs, and status aggregates. Elite, transfer, family, linkage, surrogate, and operator-credit views consume that snapshot rather than independently rescanning SQLite or artifacts. One-shape proposal elites consume the shape-local ranking directly. Multi-shape proposal parents derive specialist and coverage-aware generalist lanes from shape-local incumbent-normalized regret. They do not treat the globally sorted pair rows as a candidate ranking.

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

Use a new database for incompatible hardware, software, benchmark, or validation identities. The runtime schema contains only current operational tables. One-time historical migration code is not retained.
