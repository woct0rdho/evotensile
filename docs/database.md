# Database Design

This document describes EvoTensile's SQLite result database and cache semantics.

## Evidence Namespace

Each SQLite file is one evidence namespace for a target hardware/environment/campaign. Use separate DB files for incompatible hardware or software environments.

Every database stores one opaque `environment_compatibility_tag`. The selected target profile supplies the expected tag whenever the database is opened. A missing or different tag rejects the database before evidence is read or mutated. The tag is a manual operator assertion that external factors affecting code generation, validation, loading, or timing remain compatible. It is not a schema version, semantic version, or automatic environment fingerprint. The current gfx1151 profile tag is `gfx1151-nt-hhs-v1`.

Changes that may invalidate generated or executed code require a new tag and fresh timing/validation evidence rather than relabeling old evidence.

Timing identity is:

```text
problem_type_hash
benchmark_protocol_hash
shape_id
candidate_hash
```

The database interns problem types and benchmark protocols into a benchmark namespace, then uses compact integer candidate and shape keys in evidence tables. Stable hashes and shape IDs remain the external API identities in their catalog owner rows.

Correctness identity is independent:

```text
problem_type_hash
validation_protocol_hash
shape_id
candidate_hash
```

Problem types and validation protocols similarly form validation namespaces. Catalog definitions are nullable when only an authoritative historical hash is known. Unavailable definitions are never invented.

This separation lets timing budgets change without repeating correctness while still invalidating correctness evidence when the declared validation backend, extent, initialization, or manually maintained validation-protocol schema version changes. The hash does not fingerprint the structured-runner binary, hipBLASLt or ROCm version, GPU identity, or generated-library contents. Incompatible hardware, software, or generated-code environments require a separate DB evidence namespace and fresh validation. An equivalent artifact rebuild does not. Frozen campaign configuration records binary and implementation fingerprints for campaign resume identity, while artifact bundles independently verify the exact library contents used for later execution.

## Tables

`EvoTensileDB.init()` creates the schema and enables WAL mode.

### database_metadata

Stores the single database-level `environment_compatibility_tag`. EvoTensile does not infer compatibility from software versions or repository state. The operator changes this opaque tag when the environment is no longer evidence-compatible.

### Identity Catalogs

`problem_types`, `benchmark_protocols`, and `validation_protocols` own unique external hashes and optional canonical definitions. `benchmark_namespaces` and `validation_namespaces` own the corresponding problem/protocol pair identities used by evidence rows.

### candidates

Stores one integer key, unique parameter-only candidate hash, compact canonical `params_json`, and creation time. Source, lineage, island, restart, operator metadata, and proposal cost are occurrence data and are not duplicated on candidate identity. Restoring a candidate from this table restores parameters only.

### proposal_events and proposal_candidates

`proposal_events` owns one proposal call's benchmark namespace, scope, canonical arguments, island/restart identity, duration, and timestamp. `proposal_candidates` owns each candidate occurrence's source, parent candidate keys, operator metadata, generated/preserved state, and selected state.

Rows are append-only. The latest compatible selected occurrence becomes one operator-credit trial only when compatible child timing was queried after it. Repeated occurrences cannot claim the same current measurement. Unselected oversized-pool proposals and cache-only reproposals remain auditable but receive no reward. Proposal cost is the event duration divided across its distinct generated candidates. Preserved candidates receive no new proposal cost.

### baseline_discoveries and baseline_selections

Stores installed hipBLASLt discovery context separately from adaptive proposals and evidence. One discovery owns runtime/logic query context, duration, and timestamp. Its selections map exact shapes to canonical candidates plus external hipBLASLt and logic solution identifiers. Query GFLOP/s and time are audit-only.

A selection with zero EvoTensile runs has no benchmark status, validation status, samples, cache effect, or operator reward. `scripts/discover_hipblaslt_baselines.py` groups stored exact pairs by candidate and hands them to the normal scheduler, which creates ordinary native evidence.

### shapes

Stores an integer internal key and exact `M`, `N`, batch, and `K` dimensions under a unique external shape ID. Shape IDs use `m{M}_n{N}_b{batch}_k{K}`.

### evidence_sources

Owns the provenance of every benchmark and validation observation. Current kinds are native subprocess runs, retained historical migration, deterministic replay, and static source-backed rule decisions. Provenance is audit metadata only. All observations use the same cache and ranking semantics. hipBLASLt discovery is planning data and creates no evidence source.

### native_runs

Stores one typed detail row for each native subprocess source: phase, outcome, measured duration, and return code. Commands, YAML/output/log paths, repeated protocol hashes, duplicate timeout flags, and metadata JSON are not persisted. Generated artifact locations belong to artifact records instead.

### run_candidate_costs

Indexes the shared duration attributed to each distinct candidate in a native build, diagnostic, validation, probe, or screening run. Execution boundaries already know the exact candidate set, so `insert_run()` divides duration once and records the source, candidate, phase, and duration without rereading manifests or runner pair files. Proposal cost is derived separately from proposal-event duration and distinct generated candidates.

### benchmark_events

Owns one benchmark invocation outcome per provenance source, namespace, shape, and candidate. A successful event has status, solution mapping, creation time, and one or more ordered child samples. A negative or audit event has no timing samples. Reusable-negative idempotence applies to events.

Successful benchmark events store the exact validation namespace that admitted the pair. Insertion requires the latest compatible validation event to be `passed`. Cache, ranking, adaptive retiming, and proposal evidence recheck latest-state ownership and never parse correctness from benchmark text. A later compatible validation failure suppresses the pair until a later pass supersedes it.

### benchmark_samples

Stores only `(event_id, sample_index, time_us)`. Sample indices are unique and ordered within an event. Every timing must be finite and positive. Repeated equal values are preserved as distinct samples. Insertion of an event and all of its samples is transactional.

### validations

Owns one correctness outcome through compact foreign keys to a validation namespace, shape, candidate, and evidence source, plus status, optional detail/solution index, and creation time. Validation-only runs insert one row per pair. Validation rows never count as timing samples.

### artifact_bundles, artifact_solution_yamls, and artifact_mappings

`artifact_bundles` owns one generated library's build run, build/output/library roots, optional manifest, SHA-256-derived library content identity, and creation time. `artifact_solution_yamls` owns its normalized generated solution-YAML paths. `artifact_mappings` maps problem type, shape, candidate, and runnable problem/requested/library/manifest indices to that shared bundle.

Registration occurs immediately after authoritative final-YAML mapping and before validation/timing. Loading requires every recorded path to exist and the current library content identity to match. Stale, moved, incomplete, or modified bundles are unavailable rather than inferred from surrounding directories. Multiple pair mappings reuse one verified bundle without duplicating paths or content identity.

## Candidate And Shape Registration

Before scheduling, candidates and shapes are inserted with `INSERT OR IGNORE`. Discovered baselines, random candidates, mutations, DE/GOMEA children, transfer candidates, and repair seeds share the same parameter-only candidate identity and exact shape identity.

## Benchmark Statuses

Reusable benchmark-cache groups are:

```text
POSITIVE_CACHE_STATUSES = ('ok',)
NEGATIVE_CACHE_STATUSES = ('rejected', 'build_failed')
```

Important statuses:
- `ok`: benchmark event with one or more finite positive samples admitted after compatible validation.
- `rejected`: pair did not survive source-backed rules or final solution mapping.
- `build_failed`: attributable build/codegen failure.
- `build_timeout`: singleton build timeout. Audit-only.
- `runner_timeout`: benchmark timeout. Audit-only.
- `build_failed_unattributed` / `build_timeout_unattributed`: mixed-build failure without candidate attribution. Audit-only.
- `unmapped`: planned pair absent from manifest or mapping. Audit/debug evidence.

Identical reusable negative events are idempotent by problem/protocol/pair/source/status/solution identity. Other audit events remain append-only.

## Validation Semantics

`validation_cache_states()` selects the latest row by `(created_at, validation_id)` for each pair under the active validation-protocol hash. `validated_cache_entries()` returns only pairs whose latest compatible state is `passed`.

A latest `failed` state suppresses the pair only for that exact validation identity. Changing a hashed backend, extent, initialization, or validation-protocol schema field produces a different hash and requires fresh validation. Software, GPU, and generated-library content changes do not alter this hash automatically. Use a separate DB namespace and fresh validation when those changes can alter generated or executed code. A later pass supersedes an earlier failure, and a later failure supersedes an earlier pass.

There is no trusted no-validation path. Benchmark mode is allowed only for pairs already represented in the prepared validation-passed set or compatible cached validation evidence.

## Cache Lookup

`benchmark_evidence_states()` resolves one benchmark state per pair under the active problem and benchmark protocol:
- Valid finite `ok` samples have durable precedence over every reusable negative, regardless of insertion order.
- Without positive timing, the latest `rejected` or `build_failed` event by `(created_at, event_id)` is the reusable negative state.
- Audit-only statuses do not control planning.

`_missing_candidate_indices_by_shape()` combines the resolved positive sample count or negative state with latest compatible correctness state and shape-dependent source-backed invalidity. A proven timed pair can therefore receive additional samples even when older or newer build/mapping failures remain in raw audit history.

## Ranking

`rank_benchmarks()` groups child samples from `status='ok'` benchmark events by `(shape_id, candidate_hash)` and computes sample count, median/best time, and median/best GFLOP/s. Validation-only rows cannot enter ranking because they are stored separately and have no timing.

Ranking feeds CLI reports, transfer seeds, learned linkage, outlier repair, family archives, and final GridBased updates. One proposal call builds one immutable `ProposalEvidenceSnapshot` containing compatible ranking summaries, candidates, selected occurrences, latest positive timestamps, indexed costs, and status aggregates. Elite, transfer, family, linkage, surrogate, and operator-credit views consume that snapshot rather than independently rescanning SQLite or artifacts. One-shape proposal elites consume the shape-local ranking directly. Multi-shape proposal parents derive specialist and coverage-aware generalist lanes from shape-local incumbent-normalized regret. They do not treat the globally sorted pair rows as a candidate ranking.

## Indexes

Current evidence indexes follow measured integer-key workloads:
- `idx_benchmark_events_pair` serves exact and multi-pair namespace cache lookup.
- partial `idx_benchmark_events_positive` covers successful ranking and latest-positive scans.
- partial `idx_benchmark_events_negative` limits reusable-negative resolution to the two controlling statuses.
- `idx_validations_latest` resolves exact latest validation state in timestamp/ID order.
- `idx_run_candidate_costs_candidate` serves candidate/phase cost aggregation.
- `idx_artifact_mappings_pair` serves latest problem/shape/candidate bundle lookup.

Catalog uniqueness and parent/child primary keys provide the remaining required indexes. Use `EXPLAIN QUERY PLAN`, representative timings, `dbstat`, and `PRAGMA freelist_count` when changing these indexes.

## Phase Metadata

`schedule_metadata.json` and `repair_metadata.json` record:
- benchmark and validation protocol hashes.
- prepare- and optional validation-worker counts.
- proposal, surrogate, group/donor-credit, and cost-aware flags.
- planned batches.
- staged probe, initial, stabilization, and adaptive execution phases when applicable.
- build, validation, and benchmark return codes.
- status counts and errors.

`evidence_sources` and `native_runs` provide typed subprocess provenance. Artifact bundles own reusable generated paths.

## Concurrency

SQLite uses WAL mode, a 60-second busy timeout, and short independent connections. Parallel prepare workers may insert build/diagnostic/validation evidence concurrently. An optional validation-worker semaphore can reduce validator concurrency without reducing compilation concurrency.

The prepare-worker pool is fully joined before serial benchmark insertion begins. Compile-cache population has a separate per-cache lock. A machine-wide shared/exclusive APU gate prevents timing from overlapping preparation activity across cooperating processes.

## Authoritative Audit

The retained gfx1151 corpus is `out/grid100_full_20260618_repaired.sqlite`.

Run structural checks directly:

```bash
sqlite3 out/grid100_full_20260618_repaired.sqlite 'PRAGMA integrity_check; PRAGMA foreign_key_check; PRAGMA freelist_count;'
sqlite3 out/grid100_full_20260618_repaired.sqlite \
  "SELECT name, SUM(pgsize) FROM dbstat WHERE name NOT LIKE 'sqlite_%' GROUP BY name ORDER BY SUM(pgsize) DESC;"
```

Run operational checks through current APIs:

```bash
python3 -m evotensile.cli summarize-cache --db out/grid100_full_20260618_repaired.sqlite --profile gfx1151-nt-hhs
python3 -m evotensile.cli rank-benchmarks --db out/grid100_full_20260618_repaired.sqlite --profile gfx1151-nt-hhs
```

The consolidated historical corpus intentionally has no current-protocol validation passes, so current ranking is empty until the normal scheduler validates and benchmarks pairs under current identities.

## Portability

Use a new database and compatibility tag for incompatible hardware, software, benchmark, or validation environments. Runtime code reads and writes only this schema. It contains no historical-schema readers, migration dispatch, or compatibility views.
