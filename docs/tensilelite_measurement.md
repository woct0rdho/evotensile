# TensileLite Measurement Design

This document describes how EvoTensile communicates with TensileLite when measuring candidate speed.

## Boundary

EvoTensile orchestrates TensileLite externally:
- It writes TensileLite YAML configs.
- It invokes TensileLite build/codegen in build-only mode.
- It maps accepted final-YAML solutions back to EvoTensile candidate hashes.
- It runs an external structured HIP/TensileLite backend for exact `(shape, candidate)` timing.
- It ingests structured JSONL rows into SQLite.

The old TensileLite `LibraryClient` CSV/log ingestion path has been removed from the production scheduler.

## YAML Contract

`write_tensilelite_yaml()` writes a config with:
- Profile global parameters from `TargetProfile.global_parameters(protocol)`.
- One `BenchmarkProblems` entry for the profile problem type.
- Candidate solution dictionaries emitted as a single `Groups` list under `ForkParameters`.
- Exact shape problem sizes under `BenchmarkFinalParameters`.
- Target `LibraryLogic` for GridBased gfx1151 output.

Candidates are complete solution dictionaries. EvoTensile does not emit independent multi-valued fork parameters unless a caller intentionally constructs such a candidate dictionary.

The active `gfx1151-nt-hhs` problem type includes FP16 NT HHS, bias from `D`, `scaleAlpha_vector`, activation support, and `UseE=False`.

## Manifest Contract

For every generated YAML, EvoTensile writes `config.manifest.csv` with:

```text
candidate_hash
shape_id
candidate_index
problem_index
solution_index
params_json
```

The manifest records the intended candidate order for each shape. Because each candidate is one `Groups` entry, the requested manifest `solution_index` follows candidate index before TensileLite filtering.

The manifest is audit data and the initial mapping hint. Final accepted mapping still comes from TensileLite-generated solution YAMLs.

## TensileLite Build Phase

`build_then_structured_benchmark()` invokes `run_tensilelite()` with:
- `build_only=True`.
- The generated YAML path.
- Profile global parameter items.
- The selected TensileLite executable.
- Optional `CpuThreads` through `--compile-threads`.
- Optional timeout from the profile or CLI.
- Optional stable compile-cache directory.

Build output can come from either full-client layout or build-only cache layout. EvoTensile searches for library artifacts under:

```text
4_LibraryClient/library/gfx*
1_BenchmarkProblems/**/source/library/gfx*
**/source/library/gfx*
```

## Compile Cache

The scheduler can reuse a stable TensileLite build cache under `OUTPUT_DIR/compile_cache` unless `--no-compile-cache` is passed.

The compile-cache key includes:
- Candidate hashes in the batch.
- Compile-relevant global parameters.
- Library logic.
- Problem type hash.

Timing execution parameters such as `NumBenchmarks` and validation count are excluded when they are not compile-relevant. A success marker plus TensileLite cache files are required before reuse. A filesystem lock prevents multiple workers from populating the same cache directory concurrently.

## Accepted-Solution Mapping

After build/codegen, EvoTensile finds final solution YAMLs and calls `build_runnable_pairs()`:
- Read the manifest.
- Build a solution-to-candidate mapper from final YAML solution dictionaries.
- Emit `RunnablePair` records for accepted planned pairs.
- Emit `rejected` rows for planned manifest pairs that did not survive final mapping.
- Emit `unmapped` rows for planned pairs missing from the manifest.

Rejected and unmapped pairs are persisted as negative evidence under the active problem/protocol/shape/candidate key.

## Structured Runner Input

`run_structured_backend()` writes a JSONL pairs file. Each row contains:

```text
shape_id
candidate_hash
m
n
batch
k
problem_index
requested_solution_index
library_solution_index
manifest_solution_index
num_warmups
num_benchmarks
enqueues_per_sync
syncs_per_benchmark
num_elements_to_validate
```

The external runner receives:

```text
--pairs <pairs.jsonl>
--output <results.jsonl>
--validation-backend <cpu|hipblaslt|none>
--library-dir <generated-library-dir>
```

`--runner-bin` defaults to the profile's `./build/evotensile-structured-runner` path. The production scheduler requires a runner binary unless the command is `--dry-run` or `--generate-only`.

## Structured Runner Output

The runner emits one JSON object per sample or negative result. EvoTensile reads fields including:

```text
shape_id
candidate_hash
status
sample_index
time_us
validation or validation_detail
solution_index
```

`validate_structured_samples()` enforces:
- Every emitted pair must be expected.
- The emitted `solution_index` must match the mapped library solution index.
- Positive samples must cover exactly `0..NumBenchmarks-1` with no duplicates.
- Positive samples must have finite positive `time_us`.
- Positive samples must have passing validation unless a timing-only top-up is explicitly allowed.
- A nonzero runner return code cannot be combined with positive rows.

Rows are normalized before DB insertion:
- Passing positive rows become `status='ok'`.
- Unknown validation becomes `validation_unknown`.
- Failed validation becomes `validation_fail`.
- Missing or non-positive time becomes `invalid`.
- Runner timeout rows can be recorded as `runner_timeout` for the whole batch.

## Validation Backends

`BenchmarkProtocol.validation_backend` is passed only to the structured runner. Valid values are:
- `hipblaslt`: GPU-oracle validation, used by default.
- `cpu`: CPU/OpenBLAS-style audit validation when supported by the backend.
- `none`: no validation. Accepted only for trusted timing-only top-ups with prior validation evidence.

`NumElementsToValidate` controls validation execution count in TensileLite/runner parameters. It is not part of benchmark-protocol identity because it does not change timing compatibility.

## Benchmark Protocol Parameters

`BenchmarkProtocol` writes hot-loop steady-state defaults:

```yaml
KernelTime: True
PreciseKernelTime: True
NumWarmups: 10
NumBenchmarks: 10
EnqueuesPerSync: 10
SyncsPerBenchmark: 1
SleepPercent: 0
HardwareMonitor: False
NumElementsToValidate: -1
PredictionThreshold: 2.0
SkipSlowSolutionRatio: 0.0
ParallelGpuExecution: 1
```

The structured runner currently requires `SleepPercent=0`, `HardwareMonitor=False`, and `ParallelGpuExecution=1`.

Cold-loop behavior is intentionally outside the tuning loop because it increases wall time and optimizes for first-request or bursty-idle latency instead of sustained throughput.

## Diagnostic Attribution

If a multi-candidate build fails and some candidates cannot be attributed through normal final-YAML mapping, EvoTensile runs structured TensileLite diagnostics:
- The diagnostics module imports TensileLite Python internals.
- It reconstructs fork permutations from the YAML.
- It captures `SolutionStructs` rejection reasons.
- It runs KernelWriter source processing for candidate-level failures.
- It emits diagnostic JSONL records keyed by candidate hash and phase.

Attributed failures become reusable `build_failed` rows. Unattributed failures become `build_failed_unattributed` or `build_timeout_unattributed`. Those are audit evidence and are not reusable negative cache statuses.

## Run Records

Both build and structured-run phases insert `runs` rows with command, paths, return code, duration, timeout status, and other metadata. Evaluation rows reference `run_id` when a specific run produced them.

This DB-backed contract keeps measurement reproducible without scraping stdout order, kernel names, or client CSV files.
