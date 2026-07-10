# TensileLite Measurement Design

This document describes how EvoTensile builds, validates, and times candidates with TensileLite.

## Boundary

EvoTensile orchestrates TensileLite externally:
- It writes TensileLite YAML and manifest files.
- It invokes TensileLite build/codegen in build-only mode.
- It maps accepted final-YAML solutions back to exact EvoTensile candidates.
- It runs correctness verification and timing as separate structured-runner modes.
- It stores correctness evidence separately from timing samples.

The old TensileLite `LibraryClient` CSV/log ingestion path and the combined validation-plus-timing runner path are not used by the production scheduler.

## YAML And Manifest

`write_tensilelite_yaml()` writes one complete candidate dictionary per `Groups` entry and exact shapes under `BenchmarkFinalParameters`. The active profile supplies problem type, global parameters, and target library logic.

Every YAML has a `config.manifest.csv` containing:

```text
candidate_hash
shape_id
candidate_index
problem_index
solution_index
params_json
```

The manifest records intended ordering. Final accepted mapping remains authoritative and comes from TensileLite-generated solution YAMLs.

## Prepare Queue

One scheduler wave has two phases separated by a hard barrier.

The parallel prepare queue performs, for every batch:
- TensileLite build/codegen.
- Final-YAML mapping and positive salvage.
- Structured diagnostics for unattributed mixed-build failures.
- Correctness verification for accepted pairs that lack compatible cached validation.

`--prepare-workers` controls this queue and defaults to available CPU cores. `--compile-threads` controls CPU threads inside one TensileLite build and defaults to `1`.

Compilation, diagnostics, CPU validation, and GPU validation may overlap each other. After all prepare futures finish, the worker pool is shut down. No timing starts until every prepare subprocess has exited.

Timeouts kill the complete subprocess process group before a prepare future completes. Compiler descendants therefore cannot survive the phase barrier.

## Serial Benchmark Queue

After the prepare pool has fully drained, the scheduler benchmarks prepared batches one at a time. Benchmark mode:
- Reuses the exact generated library and code object produced during preparation.
- Accepts only pairs that compiled, mapped, and passed correctness verification.
- Requires `num_elements_to_validate=0`.
- Runs warmups and timed samples only.
- Emits `NO_CHECK` from the runner. Python stores timing rows as backed by prior validation.

No compilation, diagnostics, or correctness verification is launched from the benchmark queue.

## APU Activity Gate

The scheduler barrier is the primary ordering mechanism. A machine-wide shared/exclusive filesystem gate provides cross-process protection:
- TensileLite builds and diagnostics acquire shared access.
- The structured runner acquires shared access in `validate` mode.
- The structured runner acquires exclusive access in `benchmark` mode.
- Repository-owned standalone hipBLASLt benchmark/verification utilities acquire exclusive access.

The default path is `/tmp/evotensile-apu.lock`. `EVOTENSILE_APU_LOCK_PATH` overrides it.

Consequences:
- Two benchmarks cannot overlap.
- A benchmark cannot overlap compilation, diagnostics, CPU validation, or GPU validation.
- Compilation and correctness verification may run concurrently.
- Direct invocation of `evotensile-structured-runner` still honors the gate because locking is enforced in the binary.

External GPU programs that do not participate in this gate remain outside EvoTensile's control.

## Compile Cache

The scheduler can reuse a stable TensileLite build cache under `OUTPUT_DIR/compile_cache` unless `--no-compile-cache` is passed.

The cache key includes:
- Candidate hashes in the batch.
- Compile-relevant global parameters.
- Library logic.
- Problem type hash.

Timing budgets and validation extent are excluded when they do not affect code generation. A success marker and TensileLite cache files are required before reuse. A cache-specific lock prevents duplicate population by prepare workers.

Validation, probe, main benchmark, and adaptive top-up modes always use the same prepared library directory. None of the timing stages invoke TensileLite again.

## Accepted-Solution Mapping

After build/codegen, `build_runnable_pairs()`:
- Reads the manifest.
- Maps generated solution dictionaries to candidate hashes.
- Emits `RunnablePair` records for accepted planned pairs.
- Emits `rejected` for planned manifest pairs absent from final mapping.
- Emits `unmapped` for planned pairs absent from the manifest.

If a mixed build returns nonzero but contains accepted final-YAML solutions, those accepted candidates continue through validation and timing. Missing candidates are attributed through structured diagnostics. They are not inferred rejected from absence alone.

## Structured Runner Modes

The runner receives:

```text
--mode validate|benchmark
--pairs <pairs.jsonl>
--output <results.jsonl>
--validation-backend <cpu|hipblaslt>
--library-dir <generated-library-dir>
```

Each pair row contains exact shape identity, candidate identity, mapped solution index, warmup/timing counts, and validation extent.

### Validate

`--mode validate`:
- Requires nonzero validation extent.
- Launches the candidate once.
- Runs the selected CPU or hipBLASLt GPU oracle.
- Emits exactly one result row per pair.
- Emits no timing value.

### Benchmark

`--mode benchmark`:
- Requires `num_elements_to_validate=0`.
- Performs no correctness verification.
- Runs requested warmups and timed launches.
- Emits exactly `NumBenchmarks` finite positive samples per pair.
- Emits `NO_CHECK` to make accidental combined validation detectable.

Python validates pair identity, solution index, row count, sample indices, mode-specific timing fields, and return-code consistency before insertion.

## Correctness Identity

Correctness evidence has its own validation-protocol hash. It includes:
- Validator protocol version.
- Validation backend.
- Validation extent.
- Input initialization settings.
- `CEqualD` behavior.

Timing compatibility uses the benchmark-protocol hash. `NumBenchmarks` remains an execution budget rather than a compatibility field. `BenchmarkRole` distinguishes low-fidelity probe timing from main timing even if their launch settings are configured identically.

A timing row can be produced only for a pair present in the prepared batch's validation-passed set or in compatible cached validation evidence.

## Validation Backends

Supported validation backends are:
- `hipblaslt`: GPU-oracle validation, used by default for production tuning.
- `cpu`: CPU/OpenBLAS-style audit validation when supported.

CPU validation is not feasible as a production tuning backend. OpenBLAS otherwise creates a large thread pool per concurrent validator, so audit runs must explicitly limit `OPENBLAS_NUM_THREADS`. Even with that limit, the CPU reference is substantially slower and retains additional host tensors. EvoTensile deliberately does not distribute validation work across both CPU and GPU: the expected throughput gain does not justify separate resource pools, backend-aware dispatch, evidence reconciliation, and the additional failure modes that mixed scheduling would introduce.

Benchmark mode and hipBLASLt validation initialize deterministic A, B, C, bias, and scale tensors directly in `hipMalloc` storage. They do not retain full host tensor copies. HipBLASLt validation additionally allocates only the device reference output and comparison summary required by the GPU oracle. CPU validation retains host A, B, C, bias, scale, and result tensors because its reference calculation consumes them.

On the `8192,8192,1,8192` shape, the device-initialized hipBLASLt path reduced observed structured-runner RSS from roughly `891 MiB` to `235 MiB`. The measured GTT increase was about `947 MiB`, including tensors, the fixed workspace, loaded code objects, and runtime overhead.

There is no public `none` validation backend and no trusted-validation bypass. Skipping correctness is represented only by benchmark mode after compatible validation evidence already exists.

## Adaptive Timing

Adaptive sampling prepares the candidate set once. The scheduler then:
- Runs a separate three-sample, one-enqueue, zero-warmup probe for every validation-passed pair.
- Screens only candidates confidently slower than the best compatible shape reference by more than the configured coarse factor.
- Runs the main timing protocol for probe survivors.
- Loads main-protocol timing statistics and selects plausible contenders within the final indifference zone.
- Runs benchmark-only top-up subsets from the original prepared-artifact index.
- Repeats up to the configured adaptive-round limit.

Probe evidence has a separate protocol hash and cannot enter main ranking. Missing probe evidence fails open. Probe, main, and adaptive rounds do not compile, remap, diagnose, or validate candidates. A contender without a successfully prepared artifact is ineligible for timing.

## Hot-Loop Confirmation

Broad search normally uses a cheap main protocol so many candidates can provide feedback. Final performance claims use a separate hot-loop confirmation over a small ranked set.

`hot_confirm_topk()`:
- ranks screening candidates under the requested main protocol.
- requires compatible passed validation evidence.
- resolves each finalist's generated library and mapped solution from run metadata.
- runs benchmark mode with `20` warmups, `10` samples, and `10` enqueues per sample.
- writes JSON and CSV rankings without recompilation or repeated validation.

Confirmation uses `NumElementsToValidate=0` in benchmark mode and relies on the validation table as its correctness gate. The helper is used by real blind campaign tooling described in `docs/blind_experiment_infrastructure.md`. Statistical interpretation of screening and confirmation is documented in `docs/noisy_measurements.md`.

## Diagnostic Attribution

If a multi-candidate build fails and final mapping cannot attribute every candidate, structured diagnostics reconstruct TensileLite solution permutations and KernelWriter processing.

Attributed failures become reusable `build_failed` rows. Unattributed failures become `build_failed_unattributed` or `build_timeout_unattributed`, which remain audit-only. Diagnostic failure does not suppress successfully built and validated candidates from the same mixed batch.

## Run Records

Build, diagnostic, validation, and benchmark invocations insert separate `runs` rows with command, mode, paths, return code, duration, and timeout status. CLI metadata records the probe policy/hash, survivor and screened pair counts, and whether an executed batch belongs to probe, initial main timing, or an adaptive top-up.
