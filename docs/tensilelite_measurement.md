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

`write_tensilelite_yaml()` writes one complete candidate dictionary per `Groups` entry and the batch's explicit artifact-shape scope under `BenchmarkFinalParameters`. The active profile supplies problem type, global parameters, and target library logic. The artifact scope may be an explicit superset of the shapes currently requested for evaluation.

Every YAML has a `config.manifest.csv` containing only exact requested pairs:

```text
candidate_hash
shape_id
candidate_index
problem_index
solution_index
params_json
```

Candidate and problem indices refer to the artifact candidate and shape positions in the YAML. The manifest does not contain unrequested artifact-scope combinations. Final accepted mapping remains authoritative and comes from TensileLite-generated solution YAMLs. The complete request and artifact-scope contract is documented in `docs/exact_pair_scheduling.md`.

## Prepare Queue

One scheduler wave has two phases separated by a hard barrier. The target profile bounds the number of planned batches admitted to one wave. gfx1151 defaults to `32` batches.

The parallel prepare queue performs, for every batch:
- TensileLite build/codegen.
- Final-YAML mapping and positive salvage.
- Structured diagnostics for unattributed mixed-build failures.
- Correctness verification for accepted pairs that lack compatible cached validation.

`--prepare-workers` and `--validation-workers` resolve from the selected target profile unless explicitly overridden. The gfx1151 profile uses `32` preparation workers and independently caps structured validation at one process because this split has provided fast compilation without the ROCr/KFD instability observed under concurrent large-library validation. `--compile-threads` controls CPU threads inside one TensileLite build and defaults to `1`.

Compilation, diagnostics, CPU validation, and GPU validation may overlap subject to the profile validation cap. After all preparation futures in the admitted wave finish, the worker pool is shut down. No timing starts until every prepare subprocess in that wave has exited. The wave then drains serialized timing before another wave can be admitted, so a coordinator may inspect durable DB feedback, resource state, and its soft budget between waves without ever overlapping preparation and timing.

Build, diagnostic, validation, and benchmark subprocesses use their configured operational timeouts. Campaign soft deadlines do not clamp those timeouts after a job starts. A timeout kills the complete subprocess process group before the future completes, so compiler descendants cannot survive the phase barrier.

## Serial Benchmark Queue

After the prepare pool has fully drained, the scheduler benchmarks prepared batches one at a time. Preparation submission order and timing order are separate: cost-aware scheduling can submit longest-predicted work first, while timing defaults to stable planned order and may be replaced by an explicit allocator callback. Benchmark mode:
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

Stable caching uses candidate-centric TensileLite libraries: each prepared batch contains one candidate and its explicit artifact-shape scope. This makes reuse independent of the proposal cohort without attempting to merge generated libraries or code objects. A later request outside that scope requires a distinct cache identity. The cache key includes:
- Candidate hash.
- Sorted exact shape identities, because GridBased generation and final mapping are shape-dependent.
- Compile-relevant global parameters.
- Library logic.
- Problem type hash.

Timing budgets and validation extent are excluded when they do not affect code generation. A success marker and TensileLite cache files are required before reuse. Cache-disabled and generation-only workflows may retain multi-candidate batches.

A cache-specific kernel advisory file lock prevents duplicate population by preparation workers. Its owner record contains a PID, host, creation time, and unique token for diagnostics. Process exit releases the lock automatically, including catastrophic termination. The persistent lock file is reused rather than unlinked while waiters may hold descriptors. Live ownership has a bounded wait and explicit `TimeoutError`.

Validation, probe, main benchmark, and adaptive top-up modes always use the same prepared library directory. None of the timing stages invoke TensileLite again.

After final-YAML mapping, the scheduler registers one shared artifact bundle for the generated library and one mapping per runnable requested pair. The bundle owns build roots, normalized generated solution-YAML paths, library path, optional manifest, and a SHA-256-derived library content identity. Mappings own exact solution indices. Unrequested artifact-scope combinations receive no mapping. Registration failure blocks validation and timing for that prepared batch. Later stabilization, confirmation, and production export use only content-verified bundles. They do not rediscover artifacts by scanning run directories.

## Accepted-Solution Mapping

After build/codegen, `build_runnable_pairs()`:
- Reads the exact requested-pair manifest.
- Maps generated solution dictionaries to candidate hashes.
- Emits `RunnablePair` records for accepted requested pairs.
- Emits `rejected` for requested manifest pairs absent from final mapping.
- Emits `unmapped` for requested pairs absent from the manifest.

Build failures, diagnostic attribution, validation, and timing are likewise restricted to requested keys rather than the artifact candidate/shape product.

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

Python validates pair identity, solution index, row count, sample indices, mode-specific timing fields, and return-code consistency before insertion. Probe, main, and adaptive launches use the configured runner timeout. Campaign control decides whether to admit the enclosing schedule before it starts.

## Correctness Identity

Correctness evidence has its own validation-protocol hash. It includes:
- Validator protocol version.
- Validation backend.
- Validation extent.
- Input initialization settings.
- `CEqualD` behavior.

Timing compatibility uses the benchmark-protocol hash. `NumBenchmarks` remains an execution budget rather than a compatibility field. `BenchmarkRole` distinguishes low-fidelity probe timing from main timing even if their launch settings are configured identically.

A timing row can be produced only for a pair present in the prepared batch's validation-passed set or whose latest compatible validation row is `passed`. Validation failures are stored only under validation identity. They do not create benchmark-cache negatives. Changing validation backend or extent therefore requests fresh validation instead of reusing an incompatible failure.

## Validation Backends

Supported validation backends are:
- `hipblaslt`: GPU-oracle validation, used by default for production tuning.
- `cpu`: CPU/OpenBLAS-style audit validation when supported.

CPU validation is not feasible as a production tuning backend. OpenBLAS otherwise creates a large thread pool per concurrent validator, so audit runs must explicitly limit `OPENBLAS_NUM_THREADS`. Even with that limit, the CPU reference is substantially slower and retains additional host tensors. EvoTensile deliberately does not distribute validation work across both CPU and GPU: the expected throughput gain does not justify separate resource pools, backend-aware dispatch, evidence reconciliation, and the additional failure modes that mixed scheduling would introduce.

Benchmark mode and hipBLASLt validation initialize deterministic A, B, C, bias, and scale tensors directly in `hipMalloc` storage. They do not retain full host tensor copies. HipBLASLt validation additionally allocates only the device reference output and comparison summary required by the GPU oracle. CPU validation retains host A, B, C, bias, scale, and result tensors because its reference calculation consumes them.

On the `8192,8192,1,8192` shape, the device-initialized hipBLASLt path reduced observed structured-runner RSS from roughly `891 MiB` to `235 MiB`. The measured GTT increase was about `947 MiB`, including tensors, the fixed workspace, loaded code objects, and runtime overhead.

There is no public `none` validation backend and no trusted-validation bypass. Skipping correctness is represented only by benchmark mode after compatible validation evidence already exists.

## Adaptive Timing

Adaptive sampling prepares the explicit candidate artifact scopes once. The scheduler then, for requested pairs only:
- runs one one-enqueue, zero-warmup probe sample for every validation-passed pair.
- screens candidates outside the configured coarse factor while retaining the minimum survivor floor.
- gives provisional survivors the remaining samples needed for the three-sample probe target.
- runs the main timing protocol only for candidates with complete surviving probe evidence.
- Loads main-protocol timing statistics and selects plausible contenders within the final indifference zone.
- Runs benchmark-only top-up subsets from the original prepared-artifact index.
- Repeats up to the configured adaptive-round limit.

Probe evidence has a separate protocol hash and cannot enter main ranking. Missing or incomplete probe evidence is not admitted to main timing in that schedule and creates no reusable negative cache row. Before preparation, the scheduler recomputes the initial-stage screen over currently proposed candidates with compatible probe samples. It skips a pair only when the current `ProbePolicy` threshold still screens it and the current cached cohort contains more than the configured minimum survivor floor. The deterministic policy hash is reported with the pre-prepare screened-pair count. Changing thresholds naturally retries the pair. This is an ephemeral timing-allocation decision, never static invalidity or a reusable benchmark-negative row.

Probe, main, and adaptive rounds do not compile, remap, diagnose, or validate candidates. A contender without a successfully prepared artifact is ineligible for timing.

## Hot-Loop Confirmation

Broad search normally uses a cheap main protocol so many candidates can provide feedback. The legacy singleton diagnostic uses a separate hot-loop ranking over a small set. Production deployment uses the campaign confirmation path described in `docs/deployment_selection.md`.

`hot_confirm_topk()`:
- ranks screening candidates under the requested main protocol.
- requires latest compatible passed validation evidence.
- resolves each finalist's generated library and mapped solution from the content-verified artifact registry.
- executes the explicit profile-derived hot protocol through `run_structured_phase()`.
- validates pair identity, solution index, exact sample indices, finite positive timing, `NO_CHECK`, and return-code consistency through `validate_benchmark_samples()`.
- records malformed or timed-out finalists in `summary.json` and continues while budget remains.
- writes JSON and CSV rankings without recompilation, repeated validation, or DB timing insertion.

The blind campaign supplies `20` warmups, `10` samples, and `10` enqueues per sample. This diagnostic uses `NumElementsToValidate=0` in benchmark mode and relies on the validation table as its correctness gate. It is not the production export gate because it neither revalidates nor inserts confirmation timing into the DB.

Production confirmation emits explicit `CONFIRMATION` requests through a dedicated `RealEvaluatorContext(ignore_cache=True)`. Ignoring the cache forces fresh validation and timing for every selected production pair under the requested protocol identities. The soft controller checks each candidate group before admission, lets admitted work drain, records overrun, and stops later admission. Statistical interpretation remains in `docs/noisy_measurements.md`.

## Diagnostic Attribution

If a multi-candidate build fails and final mapping cannot attribute every candidate, structured diagnostics reconstruct TensileLite solution permutations and KernelWriter processing.

Attributed failures become reusable `build_failed` rows. Unattributed failures become `build_failed_unattributed` or `build_timeout_unattributed`, which remain audit-only. Diagnostic failure does not suppress successfully built and validated candidates from the same mixed batch.

## Run Records

Build, diagnostic, validation, and benchmark invocations create `evidence_sources` plus typed `native_runs` rows containing phase, status, duration, and return code. Commands and logs remain filesystem artifacts. Reusable generated paths belong to artifact bundles. CLI metadata records the probe policy/hash, survivor and screened pair counts, and executed timing phase.
