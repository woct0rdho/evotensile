# EvoTensile

Work in progress. README is AI-generated.

EvoTensile is a smart-search autotuner for TensileLite. It proposes complete TensileLite candidate bundles, emits them as TensileLite `Groups`, uses TensileLite for solution/code-object generation, and records structured timing/cache metadata for iterative search. It is inspired by [Helion](https://github.com/pytorch/helion) and [rocm_wmma_gemm](https://github.com/adelj88/rocm_wmma_gemm).

The repository currently includes one concrete target configuration, but the core code is intended to stay reusable: candidate hashing, shape handling, search-space encoding, YAML emission, runner orchestration, benchmark-protocol hashing, validation-aware ingestion, ranking, adaptive finalist top-ups, hipBLASLt baseline import, and logic-file update helpers.

Target-specific notes, exact artifacts, measured results, and remaining kernel-specific work are under `docs/`.

## Workflow

1. Define problem type, input shapes, and config search space.
2. Search configs with imported hipBLASLt baseline and evolutionary algorithms.
3. Repair local outliers by rerunning search with neighbor-seeded configs.
4. Inspect and rank results.
5. Update hipBLASLt configs.
6. Rebuild and reinstall hipBLASLt.
7. Verify correctness and performance of reinstalled hipBLASLt.

### 1. Define Problem, Shapes, And Search Space

A target profile defines:
- a TensileLite problem type, including data types, layout, batching, epilogue flags, and validation settings.
- exact input shapes, usually represented as `shape_id = m{M}_n{N}_b{batch}_k{K}`.
- a typed benchmark protocol used consistently for YAML generation, runner JSONL, and cache hashing.
- a candidate search space made of complete TensileLite solution dictionaries, not independent Cartesian products.

Each target profile derives `problem_type_hash` and `benchmark_protocol_hash`. Pass the selected profile with `--profile <profile-name>`.

Inspect a target search space with:

```bash
python3 -m evotensile.cli summarize-space --profile <profile-name>
```

`proposal-coverage` helps define and maintain a profile's search space by generating proposals without executing them, then reporting value coverage and invalid-rule counts so proposal bias can be tuned without shrinking the underlying domains.

During real schedules, failed multi-candidate TensileLite builds are attributed through structured TensileLite diagnostics instead of log scraping or recursive isolation. Use those diagnostics to keep hard rules source-backed and exact, while keeping proposal heuristics separate from validity.

### 2. Search Configs

`schedule-batches` is the main entry point for searching. It plans missing `(shape, candidate)` work against the SQLite cache, emits TensileLite YAML batches, runs TensileLite build/codegen, maps accepted solutions from final YAML once, and ingests structured validation-gated result rows keyed directly by `shape_id` and `candidate_hash`.

Dry-run a plan:

```bash
python3 -m evotensile.cli schedule-batches \
  --db out/evotensile.sqlite \
  --output-dir out/search \
  --profile <profile-name> \
  --dry-run
```

Build rocisa, TensileLite client, and EvoTensile structured runner:

```bash
cd ~/rocm-libraries/projects/hipblaslt/tensilelite/rocisa
CXX=$ROCM_PATH/llvm/bin/amdclang++ pip install -U --no-deps -e .
cd ~/rocm-libraries
./build_tensilelite_client.sh
cd ~/evotensile
scripts/build_structured_runner.sh
```

The runner build script validates that the expected TensileLite client static libraries exist under `~/rocm-libraries/build/tensilelite-client` before compiling `./build/evotensile-structured-runner`.

Import the current hipBLASLt-selected configs once per DB/problem/grid so they participate in search and adaptive sampling:

```bash
python3 scripts/import_hipblaslt_baselines.py \
  --db out/evotensile.sqlite \
  --output-dir out/hipblaslt_baselines \
  --profile <profile-name> \
  --tensile-libpath "$ROCM_PATH/lib/hipblaslt/library/<gfx-target>"
```

Run planned batches with adaptive sampling:

```bash
python3 -m evotensile.cli schedule-batches \
  --db out/evotensile.sqlite \
  --output-dir out/search \
  --profile <profile-name>
```

The external runner consumes TensileLite build artifacts from either full-client `4_LibraryClient/library/gfx*` output or build-only `1_BenchmarkProblems/**/source/library/gfx*` cache output. Each SQLite DB file is one evidence namespace for a target hardware/environment/campaign. Use separate DB paths when comparing incompatible campaigns. Each `schedule-batches` invocation writes `schedule_metadata.json` in `--output-dir` so runs can be audited without parsing stdout. Profiles provide compile and runner timeout defaults. Pass `0` to a timeout flag to disable it or `--stop-on-error` to fail fast.

Production CLI defaults favor throughput: `--prepare-workers` defaults to available CPU cores, `--compile-threads` defaults to `1`, compile-cache reuse is enabled under `OUTPUT_DIR/compile_cache`, and `--candidate-batch-size` is chosen as the largest profile-bounded value that still leaves enough candidate/shape batches to saturate preparation. Preparation performs build/map/diagnostic/validation in parallel. Timing starts only after that pool drains and always runs serially.

Useful proposal modes include `random`, `seed-random`, `local`, `seed-random-local`, `de`, `seed-random-de`, `gomea`, `seed-random-gomea`, and `evolutionary`. Exact-shape and nearest-shape validation-passed winners, including imported hipBLASLt baselines when they remain best, can initialize non-random proposal operators through `--transfer-shapes` / `--transfer-per-shape`. Command examples omit hyperparameters when the intended value is already the profile or CLI default.

Supported protocol overrides include `--num-benchmarks`, `--num-warmups`, `--enqueues-per-sync`, `--syncs-per-benchmark`, `--num-elements-to-validate`, and `--validation-backend`. The default performs full hipBLASLt GPU-oracle validation with `NumElementsToValidate=-1`. `--validation-backend cpu` selects CPU audit validation. There is no no-validation backend: benchmark-only execution is admitted only after compatible correctness evidence exists.

Validation is a hard gate stored independently from timing. Adaptive top-ups reuse the original compiled and correctness-verified artifacts. They perform no recompilation or repeated verification.

Search-time timing is noisy enough that top-1 screening can miss the final winner. `schedule-batches` uses adaptive sampling by default: it prepares all candidates once, runs a small serial timing budget, then appends only missing benchmark samples for plausible contenders from the prepared-artifact index. Use `--fixed-sampling` only for debugging or fixed-budget utility runs.

Structured scheduler runs ingest their own JSONL results directly into SQLite. The old TensileLite `LibraryClient` CSV/log ingestion path has been removed.

### 3. Repair Local Outliers

Before manual inspection or GridBased updates, `repair-outliers` can identify shapes whose current best config sits below a robust local neighbor envelope in log GFLOP/s space. It then reruns only those shapes, seeding candidates from the outlier's current winner, nearest-shape winners/top candidates, and the selected proposal mode.

```bash
python3 -m evotensile.cli repair-outliers \
  --db out/evotensile.sqlite \
  --output-dir out/repair_outliers \
  --profile <profile-name> \
  --num-random 32 \
  --gomea-count 32
```

This is a search-budget heuristic, not a correctness rule: real performance cliffs from divisibility, edge handling, LDS pressure, or occupancy can legitimately sit below nearby shapes. The command writes `repair_metadata.json` with detected residuals, neighbors, candidate hashes, and planned/executed batch summaries.

### 4. Inspect And Rank Cached Results

Summarize cache status:

```bash
python3 -m evotensile.cli summarize-cache \
  --db out/evotensile.sqlite \
  --profile <profile-name>
```

Rank validation-passed observations:

```bash
python3 -m evotensile.cli rank-evals \
  --db out/evotensile.sqlite \
  --profile <profile-name> \
  --min-samples 2
```

### 5. Update hipBLASLt GridBased Logic

Update checked-in hipBLASLt logic YAMLs directly from the SQLite DB. The updater uses the selected profile to locate and retarget supported logic files. No intermediate winner export is required.

```bash
python3 scripts/update_hipblaslt_gridbased_logic.py \
  --db out/evotensile.sqlite \
  --profile <profile-name>
```

The updater writes TensileLite-style YAML formatting, retargets solution names, trims generated solution dictionaries to the key schema/order used by existing checked-in GridBased YAMLs, strips benchmark-only embedded `ProblemType`, and applies target-specific build-valid normalizations.

Review the hipBLASLt source diff before rebuilding:

```bash
cd ~/rocm-libraries
git diff --stat -- projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/<gfx-target>/GridBased
```

### 6. Rebuild And Validate Performance

Rebuild hipBLASLt from the modified `~/rocm-libraries` tree and install into the intended ROCm SDK prefix. By convention, keep the normal hipBLASLt build tree at `~/rocm-libraries/build/hipblaslt/`. Only override `BUILD_DIR` when comparing multiple versions.

```bash
cd ~/rocm-libraries
GPU_TARGETS=<gfx-target> ./build_hipblaslt.sh
```

Build client tools in the normal client build tree `~/rocm-libraries/build/hipblaslt-bench/`, also avoiding `BUILD_DIR` overrides unless comparing versions:

```bash
cd ~/rocm-libraries
TARGET=hipblaslt-bench GPU_TARGETS=<gfx-target> ./build_hipblaslt_bench.sh
TARGET=hipblaslt-test GPU_TARGETS=<gfx-target> ./build_hipblaslt_bench.sh
```

Then run an application-level benchmark with the intended runtime environment. If the Python runtime uses a separate TensileLite asset package, point `HIPBLASLT_TENSILE_LIBPATH` at the newly installed assets so the rebuilt logic is actually used.

### 7. Verify Installed Correctness

After performance validation, run repeatable installed-library correctness checks. The lightweight target-specific gate uses `hipblaslt-bench --verify` through the EvoTensile verifier and writes `summary.json`, `results.csv`, and per-case logs:

```bash
cd ~/evotensile
python3 scripts/verify_installed_hipblaslt.py \
  --bench ~/rocm-libraries/build/hipblaslt-bench/clients/hipblaslt-bench \
  --tensile-libpath "$ROCM_PATH/lib/hipblaslt/library/<gfx-target>"
```

For broader upstream regression coverage, run `hipblaslt-test` with GTest XML output:

```bash
cd ~/rocm-libraries/build/hipblaslt-bench/clients
HIPBLASLT_TENSILE_LIBPATH="$ROCM_PATH/lib/hipblaslt/library/<gfx-target>" \
LD_LIBRARY_PATH="$ROCM_PATH/llvm/lib:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}" \
./hipblaslt-test --gtest_filter='<test-filter>' --gtest_output=xml:/tmp/hipblaslt_test.xml
```

## Benchmark Protocol

The default generated YAML uses hot-loop / steady-state timing:

```yaml
NumWarmups: 10
NumBenchmarks: 10
EnqueuesPerSync: 10
SyncsPerBenchmark: 1
SleepPercent: 0
HardwareMonitor: False
NumElementsToValidate: -1
```

Cold-loop behavior is intentionally not tracked during tuning because it increases tuning time and optimizes for first-run or bursty-idle effects rather than sustained throughput. Analyze cold-loop behavior later only if first-request latency becomes important.

Benchmark protocol is represented by the typed `BenchmarkProtocol` profile object and included in the benchmark-protocol hash. `NumBenchmarks` and `NumElementsToValidate` are intentionally excluded from benchmark protocol identity because they control sampling/validation execution, not timing compatibility. Compile-only settings such as `CpuThreads` are phase-specific and also excluded.

## Current Limitations

- Bundled profiles and runners are target-specific. Broader data types, layouts, and epilogues need profile and backend coverage.
- Surrogate/LFBO proposals are not implemented. Keep `PredictionThreshold: 2.0` to disable heuristics like Formocast and Origami in TensileLite, until they are accurate enough on gfx1151.
- Logic file update helpers are profile-aware, but each new target variant needs validation before measured-performance claims.
- The production structured backend is intentionally narrower than the generic search abstractions and needs broader target coverage before it is a general GEMM runner.
