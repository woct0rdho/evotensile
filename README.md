# EvoTensile

Work in progress. README is AI-generated. I'm working on some better math to determine what configs to keep.

EvoTensile is an external smart-search autotuner for TensileLite / hipBLASLt. It proposes complete TensileLite candidate bundles, emits them as TensileLite `Groups`, uses TensileLite for solution/code-object generation, and records structured timing/cache metadata for iterative search. It is inspired by [Helion](https://github.com/pytorch/helion) and [rocm_wmma_gemm](https://github.com/adelj88/rocm_wmma_gemm).

The repository currently includes one concrete target configuration, but the core code is intended to stay reusable: candidate hashing, shape handling, search-space encoding, YAML emission, runner orchestration, benchmark-protocol hashing, validation-aware ingestion, ranking, finalist retiming, current-hipBLASLt comparison, and logic-file update helpers.

Target-specific notes, exact artifacts, measured results, and remaining kernel-specific work are in `PLAN.md`.

## Workflow

A normal EvoTensile tuning loop is:

1. Define problem, shapes, and search space.
2. Search configs with validation-gated TensileLite measurements.
3. Inspect and rank cached results.
4. Retime top-K finalists with stronger validation.
5. Compare with current hipBLASLt and export hybrid winners.
6. Update checked-in hipBLASLt GridBased logic YAMLs from the hybrid export.
7. Rebuild hipBLASLt and validate installed performance.
8. Verify installed hipBLASLt correctness with repeatable CPU-reference checks.

## Repository Pieces

- `evotensile/candidate.py`: canonical candidate representation and stable hashes.
- `evotensile/shapes.py`: exact-shape helpers and bundled pilot-grid shape generation.
- `evotensile/search_space.py`: target search-space domains, constraints, and seeded/random candidates.
- `evotensile/search/`: random, local, differential-evolution, and GOMEA-style proposal operators.
- `evotensile/yaml_writer.py`: TensileLite config emission using complete candidate `Groups`.
- `evotensile/runner.py`: TensileLite codegen/build subprocess orchestration.
- `evotensile/structured_runner.py`: exact pair mapping, JSONL runner contract, external runner dispatch, and direct DB ingestion.
- `csrc/structured_runner.cpp`: narrow production HIP/TensileLite backend for the current gfx1151 FP16 NT HHS target.
- `scripts/build_structured_runner.sh`: builds `./build/evotensile-structured-runner` against an existing TensileLite client build.
- `evotensile/database.py` and `evotensile/cache.py`: SQLite storage and cache queries.
- `evotensile/scheduler.py`: cache-aware batch planning and execution.
- `scripts/retime_topk.py`: exact top-K finalist retiming.
- `scripts/compare_hipblaslt_bench.py`: installed hipBLASLt comparison and hybrid export.
- `scripts/update_hipblaslt_gridbased_logic.py`: update checked-in hipBLASLt GridBased YAMLs from a hybrid export.
- `scripts/verify_installed_hipblaslt.py`: quick installed hipBLASLt correctness test using `hipblaslt-bench --verify`.

## 1. Define Problem, Shapes, And Search Space

A target profile defines:

- a TensileLite problem type, including data types, layout, batching, epilogue flags, and validation settings;
- exact input shapes, usually represented as `shape_id = m{M}_n{N}_b{batch}_k{K}`;
- a typed benchmark protocol used consistently for YAML generation, runner JSONL, and cache hashing;
- a candidate search space made of complete TensileLite solution dictionaries, not independent Cartesian products.

The current bundled profile is `gfx1151-nt-hhs`. Profile code derives `problem_type_hash` and `benchmark_protocol_hash`; the search CLI no longer accepts raw hash overrides or arbitrary TensileLite global parameters.

Inspect a target search space with:

```bash
python3 -m evotensile.cli summarize-space --num-random 128
```

## 2. Search Configs

`schedule-batches` is the main entry point for searching. It plans missing `(shape, candidate)` work against the SQLite cache, emits TensileLite YAML batches, runs TensileLite build/codegen, maps accepted solutions from final YAML once, and ingests structured validation-gated result rows keyed directly by `shape_id` and `candidate_hash`.

Dry-run a plan:

```bash
python3 -m evotensile.cli schedule-batches \
  --db out/evotensile.sqlite \
  --output-dir out/search \
  --version-name my_target_hotloop_v0 \
  --profile gfx1151-nt-hhs \
  --limit-shapes 100 \
  --candidate-batch-size 32 \
  --shape-batch-size 100 \
  --dry-run
```

Build the TensileLite client prerequisites and then the external structured runner before scheduling real measurements:

```bash
cd ~/rocm-libraries
./build_tensilelite_client.sh
cd ~/evotensile
scripts/build_structured_runner.sh
```

The runner build script validates that the expected TensileLite client static libraries exist under `~/rocm-libraries/build/tensilelite-client` before compiling `./build/evotensile-structured-runner`.

Run planned batches:

```bash
python3 -m evotensile.cli schedule-batches \
  --db out/evotensile.sqlite \
  --output-dir out/search \
  --version-name my_target_hotloop_v0 \
  --profile gfx1151-nt-hhs \
  --proposal seed-random-gomea \
  --num-random 64 \
  --gomea-count 64 \
  --transfer-shapes 4 \
  --transfer-per-shape 2 \
  --candidate-batch-size 32 \
  --shape-batch-size 100 \
  --compile-threads 4 \
  --runner-bin ./build/evotensile-structured-runner \
  --build-timeout 1800 \
  --runner-timeout 600 \
  --keep-going
```

The external runner consumes TensileLite build artifacts from either full-client `4_LibraryClient/library/gfx*` output or build-only `1_BenchmarkProblems/**/source/library/gfx*` cache output. Each `schedule-batches` invocation writes `schedule_metadata.json` in `--output-dir` so runs can be audited without parsing stdout.

Useful proposal modes include `seed-random`, `local`, `seed-random-local`, `de`, `seed-random-de`, `gomea`, `seed-random-gomea`, and `evolutionary`.

Supported protocol overrides are typed CLI options such as `--num-benchmarks`, `--num-warmups`, `--enqueues-per-sync`, `--syncs-per-benchmark`, and `--num-elements-to-validate`. The default uses full validation with `NumElementsToValidate=-1`; unsupported TensileLite global parameters are intentionally not accepted by the search CLI.

Validation is a hard gate: only `status=ok` rows with passing validation should be ranked or used as positive cache entries. Unknown validation is never ranked as positive.

## 3. Inspect And Rank Cached Results

Summarize cache status:

```bash
python3 -m evotensile.cli cache-summary \
  --db out/evotensile.sqlite \
  --version-name my_target_hotloop_v0 \
  --profile gfx1151-nt-hhs
```

Rank validation-passed observations:

```bash
python3 -m evotensile.cli rank-evals \
  --db out/evotensile.sqlite \
  --version-name my_target_hotloop_v0 \
  --profile gfx1151-nt-hhs \
  --min-samples 2
```

Structured schedule/retime runs ingest their own JSONL results directly into SQLite. The old TensileLite `LibraryClient` CSV/log ingestion path has been removed.

## 4. Retime Top-K Finalists

Search-time timing is noisy enough that top-1 screening can miss the final winner. Use `scripts/retime_topk.py` to retime the top candidates per shape with the same full-validation correctness gate and more reliable finalist timing.

```bash
python3 scripts/retime_topk.py \
  --db out/evotensile.sqlite \
  --output-dir out/topk_retime \
  --source-version-name my_target_hotloop_v0 \
  --target-version-name my_target_hotloop_v0_top4_fullval \
  --profile gfx1151-nt-hhs \
  --top-k 4 \
  --compile-threads 4 \
  --runner-bin ./build/evotensile-structured-runner \
  --build-timeout 1800 \
  --runner-timeout 600 \
  --keep-going
```

After retiming, export the per-shape winners using the project export script for the current target/artifact layout:

```bash
python3 scripts/export_winners.py \
  --db out/evotensile.sqlite \
  --output-dir out/topk_retime_export \
  --version-name my_target_hotloop_v0_top4_fullval \
  --profile gfx1151-nt-hhs \
  --min-samples 10
```

## 5. Compare With Current hipBLASLt And Export Hybrid Winners

Use `scripts/compare_hipblaslt_bench.py` when `hipblaslt-bench` can express the target operation. The comparison reports installed/current hipBLASLt performance and can export a hybrid winner set that keeps the tuned candidate only when it beats the current installed solution.

```bash
python3 scripts/compare_hipblaslt_bench.py \
  --winners-csv out/topk_retime_export/winners.csv \
  --output-dir out/hipblaslt_bench_compare \
  --bench ~/rocm-libraries/build/hipblaslt-bench/clients/hipblaslt-bench \
  --tensile-libpath "$ROCM_PATH/lib/hipblaslt/library/gfx1151" \
  --hybrid-export-dir out/hybrid_best_export
```

This is a practical installed-library comparison, not necessarily an identical timing protocol: `hipblaslt-bench` reports one average over hot launches, while EvoTensile uses the configured TensileLite benchmark groups.

## 6. Update hipBLASLt GridBased Logic

Once a hybrid export exists, update checked-in hipBLASLt logic YAMLs. The updater is target-aware today; it defaults to the bundled gfx1151 HHS/HHS+AuxH/BBS/BBS+AuxB files.

```bash
python3 scripts/update_hipblaslt_gridbased_logic.py --dry-run
python3 scripts/update_hipblaslt_gridbased_logic.py
```

The updater writes TensileLite-style YAML formatting, retargets solution names, trims generated solution dictionaries to the key schema/order used by existing checked-in GridBased YAMLs, strips benchmark-only embedded `ProblemType`, and applies target-specific build-valid normalizations.

Review the hipBLASLt source diff before rebuilding:

```bash
cd ~/rocm-libraries
git diff --stat -- projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/GridBased
```

## 7. Rebuild And Validate Performance

Rebuild hipBLASLt from the modified `~/rocm-libraries` tree and install into the intended ROCm SDK prefix. By convention, keep the normal hipBLASLt build tree at `~/rocm-libraries/build/hipblaslt/`; only override `BUILD_DIR` when comparing multiple versions.

```bash
cd ~/rocm-libraries
GPU_TARGETS=gfx1151 ./build_hipblaslt.sh
```

Build client tools in the normal client build tree `~/rocm-libraries/build/hipblaslt-bench/`, also avoiding `BUILD_DIR` overrides unless comparing versions:

```bash
cd ~/rocm-libraries
TARGET=hipblaslt-bench GPU_TARGETS=gfx1151 ./build_hipblaslt_bench.sh
TARGET=hipblaslt-test GPU_TARGETS=gfx1151 ./build_hipblaslt_bench.sh
```

Then run an application-level benchmark with the intended runtime environment. If the Python runtime uses a separate TensileLite asset package, point `HIPBLASLT_TENSILE_LIBPATH` at the newly installed assets so the rebuilt logic is actually used.

## 8. Verify Installed Correctness

After performance validation, run repeatable installed-library correctness checks. The lightweight target-specific gate uses `hipblaslt-bench --verify` through the EvoTensile verifier and writes `summary.json`, `results.csv`, and per-case logs:

```bash
cd ~/evotensile
python3 scripts/verify_installed_hipblaslt.py \
  --bench ~/rocm-libraries/build/hipblaslt-bench/clients/hipblaslt-bench \
  --tensile-libpath "$ROCM_PATH/lib/hipblaslt/library/gfx1151" \
  --output-dir out/hipblaslt_correctness
```

For broader upstream regression coverage, run `hipblaslt-test` with GTest XML output:

```bash
cd ~/rocm-libraries/build/hipblaslt-bench/clients
HIPBLASLT_TENSILE_LIBPATH="$ROCM_PATH/lib/hipblaslt/library/gfx1151" \
LD_LIBRARY_PATH="$ROCM_PATH/llvm/lib:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}" \
./hipblaslt-test --gtest_filter='*smoke*' --gtest_output=xml:/tmp/hipblaslt_test_smoke.xml
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

Benchmark protocol is represented by the typed `BenchmarkProtocol` profile object and included in the benchmark-protocol hash. Compile-only settings such as `CpuThreads` are phase-specific and intentionally excluded from benchmark protocol identity.

## Current Limitations

- The bundled problem type and search-space domains target gfx1151 FP16 NT HHS first.
- Surrogate/LFBO proposal is planned but not implemented.
- Timeout classification and multi-candidate build-failure attribution are still incomplete.
- BBS/AuxH/AuxB retargeting needs target-specific validation before making measured-performance claims for those variants.
- The production structured backend is intentionally narrow: it supports the current gfx1151 FP16 NT HHS bias + `scaleAlpha_vector` target and still needs broader target coverage before it is a general GEMM runner.
- Keep `PredictionThreshold: 2.0` to disable heuristics like Formocast and Origami in TensileLite, until they're accurate enough on gfx1151.
