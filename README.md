# EvoTensile

Work in progress. README is AI-generated.

EvoTensile is an external smart-search autotuner for TensileLite / hipBLASLt. It proposes complete TensileLite candidate bundles, emits them as TensileLite `Groups`, uses TensileLite for solution/code-object generation, and records structured timing/cache metadata for iterative search. It is inspired by [Helion](https://github.com/pytorch/helion) and [rocm_wmma_gemm](https://github.com/adelj88/rocm_wmma_gemm).

The repository currently includes one concrete target configuration, but the core code is intended to stay reusable: candidate hashing, shape handling, search-space encoding, YAML emission, runner orchestration, benchmark-protocol hashing, validation-aware ingestion, ranking, adaptive finalist top-ups, hipBLASLt baseline import, and logic-file update helpers.

Target-specific notes, exact artifacts, measured results, and remaining kernel-specific work are in `PLAN.md`.

## Workflow

A normal EvoTensile tuning loop is:

1. Define problem, shapes, and search space.
2. Search configs with validation-gated TensileLite measurements, including current hipBLASLt baseline import.
3. Inspect and rank cached results.
4. Adaptively top up uncertain finalists until timing confidence is sufficient.
5. Update checked-in hipBLASLt GridBased logic YAMLs directly from the DB.
6. Rebuild hipBLASLt and validate installed performance.
7. Verify installed hipBLASLt correctness with repeatable CPU-reference checks.

### 1. Define Problem, Shapes, And Search Space

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

### 2. Search Configs

`schedule-batches` is the main entry point for searching. It plans missing `(shape, candidate)` work against the SQLite cache, emits TensileLite YAML batches, runs TensileLite build/codegen, maps accepted solutions from final YAML once, and ingests structured validation-gated result rows keyed directly by `shape_id` and `candidate_hash`.

Dry-run a plan:

```bash
python3 -m evotensile.cli schedule-batches \
  --db out/evotensile.sqlite \
  --output-dir out/search \
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

Import the current hipBLASLt-selected configs once per DB/problem/grid so they participate in search and adaptive sampling:

```bash
python3 scripts/import_hipblaslt_baselines.py \
  --db out/evotensile.sqlite \
  --output-dir out/hipblaslt_baselines \
  --profile gfx1151-nt-hhs \
  --bench ~/rocm-libraries/build/hipblaslt-bench/clients/hipblaslt-bench \
  --tensile-libpath "$ROCM_PATH/lib/hipblaslt/library/gfx1151" \
  --runner-bin ./build/evotensile-structured-runner \
  --build-timeout 1800 \
  --runner-timeout 600 \
  --keep-going
```

Run planned batches with adaptive sampling:

```bash
python3 -m evotensile.cli schedule-batches \
  --db out/evotensile.sqlite \
  --output-dir out/search \
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
  --adaptive-sampling \
  --adaptive-initial-samples 3 \
  --adaptive-min-samples 20 \
  --adaptive-max-samples 80 \
  --keep-going
```

The external runner consumes TensileLite build artifacts from either full-client `4_LibraryClient/library/gfx*` output or build-only `1_BenchmarkProblems/**/source/library/gfx*` cache output. Each SQLite DB file is one evidence namespace for a target hardware/environment/campaign. Use separate DB paths when comparing incompatible campaigns. Each `schedule-batches` invocation writes `schedule_metadata.json` in `--output-dir` so runs can be audited without parsing stdout.

Useful proposal modes include `seed-random`, `local`, `seed-random-local`, `de`, `seed-random-de`, `gomea`, `seed-random-gomea`, and `evolutionary`. Exact-shape and nearest-shape validation-passed winners, including imported hipBLASLt baselines when they remain best, can seed first-pass proposals through `--transfer-shapes` / `--transfer-per-shape`.

Supported protocol overrides are typed CLI options such as `--num-benchmarks`, `--num-warmups`, `--enqueues-per-sync`, `--syncs-per-benchmark`, and `--num-elements-to-validate`. `NumBenchmarks` and `NumElementsToValidate` are execution budgets rather than cache identity fields, so adaptive top-ups pool with the fully validated timing evidence. The default uses full validation with `NumElementsToValidate=-1`; unsupported TensileLite global parameters are intentionally not accepted by the search CLI.

Validation is a hard gate: only `status=ok` rows with passing validation, or GPU-only top-up rows backed by prior passing validation for the same pair, should be ranked or used as positive cache entries. Unknown validation is never ranked as positive.

### 3. Inspect And Rank Cached Results

Summarize cache status:

```bash
python3 -m evotensile.cli summarize-cache \
  --db out/evotensile.sqlite \
  --profile gfx1151-nt-hhs
```

Rank validation-passed observations:

```bash
python3 -m evotensile.cli rank-evals \
  --db out/evotensile.sqlite \
  --profile gfx1151-nt-hhs \
  --min-samples 2
```

Structured scheduler runs ingest their own JSONL results directly into SQLite. The old TensileLite `LibraryClient` CSV/log ingestion path has been removed.

### 4. Adaptive Finalist Top-Ups

Search-time timing is noisy enough that top-1 screening can miss the final winner. `schedule-batches --adaptive-sampling` starts with a small timing budget, then appends only the missing samples for statistically plausible contenders. The first validated run for each `(shape, candidate)` pair performs CPU/OpenBLAS validation; later GPU-only top-ups use `NumElementsToValidate=0` and are accepted only when prior validation evidence exists.

After adaptive sampling, `scripts/update_hipblaslt_gridbased_logic.py` queries validation-passed DB winners directly; no intermediate artifact is required.

### 5. Update hipBLASLt GridBased Logic

Update checked-in hipBLASLt logic YAMLs directly from the SQLite DB. The updater is target-aware today; it defaults to the bundled gfx1151 HHS/HHS+AuxH/BBS/BBS+AuxB files.

```bash
python3 scripts/update_hipblaslt_gridbased_logic.py \
  --db out/evotensile.sqlite \
  --profile gfx1151-nt-hhs \
  --min-samples 10 \
  --dry-run
python3 scripts/update_hipblaslt_gridbased_logic.py \
  --db out/evotensile.sqlite \
  --profile gfx1151-nt-hhs \
  --min-samples 10
```

The updater writes TensileLite-style YAML formatting, retargets solution names, trims generated solution dictionaries to the key schema/order used by existing checked-in GridBased YAMLs, strips benchmark-only embedded `ProblemType`, and applies target-specific build-valid normalizations.

Review the hipBLASLt source diff before rebuilding:

```bash
cd ~/rocm-libraries
git diff --stat -- projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/GridBased
```

### 6. Rebuild And Validate Performance

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

### 7. Verify Installed Correctness

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

Benchmark protocol is represented by the typed `BenchmarkProtocol` profile object and included in the benchmark-protocol hash. `NumBenchmarks` and `NumElementsToValidate` are intentionally excluded from benchmark protocol identity because they control sampling/validation execution, not timing compatibility. Compile-only settings such as `CpuThreads` are phase-specific and also excluded.

## Winner Selection Math

For each shape, EvoTensile treats each validation-passed candidate as one noisy timing arm. Timing samples are analyzed in log-time space so multiplicative noise and percent gaps are handled consistently.

For candidate $c$ with positive timing samples $t_{c,1}, \dots, t_{c,n}$:

$$
y_{c,i} = \log(t_{c,i}), \qquad s_c = \mathrm{median}(y_{c,*}).
$$

The score $s_c$ is the median log time, so lower is better. The robust log-noise estimate is:

$$
\sigma_c = \max\left(\mathrm{stdev}(y_c), 1.483 \mathrm{MAD}(y_c), \frac{\mathrm{IQR}(y_c)}{1.349}\right).
$$

The approximate standard error of the median log time is:

$$
\mathrm{SE}_c = 1.253 \frac{\sigma_c}{\sqrt{n}}.
$$

Let $b$ be the current best candidate by lowest $s_c$. For another candidate $c$, define the log-time gap and confidence interval:

$$
g_c = s_c - s_b,
$$

$$
\mathrm{CI}_c = g_c \pm z_\alpha \sqrt{\mathrm{SE}_c^2 + \mathrm{SE}_b^2}.
$$

A candidate remains plausible if its lower confidence bound is within the indifference zone $\epsilon$:

$$
\mathrm{CI}_{c,\mathrm{low}} \le \log(1 + \epsilon).
$$

This means the candidate is not confidently slower than the current best by more than the requested percent tolerance. If no non-best candidate is plausible, the best is resolved. If all plausible candidates are mutually inside the $\pm\epsilon$ zone, the shape is marked practically equivalent and no more samples are scheduled.

When a shape remains unresolved, EvoTensile estimates a target sample count for the plausible contenders. For contender $c$:

$$
d_c = \max\left(\left| |s_c - s_b| - \epsilon_{\log} \right|, \delta_{\min}\right),
$$

$$
n_c = \left\lceil\left(\frac{z_\alpha \cdot 1.253 \sqrt{\sigma_b^2 + \sigma_c^2}}{d_c}\right)^2\right\rceil.
$$

The scheduled target is the maximum requested $n_c$, rounded up to `--adaptive-sample-step` and clamped to `--adaptive-min-samples` / `--adaptive-max-samples`. `--adaptive-max-k` limits how many plausible candidates are topped up for one shape.

Pseudocode:

```text
for each shape:
  samples = validation-passed timing rows for this DB/profile/protocol
  stats = robust log-time stats per candidate
  best = candidate with lowest median log time
  plausible = [best]

  for contender in candidates sorted by median log time:
    gap = contender.score - best.score
    ci = gap ± z * sqrt(best.se^2 + contender.se^2)
    if ci.low <= epsilon_log:
      plausible.append(contender)

  if plausible == [best]:
    accept best
  elif plausible candidates are pairwise equivalent within epsilon_log:
    accept the fastest as representative of an equivalent set
  else:
    target_samples = estimate_needed_samples(plausible)
    schedule only missing samples for plausible[:adaptive_max_k]
```

Correctness is handled separately from repetition count. The first accepted run for a `(shape, candidate)` pair performs CPU/OpenBLAS validation. Later adaptive top-ups set `NumElementsToValidate=0` and are accepted only if the DB already contains passing validation evidence for that pair.

## Current Limitations

- The bundled problem type and search-space domains target gfx1151 FP16 NT HHS first.
- Surrogate/LFBO proposal is planned but not implemented. Keep `PredictionThreshold: 2.0` to disable heuristics like Formocast and Origami in TensileLite, until they're accurate enough on gfx1151.
- Timeout classification and multi-candidate build-failure attribution are still incomplete.
- BBS/AuxH/AuxB retargeting needs target-specific validation before making measured-performance claims for those variants.
- The production structured backend is intentionally narrow: it supports the current gfx1151 FP16 NT HHS bias + `scaleAlpha_vector` target and still needs broader target coverage before it is a general GEMM runner.
