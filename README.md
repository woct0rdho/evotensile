# EvoTensile

Work in progress. README and docs are AI-generated and intended for AI to read.

EvoTensile is a framework for TensileLite kernel tuning using smart search algorithms. It's inspired by [Helion](https://github.com/pytorch/helion), [rocm\_wmma\_gemm](https://github.com/adelj88/rocm_wmma_gemm), [Ductile](https://github.com/ROCm/rocm-libraries/pull/8831), and [GEKO](https://github.com/ROCm/rocm-libraries/pull/8832).

It's equipped with family stratified seeding, GOMEA, learned linkage, learned tree surrogate, and joint search on neighboring shapes. It's suitable to search on whether a single shape or a large grid of shapes.

It separates search strategies from underlying measurements. When evaluating the efficiency of a search strategy, it supports simulated timing based on previous measurement results and running times, rather than actually rerun the measurements.

## Workflow

1. Define problem type, input shapes, and config search space.
2. Discover installed hipBLASLt selections, then search them through the normal scheduler alongside family-QD candidates.
3. Repair local outliers by rerunning search with neighbor-seeded configs.
4. Update hipBLASLt configs.
5. Rebuild and reinstall hipBLASLt.
6. Verify correctness and performance of reinstalled hipBLASLt.

### 1. Define Problem, Shapes, And Search Space

A target profile defines:
- a TensileLite problem type, including data types, layout, batching, epilogue flags, and validation settings.
- exact input shapes, usually represented as `shape_id = m{M}_n{N}_b{batch}_k{K}`.
- a typed benchmark protocol used consistently for YAML generation, runner JSONL, and cache hashing.
- a candidate search space made of complete TensileLite solution dictionaries, not independent Cartesian products.
- exact candidate-shape scheduler requests with separate candidate-centric artifact scopes, so shared builds never imply extra validation or timing work.

Each target profile derives `problem_type_hash` and `benchmark_protocol_hash`. Commands use the registry's default profile when `--profile` is omitted.

Inspect the default target search space with:

```bash
python3 -m evotensile.cli summarize-space
```

When an installation provides another profile, select it explicitly with `--profile <profile-name>`.

`proposal-coverage` helps define and maintain a profile's search space by generating proposals without executing them, then reporting value coverage and invalid-rule counts so proposal bias can be tuned without shrinking the underlying domains.

During real schedules, failed multi-candidate TensileLite builds are attributed through structured TensileLite diagnostics instead of log scraping or recursive isolation. Use those diagnostics to keep hard rules source-backed and exact, while keeping proposal heuristics separate from validity.

### 2. Search Configs

`schedule-batches` is the main entry point for searching. It plans missing `(shape, candidate)` work against the SQLite cache, emits TensileLite YAML batches, runs TensileLite build/codegen, maps accepted solutions from final YAML once, and ingests structured validation-gated result rows keyed directly by `shape_id` and `candidate_hash`.

Dry-run a plan:

```bash
python3 -m evotensile.cli schedule-batches \
  --db out/evotensile.sqlite \
  --output-dir out/search \
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

Discover the current hipBLASLt-selected configs once per DB/problem/grid. Discovery records planning pairs only. Unless `--query-only` is used, the command then schedules those exact pairs through normal compilation, validation, timing, cost, and artifact paths:

```bash
python3 scripts/discover_hipblaslt_baselines.py \
  --db out/evotensile.sqlite \
  --output-dir out/hipblaslt_baselines \
  --tensilelite-libpath "$ROCM_PATH/lib/hipblaslt/library/<gfx-target>"
```

Run planned batches with adaptive sampling:

```bash
python3 -m evotensile.cli schedule-batches \
  --db out/evotensile.sqlite \
  --output-dir out/search
```

The external runner consumes TensileLite build artifacts from either full-client `4_LibraryClient/library/gfx*` output or build-only `1_BenchmarkProblems/**/source/library/gfx*` cache output. Each SQLite DB file is one evidence namespace for a target hardware/environment/campaign. Use separate DB paths when comparing incompatible campaigns. Compatible benchmark overlays can be consolidated into a new read-only replay snapshot with `scripts/merge_compatible_databases.py`. Source DBs remain unchanged and the merge records a source manifest. Each `schedule-batches` invocation writes `schedule_metadata.json` in `--output-dir` so runs can be audited without parsing stdout. Profiles provide compile and runner timeout defaults. Pass `0` to a timeout flag to disable it or `--stop-on-error` to fail fast.

Production CLI defaults favor reusable throughput: the selected profile supplies the preparation-worker cap, compile threads default to one, and compile-cache reuse is enabled under `OUTPUT_DIR/compile_cache`. Cache-backed schedules use singleton candidate libraries so artifacts remain reusable across proposal cohorts. With `--no-compile-cache`, a profile-bounded throughput heuristic chooses the candidate batch size. Preparation performs build/map/diagnostic/validation in parallel. Timing starts only after that pool drains and always runs serially.

With no custom provider, proposal-generating commands run the built-in family-QD policy and record durable provider provenance `builtin:family-qd`. Exact-shape and nearest-shape validation-passed winners, including measured discovered hipBLASLt candidates when they remain best, can initialize its operators through `--transfer-shapes` / `--transfer-per-shape`. Random, local/semantic mutation, DE, GOMEA, family archive, linkage, covering, adaptive operator allocation, and contextual bundle acquisition remain supported building blocks. Singleton oversized-pool selection uses bundle acquisition after sufficient exact evidence while preserving mechanical cold start. Trusted custom compositions use `--proposal-script`. See `docs/custom_proposals.md` and `docs/campaign_policy_tuning.md`.

Supported protocol overrides include `--num-benchmarks`, `--num-warmups`, `--enqueues-per-sync`, `--syncs-per-benchmark`, `--num-elements-to-validate`, and `--validation-backend`. The default performs full hipBLASLt GPU-oracle validation with `NumElementsToValidate=-1`. `--validation-backend cpu` selects CPU audit validation. There is no no-validation backend: benchmark-only execution is admitted only after compatible correctness evidence exists.

Validation is a hard gate stored independently from timing. Adaptive top-ups reuse the original compiled and correctness-verified artifacts. They perform no recompilation or repeated verification.

Search-time timing is noisy enough that top-1 screening can miss the final winner. `schedule-batches` uses adaptive sampling by default: it prepares all candidates once, gives each validation-passed pair one probe launch, tops up provisional probe survivors to three launches, runs the main timing protocol only for final probe survivors, then appends missing main-protocol samples for plausible contenders from the prepared-artifact index. Use `--fixed-sampling` only for debugging or fixed-budget utility runs.

Structured scheduler runs ingest their own JSONL results directly into SQLite. The old TensileLite `LibraryClient` CSV/log ingestion path has been removed.

### 3. Repair Weak Shapes

Multi-shape campaigns reserve a staged repair phase for evidence-supported weak shapes. Repair combines capped reference, nearest-shape, cluster, and model-uncertainty deficits with each available candidate's posterior probability of closing useful headroom. Incumbent, neighbor, cluster, broad, and generic mutation seeds enter the same shared-cost bundle acquisition used by the rest of the campaign. Uniform shape weighting remains the default. Explicit call-count * baseline-latency workload weighting is documented in `docs/workload_weighting.md`.

Repair is a search-budget heuristic, not a correctness rule: real performance cliffs from divisibility, edge handling, LDS pressure, or occupancy can legitimately sit below nearby shapes. Every selected repair pair still passes exact validation and timing on its target shape. See `docs/search_outlier_repair.md` and `scripts/simulate_repair_acquisition.py`.

After searching, we can inspect the results.

Summarize cache status:

```bash
python3 -m evotensile.cli summarize-cache \
  --db out/evotensile.sqlite
```

Rank validation-passed observations:

```bash
python3 -m evotensile.cli rank-benchmarks \
  --db out/evotensile.sqlite \
  --min-samples 2
```

### 4. Update hipBLASLt configs

The updater requires the complete profile shape set, positive confirmation timing, latest compatible passed validation, and a content-verified registered artifact for every selected exact pair. Production export passes a serialized deployment assignment with `--selection-json`. This preserves optional solution-bank consolidation instead of re-ranking the DB. The no-selection path remains a DB-rank preview. See `docs/deployment_selection.md`.

```bash
python3 scripts/update_hipblaslt_gridbased_logic.py \
  --db out/evotensile.sqlite
```

Stage the complete variant set outside the hipBLASLt source tree for review:

```bash
python3 scripts/update_hipblaslt_gridbased_logic.py \
  --db out/evotensile.sqlite \
  --selection-json out/deployment-selection.json \
  --output-dir out/gridbased-logic-staged
```

After reviewing the staged YAML, use `--write-source` to overwrite the selected checked-in files. `--allow-partial` is an explicit development-only escape hatch. Normal production export requires exactly the profile shape set and rejects empty, duplicate, missing, or extra mappings.

The updater writes TensileLite-style YAML formatting, retargets solution names, trims generated solution dictionaries to the key schema/order used by existing checked-in GridBased YAMLs, strips benchmark-only embedded `ProblemType`, and applies target-specific build-valid normalizations.

Review the hipBLASLt source diff before rebuilding:

```bash
cd ~/rocm-libraries
git diff --stat -- projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/<gfx-target>/GridBased
```

### 5. Rebuild hipBLASLt

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

### 6. Verify Correctness and Performance

The lightweight target-specific gate uses `hipblaslt-bench --verify` through the EvoTensile verifier and writes `summary.json`, `results.csv`, and per-case logs:

```bash
cd ~/evotensile
python3 scripts/verify_installed_hipblaslt.py \
  --bench ~/rocm-libraries/build/hipblaslt-bench/clients/hipblaslt-bench \
  --tensilelite-libpath "$ROCM_PATH/lib/hipblaslt/library/<gfx-target>"
```

For broader upstream regression coverage, run `hipblaslt-test` with GTest XML output:

```bash
cd ~/rocm-libraries/build/hipblaslt-bench/clients
HIPBLASLT_TENSILE_LIBPATH="$ROCM_PATH/lib/hipblaslt/library/<gfx-target>" \
LD_LIBRARY_PATH="$ROCM_PATH/llvm/lib:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}" \
./hipblaslt-test --gtest_filter='<test-filter>' --gtest_output=xml:/tmp/hipblaslt_test.xml
```

Then run an application-level benchmark, such as:

```bash
TORCH_BLAS_PREFER_HIPBLASLT=1 \
python3 ~/ComfyUI-FeatherOps/benchmark_mm_hipblaslt_fp16.py
```

If the Python runtime uses a separate TensileLite asset package, point `HIPBLASLT_TENSILE_LIBPATH` at the newly installed assets so the rebuilt logic is actually used.

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
- The proposal ExtraTrees surrogate remains per-campaign shortlisting. A shared contextual ExtraTrees pair model now supports normalized performance, validity, and calibrated uncertainty for campaign acquisition, but persistent cross-campaign transfer, LFBO, and trusted TensileLite prediction pruning are not implemented. Keep `PredictionThreshold: 2.0` to disable heuristics like Formocast and Origami in TensileLite until they are accurate enough on gfx1151.
- Logic file update helpers are profile-aware, but each new target variant needs validation before measured-performance claims.
- The production structured backend is intentionally narrower than the generic search abstractions and needs broader target coverage before it is a general GEMM runner.
