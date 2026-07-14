# EvoTensile

Work in progress. README and docs are AI-generated and intended for AI to read.

EvoTensile is a framework for TensileLite kernel tuning using smart search algorithms. It's inspired by [Helion](https://github.com/pytorch/helion), [rocm\_wmma\_gemm](https://github.com/adelj88/rocm_wmma_gemm), [Ductile](https://github.com/ROCm/rocm-libraries/pull/8831), and [GEKO](https://github.com/ROCm/rocm-libraries/pull/8832).

Notable features:
- high-throughput search on whether one shape or thousands of shapes.
- family stratified seeding, GOMEA, learned linkage, learned tree surrogate on one or multi shapes.
- Bayesian joint search on multi shapes, which borrows configs from neighboring shapes.
- early screening of obviously slow configs.
- multi-armed bandit algorithm to rank candidates with noisy measurements.
- compilation cache and multiprocess compilation.
- correctness validation against current hipBLASLt on GPU.
- database for persisted search history.
- simulated timing from known history when evaluating search strategy, without rerunning measurements.

There is not yet a fully automated top-level search loop, but the loop can be driven by AI.

I've tuned the gfx1151 GEMM NT HHS kernel on an 1,135-shape grid. It's possible to support other kernels, notably GroupedGEMM.

## Workflow

1. Define problem type, input shapes, and config search space.
2. Import current hipBLASLt configs.
3. Search configs in iterative rounds, interleaving generic search, measured promotion, and outlier repair. Finish with finalist selection.
4. Update tuned hipBLASLt configs.
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

### 2. Import hipBLASLt Configs

Start each campaign by importing the configs selected by the relevant hipBLASLt installations or preserved logic packages. Use one mutable SQLite database for all evidence-compatible imports and later search rounds. Keep incompatible hardware, toolchain, runtime, or validation environments in separate databases.

An imported hipBLASLt query is seed provenance, not native EvoTensile performance evidence. `discover_hipblaslt_baselines.py` identifies the selected solution for every target shape and records the external query context. Unless `--query-only` is used, it then schedules those exact candidate-shape pairs through the normal compilation, validation, timing, cost, and artifact pipeline.

Build rocisa, the TensileLite client, and the EvoTensile structured runner before importing:

```bash
cd ~/rocm-libraries/projects/hipblaslt/tensilelite/rocisa
CXX=$ROCM_PATH/llvm/bin/amdclang++ pip install -U --no-deps -e .
cd ~/rocm-libraries
./build_tensilelite_client.sh
cd ~/evotensile
scripts/build_structured_runner.sh
```

The runner build script validates that the expected TensileLite client static libraries exist under `~/rocm-libraries/build/tensilelite-client` before compiling `./build/evotensile-structured-runner`.

Import and natively measure the current installed selections:

```bash
python3 scripts/discover_hipblaslt_baselines.py \
  --db out/evotensile.sqlite \
  --profile <profile-name> \
  --output-dir out/hipblaslt-current \
  --baseline-label current \
  --tensilelite-libpath "$ROCM_PATH/lib/hipblaslt/library/<gfx-target>"
```

Repeat the import for any evidence-compatible comparison package that should remain a mandatory control, such as preserved untuned logic. Isolate its runtime assets, give it a distinct `--baseline-label`, and provide the matching `--logic-yaml`. Do not pool query timings or measurements from incompatible environments.

After all seed imports, preserve an immutable database snapshot for final baseline comparisons and keep searching in the mutable campaign database:

```bash
sqlite3 out/evotensile.sqlite ".backup 'out/evotensile-seed.sqlite'"
```

### 3. Search Configs

The 1,135-shape campaign used sparse evidence-driven rounds rather than one dense restart or a separate repair-only tail. Generic interaction search, measured promotion, integrated outlier repair, stabilization, and checkpoint refreshes can be interleaved according to the evidence produced by the previous round.

The generalized practical-round and finalization scripts retain historical `grid100` filenames, but support every registered profile, including `gfx1151-nt-hhs-comfy1135`. They default to `out/evotensile.sqlite`, derive the seed database and campaign directories from that path, derive the incumbent report beside its deployment file, and use four contenders within 5% for finalization.

First establish a fresh initial checkpoint from the imported corpus. Checkpoints anchor later comparisons. They are selection state, not attribution evidence for a search operator:

```bash
python3 scripts/finalize_grid100_production_search.py \
  --profile <profile-name> \
  --output-dir out/evotensile/checkpoint-initial \
  --maximum-contenders 2 \
  --samples 10
```

Run sparse interaction rounds over evidence-backed parameter families such as store, staging, mapping, vector, and LDS. Integrated repair can spend part of the same round on shapes with fresh checkpoint deficits, model uncertainty, weak gains, or nearby measured candidates:

```bash
python3 scripts/run_grid100_practical_round.py \
  --profile <profile-name> \
  --incumbent-deployment out/evotensile/checkpoint-initial/deployment_0.000.json \
  --round-id round08-vector-repair \
  --interaction-profile vector \
  --integrated-repair-targets 24 \
  --integrated-repair-weight 1.0 \
  --seed <change seed each time>
```

Promote only children with exact measured gains, using the measured comparison parent from the originating round:

```bash
python3 scripts/run_grid100_practical_round.py \
  --profile <profile-name> \
  --incumbent-deployment out/evotensile/checkpoint-initial/deployment_0.000.json \
  --round-id round09-promotion \
  --strategy promotion \
  --promote <candidate-hash>:<parent-hash> \
  --seed <change seed each time>
```

A practical campaign loop is:
- probe a broad or evidence-backed interaction family on a bounded shape set.
- promote measured winners across parent-competitive shapes and mechanical neighbors.
- interleave integrated repair when checkpoint deficits or uncertainty expose local headroom. Repair is not a correctness rule and every pair still requires exact validation and timing.
- refresh the checkpoint after material generalists, stale assignments, or several rounds of accumulated gains. Include explicit baseline and incumbent controls whenever same-session comparison is required.
- stabilize close or noisy contenders before attributing gains, and preserve failed validations and rejected proposals as evidence.
- reopen a search family only for transferable gains. Isolated noise or a dominated same-shape competitor does not justify a broad restart.
- declare convergence only after promotion is exhausted and diverse closure probes produce no material transferable multi-shape gain.

`schedule-batches` remains the lower-level entry point for explicit candidate-shape work. It plans missing pairs against SQLite, emits TensileLite YAML, runs build/codegen, maps final solutions, and ingests validation-gated timing rows. It uses adaptive probe, screening, main-protocol, and top-up sampling by default:

```bash
python3 -m evotensile.cli schedule-batches \
  --db out/evotensile.sqlite \
  --profile <profile-name> \
  --output-dir out/search \
  --dry-run
```

The external runner consumes TensileLite artifacts from either full-client `4_LibraryClient/library/gfx*` output or build-only `1_BenchmarkProblems/**/source/library/gfx*` cache output. Compile-cache builds are candidate-centric and reusable across proposal cohorts. Preparation performs build, mapping, diagnostics, and validation in parallel. Timing starts only after preparation drains and always runs serially. Validation is stored independently and remains a hard gate.

With no custom provider, proposal-generating commands use the built-in family-QD policy. Random, local and semantic mutation, DE, GOMEA, family archives, learned linkage, mechanical covering, adaptive operator allocation, contextual bundle acquisition, and measured transfer remain available building blocks. See `docs/custom_proposals.md`, `docs/campaign_policy_tuning.md`, and `docs/search_outlier_repair.md`.

After convergence, run authoritative fresh finalization. It remeasures the bounded contender set for every shape, requires the original compatible control and current incumbent, and emits zero-tolerance plus explicit loss-bounded deployment selections:

```bash
python3 scripts/finalize_grid100_production_search.py \
  --profile <profile-name> \
  --incumbent-deployment out/evotensile/checkpoint-latest/deployment_0.000.json
```

Production updates must use a deployment file from fresh finalization, normally `deployment_0.000.json` for maximum measured speed or an explicitly selected loss-bounded alternative. Historical pooled rankings and intermediate checkpoints remain diagnostic only. See `docs/experiment_1135_shape.md` for the concrete round, repair, checkpoint, convergence, and finalization sequence from the 1,135-shape campaign.

Summarize cache status and inspect validation-passed observations at any point:

```bash
python3 -m evotensile.cli summarize-cache --db out/evotensile.sqlite
python3 -m evotensile.cli rank-benchmarks --db out/evotensile.sqlite --min-samples 2
```

### 4. Update hipBLASLt Configs

The updater requires the complete profile shape set, positive confirmation timing, latest compatible passed validation, and a content-verified registered artifact for every selected exact pair. Production export passes a serialized deployment assignment with `--selection-json`. This preserves optional solution-bank consolidation instead of re-ranking the DB. The no-selection path remains a DB-rank preview. See `docs/deployment_selection.md`.

```bash
python3 scripts/update_hipblaslt_gridbased_logic.py \
  --db out/evotensile.sqlite
```

Stage the complete variant set outside the hipBLASLt source tree for review:

```bash
python3 scripts/update_hipblaslt_gridbased_logic.py \
  --db out/evotensile.sqlite \
  --selection-json out/evotensile/finalization/deployment_0.000.json \
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
python3 scripts/verify_installed_hipblaslt.py
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
