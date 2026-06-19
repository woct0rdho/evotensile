# EvoTensile

Work in progress. README is AI-generated.

EvoTensile is an external smart-search autotuner for TensileLite / hipBLASLt. It proposes complete TensileLite candidate bundles, emits them as TensileLite `Groups`, runs TensileLite as the evaluator, and stores reproducible timing/cache metadata for iterative search. It is inspired by [Helion](https://github.com/pytorch/helion) and [rocm_wmma_gemm](https://github.com/adelj88/rocm_wmma_gemm).

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
7. Rebuild and validate the installed hipBLASLt behavior with an application-level benchmark.

## Repository Pieces

- `evotensile/candidate.py`: canonical candidate representation and stable hashes.
- `evotensile/shapes.py`: exact-shape helpers and bundled pilot-grid shape generation.
- `evotensile/search_space.py`: target search-space domains, constraints, and seeded/random candidates.
- `evotensile/search/`: random, local, differential-evolution, and GOMEA-style proposal operators.
- `evotensile/yaml_writer.py`: TensileLite config emission using complete candidate `Groups`.
- `evotensile/runner.py`: TensileLite subprocess orchestration, including build/benchmark separation.
- `evotensile/parser.py` and `evotensile/ingest.py`: validation-aware CSV/log/final-YAML ingestion.
- `evotensile/database.py` and `evotensile/cache.py`: SQLite storage and cache queries.
- `evotensile/scheduler.py`: cache-aware batch planning and execution.
- `scripts/retime_topk.py`: exact top-K finalist retiming.
- `scripts/compare_hipblaslt_bench.py`: installed hipBLASLt comparison and hybrid export.
- `scripts/update_hipblaslt_gridbased_logic.py`: update checked-in hipBLASLt GridBased YAMLs from a hybrid export.

## 1. Define Problem, Shapes, And Search Space

A target needs:

- a TensileLite problem type, including data types, layout, batching, epilogue flags, and validation settings;
- exact input shapes, usually represented as `shape_id = m{M}_n{N}_b{batch}_k{K}`;
- a candidate search space made of complete TensileLite solution dictionaries, not independent Cartesian products;
- cache identity names: `version_name`, `problem_type_hash`, and `benchmark_protocol_hash`.

The current code has a bundled gfx1151 FP16 NT HHS target, but new targets should keep target-specific choices in config/data code and leave the generic runner/cache/search flow unchanged.

Inspect a target search space with:

```bash
python3 -m evotensile.cli summarize-space --num-random 128
```

## 2. Search Configs

`schedule-batches` is the main entry point for searching. It plans missing `(shape, candidate)` work against the SQLite cache, emits TensileLite YAML batches, optionally separates compile and benchmark phases, and ingests validation-gated results.

Dry-run a plan:

```bash
python3 -m evotensile.cli schedule-batches \
  --db out/evotensile.sqlite \
  --output-dir out/search \
  --version-name my_target_hotloop_v0 \
  --limit-shapes 100 \
  --candidate-batch-size 32 \
  --shape-batch-size 100 \
  --dry-run
```

Run planned batches:

```bash
python3 -m evotensile.cli schedule-batches \
  --db out/evotensile.sqlite \
  --output-dir out/search \
  --version-name my_target_hotloop_v0 \
  --proposal seed-random-gomea \
  --num-random 64 \
  --gomea-count 64 \
  --transfer-shapes 4 \
  --transfer-per-shape 2 \
  --candidate-batch-size 32 \
  --shape-batch-size 100 \
  --compile-threads 4 \
  --benchmark-threads 1 \
  --keep-going
```

Useful proposal modes include `seed-random`, `local`, `seed-random-local`, `de`, `seed-random-de`, `gomea`, `seed-random-gomea`, and `evolutionary`.

Validation is a hard gate: only `status=ok` rows with passing validation should be ranked or used as positive cache entries. Unknown validation should be treated as debug-only unless a target explicitly opts into it.

## 3. Inspect And Rank Cached Results

Summarize cache status:

```bash
python3 -m evotensile.cli cache-summary \
  --db out/evotensile.sqlite \
  --version-name my_target_hotloop_v0
```

Rank validation-passed observations:

```bash
python3 -m evotensile.cli rank-evals \
  --db out/evotensile.sqlite \
  --version-name my_target_hotloop_v0 \
  --min-samples 2
```

Manual ingestion is available when a TensileLite run was produced outside `schedule-batches`:

```bash
python3 -m evotensile.cli ingest-csv out/tensilelite_run_000 \
  --db out/evotensile.sqlite \
  --manifest out/search/config.manifest.csv \
  --version-name my_target_hotloop_v0 \
  --include-logs
```

When given a run directory, ingestion uses TensileLite final YAML as the source of truth for accepted/rejected/deduplicated solution mapping.

## 4. Retime Top-K Finalists

Search-time timing is noisy enough that top-1 screening can miss the final winner. Use `scripts/retime_topk.py` to retime the top candidates per shape, normally with stronger validation such as `NumElementsToValidate=-1`.

```bash
python3 scripts/retime_topk.py \
  --db out/evotensile.sqlite \
  --output-dir out/topk_retime \
  --source-version-name my_target_hotloop_v0 \
  --target-version-name my_target_hotloop_v0_top4_fullval \
  --top-k 4 \
  --global-parameter NumElementsToValidate=-1 \
  --compile-threads 4 \
  --benchmark-threads 1 \
  --keep-going
```

After retiming, export the per-shape winners using the project export script for the current target/artifact layout.

## 5. Compare With Current hipBLASLt And Export Hybrid Winners

Use `scripts/compare_hipblaslt_bench.py` when `hipblaslt-bench` can express the target operation. The comparison reports installed/current hipBLASLt performance and can export a hybrid winner set that keeps the tuned candidate only when it beats the current installed solution.

```bash
python3 scripts/compare_hipblaslt_bench.py \
  --winners-csv out/topk_retime_export/winners.csv \
  --output-dir out/hipblaslt_bench_compare \
  --bench ~/rocm-libraries/build/hipblaslt-bench/clients/hipblaslt-bench \
  --tensile-libpath ~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_libraries/lib/hipblaslt/library/gfx1151 \
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

## 7. Rebuild And Validate hipBLASLt

Rebuild hipBLASLt from the modified `~/rocm-libraries` tree and install into the intended ROCm SDK prefix. The current local helper is:

```bash
cd ~/rocm-libraries
BUILD_DIR="$PWD/build/hipblaslt-gfx1151-grid100-hybrid" \
GPU_TARGETS=gfx1151 \
./build_hipblaslt.sh
```

Then run an application-level benchmark with the intended runtime environment. If the Python runtime uses a separate TensileLite asset package, point `HIPBLASLT_TENSILE_LIBPATH` at the newly installed assets so the rebuilt logic is actually used.

## Benchmark Protocol

The default generated YAML uses hot-loop / steady-state timing:

```yaml
NumWarmups: 10
NumBenchmarks: 10
EnqueuesPerSync: 10
SyncsPerBenchmark: 1
SleepPercent: 0
HardwareMonitor: False
```

Cold-loop behavior is intentionally not tracked during tuning because it increases tuning time and optimizes for first-run or bursty-idle effects rather than sustained throughput. Analyze cold-loop behavior later only if first-request latency becomes important.

Benchmark-affecting global parameters are included in the benchmark-protocol hash. Compile-only settings such as `CpuThreads` are phase-specific and intentionally excluded from benchmark protocol identity.

## Current Limitations

- The bundled problem type and search-space domains target gfx1151 FP16 NT HHS first.
- Surrogate/LFBO proposal is planned but not implemented.
- Timeout classification and multi-candidate build-failure attribution are still incomplete.
- BBS/AuxH/AuxB retargeting needs target-specific validation before making measured-performance claims for those variants.
- A purpose-specific benchmark runner is still needed before scaling to much larger grids; the generic TensileLite client path is dominated by orchestration/log/validation overhead.
- Keep `PredictionThreshold: 2.0` to disable heuristics like Formocast and Origami in TensileLite, until they're accurate enough on gfx1151.
