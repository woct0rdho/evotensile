# EvoTensile Plan

EvoTensile is an external smart-search autotuner for TensileLite / hipBLASLt. The current completed pilot target is gfx1151 FP16 NT HHS GridBased GEMM tuning for the 100-shape grid described in `~/ComfyUI-FeatherOps/doc/tensile_fp16_nt_hhs_grid.md`, followed by checked-in hipBLASLt HHS/HHS+AuxH/BBS/BBS+AuxB GridBased YAML updates, rebuild, PyTorch-level performance validation, and repeatable installed-library correctness validation. The current workflow uses imported current-hipBLASLt baseline candidates, a structured exact-pair backend, and adaptive sampling before scaling to finer/larger shape grids.

## 1. Motivation

Hand-picking a small candidate set is fragile because:
- performance far from `8192^3` is hard to predict;
- TN/NN winners may or may not transfer to NT;
- TensileLite knobs have strong interactions / epistasis;
- local optima are likely;
- Origami ranking is not reliable enough for hard pruning on this target;
- TensileLite itself enumerates Cartesian products unless candidates are bundled through `Groups`.

EvoTensile will treat TensileLite configurations as a constrained, noisy, mixed-categorical black-box optimization problem and search it with random, evolutionary, local, and later surrogate-assisted methods.

## 2. Initial Scope

### Target problem type

Initial target:
- GPU: `gfx1151`
- Operation: GEMM
- Layout: NT (`TransposeA=False`, `TransposeB=True`)
- Data: FP16 input/output, FP32 accumulation (`HHS`)
- Batched: true, batch initially `1`
- hipBLASLt epilogue-capable UserArgs problem type:
  - `UseBias=1`
  - `BiasSrc=D`
  - `UseScaleAlphaVec=1`
  - `Activation=True`
  - `ActivationType=hipblaslt_all`
  - `UseE=False` for first pass

### Pilot shape grid

Start with 100 exact shapes:

```text
M:     [512, 640, 896, 1024]
N:     [128, 256, 512, 768, 1024]
batch: [1]
K:     [256, 512, 1024, 2048, 4096]
```

The pilot may benchmark more than 10 configs per shape. Initial budget target: roughly 64-256 evaluated candidate configs per shape, subject to wall-time results from the harness.

## 3. Non-goals for the First Version

- Do not modify TensileLite internals initially.
- Do not rely on Origami / Formocast as a hard pruning gate.
- Do not attempt to solve all data types/layouts immediately.
- Do not scale to production-size grids before the benchmark/evaluation and GridBased update loop is stable.
- Do not assume one unique candidate set per shape; use buckets/batches to control compilation and process overhead.

## 4. Design Principles

- External orchestration first. EvoTensile generates candidate batches, emits TensileLite YAML with `Groups`, runs TensileLite, parses outputs, and records results.
- Canonical candidate hashes. Every candidate config gets a stable canonical representation and hash. This enables deduplication, caching, and reproducibility.
- Persistent result database. Every attempted `(shape, candidate, environment)` should be stored, including invalid builds, correctness failures, timeouts, and benchmark results.
- Batch evaluation. Avoid per-candidate/per-shape process launches. Emit batches of candidates into `Groups` and evaluate shape buckets in one TensileLite run when possible.
- Search data is valuable. The pilot should produce a reusable dataset for later nearest-shape seeding and surrogate training.
- Hot-loop throughput is the tuning objective. Search-time and final timings should both use a hot-loop protocol that represents steady-state long-running inference/training throughput. Cold-loop behavior is not tracked during tuning because it increases wall time; analyze it later only if first-request or bursty-idle latency becomes important.

## 5. Current Implementation Status

Generic implemented capabilities are summarized in `README.md`. Target-specific status:
- the gfx1151 FP16 NT HHS problem type and 100-shape pilot grid are encoded;
- the search space can express the documented `8192^3` SIA3/no-store-priority winner family;
- the completed first-pass scan, repaired ingestion, top-4 full-validation retime, current-hipBLASLt baseline import, DB-driven GridBased YAML update, hipBLASLt rebuild/install, PyTorch benchmark validation, `hipblaslt-bench --verify` correctness test, and upstream `hipblaslt-test` smoke/quick/pre_checkin validation are done for the 100-shape pilot;
- generated YAML uses complete candidate `Groups` rather than independent Cartesian products;
- runner support exists for direct runs, compile-then-serial-benchmark runs, and an exact-pair structured backend path;
- DB schema uses each SQLite file as the evidence namespace and keys cache reuse by `problem_type_hash`, `benchmark_protocol_hash`, exact shape, and candidate hash;
- structured JSONL result ingestion exists, using TensileLite final solution YAML as the source of truth for candidate mapping;
- cache-aware batch scheduling exists for missing `ok` observations, including adaptive top-ups that reuse prior validation evidence and run GPU-only timing when safe;
- current hipBLASLt-selected configs can be imported once per DB/problem/grid as baseline candidates and then compete in normal EvoTensile selection;
- `scripts/update_hipblaslt_gridbased_logic.py` now emits build-valid checked-in YAMLs for HHS/HHS+AuxH/BBS/BBS+AuxB variants, including Aux `UseE` handling and scalar type normalization;
- a real one-shape harness under `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/evotensile_one_shape/` showed that hindsight-directed local refinement can reproduce the documented `8192^3` winner, but this operator is not part of the generic scheduler because it bakes in the known winner neighborhood.

## 6. Candidate Model

A candidate is a complete TensileLite solution parameter bundle, not a single independent knob. EvoTensile will emit each candidate as one `Groups` entry.

Initial candidate fields:

```python
Candidate = {
    # required / fixed-ish
    "KernelLanguage": "Assembly",
    "WavefrontSize": 32,

    # tile / MI family
    "MatrixInstruction": [...],
    "WorkGroup": [...],
    "DepthU": 16 | 32 | 64,
    "GlobalSplitU": 1 | 2 | 4,
    "GlobalSplitUAlgorithm": "MultipleBuffer",

    # pipeline / schedule
    "PrefetchGlobalRead": 0 | 1 | 2,
    "PrefetchLocalRead": 0 | 1,
    "ScheduleGlobalRead": 1,
    "ScheduleLocalWrite": 1,
    "ScheduleIterAlg": 1 | 2 | 3,

    # spatial ordering / cache behavior
    "WorkGroupMapping": 4 | 5 | 8 | 16,
    "StaggerU": 0 | 8 | 16 | 32 | 64,
    "StaggerUStride": 256,
    "StaggerUMapping": 0 | 1,

    # LDS / source layout
    "SourceSwap": 0 | 1,
    "1LDSBuffer": 0 | 1,
    "ClusterLocalRead": 0 | 1,
    "TransposeLDS": 0 | 2,

    # vectorization
    "VectorWidthA": 1,
    "VectorWidthB": 1 | 2,
    "GlobalReadVectorWidthA": 1 | 2 | 4 | 8,
    "GlobalReadVectorWidthB": 1 | 2 | 4 | 8,
    "LocalReadVectorWidth": 16,

    # stores / assertions
    "StoreVectorWidth": -1 | 1,
    "StoreRemapVectorWidth": 0,
    "StorePriorityOpt": True | False,
    "NumElementsPerBatchStore": 0 | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 20 | 24 | 32,
    "StoreSyncOpt": 0 | 1 | 2 | 4,
    "GroupLoadStore": False | True,
    "LdsBlockSizePerPadA/B": 0 | 128 | 256 | 512 | 1024 | 2048,
    "LdsPadA/B": 0 | 4 | 8 | 16,
    "AssertFree0ElementMultiple": 8,
    "AssertFree1ElementMultiple": 8,
    "AssertSummationElementMultiple": 16,
}
```

The search space should support conditional constraints and linked mutations. Examples:
- mutate `MatrixInstruction`, `WorkGroup`, and macro-tile-related choices together;
- mutate `DepthU` with vector-width / prefetch choices;
- mutate `WorkGroupMapping`, `StaggerU`, and `StaggerUMapping` together;
- keep LDS padding as linked artifact-backed profiles instead of independent random padding choices;
- keep known-invalid or repeatedly failing combinations out of later batches.

## 7. TensileLite Evaluation Strategy

### YAML generation

EvoTensile generates TensileLite YAML where candidate configs are emitted as `Groups`, for example:

```yaml
ForkParameters:
  - Groups:
    - - MatrixInstruction: [16, 16, 16, 1, 1, 4, 4, 2, 2]
        WorkGroup: [16, 16, 1]
        DepthU: 64
        GlobalSplitU: 1
        ...
      - MatrixInstruction: [16, 16, 16, 1, 1, 4, 2, 2, 2]
        WorkGroup: [16, 16, 1]
        DepthU: 32
        GlobalSplitU: 1
        ...
```

Avoid normal multi-valued `ForkParameters` except when a Cartesian product is intentionally desired.

### Batching

Two useful batch modes:
- Candidate-batch x shape-bucket: Evaluate a batch of candidates over a bucket of shapes. Good for broad scans.
- Shape-local candidate batch: Evaluate a candidate batch specific to a single shape or small shape cluster. Useful for refinement.

The scheduler should choose batch sizes to balance code-object build overhead, process overhead, and wasted cross-evaluation.

### Caching

Use TensileLite `--build-only` and `--use-cache` where helpful, but do not rely only on TensileLite's build cache. EvoTensile treats one SQLite DB file as one evidence namespace for a target hardware/environment/campaign. Within that DB, cache identity is based on:
- problem type hash;
- benchmark protocol hash;
- exact shape;
- candidate hash.

Do not automatically key invalidation on a `rocm-libraries` commit hash. Store commit/source metadata for audit, but compare incompatible environments or campaigns through separate DB files.

## 8. Result Database

Use SQLite initially. DuckDB can be added later for analytics.

Core tables:

### `candidates`

- `candidate_hash TEXT PRIMARY KEY`
- `candidate_json TEXT`
- `created_at TEXT`
- `source TEXT` - random, seed, mutation, crossover, imported, etc.
- `parent_hashes TEXT`

### `shapes`

- `shape_id TEXT PRIMARY KEY`
- `m INTEGER`
- `n INTEGER`
- `batch INTEGER`
- `k INTEGER`
- derived features: `log_m`, `log_n`, `log_k`, aspect ratios, etc.

### `runs`

- `run_id TEXT PRIMARY KEY`
- `timestamp REAL`
- `problem_type_hash TEXT`
- `benchmark_protocol_hash TEXT`
- `yaml_path TEXT`
- `output_dir TEXT`
- `tensilelite_bin TEXT`
- `status TEXT`
- `returncode INTEGER`
- `stdout_path TEXT`
- `stderr_path TEXT`
- `metadata_json TEXT`

### `evaluations`

- `eval_id INTEGER PRIMARY KEY`
- `problem_type_hash TEXT`
- `benchmark_protocol_hash TEXT`
- `shape_id TEXT`
- `candidate_hash TEXT`
- `run_id TEXT`
- `status TEXT` - ok, invalid, compile_fail, validation_fail, timeout, parse_fail
- `time_us REAL`
- `validation TEXT`
- `solution_index INTEGER`
- `created_at REAL`

Index `(problem_type_hash, benchmark_protocol_hash, shape_id, candidate_hash)` heavily for cache lookups.

## 9. Search Algorithms

### Phase A: baseline generators

Implemented now:
- deterministic conservative seeds, including large-square, TLDS2/LDS-pad, and small/skinny checked-in-style NT seed families;
- exact-shape and nearest-shape winner transfer from validation-passed cached observations, defaulting to `4` nearby shapes and `2` top candidates per shape;
- random valid generator;
- local mutation around cached DB elites;
- scheduler proposal modes for seed/random, local-only, seed/random plus local refinement, categorical DE, GOMEA, and combined evolutionary batches;
- a ground-truth documented-winner helper for checks, not as a default random-init seed.

Still planned:
- stratified generator over macro-tile / depth / GSU / schedule families;
- shape-aware MI/GSU generator inspired by TensileLite's beta `tensile_config_generator.py`;
- configurable seed packs for known winners and transferred TN/NN candidates.

### Phase B: local/evolutionary search

Implemented now:
- categorical DE-style mutation/crossover over encoded TensileLite domain values;
- GOMEA-style linkage neighborhoods and linkage-tree mixing inspired by `~/rocm_wmma_gemm/rocm_wmma_gemm/config/tune.py`;
- generic seed/random plus GOMEA reproduction of the documented `8192^3` winner within the first 32 proposals, without inserting the documented winner or using the hindsight-directed operator;
- `schedule-batches` now defaults to the recommended 100-shape first-pass settings: nearest-shape transfer, `--proposal seed-random-gomea`, `--num-random 64`, and `--gomea-count 64`.

Still needed:
- richer shape-aware candidate proposal beyond nearest-shape winner transfer and optional `--proposal-shape-id` filtering;
- richer crossover between near-winners;
- generic refinement operators that do not bake in known-winner hindsight;
- richer failure-aware candidate filtering beyond the current reusable negative-cache statuses.

For the pilot, a simple version is enough:

```text
per shape:
  evaluate seed + random/stratified batch
  keep top K elites
  repeat R rounds:
    generate mutations/crossovers from elites
    add random restarts
    evaluate batch
    update elites
```

### Phase C: surrogate assistance

After enough observations exist:
- train random-forest / extra-trees models;
- use LFBO-style classification: predict probability that a candidate is top quantile for a shape;
- rank a large generated candidate pool and evaluate only the best/diverse subset;
- include diversity penalty to avoid evaluating near-duplicates.

Classic Gaussian-process Bayesian optimization is not the first choice because the space is high-dimensional, discrete/categorical, constrained, and noisy.

### Previous 8192^3 Reproduction Run Context

Source artifacts:
- `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/evotensile_one_shape/run_one_shape_random_repro.py`
- `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/evotensile_one_shape/runs/random12_local8_control/summary.json`
- `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/evotensile_one_shape/runs/random12_local8_directed_repro/summary.json`
- `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/evotensile_one_shape/runs/random12_local8_directed_repro/hot_loop_summary.json`

Prior plain-random/local baseline:
- shape: `m8192_n8192_b1_k8192`;
- command from Pi session line 519: `run_one_shape_random_repro.py --run-name random12_local8_control --seed 1151 --num-random 12 --num-local 8 --num-benchmarks 2 --num-warmups 1 --include-documented-winner`;
- approximate wall time from Pi command/result timestamps: `2026-06-18T08:10:00.340Z` to `2026-06-18T08:12:10.832Z`, about `130.5 s`;
- initial search budget: `12` random candidates plus `8` naive local mutations;
- one documented-winner control was benchmarked separately, so `summary.json` has `num_total=21` and `documented_winner_in_random_or_local=false`;
- plain random/local did not generate `cand_4bde2d3af447f757`;
- best non-control generated candidate was `cand_93149cee63b2ead1_MT128x128x16_SIA3_PGR2_PLR0_VWB2_NEPBS16_SPO1`, row `19` in `results.csv`;
- hot-loop retime for that best non-control was `34976.1 GFLOP/s` median;
- documented winner hot-loop retime was `46698.1 GFLOP/s` median, so the plain random/local best was `25.1%` slower (`74.9%` of winner throughput, winner `1.335x` faster).

Prior hindsight-directed-refinement run, retained only as context:
- command from Pi session line 536: `run_one_shape_random_repro.py --run-name random12_local8_directed_repro --seed 1151 --num-random 12 --num-local 8 --num-benchmarks 2 --num-warmups 1 --directed-refine`;
- approximate wall time from Pi command/result timestamps: `2026-06-18T08:20:22.541Z` to `2026-06-18T08:24:55.875Z`, about `273.3 s`;
- same `12` random plus `8` naive local starting point, followed by compact directed refinement from random/local elites;
- `summary.json` has `num_total=38`, `num_ok=27`, and `documented_winner_in_random_or_local=true`;
- directed refinement generated the documented winner `cand_4bde2d3af447f757_MT128x128x16_SIA3_PGR1_PLR1_VWB2_NEPBS10_SPO0` without inserting it as a control;
- because that operator stages candidates into the already-known 8192^3 winner neighborhood, it is treated as a non-generic hindsight baseline and should not be used for normal unknown-shape search;
- the documented winner was row `34` in `results.csv`, so reproduction happened after `34` benchmarked candidates in that run;
- cool-loop screening ranked the `NEPBS16` sibling first and the documented `NEPBS10` winner third;
- hot-loop retime reversed that order: documented `NEPBS10` median `46698.1 GFLOP/s`, sibling `NEPBS16` median `45205.2 GFLOP/s`.

Current non-hindsight evolutionary reproduction check:
- `propose_candidates(... proposal="seed-random-gomea", num_random=12, gomea_count=64, seed=1151)` generates the documented winner at candidate position `32` from an empty DB;
- when seeded with the prior plain random/local observations excluding the documented control, `seed-random-gomea` still generates the winner at proposal position `32`, requiring `20` new uncached candidates before the winner if the previous `20` non-control evaluations are treated as cached;
- pure categorical DE did not generate the exact winner in the checked budgets, so GOMEA-style linked schedule/store neighborhood expansion is currently the useful generic evolutionary operator for this reproduction case;
- this is generic in the sense that it sweeps linked categorical groups around seeds/parents and uses the documented winner only as an external success predicate.

## 10. Shape Transfer Strategy

For finer grids:
- Represent each shape in log/ratio feature space.
- Find nearest tuned shapes.
- Seed new shape with:
  - exact-shape and nearest winners, implemented in `schedule-batches` through `--transfer-shapes` and `--transfer-per-shape`;
  - nearest near-winners;
  - mutations around nearest winners;
  - a few global robust candidates;
  - a few random exploratory candidates.
- Run a smaller budget, e.g. 16-64 configs/shape, then let adaptive sampling top up uncertain finalists.

Shape features:

```text
log2(M), log2(N), log2(K), log2(batch)
log2(M/N), log2(K/M), log2(K/N)
ceil(M/MT0), ceil(N/MT1), tile count, edge fraction
```

## 11. Benchmark Protocol

### Default hot-loop protocol

Goal: steady-state throughput for long-running inference/training, plus correctness screening.

Default settings:

```yaml
PredictionThreshold: 2.0   # do not use Formocast prediction on gfx1151
NumWarmups: 10
NumBenchmarks: 10
EnqueuesPerSync: 10
SyncsPerBenchmark: 1
SleepPercent: 0
HardwareMonitor: False
NumElementsToValidate: -1
SkipSlowSolutionRatio: 0.0 initially; optional after validation
```

Cold-loop behavior is intentionally not part of the tuning loop. Tracking it would add measurement cost and optimize for first-run / bursty-idle effects rather than sustained throughput. If needed, add a separate later analysis pass for first-request latency, module-load/JIT effects, allocator warmup, and idle-to-active behavior.

### Final confirmation protocol

For top candidates per shape:
- adaptively top up statistically plausible contenders with the same hot-loop timing protocol;
- use repeated samples and report robust medians / timing summaries;
- validate once per `(shape, candidate)` pair, then allow GPU-only timing top-ups only with prior validation evidence;
- compare against existing hipBLASLt logic and known baseline configs.

Final results should separate:
- search-time best;
- confirmed best;
- correctness failures;
- unstable/noisy candidates.

## 12. Milestones

### M0-M2: repository skeleton, primitives, YAML writer - done

- Package structure, plan, README, and config scaffold exist.
- Candidate dataclass / canonical JSON / hash exist.
- Pilot grid generator exists.
- Search-space domains, cheap constraints, random generation, and deterministic seeds exist.
- TensileLite YAML emission with `Groups` exists.
- Default hot-loop protocol is encoded.

### M3: runner, structured ingestion, and cache identity - mostly done

Done:
- invoke TensileLite in a subprocess;
- capture logs and output paths;
- initialize SQLite schema;
- record run metadata with DB-file evidence namespaces;
- query cache identity/status/missing evaluations and rank only validation-passed observations;
- schedule missing `ok` observations into candidate/shape batches and ingest each completed batch;
- use the structured exact-pair JSONL backend instead of CSV/log parsing;
- reuse rejected final-YAML candidates, validation failures, and single-candidate build failures as negative-cache entries that scheduler skips;
- adaptively top up unresolved contenders and skip repeated CPU validation when prior validation evidence exists.

Remaining:
- classify multi-candidate build failures robustly beyond validation pass/fail rows.

### M4: first pilot scan - done

- Evaluated `135` proposed candidates over the 100-shape pilot grid.
- Repaired final-YAML mapping and produced a validation-gated database with `75,000 ok`, `2,000 validation_fail`, and `5,800 rejected` rows.
- Historical winner/near-winner analysis reports were reviewed and later removed from `out/` after the DB-driven workflow replaced artifact-based handoffs.
- Verified that generic TensileLite client orchestration dominates wall time, motivating the custom benchmark runner before larger grids.

### M5: local/evolutionary refinement - done for the pilot, open for larger grids

- `seed-random-gomea` plus nearest-shape transfer is the current first-pass proposal policy.
- GOMEA-style linked neighborhoods reproduced the documented `8192^3` winner without inserting it directly.
- Future work: richer shape-aware parent selection, elite crossover, failure-aware mutations, and non-hindsight refinement rounds for larger grids.

### M6: final confirmation + GridBased update

Done for the 100-shape pilot:
- Repaired final-YAML mapping and re-ingested the full scan into `out/grid100_full_20260618_repaired.sqlite`.
- Retimed top-4 per shape with full validation under `gfx1151_fp16_nt_hhs_grid100_20260618_repaired`.
- Historically exported selected candidate bundles after top-4 retime; that intermediate artifact path is now retired.
- Historically used a post-hoc installed-hipBLASLt replacement export; that path is now retired because current hipBLASLt-selected configs are imported up front and compete under EvoTensile's benchmark protocol.
- Imported current hipBLASLt-selected configs into `out/grid100_full_20260618_repaired.sqlite`; GridBased YAML updates now query protocol-authoritative winners directly from the DB.
- Added `scripts/update_hipblaslt_gridbased_logic.py` and used it to directly overwrite the tracked gfx1151 GridBased HHS/HHS+AuxH/BBS/BBS+AuxB YAMLs in `~/rocm-libraries`.
- The updater emits TensileLite-style YAML formatting, trims per-solution dictionaries to the key schema/order used by existing large GridBased YAMLs, strips benchmark-only embedded `ProblemType`, preserves local EvoTensile run evidence unchanged, forces `GroupLoadStore=False` for Aux `UseE` variants, and normalizes scalar types that TensileLite/msgpack expect as bool/float.
- Rebuilt hipBLASLt from `~/rocm-libraries` and installed into `$ROCM_PATH` with gfx1151 TensileLite assets under `$ROCM_PATH/lib/hipblaslt/library/gfx1151`.
- Validated the rebuilt install with `~/ComfyUI-FeatherOps/benchmark_mm_hipblaslt_fp16.py` using `TORCH_BLAS_PREFER_HIPBLASLT=1` and `HIPBLASLT_TENSILE_LIBPATH=$ROCM_PATH/lib/hipblaslt/library/gfx1151`.

### M7: transfer to finer grids

- Nearest-shape seeding is implemented for cached validation-passed winners.
- Add smaller-budget local refinement for new shapes.
- Add surrogate-assisted proposal once enough data exists.

## 13. Pilot Review Notes

Post-100-shape status and remaining risks:
- The `~/ComfyUI-FeatherOps/doc/tensile_fp16_nt_hhs_grid.md` plan is still applicable: start with the 100-shape NT HHS non-AuxH grid, use hot-loop retiming from `tensile_fp16_nt_hhs.md`, and treat the `8192^3` winner as a center-point seed/evidence rather than a shape-generic conclusion.
- Search-space review expanded the first-pass domain from the grid vocabulary plus observed NT artifacts: TLDS2/LDS-pad profiles, `NumElementsPerBatchStore=0/14/20/24/32`, `StoreSyncOpt=1/2/4`, `GroupLoadStore=True`, WGM `4/16`, stagger `16/64`, and checked-in-style small/skinny seed families.
- Exact-shape and nearest-shape transfer now seed each proposal from validation-passed winners of cached shapes before random restarts. Imported hipBLASLt baseline configs participate in this path only if they remain the best cached candidates for those shapes.
- Pair-level cache inefficiency has a first fix: scheduler now groups shapes by exact missing candidate subset within each candidate/shape chunk, so planned batches do not deliberately re-run cached pairs. Future dense-merge heuristics may allow a small number of `ok` extras if compile overhead dominates.
- APU thermal coupling: compile and benchmark are sequential, but a highly threaded compile can heat Strix Halo immediately before GPU timing. Default policy is still no deliberate compile/benchmark overlap and no deliberate cool-down sleep; reduce `--compile-threads` if pilot timings look thermally biased.
- Multi-candidate build failure attribution: only single-candidate build failures are negative-cached today. If a multi-candidate batch fails, isolate with `--candidate-batch-size 1` before marking candidates bad.
- Search-time validation now defaults to full validation (`NumElementsToValidate=-1`) after adding the OpenBLAS-backed structured-runner reference path. `NumElementsToValidate` is an execution budget, not a timing-cache identity field, so GPU-only top-ups with prior validation evidence append under the same benchmark protocol hash.
- Final-YAML mapping was repaired after the full scan. The repaired mapper handles TLDS2-derived `1LDSBuffer`/`PrefetchLocalRead` rewrites and inactive `StaggerU=0` `StaggerUMapping`/`StaggerUStride` normalization. Re-ingest now reports zero unmapped rows and zero unmatched final solutions.

Suggested pre-grid test:

```bash
python3 -m evotensile.cli schedule-batches \
  --db out/evotensile.sqlite \
  --output-dir out/pregrid_test \
  --limit-shapes 2 \
  --candidate-batch-size 4 \
  --max-batches 1 \
  --keep-going
```

## 14. Pilot Timing Data

Measured pre-structured-runner data on Radeon 8060S gfx1151 with the former standalone TensileLite client path:
- 1 shape x 1 candidate test: wall `13.28s`; build `4.464s`; benchmark `8.529s`; inserted `10 ok`; summed recorded GEMM time `0.000144s`; non-GEMM runner time `12.993s`.
- 2 shapes x 2 candidates test: wall `11.34s`; build `4.284s`; benchmark `6.623s`; inserted `20 ok` and `2 rejected`; summed recorded GEMM time `0.000352s`; non-GEMM runner time `10.907s`.
- 10 shapes x 8 candidates medium probe: wall `19.50s`; build `4.428s`; benchmark `7.655s`; inserted `700 ok` and `10 rejected`; summed recorded GEMM time `0.03047s`; non-GEMM runner time `12.053s`.

Pre-full-run estimate:
- The default 100-shape plan is `135` candidates x `100` shapes = `13,500` candidate-shape pairs in `5` batches (`32,32,32,32,7` candidates x `100` shapes).
- Raw GEMM work per candidate across the 100-shape grid is `131,063,611,392` FLOPs; at 10-20 TFLOP/s, 135 candidates x 10 benchmark samples would be only `8.85-17.69s` of kernel time.
- The medium probe shows orchestration/validation/logging dominates small and medium batches: recorded GEMM time was `0.030s` out of `12.083s` runner time.
- A conservative full-grid wall estimate before running is `20-35 minutes`, dominated by TensileLite client validation/logging and code-object/build orchestration rather than matmul kernel time.

Actual 100-shape first-pass results:
- Launch used the former standalone TensileLite client path with `seed-random-gomea`, candidate batch size `32`, shape batch size `100`, and compile threads `4`.
- Wall time was `1265.21s` (`21.1 min`) for `135` proposed candidates x `100` shapes planned as `13,500` pairs in `5` batches.
- Recorded TensileLite subprocess durations summed to `562.6s`; Python ingestion/log processing and filesystem overhead were the rest, confirming that orchestration dominates the generic client path.
- Initial DB before mapper repair had `64,000 ok`, `2,000 validation_fail`, and `6,900 rejected` rows, with accepted rows under-mapped as rejected/unmapped.
- Repaired re-ingest took `10.86s` and produced `75,000 ok`, `2,000 validation_fail`, and `5,800 rejected` rows: `7,500` ok pairs, `200` validation-failed pairs, and exactly `10` samples per ok pair.
- First-pass summed recorded ok GEMM time was `17.178s`; validation-failed GEMM time was `0.348s`. The remaining wall time is compile/client/log/validation/database overhead rather than matmul time.

Actual top-4 full-validation retime:
- Historical fixed top-4 retime selected top-4 per shape from the repaired first-pass DB and grouped exact pair sets without cross-product extras.
- Protocol: default full validation with `NumElementsToValidate=-1`, producing current benchmark protocol hash `bproto_d8085f528519ae64`.
- Coverage: `400` intended pairs, `35` unique candidates, `57` groups, `4,000 ok` samples, `0` rejected/unmapped/validation-fail rows.
- Wall time was `675.86s`; summed retime ok GEMM time was `0.256s`, so this was almost entirely generic TensileLite compile/client overhead.
- Historical fixed full-validation retime changed `57` of `100` per-shape winners versus the first-pass screen, which motivated integrated adaptive sampling before GridBased update.
- Top-k sensitivity from the historical final top-up: the final winner's first-pass rank was `1` for `43` shapes, `2` for `27`, `3` for `17`, and `4` for `13`. Topping up only top-1 would miss `57/100` final winners, top-2 would miss `30/100`, and top-3 would miss `13/100`; top-4 captured every final winner observed in this run.
- The retimed winner versus the retimed first-pass top-1 improved by median `0.367%`, mean `3.904%`, and max `35.202%`; `21` shapes improved by more than `5%`, and `13` improved by more than `10%`.
- Current policy: the 100-shape artifact needs no further top-up, but future grids should use integrated adaptive sampling with at least top-8/top-10 plausible contenders on staged subsets to check whether rank-5+ candidates sometimes overtake top-4.
- The old rebuild-ready export artifacts were removed; `scripts/update_hipblaslt_gridbased_logic.py` now selects winners directly from `out/grid100_full_20260618_repaired.sqlite` and reconstructs full solution dictionaries from DB run outputs plus checked-in source logic.
- Historical analysis artifacts were removed from `out/`; use ad-hoc DB queries for future analysis.

Build-directory convention for current work:
- Use `~/rocm-libraries/build/hipblaslt/` for the normal `~/rocm-libraries/build_hipblaslt.sh` build tree.
- Use `~/rocm-libraries/build/hipblaslt-bench/` for the normal `~/rocm-libraries/build_hipblaslt_bench.sh` client build tree, including `hipblaslt-bench` speed comparisons, `hipblaslt-bench --verify` correctness checks, and `hipblaslt-test`.
- Override `BUILD_DIR` only when comparing multiple versions or preserving a specific historical build tree.

Current hipBLASLt baseline import:
- Built `hipblaslt-bench` with `~/rocm-libraries/build_hipblaslt_bench.sh`. The normal binary path is `~/rocm-libraries/build/hipblaslt-bench/clients/hipblaslt-bench`.
- Before the tuned rebuild/install, the installed `_rocm_sdk_devel` package had `libhipblaslt.so` but not the gfx1151 TensileLite assets at the default path, so baseline import used `HIPBLASLT_TENSILE_LIBPATH=~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_libraries/lib/hipblaslt/library/gfx1151`. Post-install validation should use `HIPBLASLT_TENSILE_LIBPATH=$ROCM_PATH/lib/hipblaslt/library/gfx1151`.
- Current hipBLASLt-selected configs were imported with `scripts/import_hipblaslt_baselines.py` into `out/grid100_full_20260618_repaired.sqlite`: `100` queried shapes, `22` unique installed candidates, and `1,000 ok` structured samples under benchmark protocol hash `bproto_d8085f528519ae64`.
- Baseline import protocol: FP16 NT HHS, batch `1`, bias vector type `f16_r`, bias source `d`, `scaleAlpha_vector`, activation `none`, `alpha=2`, `beta=2`, `--initialization hpl`, `--cold_iters 10`, `--iters 100`, `--use_gpu_timer`, `--requested_solution 1`. `alpha=2`/`beta=2` match the EvoTensile structured-runner init modes (`init-alpha=Two`, `init-beta=Two`); use `--beta 0` later for the wrapper-contract variant if needed.
- The old post-export `hipblaslt-bench` comparison/replacement script was removed. EvoTensile DB winner selection is now authoritative because it uses the benchmark protocol selected for long-running inference/training, while `hipblaslt-bench` reports one average over hot launches and does not expose matching repeated-group median timing through CLI flags alone.

Actual rebuilt hipBLASLt validation:
- Rebuilt hipBLASLt from `~/rocm-libraries` with `GPU_TARGETS=gfx1151`, then installed into `$ROCM_PATH`. Going forward, use the normal build tree `~/rocm-libraries/build/hipblaslt/` unless a version-comparison build needs a separate directory.
- Runtime performance validation used `~/ComfyUI-FeatherOps/benchmark_mm_hipblaslt_fp16.py` with `TORCH_BLAS_PREFER_HIPBLASLT=1` and `HIPBLASLT_TENSILE_LIBPATH=$ROCM_PATH/lib/hipblaslt/library/gfx1151`.
- Benchmark output: `~/ComfyUI-FeatherOps/mm_hipblaslt_fp16.csv` and `/tmp/benchmark_mm_hipblaslt_fp16_grid100_20260619_125807.log`.
- The 1024^3 NT path showed the expected improvement versus the TheRock issue baseline: `torch_mm_NT` `16.007 -> 23.434 TFLOP/s` (`1.464x`), `torch_linear_NT` `15.998 -> 23.465 TFLOP/s` (`1.467x`), and direct `hipblaslt_NT` `14.417 -> 25.554 TFLOP/s` (`1.772x`).
- Larger square NT cases also improved strongly in that benchmark: direct `hipblaslt_NT` speedup was `1.829x` at `2048`, `2.218x` at `4096`, and `4.804x` at `8192` versus the issue baseline.
- Lightweight installed correctness test used `scripts/verify_installed_hipblaslt.py`, which drives `hipblaslt-bench --verify` against CPU reference for six curated target and off-grid cases; result was `6/6 ok`, `0` failures, in `2.16s`. The old `out/` correctness artifact was removed after recording the result here.
- Upstream `hipblaslt-test` was built in the normal client build tree and run with GTest machine-readable XML. `*smoke*` passed `911/911` tests in `5.916s`. Full `*quick*` ran `7606` tests in `45.039s` with `248` `NO solution found!` availability failures limited to FP16/BF16 NT `quick_matmul_one` edge/skinny cases; excluding only that no-solution family, `7358/7358` passed in `42.821s`. Full `*pre_checkin*` ran `6401` tests in `197.59s` with `8` `NO solution found!` availability failures limited to FP16/BF16 NT `k=0` cases; excluding only that no-solution family, `6393/6393` passed in `196.938s`.
- Validation artifacts include `/tmp/hipblaslt_test_validation_summary_20260619.json`, `/tmp/hipblaslt_test_quick_20260619_134911.xml`, `/tmp/hipblaslt_test_quick_minus_nosol_20260619_135522.xml`, `/tmp/hipblaslt_test_pre_checkin_20260619_135631.xml`, and `/tmp/hipblaslt_test_pre_checkin_minus_k0_nosol_20260619_141122.xml`.
- The benchmark log had no warning/error/exception/fallback/nan hits. The successful build log still had TensileLite YAML type-mismatch warnings in five pre-existing non-target files; the four updated `Ailk_Bjlk` files were normalized before the final install.

Structured runner refactor status:
- `evotensile/profile.py` and `evotensile/protocol.py` define the bundled `gfx1151-nt-hhs` target profile and typed benchmark protocol. The search CLI derives `problem_type_hash` and `benchmark_protocol_hash` from those objects rather than accepting raw hash/global-parameter overrides.
- `evotensile/structured_runner.py` now defines the exact-pair JSONL contract, maps accepted final-YAML solutions back to `(shape_id, candidate_hash)` once after codegen, dispatches the external structured runner, validates emitted rows, and writes DB evaluations directly without TensileLite client CSV/log archaeology.
- `schedule-batches` uses the structured path only; the old TensileLite `LibraryClient` CSV/log runner path, standalone retime scripts, and in-process test backend have been removed.
- The structured result format carries `shape_id` and `candidate_hash` in every sample row, so IO mapping correctness no longer depends on stdout order, problem-progress strings, or kernel names.
- `csrc/structured_runner.cpp` now implements a narrow production HIP/TensileLite backend for the current gfx1151 FP16 NT HHS bias + `scaleAlpha_vector` target. It loads generated `TensileLibrary_gfx1151.yaml` or build-only `TensileLibrary.yaml` plus `.co`/`.hsaco` artifacts, selects exact solution indices, launches via `TensileLite::hip::SolutionAdapter`, validates CPU-reference samples, and emits JSONL rows.
- `scripts/build_structured_runner.sh` builds `./build/evotensile-structured-runner` against the existing TensileLite client build under `~/rocm-libraries/build/tensilelite-client`.
- The dispatcher now accepts both full-client `4_LibraryClient/library/gfx*` output and build-only `1_BenchmarkProblems/**/source/library/gfx*` cache output.
- A real `schedule-batches --runner-bin ./build/evotensile-structured-runner` test passed after refreshing stale `rocisa` bindings: 1 shape x 1 candidate produced `2 ok` validation-passed DB samples; 2 shapes x 2 candidates produced `2 ok` plus `2 rejected` rows, matching final-YAML accepted/rejected mapping.
- An 8-pair rerun from top-4 retime artifacts (`2` shapes x `4` candidates, solution indices `0-3`) passed validation for every pair. Under the normal hot-loop envelope with `10` samples, structured medians were within about `0.4-2.7%` of legacy per-solution CSV medians; isolated low outliers remain a measurement-noise issue, not a mapping/correctness issue.
- A one-pair clean generated-library test passed on `m512_n128_b1_k256`, emitting `2` validation-passed structured samples. The same test against an old generated artifact with hard-coded `CUCount: 16` correctly returned `WRONG_HARDWARE` on the 20-CU Strix Halo, matching the known CUCount pitfall.
- Tests use fake external runner scripts and fake TensileLite build outputs to verify exact mapping, validation-gated direct DB ingestion, build-only library layout discovery, rejected-candidate handling, and bad-runner detection without carrying an in-process test backend.

## 15. Open Questions

- Best batch size for TensileLite compile/run overhead on the target machine.
- How much candidate union across shapes is acceptable before wasted cross-evaluation dominates.
- Which cheap constraints can predict invalid TensileLite solutions before invoking TensileLite.
- Whether `SkipSlowSolutionRatio` biases search results for small/skinny shapes.
- Whether any subset of Origami features is useful as a weak feature for surrogate training.
- How broad the routine upstream `hipblaslt-test` tier should be after each future GridBased YAML update: smoke only, filtered quick, or filtered pre_checkin.

## 16. Immediate Next Steps

- Expand production backend validation from the current accepted top-up pairs to `10-20` pairs spanning more shapes, candidates, and generated libraries before starting the 9,681-shape grid.
- Decide whether to use explicit runner priming, additional warmups, or robust sample filtering for occasional first-use/timing outliers while keeping benchmark protocol identity user-controlled.
- Improve multi-candidate build-failure/timeout attribution once failure signatures are better understood; single-candidate build timeouts are classified, but multi-candidate failures are still intentionally not negative-cached.
- Decide whether profile-owned runner build commands should be executable workflow steps; profiles currently record the default runner path/build command, but users still run the build command explicitly.
- Add non-hindsight refinement operators that learn from cached winners/near-winners without hard-coding the documented `8192^3` neighborhood.
- Keep the rebuilt-hipBLASLt validation recipe as the standard post-install gate: target `hipblaslt-bench --verify` test, upstream `hipblaslt-test` smoke/filtered quick as needed, and the PyTorch/FeatherOps 1024^3 NT performance path before scaling further.
