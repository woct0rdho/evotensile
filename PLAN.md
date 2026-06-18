# EvoTensile Plan

EvoTensile is an external smart-search autotuner for TensileLite / hipBLASLt. The initial target is gfx1151 FP16 NT HHS non-AuxH GridBased GEMM tuning for the 100-shape pilot grid described in `~/ComfyUI-FeatherOps/doc/tensile_fp16_nt_hhs_grid.md`, then expansion to finer/larger shape grids.

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
- Do not generate final production hipBLASLt logic before the benchmark/evaluation loop is stable.
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
- the gfx1151 FP16 NT HHS problem type and pilot grid are encoded;
- the search space can express the documented `8192^3` SIA3/no-store-priority winner family;
- generated YAML uses complete candidate `Groups` rather than independent Cartesian products;
- runner support exists for direct runs and compile-then-serial-benchmark runs;
- DB schema includes manual cache namespace fields (`version_name`, `problem_type_hash`, `benchmark_protocol_hash`);
- validation-aware CSV/log parsing and SQLite ingestion exist, using TensileLite final solution YAML as the source of truth for candidate mapping;
- cache-aware batch scheduling exists for missing `ok` observations, with compile-only, serial benchmark, and immediate ingestion phases;
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

Use TensileLite `--build-only` and `--use-cache` where helpful, but do not rely only on TensileLite's build cache. EvoTensile maintains DB-level timing-cache identity fields based on:
- user-controlled `version_name` / `tensilelite_version_name` namespace;
- problem type hash;
- benchmark protocol hash;
- exact shape;
- candidate hash.

Do not automatically key invalidation on a `rocm-libraries` commit hash. Store commit/source metadata for audit, but make cache refresh explicit through the user-controlled version namespace.

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
- `version_name TEXT`
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
- `version_name TEXT`
- `problem_type_hash TEXT`
- `benchmark_protocol_hash TEXT`
- `shape_id TEXT`
- `candidate_hash TEXT`
- `run_id TEXT`
- `status TEXT` - ok, invalid, compile_fail, validation_fail, timeout, parse_fail
- `time_us REAL`
- `gflops REAL`
- `validation TEXT`
- `solution_index INTEGER`
- `raw_csv_row TEXT`
- `created_at REAL`

Index `(version_name, problem_type_hash, benchmark_protocol_hash, shape_id, candidate_hash)` heavily for cache lookups.

## 9. Search Algorithms

### Phase A: baseline generators

Implemented now:
- deterministic conservative seeds, including large-square, TLDS2/LDS-pad, and small/skinny checked-in-style NT seed families;
- nearest-shape winner transfer from validation-passed cached observations, defaulting to `4` nearby shapes and `2` top candidates per shape;
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
  - nearest winners, implemented in `schedule-batches` through `--transfer-shapes` and `--transfer-per-shape`;
  - nearest near-winners;
  - mutations around nearest winners;
  - a few global robust candidates;
  - a few random exploratory candidates.
- Run a smaller budget, e.g. 16-64 configs/shape, then retime finalists.

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
NumElementsToValidate: 128 or stronger for risky candidates
SkipSlowSolutionRatio: 0.0 initially; optional after validation
CSVExportWinner: True if helpful for parsing
```

Cold-loop behavior is intentionally not part of the tuning loop. Tracking it would add measurement cost and optimize for first-run / bursty-idle effects rather than sustained throughput. If needed, add a separate later analysis pass for first-request latency, module-load/JIT effects, allocator warmup, and idle-to-active behavior.

### Final confirmation protocol

For top candidates per shape:
- retime top 3-10 candidates with the same hot-loop protocol;
- use repeated samples;
- report median / trimmed mean;
- keep validation enabled;
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

### M3: runner, parser, and cache identity - partially done

Done:
- invoke TensileLite in a subprocess;
- capture logs and output paths;
- parse CSV files for inspection;
- initialize SQLite schema;
- record run metadata and manual cache identity;
- query cache identity/status/missing evaluations and rank only validation-passed observations;
- schedule missing `ok` observations into candidate/shape batches and ingest each completed batch.

Implemented now:
- rejected final-YAML candidates, validation failures, and single-candidate build failures are reusable negative-cache entries that scheduler skips.

Remaining:
- classify timeouts, multi-candidate build failures, and parse failures robustly beyond validation pass/fail rows.

### M4: first pilot scan - next major milestone

- Evaluate 64-128 candidates per shape or shape bucket.
- Generate winner/near-winner report.
- Identify invalid/high-failure regions of search space.
- Verify batch sizes against compile time and serial benchmark time.

### M5: local/evolutionary refinement

- Improve the current DE/GOMEA scheduler path with shape-aware parent selection and non-hindsight refinement.
- Add elite crossover and failure-aware mutations.
- Run 1-3 refinement rounds on pilot shapes.

### M6: final confirmation + export

- Retime finalists with the same hot-loop protocol.
- Export selected candidate bundles.
- Generate candidate-to-shape mapping suitable for later TensileLite logic generation/merge.

### M7: transfer to finer grids

- Nearest-shape seeding is implemented for cached validation-passed winners.
- Add smaller-budget local refinement for new shapes.
- Add surrogate-assisted proposal once enough data exists.

## 13. Pre-Pilot Review Notes

Remaining risks to track before/during the first 100-shape run:
- The `~/ComfyUI-FeatherOps/doc/tensile_fp16_nt_hhs_grid.md` plan is still applicable: start with the 100-shape NT HHS non-AuxH grid, use hot-loop retiming from `tensile_fp16_nt_hhs.md`, and treat the `8192^3` winner as a center-point seed/evidence rather than a shape-generic conclusion.
- Search-space review expanded the first-pass domain from the grid vocabulary plus observed NT artifacts: TLDS2/LDS-pad profiles, `NumElementsPerBatchStore=0/14/20/24/32`, `StoreSyncOpt=1/2/4`, `GroupLoadStore=True`, WGM `4/16`, stagger `16/64`, and checked-in-style small/skinny seed families.
- Nearest-shape transfer now seeds each proposal from validation-passed winners of nearby cached shapes before random restarts, which helps staged 100-shape grid tuning reuse earlier shape results without trusting validation-failed or unknown rows.
- Pair-level cache inefficiency has a first fix: scheduler now groups shapes by exact missing candidate subset within each candidate/shape chunk, so planned batches do not deliberately re-run cached pairs. Future dense-merge heuristics may allow a small number of `ok` extras if compile overhead dominates.
- APU thermal coupling: compile and benchmark are sequential, but a highly threaded compile can heat Strix Halo immediately before GPU timing. Default policy is still no deliberate compile/benchmark overlap and no deliberate cool-down sleep; reduce `--compile-threads` if pilot timings look thermally biased.
- Multi-candidate build failure attribution: only single-candidate build failures are negative-cached today. If a multi-candidate batch fails, isolate with `--candidate-batch-size 1` before marking candidates bad.
- Search-time validation is partial: `NumElementsToValidate=128` is acceptable for screening, but final winners should be retimed with stronger/full validation before trusting them.
- Real mapping smoke is still needed: run a small multi-candidate schedule with accepted, rejected, and deduplicated candidates before the full grid to validate final-YAML mapping and ingestion behavior on actual TensileLite output.

Suggested pre-grid smoke:

```bash
python3 -m evotensile.cli schedule-batches \
  --db out/evotensile.sqlite \
  --output-dir out/pregrid_smoke \
  --version-name gfx1151_hotloop_pregrid_smoke \
  --limit-shapes 2 \
  --candidate-batch-size 4 \
  --max-batches 1 \
  --keep-going
```

## 14. Open Questions

- Best batch size for TensileLite compile/run overhead on the target machine.
- How much candidate union across shapes is acceptable before wasted cross-evaluation dominates.
- Which cheap constraints can predict invalid TensileLite solutions before invoking TensileLite.
- Whether `SkipSlowSolutionRatio` biases search results for small/skinny shapes.
- Whether any subset of Origami features is useful as a weak feature for surrogate training.
- How to export final results into the existing hipBLASLt GridBased logic workflow with minimal manual steps.

## 15. Immediate Next Steps

- Validate final-solution YAML mapping and `schedule-batches` on a real multi-candidate TensileLite run with known rejected/deduplicated candidates.
- Add timeout and multi-candidate build-failure attribution once failure signatures are better understood.
- Add non-hindsight refinement operators that learn from cached winners/near-winners without hard-coding the documented `8192^3` neighborhood.
- Run the first 100-shape hot-loop pilot scan and produce a winner/near-winner report.
- Add final export of selected candidate bundles for later GridBased logic generation/merge.
