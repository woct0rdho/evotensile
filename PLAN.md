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
- a real one-shape harness under `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/evotensile_one_shape/` showed that random init plus directed local refinement can reproduce the documented `8192^3` winner.

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
    "WorkGroupMapping": 5 | 8,
    "StaggerU": 0 | 8 | 32,
    "StaggerUStride": 256,
    "StaggerUMapping": 0 | 1,

    # LDS / source layout
    "SourceSwap": 0 | 1,
    "1LDSBuffer": 0 | 1,
    "ClusterLocalRead": 0 | 1,
    "TransposeLDS": 0,

    # vectorization
    "VectorWidthA": 1,
    "VectorWidthB": 1 | 2,
    "GlobalReadVectorWidthA": 1 | 2 | 4 | 8,
    "GlobalReadVectorWidthB": 1 | 2 | 4 | 8,
    "LocalReadVectorWidth": 16,

    # stores / assertions
    "StoreVectorWidth": -1,
    "StoreRemapVectorWidth": 0,
    "StorePriorityOpt": True | False,
    "NumElementsPerBatchStore": 4 | 8 | 10 | 12 | 16,
    "StoreSyncOpt": 0,
    "GroupLoadStore": False,
    "AssertFree0ElementMultiple": 8,
    "AssertFree1ElementMultiple": 8,
    "AssertSummationElementMultiple": 16,
}
```

The search space should support conditional constraints and linked mutations. Examples:
- mutate `MatrixInstruction`, `WorkGroup`, and macro-tile-related choices together;
- mutate `DepthU` with vector-width / prefetch choices;
- mutate `WorkGroupMapping`, `StaggerU`, and `StaggerUMapping` together;
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
- deterministic conservative seeds;
- random valid generator;
- local mutation around cached DB elites;
- scheduler proposal modes for seed/random, local-only, and seed/random plus local refinement;
- a ground-truth documented-winner helper for checks, not as a default random-init seed.

Still planned:
- stratified generator over macro-tile / depth / GSU / schedule families;
- shape-aware MI/GSU generator inspired by TensileLite's beta `tensile_config_generator.py`;
- configurable seed packs for known winners, transferred TN/NN candidates, and nearest-shape winners.

### Phase B: local/evolutionary search

Still needed:
- shape-aware candidate proposal beyond optional `--proposal-shape-id` filtering;
- differential evolution over encoded categorical values;
- GOMEA-like linkage-aware mixing inspired by `~/rocm_wmma_gemm/rocm_wmma_gemm/config/tune.py`;
- crossover between near-winners;
- directed/refinement operators promoted from the one-shape harness into generic search code;
- robust failure-aware candidate filtering.

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

## 10. Shape Transfer Strategy

For finer grids:
- Represent each shape in log/ratio feature space.
- Find nearest tuned shapes.
- Seed new shape with:
  - nearest winners;
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

Remaining:
- classify invalid builds, timeouts, rejected candidates, and parse failures robustly beyond validation pass/fail rows.

### M4: first pilot scan - next major milestone

- Evaluate 64-128 candidates per shape or shape bucket.
- Generate winner/near-winner report.
- Identify invalid/high-failure regions of search space.
- Verify batch sizes against compile time and serial benchmark time.

### M5: local/evolutionary refinement

- Extend the current local-mutation scheduler path with differential evolution, GOMEA-like mixing, and directed refinement.
- Add elite crossover and failure-aware mutations.
- Run 1-3 refinement rounds on pilot shapes.

### M6: final confirmation + export

- Retime finalists with the same hot-loop protocol.
- Export selected candidate bundles.
- Generate candidate-to-shape mapping suitable for later TensileLite logic generation/merge.

### M7: transfer to finer grids

- Add nearest-shape seeding.
- Add smaller-budget local refinement for new shapes.
- Add surrogate-assisted proposal once enough data exists.

## 13. Open Questions

- Best batch size for TensileLite compile/run overhead on the target machine.
- How much candidate union across shapes is acceptable before wasted cross-evaluation dominates.
- Which cheap constraints can predict invalid TensileLite solutions before invoking TensileLite.
- Whether `SkipSlowSolutionRatio` biases search results for small/skinny shapes.
- Whether any subset of Origami features is useful as a weak feature for surrogate training.
- How to export final results into the existing hipBLASLt GridBased logic workflow with minimal manual steps.

## 14. Immediate Next Steps

- Validate final-solution YAML mapping and `schedule-batches` on a real multi-candidate TensileLite run with known rejected/deduplicated candidates.
- Record rejected/unmapped/build-failed candidates as reusable non-`ok` observations.
- Promote the directed-refinement operator from the one-shape harness into reusable search code.
- Run the first 100-shape hot-loop pilot scan and produce a winner/near-winner report.
- Add final export of selected candidate bundles for later GridBased logic generation/merge.
