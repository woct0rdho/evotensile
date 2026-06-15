# EvoTensile Plan

EvoTensile is an external smart-search autotuner for TensileLite / hipBLASLt.  The initial target is gfx1151 FP16 NT HHS non-AuxH GridBased GEMM tuning for the 100-shape pilot grid described in `ComfyUI-FeatherOps/doc/tensile_fp16_nt_hhs_grid.md`, then expansion to finer/larger shape grids.

## 1. Motivation

Hand-picking a small candidate set is fragile because:

- performance far from `8192^3` is hard to predict;
- TN/NN winners may or may not transfer to NT;
- Tensile knobs have strong interactions / epistasis;
- local optima are likely;
- Origami ranking is not reliable enough for hard pruning on this target;
- TensileLite itself enumerates Cartesian products unless candidates are bundled through `Groups`.

EvoTensile will treat Tensile configurations as a constrained, noisy, mixed-categorical black-box optimization problem and search it with random, evolutionary, local, and later surrogate-assisted methods.

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

The pilot may benchmark more than 10 configs per shape.  Initial budget target: roughly 64-256 evaluated candidate configs per shape, subject to wall-time results from the harness.

## 3. Non-goals for the First Version

- Do not modify TensileLite internals initially.
- Do not rely on Origami / Formocast as a hard pruning gate.
- Do not attempt to solve all data types/layouts immediately.
- Do not generate final production hipBLASLt logic before the benchmark/evaluation loop is stable.
- Do not assume one unique candidate set per shape; use buckets/batches to control compilation and process overhead.

## 4. Design Principles

1. **External orchestration first**
   EvoTensile generates candidate batches, emits TensileLite YAML with `Groups`, runs TensileLite, parses outputs, and records results.

2. **Canonical candidate hashes**
   Every candidate config gets a stable canonical representation and hash.  This enables deduplication, caching, and reproducibility.

3. **Persistent result database**
   Every attempted `(shape, candidate, environment)` should be stored, including invalid builds, correctness failures, timeouts, and benchmark results.

4. **Batch evaluation**
   Avoid per-candidate/per-shape process launches.  Emit batches of candidates into `Groups` and evaluate shape buckets in one TensileLite run when possible.

5. **Search data is valuable**
   The pilot should produce a reusable dataset for later nearest-shape seeding and surrogate training.

6. **Final winners are retimed**
   Search-time timings can be cheaper/noisier.  Final top candidates must be validated and retimed with a stricter protocol.

## 5. Proposed Repository Layout

```text
evotensile/
  PLAN.md
  README.md
  pyproject.toml
  evotensile/
    __init__.py
    candidate.py          # canonical candidate representation and hashing
    search_space.py       # NT HHS gfx1151 parameter domains and constraints
    shapes.py             # pilot grid, large grid, bucketing, nearest neighbors
    yaml_writer.py        # TensileLite YAML writer using ForkParameters/Groups
    runner.py             # TensileLite subprocess runner
    parser.py             # parse CSV/YAML/log outputs
    database.py           # SQLite/DuckDB persistence layer
    metrics.py            # timing, median, variance, TFLOP/s, validation status
    scheduler.py          # batch planning and resumable execution
    report.py             # summarize winners and failures
    export.py             # export winner YAML / candidate bundles
    search/
      __init__.py
      random_search.py
      stratified.py
      local_search.py
      differential_evolution.py
      gomea.py
      surrogate_lfbo.py
  configs/
    fp16_nt_hhs_gfx1151.yaml
  scripts/
    run_pilot_100.py
    resume.py
    report_pilot.py
    export_winners.py
  tests/
```

Implementation can start with fewer modules, but the boundaries above should guide the design.

## 6. Candidate Model

A candidate is a complete Tensile solution parameter bundle, not a single independent knob.  EvoTensile will emit each candidate as one `Groups` entry.

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
    "StoreVectorWidth": 1,
    "StoreRemapVectorWidth": 0,
    "StorePriorityOpt": True,
    "NumElementsPerBatchStore": 8,
    "StoreSyncOpt": 0,
    "GroupLoadStore": False,
    "AssertFree0ElementMultiple": 1,
    "AssertFree1ElementMultiple": 1,
    "AssertSummationElementMultiple": 1,
}
```

The search space should support conditional constraints and linked mutations.  Examples:

- mutate `MatrixInstruction`, `WorkGroup`, and macro-tile-related choices together;
- mutate `DepthU` with vector-width / prefetch choices;
- mutate `WorkGroupMapping`, `StaggerU`, and `StaggerUMapping` together;
- keep known-invalid or repeatedly failing combinations out of later batches.

## 7. TensileLite Evaluation Strategy

### YAML generation

EvoTensile should generate TensileLite YAML where candidate configs are emitted as `Groups`, for example:

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

1. **Candidate-batch x shape-bucket**
   Evaluate a batch of candidates over a bucket of shapes.  Good for broad scans.

2. **Shape-local candidate batch**
   Evaluate a candidate batch specific to a single shape or small shape cluster.  Useful for refinement.

The scheduler should choose batch sizes to balance code-object build overhead, process overhead, and wasted cross-evaluation.

### Caching

Use TensileLite `--build-only` and `--use-cache` where helpful, but do not rely only on Tensile's cache.  EvoTensile must maintain its own DB-level cache keyed by:

- candidate hash;
- exact shape;
- problem type hash;
- GPU target / arch;
- ROCm/hipBLASLt/Tensile commit or path hash if available;
- critical environment variables;
- benchmark protocol version.

## 8. Result Database

Use SQLite initially.  DuckDB can be added later for analytics.

Core tables:

### `candidates`

- `candidate_hash TEXT PRIMARY KEY`
- `candidate_json TEXT`
- `created_at TEXT`
- `source TEXT` — random, seed, mutation, crossover, imported, etc.
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
- `timestamp TEXT`
- `git_info TEXT`
- `rocm_info TEXT`
- `tensile_path TEXT`
- `yaml_path TEXT`
- `output_dir TEXT`
- `status TEXT`

### `evaluations`

- `shape_id TEXT`
- `candidate_hash TEXT`
- `run_id TEXT`
- `status TEXT` — ok, invalid, compile_fail, validation_fail, timeout, parse_fail
- `time_us REAL`
- `gflops REAL`
- `validation TEXT`
- `raw_solution_index INTEGER`
- `raw_csv_row TEXT`
- `created_at TEXT`

Index `(shape_id, candidate_hash)` heavily.

## 9. Search Algorithms

### Phase A: baseline generators

Implement first:

- deterministic seeds:
  - known `8192^3` winner;
  - variants around known winner;
  - transferred TN/NN candidates if available;
- random valid generator;
- stratified generator over macro-tile / depth / GSU / schedule families;
- shape-aware MI/GSU generator inspired by TensileLite's beta `tensile_config_generator.py`.

### Phase B: local/evolutionary search

Implement after the runner is stable:

- local mutation around top-k winners;
- crossover between near-winners;
- differential evolution over encoded categorical values;
- GOMEA-like linkage-aware mixing inspired by `~/rocm_wmma_gemm/rocm_wmma_gemm/config/tune.py`.

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

1. Represent each shape in log/ratio feature space.
2. Find nearest tuned shapes.
3. Seed new shape with:
   - nearest winners;
   - nearest near-winners;
   - mutations around nearest winners;
   - a few global robust candidates;
   - a few random exploratory candidates.
4. Run a smaller budget, e.g. 16-64 configs/shape, then retime finalists.

Shape features:

```text
log2(M), log2(N), log2(K), log2(batch)
log2(M/N), log2(K/M), log2(K/N)
ceil(M/MT0), ceil(N/MT1), tile count, edge fraction
```

## 11. Benchmark Protocol

### Search-time protocol

Goal: throughput and correctness screening.

Suggested initial settings:

```yaml
PredictionThreshold: 2.0   # do not use Formocast prediction on gfx1151
NumWarmups: 1-3
NumBenchmarks: small
NumElementsToValidate: 128 or stronger for risky candidates
SkipSlowSolutionRatio: optional, e.g. 0.75-0.9 after validation
CSVExportWinner: True if helpful for parsing
```

### Final confirmation protocol

For top candidates per shape:

- retime top 3-10 candidates;
- repeated samples;
- median / trimmed mean;
- validation enabled;
- compare against existing hipBLASLt logic and known baseline configs.

Final results should separate:

- search-time best;
- confirmed best;
- correctness failures;
- unstable/noisy candidates.

## 12. Milestones

### M0: repository skeleton

- Create package structure.
- Add config file for gfx1151 FP16 NT HHS.
- Add plan and README.

### M1: candidate + shape primitives

- Candidate dataclass / canonical JSON / hash.
- Pilot grid generator.
- Search-space domains and cheap constraints.
- Random and deterministic seed generation.

### M2: YAML writer

- Emit valid TensileLite YAML using `Groups`.
- Emit fixed problem type and 100-shape pilot grid.
- Support candidate batches and shape buckets.

### M3: runner and parser

- Invoke TensileLite in a subprocess.
- Capture logs and output paths.
- Parse CSV/YAML results.
- Store all observations in SQLite.
- Support resume without repeating known `(shape, candidate)` evaluations.

### M4: first pilot scan

- Evaluate 64-128 candidates per shape or shape bucket.
- Generate winner/near-winner report.
- Identify invalid/high-failure regions of search space.

### M5: local/evolutionary refinement

- Add elite mutation and crossover.
- Add simple differential evolution or GOMEA-like mixing.
- Run 1-3 refinement rounds on pilot shapes.

### M6: final confirmation + export

- Retime finalists.
- Export selected candidate bundles.
- Generate candidate-to-shape mapping suitable for later Tensile logic generation/merge.

### M7: transfer to finer grids

- Add nearest-shape seeding.
- Add smaller-budget local refinement for new shapes.
- Add surrogate-assisted proposal once enough data exists.

## 13. Open Questions

- Best batch size for TensileLite compile/run overhead on the target machine.
- How much candidate union across shapes is acceptable before wasted cross-evaluation dominates.
- Which cheap constraints can predict invalid Tensile solutions before invoking Tensile.
- Whether `SkipSlowSolutionRatio` biases search results for small/skinny shapes.
- Whether any subset of Origami features is useful as a weak feature for surrogate training.
- How to export final results into the existing hipBLASLt GridBased logic workflow with minimal manual steps.

## 14. Immediate Next Steps

1. Create the Python package skeleton.
2. Implement candidate hashing and pilot shape generation.
3. Implement a minimal YAML writer for one batch of candidates and the 100-shape grid.
4. Implement a dry-run command that prints candidate counts and writes YAML without running Tensile.
5. Run a tiny smoke test: 2 shapes x 2 candidates.
6. Add SQLite result tracking.
7. Scale to first broad random/stratified pilot batch.
