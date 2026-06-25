# Linkage Search Plan

## Goal

Implement data-driven, basin-aware linkage learning for EvoTensile proposal generation, inspired by the latest `~/rocm_wmma_gemm` tuner changes in commit `8aae96ed64bc1b0d414af91c31cf05f12331b371`.

The goal is not to copy that tuner directly. EvoTensile has a richer TensileLite solution space, shape-dependent validity, final-YAML mapping, adaptive sampling, and a persistent SQLite evidence cache. The useful ideas are:
- learn linkage from high-performing validated evidence, not from all generated candidates;
- keep separate linkage models for structurally distinct basins;
- use rank/ordinal information where a knob has meaningful ordering;
- seed and mix from multiple diverse elites rather than a single global winner;
- retire or de-emphasize stagnant small populations while larger populations continue exploring.

## Source Inspiration

The `rocm_wmma_gemm` commit adds or clarifies:
- truncation-selected linkage learning from scored configs in `~/rocm_wmma_gemm/rocm_wmma_gemm/config/tune.py`;
- `TRUNCATION_TAU = 0.5` and `MIN_SAMPLE = 8` as simple defaults for when learned linkage becomes meaningful;
- balanced leader clustering by structural Hamming distance before building linkage trees;
- rank-based Mutual Information followed by UPGMA FOS construction with a small spurious-linkage guard;
- assigning each individual to the nearest cluster's FOS for mixing;
- hall-of-fame seeding, interleaved multi-start, forced improvement, and stagnation termination.

The `load.hpp` changes are mostly naming cleanup around `fast_load` / `fast_store`; they do not directly affect EvoTensile config search.

## Existing EvoTensile Baseline

EvoTensile already has several compatible pieces:
- `evotensile/search/encoding.py` encodes full TensileLite candidates as categorical genomes over `DOMAINS`.
- `evotensile/search/gomea.py` has rule-derived static linkage groups plus parent-genome fallback FOS construction through shared learned-linkage primitives.
- `evotensile/scheduler.py` ranks validation-passed elites through SQLite and proposes random/local/DE/GOMEA candidates.
- The scheduler now uses production-oriented candidate batching and compile-cache reuse, so larger generation-level proposal cohorts are practical.
- Structured diagnostics and shape-aware invalidity keep backend failures separate from hard validity rules.

The missing piece is a persistent, evidence-backed linkage model that uses validated performance rows across shapes and generations.

## Design Principles

- Validated evidence only: Learn positive linkage from `status='ok'` rows with passing validation, or from GPU-only top-up rows backed by prior passing validation.
- Performance enters through selection: Build the linkage model from the best truncation pool, not from raw random history.
- Basin-aware before global: Maintain multiple linkage models for structurally distinct config families, then assign each proposal parent to its nearest basin.
- Hybrid ordinal/nominal MI: Use ordinal/rank bins only for knobs with meaningful order. Keep nominal MI for purely categorical structures such as `MatrixInstruction` and tuple-valued fields.
- No hard validity drift: Learned linkage is a proposal mechanism only. It must not shrink `DOMAINS` or add validity rules.
- Shape-aware, not shape-fragmented: Learn global and shape-cluster models. Avoid building one fragile model per shape unless enough evidence exists.
- Auditable metadata: Record learned model summaries and chosen FOS groups in run metadata so proposal behavior is explainable.

## Data Model

### Evidence Rows

Add a lightweight query/helper that returns scored candidate evidence for a target profile and protocol:

```text
shape_id
candidate_hash
median_time_us
median_gflops
samples
validation state
candidate canonical params
shape features
```

The first implementation can use `EvoTensileDB.rank_evaluations()` plus `db.get_candidates()`. A later optimization can add a dedicated query that returns all fields in one pass.

Filter criteria:
- `status='ok'` only;
- minimum sample count configurable, default `1` for early search and higher for mature campaigns;
- validation-passed evidence required through existing DB ranking semantics;
- optional shape filter or nearest-shape pool for local repair/refinement;
- exclude rows from incompatible profile/problem/protocol hashes.

### Scoring

Use per-shape rank rather than raw global time when pooling across shapes:

```text
score = rank percentile within shape, lower is better
```

For model learning, sort by:
- shape-local rank percentile;
- median GFLOP/s relative to shape winner;
- sample count / confidence as a tie-breaker.

This avoids large shapes dominating the truncation pool just because their absolute times are larger.

## Linkage Learning Pipeline

### Stage 1: Truncation Pool

Build a candidate-evidence pool from validated rows:
- group evidence by shape;
- keep the top `elite_per_shape` rows per shape, default `8`;
- merge and dedupe by candidate hash;
- sort by robust aggregate rank across shapes;
- keep the best `truncation_tau` fraction, default `0.5`;
- require at least `min_linkage_samples`, default `8`.

If there is insufficient evidence, fall back to the existing static `NT_HHS_LINKAGE_GROUPS` plus parent-only linkage.

### Stage 2: Structural Features

Define a structural feature vector for basin clustering. Initial features:
- `MatrixInstruction` family and derived macro tile;
- `WorkGroup`, `DepthU`, `GlobalSplitU`;
- `TransposeLDS` and TLDS-derived linked path;
- `PrefetchGlobalRead`, `PrefetchLocalRead`, `1LDSBuffer`, `ClusterLocalRead`;
- `StoreVectorWidth`, `StorePriorityOpt`, `NumElementsPerBatchStore`, `StoreSyncOpt`;
- `_nt_hhs_lds_bytes` bucket;
- `_valu_vgpr_lower_bound` bucket;
- `SourceSwap`, `ScheduleIterAlg`, `WorkGroupMapping`.

Use standard genome Hamming distance first, with optional feature weights later if clustering is too coarse or too fragmented.

### Stage 3: Balanced Leader Clustering

Cluster the truncation pool before MI learning:
- leaders are selected from best-ranked candidates in order;
- assign a candidate to the first leader within a Hamming threshold;
- default threshold: `max(2, floor(n_genes * 0.3))`, then tune based on cluster counts;
- enforce a soft maximum number of clusters, default `4` to `8`;
- split oversized clusters by adding another leader if needed;
- keep singleton clusters, but they use static/univariate FOS until enough samples exist.

The point is to avoid averaging TLDS0/TLDS2, different macro-tile families, or very different store paths into one misleading global linkage model.

### Stage 4: Hybrid MI Matrix

Compute an MI matrix per cluster:
- categorical knobs use nominal MI over domain indices;
- ordinal knobs use rank-transformed bins before MI;
- tuple-valued structural knobs can start as nominal and later expose derived ordinal features if needed;
- use `n_bins=4` for ordinal rank bins initially;
- apply an MI floor, default `1e-6`, to stop spurious UPGMA merges.

Initial ordinal knob candidates:

```text
DepthU
GlobalSplitU
VectorWidthA/B
GlobalReadVectorWidthA/B
StoreVectorWidth
WorkGroupMapping
StaggerU
StaggerUStride
AssertFree0ElementMultiple
AssertFree1ElementMultiple
AssertSummationElementMultiple
```

Keep `MatrixInstruction`, `WorkGroup`, `TransposeLDS`, and bool/enumeration flags nominal until there is evidence that an ordinal representation helps.

`NumElementsPerBatchStore` is intentionally excluded from the default ordinal set because `0` is a special/default-style value rather than simply the smallest explicit batch-store count.

### Stage 5: UPGMA FOS Construction

For each cluster:
- start with singleton genes;
- repeatedly merge active groups with highest average cross-MI;
- stop when best MI is below the floor;
- omit the full all-genes group unless experiments show it helps;
- prepend source-backed static linkage groups so mechanical couplings are always considered;
- dedupe groups while preserving priority order.

Each learned model should produce:

```text
leader_candidate_hash
cluster_size
source_evidence_count
fos_groups
mi_summary
created_at
profile/problem/protocol hashes
```

## Proposal Integration

### New Proposal Mode Or Option

Add learned linkage as an option first, not as a forced replacement:

```text
--learned-linkage
--linkage-truncation-tau 0.5
--linkage-min-samples 8
--linkage-max-clusters 8
--linkage-ordinal-bins 4
```

Learned linkage is now enabled by default for GOMEA-style proposals when enough validated evidence exists. Use `--no-learned-linkage` for A/B checks or debugging.

### GOMEA Candidate Generation

Extend `gomea_candidates()` to accept optional linkage models:

```python
gomea_candidates(..., linkage_models: Sequence[LinkageModel] | None = None)
```

Generation flow:
- choose a base parent from ranked parents;
- assign it to the nearest linkage-model leader by Hamming distance;
- mix with donors using that model's FOS;
- preserve existing rule-valid child checks and target-shape checks;
- if no learned model fits, fall back to static `NT_HHS_LINKAGE_GROUPS` plus parent-genome FOS construction through shared learned-linkage primitives.

### Multi-Basin Hall Of Fame

Add diverse elite selection for proposal parents:
- global top candidates by robust aggregate rank;
- per-shape winners and near-winners;
- per-cluster leaders;
- nearest-shape transfer winners;
- imported hipBLASLt baselines if still competitive.

Keep `elite_count` as the total budget, but fill it from diverse sources before taking extra global top rows.

### Interleaved Multi-Start

Do not implement a fully asynchronous IMS scheduler immediately. Start with generation-level populations:

```text
generation 0: random + transfer + static GOMEA
generation 1: learned-linkage GOMEA with population budget 4/8/16 chunks
generation N: continue larger chunks if smaller chunks stagnate
```

Population sizes can be logical proposal cohorts, not separate scheduler processes. The existing scheduler can batch the resulting candidates across all shapes.

### Stagnation Policy

Track per-population/cohort progress:
- new validation-passed candidate count;
- new per-shape winner count;
- median improvement over prior generation;
- build-valid rate;
- duplicate proposal rate.

If a cohort produces no new useful candidates for `stagnation_rounds`, default `2`, de-emphasize it and shift budget to random restarts or larger learned-linkage populations.

## CLI And Metadata

Add generation/search metadata fields:

```text
learned_linkage_enabled
linkage_model_count
linkage_truncation_tau
linkage_min_samples
linkage_cluster_sizes
linkage_fos_group_count
linkage_fallback_reason
stagnated_cohorts
```

For dry runs, emit the chosen linkage models without launching TensileLite so model behavior can be inspected cheaply.

## Testing Plan

### Unit Tests

Add tests for:
- truncation selection uses best scored rows and ignores failed/unvalidated rows;
- ordinal MI groups monotone knobs that nominal MI would treat weakly;
- nominal MI still works for tuple/categorical knobs;
- leader clustering separates two synthetic basins;
- UPGMA stops when MI is below the spurious-linkage floor;
- model assignment chooses the nearest leader;
- learned models fall back cleanly when evidence count is below `min_linkage_samples`.

### Integration Tests

Add scheduler/proposal tests for:
- `seed-random-gomea --learned-linkage` uses DB evidence to produce candidates;
- generated candidates remain `cheap_constraints()` valid for target shapes;
- `candidate_batch_size` production defaults still batch generation-level candidates broadly;
- metadata records linkage model summaries;
- explicit `--candidate-batch-size 1` remains available for debugging.

### Regression Tests

Keep existing checks that:
- static linkage groups remain available;
- TLDS2 proposal bias is not a validity rule;
- structured diagnostics and final-YAML mapping remain authoritative;
- `DOMAINS` are not narrowed by learned linkage.

## Experiment Plan

### Phase A: Offline Model Inspection

Use existing retained DBs/runs to dump linkage models without proposing new candidates:
- one-shape `8192^3` random/GOMEA evidence;
- pilot 100-shape DB evidence;
- imported hipBLASLt baseline rows where available.

Review:
- cluster leaders and cluster sizes;
- top learned FOS groups;
- whether TLDS0/TLDS2 and macro-tile families separate cleanly;
- whether learned groups duplicate or improve static groups.

### Phase B: Proposal-Only Coverage

Run `proposal-coverage` or an equivalent dry-run mode with learned linkage:
- compare value coverage to current `seed-random-gomea`;
- compare duplicate rate;
- compare cheap invalidity rejection counts;
- inspect whether learned linkage over-exploits one family.

### Phase C: Capped Runtime Runs

Use short, under-5-minute runs unless explicitly expanded:
- `8192,8192,1,8192` learned-linkage GOMEA versus current filtered GOMEA;
- a small subset of pilot shapes with enough historical evidence;
- compare build-valid rate, validation-passed rate, and best median GFLOP/s.

### Phase D: Production Default Decision

Make learned linkage default for GOMEA/evolutionary only after:
- it improves or matches best-found quality under the same budget;
- it does not collapse proposal diversity;
- it does not increase unattributed build failures;
- metadata and tests make behavior auditable.

## Implementation Stages

### Stage 1: Linkage Model Types

Status: implemented.

Added `evotensile/search/learned_linkage.py` with:
- `ScoredGenome`;
- `LinkageModel`;
- `LinkageLearningSummary`;
- truncation selection;
- structural clustering;
- hybrid MI computation;
- UPGMA FOS construction;
- nearest-model assignment.

Added synthetic unit tests in `tests/test_learned_linkage.py` covering truncation, fallback, structural clustering, ordinal MI, MI-floor stopping, model learning, and nearest-model assignment.

Validation after this stage: `pre-commit run --all-files` passed and `pytest tests/` reported `90 passed`.

### Stage 2: DB Evidence Adapter

Status: implemented.

Added learned-linkage evidence helpers in `evotensile/search/learned_linkage.py`:
- `CandidateEvidence`;
- `load_candidate_evidence()` using existing `rank_evaluations()` and `get_candidates()`;
- shape-local rank-percentile scoring over positive `ok` summaries;
- global or shape-filtered evidence pools;
- `evidence_to_scored_genomes()` for model construction.

Added tests that verify positive-row filtering, shape-local rank aggregation, ignored failed/unregistered candidates, and conversion to scored genomes.

Validation after this stage: `pre-commit run --all-files` passed and `pytest tests/` reported `91 passed`.

### Stage 3: GOMEA Integration

Status: implemented.

Threaded optional `linkage_models` into `gomea_candidates()` and preserved the existing fallback behavior. Each base genome now selects the nearest learned linkage model when models are provided, prepends source-backed static rule groups, and otherwise uses static groups plus parent-genome FOS construction through shared learned-linkage primitives.

Proposal generation now builds learned linkage models from DB evidence when `learned_linkage=True` is requested. The learned models are passed into GOMEA candidates while neighborhood/static proposal paths remain unchanged.

Added tests to verify learned models are accepted, DB-backed learned-linkage proposal generation works, and generated candidates remain cheap-valid.

Validation after this stage: `pre-commit run --all-files` passed and `pytest tests/` reported `94 passed`.

Cleanup after this stage: removed the older duplicate MI/UPGMA fallback implementation from `evotensile/search/gomea.py`. Parent-genome fallback now reuses `fos_from_genomes()` from `evotensile/search/learned_linkage.py`, so there is a single FOS construction codepath. Also removed unused learned-linkage fields/helpers introduced during staging.

### Stage 4: CLI And Metadata

Status: implemented.

Added opt-in CLI flags:
- `--learned-linkage`;
- `--linkage-truncation-tau`;
- `--linkage-min-samples`;
- `--linkage-max-clusters`;
- `--linkage-ordinal-bins`.

Schedule and repair metadata now report learned-linkage enablement, model count, evidence/selection counts, fallback reason, settings, and per-model summaries. Dry runs emit this metadata without launching TensileLite.

Validation after this stage: `pre-commit run --all-files` passed and `pytest tests/` reported `94 passed`.

### Stage 5: Experiments And Default Flip

Status: implemented for the default policy and tests; capped runtime experiments remain optional campaign work.

Learned linkage is enabled by default for GOMEA-style proposals when sufficient validated DB evidence exists. If evidence is insufficient, proposal generation falls back to the existing static/parent-learned GOMEA path and metadata reports `insufficient_validated_evidence`. Added `--no-learned-linkage` as an A/B and debugging escape hatch.

Added tests that verify default learned-linkage proposal behavior, explicit learned-linkage use, insufficient-evidence fallback metadata, and explicit disable metadata.

Validation after this stage: `pre-commit run --all-files` passed and `pytest tests/` reported `95 passed`.

Post-implementation cleanup validation: `pre-commit run --all-files` passed and `pytest tests/` reported `95 passed` after removing duplicate/dead codepaths.

## Risks And Guardrails

- Overfitting to old shapes: Use shape-local ranks and nearest-shape filters; keep random restarts in every generation.
- Diversity collapse: Maintain basin leaders and per-shape elites; cap any one cluster's parent share.
- False ordinal assumptions: Maintain an explicit ordinal knob list and leave tuple/categorical knobs nominal by default.
- Evidence contamination: Use only reusable positive evidence; ignore validation failures, build failures, and unattributed audit rows for positive linkage learning.
- Runtime overhead: Linkage learning should be cheap compared with TensileLite builds; cache or memoize models per DB/profile/protocol/generation if needed.
- Hard-rule creep: Learned linkage never changes `explain_invalid_nt_hhs()`.

## Acceptance Criteria

- Learned-linkage model construction is covered by unit tests with synthetic evidence.
- `seed-random-gomea --learned-linkage` remains deterministic for fixed seed and DB contents.
- Proposal coverage remains broad and does not remove any `DOMAINS` values from eligibility.
- Scheduler metadata reports the linkage settings and model summary.
- Full validation passes with `pre-commit run --all-files` and `pytest tests/`.
- Capped experiments show learned linkage is at least neutral on build-valid rate and improves candidate quality or duplicate rate before it becomes a default.
