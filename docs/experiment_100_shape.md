# 100-Shape Experiment

This document records the completed evidence, current capabilities, experiment policy, and remaining work for improving the gfx1151 FP16 NT HHS 100-shape GridBased workload. General subsystem design remains in the focused documents under `docs/`. The blind one-shape experiment is recorded separately in `docs/experiment_blind_one_shape.md`.

## Objective

The experiment has two related goals:
- Improve performance across every shape in the `gfx1151-nt-hhs` 100-shape grid.
- Evaluate multi-shape search strategies by exact historical simulation and then by controlled real searches.

The experiment should minimize candidate-shape trials and wall time, not merely maximize the number of generated candidates. Final production results must still contain current validation and measured timing for every selected candidate-shape pair.

## Experiment Policy

This is not a blind experiment. It may import and use:
- all retained 100-shape candidates and measurements.
- current installed hipBLASLt-selected configurations.
- hand-authored and hand-tuned configurations.
- the best guarded SIA3/no-store-priority configuration documented in `~/ComfyUI-FeatherOps/doc/tensile_fp16_nt_hhs.md`.
- known strong candidates as initial parents, transfer seeds, controls, or local-search centers.

Those imports are already represented in the retained corpus. In particular, the hand-tuned `ScheduleIterAlg=3`, `StorePriorityOpt=False`, `NumElementsPerBatchStore=10` configuration is normalized candidate `cand_07ba5e67b99df4ba` and has retained timing evidence on all 100 shapes.

The search strategy itself must remain generic. Known configurations may initialize or guide generic operators, but they must not become hidden validity rules, winner-specific linkage groups, hard-coded parameter bundles in default proposal logic, or special cases that work only because the final answer is known.

The reusable constraints from `docs/experiment_blind_one_shape.md` still apply to strategy evaluation:
- validity comes from source-backed rules and real validation, not performance hindsight.
- unknown simulated candidate-shape results remain unknown rather than imputed from a nearby winner.
- simulated evidence is disclosed query-causally before it affects archive, surrogate, linkage, or operator credit.
- screening evidence is not a final performance claim.
- final claims use validation-gated confirmation evidence.
- performance-derived observations belong in experiment policy and evidence, not generic search-space restrictions.

Unlike the blind experiment, explicit imported candidates and their prior measurements are allowed inputs and must be labeled as such.

## One-Shape And Multi-Shape Search

One-shape search is the one-member special case of multi-shape search. It should use the same candidate representation, proposal operators, evidence model, validation gate, timing policy, and final ranking semantics.

Multi-shape search is not one-shape search repeated independently for every shape. Joint search can reduce expensive trials by exploiting structure that does not exist in an isolated run:
- a candidate measured on one shape can be promoted to mechanically related shapes.
- exact and nearest-shape winners can seed mutation, DE, GOMEA, and family archives elsewhere.
- representative shapes can screen a family before broader promotion.
- specialist and generalist candidates can share one proposal pool without pooling absolute latency across shapes.
- compilation artifacts can be reused when one candidate is evaluated on several shapes.
- uncertainty and expected improvement can allocate measurements across shapes rather than assigning equal independent budgets.
- full-grid outlier detection can identify shapes that need dedicated repair after broad promotion.
- a final solution bank can trade a small bounded performance loss for fewer deployed solutions and code objects.

Every promotion remains a measurement decision. Mechanical similarity may prioritize a candidate-shape pair, but it must never turn an unmeasured pair into a production winner.

## 100-Shape Grid

The profile grid is the Cartesian product defined in `evotensile/shapes.py`:

```text
M:     512, 640, 896, 1024
N:     128, 256, 512, 768, 1024
batch: 1
K:     256, 512, 1024, 2048, 4096
```

This produces 100 exact shapes covering small and large output tiles, skinny and square aspect ratios, and short through long reductions.

## Completed Evidence

### Initial Pilot Search

The first pilot used the former standalone TensileLite client path:
- `135` proposed candidates across `100` shapes.
- `13,500` planned candidate-shape pairs in five batches.
- `1265.21s` wall time.
- `75,000` successful timing samples, corresponding to `7,500` accepted pairs with ten samples each.
- `200` validation-failed pairs and `5,800` rejected observations.
- `17.178s` of summed successful GEMM time. Compilation, client startup, validation, logs, and ingestion dominated wall time.

### Historical Top-Four Retiming

Before adaptive timing existed, the top four screened candidates for every shape were rerun:
- `400` intended candidate-shape pairs.
- `4,000` successful timing samples.
- `675.86s` wall time.
- the winner changed on `57/100` shapes.
- final winners had first-pass ranks 1, 2, 3, and 4 for `43`, `27`, `17`, and `13` shapes respectively.

This result motivated adaptive sampling and confirms that a one-pass top-one screen is not reliable enough for final grid selection.

### Outlier Repair And Imported Baselines

The retained corpus includes:
- `15,204` successful samples and `816` rejections from historical outlier repair.
- current installed hipBLASLt selections for all 100 shapes.
- `22` unique installed hipBLASLt candidates and `1,000` scheduled timing samples.
- hand-tuned candidates, including the guarded SIA3/no-store-priority configuration from the FeatherOps investigation.

Imported candidates compete as ordinary candidates. Their known origin may guide seeding and diagnostics, but final production use still requires the same current validation and timing path as generated candidates.

### Consolidated Oracle Database

The retained corpus is:

```text
out/grid100_full_20260618_repaired.sqlite
```

Current contents include:
- `219` canonical candidates.
- `100` shapes.
- `8,728` successful candidate-shape pairs.
- `165,604` successful timing samples.
- `6,616` retained source-backed rejections.
- `200` historical failed-validation events.
- benchmark protocol hash `bproto_9f4055f5f13232a3`.

The DB uses the current table and index schema. A one-time canonical migration converted legacy integer encodings of `ExpandPointerSwap`, `MIArchVgpr`, and `SourceSwap` to JSON booleans and recomputed `183` candidate hashes with no collisions. Candidate integer IDs and every evidence row were preserved. The complete local hash map is `out/grid100_boolean_migration_20260711.json`.

The corpus intentionally does not synthesize current validation passes for historical timing. It is authoritative as a simulation oracle and imported-candidate catalog. Real production promotion must obtain current compatible validation evidence.

### Reproduction Check

Four historical winners spanning the retained timing range were rebuilt with the current TensileLite checkout, fully validated through the hipBLASLt oracle, and measured with 30 fresh main-protocol samples. The acceptance bound was `max(5%, 3 * historical relative MAD)`.

| Shape | Historical TFLOP/s | Fresh TFLOP/s | Speed error | Bound |
| --- | ---: | ---: | ---: | ---: |
| `m512_n128_b1_k256` | `5.028` | `5.063` | `0.70%` | `5.00%` |
| `m640_n512_b1_k1024` | `19.451` | `19.792` | `1.75%` | `5.00%` |
| `m896_n768_b1_k2048` | `28.711` | `28.744` | `0.12%` | `5.00%` |
| `m1024_n1024_b1_k4096` | `38.672` | `38.353` | `0.83%` | `6.00%` |

All four passed. The scheduler also validated all 16 cross-product pairs between the four candidates and four selected shapes and recorded 480 fresh timing samples. The machine-readable report is `out/grid100_winner_reproduction_20260711/report_normalized.json`.

### Rebuilt hipBLASLt Validation

The GridBased YAMLs were updated from retained winners, hipBLASLt was rebuilt and installed for gfx1151, and the installed runtime was checked:
- PyTorch-level tests used `~/ComfyUI-FeatherOps/benchmark_mm_hipblaslt_fp16.py` with `TORCH_BLAS_PREFER_HIPBLASLT=1`.
- the `1024^3` NT path improved from `16.007` to `23.434 TFLOP/s` for `torch_mm_NT`, from `15.998` to `23.465 TFLOP/s` for `torch_linear_NT`, and from `14.417` to `25.554 TFLOP/s` for direct hipBLASLt NT.
- direct hipBLASLt NT improved by `1.829x`, `2.218x`, and `4.804x` at square sizes `2048`, `4096`, and `8192` relative to the earlier reference.
- installed correctness passed all six curated target/off-grid cases.
- upstream `hipblaslt-test` passed after excluding known no-solution availability families.

## Implemented Multi-Shape Building Blocks

The repository already provides:
- exact proposal scopes for one shape and shape sets.
- shape-local ranking and incumbent-normalized multi-shape parent selection.
- nearest-shape transfer seeds and measured elite reuse.
- family-QD archives with specialist and generalist objectives.
- random, semantic mutation, DE, GOMEA, learned linkage, adaptive operator/group/donor allocation, covering, and surrogate selection.
- candidate-shape mechanical features including tile fill, WGP rounds, reduction depth, LDS/VGPR pressure, and arithmetic intensity.
- query-causal immutable proposal evidence snapshots.
- compile-cache reuse across candidate cohorts.
- separate parallel preparation and serialized validation-gated timing.
- staged probing, screening stabilization, adaptive top-ups, and hot confirmation.
- measured candidate cost attribution and analytical cost-aware preparation ordering.
- neighbor-seeded `repair-outliers` over the complete grid.
- shared one-shape and multi-shape exact-oracle replay state with query-causal DB evidence, exact unknown-pair handling, per-shape incumbents, candidate coverage, and one-time candidate preparation across shapes.
- complete GridBased preview/write checks requiring full shape coverage, current validation, and registered artifacts.

These mechanisms are components. The production multi-shape controller that coordinates them over the entire grid is not yet implemented.

## Simulated Multi-Shape Evaluation

### Oracle Role

Use the migrated retained DB as an exact candidate-shape timing oracle. Simulation may expose all imported candidate parameters because this experiment is non-blind, but timing must still be revealed only when a strategy queries the exact pair.

A missing candidate-shape pair is unknown. Do not fill it with a neighbor's value, a surrogate prediction, or a hidden incumbent. Predictions may choose the next query but may not become oracle answers.

`ExactOracleReplayState` now provides the shared candidate catalog, exact shape-by-candidate matrix, known/unknown/queried/disclosed pair state, query-causal simulated DB, per-shape incumbents and unresolved status, candidate coverage, and separate shared-preparation and pair-timing ledgers. The one-shape simulator uses the same state with one registered shape. Policy drivers still need to implement cross-shape promotion, transfer, budget allocation, regret trajectories, repair reserves, and final-confirmation decisions.

### Policy Comparisons

At equal simulated wall time and candidate-shape query budgets, compare at least:
- Independent one-shape search: no transfer or shared allocation. The required baseline.
- Global candidate evaluation: evaluate selected candidates broadly without mechanical clustering.
- Nearest-shape transfer: seed each shape from measured neighboring winners and near-winners.
- Representative-first clustering: search medoids, then promote candidates within and between nearby clusters.
- Specialist/generalist family-QD: maintain shape specialists plus candidates with broad low regret.
- Joint uncertainty and improvement allocation: choose the next candidate-shape pair across the whole grid.
- Outlier repair: run the same broad policy, then spend a reserved budget on residual weak shapes.
- Combined staged controller: clustering, promotion, joint allocation, and final repair.

Known hipBLASLt and hand-tuned candidates should be available to every policy under the same declared import condition. Ablations must not give one policy a stronger seed catalog than another.

### Simulation Metrics

Report trajectories, not only the final average:
- unweighted and optional workload-weighted incumbent-normalized regret.
- worst-shape regret.
- unresolved-shape count.
- candidate-shape queries and unique candidates prepared.
- simulated preparation, validation, probe, screening, and confirmation time.
- time and queries to reach fixed regret thresholds.
- representative-to-cluster promotion precision and missed specialists.
- gains attributable to transfer and outlier repair.
- final unique solution count and solution-bank coverage at each tolerance.

Simulation evaluates allocation over the known oracle, not the ability to generate genuinely unseen fast candidates. A strategy that succeeds only because all historical candidates are visible must be labeled as candidate-selection evidence rather than end-to-end search evidence.

## Remaining Implementation

### Multi-Shape Campaign State

Implement the policy/controller layer that owns cluster assignments, promotion queues, global and per-phase budgets, regret trajectories, repair and confirmation reserves, and checkpoints. One-shape policy execution should use this controller with one shape rather than a separate algorithm.

### Staged Shape Evaluation

Add a controller that clusters shapes by mechanical behavior rather than only raw dimension distance. Inputs should include aspect ratio, batch/K regimes, arithmetic intensity, tile efficiency across macro-tile families, WGP-round behavior, and compatible coarse kernel families.

The controller should:
- choose representative or medoid shapes.
- search representatives first with the normal family-QD/operator/surrogate flow.
- promote promising specialists and generalists to remaining cluster members.
- migrate candidates between mechanically adjacent clusters.
- retain artifact reuse across promoted shapes.
- finish with complete-grid outlier detection and repair.

### Workload-Aware Allocation And Timing

Add an explicit workload-weighted mode in which shape priority combines call count, baseline latency, predicted improvement headroom, uncertainty, and expected evaluation cost. Apply weights consistently to acquisition, archive and parent selection, operator feedback, timing admission, and reporting.

Fixed GridBased coverage remains a distinct mode and must evaluate every required shape. Workload weighting may defer low-contribution shapes only when explicitly requested.

Replace the analytical preparation heuristic with a measured predictor when enough build and validation history exists. Keep preparation order separate from serialized benchmark order. Benchmark admission should use expected improvement or information per second, unresolved-shape coverage, and soft-deadline fit.

### Deployment Solution Bank

After confidence-aware retiming, add an optional greedy set-cover pass that selects the smallest solution bank covering all required shapes within a requested tolerance.

A zero tolerance must preserve exact per-shape winners. Nonzero tolerances must report weighted and worst-shape loss, retained solution/code-object count, shapes covered by each generalist, and shapes requiring specialists. Revalidate and remeasure every selected pair before GridBased generation.

### Audit And Controlled Evaluation

Add audit tools for:
- winner sensitivity to sample count and timing noise.
- DB-level candidate/hash/shape coverage.
- representative-shape promotion precision.
- transfer and repair effectiveness.
- policy comparison from identical DB snapshots.
- equal-wall-time and equal-pair-budget ablations.

Run controlled ablations for clustering, transfer, joint acquisition, workload weighting, measured cost prediction, timing allocation, outlier repair, and solution-bank minimization before any becomes a production default.

## Real Search Sequence

After simulation identifies robust policies:
- Freeze the imported candidate catalog and experiment snapshot.
- Start fresh real campaign DBs under the current environment tag. Import candidate parameters and provenance without inventing validation evidence.
- Validate and screen imported hipBLASLt, historical, and hand-tuned seeds through the current scheduler.
- Run representative-first and joint multi-shape search with explicit wall-time and pair budgets.
- Promote candidates only through measured candidate-shape pairs.
- Reserve time for finalist stabilization, full-grid outlier repair, and confirmation.
- Recheck every final selected pair with current validation and confirmation evidence.
- Generate complete GridBased logic, rebuild hipBLASLt, and run installed correctness and performance validation.

The final report must distinguish gains from imported candidates, gains from generic search around those candidates, gains from cross-shape transfer, and gains from repair.

## Immediate Next Steps

- Implement the independent one-shape policy baseline on the shared replay state.
- Add nearest-shape transfer and representative-cluster policies over the same oracle snapshot.
- Define equal-budget regret and unresolved-shape reports.
