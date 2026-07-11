# EvoTensile Live Plan

EvoTensile is a smart-search autotuner for TensileLite. The implemented pilot target is gfx1151 FP16 NT HHS GridBased GEMM tuning for the 100-shape grid in the `gfx1151-nt-hhs` profile.

This file is the live project log and forward plan. Stable design details live in focused docs. The current one-shape and production-grid readiness audit is in `docs/one_shape_workflow_review.md`.

## Current Status

Implemented and working:
- `gfx1151-nt-hhs` target profile with FP16 NT HHS problem type and the 100-shape pilot grid.
- Broad NT HHS candidate construction with explicit linked repairs and explainable invalidity rules.
- Candidate emission through complete TensileLite `Groups`, not Cartesian-product fork parameters.
- Structured scheduler path only: YAML + manifest generation, parallel build/map/diagnostic/validation preparation, a hard barrier, serial benchmark-only execution, and direct SQLite ingestion.
- Separate correctness and timing identities: validation evidence is stored independently from benchmark samples.
- Cache-aware exact-pair planning keyed by problem type, benchmark protocol, validation protocol, shape, and candidate, with latest-compatible correctness-state resolution.
- Random, local and semantic mutation, categorical DE, GOMEA, learned-linkage GOMEA, family-QD proposals, adaptive operator allocation, transfer seeding, and imported hipBLASLt baseline participation.
- Optional ExtraTrees shortlisting and mechanical covering from oversized proposal pools using validation-passed DB evidence or evidence-free soft mechanics.
- Adaptive finalist top-ups, screening stabilization, and strict hot confirmation that reuse content-verified registered build artifacts without recompilation or revalidation.
- Optional cost-aware operator credit and longest-predicted-work-first preparation ordering.
- A deterministic blind one-shape campaign harness with two isolated cold islands, later migration, exact-proposal checkpoints, and opt-in convergence stopping.
- `repair-outliers` neighbor-seeded second-stage search.
- DB-driven hipBLASLt GridBased YAML preview/output helper for HHS/HHS+AuxH/BBS/BBS+AuxB variants, requiring complete-profile shape coverage, current validation, and complete registered artifacts before explicit source overwrite.
- Installed-library verification helper using `hipblaslt-bench --verify`.

Current default workflow:
- Import current hipBLASLt-selected baselines into the campaign DB.
- Run `schedule-batches` with the target profile and structured runner.
- Let adaptive sampling top up plausible finalists.
- Run `repair-outliers` before final GridBased update when expanding or retuning a grid.
- Preview and write complete hipBLASLt GridBased logic from the DB, review it, then explicitly request source overwrite.
- Rebuild/install hipBLASLt and validate performance/correctness.

## Recent Project Log

### 100-Shape Pilot Search

Completed the first gfx1151 FP16 NT HHS 100-shape pilot using the former standalone TensileLite client path:
- Planned `135` proposed candidates across `100` shapes, for `13,500` candidate-shape pairs in `5` batches.
- Wall time was `1265.21s` (`21.1 min`).
- Repaired ingestion produced `75,000 ok`, `2,000 validation_fail`, and `5,800 rejected` rows.
- Effective accepted coverage was `7,500` ok pairs and `200` validation-failed pairs, with `10` samples per ok pair.
- Summed recorded ok GEMM time was `17.178s`. The rest was compile/client/log/validation/database overhead.

### Historical Top-4 Retiming

A fixed top-4 full-validation retime was run before adaptive sampling existed:
- Coverage was `400` intended pairs: top 4 per shape.
- Result rows: `4,000 ok` samples, `0` rejected/unmapped/validation-fail.
- Wall time was `675.86s`, again dominated by generic TensileLite overhead.
- Retiming changed `57` of `100` per-shape winners versus the first-pass screen.
- The final winner's first-pass rank was `1` for `43` shapes, `2` for `27`, `3` for `17`, and `4` for `13`.

This result motivated integrated adaptive sampling, which is now the default scheduler behavior.

### Current hipBLASLt Baseline Import

Imported current hipBLASLt-selected configs into `out/grid100_full_20260618_repaired.sqlite`:
- `100` queried shapes.
- `22` unique installed candidates.
- `1,000 ok` structured samples, now retained under current benchmark protocol hash `bproto_9f4055f5f13232a3` with the complete 100-shape corpus.
- Baselines now compete as normal DB candidates and can become proposal parents, transfer seeds, or final winners.

### 100-Shape Protocol Consolidation

The complete retained corpus was migrated through the proven role-only identity change and consolidated in `out/grid100_full_20260618_repaired.sqlite`:
- `165,604 ok`, `6,616 rejected`, and `2,000` historical validation-failure audit rows remain under current benchmark protocol hash `bproto_9f4055f5f13232a3`.
- Source and destination row multisets matched exactly before the obsolete identity was removed.
- Before deleting historical run trees, their DB contributions were verified: `4,000 ok` top-four retiming rows, `15,204 ok` plus `816 rejected` outlier-repair rows, and `1,000 ok` installed-baseline rows.
- No current validation evidence was synthesized. Future production promotion still requires the current validation protocol.

### Rebuilt hipBLASLt Validation

Updated checked-in GridBased YAMLs from the DB, rebuilt hipBLASLt for `gfx1151`, installed into `$ROCM_PATH`, and validated the installed runtime:
- PyTorch-level benchmark used `~/ComfyUI-FeatherOps/benchmark_mm_hipblaslt_fp16.py` with `TORCH_BLAS_PREFER_HIPBLASLT=1`.
- The `1024^3` NT path improved over the TheRock issue baseline: `torch_mm_NT` `16.007 -> 23.434 TFLOP/s`, `torch_linear_NT` `15.998 -> 23.465 TFLOP/s`, and direct `hipblaslt_NT` `14.417 -> 25.554 TFLOP/s`.
- Larger direct `hipblaslt_NT` square cases improved by `1.829x` at `2048`, `2.218x` at `4096`, and `4.804x` at `8192` versus the issue baseline.
- Lightweight installed correctness via `scripts/verify_installed_hipblaslt.py` passed `6/6` curated target/off-grid cases.
- Upstream `hipblaslt-test` validation passed when excluding known no-solution availability cases in FP16/BF16 NT edge/skinny and `k=0` families.

### Structured Runner And Phase Queues

The production scheduler uses two explicit queues:
- Parallel preparation performs TensileLite build/codegen, final-YAML mapping and salvage, diagnostics, and correctness verification.
- A hard worker-pool barrier completes before the serial benchmark queue starts.
- `csrc/structured_runner.cpp` exposes strict `validate` and `benchmark` modes and enforces the machine-wide shared/exclusive APU gate itself.
- Validation mode emits no timing. Benchmark mode requires validation disabled and performs no correctness work.
- Adaptive top-ups benchmark subsets from the original prepared artifacts.
- Tests assert compiler/validator completion before timing, serial benchmark execution, and no adaptive recompilation/revalidation.
- A real generated-library check passed hipBLASLt GPU validation followed by benchmark-only timing from the same code object.

## Blind Search Experiments

Blind one-shape baselines, simulated policy selection, real 20-minute campaigns, unsuccessful diagnostics, utilization evidence, and the completed family-aware follow-up experiment are recorded in `docs/blind_one_shape_experiment.md`.

## Build And Runtime Conventions

Use these build directories for current work:
- `~/rocm-libraries/build/hipblaslt/` for the normal `~/rocm-libraries/build_hipblaslt.sh` build tree.
- `~/rocm-libraries/build/hipblaslt-bench/` for `hipblaslt-bench`, `hipblaslt-test`, speed comparisons, and installed correctness checks.
- Override `BUILD_DIR` only when comparing versions or preserving a specific historical tree.

Runtime validation should point `HIPBLASLT_TENSILE_LIBPATH` at the installed gfx target library path when the Python/runtime package might otherwise use stale packaged assets.

## Next Production Search Plan

The completed blind one-shape follow-up is recorded in `docs/blind_one_shape_experiment.md`. This section focuses on ideas whose main benefit appears only when searching many shapes, allocating a production workload budget, or building a compact final solution library. The ideas adapt useful signals from Ductile and GEKO while keeping EvoTensile layered directly over TensileLite.

### Grid-Aware Acquisition

Multi-shape surrogate acquisition should compare every candidate with the incumbent for each shape rather than averaging predicted log time across all target shapes. The shortlist should greedily maximize marginal predicted improvement over unresolved shapes and preserve separate lanes for:
- per-shape and per-cluster specialists.
- workload-weighted generalists.
- uncertainty and poorly modeled shapes.
- family, mechanical, and random diversity.

Equal weights should remain the default for an authoritative fixed grid. Optional production weights may come from shape frequency, baseline latency, or measured end-to-end contribution. A candidate that is exceptional for a small subset of shapes must not be hidden by poor grid-wide average performance.

Candidate-shape interaction features should include tile fill, total tiles, tiles per effective CU, CU-round and wave granularity, K-tail behavior, LDS and VGPR pressure, GSU workspace fraction, and arithmetic intensity. Architecture-specific MI filters and hand-selected parameter lists may inform soft priors, but must not become permanent validity constraints.

### Staged Shape Evaluation

Production shapes should be clustered by mechanical behavior rather than only raw dimension distance. Clustering inputs should include aspect, batch and K regimes, arithmetic intensity, tile efficiency across macro-tile families, CU-round behavior, and compatible coarse kernel families.

The staged search should:
- choose representative or medoid shapes for each cluster.
- search representatives first using the normal family-QD, operator portfolio, and surrogate flow.
- promote promising specialists and generalists to the remaining cluster members.
- migrate candidates between mechanically adjacent clusters.
- finish with full-grid outlier detection and `repair-outliers` so every authoritative shape is covered.

Compilation artifacts should remain candidate-centric and reusable across promoted shapes. Representative-shape screening reduces candidate-shape measurements. It must not infer an unmeasured production winner.

### Workload And Cost Allocation

For application-derived workloads, shape priority should reflect `call_count * baseline_latency`, predicted improvement headroom, uncertainty, and expected evaluation cost. Low-contribution shapes may be deferred or omitted only in an explicitly workload-weighted mode. A fixed GridBased coverage campaign must continue to evaluate every required shape.

Preparation cost should be predicted from observed build, validation, candidate-count, and resource-complexity history. Parallel preparation should use longest-processing-time-first ordering to reduce the compile/validation barrier. Serialized benchmark work should be ordered by expected improvement or information per second, and campaign admission should reserve enough measured time for finalist confirmation and outlier repair.

### Specialist And Generalist Bank

The production archive should track both shape specialists and candidates that remain within a configurable tolerance of the winner across many shapes. After confidence-aware retiming, an optional greedy set-cover pass should choose the smallest solution bank that covers all required shapes within the requested tolerance.

A zero tolerance must preserve exact per-shape winners. Nonzero tolerance is a deployment trade-off and should report:
- weighted and worst-shape performance loss.
- number of retained solutions and code objects.
- shapes covered by each generalist.
- any shapes requiring dedicated specialists.

The selected candidate-shape pairs must be rechecked through the normal validation and final measurement path before GridBased YAML generation.

### Production Evaluation

Grid policies should be compared at equal wall time and report every required shape. Primary metrics should include:
- weighted and unweighted incumbent-normalized regret.
- worst-shape regret and unresolved-shape count over time.
- candidate-shape pairs prepared, validated, probed, screened, and confirmed.
- compile/validation barrier time and serialized benchmark time.
- representative-to-cluster promotion precision and missed specialists.
- unique final solutions and solution-bank coverage at each tolerance.

Controlled ablations should cover grid-aware acquisition, shape clustering, workload weighting, cost-aware scheduling, and solution-bank minimization before they become production defaults.

## Broader Future Plan

Near-term:
- Run larger or finer NT HHS grids with the staged structured workflow, adaptive sampling, surrogate shortlisting, and `repair-outliers` as the standard loop.
- Compare learned-linkage, surrogate, clustering, and acquisition features with fixed DB snapshots and equal wall-time budgets.
- Add audit scripts for DB-level winner sensitivity, sample-count sensitivity, representative-shape promotion, and repair effectiveness.
- Broaden installed-library correctness cases beyond the six curated verifier cases.

Medium-term:
- Extend structured runner/profile coverage beyond the current gfx1151 FP16 NT HHS epilogue path.
- Add target profiles for additional layouts, data types, and epilogue variants only after backend validation exists.
- Improve failure attribution and negative-cache reporting for new TensileLite rejection families.
- Add explicit migration/version handling if the SQLite schema grows beyond additive changes.

Longer-term:
- Evaluate LFBO or persistent transfer surrogates beyond the current per-campaign ExtraTrees model.
- Add cross-grid transfer workflows for production-size shape sets.
- Evaluate cold-loop or first-request latency separately if that becomes a product requirement.
- Decide whether generalized structured-runner support should move closer to TensileLite upstream APIs.
