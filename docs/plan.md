# EvoTensile Live Plan

EvoTensile is a smart-search autotuner for TensileLite. The implemented pilot target is gfx1151 FP16 NT HHS GridBased GEMM tuning for the 100-shape grid in the `gfx1151-nt-hhs` profile.

This file is the live project log and forward plan. Stable design details live in focused docs:
- `docs/nt_hhs_search_space.md`: how the NT HHS candidate search space is constructed.
- `docs/search_algorithms.md`: general search loop and proposal modes.
- `docs/gomea.md`: GOMEA proposal mechanics.
- `docs/linkage_learning.md`: learned linkage mechanics.
- `docs/noisy_measurements.md`: adaptive timing and noisy winner-selection math.
- `docs/tensilelite_measurement.md`: TensileLite YAML/build/runner/JSONL measurement contract.
- `docs/outlier_repair.md`: local outlier detection and repair math.
- `docs/database.md`: SQLite schema, ranking, cache, and validation semantics.

## Current Status

Implemented and working:
- `gfx1151-nt-hhs` target profile with FP16 NT HHS problem type and the 100-shape pilot grid.
- Broad NT HHS candidate construction with explicit linked repairs and explainable invalidity rules.
- Candidate emission through complete TensileLite `Groups`, not Cartesian-product fork parameters.
- Structured scheduler path only: YAML + manifest generation, TensileLite build-only codegen, final-YAML mapping, external structured runner, and direct SQLite ingestion.
- Cache-aware exact-pair batch planning keyed by problem type hash, benchmark protocol hash, shape id, and candidate hash.
- Random, local mutation, categorical DE, GOMEA, learned-linkage GOMEA, transfer seeding, and imported hipBLASLt baseline participation.
- Adaptive finalist top-ups with validation-gated timing-only reruns.
- `repair-outliers` neighbor-seeded second-stage search.
- DB-driven hipBLASLt GridBased YAML update helper for HHS/HHS+AuxH/BBS/BBS+AuxB variants.
- Installed-library verification helper using `hipblaslt-bench --verify`.

Current default workflow:
- Import current hipBLASLt-selected baselines into the campaign DB.
- Run `schedule-batches` with the target profile and structured runner.
- Let adaptive sampling top up plausible finalists.
- Run `repair-outliers` before final GridBased update when expanding or retuning a grid.
- Update hipBLASLt GridBased logic directly from the DB.
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
- `1,000 ok` structured samples under benchmark protocol hash `bproto_d8085f528519ae64`.
- Baselines now compete as normal DB candidates and can become proposal parents, transfer seeds, or final winners.

### Rebuilt hipBLASLt Validation

Updated checked-in GridBased YAMLs from the DB, rebuilt hipBLASLt for `gfx1151`, installed into `$ROCM_PATH`, and validated the installed runtime:
- PyTorch-level benchmark used `~/ComfyUI-FeatherOps/benchmark_mm_hipblaslt_fp16.py` with `TORCH_BLAS_PREFER_HIPBLASLT=1`.
- The `1024^3` NT path improved over the TheRock issue baseline: `torch_mm_NT` `16.007 -> 23.434 TFLOP/s`, `torch_linear_NT` `15.998 -> 23.465 TFLOP/s`, and direct `hipblaslt_NT` `14.417 -> 25.554 TFLOP/s`.
- Larger direct `hipblaslt_NT` square cases improved by `1.829x` at `2048`, `2.218x` at `4096`, and `4.804x` at `8192` versus the issue baseline.
- Lightweight installed correctness via `scripts/verify_installed_hipblaslt.py` passed `6/6` curated target/off-grid cases.
- Upstream `hipblaslt-test` validation passed when excluding known no-solution availability cases in FP16/BF16 NT edge/skinny and `k=0` families.

### Structured Runner Refactor

The production scheduler now uses only the structured runner path:
- `evotensile/profile.py` and `evotensile/protocol.py` define profile/protocol objects and hashes.
- `evotensile/structured_runner.py` maps final-YAML accepted solutions to exact `(shape_id, candidate_hash)` pairs and validates JSONL rows before DB insertion.
- `csrc/structured_runner.cpp` implements the narrow production backend for current gfx1151 FP16 NT HHS bias + `scaleAlpha_vector` timing.
- Tests use fake external runner scripts and fake TensileLite build outputs instead of an in-process backend.
- Real generated-library checks passed for one-pair and small multi-pair runs, including validation of accepted/rejected mapping.

## Historical One-Shape Reproduction Context

A one-shape harness under `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/evotensile_one_shape/` reproduced the documented `8192^3` winner with hindsight-directed local refinement. That operator is historical evidence only and is not part of the generic scheduler because it bakes in the known winner neighborhood.

Key retained facts:
- Plain random/local control: `12` random + `8` local mutations did not generate `cand_4bde2d3af447f757`.
- Best non-control generated candidate reached `34976.1 GFLOP/s` hot-loop median.
- Documented winner reached `46698.1 GFLOP/s` hot-loop median.
- Hindsight-directed refinement generated the documented winner after `34` benchmarked candidates in that run.
- Cool-loop screening ranked a sibling first, while hot-loop retime restored the documented winner, reinforcing the current hot-loop protocol choice.

## Build And Runtime Conventions

Use these build directories for current work:
- `~/rocm-libraries/build/hipblaslt/` for the normal `~/rocm-libraries/build_hipblaslt.sh` build tree.
- `~/rocm-libraries/build/hipblaslt-bench/` for `hipblaslt-bench`, `hipblaslt-test`, speed comparisons, and installed correctness checks.
- Override `BUILD_DIR` only when comparing versions or preserving a specific historical tree.

Runtime validation should point `HIPBLASLT_TENSILE_LIBPATH` at the installed gfx target library path when the Python/runtime package might otherwise use stale packaged assets.

## Future Plan

Near-term:
- Run larger or finer NT HHS grids with the structured scheduler, adaptive sampling, and `repair-outliers` as the standard loop.
- Compare learned-linkage enabled/disabled runs on the same DB snapshots to quantify proposal value.
- Add more audit scripts for DB-level winner sensitivity, sample-count sensitivity, and repair effectiveness.
- Broaden installed-library correctness cases beyond the six curated verifier cases.

Medium-term:
- Extend structured runner/profile coverage beyond the current gfx1151 FP16 NT HHS epilogue path.
- Add target profiles for additional layouts, data types, and epilogue variants only after backend validation exists.
- Improve failure attribution and negative-cache reporting for new TensileLite rejection families.
- Add explicit migration/version handling if the SQLite schema grows beyond additive changes.

Longer-term:
- Implement surrogate/LFBO-style proposal ranking once enough validated DB evidence exists.
- Add cross-grid transfer workflows for production-size shape sets.
- Evaluate cold-loop or first-request latency separately if that becomes a product requirement.
- Decide whether generalized structured-runner support should move closer to TensileLite upstream APIs.
