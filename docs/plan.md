# EvoTensile Live Plan

This file summarizes repository-level status and broader direction. Experiment histories and experiment-specific plans live in `docs/experiment_*.md`. Stable subsystem design lives in the other focused documents under `docs/`.

## Current Status

Implemented and working:
- `gfx1151-nt-hhs` target profile with FP16 NT HHS problem type and the 100-shape pilot grid.
- Broad NT HHS candidate construction with explicit linked repairs and explainable invalidity rules.
- Candidate emission through complete TensileLite `Groups`, not Cartesian-product fork parameters.
- Structured scheduler path only: YAML + manifest generation, parallel build/map/diagnostic/validation preparation, a hard barrier, serial benchmark-only execution, and direct SQLite ingestion.
- Separate correctness and timing identities: validation evidence is stored independently from benchmark samples.
- Cache-aware exact-pair planning keyed by problem type, benchmark protocol, validation protocol, shape, and candidate, with latest-compatible correctness-state resolution.
- Random, local and semantic mutation, categorical DE, GOMEA, learned-linkage GOMEA, family-QD proposals, adaptive operator allocation, transfer seeding, and installed hipBLASLt discovery followed by normal scheduled evidence.
- Optional ExtraTrees shortlisting and mechanical covering from oversized proposal pools using validation-passed DB evidence or evidence-free soft mechanics.
- Adaptive finalist top-ups, screening stabilization, and strict hot confirmation that reuse content-verified registered build artifacts without recompilation or revalidation.
- Optional cost-aware operator credit and longest-predicted-work-first preparation ordering.
- A deterministic blind one-shape campaign harness with two isolated cold islands, later migration, exact-proposal checkpoints, and opt-in convergence stopping.
- `repair-outliers` neighbor-seeded second-stage search.
- DB-driven hipBLASLt GridBased YAML preview/output helper for HHS/HHS+AuxH/BBS/BBS+AuxB variants, requiring complete-profile shape coverage, current validation, and complete registered artifacts before explicit source overwrite.
- Installed-library verification helper using `hipblaslt-bench --verify`.

Current default workflow:
- Discover current hipBLASLt-selected pairs and schedule them through the normal evidence path.
- Run `schedule-batches` with the target profile and structured runner.
- Let adaptive sampling top up plausible finalists.
- Run `repair-outliers` before final GridBased update when expanding or retuning a grid.
- Preview and write complete hipBLASLt GridBased logic from the DB, review it, then explicitly request source overwrite.
- Rebuild/install hipBLASLt and validate performance/correctness.

## Infrastructure Status

### Structured Runner And Phase Queues

The production scheduler uses two explicit queues:
- Parallel preparation performs TensileLite build/codegen, final-YAML mapping and salvage, diagnostics, and correctness verification.
- A hard worker-pool barrier completes before the serial benchmark queue starts.
- `csrc/structured_runner.cpp` exposes strict `validate` and `benchmark` modes and enforces the machine-wide shared/exclusive APU gate itself.
- Validation mode emits no timing. Benchmark mode requires validation disabled and performs no correctness work.
- Adaptive top-ups benchmark subsets from the original prepared artifacts.
- Tests assert compiler/validator completion before timing, serial benchmark execution, and no adaptive recompilation/revalidation.
- A real generated-library check passed hipBLASLt GPU validation followed by benchmark-only timing from the same code object.

## Build And Runtime Conventions

Use these build directories for current work:
- `~/rocm-libraries/build/hipblaslt/` for the normal `~/rocm-libraries/build_hipblaslt.sh` build tree.
- `~/rocm-libraries/build/hipblaslt-bench/` for `hipblaslt-bench`, `hipblaslt-test`, speed comparisons, and installed correctness checks.
- Override `BUILD_DIR` only when comparing versions or preserving a specific historical tree.

Runtime validation should point `HIPBLASLT_TENSILE_LIBPATH` at the installed gfx target library path when the Python/runtime package might otherwise use stale packaged assets.

## Broader Future Plan

Near-term:
- Use the 100-shape experiment to decide which multi-shape mechanisms graduate into the general production workflow.
- Apply validated multi-shape policies to larger or finer NT HHS grids only after the pilot ablations complete.
- Broaden installed-library correctness cases beyond the six curated verifier cases.

Medium-term:
- Extend structured runner/profile coverage beyond the current gfx1151 FP16 NT HHS epilogue path.
- Add target profiles for additional layouts, data types, and epilogue variants only after backend validation exists.
- Improve failure attribution and negative-cache reporting for new TensileLite rejection families.
- Keep checked-out code and authoritative databases synchronized when the SQLite schema changes.

Longer-term:
- Evaluate LFBO or persistent transfer surrogates beyond the current per-campaign ExtraTrees model.
- Add cross-grid transfer workflows for production-size shape sets.
- Evaluate cold-loop or first-request latency separately if that becomes a product requirement.
- Decide whether generalized structured-runner support should move closer to TensileLite upstream APIs.
