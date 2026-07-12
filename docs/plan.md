# EvoTensile Live Plan

This file summarizes repository-level status and broader direction. Experiment histories and experiment-specific plans live in `docs/experiment_*.md`. Stable subsystem design lives in the other focused documents under `docs/`.

## Current Status

Implemented and working:
- `gfx1151-nt-hhs` target profile with FP16 NT HHS problem type and the 100-shape pilot grid.
- Broad NT HHS candidate construction with explicit linked repairs and explainable invalidity rules.
- Candidate emission through complete TensileLite `Groups`, not Cartesian-product fork parameters.
- Structured exact-pair scheduler path only: explicit pair requests, candidate-centric artifact scopes, exact manifests, parallel build/map/diagnostic/validation preparation, a hard barrier, serial benchmark-only execution, and direct SQLite ingestion.
- Separate correctness and timing identities: validation evidence is stored independently from benchmark samples.
- Cache-aware exact-pair planning keyed by problem type, benchmark protocol, validation protocol, shape, and candidate, with explicit artifact-shape scopes and latest-compatible correctness-state resolution.
- Random, local and semantic mutation, categorical DE, GOMEA, learned-linkage GOMEA, family-QD proposals, adaptive operator allocation, transfer seeding, and installed hipBLASLt discovery followed by normal scheduled evidence.
- Optional ExtraTrees shortlisting and mechanical covering from oversized proposal pools using validation-passed DB evidence or evidence-free soft mechanics.
- Adaptive finalist top-ups and diagnostic singleton hot ranking that reuse registered artifacts, plus a production deployment path that stabilizes posterior-close finalists and deliberately revalidates and remeasures every selected exact pair.
- Optional cost-aware operator credit and longest-predicted-work-first preparation ordering.
- A deterministic blind singleton campaign profile running through the shared controller and real evaluator, with two isolated cold islands, later migration, exact-proposal checkpoints, and opt-in convergence stopping.
- Integrated weak-shape repair using capped evidence deficits, candidate-specific posterior close probability, shared-cost exact bundles, and an explicit staged reserve.
- Explicit deployment solution-bank selection and DB-driven hipBLASLt GridBased YAML preview/output for HHS/HHS+AuxH/BBS/BBS+AuxB variants, requiring complete-profile assignment coverage, confirmation timing, current validation, and complete registered artifacts before explicit source overwrite.
- Installed-library verification helper using `hipblaslt-bench --verify`.

Current default workflow:
- Discover current hipBLASLt-selected pairs and schedule them through the normal evidence path.
- Use the shared replay, real, or hybrid exact-pair evaluator with a fresh/labeled overlay for controller-driven experiments.
- Partition multi-shape workloads with persisted deterministic mechanical medoids. Compare independent, dense, and representative-only exact-pair baselines.
- Race exact nearest, representative, specialist, and adjacent-cluster broad promotions through shared artifact bundles and staged probe/main evidence.
- Fit a disclosed-evidence-only contextual ExtraTrees pair model with normalized performance, validity probability, calibrated uncertainty, and posterior samples.
- Select finite candidate bundles by posterior marginal utility per shared preparation/expansion plus exact pair cost, with separate preparation and timing order.
- Execute reusable five-minute staged rounds with cumulative phase deadlines, exact pending-wave checkpoints, durable replanning, and soft overrun.
- Run `schedule-batches` with the target profile and structured runner.
- Let adaptive sampling top up plausible finalists.
- Spend the staged repair reserve on evidence-supported weak shapes before final confirmation and GridBased selection.
- Preview and write complete hipBLASLt GridBased logic from the DB, review it, then explicitly request source overwrite.
- After explicit approval, rebuild/install hipBLASLt and validate performance/correctness.

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

## Singleton And Multi-Shape Unification Gaps

The lower execution and evidence layers already share exact pair requests, scheduling, controller state, replay/real/hybrid evaluators, contextual pair modeling, workload weighting, and deployment selection. Clustering, promotion, and repair use shared data structures with explicit singleton degenerations to one cluster or no cross-shape work.

The following paths are not yet fully unified:
- `CampaignConfiguration` and `evotensile/campaign/runner.py` remain singleton-oriented: configuration owns one `shape_id`, and `run_campaign()` rejects shape sets larger than one.
- the real singleton runner uses its own island round loop, plateau/restart/convergence handling, dense candidate evaluation, reserve calculation, and leader reporting instead of the generic staged-round planner used by multi-shape policy replay.
- singleton proposal shortlisting explicitly calls `select_singleton_bundle_pool()` after the evidence threshold. It wraps the shared pair model and bundle planner but constructs a separate temporary controller, cost model, fallback costs, planning cap, and singleton acquisition projection.
- singleton cold start uses dedicated mechanical covering, while multi-shape cold start uses unresolved-shape mechanical ordering plus diversity.
- family archive objective selection explicitly keeps only the specialist objective for one shape rather than obtaining that reduction solely from the general multi-objective path.
- the real singleton runner finishes through diagnostic `hot_confirm_topk()` rather than the shared stabilization, model-refit, fresh production confirmation, and deployment-selection path.
- singleton and multi-shape policy parameters live in partially separate schemas: `CampaignConfiguration` controls the real blind runner, while `CampaignPolicyConfiguration` controls clustering, promotion, bundle acquisition, repair, and staged-round policy.

The intended end state is one shape-set campaign runner and one policy schema. A singleton should select the same controller, staged-round, acquisition, evaluator, confirmation, and deployment codepaths with a one-shape hyperparameter preset. Explicit singleton branches should remain only where the mathematical operation is genuinely empty or uniquely normalized, such as no cross-shape promotion, no cross-shape repair, one deterministic cluster, and workload weight one.

## Future Search Intelligence

EvoTensile should grow from configuration-only statistical search toward artifact-aware experimental guidance while keeping generated-code interpretation and source modification outside the autonomous core.

Planned directions:
- add incumbent-centered trust-region searches that freeze a strong structural family and systematically sweep selected parameters, nearby ordered values, and bounded cross-group interactions instead of relying only on random categorical transitions.
- support explicit interaction experiments for hypotheses such as scheduling algorithm * store policy * store batch size, with equal budgets, stable controls, and reports that preserve the conditional context of each result.
- condition operator, semantic-group, and donor credit by structural family and campaign phase when enough evidence exists, while retaining minimum exploration and unconditioned fallback estimates.
- add paired or blocked finalist timing that alternates candidates with a repeated control anchor, records timing-session identity and execution order, and reports within-block speedups to detect drift that pooled historical medians may hide.
- extract machine-readable generated-artifact features from final YAML, code-object metadata, and deterministic assembly/disassembly parsing. Initial features should include actual VGPR, SGPR, LDS, scratch, code-object size, kernel size, occupancy-relevant resource buckets, and counts of broad instruction classes such as global loads/stores, LDS reads/writes, matrix instructions, barriers, waits, branches, and scalar/vector arithmetic.
- retain cheap analytical mechanics beside measured artifact features so reports can expose where predicted VGPR/LDS/cost diverges from generated reality. Artifact features should be keyed by candidate, exact artifact shape scope, target/toolchain identity, and content hash.
- make artifact features available to cost models, contextual surrogates, family diagnostics, local-neighborhood reports, and plateau analysis. They remain ranking and explanation signals, not automatic correctness or validity rules.
- add structured configuration-delta reports around finalists: changed parameters, linked repairs, measured performance, generated resources, instruction-mix deltas, validation status, and shape coverage. These reports should help a human or external AI identify a narrow next experiment without requiring EvoTensile itself to understand assembly semantics.
- detect and report possible TensileLite correctness issues from evidence patterns such as wrong-but-fast candidates, repeated validation failures in a narrowly defined configuration neighborhood, shape-boundary failures, inconsistent generated mappings, and disagreements between expected and generated resource/layout metadata.
- emit self-contained correctness-anomaly bundles containing candidate dictionaries, exact shapes, protocol and environment identities, logs, final YAML, manifests, relevant assembly/disassembly, nearest passing controls, and minimal reproduction commands. Reports may rank suspected correlations but must distinguish observed association from a proven generator defect.
- add plateau reports that separate exhausted configuration neighborhoods from insufficient evidence, timing noise, repeated generation failure, and unexplained generated-artifact changes. A plateau may recommend assembly or TensileLite investigation as a human action, but it must not modify external source trees.

Explicit non-goals and safety boundaries:
- EvoTensile will not perform LLM-like semantic interpretation of generated assembly or claim causal understanding from instruction text.
- EvoTensile will not autonomously edit, patch, rebuild, install, or switch the checked-out TensileLite or hipBLASLt source tree.
- performance correlations, artifact-feature correlations, and recurring failures will not automatically become hard validity constraints. New reusable rules require source-backed justification, focused tests, and human review.
- generated-code reports are evidence packages for maintainers and external analysis. Any source patch, upstream issue, guarded specialization, or host-side selection predicate remains an explicit reviewed change outside the search loop.

## Broader Future Plan

Near-term:
- Review the converged practical 100-shape result in `out/grid100_production_search_20260712/finalization_v4/`. Zero tolerance maximizes freshly confirmed speed with 42 solutions. 0.5%, 1%, and 2% alternatives explicitly trade up to 0.399%, 0.980%, and 1.884% worst-shape loss for 36, 31, and 26 solutions.
- Treat the 47-round practical search, including integrated repair, measured promotion, focused trust regions, and the final staging restart, as complete under its convergence criteria. Resume native configuration search only for new evidence, a changed workload/target/environment, or a focused unresolved correctness/performance hypothesis rather than rerunning the campaign.
- Use an explicit workload when practical call frequencies are available, while retaining unweighted tail and worst-shape reporting.
- Use only the post-convergence fresh `finalization_v4` assignment for production review. Require complete registered artifacts before generating GridBased logic, and rebuild/install only after explicit approval.
- Broaden installed-library correctness cases beyond the six curated verifier cases.
- Apply the validated mechanisms to larger or finer NT HHS grids only after the practical 100-shape deployment is reviewed.

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
