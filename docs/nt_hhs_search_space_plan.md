# NT HHS Search Space Plan

## Goal

Define the EvoTensile search space for gfx1151 FP16 NT HHS GridBased GEMM tuning as a broad reusable parameter domain plus explicit, explainable invalidity rules.

This plan is intentionally not a list of known-valid NT configs. Known-valid configs are controls, regression cases, and inspiration only. The search should explore the widest practical NT space that TensileLite can compile and validate, while rejecting configurations only when we have a clear reason they are disallowed.

## Principles

- Keep `DOMAINS` broad and reusable across NN/NT/TN/TT unless a value is globally invalid or unencodable.
- Define NT validity as broad-domain values minus explicit disallowed rules.
- Prefer negative rules over positive templates.
- Treat TensileLite as the authoritative validator for accepted, rejected, and normalized solutions.
- Keep shape-dependent invalidity separate from kernel-global invalidity.
- Do not assume NN/TN/TT configs are valid for NT. Use them only as inspiration for values and linkages.
- Do not rely on legacy known-valid seeds in random/evolutionary proposal modes.

## Target Interface

Add an NT-specific validity layer next to the broad domain:

```python
def explain_invalid_nt_hhs(params: dict[str, Any], *, shape: Shape | None = None) -> list[InvalidReason]:
    ...
```

`InvalidReason` should include:
- `rule_id`: stable identifier such as `nt_hhs.tlds2.requires_pgr2_plr0`.
- `message`: concise human explanation.
- `params`: names of involved parameters.
- `source`: `schema`, `solutionstructs`, `kernelwriter`, `resource`, `empirical`, or `heuristic`.
- `shape_dependent`: whether the rule depends on `M/N/K/batch`.

`cheap_constraints()` can remain as a fast boolean wrapper, but the explainable API should be the source of truth for debugging and mining.

## Candidate Domain Model

An EvoTensile candidate is a complete TensileLite solution parameter bundle emitted as one `Groups` entry, not an independent Cartesian product. The current NT HHS domain keeps broad categorical coverage for:
- tile and work shape: `MatrixInstruction`, `WorkGroup`, `DepthU`, `GlobalSplitU`;
- scheduling and prefetch: `PrefetchGlobalRead`, `PrefetchLocalRead`, `ScheduleIterAlg`, `1LDSBuffer`, `ClusterLocalRead`;
- LDS layout and padding: `TransposeLDS`, `LdsBlockSizePerPadA/B`, `LdsPadA/B`, `SourceSwap`;
- vectorization and stores: `VectorWidthA/B`, `GlobalReadVectorWidthA/B`, `StoreVectorWidth`, `StorePriorityOpt`, `StoreSyncOpt`, `GroupLoadStore`, `NumElementsPerBatchStore`;
- shape assertions and pointer/cache behavior: assert multiples, `WorkGroupMapping`, `StaggerU`, `StaggerUStride`, `StaggerUMapping`, `ExpandPointerSwap`.

Keep this domain broad. Encode hard validity only as explicit negative rules in `explain_invalid_nt_hhs()`, and use repairs/proposal bias only to make generated candidates practical.

## Disallowed Rule Categories

### 1. Schema And Type Rules

Mine and encode TensileLite front-end validation failures:
- Unsupported enum values.
- Parameter type expectations such as bool vs int for the active TensileLite version.
- Value range errors.
- Missing required linked fields.

These rules are mechanical and should be hard-coded once observed.

### 2. Matrix Instruction And Macro Tile Rules

Keep broad matrix-instruction coverage, but reject combinations that are mechanically impossible:
- Non-positive macro tiles.
- Macro tiles above supported generator limits.
- Unencodable `MatrixInstruction` values are handled by domain/encoding validation before the NT invalidity layer.
- `WorkGroup`/`MatrixInstruction` combinations rejected by `SolutionStructs` for occupancy or thread mapping reasons.

Avoid positive whitelisting by known macro tiles. Instead, encode explicit bounds and rejection-derived incompatibilities.

### 3. LDS And Prefetch Rules

Treat LDS settings as linked constraints, not independent scalar knobs:
- Explicit TensileLite source-backed LDS rejections, such as TLDS2 block-size-per-pad divisibility by `DepthU * 2` for FP16.
- `TransposeLDS` paths that require specific `PrefetchGlobalRead`, `PrefetchLocalRead`, `VectorWidthB`, or `1LDSBuffer` settings.
- LDS footprint beyond hardware or TensileLite generator limits.
- Layout-specific NT LDS restrictions discovered via rejection mining.

Do not use an LDS tuple allowlist as validity. Port only mechanical TensileLite rejection checks or minimized rejection-mining results.

### 4. Store Path Rules

Keep store-path domains broad, but reject known invalid couplings:
- `StoreSyncOpt` values unsupported by TensileLite for this problem type.
- `StoreSyncOpt != 0` with unsupported `NumElementsPerBatchStore` values.
- `GroupLoadStore=True` without required `StoreSyncOpt`, `StorePriorityOpt`, or batch-store settings.
- Store vector widths rejected by `SolutionStructs` or KernelWriter for specific macro-tile/store-path combinations.

Do not infer performance preference from current winners. Only encode build/validation invalidity.

### 5. Resource Rules

Mine resource failures and turn mechanical limits into rules:
- VGPR count above generator/hardware limit.
- LDS allocation above per-workgroup limit.
- Thread/tile mapping that exceeds supported codegen assumptions.
- KernelWriter failures with stable parameter signatures.

Resource rules may be approximate at first, but should report whether they are exact mechanical limits or conservative predictors.

### 6. Shape-Dependent Rules

Keep shape-dependent checks separate:
- Assert multiples for free/summation indices.
- Vector-width divisibility by `M/N/K` or leading dimensions.
- GlobalSplitU behavior that depends on `K` or batch.
- Edge cases for small shapes.

The same kernel may be valid for one NT shape and invalid for another. Shape-dependent failures should not globally poison a candidate.

## TensileLite Rejection Mining

### Build-Only Mining Loop

Run controlled build-only batches with:
- one candidate per batch for attributable failures,
- `PrintSolutionRejectionReason=True`,
- no timing requirement,
- fixed representative NT shapes first,
- sampled broad-domain candidates, not known seeds.

For each candidate, record:
- candidate hash and canonical params,
- shape id,
- build return code,
- final YAML presence,
- `Actual Solutions: X / Y after SolutionStructs`,
- `KernelWriter` resource messages,
- structured rejection reason lines,
- final accepted/normalized params if available.

### Failure Classification

Classify failures into:
- `schema`: fails before solution construction.
- `solutionstructs_zero`: no solution survives `SolutionStructs`.
- `solutionstructs_partial`: some candidates rejected, some accepted.
- `kernelwriter_resource`: VGPR/LDS/occupancy/codegen resource failures.
- `kernelwriter_bug_or_unknown`: codegen exception without clear rule.
- `runtime_validation`: builds and runs but fails correctness.
- `timeout`: unattributed unless isolated and repeatable.

### Delta Debugging

For frequent rejection signatures:
- Pick a nearby accepted parent.
- Apply the failing candidate changes one linkage group at a time.
- Minimize the parameter subset required to reproduce the rejection.
- Promote only stable minimal failures into `explain_invalid_nt_hhs()`.
- Leave non-minimized patterns in the negative cache.

## Negative Cache Policy

Use a negative cache for isolated, attributable failures:
- Cache only single-candidate build failures, validation failures, and repeated single-candidate timeouts.
- Do not poison the negative cache on multi-candidate batch failures.
- Include TensileLite version, profile, problem type hash, and relevant protocol/build settings.
- Separate kernel-global invalidity from shape-dependent invalidity.

Use negative cache rows to avoid repeating known failures, but do not turn them into global rules until explained.

## GOMEA Exploration Strategy

### Constraint-Aware Sampling

Replace template-first random as the long-term main source with broad-domain sampling plus rejection rules:
- Sample from `DOMAINS`.
- Apply only mechanical repairs and canonicalizations.
- Reject if `explain_invalid_nt_hhs()` returns hard invalid reasons.
- Use negative-cache lookup to skip known failures.
- Keep sampling statistics by value and value-pair coverage.

Random generation should maximize coverage of allowed values and pairs, not similarity to known winners. The current random path keeps `DOMAINS["TransposeLDS"] == [0, 1, 2]` but uses `NT_HHS_RANDOM_TLDS2_PROBABILITY` as a proposal-only bias toward the higher-yield TLDS2 path. For target shapes such as `8192,8192,1,8192`, shape-aware random generation sets `GlobalSplitU=1` directly because the workspace rule would reject larger GSU values for that shape. The fast direct TLDS2 sampler generates TLDS2 candidates with compatible `PrefetchGlobalRead=2`, `PrefetchLocalRead=0`, `VectorWidthB=1`, valid TLDS2 LDS pad-block choices, and proposal-only VALU lower-bound headroom. Explicit non-random candidates are still allowed to exceed that headroom and can still use TLDS0 when valid.

### Linkage Groups

Use invalidity rules and TensileLite rejection evidence to build GOMEA linkage groups:
- `MatrixInstruction`, `WorkGroup`, `DepthU`, `GlobalSplitU`.
- `TransposeLDS`, `LdsBlockSizePerPadA`, `LdsBlockSizePerPadB`, `LdsPadA`, `LdsPadB`.
- `PrefetchGlobalRead`, `PrefetchLocalRead`, `1LDSBuffer`, `ClusterLocalRead`, `VectorWidthB`.
- `GlobalReadVectorWidthA`, `GlobalReadVectorWidthB`, `VectorWidthA`, `VectorWidthB`.
- `ScheduleIterAlg`, `WorkGroupMapping`, `StaggerU`, `StaggerUStride`, `StaggerUMapping`, `SourceSwap`.
- `StorePriorityOpt`, `NumElementsPerBatchStore`, `StoreSyncOpt`, `GroupLoadStore`, `StoreVectorWidth`.
- `ExpandPointerSwap`.
- `AssertFree0ElementMultiple`, `AssertFree1ElementMultiple`, `AssertSummationElementMultiple`.

When GOMEA mixes a linkage group:
- validate the child immediately,
- keep valid children,
- retry or backtrack invalid linkage changes,
- record invalid reasons to improve linkage learning.

### Coverage Metrics

Track whether proposals cover the broad allowed space:
- per-parameter value coverage,
- pairwise coverage for linked groups,
- rejection rate by rule id,
- build-valid rate after TensileLite,
- final YAML normalization frequency,
- unique accepted matrix-instruction/macro-tile families.

Use these metrics to identify overly conservative rules or missing linkage groups. `proposal-coverage` is the lightweight check for proposal-side coverage before spending TensileLite build/run time. `summarize-rejections` is the offline loop for turning repeated TensileLite rejection signatures into reviewable evidence without adding hard rules prematurely.

## Validation Stages

### Stage 1: Rule Infrastructure

Status: implemented.

- Added `InvalidReason` and `explain_invalid_nt_hhs()`.
- Converted current boolean checks into explainable NT HHS rules.
- Kept `cheap_constraints()` as a wrapper over the explainable rule API.
- Added tests asserting representative invalid configs report expected rule ids.

### Stage 2: Rejection Mining

Status: initial tooling implemented.

- Added a TensileLite log classifier for schema, `SolutionStructs`, KernelWriter/resource, runtime validation, and unknown failures.
- Added `summarize-rejections` CLI support for scanning log files or run directories.
- Added tests for schema errors, zero accepted solutions, partial accepted solutions, resource messages, and JSON CLI output.
- Remaining work: run broad build-only mining batches with `PrintSolutionRejectionReason=True`, delta-debug frequent signatures, and promote stable mechanical failures into hard rules.

### Stage 3: Constraint-Aware Random

Status: implemented and tuned for the retained `8192^3` path.

- Replaced active template-first random generation with broad-domain sampling from `DOMAINS` plus mechanical repairs and `explain_invalid_nt_hhs()` filtering.
- Removed active dependence on curated `RANDOM_KERNEL_BUNDLES` and per-bundle mutation fences for random proposal generation.
- Removed the LDS tuple allowlist from validity. Broad LDS tuples are allowed unless rejected by explicit TensileLite-backed rules.
- Added proposal-only TLDS2 bias and a direct TLDS2 sampler so random generation remains broad but spends most attempts on high-yield NT HHS candidates.
- Added shape-aware random generation so `8192^3` proposals avoid shape-invalid `GlobalSplitU > 1` instead of sampling and rejecting those slots.
- Added proposal-only VALU lower-bound headroom for random generation. This is not a hard validity cutoff.
- Added tests that random candidates remain cheap-valid, cover many matrix-instruction and LDS-profile values, preserve TLDS0 availability, and respect the random headroom target.
- Remaining work: continue using rejection-mining output to add only explicit negative rules.

### Stage 4: GOMEA Integration

Status: implemented and validated on retained `8192^3` evidence.

- Added exported `NT_HHS_LINKAGE_GROUPS` derived from rule families.
- Fed the rule-derived linkage groups into stochastic GOMEA.
- Added invalid-child backtracking: stochastic GOMEA applies linkage groups one at a time and keeps only rule-valid changes.
- Added shape-aware and random-headroom filtering to stochastic GOMEA proposals, so GOMEA exploitation stays in the useful target-shape region without declaring those filters as global invalidity.
- Added neighborhood singleton coverage for every multi-valued `DOMAINS` knob, including `VectorWidthA`, `StaggerUStride`, `ExpandPointerSwap`, and assert-multiple knobs.
- Removed production known-seed and documented-winner constants from proposal initialization. Fixed controls now live only in tests.
- Added proposal coverage metrics and a `proposal-coverage` CLI to summarize unique value coverage and invalid-rule counts for generated candidates.
- Earlier non-hindsight `seed-random-gomea` reproduction checks generated the documented `8192^3` winner from proposal operators without inserting it directly. Pure categorical DE did not hit that exact winner in the checked budgets.
- Latest retained filtered GOMEA check generated only new `ok` rows for `8192^3` under the capped runtime experiment. The prior unfiltered misses were backend/resource/runtime outcomes, not new hard rules.
- Remaining work: record invalid-child reason counters inside each GOMEA generation and use them to refine linkage groups automatically.

### Stage 5: Search Runs

- Retain `out/one_shape_8192_single_fasttlds2_20260624_130547` as the latest useful fast random provenance run.
- Retain `out/one_shape_8192_gomea16_filtered_20260624_132622` as the latest capped GOMEA exploitation check.
- Use validation-passed random evidence to seed further GOMEA/local exploitation once residual backend failures remain stable.
- Then test staged multi-generation on larger grids.

## Acceptance Criteria

- NT search space is defined as broad `DOMAINS` minus explicit disallowed rules.
- Every hard invalid rule has a stable source and a test.
- Rejection mining artifacts can reproduce why rules were added.
- Random proposal modes do not depend on known valid seeds.
- GOMEA can explore non-template combinations while maintaining a practical build-valid rate.
- Accepted/rejected/normalized final TensileLite YAML remains the authoritative truth.

## Open Questions

- Which TensileLite type expectations differ across the reverted and tuned source trees, and should EvoTensile normalize them by profile/version?
- How many rejection-mining samples are needed before promoting an empirical pattern into a hard rule?
- Which rules are global for NT HHS and which must be shape-dependent?
- Can resource prediction for VGPR/LDS be made accurate enough before KernelWriter, or should it remain a learned negative-cache layer?
- How should linkage groups be updated automatically from repeated invalid-child reasons?
