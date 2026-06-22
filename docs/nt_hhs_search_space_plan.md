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
- Do not assume NN/TN/TT configs are valid for NT; use them only as inspiration for values and linkages.
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

Do not infer performance preference from current winners; only encode build/validation invalidity.

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

Random generation should maximize coverage of allowed values and pairs, not similarity to known winners.

### Linkage Groups

Use invalidity rules and TensileLite rejection evidence to build GOMEA linkage groups:
- `MatrixInstruction`, `WorkGroup`, `DepthU`, `GlobalSplitU`.
- `TransposeLDS`, `LdsBlockSizePerPadA`, `LdsBlockSizePerPadB`, `LdsPadA`, `LdsPadB`.
- `PrefetchGlobalRead`, `PrefetchLocalRead`, `1LDSBuffer`, `ClusterLocalRead`, `VectorWidthB`.
- `GlobalReadVectorWidthA`, `GlobalReadVectorWidthB`, `VectorWidthB`.
- `ScheduleIterAlg`, `WorkGroupMapping`, `StaggerU`, `StaggerUMapping`, `SourceSwap`.
- `StorePriorityOpt`, `NumElementsPerBatchStore`, `StoreSyncOpt`, `GroupLoadStore`, `StoreVectorWidth`.

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

Use these metrics to identify overly conservative rules or missing linkage groups.

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

Status: initial implementation complete.

- Replaced active template-first random generation with broad-domain sampling from `DOMAINS` plus mechanical repairs and `explain_invalid_nt_hhs()` filtering.
- Removed active dependence on curated `RANDOM_KERNEL_BUNDLES` and per-bundle mutation fences for random proposal generation.
- Removed the LDS tuple allowlist from validity; broad LDS tuples are allowed unless rejected by explicit TensileLite-backed rules.
- Added tests that random candidates remain cheap-valid and cover many matrix-instruction and LDS-profile values.
- Remaining work: measure TensileLite build-valid rate on representative NT shapes and use rejection-mining output to add only explicit negative rules.

### Stage 4: GOMEA Integration

Status: initial implementation complete.

- Added exported `NT_HHS_LINKAGE_GROUPS` derived from rule families.
- Fed the rule-derived linkage groups into stochastic GOMEA.
- Added invalid-child backtracking: stochastic GOMEA applies linkage groups one at a time and keeps only rule-valid changes.
- Added tests that GOMEA from broad random parents stays cheap-valid and that non-random GOMEA returns no candidates without cached/imported evidence.
- Removed production known-seed and documented-winner constants from proposal initialization; fixed controls now live only in tests.
- Added proposal coverage metrics and a `proposal-coverage` CLI to summarize unique value coverage and invalid-rule counts for generated candidates.
- Remaining work: record invalid-child reason counters inside each GOMEA generation and use them to refine linkage groups automatically.

### Stage 5: Search Runs

- Re-run the one-shape `8192,8192,1,8192` experiment from an empty DB.
- Require generation 0 to produce enough timed candidates before adding multi-generation search.
- Then test staged multi-generation on the 100-shape grid.

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
