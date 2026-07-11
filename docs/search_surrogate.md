# Surrogate-Guided Proposal Search

This document describes EvoTensile's optional ExtraTrees surrogate and oversized-pool shortlisting. The surrogate ranks already generated candidates. It does not invent parameters, bypass validity, or replace the underlying random, semantic, DE, GOMEA, or family-QD operators.

## Purpose

Compilation and validation are expensive, while generating and feature-encoding valid candidates is cheap. The surrogate uses this asymmetry to consider a much larger proposal pool than the measurement budget can afford.

The scheduler can therefore:
- Generate several times the requested random and variation budget.
- Fit a model from compatible queried DB evidence.
- Preserve archive and transfer parents.
- Select a fixed-size mixture of predicted performance, uncertainty, family diversity, and random exploration.
- Compile and measure only the shortlist.

This is the main mechanism for increasing structural search breadth inside a practical wall-time budget.

## Activation

The relevant controls are:

```text
--surrogate-pool-multiplier N
--surrogate-min-evidence N
--covering-cold-start
```

`--covering-cold-start` changes only the one-shape evidence-free fallback and is independent of model activation.

The default multiplier is `1`, so existing proposal behavior is unchanged unless shortlisting is requested. The default evidence threshold is `24` unique candidates per shape. A shape model also requires variation in at least four candidate genes, so repeated rows or one candidate measured across many shapes cannot activate the surrogate.

For example, a multiplier of `8` generates eight times the configured random and variation counts, then returns the original requested measurement count plus preserved archive or transfer candidates.

The implementation requires `scikit-learn>=1.4`.

## Training Evidence

Training rows come from the immutable proposal evidence snapshot, whose ranking summaries were loaded once through `EvoTensileDB.rank_evaluations()` under the requested problem type and benchmark protocol. The surrogate filters those summaries to its target shapes without another DB scan.

Eligible evidence has:
- `status='ok'` timing samples.
- a positive median time.
- compatible problem and benchmark identity.
- a known candidate and shape.

Probe rows use a different benchmark protocol and therefore cannot enter a main-protocol surrogate accidentally. Hidden or unqueried oracle values are never training data.

The target is:

```text
log(median_time_us)
```

Lower values are better. Log time makes multiplicative performance differences more uniform across candidates and shapes.

## Candidate And Shape Features

`candidate_shape_features()` combines literal categorical genes with generic mechanical features.

Categorical features include the canonical value of every parameter in `PARAM_NAMES`.

Derived features include:
- `log2(M)`, `log2(N)`, `log2(K)`, batch, aspect ratio, and arithmetic intensity.
- macro-tile dimensions, area, aspect, and M/N tile fill.
- output tiles, GSU-expanded workgroups, workgroups per WGP, WGP rounds, and WGP granularity.
- reduction iterations and K fill from `K`, `DepthU`, and `GlobalSplitU`.
- workgroup threads, waves, WMMA wave-tile area, and wave-group size.
- proposal-side VALU VGPR lower bound and fraction.
- LDS bytes and fraction plus GSU workspace fraction.
- local and global vector widths in bytes.
- whether store batching uses the nominal auto value.

These features express shape compatibility and resource mechanics without encoding a known winning parameter bundle. Their shared analytical definitions and limitations are documented in `docs/search_mechanical_coverage.md`.

## Model

`ExtraTreesSurrogate` uses:

```text
ExtraTreesRegressor
n_estimators = 192
min_samples_leaf = 2
max_features = 0.7
n_jobs = target-profile surrogate job limit
```

The gfx1151 profile defaults to one ExtraTrees job so proposal fitting cannot consume the entire CPU pool or oversubscribe preparation work in a coordinator.

A `DictVectorizer` converts mixed categorical and numeric feature dictionaries to a dense matrix.

A separate model is fitted for every target shape with enough unique varied candidates. For each candidate-shape pair, predictions from every tree provide:
- mean predicted log time.
- tree-to-tree standard deviation as shape-local epistemic uncertainty.

Between-shape latency or performance variation is never added to uncertainty. Each prediction is compared with that shape's incumbent log time. One-shape search uses the same path with one shape model.

## Shortlist Composition

The fixed-size shortlist is assembled in explicit grid lanes:
- `35%` greedy marginal optimistic gain over shape incumbents, preserving complementary specialists.
- `25%` lowest mean incumbent-normalized log regret among modeled shapes.
- `15%` highest shape-local ensemble disagreement.
- `10%` per-shape mechanical-priority coverage for unresolved shapes.
- family-diverse and seeded-random fill, followed by maximum-Hamming-distance fill if needed.

Equal shape weights are used for the fixed authoritative grid. The greedy gain lane tracks gain already covered per shape, so a candidate with the same strengths as an existing shortlist member has less marginal value than a complementary specialist. This ensures that a candidate exceptional on a small subset of shapes is not hidden by grid-wide average latency.

Workload-derived acquisition weights are not implemented. A future explicit workload mode may derive them from shape frequency, baseline latency, or measured end-to-end contribution, but a fixed authoritative GridBased campaign retains equal weights and complete shape coverage.

This mixture prevents the model from converting early noisy evidence into a fully greedy search.

## Cold-Start Fallback

A shape without enough unique varied candidates remains unresolved and gets no fitted model. If no shape is modeled, multi-shape selection uses per-shape mechanical-priority round-robin for half the budget and family diversity for the remainder. If only some shapes are modeled, the unresolved-shape lane remains separate from model exploitation and uncertainty.

For an explicitly enabled one-shape covering cold start, the fallback uses the quality-weighted mechanical covering selector from `docs/search_mechanical_coverage.md`. Exact MatrixInstruction identities are not coverage tokens. This policy changes only which generated candidates are shortlisted and preserves the same measurement budget.

## Scheduler Integration

`propose_candidates()` applies the pool multiplier to random, semantic mutation, DE, and GOMEA generation counts. It then:
- deduplicates candidates by hash.
- separates generated candidates from preserved archive and transfer candidates.
- computes the requested generated-candidate measurement count.
- calls `select_surrogate_pool()`.
- returns preserved parents plus the selected generated candidates.

Family archive parents are never discarded merely because the surrogate predicts another family to be faster.

The surrogate can be combined with `--adaptive-operators`. Operator allocation determines how the oversized pool is generated. The surrogate determines which generated children receive expensive measurement.

## Blindness And Reproducibility

The surrogate is query-causal:
- it trains only from rows already present in the active DB.
- unknown candidates remain unknown until measured.
- candidate order, model seed, diversity fallback, and random fill are deterministic for a fixed search seed and DB snapshot.
- external control measurements are not imported into production campaign DBs.

The replay infrastructure in `docs/blind_experiment_infrastructure.md` enforces the same exact-query rule for simulated campaigns.

## Limitations

Current limitations include:
- two-sample medians can be noisy enough to misrank provisional leaders.
- the model has no explicit compile-failure or validation-failure classifier.
- model fitting is rebuilt from the active immutable evidence snapshot on each proposal call.
- acquisition weights and shortlist fractions are fixed rather than cost-adaptive.
- uncertainty is ensemble disagreement, not a calibrated posterior interval.
- there is no cross-campaign transfer model beyond evidence explicitly present in the DB.
- campaign-level screening stabilization improves a few global contenders but is not a general surrogate-training fidelity policy.

Screening stabilization is documented in `docs/search_screening_stabilization.md`.
