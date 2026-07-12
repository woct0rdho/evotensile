# Contextual Pair Model

This document defines the shared probabilistic exact-pair model in `evotensile/search/pair_model.py`. The existing proposal-only per-shape shortlister remains documented in `docs/search_surrogate.md`. Promotion heuristics are documented in `docs/shape_promotion_racing.md`.

## Boundary

`ContextualPairModel` predicts one candidate-shape pair. It does not disclose evidence, declare a winner, bypass validation, or mutate a controller. It trains only from `PairEvaluationOutcome` records that are both known and disclosed. Replay-unknown, undisclosed, and hidden pairs are ignored.

The model is shared by one-shape and multi-shape campaigns. One shape follows the same fit and predict path with one shape reference. There is no separate singleton model.

## Target

For every shape with disclosed positive evidence, fitting records the best visible performance reference `r_s`. The regression target is:

```text
z(c, s) = log(performance(c, s) / r_s)
```

This retains shape-local normalization and avoids pooling raw latency or throughput across workloads. The best visible pair on a shape has target zero. Weaker pairs are negative. A prediction on a shape with no training evidence still returns a normalized score and uncertainty, but has no absolute `predicted_performance` because no compatible visible reference exists.

## Features

The model reuses `candidate_shape_features()` so its fitted contract includes:
- literal canonical values for every mixed and conditional candidate gene.
- log shape dimensions, ratios, batch, and arithmetic intensity.
- candidate macro-tile dimensions, fill, and remainders.
- output tiles, workgroups, WGP rounds, and WGP granularity.
- reduction iterations and K fill.
- wave/workgroup geometry.
- analytical VGPR, LDS, workspace, vector-width, and store-batching interactions.

`PairModelFitSummary` records the complete configuration, feature count, SHA-256 feature-contract identity, evidence counts, shape references, uncertainty calibration, and candidate/shape coverage. The retained-corpus evaluation artifact stores this summary for every split.

## Performance Ensemble

The selected model is a bootstrapped `ExtraTreesRegressor`:
- 192 trees in production configuration.
- minimum leaf size 2.
- 70% feature subsampling.
- 80% bootstrap sample per tree.
- target-profile-controlled CPU jobs.

Tree predictions provide posterior-like samples. Their mean is the normalized log-performance estimate. Tree disagreement is calibrated with an internal deterministic 20% exact-pair split, then the final model is refit on all visible rows. A conservative margin is applied, and uncertainty is inflated for a candidate or shape absent from the fit set.

`PairPrediction` exposes:
- normalized posterior mean.
- calibrated epistemic standard deviation.
- every calibrated tree sample.
- validity probability.
- optional shape reference and predicted absolute performance.
- validity-weighted probability of exceeding an incumbent by a requested gain.

These samples are an ensemble approximation, not an exact Bayesian posterior.

## Validity Model

A separate bootstrapped `ExtraTreesClassifier` trains on every known disclosed outcome:
- positive class: finite positive performance.
- negative class: known validation, preparation, or timing failure without positive performance.

If visible evidence contains only one class, the model uses that constant probability. Unknown pairs never become negative labels. Acquisition can multiply improvement or information value by the returned validity probability.

## Sparse Evidence

The shared contextual model can predict:
- a candidate absent from fitting, using its literal and mechanical features with novelty-inflated uncertainty.
- a shape absent from fitting, using its descriptor interactions with novelty-inflated uncertainty and no invented absolute reference.
- a sparsely observed candidate or shape using all other visible pair rows.

This differs from the existing shape-local shortlister, which requires enough varied candidates independently on every modeled shape.

## Metrics

`evaluate_pair_predictions()` reports:
- normalized-log mean absolute error.
- empirical 50%, 80%, and 90% interval coverage.
- mean per-shape Spearman-style rank correlation.
- mean per-shape top-k recall.
- validity Brier score.
- probability-of-improvement Brier score and binned calibration error.

Metrics use only supplied visible evaluation outcomes. Held-out shape evaluation normalizes actual performance within each held-out shape and does not invent an absolute training reference.

## Controlled Retained-Corpus Evaluation

`scripts/evaluate_pair_model.py` uses `out/grid100_full_20260618_repaired.sqlite` read-only and writes `out/grid100_pair_model_20260712.json`. The run uses:
- 100 shapes.
- 15,344 exact known oracle records, including positive and failed pairs.
- five deterministic folds.
- 96 trees per model for evaluation speed.
- separate held-out candidate, held-out shape, and masked exact-pair splits.

The comparison baselines are:
- existing shape-local ExtraTrees on held-out candidates and masked pairs.
- exact nearest-shape normalized transfer on held-out shapes and masked pairs.
- iterative rank-8 candidate/shape factorization on masked pairs.

Key results are:

| Split and model | Positive test pairs | MAE | Rank correlation | Top-3 recall | 90% coverage | Validity Brier |
|---|---:|---:|---:|---:|---:|---:|
| held candidate, shared contextual | 1,730 | 0.396 | 0.689 | 0.457 | 0.876 | 0.060 |
| held candidate, shape-local ExtraTrees | 1,730 | 0.422 | 0.707 | 0.463 | 0.375 | 0.414 |
| held shape, shared contextual | 1,708 | 0.231 | 0.912 | 0.600 | 0.904 | approximately 0 |
| held shape, nearest transfer | 1,707 | 0.281 | 0.875 | 0.517 | 0.404 | 0 |
| masked pair, shared contextual | 1,790 | 0.249 | 0.895 | 0.693 | 0.868 | approximately 0 |
| masked pair, low rank | 1,790 | 0.242 | 0.919 | 0.683 | 0.556 | 0.430 |
| masked pair, nearest transfer | 1,790 | 0.304 | 0.859 | 0.607 | 0.423 | 0 |
| masked pair, shape-local ExtraTrees | 1,790 | 0.385 | 0.725 | 0.530 | 0.424 | 0.430 |

The contextual ensemble is selected for campaign acquisition because it is the only compared implementation that simultaneously:
- predicts unseen candidates.
- predicts unseen shapes.
- handles mixed conditional candidate features.
- exposes calibrated uncertainty samples.
- models validity.
- retains strong rank and top-k behavior.

Low-rank factorization remains the strongest masked-pair MAE/rank baseline, so later acquisition ablations should retain it as a diagnostic. It cannot represent a genuinely unseen candidate or shape without an additional side-feature model.

Probability-of-improvement Brier score is approximately `0.0085` on held-out candidates and `0.0108` on masked pairs. The event rate is low because visible references are already the best training performance, so this metric must be interpreted with calibration error and acquisition recall rather than alone.

## One-Shape Default

The shared model supports one shape without a separate implementation. Mechanical covering remains the cold-start proposal selector. Once the singleton campaign has at least 24 exact positive observations, the built-in provider fits this model and applies singleton bundle acquisition to the oversized generated pool. Controlled policy tuning selected information weight `0.05`. Insufficient evidence or an explicitly disabled policy falls back to the existing shape-local surrogate. See `docs/shared_bundle_acquisition.md`, `docs/campaign_policy_tuning.md`, and `docs/blind_campaign_control.md`.

## Limitations

- Tree samples are correlated and should not be interpreted as exact independent posterior draws.
- Held-out candidate interval coverage remains below the nominal 90% target at 87.6%.
- The retained corpus is sparse and selection-biased toward historically proposed configurations.
- Failure labels combine several known terminal statuses. Future data may justify separate build, validation, and runtime classifiers.
- Model fitting is in-memory and currently refits from the complete supplied visible snapshot.
