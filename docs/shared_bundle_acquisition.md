# Shared-Cost Bundle Acquisition

This document defines cost-aware finite candidate-shape bundle selection in `evotensile/campaign/acquisition.py`. Pair posterior modeling is documented in `docs/contextual_pair_model.md`. Exact scheduling is documented in `docs/exact_pair_scheduling.md`.

## Boundary

Bundle acquisition consumes:
- a finite candidate set.
- the exact registered shape set.
- policy-visible `PairPrediction` values.
- current controller incumbents, queried pairs, and artifact coverage.
- one fitted or fallback `BundleCostModel`.
- optional shape weights and repair values.

It does not generate candidates, disclose evidence, run the scheduler, or infer an unmeasured winner. Its output is an `AcquisitionPlan` containing exact requests, explicit artifact scopes, bundle scores, predicted cost, preparation order, and timing priorities.

One five-minute round may execute one or more such waves. The staged round controller owns admission, persistence, and recomputation after durable results.

## Candidate Bundles

For each candidate, the planner ranks currently unqueried predicted pairs and creates deterministic prefix bundles at configured sizes. The selected default size set is `1, 2, 4, 8`. Controlled replay additionally tests size 16. The full available prefix is included as an option even when it is not a configured size.

At most one bundle per candidate is selected in a wave. This prevents paying shared setup twice and makes the chosen artifact scope explicit.

If a candidate already has artifact coverage, its proposed artifact scope is the union of the existing scope and newly requested destinations. A request fully inside the existing scope has no preparation cost. A scope with new shapes is an artifact expansion and pays preparation once. Exact validation and timing remain per requested pair.

## Posterior Utility

For every pair, posterior tree samples are compared with the shape-local incumbent. Marginal expected improvement is:

```text
E[max(0, posterior_log_performance - incumbent_log_performance)] * validity_probability
```

For unresolved shapes, no incumbent improvement is invented. Coverage value instead uses posterior expected relative quality:

```text
E[exp(min(0, posterior_log_performance))] * validity_probability
```

This distinguishes a likely useful first incumbent from a merely valid but predicted-poor pair. An earlier equal-value coverage implementation resolved many shapes with poor candidates and was removed after controlled replay exposed the failure.

Information value is calibrated posterior standard deviation times validity probability. Optional repair value is supplied by the later repair layer. Shape weights default to one. Explicit workload mode obtains the same normalized weights from call count multiplied by baseline latency and uses them for every pair utility and resulting timing priority. See `docs/workload_weighting.md`.

The scalar bundle utility is:

```text
improvement_weight * marginal_improvement
+ coverage_weight * unresolved_coverage
+ information_weight * information
+ repair_weight * repair_value
```

Utility is divided by predicted marginal makespan. The scalar is an admission priority, not a replacement for per-shape incumbents or regret reporting.

## Posterior Competition

The selector tracks gain already covered for every shape and posterior sample index. After selecting a bundle, a second candidate competing for the same sampled shape improvement receives only uncovered marginal gain. This avoids simply summing independent expected improvements for candidates that cannot all become the incumbent.

The implementation uses deterministic lazy greedy selection:
- push every candidate-prefix option with an initial utility-per-cost upper score.
- pop the current best option.
- recompute its true marginal score against selected posterior coverage.
- reinsert it if another stale upper score may still dominate.
- select it only when current-best, within pair/bundle/cost caps, and above the minimum utility rate.
- remove every other prefix for that candidate.

This is a one-wave lookahead. No exact combinatorial optimizer is used.

## Cost Model

`BundleCostModel` separates:
- candidate preparation or artifact expansion.
- exact-pair validation.
- exact-pair probe/screening timing.

`fit()` consumes `CandidateMeasuredCost`, candidate identity, and exact observed shape coverage. It fits three conservative bootstrapped ExtraTrees regressors when each phase has enough rows:
- preparation on candidate features aggregated across artifact scope.
- validation on exact candidate-shape features.
- probe plus screening timing on exact candidate-shape features.

Prediction uses tree median plus twice tree MAD, with a robust measured fallback floor. When evidence is insufficient:
- preparation uses a conservative fallback scaled by the analytical candidate preparation weight.
- validation and timing use conservative per-pair fallbacks.

When a retained historical corpus has no typed `run_candidate_costs`, replay uses declared fallback costs. New real campaign overlays provide fitted phase rows when enough measured costs are available.

`BundleCostFitSummary` reports phase row counts, fitted/fallback status, and fallback durations.

## Scheduler Ordering

Acquisition keeps two orders separate:
- `preparation_order` is longest predicted preparation first, which reduces parallel preparation tail.
- `timing_requests` carry utility-per-cost priority for serialized exact timing.

The scheduler still controls safe batch construction and never derives evaluation pairs from artifact scope.

## Singleton Law

For one shape, every bundle has one pair. There is no shared cross-shape setup decision, posterior competition is shape-local, and utility reduces to expected improvement, unresolved coverage, information, or repair value divided by exact predicted cost.

P09's initial five-case characterization did not justify a default change. P12 repeated the comparison across five representative shapes and three deterministic seeds using the shared policy schema, 32 seed measurements, and 16 equal shortlist requests. Results are stored in `out/grid100_singleton_policy_tuning_20260712.json`.

Bundle acquisition with information weights `0.05`, `0.10`, and `0.25` tied at mean log regret `0.01027`, p95 `0.07091`, and worst `0.09227`. The existing surrogate produced `0.02106`, `0.10920`, and `0.14871`. The selected default is therefore the smallest winning information weight, `0.05`.

Production family-QD preserves mechanical cold-start covering. Singleton bundle acquisition activates only after at least 24 exact positive observations. Otherwise selection falls back to the existing surrogate. See `docs/campaign_policy_tuning.md`.

## Multi-Shape Replay Characterization

`scripts/simulate_bundle_acquisition.py` writes `out/grid100_bundle_acquisition_20260712.json`. Every policy starts from the same:
- retained exact-oracle snapshot.
- 100 shapes and 217 candidate catalog.
- 80 seed candidates evaluated on 16 deterministic medoids.
- shared contextual model fitted only from that seed result.
- added exact-pair budget of 385.

Unknown retained pairs remain unknown. The test is one acquisition wave, not a full five-minute or multi-round campaign.

| Policy | Added pairs | Resolved shapes | Mean log regret | P95 log regret | Worst log regret | Prepared candidates |
|---|---:|---:|---:|---:|---:|---:|
| representative only | 0 | 16 | 0.0902 | 0.4597 | 0.6307 | 80 |
| observed transfer | 279 | 94 | 0.1809 | 0.6127 | 1.2086 | 80 |
| global dense | 336 | 16 | 0.0902 | 0.4597 | 0.6307 | 80 |
| independent model rank | 385 | 93 | 0.1293 | 0.5736 | 1.7508 | 100 |
| joint quality (`coverage=0.2`) | 385 | 53 | 0.1039 | 0.4150 | 0.7151 | 83 |
| joint balanced (`coverage=0.5`) | 385 | 67 | 0.0908 | 0.3783 | 1.7508 | 83 |
| joint coverage (`coverage=1.0`) | 385 | 68 | 0.0980 | 0.3806 | 1.7508 | 82 |
| joint information (`coverage=0.5`, `information=0.3`) | 385 | 53 | 0.1039 | 0.4150 | 0.7151 | 83 |

The balanced policy is the P09 characterization point:
- substantially better mean and p95 resolved-shape regret than transfer and independent ranking.
- only three newly prepared candidates beyond the seed, versus 20 for independent ranking.
- fewer resolved shapes than transfer or independent ranking, which is acceptable for one bounded round and leaves explicit work for later rounds.

The result is the P09 characterization point rather than the final default. P12 selects profile-specific weights, scopes, cluster counts, reserves, repair caps, and round roles in `docs/campaign_policy_tuning.md`. Complete one-round resolution remains explicitly unnecessary.

## Tests

Tests verify:
- preparation is paid once while validation and timing scale per exact pair.
- posterior sample coverage suppresses redundant same-shape candidates.
- artifact expansion preserves prior scope while requesting only the new exact pair.
- preparation order and timing priority remain separate.
- singleton expected-improvement-per-cost reduction.
- preparation, validation, and timing regressors fit when measured rows are available.
