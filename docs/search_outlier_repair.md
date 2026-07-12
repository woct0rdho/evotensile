# Integrated Weak-Shape Repair

This document describes the generic repair acquisition in `evotensile/campaign/repair.py`. Staged admission and the explicit repair reserve are documented in `docs/staged_round_controller.md`. Shared-cost candidate bundles are documented in `docs/shared_bundle_acquisition.md`.

The former standalone `repair-outliers` command and its dense candidate-by-shape schedule have been removed. Repair now uses the same contextual model, candidate bundles, exact requests, shared preparation accounting, durable evidence, and soft round admission as broad and promotion work.

## Boundary

Repair is a budget-allocation heuristic. It does not:
- inspect hidden oracle winners.
- encode retained-corpus winners or candidate hashes.
- impose smoothness as a validity or selection rule.
- copy performance between shapes.
- infer evaluation pairs from artifact scope.

Every selected repair pair is measured exactly on its target shape. Unknown replay pairs remain unknown and still consume their attempted budget.

Grid-outlier detection is a no-op for one shape. Ordinary singleton incumbent improvement, stabilization, and confirmation remain active.

## Deficit Evidence

`assess_repair_deficits()` requires the current controller, the campaign's mechanical clustering, and optional compatible reference performance and contextual predictions.

For each resolved shape, it derives candidate-independent target signals from:
- an installed or other explicitly supplied compatible reference for that exact shape.
- a distance-weighted upper quantile of nearest resolved-shape incumbents.
- an upper quantile of incumbents in the same mechanical cluster.
- calibrated epistemic uncertainty for still-unqueried candidate pairs on the target shape.

Nearest-shape distance uses the same normalized mechanical descriptors as clustering. Neighbor and cluster targets use log-performance quantiles to reduce sensitivity to one noisy maximum. The available reference, neighbor, and cluster targets are combined by their median. Model uncertainty contributes only a configurable fraction of log headroom.

For incumbent performance `P_s` and robust evidence target `T_s`, the raw deficit is:

```text
max(0, log(T_s / P_s) + uncertainty_weight * epistemic_uncertainty_s)
```

Deficits below `minimum_deficit_fraction` are ignored. Larger deficits are capped at `maximum_deficit_fraction`. The cap prevents one apparently weak shape from consuming the complete reserve.

This estimate can be wrong because real kernels have shape-specific cliffs. It allocates exact measurements rather than changing the incumbent directly.

## Candidate Support

A deficit alone is insufficient. For every available unqueried candidate-shape prediction, repair computes the posterior probability of reaching a configurable fraction of capped headroom:

```text
useful_target = incumbent * exp(useful_close_fraction * capped_deficit_log)
close_probability = P(candidate >= useful_target) * P(valid)
```

Probabilities below `minimum_close_probability` become zero. A shape therefore receives no repair utility when the candidate catalog and model provide no support for useful improvement.

The bundle allocator receives:
- shape repair value equal to capped log deficit.
- exact candidate-shape close probability.
- the normal expected-improvement, information, shape-weight, and cost terms.

The repair component for a pair is capped deficit multiplied by its candidate-specific close probability. It is not assigned uniformly to every valid candidate. Lazy greedy selection still accounts for candidates competing for the same shape, shared preparation, artifact expansion, exact validation/timing cost, pair and bundle caps, and utility per second.

## Candidate Seeds

`build_repair_candidate_pool()` produces one deduplicated, auditable finite pool from:
- the weak shape's current incumbent.
- nearest-shape winners and near-winners from disclosed evidence.
- champions from the target's mechanical cluster.
- broad candidates supplied by the current round planner.
- deterministic generic semantic mutations around those sources.

Every candidate records its contributing lanes and target shapes. Mutations use existing linked-parameter repair and shape-scope eligibility. No source is assumed to transfer. The contextual model scores the exact target pair and the evaluator measures selected requests.

Historical exact-oracle ablations may disable new mutations because absent native pairs must remain unknown. Real or hybrid campaigns may retain them and measure the exact new pairs.

## Reserve And Admission

`StagedRoundConfiguration.repair_fraction` creates an explicit cumulative repair reserve. A broad expected-improvement wave cannot consume it because an over-budget broad plan closes its phase while later phases remain eligible.

Within the repair phase, callers refit the contextual model and recompute deficits, candidate support, and bundle acquisition after each durable wave. Unsupported shapes naturally lose utility. Admitted work drains under unchanged operational timeouts. Overrun blocks all later admission.

The repair module does not bypass the staged planner boundary. It returns a normal `AcquisitionPlan` with exact `PairRequest` objects, artifact scopes, predicted costs, preparation order, and timing priorities.

## Reporting

`RepairReport` records:
- eligible weak shapes.
- exact repair queries.
- queries reusing prepared target artifacts.
- deficits reaching the useful-close target.
- mean and worst per-shape incumbent gain.
- queries with no incumbent gain.
- predicted cost attributed to those false repairs.

Candidate-pool origins, deficits, pair close probabilities, and the full acquisition plan are serializable for round reports.

## Retained-Corpus Ablation

`scripts/simulate_repair_acquisition.py` compares two equal-budget staged policies on `out/grid100_full_20260618_repaired.sqlite`. Both receive the identical 373-pair broad wave after the same 16-medoid seed. One spends the final 12 pairs on another broad acquisition. The other spends them through repair. New mutations are disabled only for this exact-oracle comparison because their historical pairs do not exist.

The selected characterization in `out/grid100_repair_acquisition_20260712.json` reports:

| Final 12-pair policy | Mean log regret | P95 log regret | Worst log regret | Resolved shapes |
|---|---:|---:|---:|---:|
| Broad continuation | 0.07376 | 0.38236 | 0.66922 | 66 |
| Repair reserve | 0.06394 | 0.31275 | 0.63070 | 62 |

Repair resolves one flagged deficit and improves mean, p95, and worst regret, but broad continuation resolves four more previously unseen shapes. This supports a small explicit weak-tail reserve without claiming that repair should replace broad coverage. P12 subsequently tested the reserve across seeds, orderings, initialization profiles, equal pair budgets, and targeted hybrid measurements. Selected repair fractions remain small and profile-specific. See `docs/campaign_policy_tuning.md`.

## Tests

Tests cover:
- singleton no-op behavior.
- reference/neighbor/cluster/uncertainty deficit construction and caps.
- candidate-specific close-probability gating.
- exact repair bundle selection.
- incumbent, neighbor, cluster, broad, and mutation seed provenance.
- preparation reuse, resolved-deficit, gain, and false-repair reporting.
