# Campaign Policy Tuning

This document records the controlled P12 hyperparameter selection for the gfx1151 FP16 NT HHS 100-shape campaign. The reusable configuration schema lives in `evotensile/campaign/policy.py`. Fold construction, aggregation, Pareto filtering, and robust selection live in `evotensile/campaign/tuning.py`. The complete machine-readable result is `out/grid100_policy_tuning_20260712.json`. `selected_campaign_policy()` and `selected_campaign_round_schedule()` expose the frozen defaults to later campaign layers without reparsing the artifact.

## Compatible Oracle Snapshot

Policy replay uses the consolidated read-only snapshot:

```text
out/grid100_compatible_20260712.sqlite
```

`scripts/merge_compatible_databases.py` created it from:
- `out/grid100_full_20260618_repaired.sqlite`.
- `out/grid100_untuned_hipblaslt_baseline_20260712.sqlite`.
- `out/grid100_tuned_hipblaslt_baseline_20260712.sqlite`.
- `out/grid100_policy_hybrid_20260712.sqlite`.
- `out/grid100_policy_hybrid_round2_20260712.sqlite`.

Every imported timing row uses problem type `ptype_445e2e64534751bc`, benchmark protocol `bproto_9f4055f5f13232a3`, and environment compatibility tag `gfx1151-nt-hhs-v1`. Current validation rows use `vproto_54c03ca125088879`. Retained legacy validation remains separately identified. The merge copies raw benchmark events and samples, validation, native phase/candidate costs, baseline discoveries, and baseline selections. It deliberately excludes proposal events and artifact registrations because those are campaign-local execution records rather than portable measurement evidence.

The source manifest is `out/grid100_compatible_20260712_manifest.json`. The final snapshot contains:
- `224` candidates and `100` shapes.
- `17,171` benchmark events and `168,214` timing samples.
- `15,493` exact candidate-shape pairs, of which `8,858` have positive timing.
- `461` validation events.
- `2` labeled baseline discoveries and `200` exact baseline selections.
- `435` native-run records and `141` candidate-phase cost rows.

Compatible measured pairs are simulated directly. Missing pairs remain unknown. Native execution is used only for exact pairs absent from the consolidated snapshot.

## Initialization Profiles

One common schema represents three separately tuned regimes:
- `blind`: no disclosed baseline incumbent at campaign start.
- `anchored-untuned`: all 100 exact selections from the untuned GridBased logic are disclosed incumbents and references.
- `anchored-tuned`: all 100 exact selections from the tuned GridBased logic are disclosed incumbents and references.

The anchored overlays each contain current validation plus ten timing samples for all 100 selected pairs. Untuned uses 10 unique candidates. Tuned uses 22. Initialization evidence does not mark candidates as prepared search artifacts.

The untuned discovery records source-logic SHA-256 `9cbf840639705192fb0a8123ef08aaa7c6ecd2bf63e0b71d8dc44e86a6187ab6` and installed-logic SHA-256 `900723b8a5fd64bd4024b875893fcd318fa672d6a7b5d20d4e13618b2a2317a1`. Tuned records source-logic SHA-256 `16e55e16636edf6a328c0bc0f0d55603b2636e874b9e0e9fe8bfb201190df2e5` and installed-logic SHA-256 `6538190f1e53c39c4ffa8448393b43c1378d1082a0d877ac18946b39dd37f87c`.

## Controlled Sweep

`scripts/tune_campaign_policy.py` evaluates 12 generic configurations across the three initialization profiles. Each configuration receives:
- three deterministic candidate-order seeds: `20260712`, `20260713`, and `20260714`.
- five mechanically stratified shape folds derived from deterministic 16-medoid clustering.
- one equal added-pair budget of `385` requests.
- exact unknown-pair accounting and attempted-cost charging.
- the same generic promotion, posterior model, shared-cost acquisition, staged-round, and repair implementations.

The artifact contains 180 fold observations. Candidate ordering uses a seeded hash of each candidate identity rather than in-place shuffle. Adding a compatible overlay candidate therefore preserves the relative order of every existing candidate.

Blind replay starts with 80 candidates queried on the 16 medoids. Anchored replay starts with its 100 exact baseline pairs and uses a small first-round calibration set only while each shape has at most one positive disclosed observation. Calibration counts are tuned independently because one observation per shape normalizes every pair-model target to zero.

Configurations are compared within their initialization profile. Pareto objectives are mean, p95, and worst fold regret, unresolved shapes, prepared candidates, unknown pairs, and variance of seed-level mean regret. Robust selection minimizes the worst normalized objective plus a small normalized mean penalty, then uses tail regret for deterministic ties.

## Selected One-Round Defaults

| Initialization | Policy | Identity | Clusters | Calibration | Artifact scope | Coverage | Information | Repair cap | Guard | Mean log regret | P95 | Worst | Mean unresolved shapes |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| blind | `balanced-16-requested` | `campaign_policy_46baa1a9` | 16 | 0 | requested | 0.50 | 0.10 | 0.30 | 30s | 0.5641 | 1.4444 | 2.1841 | 16.67 |
| anchored-untuned | `tail-16-cluster` | `campaign_policy_89ea03a4` | 16 | 4 | cluster | 0.35 | 0.05 | 0.20 | 20s | 0.1998 | 0.5073 | 0.8286 | 0 |
| anchored-tuned | `information-20-requested` | `campaign_policy_e7961c9f` | 20 | 8 | requested | 0.50 | 0.25 | 0.30 | 20s | 0.0757 | 0.2732 | 0.4010 | 0 |

Mean unresolved shapes converts the fold aggregate back to the 100-shape run scale. Blind seeds resolved 100, 50, and 100 shapes respectively. Unresolved shapes remain explicit and are not assigned synthetic performance.

The selected repair settings differ generically by regime. Blind and anchored-tuned retain a 30% maximum deficit and 10% minimum close probability. Anchored-untuned uses a 20% cap and 20% minimum close probability. Promotion mechanics remain generic and unchanged: one protected specialist lane, two broad slots, neighbor depth two, adjacent-cluster depth one, one-sample probes, three-sample main top-ups, and observed regret stop rules.

The selected staged phase fractions are:
- blind: broad `0.35`, promotion `0.418831`, repair `0.031169`, stabilization `0.10`, confirmation `0.10`.
- anchored-untuned: broad `0.35`, promotion `0.408442`, repair `0.041558`, stabilization `0.10`, confirmation `0.10`.
- anchored-tuned: broad `0.35`, promotion `0.429221`, repair `0.020779`, stabilization `0.10`, confirmation `0.10`.

## Multi-Round Composition

The selected one-round configuration for each profile was run for two durable increments under the same total 385-pair budget:
- fixed: `50/50`, with the same full policy in both rounds.
- role-specialized: `60/40`, with a discovery-heavy first round that defers repair and a second round that enables repair.

Selection uses a Pareto-normalized compromise over mean, p95, and worst regret, unresolved shapes, prepared candidates, and unknown pairs.

| Initialization | Selected schedule | Pair split | Mean log regret | P95 | Worst | Unresolved | Prepared | Unknown |
|---|---|---|---:|---:|---:|---:|---:|---:|
| blind | fixed | 50/50 | 0.8290 | 1.9195 | 2.1841 | 0 | 91 | 561 |
| anchored-untuned | role-specialized | 60/40 | 0.1958 | 0.5474 | 0.8286 | 0 | 20 | 130 |
| anchored-tuned | role-specialized | 60/40 | 0.0673 | 0.2729 | 0.3846 | 0 | 23 | 79 |

Blind fixed scheduling has slightly worse mean regret than role specialization but materially better p95, worst, and unknown-pair cost, so the robust compromise selects fixed. Anchored-untuned role specialization improves mean regret and unknown coverage at the cost of seven additional prepared candidates. Anchored-tuned role specialization improves mean and p95 regret while preserving worst regret.

## Targeted Native Evidence

Two hybrid finalist passes measured only pairs absent from the compatible oracle:
- `out/grid100_policy_hybrid_20260712.sqlite`: 32 exact pairs, 21 successful ten-sample timings, and 11 attributable build failures.
- `out/grid100_policy_hybrid_round2_20260712.sqlite`: 48 exact pairs, 40 successful ten-sample timings, and 8 attributable build failures.

Together they add 80 durable exact outcomes, 61 positive timings, 610 samples, and 19 generic build-failure observations. After the second merge and rerun, canonical selected-policy unknown counts are 180 blind, 25 anchored-untuned, and 89 anchored-tuned. These remain explicit missing oracle pairs. The targeted evidence materially reduced anchored uncertainty without requiring exhaustive native completion of every replay proposal.

## Singleton Default

`scripts/tune_singleton_policy.py` uses the same acquisition schema on five representative shapes, three deterministic seeds, 32 disclosed seed measurements, and 16 equal shortlist requests. Results are in `out/grid100_singleton_policy_tuning_20260712.json`.

| Policy | Trials | Mean log regret | P95 | Worst |
|---|---:|---:|---:|---:|
| existing surrogate | 15 | 0.02106 | 0.10920 | 0.14871 |
| bundle information 0.05 | 15 | 0.01027 | 0.07091 | 0.09227 |
| bundle information 0.10 | 15 | 0.01027 | 0.07091 | 0.09227 |
| bundle information 0.25 | 15 | 0.01027 | 0.07091 | 0.09227 |

All bundle settings tie and dominate the existing surrogate on mean, p95, and worst regret. The selected singleton default is therefore bundle acquisition with information weight `0.05`, the smallest winning value.

Production family-QD keeps mechanical covering for cold start. Once a singleton campaign has at least 24 exact positive observations, it fits the contextual pair model and applies singleton bundle acquisition to the oversized generated pool. Proposal metadata records `selection_method = "singleton-bundle-acquisition"`. Disabling the policy or insufficient evidence falls back to the existing surrogate path.

## Limitations

This is candidate-selection and allocation evidence over the retained candidate catalog, augmented by targeted real exact pairs. It is not an end-to-end proof that proposal operators will generate the same candidate sets in a fresh real campaign. Blind replay remains sensitive to which historical candidates enter its initial 80-candidate calibration set, although stable seeded ordering prevents unrelated catalog insertions from reordering existing candidates. Final production claims still require current exact validation, timing, stabilization, confirmation, and complete deployment selection.
