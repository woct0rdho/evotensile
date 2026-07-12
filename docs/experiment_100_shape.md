# 100-Shape Campaign Experiment Log

This document is the historical log for the completed gfx1151 FP16 NT HHS 100-shape campaign work. It records experiment conditions, retained evidence, ordered P01-P14 results, selected policies, and artifacts. Stable subsystem behavior belongs to the focused design documents under `docs/`.

The controlled campaign concluded after P14. The next objective is to improve the existing 100-shape result for practical use, not to rerun the same controlled experiment from scratch. Current direction is tracked in `docs/plan.md`.

## Experiment Scope

The campaign studied the 100-shape Cartesian grid defined by the `gfx1151-nt-hhs` profile:

```text
M:     512, 640, 896, 1024
N:     128, 256, 512, 768, 1024
batch: 1
K:     256, 512, 1024, 2048, 4096
```

The experiment was non-blind. It allowed retained 100-shape candidates and measurements, installed hipBLASLt selections, hand-authored configurations, and the guarded SIA3/no-store-priority configuration from the FeatherOps investigation. That normalized candidate is `cand_07ba5e67b99df4ba` and has retained timing evidence on all 100 shapes.

Imported configurations were allowed as seeds, parents, controls, local-search centers, and measured incumbents. The generic strategy was not allowed to turn known winners into hidden validity rules, winner-specific linkage, hard-coded default bundles, or inferred performance on unmeasured pairs.

All replay comparisons used exact query-causal evidence:
- an exact retained `(candidate, shape)` pair could answer only after that pair was requested.
- a missing retained pair remained unknown.
- neighboring measurements and model predictions could prioritize a request but could not answer it.
- screening evidence was not treated as a final production claim.
- production eligibility still required current exact validation, timing, confirmation, and artifact registration.

The shared design resulting from these conditions is documented in:
- `docs/multi_shape_campaign_control.md`.
- `docs/exact_pair_scheduling.md`.
- `docs/pair_evaluators.md`.
- `docs/staged_round_controller.md`.
- `docs/deployment_selection.md`.

## Evidence Before P01

### Initial Pilot Search

The first pilot used the former standalone TensileLite client path:
- `135` proposed candidates across `100` shapes.
- `13,500` planned candidate-shape pairs in five batches.
- `1265.21s` wall time.
- `75,000` successful timing samples for `7,500` accepted pairs.
- `200` validation-failed pairs and `5,800` rejected observations.
- `17.178s` of summed successful GEMM time.

Compilation, client startup, validation, logging, and ingestion dominated wall time. This motivated shared preparation, exact sparse scheduling, and explicit cost-aware admission.

### Historical Top-Four Retiming

The top four screened candidates for every shape were rerun before adaptive timing existed:
- `400` intended exact pairs.
- `4,000` successful timing samples.
- `675.86s` wall time.
- the winner changed on `57/100` shapes.
- final winners had first-pass ranks 1, 2, 3, and 4 for `43`, `27`, `17`, and `13` shapes.

This showed that one-pass top-one screening was not reliable enough for final assignment.

### Imported Baselines And Repair Evidence

The retained corpus included:
- `15,204` successful samples and `816` rejections from historical outlier repair.
- installed hipBLASLt selections for all 100 shapes.
- `22` unique installed hipBLASLt candidates and `1,000` scheduled timing samples.
- hand-tuned candidates, including the guarded SIA3/no-store-priority configuration.

### Historical Corpus And Canonical Migration

The historical source database is:

```text
out/grid100_full_20260618_repaired.sqlite
```

At the initial consolidation point it contained:
- `219` canonical candidates.
- `100` shapes.
- `8,728` successful exact candidate-shape pairs.
- `165,604` successful timing samples.
- `6,616` source-backed rejections.
- `200` historical failed-validation events.
- benchmark protocol `bproto_9f4055f5f13232a3`.

A one-time canonical migration converted legacy integer encodings of `ExpandPointerSwap`, `MIArchVgpr`, and `SourceSwap` to JSON booleans. It changed `183` candidate hashes without collisions while preserving candidate integer IDs and all evidence rows. The local map is `out/grid100_boolean_migration_20260711.json`.

The historical corpus intentionally did not synthesize current validation passes. It remained suitable as a replay oracle and imported-candidate catalog, not as direct production confirmation.

### Winner Reproduction

Four historical winners spanning the retained timing range were rebuilt with the current TensileLite checkout, validated through the hipBLASLt oracle, and measured with 30 fresh main-protocol samples. Acceptance used `max(5%, 3 * historical relative MAD)`.

| Shape | Historical TFLOP/s | Fresh TFLOP/s | Speed error | Bound |
| --- | ---: | ---: | ---: | ---: |
| `m512_n128_b1_k256` | `5.028` | `5.063` | `0.70%` | `5.00%` |
| `m640_n512_b1_k1024` | `19.451` | `19.792` | `1.75%` | `5.00%` |
| `m896_n768_b1_k2048` | `28.711` | `28.744` | `0.12%` | `5.00%` |
| `m1024_n1024_b1_k4096` | `38.672` | `38.353` | `0.83%` | `6.00%` |

All four passed. The scheduler also validated all 16 cross-product pairs between those candidates and shapes and recorded 480 fresh samples. The report is `out/grid100_winner_reproduction_20260711/report_normalized.json`.

### Historical Installed-Library Check

An earlier GridBased update and rebuilt hipBLASLt installation produced:
- `1024^3` NT improvement from `16.007` to `23.434 TFLOP/s` for `torch_mm_NT`.
- improvement from `15.998` to `23.465 TFLOP/s` for `torch_linear_NT`.
- direct hipBLASLt NT improvement from `14.417` to `25.554 TFLOP/s`.
- direct NT speedups of `1.829x`, `2.218x`, and `4.804x` at square sizes `2048`, `4096`, and `8192` relative to the earlier reference.
- correctness passes on six curated target/off-grid cases.
- an upstream `hipblaslt-test` pass after excluding known no-solution availability families.

These were historical validation results, not actions performed during P01-P14.

## Ordered Campaign Results

### P01-P05: Shared Exact Campaign Infrastructure

P01-P05 replaced one-shape-specific control with shared campaign infrastructure:
- one checkpointed `CampaignControllerState` for phases, exact pair state, incumbents, artifacts, clustering, workload state, costs, and traces.
- one exact-oracle replay state shared by singleton and multi-shape simulation.
- explicit `PairRequest(candidate, shape)` scheduling with no inferred Cartesian evaluation product.
- a generic campaign runner with durable checkpoints.
- replay, real, and hybrid evaluators using one controller-facing result contract and labeled provenance.

The old `evotensile/campaign/one_shape.py` implementation was removed in favor of the shared runner. Design details are in `docs/multi_shape_campaign_control.md`, `docs/exact_pair_scheduling.md`, and `docs/pair_evaluators.md`.

### P06: Mechanical Shape Clustering

Deterministic candidate-independent descriptors and clustering were evaluated against the retained oracle. Fixed-count 16 clustering was selected as the moderate-cost baseline:
- `55.2%` representative promotion precision at 5% regret tolerance.
- `44.8%` missed specialists among assessed shapes.
- `2.39%` median assessed regret.

Representative-only transfer was therefore insufficient. Results are in `out/grid100_shape_clustering_20260712.json` and `docs/shape_clustering.md`.

### P07: Exact Promotion Racing

The selected observed-evidence racer combined nearest, representative, specialist, and adjacent-cluster broad lanes with exact probe and main requests. In retained replay it:
- resolved all 100 shapes.
- issued 385 probe pairs and 166 main pairs after 3,472 representative seed requests.
- achieved `27.5%` promotion precision.
- ended at mean log regret `0.0879` and worst log regret `0.6432` over resolved shapes.

This characterized the transfer mechanism. It did not make complete one-round resolution a production requirement. Results are in `out/grid100_shape_promotion_20260712.json` and `docs/shape_promotion_racing.md`.

### P08: Contextual Exact-Pair Model

The shared bootstrapped ExtraTrees model was selected because it predicted unseen candidates and shapes, modeled validity, and exposed calibrated tree samples. Retained five-fold evaluation reported:
- held-candidate MAE `0.396` and 90% interval coverage `0.876`.
- held-shape MAE `0.231` and coverage `0.904`.
- masked-pair MAE `0.249` and coverage `0.868`.

The evaluation artifact is `out/grid100_pair_model_20260712.json`. Design and complete baselines are in `docs/contextual_pair_model.md`.

### P09: Shared-Cost Bundle Acquisition

The balanced one-wave characterization added 385 exact pairs, resolved 67 shapes, prepared 83 candidates, and achieved:
- mean resolved-shape log regret `0.0908`.
- p95 log regret `0.3783`.

It improved mean and tail quality over observed transfer and independent model ranking while preparing far fewer new candidates. Results are in `out/grid100_bundle_acquisition_20260712.json` and `docs/shared_bundle_acquisition.md`.

### P10: Staged Soft-Deadline Rounds

Reusable broad, promotion, repair, stabilization, and confirmation phases were implemented with cumulative phase deadlines, explicit reserves, conservative admission, exact pending-wave persistence, resume-before-replan behavior, replay simulated time, and overrun stopping. Admitted work drained normally instead of receiving shrinking lower-layer timeouts.

The resulting execution contract is documented in `docs/staged_round_controller.md`.

### P11: Integrated Weak-Shape Repair

Equal-wave retained replay compared broad continuation with bounded weak-shape repair. Repair improved:
- mean log regret from `0.07376` to `0.06394`.
- p95 log regret from `0.38236` to `0.31275`.
- worst log regret from `0.66922` to `0.63070`.

Repair remained a small reserved phase rather than a replacement for broad acquisition. Results are in `out/grid100_repair_acquisition_20260712.json` and `docs/search_outlier_repair.md`.

### P12: Policy Tuning And Compatible Oracle

A consolidated read-only oracle was created at:

```text
out/grid100_compatible_20260712.sqlite
```

Its manifest is `out/grid100_compatible_20260712_manifest.json`. After compatible baseline and targeted hybrid overlays, the snapshot contained 224 candidates, 100 shapes, 17,171 benchmark events, 168,214 samples, and 8,858 positive exact pairs. Missing compatible pairs remained unknown.

Twelve generic configurations were compared across blind, anchored-untuned, and anchored-tuned initialization. The selected one-round policies were:

| Initialization | Policy identity | Mean log regret | P95 | Worst | Mean unresolved |
| --- | --- | ---: | ---: | ---: | ---: |
| blind | `campaign_policy_46baa1a9` | `0.5641` | `1.4444` | `2.1841` | `16.67` |
| anchored-untuned | `campaign_policy_89ea03a4` | `0.1998` | `0.5073` | `0.8286` | `0` |
| anchored-tuned | `campaign_policy_e7961c9f` | `0.0757` | `0.2732` | `0.4010` | `0` |

Two-round selection retained fixed `50/50` scheduling for blind and role-specialized `60/40` scheduling for both anchored regimes. The complete artifact is `out/grid100_policy_tuning_20260712.json`. Details are in `docs/campaign_policy_tuning.md`.

The singleton comparison selected shared bundle acquisition with information weight `0.05` after at least 24 positive observations. Its artifact is `out/grid100_singleton_policy_tuning_20260712.json`.

### P13: Explicit Workload Weighting

Three stable anchored-untuned replays compared uniform allocation with an explicit workload proportional to untuned baseline latency. Workload weighting shifted top-quartile pair allocation from `22.7%` to `26.6%`, reduced mean unknown pairs from `148.7` to `128.7`, and modestly improved:
- unweighted mean regret from `0.19079` to `0.18810`.
- workload-weighted mean regret from `0.26605` to `0.26502`.
- unweighted p95 regret from `0.53466` to `0.52816`.

The effect did not justify changing the default. Uniform weighting remained default. Explicit workload mode remained available with persisted provenance. Results are in `out/grid100_workload_weighting_20260712.json` and `docs/workload_weighting.md`.

### P14: Final Confirmation And Deployment Selection

Three stable anchored-untuned replays stabilized posterior-close finalists, refit the contextual model, and confirmed exact finalists under a 300-second soft budget. All trials retained complete positive coverage for 100 shapes.

Zero-tolerance deployment selected 11-12 exact confirmed winners, averaging:
- `11.33` solutions.
- `10.33` multi-shape generalists.
- one specialist shape.

Tolerances from 1% through 5% produced no additional reduction because the exact winners were already broadly shared. The artifact is `out/grid100_deployment_selection_20260712.json`. Production confirmation and export requirements are in `docs/deployment_selection.md`.

## Experiment Conclusion

The experiment established that the retained 100-shape result can be managed as one exact, cost-aware campaign rather than 100 isolated searches. The selected mechanisms support sparse exact evaluation, shared artifact preparation, measured promotion, contextual allocation, bounded repair, explicit workload priorities, fresh final confirmation, and deterministic deployment assignment.

The evidence also established important limits:
- retained-oracle success is candidate-selection and allocation evidence, not proof that a fresh proposal run regenerates the same catalog.
- missing historical pairs remain unknown and may require targeted native execution.
- model predictions and screening incumbents are not production evidence.
- nonzero deployment tolerance is useful only when it actually reduces the confirmed solution bank and its measured loss is acceptable.

Practical improvement should start from the existing candidates, compatible oracle, selected anchored policies, workload information when available, and current deployment assignment. New native work should be targeted at exact weak-shape, missing-pair, stabilization, confirmation, and artifact-completion needs rather than repeating the full campaign. Production logic generation and any hipBLASLt rebuild or installation remain separate operational actions requiring explicit approval.

## Artifact Index

- Historical source DB: `out/grid100_full_20260618_repaired.sqlite`.
- Compatible oracle: `out/grid100_compatible_20260712.sqlite`.
- Compatible oracle manifest: `out/grid100_compatible_20260712_manifest.json`.
- Boolean migration map: `out/grid100_boolean_migration_20260711.json`.
- Winner reproduction: `out/grid100_winner_reproduction_20260711/report_normalized.json`.
- Shape clustering: `out/grid100_shape_clustering_20260712.json`.
- Shape promotion: `out/grid100_shape_promotion_20260712.json`.
- Contextual pair model: `out/grid100_pair_model_20260712.json`.
- Bundle acquisition: `out/grid100_bundle_acquisition_20260712.json`.
- Repair acquisition: `out/grid100_repair_acquisition_20260712.json`.
- Policy tuning: `out/grid100_policy_tuning_20260712.json`.
- Singleton policy tuning: `out/grid100_singleton_policy_tuning_20260712.json`.
- Workload weighting: `out/grid100_workload_weighting_20260712.json`.
- Deployment selection: `out/grid100_deployment_selection_20260712.json`.
