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

## Practical Production-Improvement Campaign

The controlled P01-P14 experiment remains complete. A separate practical campaign is now active to improve the retained 100-shape result until additional configuration search no longer produces significant robust gains.

### Operating Rules

- Start every round from all compatible retained and newly measured evidence. Do not restart the search or densely remeasure the existing candidate catalog.
- Use `out/grid100_production_search_20260712.sqlite` as the mutable campaign database. It starts as a copy of `out/grid100_compatible_20260712.sqlite`, which already consolidates the historical, untuned/tuned baseline, and both policy-hybrid evidence overlays.
- Submit explicit sparse `(candidate, shape)` requests. Preparing one candidate for several shapes may share work, but artifact scope must not imply unrequested timing pairs.
- Give new candidates short exact screening on shapes where their parent or model makes them plausible. Promote only measured improvements or statistically close candidates to additional shapes and samples.
- Keep ordinary rounds near a five-minute soft budget. Admitted builds, validation, and timing drain normally so timeout pressure does not discard evidence or corrupt ingestion.
- Preserve full GPU-oracle validation as a hard gate. Historical timing may guide proposals and define comparison thresholds, but production assignment still requires current compatible validation and fresh confirmation.
- Keep production GridBased generation, source overwrite, and hipBLASLt rebuild/install as separate approval-gated actions after convergence.

### Round Strategy

- Establish the consolidated incumbent and uncertainty report from retained exact evidence, including winner frequency, candidate coverage, close gaps, and noisy shapes.
- Run incumbent-centered trust-region interaction sweeps. Freeze each strong structural family and vary a bounded set of scheduling/store parameters, initially emphasizing `ScheduleIterAlg`, `StorePriorityOpt`, `NumElementsPerBatchStore`, and `StoreVectorWidth`.
- Fit the contextual pair model to all disclosed compatible exact evidence and use shared-cost acquisition to choose only high-value unknown child-shape pairs. Limit each child initially to shapes where its parent is incumbent or close to incumbent.
- Promote measured improvements across mechanically nearby or parent-competitive shapes. Do not infer transfer performance from shared artifacts or neighboring results.
- Repair weak or noisy shapes with focused structural and interaction neighborhoods, then retime close finalists with paired or blocked controls when ordinary pooled timing is insufficient.
- Refit and repeat after every round. Change parameter neighborhoods, acquisition weights, or proposal mechanics when round evidence identifies a systematic blind spot or execution inefficiency.
- After search convergence, freshly validate and confirm the final candidate bank on all assigned exact shapes, then produce a reviewed deployment-selection artifact. Do not generate production logic without explicit approval.

### Initial Evidence Audit

The pre-round audit of the compatible database found:
- `224` candidates, `100` shapes, `17,171` benchmark events, `168,214` samples, and `8,858` positive exact pairs.
- `24` distinct historical per-shape winners. The most frequent winner covered `14` shapes, so the retained optimum is not one homogeneous configuration family.
- several winner candidates have exact timing on only `16-37` shapes, leaving useful targeted transfer pairs without justifying dense 100-shape evaluation.
- several top-two gaps are below one percent, while some current winners have relative MAD above two percent. These shapes require stabilization rather than treating a pooled median ordering as final.

### Implemented Round Infrastructure

The practical campaign added:
- `evotensile/search/trust_region.py`, which deterministically enumerates bounded interaction grids around complete measured parents while preserving lineage, scope eligibility, and optional linked repair.
- `scripts/run_grid100_practical_round.py`, which initializes one consolidated mutable DB, reconstructs all compatible exact evidence, builds parent-competitive target sets, fits the contextual pair model, excludes known pairs, selects shared-cost bundles, executes exact GPU-validated requests, and appends auditable plans/reports to the campaign manifest.
- interaction profiles for store/scheduling, staging/prefetch, mapping/stagger, and read/vectorization neighborhoods, plus a promotion mode for measured children and their explicit comparison parents.
- an explicit random seed default of `12345`. Later round seeds use visibly deliberate incremental values rather than date-like or architecture-like values.
- `scripts/finalize_grid100_production_search.py`, which freshly validates and measures current contenders plus the original compatible winner control, requires complete selected artifacts, and writes zero- and nonzero-tolerance deployment assignments without generating production logic.

### Native Search Rounds

All 25 rounds operated on `out/grid100_production_search_20260712.sqlite`, selected only previously unknown exact pairs, and retained all positive and negative evidence. They requested 899 pairs from 105 candidate-round bundles. Ordinary rounds completed well below five minutes. Admitted work always drained and was ingested.

| Round | Strategy | Pairs | Candidates | OK | Validation failed | >1% shapes | Maximum gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `round01_store_interactions` | store interaction | 48 | 6 | 40 | 8 | 8 | 9.39% |
| `round02_promote_nepbs20` | measured promotion | 12 | 1 | 12 | 0 | 2 | 2.09% |
| `round03_expanded_store_sweep` | expanded store interaction | 48 | 5 | 48 | 0 | 2 | 3.09% |
| `round04_promote_nepbs24` | measured promotion | 13 | 1 | 13 | 0 | 1 | 1.40% |
| `round05_staging_interactions` | staging interaction | 48 | 7 | 48 | 0 | 19 | 12.43% |
| `round06_promote_staging_winners` | measured promotion | 45 | 4 | 45 | 0 | 4 | 14.61% |
| `round07_mapping_interactions` | mapping interaction | 48 | 6 | 48 | 0 | 3 | 2.45% |
| `round08_promote_mapping_winners` | measured promotion | 25 | 2 | 25 | 0 | 1 | 6.31% |
| `round09_vector_interactions` | vector interaction | 48 | 5 | 36 | 12 | 1 | 1.11% |
| `round10_store_after_staging` | conditional store interaction | 48 | 6 | 48 | 0 | 0 | 0.43% |
| `round11_staging_convergence_probe` | staging interaction | 48 | 6 | 48 | 0 | 4 | 2.04% |
| `round12_promote_pgr1_variant` | measured promotion | 4 | 1 | 4 | 0 | 0 | 0.98% |
| `round13_mapping_convergence_probe` | mapping interaction | 48 | 6 | 48 | 0 | 0 | 1.00% |
| `round14_vector_convergence_probe` | vector interaction | 48 | 6 | 36 | 12 | 11 | 3.75% |
| `round15_promote_vector_winners` | measured promotion | 24 | 2 | 24 | 0 | 5 | 4.04% |
| `round16_vector_refit_probe` | vector refit | 48 | 5 | 36 | 12 | 1 | 1.34% |
| `round17_promote_grvwb4_vwb2` | measured promotion | 16 | 1 | 16 | 0 | 2 | 4.10% |
| `round18_vector_final_refit` | vector refit | 48 | 5 | 48 | 0 | 2 | 4.84% |
| `round19_promote_final_vector_children` | measured promotion | 20 | 2 | 20 | 0 | 1 | 2.76% |
| `round20_vector_exhaustion_probe` | vector refit | 48 | 7 | 37 | 11 | 2 | 5.11% |
| `round21_promote_grvwa4` | measured promotion | 8 | 1 | 8 | 0 | 0 | 0.00% |
| `round22_final_store_probe` | conditional store interaction | 48 | 5 | 48 | 0 | 1 | 1.39% |
| `round23_promote_nepbs24_vector_family` | measured promotion | 12 | 1 | 12 | 0 | 0 | 0.69% |
| `round24_mapping_convergence_restart` | mapping convergence | 48 | 7 | 48 | 0 | 0 | 0.68% |
| `round25_staging_convergence_restart` | staging convergence | 48 | 7 | 48 | 0 | 0 | 0.00% |

The most important findings were conditional rather than universal:
- changing `NumElementsPerBatchStore` from 10 to 20 in one strong family improved several `N=512-1024` shapes by 1-9%, but was substantially slower on square and some `M=512` cases.
- staging changes were the largest source of improvement. `PrefetchGlobalRead 1->2` and one `DepthU 32->16` child produced gains up to 14.61%. Promotion confirmed other shape-specific gains but also large regressions outside their regimes.
- mapping/stagger changes mostly produced sub-percent effects, with isolated gains through 6.31% after promotion.
- the late vector basin required conditional combinations. `GlobalReadVectorWidthB 8->4` with `VectorWidthB 2->4`, then `VectorWidthB 4->2`, and later `VectorWidthA=2` produced distinct shape-local gains. None was safe to generalize across families.
- the final conditional store change on the vector family produced one 1.39% gain, but promotion found no further gain above 0.69%.

Correctness failures were also conditional. Four generated children failed validation on every requested pair, totaling 43 failed exact pairs. The observed patterns included one `ScheduleIterAlg=3`/NEPBS8 child and several `VectorWidthA=2` children in specific families, while `VectorWidthA=2` was valid and fast in another family. These remain structured anomaly evidence. The campaign did not convert them into global validity rules or patch TensileLite.

### Convergence And Final Confirmation

After the last measured promotion, two diverse current-incumbent proposal rounds established broad-search convergence:
- `round24_mapping_convergence_restart` measured 48 new valid pairs and found no gain above 0.68%.
- `round25_staging_convergence_restart` measured 48 new valid pairs and found no improvement.

The corrected final run at `out/grid100_production_search_20260712/finalization_v2/` freshly measured the current contenders and mandatory original-compatible-winner control for every shape in one session:
- 298 exact candidate-shape pairs across 73 candidates.
- all 298 pairs passed fresh hipBLASLt GPU-oracle validation.
- 8,940 fresh main-protocol samples, 30 per pair.
- every selected pair has a registered artifact.
- zero-tolerance selection improved 69/100 shapes against the freshly measured original winner controls, with 43 gains of at least 1%, 3.19% mean gain, 0.45% median gain, and 24.58% maximum gain. The minimum same-session gain is zero because the original control remains eligible.

The zero-tolerance assignment uses 45 solutions. Optional consolidation produces:
- 34 solutions at 0.5% tolerance, with 0.036% mean measured loss and 0.445% worst loss.
- 30 solutions at 1% tolerance, with 0.072% mean measured loss and 0.966% worst loss.
- 27 solutions at 2% tolerance, with 0.133% mean measured loss and 1.408% worst loss.

`finalization/` is superseded by `finalization_v2/` because the first finalization did not force the original compatible winner into every same-session contender group. Its data remains compatible and retained, but production decisions must use `finalization_v2/deployment_*.json`.

### Convergence Criteria

Configuration search is considered converged only after all of the following hold:
- at least two consecutive diverse proposal rounds produce no fresh validated improvement above one percent on any shape after stabilization.
- no unmeasured high-probability pair remains above the configured practical-improvement threshold in the contextual model or focused trust regions.
- weak/noisy shapes have stable finalists or an explicit evidence-backed explanation for remaining uncertainty.
- a final fresh confirmation round finds no material winner reversal and covers every intended deployment assignment with registered artifacts.

Sub-one-percent changes may still be retained when repeated paired evidence is stable and they do not increase deployment risk, but they do not by themselves reset the broad-search convergence counter.

## Experiment Conclusion

The experiment established that the retained 100-shape result can be managed as one exact, cost-aware campaign rather than 100 isolated searches. The selected mechanisms support sparse exact evaluation, shared artifact preparation, measured promotion, contextual allocation, bounded repair, explicit workload priorities, fresh final confirmation, and deterministic deployment assignment.

The evidence also established important limits:
- retained-oracle success is candidate-selection and allocation evidence, not proof that a fresh proposal run regenerates the same catalog.
- missing historical pairs remain unknown and may require targeted native execution.
- model predictions and screening incumbents are not production evidence.
- nonzero deployment tolerance is useful only when it actually reduces the confirmed solution bank and its measured loss is acceptable.

The practical configuration search is now converged under the stated criteria. Production decisions should use the fresh `finalization_v2` assignment rather than pooled historical rankings. The zero-tolerance assignment maximizes freshly confirmed speed. The 0.5%, 1%, and 2% artifacts provide explicit solution-count/loss tradeoffs for review. Production logic generation and any hipBLASLt rebuild or installation remain separate operational actions requiring explicit approval.

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
- Practical mutable evidence DB: `out/grid100_production_search_20260712.sqlite`.
- Practical campaign manifest: `out/grid100_production_search_20260712_manifest.json`.
- Practical round plans/reports and compile cache: `out/grid100_production_search_20260712/`.
- Superseded first finalization: `out/grid100_production_search_20260712/finalization/report.json`.
- Production finalization report: `out/grid100_production_search_20260712/finalization_v2/report.json`.
- Zero-tolerance assignment: `out/grid100_production_search_20260712/finalization_v2/deployment_0.000.json`.
- Consolidated assignments: `out/grid100_production_search_20260712/finalization_v2/deployment_0.005.json`, `out/grid100_production_search_20260712/finalization_v2/deployment_0.010.json`, and `out/grid100_production_search_20260712/finalization_v2/deployment_0.020.json`.
