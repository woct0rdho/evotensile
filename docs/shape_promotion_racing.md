# Shape Promotion Racing

This document defines the policy-visible exact-pair promotion racer in `evotensile/campaign/promotion.py`. Mechanical descriptors and medoids are documented in `docs/shape_clustering.md`. Evaluator provenance and overlay rules are documented in `docs/pair_evaluators.md`.

## Boundary

Promotion is an evaluation decision. It never copies a source performance to a destination, inserts a predicted destination winner, or marks an unmeasured pair as resolved. Every destination pair is an explicit `PairRequest`, and only its evaluator outcome can update the destination incumbent.

The racer consumes:
- the registered shapes and deterministic clustering.
- the current `CampaignControllerState`.
- policy-visible successful `PairEvaluationOutcome` records.
- one replay, real, or hybrid `PairEvaluator`.
- an immutable `PromotionPolicy`.

It produces exact probe and main results plus an auditable event for every proposed destination pair.

## Promotion Lanes

`plan_shape_promotions()` fills a bounded per-shape request set through four generic lanes:
- `specialist`: the nearest measured source's winner, protected by explicit specialist slots.
- `nearest`: source winners and near-winners from the mechanically nearest measured shapes.
- `representative`: top measured candidates from the destination cluster's medoid.
- `broad`: candidates with successful evidence on multiple shapes and multiple clusters, transferred only from the destination cluster or mechanically adjacent clusters.

Source rankings use disclosed throughput only. Near-winner eligibility is relative to the best disclosed source result. Pair deduplication is exact and deterministic. Specialist, nearest, representative, and broad priorities arbitrate overlapping rationales without creating duplicate requests.

A candidate is not promoted again to an already queried destination. A candidate-cluster lane is blocked when policy-visible evidence in that cluster is worse than the local incumbent by more than `stop_regret_fraction`. This is an observed-evidence stopping rule, not a hidden oracle rule.

## Mechanical Neighborhoods

Nearest source shapes use standardized distances reconstructed from the persisted mechanical descriptors. Adjacent clusters use the same metric between medoids. Stable shape and cluster IDs break all ties.

The selected defaults are:
- neighbor depth `2`.
- representative finalists `3`.
- maximum promotions per destination `6`.
- specialist slots `1`.
- broad slots `2`, requiring evidence on at least two shapes and two clusters.
- adjacent-cluster depth `1`.
- source near-winner tolerance `2%`.
- observed cluster-stop regret `30%`.

## Probe And Main Stages

`execute_promotion_race()` runs two durable waves:
- materialize every admitted promotion as a one-sample `EvidenceStage.PROBE` request.
- apply the probe result to the controller.
- retain candidates within the destination's probe regret threshold, while enforcing protected specialist slots and a general survivor floor.
- top surviving exact pairs up to the main sample target with `EvidenceStage.SCREENING` requests.
- apply the main result and append promotion audit events.

The selected probe threshold is `5%`, the survivor floor is `1`, and the main target is three total samples. Replay top-ups now preserve target-sample semantics: a probe inserts one sample, a main request for three inserts only two additional samples. Replay phase cost uses exact pair throughput and protocol launch counts for each admitted stage.

## Shared Artifact Bundles

Before the probe wave, destinations are grouped by candidate. One explicit `artifact_shapes_by_candidate` scope contains every admitted destination for that candidate. The same full scope is reused for the main wave even when only some destinations survive, preserving compile-cache identity and avoiding a false artifact contraction or rebuild.

Events record whether preparation was already available or shared across multiple destinations. Artifact reuse never creates evidence for unrequested combinations.

## Audit Events

Each `PromotionEvent` records:
- source and destination shape.
- source and destination cluster.
- candidate and lane.
- final stage (`main` or `probe_rejected`).
- artifact scope and preparation reuse.
- status and destination performance.
- realized gain over a pre-race destination incumbent when one existed.
- success, defined as becoming the final destination incumbent and improving the pre-race incumbent, or resolving a previously unresolved destination as its final incumbent.

Events are appended to the controller trace after durable evaluator results.

## Singleton Law

With one shape there is no distinct source or destination. Planning returns no promotions, performs no evaluator call, adds no hyperparameter effect, and leaves the existing singleton race unchanged.

## Controlled Replay Ablations

The retained exact oracle was replayed from a fresh overlay for every policy with the same:
- 100 shapes.
- 217 imported candidates.
- 15,344 retained exact pairs.
- deterministic 16-cluster assignment.
- dense candidate evaluation on the 16 medoids before promotion.

Machine-readable results are stored in `out/grid100_shape_promotion_20260712.json`. Unavailable retained pairs remained unknown. Mean and worst log regret are over resolved shapes, so unresolved count must be read with them.

| Policy | Resolved | Total queries | Probe pairs | Main pairs | Promotion precision | Mean log regret | Worst log regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| representative seed only | 16 | 3,472 | 0 | 0 | n/a | 0.0000 | 0.0000 |
| nearest depth 1 | 94 | 3,726 | 254 | 152 | 37.9% | 0.1117 | 1.1821 |
| nearest depth 2 | 95 | 3,804 | 332 | 167 | 32.4% | 0.0866 | 0.6432 |
| nearest depth 3 | 95 | 3,847 | 375 | 183 | 30.2% | 0.0846 | 0.6432 |
| representative finalist 1 | 89 | 3,609 | 137 | 92 | 70.2% | 0.1396 | 1.1821 |
| representative finalists 3 | 94 | 3,736 | 264 | 147 | 34.7% | 0.0929 | 0.6432 |
| representative finalists 5 | 94 | 3,828 | 356 | 177 | 25.3% | 0.0916 | 0.6432 |
| selected combined | 100 | 3,857 | 385 | 166 | 27.5% | 0.0879 | 0.6432 |

Threshold and floor ablations show:
- a `5%` probe threshold preserves the deterministic replay result while reducing main top-ups relative to `10%` and `20%`.
- a `2%` source near-winner threshold is slightly better than `5%` or `10%` at similar query cost.
- survivor floor `1` avoids unnecessary top-ups relative to floor `2` in exact replay.
- a `10%` or `20%` observed stop leaves shapes unresolved. `30%` reaches all 100 with modest additional query cost.

The combined policy is selected for the P07 mechanism characterization because it reaches complete retained-oracle coverage while preserving explicit specialist, representative, nearest, and adjacent-cluster broad lanes. Nearest depth 3 has slightly lower regret over its 95 resolved shapes, but it does not close the five missing shapes. This replay coverage is not a requirement for one five-minute round: a real round is allowed to leave many shapes unresolved while gathering useful evidence for later rounds.

## Limitations

- Promotion scores are deterministic observed heuristics. P08 replaces them with calibrated pair-specific posterior estimates.
- The retained matrix is sparse, so unknown promotion outcomes cannot distinguish invalidity from absent historical measurement.
- Exact replay has no timing noise. Real campaigns still require stabilization and confirmation.
- One promotion wave cannot exhaust all potentially useful graph paths. Later staged rounds may recompute from newly disclosed evidence.
