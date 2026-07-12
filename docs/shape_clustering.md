# Mechanical Shape Clustering

This document defines deterministic workload clustering in `evotensile/search/shape_clustering.py` and the P06 replay baselines in `evotensile/campaign/baselines.py`. Clustering is a policy organization mechanism. It does not disclose evidence, infer an unmeasured winner, or add validity rules.

## Descriptor Contract

`mechanical_shape_descriptor()` computes candidate-independent features from each registered shape and target work-group-processor count:
- log M, N, K, batch, output size, and dimension ratios.
- reduction depth, reduction-to-output regime, and short-reduction response.
- GEMM arithmetic intensity from FP16 inputs and FP32 output traffic.
- coarse M/N/K alignment at 16 and 64 elements plus batched/non-batched compatibility.
- macro-tile fill, output-tile count, WGP rounds, and final-round WGP granularity across representative macro-tile families.

Macro-tile families come from the legal `MatrixInstruction` search-space domain. The selector deduplicates resulting macro tiles, chooses a central tile, and adds deterministic farthest tiles in log area/aspect space. It uses no historical candidate, timing, winner, or failure label.

Before clustering, every feature is centered and scaled across the requested workload. Constant features receive unit scale. Distances are Euclidean in this standardized descriptor space. The complete raw descriptor schema, selected macro-tile families, and configuration are serialized with the result.

## Fixed-Count Mode

Fixed-count mode uses deterministic k-medoids:
- choose the workload-wide medoid.
- add farthest-first initial medoids.
- assign each shape to its nearest medoid with shape-ID tie breaking.
- replace each medoid by the member minimizing within-cluster distance.
- repeat until stable or the configured iteration limit.

The requested count is capped at the number of shapes. Final cluster IDs are assigned after sorting by medoid shape ID, so input order does not affect identity.

`DEFAULT_SHAPE_CLUSTER_COUNT` is 16 for the current 100-shape experiment baseline. The count is an experiment policy default, not a search-space invariant.

## Distance-Threshold Mode

Threshold mode starts with singleton clusters and performs deterministic complete-link merges. Two clusters can merge only when every cross-cluster pair is within the threshold. This guarantees every final member is within the threshold of its cluster medoid. Merge ordering uses complete-link distance and stable member identities.

A threshold of zero separates distinct descriptors. A sufficiently high threshold produces one cluster.

## Singleton Law

For one shape, both modes return exactly:
- one `cluster_000`.
- the shape as its medoid.
- zero medoid distance.

Cluster count, threshold, macro-tile family count, and iterative fitting cannot alter the singleton assignment or representative. The shared one-shape campaign stores this degeneration in its controller state.

## Persistence

`ShapeClustering.to_dict()` records:
- complete clustering configuration.
- exact shape identity set.
- selected macro-tile families.
- raw descriptors by shape.
- cluster identity, medoid identity, member identities, and standardized distance to the medoid.

`CampaignControllerState.set_clustering()` requires clusters to partition exactly the registered shape set, requires every medoid to be a member, rejects overlap, records a transition, and persists the payload in summaries and current `v1` checkpoints. No compatibility layer is provided for earlier development layouts.

## Baseline Policies

The required P06 policy boundaries are explicit:
- `simulate_independent_shape_baseline()` from `evotensile/search/replay.py` runs isolated singleton controllers with no evidence or preparation sharing.
- `evaluate_global_candidate_dense_baseline()` materializes every requested candidate-shape pair and applies one evaluator result.
- `evaluate_representative_first_baseline()` materializes candidates only on cluster medoids and performs no transfer or promotion to members.

The representative-first controller therefore has incumbents only for queried medoids. Non-medoid shapes remain unresolved. This is intentional: the baseline measures the information and cost boundary before the measured promotion racer documented in `docs/shape_promotion_racing.md`.

## Oracle Characterization

`characterize_representative_promotions()` is an offline diagnostic, not policy-visible evidence. For each cluster it selects the best retained candidate on the medoid, then checks that candidate's exact retained pair on each member when available. It reports:
- explicit medoid request count and retained-known medoid count.
- assessed and unavailable member outcomes.
- precision within a configured regret tolerance.
- missed-specialist rate.
- median and worst exact regret.

The diagnostic never inserts hidden member evidence into a controller.

The retained 100-shape corpus characterization is stored in `out/grid100_shape_clustering_20260712.json`. It covers 100 shapes, 217 evidence-bearing candidates, and 15,344 exact pairs at a 5% regret tolerance.

| Mode | Clusters | Requested medoid pairs | Retained-known pairs | Precision | Missed specialists | Median regret | Unavailable shapes |
|---|---:|---:|---:|---:|---:|---:|---:|
| count 4 | 4 | 868 | 351 | 50.0% | 50.0% | 5.18% | 58 |
| count 12 | 12 | 2,604 | 1,053 | 51.0% | 49.0% | 3.31% | 51 |
| count 16 | 16 | 3,472 | 1,310 | 55.2% | 44.8% | 2.39% | 33 |
| count 20 | 20 | 4,340 | 1,802 | 68.7% | 31.3% | 0.00% | 33 |
| threshold 6 | 18 | 3,906 | 1,650 | 72.9% | 27.1% | 0.00% | 41 |
| threshold 4 | 36 | 7,812 | 3,112 | 68.3% | 31.7% | 0.00% | 18 |

Fixed count 16 is retained as the moderate-cost default baseline: it lowers median assessed regret and unavailable coverage relative to smaller fixed counts without paying the much denser threshold-4 or threshold-3 request cost. It does not make representative transfer sufficiently reliable by itself. A 44.8% missed-specialist rate among assessed shapes and worst regret above 200% clearly require P07's nearest-shape transfer, specialist lanes, and measured promotions rather than implicit cluster inheritance.

## Tests

Tests verify:
- deterministic output under input reordering.
- exact fixed-count partitions and member medoids.
- threshold separation and merging.
- descriptor coverage for arithmetic intensity, reduction, macro-tile fill, and WGP behavior.
- singleton degeneration across both configuration modes.
- checkpoint round-trip and invalid-partition rejection.
- representative-only versus dense explicit replay pair sets.
- a controlled synthetic specialist missed by representative-only evaluation.
