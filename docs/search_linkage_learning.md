# Linkage Design

This document describes EvoTensile's linkage learning: how validated evidence is converted into FOS groups for GOMEA-style proposal mixing. It does not describe the broader search loop or GOMEA execution details. Those live in `docs/search_algorithms.md` and `docs/search_gomea.md`.

## Purpose

TensileLite knobs are strongly epistatic. Mixing one field at a time often creates invalid or unhelpful candidates, while copying too much from one parent prevents useful recombination. Linkage learning identifies groups of genes that are mixed together based on validated performance evidence.

Learned linkage is proposal-only:
- It never shrinks `DOMAINS`.
- It never creates new validity rules.
- It never overrides `explain_invalid_nt_hhs()` or TensileLite validation.
- It only supplies additional FOS groups to GOMEA when there is enough DB evidence.

## Evidence Source

`learn_linkage_models_from_snapshot()` builds models through `load_candidate_evidence()` from the immutable evidence snapshot created once for the proposal call.

Evidence filtering uses:
- `status='ok'` timing rows from `evaluations`.
- The active `problem_type_hash` and `benchmark_protocol_hash` when supplied.
- Optional target shapes, so linkage can be learned for the shapes currently being scheduled.
- `min_samples`, which defaults to `1` for early campaigns and can be raised for mature campaigns.
- Validation-passed DB ranking semantics from `EvoTensileDB.rank_evaluations()`.

The loader retrieves candidate JSON from the `candidates` table and returns `CandidateEvidence` records containing the candidate, aggregate score, and total samples.

## Shape-Local Scoring

Raw time and GFLOP/s are not pooled directly across shapes. Instead, linkage learning uses per-shape rank percentiles:
- For each shape, rank candidates by median GFLOP/s, descending.
- Convert rank to percentile, where lower is better.
- Keep the top `elite_per_shape` candidates for each shape.
- Compute the generalist score across the complete requested shape set.
- Impute every unresolved target shape at worst percentile rather than ignoring it.
- Break equal generalist scores by larger sample count and then candidate hash. Coverage is reported explicitly.

This makes linkage evidence shape-aware without letting large shapes dominate because their absolute times are larger or letting a sparse specialist masquerade as broad breeding evidence. One-shape linkage is the same calculation over one shape.

## Truncation Pool

`select_truncation_pool()` keeps only the best fraction of finite scored genomes:
- `truncation_tau` defaults to `0.5`.
- The selected count is `max(1, int(n * truncation_tau))`.
- At least `min_samples` selected genomes are required. The default is `8`.
- If there is insufficient evidence, model learning returns no models and reports `insufficient_validated_evidence`.

The DB loader requests enough pre-truncation evidence to preserve this floor. `minimum_evidence_for_truncation()` computes at least `ceil(min_samples / truncation_tau)` and adjusts for integer truncation. For example, `tau=0.5` and `min_samples=8` load `16` candidates so the selected pool still contains `8`. This prevents an evidence cap from silently disabling learned linkage.

This positive truncation pool is the only input to learned linkage. Rejected, invalid, unvalidated, and low-performing generated candidates do not teach positive linkage.

## Basin Clustering

`leader_clusters()` splits the truncation pool before MI learning so structurally different basins do not get averaged into one model.

The implemented clustering is balanced leader clustering:
- Sort selected genomes by score.
- Use each unmatched high-ranked genome as a leader until `max_clusters` is reached.
- Assign a genome to an existing leader if Hamming distance is below the threshold.
- The default threshold is `max(2, int(n_genes * 0.3))`.
- When multiple leaders match, assign to the smallest matching cluster.
- Once the cluster limit is reached, assign remaining genomes to the nearest leader, again preferring smaller clusters.

`max_clusters` defaults to `8`. Singleton clusters are allowed. They produce univariate FOS groups only.

## Hybrid Mutual Information

For each non-singleton cluster, EvoTensile builds an MI matrix over genome columns with `hybrid_mi_matrix()`.

Nominal genes use raw domain indices. Ordinal-like genes are rank-binned before MI so relative order can contribute without assuming numeric distance in the raw domain index. The default ordinal set is:

```text
DepthU
GlobalSplitU
VectorWidthA
VectorWidthB
GlobalReadVectorWidthA
GlobalReadVectorWidthB
StoreVectorWidth
WorkGroupMapping
StaggerU
StaggerUStride
AssertFree0ElementMultiple
AssertFree1ElementMultiple
AssertSummationElementMultiple
```

`MatrixInstruction`, `WorkGroup`, `TransposeLDS`, booleans, and most enums remain nominal. `NumElementsPerBatchStore` is intentionally nominal because `0` has special auto/default meaning.

Ordinal bin count defaults to `4`. MI values are computed from empirical counts with natural logarithms.

## UPGMA FOS Construction

`upgma_fos()` turns each cluster's MI matrix into FOS groups:
- Start with every gene as a singleton group.
- Repeatedly merge the active group pair with highest average cross-MI.
- Stop when the best MI is at or below `mi_floor`.
- Omit the full all-genes group.
- Dedupe groups while preserving order.

The default `mi_floor` is `1e-6`, which suppresses merges driven by effectively zero MI.

Each `LinkageModel` records:
- `leader_genome` and optional `leader_candidate_hash`.
- `fos_groups` as tuples of genome indices.
- `cluster_size`.
- `evidence_count`.
- `mi_floor`.

The model summary written to schedule metadata includes leader hash, cluster size, evidence count, FOS group count, maximum FOS group size, and MI floor.

## GOMEA Assignment

`nearest_linkage_model()` assigns a GOMEA base genome to the model whose leader has the smallest Hamming distance. GOMEA then mixes with:
- source-backed static NT HHS groups.
- the assigned model's learned FOS groups.

If there is no model, GOMEA falls back to static groups plus a FOS built from current elite parent genomes. `docs/search_gomea.md` describes this mixing path.

## CLI Controls And Metadata

Learned linkage is enabled by default for GOMEA-style proposal modes. CLI controls are:

```text
--learned-linkage
--no-learned-linkage
--linkage-truncation-tau
--linkage-min-samples
--linkage-max-clusters
--linkage-ordinal-bins
```

`schedule-batches`, `repair-outliers`, and dry runs write linkage metadata into `schedule_metadata.json` or `repair_metadata.json`, including:
- Whether learned linkage was requested and enabled.
- Evidence and selected counts.
- Fallback reason, if any.
- Linkage model count and per-model summaries.
- Hyperparameters used for the learning pass.

This makes proposal behavior inspectable without replaying the run.

## Relationship To Search

Linkage learning is one internal component of GOMEA proposal generation. The scheduler still controls:
- Which shapes are being tuned.
- Which DB evidence is eligible.
- How many candidates GOMEA may propose.
- How candidates are batched and measured.
- How noisy timing finalists are topped up.
- Which measured candidates become winners.

Those broader search-loop responsibilities are documented in `docs/search_algorithms.md` and `docs/noisy_measurements.md`.
