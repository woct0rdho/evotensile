# GOMEA Design

This document describes EvoTensile's Gene-pool Optimal Mixing Evolutionary Algorithm (GOMEA) proposal operator. It assumes the candidate space and validity layer from `docs/nt_hhs_search_space.md`.

## Role In EvoTensile

GOMEA is a proposal generator used by `schedule-batches` through proposal modes such as `gomea`, `seed-random-gomea`, `evolutionary`, and `family-qd`. It produces new complete TensileLite candidates from validation-passed parent candidates already stored in the SQLite database.

GOMEA does not rank final winners and does not change validity. Its output still goes through candidate canonicalization, rule checks, TensileLite build/codegen, structured validation, timing ingestion, and DB ranking like any other proposal source.

## Genome Encoding

`evotensile/search/encoding.py` maps each candidate to a categorical genome:
- `PARAM_NAMES` is the ordered key list from `DOMAINS`.
- Each gene is the domain index for one parameter.
- Tuple/list values such as `MatrixInstruction` and `WorkGroup` are treated as single categorical genes.
- `genome_to_candidate(..., repair=True)` converts domain indices back to candidate parameters and runs linked repairs.
- Hamming distance over domain-index genomes is used for nearest-elite and linkage-model assignment.

The genome is a search representation only. The persisted identity is still the canonical candidate dictionary and `cand_...` hash.

## Static Family Of Subsets

GOMEA mixes genes by Family-of-Subsets (FOS) groups instead of mutating independent fields. EvoTensile always includes source-backed static linkage groups for known mechanical couplings:

```text
(MatrixInstruction, WorkGroup, DepthU, GlobalSplitU)
(TransposeLDS, LdsBlockSizePerPadA/B, LdsPadA/B)
(PrefetchGlobalRead, PrefetchLocalRead, 1LDSBuffer, ClusterLocalRead, VectorWidthB)
(GlobalReadVectorWidthA/B, VectorWidthA/B)
(ScheduleIterAlg, WorkGroupMapping, StaggerU, StaggerUStride, StaggerUMapping, SourceSwap)
(StorePriorityOpt, NumElementsPerBatchStore, StoreSyncOpt, GroupLoadStore, StoreVectorWidth)
(ExpandPointerSwap)
(AssertFree0ElementMultiple, AssertFree1ElementMultiple, AssertSummationElementMultiple)
```

These groups preserve important TensileLite couplings during mixing. They are defined once in `evotensile/search/semantics.py` and are also used by semantic mutation. Learned linkage models can add evidence-derived groups, but they do not replace these static groups.

## Neighborhood Sweep Operator

`gomea_neighborhood_candidates()` supports sampled and bounded micro-exhaustive one-parent neighborhoods around ranked elites:
- It forms `(parent, semantic-group)` slots across the selected parent pool.
- The default path preserves seeded sampled-neighborhood behavior.
- The opt-in micro-exhaustive path enumerates a complete Cartesian neighborhood when it fits the cap, otherwise randomized singleton and paired alternatives.
- Queried semantic-group UCB scores may bias slot order without eliminating any group.
- Every trial is linked-repaired, shape-checked, passed through `make_candidate()`, and deduplicated.
- Children record group, transition, changed-gene, and enumeration metadata for audit and future credit.

For micro-exhaustive search, the operator opens a bounded number of `(parent, group)` slots. If all non-current Cartesian variants fit the per-slot cap, it enumerates them completely and shuffles their order. Otherwise it tries randomized single-gene alternatives first and then randomized two-gene combinations. It does not enumerate higher-order combinations for an oversized group. Valid candidates from active slots are emitted round-robin so one large semantic group cannot consume the complete output budget.

The scheduler normally gives this operator part of the configured GOMEA budget. Under the adaptive portfolio it has the distinct source identity `gomea-neighborhood`, so its measured reward is separate from donor mixing.

## Mixing Operator

`gomea_candidates()` is the stochastic GOMEA mixer:
- Load ranked parents from DB evidence, ordered best-first by `rank_evaluations()`.
- Encode parents as categorical genomes.
- Pick a base parent and a different donor parent.
- When family-aware adaptive search is active, prefer a donor from the base candidate's family with probability `0.8` when one exists.
- The opt-in adaptive donor policy mixes quality donors from the best compatible quarter of the ranked pool, donors at maximum Hamming distance from the base, and random donors.
- Base donor-mode weights are `0.5` quality, `0.3` diverse, and `0.2` random. Queried donor-mode UCB scores multiply these priors while minimum whole-operator exploration remains intact.
- Select static FOS groups plus learned-linkage FOS groups when available, otherwise use static groups plus a fallback FOS learned from the current elite genomes.
- Shuffle groups and apply a small random prefix of them.
- For each group, copy donor genes into the base genome and accept the trial only if proposal-side rule checks pass.
- If no useful change happened, perform forced improvement by copying one group from the nearest elite.
- Convert the final genome back to a repaired candidate and dedupe by candidate hash.

A GOMEA child records parent hashes plus donor mode, family locality, donor distance, applied groups, and changed genes for audit. Ranking and cache identity remain based only on the child parameter hash. Adaptive search records two-parent children as `gomea-mixing` so operator and donor-mode credit can distinguish them from neighborhood trials.

## Rule-Gated Proposals

The GOMEA proposal gate is stricter than general candidate validity. A child must:
- Pass `explain_invalid_nt_hhs()` without global invalid reasons.
- Have at least one shape-specific `cheap_constraints()` pass in the declared shape or cluster scope.
- Stay under the proposal-side VALU VGPR lower-bound headroom used by random/GOMEA generation.

This is a throughput heuristic for generated proposals. Imported candidates, existing DB candidates, or explicit hand-authored candidates are not invalidated merely because they sit outside the proposal headroom.

## Learned Linkage Integration

When learned linkage is enabled and enough validated evidence exists, GOMEA assigns each base parent to the nearest linkage model by Hamming distance to the model leader genome. It then uses that model's FOS groups together with the static rule groups.

If learned linkage is disabled or there is insufficient evidence, GOMEA falls back to:
- the static NT HHS linkage groups.
- `fos_from_genomes()` built from current elite parent genomes.

The scheduler enables learned linkage by default for GOMEA-style proposals and exposes `--no-learned-linkage` for A/B checks. The learned-linkage mechanics are documented in `docs/search_linkage_learning.md`.

## Scheduler Use

`propose_candidates()` calls GOMEA when the selected proposal mode is one of:
- `gomea`
- `seed-random-gomea`
- `evolutionary`
- `family-qd`

For these modes, the scheduler:
- Loads DB elites when enough ranked evidence exists.
- Adds nearest-shape transfer elites for multi-shape proposals.
- Optionally adds random candidates first for seeded modes.
- Splits the GOMEA budget between neighborhood trials and stochastic mixing.
- Optionally allocates those arms separately from queried child-versus-parent evidence.
- Optionally uses family-local donors and an oversized surrogate-shortlisted pool.
- Deduplicates all generated candidates before planning missing `(shape, candidate)` pairs.

GOMEA is therefore one proposal source inside the broader search loop, not a standalone executor. Adaptive allocation is documented in `docs/search_operator_portfolio.md`, and family-aware parent selection is documented in `docs/search_family_qd.md`.
