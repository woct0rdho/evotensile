# Family-Aware EA And Screening Plan

This document tracks the search-loop upgrade for EvoTensile: use evolutionary algorithms to discover promising kernel families efficiently, then spend measurement budget inside those families with a staged screening funnel. It builds on `docs/search_algorithms.md`, `docs/gomea.md`, `docs/linkage_learning.md`, `docs/noisy_measurements.md`, `docs/tensilelite_measurement.md`, and `docs/nt_hhs_search_space.md`.

## Implementation Status

Implemented:
- Profile-specific coarse `gfx1151-nt-hhs` family descriptors in `evotensile/search/family.py`.
- On-demand DB family archive summaries using validation-passed timing rows and shape-local rank percentiles.
- Family health metadata from status counts, including negative statuses as audit/allocation evidence only.
- `family-qd` proposal mode with minimum family occupancy, one retry for negative-only cells, all positive family leaders as parents, and balanced TLDS branch sampling.
- Fair static/univariate GOMEA neighborhood coverage without performance-derived priority bundles.
- Parallel build/map/diagnostic/validation preparation followed by a hard barrier and serial benchmark-only execution.
- Family coverage in `proposal-coverage` and `schedule_metadata.json`.
- `summarize-families` CLI reporting for existing DB evidence.

Still planned:
- Evidence-driven bandit allocation after minimum family occupancy is satisfied.
- Family-leader-specific adaptive retiming before individual finalist retiming.
- Purely evidence-learned within-family linkage and conditional neighborhood allocation.
- Grid-time estimates from per-family throughput metadata.

## Motivation

The current random, local, DE, and GOMEA proposal modes operate on complete candidates and global DB elites. This is useful once good candidates are already represented in the parent pool, but it is sample-inefficient when high-performing kernels live in rare structural families.

Cold-start `8192^3` experiments show the failure mode clearly:
- High-performing regions are linked across tile, main-loop, traversal, vectorization, and store genes rather than controlled by one knob.
- Flat random/evolutionary search undersamples combinations inside broad mechanically valid families.
- GOMEA can preserve linked blocks only after useful blocks exist in validation-passed parents.
- Short screening timings are necessary for throughput, but noisy or contended measurements can eliminate a family before enough representatives are measured.

The reusable goal is not to hard-code the known NT HHS winner. The goal is to make search discover, preserve, and exploit families for any future kernel profile.

## Goals

- Preserve family diversity while searching, so rare but promising families are not discarded by early global selection.
- Allocate trials adaptively across families using measured evidence and uncertainty.
- Run within-family micro-sweeps for linked knobs once a family looks promising.
- Keep broad `DOMAINS` and TensileLite as the validity authority.
- Use validation-passed evidence only for positive ranking, linkage learning, and family promotion.
- Produce timing and coverage metadata that can estimate tuning cost for larger shape grids.

## Non-Goals

- Do not encode performance-derived families as validity rules.
- Do not replace adaptive sampling, GOMEA, learned linkage, or transfer. This plan adds a family-aware layer above them.
- Do not require schema changes in the first implementation. Family descriptors can initially be computed from candidate JSON and evaluation rows.
- Do not use best-sample GFLOP/s as the primary winner metric. Median timing remains authoritative.

## Family Descriptors

A family descriptor is a compact, profile-specific projection from a full candidate to structural traits that define a basin. The implemented `gfx1151-nt-hhs` descriptor intentionally stays coarse:

```text
floor(log2(MacroTile0 * MacroTile1))
MacroTile aspect bucket: M-major, balanced, or N-major
TransposeLDS branch
GlobalSplitU
```

Leaf scheduling, vectorization, and store fields remain within-family genes. Including them in the descriptor made almost every random candidate a singleton cell and prevented family-level evidence accumulation.

Descriptor design rules:
- Use fields that affect generated assembly shape, launch traversal, LDS/main-loop behavior, or store scheduling.
- Bucket high-cardinality or leaf-like knobs when exact values are too sparse.
- Keep descriptors profile-specific and documented next to profile/search-space docs.
- Treat descriptors as proposal metadata only. They are not cache identity and not validity.

## Family Archive

Add a derived family archive over the DB:
- Each family cell stores the best validation-passed candidates by median time per target shape or shape cluster.
- Each cell records sample count, validation failures, build rejections, timing variance, and last-seen generation.
- The archive keeps at least one elite per family even if it is not globally top-ranked.
- For multi-shape tuning, archive scores should use shape-local rank percentiles like learned linkage, not raw pooled GFLOP/s.

Initial implementation can compute the archive on demand from `candidates` and `evaluations`. Later, a materialized table can be added if needed for large campaigns.

## Proposal Modes

The initial family-aware proposal mode is `family-qd`. A later combined alias such as `family-evolutionary` can be added if A/B runs show it helps usability.

### Stratified Initialization

Instead of drawing all random candidates from the same distribution:
- Sample across descriptor cells deliberately.
- Cap the number of first-generation candidates per cell.
- Ensure mutually important branches such as TLDS0/TLDS2, VWB choices, schedule algorithms, and macro-tile families receive coverage.
- Continue applying linked repairs and shape-dependent cheap constraints.

This keeps broad domains while avoiding accidental over-concentration in one high-yield construction path.

### Quality-Diversity Archive

Maintain a MAP-Elites-style archive:
- Candidate fitness is validation-passed median time or rank percentile.
- Diversity comes from family descriptors.
- Proposal parents are sampled from both global elites and family elites.
- Family cells with sparse evidence are not discarded merely because global winners are elsewhere.

### Bandit Budget Allocation

Allocate new measurements across families using a simple upper-confidence policy:
- Each family gets a small initial budget.
- Additional budget goes to families with good median score, high uncertainty, or insufficient sample count.
- Rejected/build-failed/validation-failed ratios lower a family's priority but do not create hard invalidity.
- The policy writes per-family allocation decisions to metadata for audit.

A first implementation can use a heuristic score:

```text
family_priority = best_rank_percentile
                - exploration_bonus(samples, families_seen)
                + rejection_penalty
                + validation_fail_penalty
```

Lower priority is better when using rank percentiles.

### Within-Family Exploitation

When a family cell is promoted:
- Run GOMEA/local mutation using only parents from that family or adjacent family cells.
- Run compact micro-sweeps over linked leaf knobs rather than independent mutation.
- Feed within-family winners back into learned linkage as basin-specific evidence.

Within-family linkage should come from either source-backed mechanical coupling or validation-passed learned evidence. Performance-derived knob bundles must not be encoded as static priority groups merely because they match a previously observed winner.

## Screening Funnel

Use a staged funnel so broad exploration remains fast while final claims remain reliable.

### Stage 0: Build/Validity Attribution

- Generate candidates and run normal TensileLite build/final-YAML mapping.
- Preserve accepted candidates from mixed builds.
- Attribute build failures through structured diagnostics when final YAML cannot map them.
- Record `rejected`, `build_failed`, and `validation_fail` as evidence for allocation, but do not treat them as hard rules unless source-backed predicates are found.

### Stage 1: First Validation And Cheap Timing

- Validate each `(shape, candidate)` once before timed benchmarking.
- Use the default GPU oracle validation backend for throughput.
- Use a small number of timing samples for first-pass family ranking.
- Rank family cells by robust median log-time summaries, not by best sample.

### Stage 2: Family Leader Retiming

- Retain the best `K` family leaders plus uncertain contenders.
- Increase sample count with adaptive sampling only for family leaders and plausible near-leaders.
- Use shape-local rank percentiles when comparing families across multiple shapes.
- Drop families only when confidence intervals show they are outside the practical gap threshold.

### Stage 3: Hot-Loop Confirmation

- Retime final candidates with the production hot-loop protocol before declaring winners or exporting GridBased logic.
- Keep hot-loop confirmation separate from broad screening so the expensive protocol is used only on a small finalist set.
- Store confirmation runs in the same DB under a benchmark-protocol hash that captures timing compatibility.

### Stage 4: Grid Transfer

- Promote family descriptors and linkage groups that win on one shape into transfer priors for nearby shapes.
- Tune each new shape with a mixture of transferred family leaders, family-neighborhood proposals, and stratified exploratory cells.
- Track per-shape family winner changes to estimate whether the grid needs broad exploration or mostly local retiming.

## Metadata And Reports

Each family-aware run should write:
- Descriptor definition and version.
- Number of families sampled and number of live archive cells.
- Candidates, accepted candidates, validation failures, and rejections per family.
- Wall time split by build, validation, timing, diagnostics, and DB ingestion when available.
- GPU utilization snapshots when monitor data is present.
- Family allocation decisions and reasons.
- Top candidates per family and final global ranking.
- Estimated per-shape and per-grid tuning time from measured batch throughput.

The current report command is:

```bash
python -m evotensile.cli summarize-families --db <db> --shapes 8192,8192,1,8192
```

## Implementation Plan

- Add descriptor helpers. Done.
  - Implement a profile-aware `family_descriptor(candidate, shape=None)` helper.
  - Start with `gfx1151-nt-hhs` descriptors.
  - Add tests that documented candidates map to stable descriptor tuples.

- Add archive summarization. Done.
  - Compute family archive entries from existing DB evidence.
  - Use validation-passed rows for positive scores.
  - Include rejected and validation-failed counts as family health metadata.

- Add stratified proposal generation. Done.
  - Add a sampler that targets descriptor cells and avoids first-generation over-concentration.
  - Keep existing random generation available as an exploration fallback.
  - Report proposal coverage by descriptor cell.

- Add `family-qd` proposal mode. Done.
  - Combine family archive parents, stratified random candidates, and within-family GOMEA/local variants.
  - Dedupe by candidate hash before planning batches.
  - Write descriptor coverage and allocation metadata to `schedule_metadata.json`.

- Add family leader retiming.
  - Reuse adaptive sampling math, but group decisions by family leaders before individual finalists.
  - Keep final DB ranking unchanged.

- Add micro-sweep expansion.
  - For promoted families, sweep configured linked groups around the current family leader.
  - Reuse `gomea_neighborhood_candidates()` where possible, but allow profile-specific priority groups.

- Add reporting and grid-time estimates.
  - Summarize per-family throughput, acceptance rate, and score distribution.
  - Estimate 100-shape tuning wall time from observed batch duration, candidate count, and validation/timing sample count.

## Evaluation Plan

Run A/B comparisons on fixed seeds and DB snapshots:
- Current `evolutionary` versus `family-qd` on `8192^3`.
- Current `seed-random-gomea` versus `family-evolutionary` on a small multi-shape subset.
- Learned linkage on/off inside family-aware proposals.
- Fixed first-pass samples versus family-leader adaptive retiming.

Success metrics:
- Time to first candidate within `10%`, `5%`, and `2%` of the final retimed winner.
- Number of candidate builds before discovering the winning family.
- Number of family cells sampled before convergence.
- Retiming stability: how often first-pass family leaders remain leaders after hot-loop confirmation.
- Total wall time per tuned shape and projected wall time for the 100-shape grid.

## Open Questions

- Which descriptors should be exact versus bucketed for each future profile?
- Should family archive selection be shape-local, shape-cluster-local, or global with transfer weighting?
- How much initial budget should every family receive before bandit allocation starts?
- Should validation-failed families be penalized by family descriptor or only by exact candidate?
- How should hot-loop confirmation protocols be represented when they differ from cheap screening protocols?
- When a family wins on one shape but fails on nearby shapes, should the descriptor split be refined automatically?
