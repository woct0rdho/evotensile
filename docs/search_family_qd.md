# Family-QD Search Design

This document describes EvoTensile's family-aware quality-diversity search. It covers family descriptors, stratified initialization, the DB-derived family archive, and the way family evidence feeds proposal generation. General proposal flow is documented in `docs/search_algorithms.md`. GOMEA and learned linkage are documented in `docs/search_gomea.md` and `docs/search_linkage_learning.md`.

## Purpose

High-performing TensileLite kernels are usually defined by linked structural choices rather than one independent knob. A global elite list can converge before it retains enough distinct building blocks from rare but promising structural basins.

Family-QD adds a coarse quality-diversity layer that:
- gives broad structural families repeated initial evidence.
- preserves strong candidates from many families, not only global leaders.
- retains multiple diverse candidates inside a family.
- supplies family-aware parents to semantic mutation and GOMEA.
- keeps family policy separate from validity, cache identity, and final winner selection.

Family-QD does not narrow `DOMAINS`, make runtime failures into validity rules, or encode a known winner's parameter bundle.

## Family Descriptor

A family descriptor is a profile-specific projection from a complete candidate to a small structural cell. The current `gfx1151-nt-hhs` descriptor version is `nt_hhs_v2` and contains:

```text
floor(log2(MacroTile0 * MacroTile1))
MacroTile aspect bucket: M-major, balanced, or N-major
TransposeLDS branch
GlobalSplitU
```

This descriptor is intentionally coarse. Scheduling, vectorization, staging, and store fields remain within-family genes. An earlier detailed descriptor made nearly every random candidate a singleton and prevented evidence accumulation.

Descriptor design rules:
- use traits that define a broad assembly or execution basin.
- bucket high-cardinality traits when exact values make cells too sparse.
- leave leaf tuning choices available for within-family search.
- keep descriptors profile-specific and versioned.
- treat the descriptor as proposal metadata only.

`family_descriptor()` and `family_descriptor_counts()` implement descriptor calculation and coverage reporting.

## Stratified Initialization

`family_stratified_random_candidates()` targets coarse family cells rather than accepting whichever cells ordinary random sampling happens to produce.

The sampler:
- enumerates compatible family cells for the target shapes.
- loads prior attempt counts from the active DB and protocol.
- prioritizes cells with insufficient occupancy.
- retries a negative-only cell once before treating its evidence as representative.
- constructs candidates through the normal broad random generator and linked repair path.
- applies shape-dependent cheap constraints before returning proposals.
- falls back to ordinary shape-aware random generation when necessary.

The NT HHS random generator keeps compatible TLDS0 and TLDS2 construction branches balanced. This is a proposal policy, not a validity restriction.

Repeated occupancy matters because one build or validation failure is evidence about one candidate, not proof that an entire structural family is bad.

## Family Archive

`load_family_archive()` derives archive entries on demand from SQLite. It does not require a separate materialized archive table.

Positive archive scoring uses validation-passed `status='ok'` timing evidence under the requested problem and benchmark protocol. For multi-shape evidence, candidates are compared using shape-local rank percentiles rather than pooled absolute time or GFLOP/s.

Each archive entry records:
- descriptor and descriptor version.
- candidate and aggregate score.
- timing sample count and represented shape count.
- observed candidate count for the cell.
- family-level build, rejection, and validation status counts.
- rank inside the family.
- Hamming novelty distance from already selected family elites.

Negative statuses are health and audit evidence. They do not teach positive performance linkage and do not create new hard invalidity rules.

## Diverse Multi-Elite Selection

A family cell keeps up to four elites by default.

Selection proceeds as follows:
- Select the best candidate by aggregate score, then sample count and shape coverage.
- Define a quality window relative to that leader. The default score slack is `0.25`.
- Within the quality window, choose the candidate with maximum minimum Hamming distance from already selected elites.
- Break ties by quality, evidence count, shape count, and candidate hash.
- If the quality window does not contain enough candidates, continue from the remaining family candidates rather than returning duplicate or empty slots.

This preserves complementary within-family genomes while ensuring the first archive entry remains the family leader. APIs that request one elite per family retain the original single-leader behavior.

## Proposal Integration

The `family-qd` proposal mode combines:
- stratified random exploration.
- globally ranked DB elites.
- diverse family archive elites.
- semantic-group mutation.
- categorical differential evolution.
- GOMEA neighborhood trials.
- two-parent GOMEA mixing.

Family archive elites are inserted before generated proposals and are preserved when an oversized surrogate pool is shortlisted. Existing archive candidates are normally cache hits, so retaining them as parents does not imply repeating their measurements.

When adaptive operators are enabled, two-parent GOMEA chooses a donor from the base candidate's family with probability `0.8` when a compatible donor exists. This keeps recombination local enough to preserve a basin while still allowing cross-family exploration.

Semantic mutation and the adaptive operator portfolio are documented in `docs/search_operator_portfolio.md`. Surrogate shortlisting is documented in `docs/search_surrogate.md`.

## Relationship To Learned Linkage

Family descriptors and learned linkage operate at different scales:
- the family descriptor identifies a coarse structural basin.
- the archive preserves several candidate genomes in that basin.
- learned linkage derives evidence-based FOS groups from validated candidates.
- GOMEA applies static semantic groups and learned groups during neighborhood or donor mixing.

Family membership never overrides the linkage model. A base candidate is assigned to the nearest learned linkage model by genome distance as described in `docs/search_linkage_learning.md`.

## Reporting

Family information is available through proposal coverage metadata, schedule metadata, and:

```bash
python -m evotensile.cli summarize-families \
  --db <campaign.sqlite> \
  --shapes 8192,8192,1,8192
```

Reports include descriptor counts, archive leaders, family ranks, novelty, evidence counts, and negative-status summaries.

## Current Limitations

The implementation does not yet:
- allocate measurement budget directly between family cells with a family-level bandit.
- split or merge descriptor cells automatically from accumulated evidence.
- retime family leaders as a separate fidelity tier before global finalists.
- materialize archive history or per-generation family state in dedicated DB tables.
- migrate elites between independent search islands.

Those are possible extensions. The current design keeps family state derived, auditable, and independent of candidate validity.
