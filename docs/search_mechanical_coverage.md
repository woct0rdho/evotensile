# Mechanical Coverage And Cold-Start Selection

This document describes the generic candidate-shape mechanics in `evotensile/search/mechanics.py` and the optional covering cold-start selector built from them. Family descriptors and archives are documented in `docs/search_family_qd.md`. Surrogate training and acquisition are documented in `docs/search_surrogate.md`.

## Boundary

Mechanical features and coverage scores are proposal signals. They do not:
- remove values from `DOMAINS`.
- add candidate invalidity rules.
- turn runtime failures into reusable rejection predicates.
- change candidate hashes, correctness identity, or final ranking.

Every selected candidate still follows normal linked repair, source-backed rule checks, TensileLite build/codegen, validation, probe, and timing.

## Candidate-Shape Mechanics

`candidate_shape_mechanics()` projects one complete candidate and one shape into generic execution features. The current gfx1151 default uses an effective CU count of `20`. Callers may override it for another target.

Tile and dispatch features include:
- M and N tile fill.
- output tile count.
- GSU-expanded workgroup count.
- tiles per effective CU.
- integer CU rounds and final-round CU granularity.
- workgroup threads and waves.
- macro-tile area, one-instruction output area, and their ratio.
- WMMA wave-tile area and wave-group size.

Reduction and resource features include:
- reduction iterations after `DepthU * GlobalSplitU` partitioning.
- K fill after rounding to complete reduction iterations.
- LDS bytes and fraction of the profile limit.
- proposal-side VALU VGPR lower bound and fraction of the profile limit.
- GSU workspace bytes and fraction of the profile limit.
- arithmetic intensity from GEMM FLOPs and approximate input/output traffic.

These are cheap analytical features. They are not a complete occupancy, instruction-count, memory-transaction, or register-allocation model.

## Dispatch-Efficiency Prior

Very small macro tiles can have excellent divisibility while launching an excessive number of workgroups. The soft prior therefore includes:

```text
workgroup_tile_multiple = macro_tile_area / instruction_tile_area
dispatch_efficiency = 1 - 1 / sqrt(max(1, workgroup_tile_multiple))
```

A workgroup that covers only one instruction output tile receives zero dispatch-efficiency contribution. Larger workgroups approach one smoothly. This is a ranking term, not a hard minimum macro-tile rule.

`mechanical_prior_score()` multiplies tile fill, CU granularity, minimum parallel depth, K fill, and dispatch efficiency, then adds small VGPR and LDS headroom terms. The score is used only inside cold-start selection.

## Coverage Tokens

`mechanical_coverage_tokens()` converts mechanics and parameter marginals into discrete coverage tokens:
- coarse family descriptor.
- WMMA wave-tile and wave-group geometry.
- macro-area and macro-aspect log buckets.
- CU-round and CU-granularity buckets.
- wave count and K-fill bucket.
- LDS and VGPR fraction buckets.
- exact marginal values for every parameter except `MatrixInstruction`.

Exact `MatrixInstruction` identities and exact macro-tile identities are intentionally excluded. Raw identities made almost every oversized-pool candidate look uniquely valuable and could reward pathological single-instruction workgroups. The selector instead covers decomposed building blocks that can transfer between complete instructions.

Token weights are inverse-square-root frequency weights. WMMA wave and macro tokens receive weight `2.0`, family tokens receive `1.5`, and other tokens receive `1.0` before frequency normalization. Rare structural tokens therefore matter without becoming permanent quotas.

## Covering Selector

`select_covering_cold_pool()` deduplicates an oversized valid pool and fills a fixed measurement budget through three lanes:
- `80%` quality-weighted marginal coverage by default.
- `10%` highest mechanical prior by default.
- the remaining budget from seeded random order.

For one coverage candidate, the acquisition key is:

```text
marginal_uncovered_token_weight * (0.35 + 0.65 * normalized_prior)
```

The quality multiplier prevents mechanically novel but obviously under-tiled candidates from dominating solely through rare tokens. The selector remains deterministic for a fixed pool and seed.

`precovered_tokens` lets coordinated cold populations avoid duplicating each other's mechanical coverage. The blind one-shape campaign uses this when selecting its second cold island.

## Integration

The feature projection is reused by:
- `candidate_shape_features()` for ExtraTrees training.
- the one-shape `--covering-cold-start` fallback before surrogate evidence exists.
- campaign population diagnostics.
- the proposal-side preparation-cost heuristic.

The default evidence-free surrogate fallback remains family-diverse round-robin selection. Mechanical covering is opt-in and currently applies only when exactly one target shape is supplied.

## Reporting And Tests

Campaign diagnostics report:
- distinct family cells.
- distinct complete MatrixInstruction values for audit.
- union size of mechanical coverage tokens.
- mean and minimum categorical-genome Hamming distance.

Tests cover feature ranges, the dispatch-efficiency ordering, fixed shortlist size, increased token coverage, and integration with the scheduler's oversized cold pool.

## Limitations

- Effective CU count is currently a gfx1151 default rather than a profile field.
- The prior does not model full VGPR allocation, occupancy, memory coalescing, instruction scheduling, or code-object size.
- Token priorities and lane fractions are fixed policy constants and have not received a full multi-seed ablation.
- Better mechanical coverage does not imply better final performance. Experiment outcomes belong in `docs/blind_one_shape_experiment.md`.
