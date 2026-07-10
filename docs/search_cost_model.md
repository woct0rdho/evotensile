# Search Cost Modeling

This document describes the measured candidate-cost attribution and proposal-side preparation-weight heuristic in `evotensile/search/cost_model.py`. Operator allocation is documented in `docs/search_operator_portfolio.md`. Campaign round admission is documented in `docs/blind_campaign_control.md`.

## Purpose And Boundary

EvoTensile uses cost in two different ways:
- retrospective measured cost can scale operator, semantic-group, and donor-mode credit.
- a cheap pre-measurement weight can order parallel preparation batches longest-predicted-work first.

Neither mechanism changes candidate validity, correctness evidence, benchmark ranking, or the serialized timing requirement. Both are optional scheduling/proposal policies.

## Measured Candidate Cost

`CandidateEvaluationCost` separates:

```text
proposal_s
prepare_s
validation_s
probe_s
screening_s
```

`total_s` is their sum.

### Proposal Cost

The blind campaign measures one proposal call's wall time and divides it by the number of newly generated candidates. `tag_proposals()` persists this estimate as `proposal_metadata.proposal_cost_s` without changing the parameter-only candidate hash.

Preserved parents do not receive new proposal cost merely because they appear in another proposal list. Because candidate registration is first-write-wins by parameter hash, later duplicate proposals remain in round artifacts but do not replace the DB-level proposal-cost attribution.

### Run Cost Attribution

`load_candidate_evaluation_costs()` reconstructs costs from the `runs` table and retained artifacts:
- build/prepare invocations use candidate hashes from `config.manifest.csv`.
- validation, probe, and screening invocations use candidate hashes from the runner pair file in the recorded command.
- zero-warmup benchmark pair files are classified as probe work.
- validation mode is classified separately from main screening.

One invocation's duration is divided equally between the distinct candidate hashes attributed to that invocation.

This is an audit-quality shared-cost estimate, not an exact per-candidate profiler. Batch startup, library loading, one pathological candidate, and shared codegen can make equal attribution inaccurate.

## Cost-Aware Credit

Normal UCB credit uses queried child-versus-parent success and an exploration term. With `--cost-aware-operator-credit`, `credit_ucb_scores()` additionally computes a smoothed average cost for each arm:

```text
average_cost = (cumulative_cost_s + 1) / (trials + 1)
reference_cost = middle sorted arm average cost
cost_multiplier = clamp(sqrt(reference_cost / average_cost), 0.5, 2.0)
```

The UCB score is multiplied by this factor before proportional budget allocation.

The same mechanism can scale:
- whole-operator arms.
- semantic-group arms.
- GOMEA donor-mode arms.

Minimum exploration is applied before proportional allocation, so a high estimated cost cannot permanently remove an arm.

## Predicted Preparation Weight

`predicted_candidate_prepare_weight()` is available before compilation. For one candidate and shape it uses:

```text
1
+ 0.40 × VALU_VGPR_fraction
+ 0.25 × LDS_fraction
+ 0.10 × log2(max(1, WMMA_wave_tile_area))
+ 0.05 × log2(max(1, WMMA_wave_group_size))
```

A batch weight sums candidate weights averaged over its shapes.

With `--cost-aware-scheduling`, the scheduler submits higher-weight preparation batches first. This is a longest-predicted-work-first heuristic intended to reduce the tail before the hard preparation/timing barrier. It does not change the batch contents or allow timing overlap.

## Campaign Round Admission

The one-shape campaign does not use the analytical preparation weight as its wall-time estimate. It computes recent measured seconds per missing pair and applies a robust margin before admitting another round. That policy is described in `docs/blind_campaign_control.md`.

## CLI Controls

```text
--cost-aware-operator-credit
--cost-aware-scheduling
```

Both default off in the general CLI. The blind one-shape policy enables both explicitly.

## Persistence And Identity

Proposal metadata is stored inside candidate JSON and restored by `EvoTensileDB.get_candidates()`. Candidate hashes remain a function only of canonical parameter dictionaries. Cost metadata therefore affects future proposal policy without fragmenting cache identity.

Run durations and commands remain the authoritative low-level provenance. Derived candidate costs are reconstructed on demand rather than materialized in a new database table.

## Limitations

- Equal division of shared run duration cannot identify which candidate caused a long batch.
- The preparation predictor is a hand-sized generic heuristic, not a fitted duration model.
- Validation and benchmark startup costs can dominate small candidate sets.
- Cost is pooled across selected shapes and phases rather than modeled conditionally by family or hardware state.
- Cost-aware allocation has observational support but no complete real component ablation.
