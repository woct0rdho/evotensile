# Search Cost Modeling

This document describes measured candidate-cost attribution and the proposal-side preparation-weight heuristic in `evotensile/search/cost_model.py`. The campaign-level fitted preparation/validation/timing predictor and bundle marginal-cost contract are documented in `docs/shared_bundle_acquisition.md`. Operator allocation is documented in `docs/search_operator_portfolio.md`. Shared admission estimation and soft-budget accounting are documented in `docs/multi_shape_campaign_control.md`. The current one-shape state machine is documented in `docs/blind_campaign_control.md`.

## Purpose And Boundary

EvoTensile uses cost in two different ways:
- retrospective measured cost can scale operator, semantic-group, and donor-mode credit.
- a cheap pre-measurement weight can order parallel preparation batches longest-predicted-work first.

Neither mechanism changes candidate validity, correctness evidence, benchmark ranking, or the serialized timing requirement. Both are optional scheduling/proposal policies.

## Measured Candidate Cost

`CandidateMeasuredCost` separates:

```text
proposal_s
prepare_s
validation_s
probe_s
screening_s
```

`total_s` is their sum.

### Proposal Cost

The blind campaign measures one proposal call's wall time and divides it by the scheduler's explicit novel `generated` set before surrogate shortlisting. Each proposal call persists one `proposal_events` row with its duration and child `proposal_candidates` rows identifying generated versus preserved candidates. Proposal cost is derived as event duration divided across distinct generated candidates.

Preserved parents and previously registered duplicate hashes receive no new proposal cost merely because they appear in another selected set. Island, restart, lineage, operator metadata, and selection state remain occurrence-owned and never alter parameter-only candidate identity.

### Run Cost Attribution

Execution boundaries pass exact candidate hashes, phase, and duration to `insert_run()`. SQLite records one shared-cost row per distinct candidate in `run_candidate_costs` for prepare, validation, probe, and screening work. `load_candidate_measured_costs()` aggregates this index and combines it with proposal-event cost. It does not reopen manifests, pair files, or run directories.

One invocation's duration is divided equally between the distinct candidate hashes attributed to that invocation at insertion time.

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
+ 0.40 * VALU_VGPR_fraction
+ 0.25 * LDS_fraction
+ 0.10 * log2(max(1, WMMA_wave_tile_area))
+ 0.05 * log2(max(1, WMMA_wave_group_size))
```

A batch weight sums candidate weights averaged over its explicit artifact-shape scope. Requested evaluation pairs remain separate and exact.

With `--cost-aware-scheduling`, the scheduler submits higher-weight preparation batches first. This is a longest-predicted-work-first heuristic intended to reduce the tail before each hard preparation/timing barrier. It does not change artifact scope, requested pair contents, timing order, or permit timing overlap.

Preparation and timing order are independent. After preparation drains, stable planned order is the default serialized timing order. The shared bundle allocator now emits longest-predicted-preparation order separately from utility-per-cost exact timing priority. Its phase predictors fit typed measured history when available and retain conservative analytical/fixed fallbacks for cold candidates.

## Campaign Round Admission

The one-shape campaign does not use the analytical preparation weight as its wall-time estimate. It converts recent rounds to measured duration and exact requested-pair observations, then calls the shared robust estimator in `evotensile/campaign/controller.py` before admitting another round. The campaign budget is a soft admission deadline: an admitted schedule retains normal build and runner timeouts and may finish afterward. The confirmation reserve is recalculated from measured finalist launch cost with a configured minimum floor. The generic semantics are described in `docs/multi_shape_campaign_control.md`. The one-shape use is described in `docs/blind_campaign_control.md`.

## CLI Controls

```text
--cost-aware-operator-credit
--cost-aware-scheduling
```

Both default off in the general CLI. The blind one-shape policy enables both explicitly.

## Persistence And Identity

Candidate rows store only compact canonical parameters. Proposal context and cost live in proposal event/occurrence rows and affect future proposal policy without fragmenting cache identity.

Run durations and commands remain the authoritative low-level provenance. Derived shared candidate costs are indexed when each run is inserted and loaded into the immutable proposal evidence snapshot.

## Limitations

- Equal division of shared run duration cannot identify which candidate caused a long batch.
- Proposal-side preparation ordering remains a hand-sized generic heuristic. Campaign bundle costing adds a fitted model only when typed measured rows are available.
- Validation and benchmark startup costs can dominate small candidate sets.
- Cost is pooled across selected shapes and phases rather than modeled conditionally by family or hardware state.
- There is no explicit workload-priority model combining call count, baseline latency, headroom, uncertainty, and evaluation cost.
- Cost-aware allocation has observational support but no complete real component ablation.
