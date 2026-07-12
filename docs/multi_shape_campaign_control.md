# Multi-Shape Campaign Control

This document describes the shared campaign state, metric, and soft-admission primitives in `evotensile/campaign/controller.py` and their real-campaign orchestration in `evotensile/campaign/runner.py`. Historical implementation order and retained-corpus results are recorded in `docs/experiment_100_shape.md`. The current singleton proposal policy is documented in `docs/blind_campaign_control.md`.

## Boundary

The controller module is independent of proposal generation, SQLite, TensileLite preparation, structured validation, and timing. It owns the state and decisions that must have identical semantics in replay and real campaigns:
- the ordered shape set.
- optional deterministic mechanical clustering configuration, descriptors, medoids, memberships, and distances.
- exact queried, known, unknown, and disclosed pair identity.
- per-shape incumbents.
- candidate artifact shape coverage.
- phase cost ledgers, reserves, round and phase identity, and audit traces.
- elapsed time, the soft admission deadline, and budget overrun.
- checkpoint serialization and restoration.
- grid regret and coverage summaries.

Proposal providers continue to produce candidates. Replay, real, and hybrid evaluators in `evotensile/campaign/evaluator.py` produce durable exact evidence through one result contract. The controller records those results but does not infer or execute candidate-shape pairs. The promotion policy in `evotensile/campaign/promotion.py` materializes exact probe and main requests, then appends source/destination, cluster-transition, preparation-reuse, outcome, and realized-gain events after durable results. See `docs/pair_evaluators.md` and `docs/shape_promotion_racing.md`.

## Joint Campaign Semantics

One-shape search is the one-member specialization of the same controller, evaluator, proposal, evidence, validation, timing, and ranking contracts. Multi-shape search is not implemented as 100 unrelated singleton runs. A joint campaign may:
- reuse one candidate artifact across an explicit shape scope while measuring only requested exact pairs.
- seed exact destination measurements from mechanically nearby winners and near-winners.
- screen candidate families on deterministic representatives before measured promotion.
- retain specialists and broad generalists in one proposal pool without pooling absolute latency across shapes.
- allocate finite measurement work by shape-local incumbent improvement, unresolved coverage, uncertainty, repair value, workload contribution, and predicted cost.
- detect weak residual shapes after broad work and reserve explicit exact repair measurements.
- select a final confirmed solution bank with a reported tolerance and deployment loss.

Every transfer, promotion, repair, stabilization, and confirmation remains a measurement decision. Mechanical similarity and posterior predictions may prioritize an exact pair, but they never resolve it or make it production-eligible without durable exact evidence.

## Query-Causal State

`CampaignControllerState` requires a non-empty ordered tuple of unique shape IDs. Pair state is split into:

```text
queried_pairs
known_pairs
unknown_pairs
disclosed_pairs
```

A pair must be queried before it can be known or unknown. Only a known pair can be disclosed. Disclosure may carry a finite positive performance value. The best disclosed value becomes that shape's incumbent. Repeated query and disclosure calls are idempotent when their identity and known/unknown state agree.

Unknown pairs cannot later become known inside the same evidence source. A hybrid evaluator must create and label a real evidence result before recording the pair in a state where it is known. This prevents a missing replay row from silently changing meaning.

Artifact preparation is separate from pair evidence. `record_prepared()` records the exact shape scope available for one candidate artifact and returns only newly covered shapes. Artifact coverage alone does not disclose validation or timing evidence.

## Soft Admission

`SoftAdmissionBudget` owns:
- the declared positive time budget.
- the monotonic start of the current process session.
- elapsed time carried from an earlier checkpoint.
- the resulting admission deadline.

`decide()` checks whether predicted work plus an explicit reserve fits before that deadline. It returns `admitted`, `insufficient_predicted_budget`, or `soft_deadline`. It does not create a subprocess timeout or cancel work.

Once a caller admits work, that work keeps its configured build or runner timeout and may complete after the nominal budget. `elapsed_s()` and `overrun_s()` record the complete result. Callers must decline later work after the deadline.

The real campaign runner and exact replay both use this state. The reusable staged controller in `evotensile/campaign/round_controller.py` applies cumulative phase deadlines and the total soft deadline, persists exact pending waves before execution, and recomputes policy only after durable results. A resumed admitted wave drains through the evaluator before replanning. Replay applies the same rule with an explicit simulated-clock callback. A round is one bounded campaign increment and may intentionally leave shapes unresolved. Later multi-round policy may assign different budgets and phase/acquisition hyperparameters to different round roles while carrying durable controller evidence forward. See `docs/staged_round_controller.md`.

## Duration Prediction

`estimate_admission_duration_s()` is the shared robust estimator for admission units. It consumes recent `(duration_s, units)` observations, takes the median per-unit duration plus its median absolute deviation, scales by expected units, applies a multiplicative margin and fixed overhead, and enforces a minimum.

The one-shape campaign converts recent round records into `(duration_s, requested_pairs)` observations and calls this owner. The former duplicate round-specific estimator has been removed from `evotensile/search/campaign_control.py`.

## Grid Metrics

`grid_metrics()` compares disclosed incumbents with exact positive per-shape oracle best values. For each resolved shape it reports nonnegative log regret:

```text
log_regret_s = max(0, log(oracle_best_s / incumbent_s))
```

The result contains:
- the per-shape regret map, with `null` for unresolved shapes.
- resolved and unresolved counts.
- unweighted mean and workload-weighted mean regret over resolved shapes.
- median, p90, p95, and worst resolved regret.

Uniform weight one is the default. Explicit workload mode uses call count multiplied by baseline latency, persists the resolved weights in checkpoints, and propagates them through acquisition and proposal ranking without replacing unweighted tail reporting. See `docs/workload_weighting.md`.

Unresolved shapes remain explicit rather than receiving an invented regret. `summary()` combines these metrics with exact pair counts, candidate query coverage, prepared artifact coverage, phase costs, elapsed time, reserves, and overrun.

## Checkpoints

`to_checkpoint()` serializes only JSON-compatible durable state. Absolute monotonic deadlines are not persisted. The checkpoint stores elapsed time. `from_checkpoint()` combines it with the new process session start to reconstruct the remaining soft admission window.

`CampaignStore` persists this payload as the sole phase, round, elapsed, reserve, exact-pair, incumbent, artifact-coverage, clustering, staged-round, and phase-cost authority. Exact pending candidate hashes and round seed remain beside it because candidate dictionaries live in the round proposal artifact. Current development layouts are consumed directly without compatibility coercion.

Checkpoint restoration rejects:
- missing or duplicate shapes.
- pairs outside the registered shape set.
- pairs that are both known and unknown.
- disclosed pairs that are not known and queried.
- invalid incumbents, elapsed values, budgets, or phase costs.

There is no compatibility coercion or schema marker for earlier development layouts.

## Singleton Law

With one shape:
- there is one possible incumbent entry.
- every exact pair belongs to that shape.
- grid metrics reduce to that shape's log regret.
- the sole workload weight has no relative effect.
- soft admission and overrun are unchanged.

Mechanical clustering degenerates to one `cluster_000` whose medoid is the only shape, independent of fixed-count or threshold policy. Promotion returns no transfer requests, and repair is a no-op unless exact singleton evidence defines a deficit. Bundle acquisition reduces to one exact pair per candidate. P12 selects information weight `0.05` for this degeneration after at least 24 positive observations while preserving mechanical cold start. See `docs/shape_clustering.md`, `docs/shape_promotion_racing.md`, and `docs/campaign_policy_tuning.md`.

`simulate_independent_shape_baseline()` uses this law to run one isolated singleton replay per shape. Each instance receives the same declared policy and cost parameters but owns a separate simulated DB, preparation ledger, evidence snapshot, controller state, and budget. Its aggregate summary adds total cost and pair counts without sharing information between shapes.

## Deployment Boundary

After staged screening and repair, posterior-close finalists receive exact stabilization requests, the contextual model is refitted, and final confirmation groups follow the same soft-admission rule. Production confirmation forces fresh validation and timing rather than reusing cache hits. The resulting explicit per-shape assignment may preserve exact winners or apply an opt-in tolerance-based solution-bank cover. Uniform, workload-weighted, and worst-shape loss remain visible. See `docs/deployment_selection.md`.
