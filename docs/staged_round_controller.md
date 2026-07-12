# Staged Round Controller

This document defines the reusable soft-deadline round controller in `evotensile/campaign/round_controller.py`. Shared state and budget semantics are documented in `docs/multi_shape_campaign_control.md`. Bundle planning is documented in `docs/shared_bundle_acquisition.md`.

## Objective

One round is one bounded information-gathering and racing increment. It is not required to resolve every shape or find a very good configuration for every shape. Later multi-round campaign policy may vary round budgets, phase fractions, and acquisition hyperparameters while carrying durable evidence and artifact coverage forward.

The round controller owns execution and persistence semantics. It does not generate candidates, fit a model, or define one fixed acquisition policy.

## Phases

Every round exposes the ordered phases:
- `broad`: representative and unresolved-shape discovery.
- `promotion`: posterior or observed-evidence candidate promotion.
- `repair`: reserved weak-shape work using the integrated capped-deficit acquisition.
- `stabilization`: confidence-aware top-ups without requiring new preparation.
- `confirmation`: final high-fidelity evidence admitted before the same soft deadline.

`StagedRoundConfiguration` stores one configurable fraction for each phase. Fractions are nonnegative and sum to one. Defaults are initial experiment values, not invariants.

The controller records explicit repair, stabilization, and confirmation reserves from those fractions.

## Planner Boundary

A `StagedRoundPlanner` provides:

```text
plan_wave(phase, controller) -> AcquisitionPlan | None
observe(phase, durable_result)
```

The controller asks again after every durable wave. This lets a caller refit the contextual model and recompute bundle acquisition from newly disclosed evidence. A planner may use shared-cost bundle acquisition, representative requests, promotion racing, repair candidates, or stabilization finalists without changing round admission semantics.

An empty plan advances to the next phase. The round is allowed to complete with unresolved shapes.

## Soft Admission

Before persisting a new wave, the controller checks:
- total round time remaining.
- the cumulative deadline for the current phase.
- the plan's conservative predicted cost.
- the final no-new-preparation guard.

If a plan does not fit its phase deadline, that phase closes and later reserved phases remain eligible. If total time is exhausted, the round stops. If the final guard is active, a plan requiring preparation is rejected while an already-prepared stabilization or confirmation plan may still run.

No lower-layer timeout is shortened. Once admitted, preparation, validation, probe, main timing, and top-ups retain their operational timeouts and drain completely. Actual completion may cross a phase or total deadline. A total overrun is recorded and no later plan is requested.

Real mode uses monotonic wall time. Replay mode may pass a result-time callback that advances a simulated clock by the evaluator's phase costs. The admission logic is otherwise identical.

## Pending Wave Persistence

Before evaluator execution, `PendingRoundWave` records:
- exact shape and candidate identity for every request.
- evidence stage, target samples, and timing priority.
- explicit candidate artifact-shape scopes.
- predicted cost.
- complete acquisition-plan report, including bundle scores.
- phase and wave identity.

The controller embeds the current `StagedRoundState` in its checkpoint payload. The round state also records:
- round ID.
- round configuration hash.
- model/config identity supplied by the caller.
- phase and wave indices.
- every admission decision.
- completed exact waves and known/unknown counts.
- stop reason.

These are internal current-layout payloads. No schema version or compatibility layer is defined.

## Resume

If execution is interrupted after the pending checkpoint, resume reconstructs the exact requests and artifact scopes from registered candidate and shape catalogs. It executes that admitted pending wave before asking the planner for anything new. Pending candidates are not regenerated and bundle scores are not recomputed before the durable result.

After the resumed result is applied, `planner.observe()` runs and normal acquisition recomputation continues.

Round identity checks reject a different round ID, configuration hash, or model identity against an active checkpoint.

## Preparation And Timing Order

The pending plan retains both:
- longest-predicted-preparation-first ordering from the shared acquisition cost model.
- utility-per-cost priorities on exact timing requests.

In explicit workload mode, acquisition has already multiplied pair utility by the persisted workload weights, so these exact priorities carry workload contribution into staged timing admission. Uniform mode preserves equal shape weights. See `docs/workload_weighting.md`.

The scheduler remains the execution owner. Artifact scope does not create evaluation pairs.

## Singleton Law

One shape uses the same phases, admission decisions, pending request layout, overrun handling, and resume behavior. Broad or promotion planning may still propose singleton candidate requests. Clustering and cross-shape transfer remain no-ops.

The existing one-shape campaign runner is not silently replaced by the staged controller. A caller adopts staged planning explicitly. Singleton bundle acquisition can improve proposal selection without changing round execution semantics.

## Tests

Unit and integration tests verify:
- an admitted wave drains beyond the soft deadline and blocks later planner calls.
- a phase-deadline rejection preserves a later phase.
- the final guard rejects new preparation but admits prepared confirmation.
- interruption leaves the exact pending request checkpointed and resume executes it before replanning.
- replay phase costs advance a simulated clock and produce overrun.
- real evaluation executes the persisted exact request through the native scheduler and finishes from durable DB evidence.
