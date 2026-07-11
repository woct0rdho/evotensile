# Screening-Leader Stabilization

This document describes `evotensile/search/screening_stabilize.py`, which adds bounded main-protocol evidence for provisional leaders between search rounds. General timing statistics and adaptive per-schedule sampling are documented in `docs/noisy_measurements.md`. Final hot-loop confirmation is documented in `docs/tensilelite_measurement.md`.

## Purpose

A cheap main screening protocol can rank many candidates, but a two-sample leader is too noisy to strongly control archive promotion, surrogate training, learned linkage, or operator credit. Screening-leader stabilization adds a bounded amount of evidence before a later proposal call consumes that ranking.

It is distinct from:
- the staged catastrophic probe, which removes the very slow tail under a separate probe protocol.
- adaptive within-schedule finalist top-ups, which resolve close candidates from one prepared schedule.
- hot-loop confirmation, which is the final reporting protocol and does not feed the active search.

## Measured Default Basis

The retained current-protocol 100-shape database contains `153,784` validation-passed timing rows over `8,728` candidate/shape pairs. Across each shape's four fastest measured candidates, median kernel times range from `6.67` to `243.65` microseconds, with quartiles `20.56`, `35.95`, and `76.32` microseconds.

Under the campaign screening protocol's one launch per sample, the old `100,000`-microsecond accumulated-duration target required roughly `411` to `14,984` samples for those finalists and was therefore incompatible with its 10-sample cap. Observed top-four noise also shows that a narrow confidence-width criterion can require far more samples than a search budget permits. The replacement policy treats each evidence goal independently and reports which goals exceed the cap.

## Policy

`ScreeningStabilizationPolicy` defaults are:

```text
top_k = 4
contender_epsilon_pct = 3.0
confidence = 0.90
min_samples = 8
max_samples = 24
sample_step = 2
min_launches = 8
timer_resolution_us = 1.0
min_timer_ticks = 100
uncertainty_half_width_pct = 10.0
noise_floor_pct = 1.0
max_pairs_per_run = 16
max_runner_duration_s = 30.0
```

The campaign persists these values in its immutable campaign configuration. These are bounded screening defaults, not a claim that every finalist is statistically resolved after 24 samples.

## Contender Selection

For each shape, `plan_screening_stabilization()` ranks candidates by median log time and considers only the first `top_k` candidates. For contender `c` and current leader `b`:

```text
gap_log = score_c - score_b
combined_se = sqrt(SE_c**2 + SE_b**2)
ci_low_log = gap_log - z * combined_se
```

The leader is always eligible. Another contender is eligible only when its lower confidence bound does not prove it slower than the leader by more than `contender_epsilon_pct`. This is a top-up allocation decision, not a final equivalence claim.

## Independent Evidence Targets

For each eligible `(shape, candidate)` pair, the planner computes four sample targets under the unchanged main screening protocol:

```text
sample_target = min_samples
launch_target = ceil(min_launches / launches_per_sample)
timer_target = ceil(
    timer_resolution_us * min_timer_ticks /
    (median_time_us * launches_per_sample)
)
uncertainty_target = ceil(
    (z * 1.2533 * max(robust_sigma_log, noise_floor_log) /
     uncertainty_half_width_log) ** 2
)
```

The requested target is the maximum of those criteria, rounded to `sample_step`, then capped at `max_samples`. Every finalist record retains the uncapped target, each component target, and `capped_criteria`. Hitting the cap therefore does not silently imply that timer-resolution or uncertainty goals were achieved.

`EnqueuesPerSync` and `SyncsPerBenchmark` are part of benchmark-protocol identity. Stabilization never changes them to manufacture more launches. It requests additional samples under the original protocol so rows remain compatible ranking evidence.

## Grid And Cluster Queues

The planner accepts any nonempty shape sequence. One-shape use is the one-element special case. A caller may supply an exact shape-to-cluster mapping. Otherwise each shape forms its own cluster.

Eligible finalists form per-shape queues. The planner round-robins clusters, then shapes within each cluster, preserving rank within each shape. `queue_index` records the resulting deterministic order. Execution batches only adjacent compatible requests, so library sharing cannot leapfrog another cluster or shape under a tight budget.

This queue is an allocation mechanism only. It does not infer winners for unmeasured shapes or transfer evidence between cluster members.

## Artifact Reuse

`stabilize_screening_leaders()` never recompiles or revalidates candidates. It requires:
- compatible passed validation evidence for the exact `(shape, candidate)` pair.
- a content-verified artifact registered for that exact pair.
- the unchanged main screening protocol identity.

Artifact lookup uses indexed `artifact_mappings` filtered by problem type, shape, and candidate, then verifies the shared bundle. It does not scan run-directory history or reconstruct artifacts from manifests and pair files.

## Execution And Admission

Each admitted top-up run:
- uses structured-runner `benchmark` mode.
- sets `NumElementsToValidate=0`.
- preserves the original main benchmark-protocol hash while requesting only missing samples.
- records exact candidate costs and phase `screening_stabilization` in SQLite.
- validates row identity, sample count, solution index, timing fields, and runner return code before insertion.

Three separate bounds apply:
- `max_samples` caps evidence requested for one pair.
- `max_runner_duration_s` stops admission of later groups after measured stabilization runner time reaches the soft budget.
- the caller's `admission_deadline` stops admission under the surrounding campaign budget.

An admitted run keeps its full configured runner timeout. Results report measured runner duration and whether its budget was exhausted. Missing validation/artifacts, deadline skips, runner-budget skips, and ingestion failures are attributed to exact pairs with explicit reasons. They do not create reusable negative evidence.

## Search Integration

The blind one-shape driver invokes the grid-capable API with one shape after a round's normal schedule and before the next proposal call. A future grid coordinator can pass its active shapes and cluster assignments without copying one-shape logic. Resulting rows can affect:
- global, shape-local, and island leader selection.
- family archive ranking.
- ExtraTrees training.
- learned-linkage evidence.
- child-versus-parent operator, semantic-group, and donor credit.

The feature is controlled by the campaign's `leader_stabilization` policy and `--no-leader-stabilization`. It is not automatically enabled for every `schedule-batches` invocation.

## Result Records

`ScreeningStabilizationResult` reports:
- all considered finalists and requested pairs with cluster, rank, queue, gap, and component targets.
- uncapped targets and explicit capped criteria.
- completed pairs and exact skipped-pair reasons.
- runner invocation count, inserted sample count, measured runner duration, and total duration.
- runner-budget exhaustion and ingestion errors.

The blind campaign stores this structure in each round record.

## Invariants

- Only validation-passed main-protocol timing rows can become positive evidence.
- Candidate hashes and parameters are unchanged.
- No hidden or unqueried oracle value enters contender selection.
- Top-ups reuse authoritative generated artifacts.
- Cluster membership changes queue order only, never correctness or ranking evidence.
- Stabilization evidence remains screening evidence. Final claims require hot-loop confirmation.

## Limitations

- Confidence uses the same approximate median-log standard error as adaptive timing.
- Timer resolution is an explicit policy input rather than hardware-probed automatically.
- The 30-second runner budget is a soft admission limit. One admitted invocation may finish beyond it.
- The value of stabilization relative to its wall-time cost has not been isolated in a full real multi-seed grid ablation.
- Experiment outcomes and observed screening-to-hot gaps belong in `docs/experiment_blind_one_shape.md`.
