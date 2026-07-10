# Screening-Leader Stabilization

This document describes `evotensile/search/screening_stabilize.py`, which adds reliable main-protocol evidence for provisional leaders between search rounds. General timing statistics and adaptive per-schedule sampling are documented in `docs/noisy_measurements.md`. Final hot-loop confirmation is documented in `docs/tensilelite_measurement.md`.

## Purpose

A cheap main screening protocol can rank many candidates, but a two-sample leader is too noisy to strongly control archive promotion, surrogate training, learned linkage, or operator credit. Screening-leader stabilization adds a bounded amount of evidence before the next proposal round consumes that ranking.

It is distinct from:
- the staged catastrophic probe, which removes the very slow tail under a separate probe protocol.
- adaptive within-schedule finalist top-ups, which resolve close candidates from one prepared schedule.
- hot-loop confirmation, which is the final reporting protocol and does not feed the active search.

## Policy

`ScreeningStabilizationPolicy` defaults are:

```text
top_k = 4
contender_epsilon_pct = 3.0
confidence = 0.90
min_samples = 6
max_samples = 10
sample_step = 2
min_timed_duration_us = 100000
```

The campaign may override these values in its frozen policy.

## Contender Selection

`screening_topup_requests()` loads compatible main-protocol timing statistics and ranks candidates by median log time. It considers only the first `top_k` candidates.

For contender `c` and current leader `b`:

```text
gap_log = score_c - score_b
combined_se = sqrt(SE_c² + SE_b²)
ci_low_log = gap_log - z × combined_se
```

The leader is always eligible. Another contender is eligible only when its lower confidence bound does not prove it slower than the leader by more than `contender_epsilon_pct`.

This is a top-up allocation decision, not a final equivalence claim.

## Sample Target

The minimum accumulated timing-duration target is converted into samples using the current median kernel time and launches per screening sample:

```text
duration_target = ceil(
    min_timed_duration_us /
    (median_time_us × enqueues_per_sync × syncs_per_benchmark)
)
```

The target is then:

```text
max(min_samples, duration_target)
rounded up to sample_step
capped at max_samples
```

Candidates already at or above the target receive no request. This combines an evidence-count floor with a duration floor so very fast kernels are not treated as reliable after only a tiny accumulated timing interval.

## Artifact Reuse

`stabilize_screening_leaders()` never recompiles or revalidates candidates. It requires:
- compatible passed validation evidence for the exact `(shape, candidate)` pair.
- a previously recorded generated library and mapped `RunnablePair`.
- the unchanged main screening protocol identity.

`load_candidate_artifacts()` reconstructs reusable artifacts from structured-runner command metadata and pair files. Requests are grouped by generated library directory and remaining sample count so compatible candidates share one benchmark-only invocation.

## Execution And Ingestion

Each top-up run:
- uses structured-runner `benchmark` mode.
- sets `NumElementsToValidate=0`.
- preserves the original main benchmark-protocol hash even when the execution requests only the missing sample count.
- records phase `screening_stabilization` in the `runs` table.
- validates row identity, sample count, solution index, timing fields, and runner return code before insertion.

A campaign deadline can shorten the runner timeout. Requests without compatible validation/artifacts, requests that miss the deadline, and failed result ingestion are reported as skipped or errors. They do not create reusable negative candidate evidence.

## Search Integration

The blind one-shape driver invokes stabilization after one round's normal schedule and before the next proposal call. The resulting rows can therefore affect:
- global and island leader selection.
- family archive ranking.
- ExtraTrees training.
- learned-linkage evidence.
- child-versus-parent operator, semantic-group, and donor credit.

The feature is controlled by the campaign's `leader_stabilization` policy and `--no-leader-stabilization`. It is not automatically enabled for every `schedule-batches` invocation.

## Result Records

`ScreeningStabilizationResult` reports:
- all requested candidates and their rank/gap/confidence/sample targets.
- completed and skipped candidate hashes.
- runner invocation count.
- inserted positive sample count.
- total duration and errors.

The blind campaign stores this structure in each round record.

## Invariants

- Only validation-passed main-protocol timing rows can become positive evidence.
- Candidate hashes and parameters are unchanged.
- No hidden or unqueried oracle value enters contender selection.
- Top-ups reuse authoritative generated artifacts.
- Stabilization evidence is still screening evidence. Final claims require hot-loop confirmation.

## Limitations

- Confidence uses the same approximate median-log standard error as adaptive timing.
- Artifact discovery depends on retained run metadata and pair files.
- The policy is global for one shape and does not yet provide family-specific stabilization tiers.
- The value of stabilization relative to its wall-time cost has not been isolated in a full real multi-seed ablation.
- Experiment outcomes and observed screening-to-hot gaps belong in `docs/blind_one_shape_experiment.md`.
