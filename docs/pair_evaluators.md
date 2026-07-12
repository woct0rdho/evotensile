# Pair Evaluators

This document defines the controller-facing exact-pair evaluation contract in `evotensile/campaign/evaluator.py`. Scheduling details remain in `docs/exact_pair_scheduling.md`. Replay causality remains in `docs/blind_experiment_infrastructure.md`.

## Result Contract

Every evaluator consumes explicit `PairRequest` values and returns one `EvaluationResult` containing:
- one `PairEvaluationOutcome` per exact request.
- replay or native provenance and a source reference.
- resolved status, known/unknown state, disclosure state, sample count, and optional measured performance.
- exact candidate artifact-shape coverage made available by the evaluation.
- measured or simulated controller phase costs.
- native `ScheduleResult` records when a real scheduler ran.

`EvaluationResult.apply()` is the single controller handoff. It records artifact coverage, phase costs, exact query state, disclosure, incumbent updates, and an evaluation trace only after the evaluator has made its evidence durable.

## Replay Mode

`ReplayEvaluator` wraps `ExactOracleReplayState`:
- only an exact retained `(shape_id, candidate_hash)` key can produce known evidence.
- retained evidence is inserted into the evaluator's overlay DB with `source_kind='replay'` only after the exact query.
- absent pairs return `status='unknown'` and disclose nothing.
- preparation cost uses the configured replay workers and per-candidate cost while artifact coverage remains explicit by shape.
- repeated requests use target-sample semantics and add only missing samples. Exact throughput and protocol launches charge the corresponding probe, screening, stabilization, or confirmation phase.

`load_db_oracle_matrix()` opens retained SQLite corpora in URI `mode=ro`. The retained DB is therefore a read-only answer source, never the campaign evidence namespace.

## Real Mode

`RealEvaluator` wraps the exact scheduler with a fresh campaign DB or explicit overlay DB. It:
- sends only the admitted exact requests to `execute_schedule()`.
- preserves optional explicit artifact-shape supersets.
- derives outcomes from durable compatible benchmark and validation state.
- reports `source_kind='native_run'` evidence as native provenance.
- computes phase-cost deltas from typed `native_runs` rows for preparation, validation, probe, and screening.
- reports verified exact artifact mappings after the schedule.

The generic real campaign runner uses this evaluator rather than invoking the scheduler directly, so campaign policy does not depend on the evaluation mode.

## Hybrid Mode

`HybridEvaluator` requires replay and real evaluators to share one explicit overlay DB. It partitions requests by exact retained-oracle membership before querying:
- retained exact keys go to replay.
- absent exact keys go directly to the real evaluator.

The missing key is not first recorded as a replay unknown, because that would conflict with the native result that is about to become durable. Known replay evidence and native fallback evidence coexist in the overlay under distinct source kinds. The retained oracle mapping and source DB remain unchanged.

No neighboring pair, candidate, or shape is consulted. A retained neighbor can guide future policy only after its own exact request. It cannot answer the missing key.

## Artifact Expansion

The evaluator accepts `artifact_shapes_by_candidate` separately from requests. In real mode:
- the first scope determines the candidate-centric compile-cache identity.
- a later requested shape outside that scope requires an explicit expanded scope.
- the changed scope creates a distinct cache identity and a measured preparation run.
- controller artifact coverage expands only after the result is durable and applied.

Unrequested scope combinations still receive no validation, timing, artifact mapping, or benchmark evidence.

## Provenance And Visibility

The overlay DB is the only policy-visible evidence store. Replay inserts use replay provenance. Scheduler inserts use native-run provenance. `EvaluationResult.apply()` does not copy timing between DBs or invent compatibility. Newly measured native evidence can influence later controller decisions only after the exact scheduler result is present in the overlay and the result has been applied.

## Verification

Tests cover:
- replay-known and replay-missing exact outcomes.
- native evaluation with measured phase costs and controller disclosure.
- measured artifact-scope expansion and a new compile-cache identity.
- hybrid replay of one retained pair plus native fallback for one absent pair.
- absence of an unrequested retained neighbor from ranking and controller state.
- coexistence of replay and native provenance in one overlay.
