# Exact Pair Scheduling

This document defines the production scheduler boundary for candidate-shape evaluation. TensileLite build configuration remains candidate-centric, but validation, timing, cache decisions, manifests, and evidence insertion are exact-pair operations.

## Request Contract

`PairRequest` identifies:
- one canonical `Candidate`.
- one exact `Shape`.
- an `EvidenceStage` policy label.
- the minimum compatible main-protocol sample count required after the schedule.
- a finite policy priority used for stable admission and batching order.

The public `execute_schedule()` boundary accepts only `requests`. It does not accept independent candidate and shape lists and never materializes their product. A dense experiment must construct every desired `PairRequest` itself before calling the scheduler.

Requests are deduplicated by `(shape_id, candidate_hash)`. Byte-for-byte equivalent requirements collapse to one request. Different stage, minimum-sample, or priority requirements for the same exact pair raise a deterministic conflict error rather than silently selecting one.

## Planning

`plan_pair_requests()` applies every decision to requested keys only:
- shape-dependent static validity.
- compatible benchmark sample counts.
- reusable `rejected` and `build_failed` evidence.
- latest compatible validation pass or failure.
- optional probe-policy screening.
- explicit caller exclusions.

A positive pair is planned only for its remaining sample deficit. Validation need is stored per `PlannedPair`, so one build may contain a mixture of cached validation passes and pairs requiring fresh verification.

`PlannedBatch` contains exact `pairs` plus separate `artifact_candidates` and `artifact_shapes`. It reports only requested-pair and requested-sample counts. Rectangular nominal/extra accounting does not exist.

## Artifact Scope

By default, a candidate's artifact scope is exactly the shapes requested for that candidate. A caller may provide `artifact_shapes_by_candidate` with an explicit superset. Every requested shape must be present in that scope, and scopes for candidates without requests are rejected.

The generated TensileLite YAML contains the batch's explicit artifact candidates and artifact shapes because GridBased code generation is candidate- and shape-dependent. This build layout may be rectangular internally. It does not define evaluation work.

With stable compile caching, batches are candidate-centric and the cache key uses the candidate plus sorted explicit artifact-shape scope and compile-relevant profile identity. An artifact-scope change therefore creates a distinct cache identity rather than pretending an old library covers a new shape.

## Manifest And Mapping

`config.manifest.csv` contains only requested pairs. Candidate and problem indices refer to positions in the explicit artifact candidate and shape lists used by the YAML. Unrequested artifact-scope combinations receive no manifest row.

Final-YAML mapping accepts only manifest rows whose exact key is planned. Artifact registration creates mappings only for those runnable requested pairs. Missing final mappings, build failures, diagnostics, and timeouts are attributed only to requested keys, never to the artifact-scope cross product.

## Validation And Timing

Preparation divides mapped requested pairs into:
- pairs with compatible cached validation passes.
- pairs requiring fresh structured validation.

Only the second subset is sent to validation mode. The admitted union is then grouped by its pair-specific remaining sample count and sent to benchmark mode. Adaptive probe and top-up decisions are filtered through the same exact available-pair set.

Structured runner pair files, validation rows, benchmark rows, artifact mappings, rejection rows, and timeout rows therefore remain subsets of the original request keys. An unrequested artifact pair cannot become proposal-visible evidence as a side effect of sharing a build.

## Production Callers

The general CLI, generic real campaign evaluator, installed hipBLASLt baseline discovery, and integrated repair acquisition materialize their intended requests explicitly. CLI search creates its declared dense request list. Campaign repair emits only selected candidate-shape pairs from shared-cost acquisition. Baseline discovery uses the exact discovered selection pairs. Screening stabilization operates on exact finalist pairs and registered exact artifact mappings. Shared bundle acquisition emits sparse exact requests plus separate explicit artifact scopes, preparation order, and timing priorities as documented in `docs/shared_bundle_acquisition.md`. Replay/real/hybrid mode selection is documented in `docs/pair_evaluators.md`.

## Verification

Tests cover:
- identical-request deduplication and conflicting-request rejection.
- sparse non-rectangular plans.
- explicit artifact-shape supersets.
- cache filtering and remaining-sample calculation.
- mixed cached/fresh validation state in one batch.
- missing and rejected final artifact mappings.
- a sparse multi-candidate, multi-shape build where only requested diagonal pairs are manifested, validated, mapped, timed, and inserted.
