# One-Shape Workflow And Production Grid Readiness Review

## Scope

This review traces the current workflow through:

- `scripts/run_blind_one_shape.py` campaign control.
- Candidate generation and feedback in `evotensile/scheduler.py` and `evotensile/search/`.
- Cache-aware build, validation, probe, screening, stabilization, and confirmation.
- SQLite evidence, lineage, cost, and artifact handling.
- General `schedule-batches` behavior for a production shape grid.
- Downstream GridBased logic generation in `scripts/update_hipblaslt_gridbased_logic.py`.
- The corresponding design and experiment documentation.

The review distinguishes current one-shape correctness from production-grid readiness. Some mechanisms are acceptable for a fixed one-shape experiment but are not safe or well-defined when candidates, evidence, and budgets span many shapes.

## Overall Assessment

The basic measurement boundary is sound:

- TensileLite remains authoritative for build/codegen and final solution mapping.
- Correctness and timing execute as separate structured-runner phases.
- Compilation can run concurrently while benchmark timing remains serialized.
- Probe timing has a separate protocol identity and does not enter main ranking.
- Positive ranking uses finite timing rows marked as backed by validation.
- Shape-dependent static rejection is represented per `(shape, candidate)` pair.

However, the workflow is not yet safe as an unattended production shape-grid pipeline. The highest-priority problems are:

- Grid proposal generation applies shape validity to the intersection of all target shapes, preventing many specialists.
- Multi-shape parent selection, transfer, surrogate activation, and uncertainty do not implement a coherent grid objective.
- One-shape merged-round bookkeeping misclassifies archive parents as generated candidates, corrupting cost and population diagnostics.

These should be resolved before treating the one-shape driver as the template for a production grid campaign.

## Severity Definitions

- Blocker: can write incorrect production artifacts, reuse incompatible evidence, or structurally prevent the intended production search.
- High: can materially corrupt search feedback, audit claims, resume behavior, or final performance attribution.
- Medium: important performance, scaling, default, or maintainability problem that is not normally correctness-destructive by itself.
- Low: cleanup, duplicated boilerplate, missing tests, or misleading documentation.

## Blockers

### B1. Production Grid Proposals Enforce The Intersection Of All Shape Constraints

Affected code:

- `evotensile/search_space.py:random_candidate()`
- `evotensile/scheduler.py:_shape_aware_random_batch()`
- `evotensile/search/family.py:_candidate_for_family()`
- `evotensile/search/local_search.py:semantic_mutation_candidates()`
- `evotensile/search/gomea.py:_gomea_candidate_ok()`

When a list of target shapes is supplied, these proposal paths require a generated candidate to satisfy shape-dependent constraints for every target shape.

The scheduler itself is already pair-oriented and can reject a candidate on one shape while measuring it on another. Proposal generation therefore applies a stricter all-grid intersection than execution requires.

This blocks important production specialists, including candidates with:

- GSU workspace valid for small output shapes but too large for another grid member.
- assertion multiples valid for a subset of shapes.
- tile or reduction choices appropriate only for one mechanical cluster.

The behavior is also inconsistent between operators:

- random, semantic mutation, and GOMEA use all-target shape checks.
- broad mutation and categorical DE primarily use global checks and can leave pair-specific rejection to the scheduler.

`random_candidate()` additionally forces `GlobalSplitU=1` whenever any `target_shapes` value is present. `nt_hhs_family_cells()` does the same. The domain contains `[1, 2, 4]`, but even a one-element small-shape target list produces only GSU1 cold families.

Recommended correction:

- Separate candidate-global validity from candidate-shape eligibility.
- Generate candidates for an explicit target scope: one shape, one cluster, or a declared required-shape subset.
- Allow specialists to proceed when they have at least one eligible target pair.
- Keep the scheduler's per-pair rejection authoritative.
- Replace unconditional GSU1 forcing with viability-aware sampling for the target scope.
- Record intended target scope in proposal events rather than in candidate identity.

### B2. Multi-Shape Parent And Transfer Selection Is Biased Toward Arbitrary Small Shapes

Affected code:

- `evotensile/scheduler.py:_ranked_elites()`
- `evotensile/scheduler.py:_nearest_shape_ids()`
- `evotensile/scheduler.py:_transfer_elites()`

When `shape_id` is not specified, `_ranked_elites()` asks `rank_evaluations()` for a global list of shape-candidate rows. Those rows are ordered by absolute time, so the selected parents tend to come from the fastest absolute shapes, usually the smallest grid members. They are not candidate aggregates and are not normalized against each shape's incumbent.

Transfer selection has a separate multi-target issue. `_nearest_shape_ids()` assigns each source shape its minimum distance to any requested target. If the DB already contains the full requested grid, every exact source shape has distance zero. The default limit then selects the first shapes by lexical shape ID rather than representative or per-target neighbors.

For a 100-shape request, `--transfer-shapes 4` therefore means approximately "four lexically first exact grid shapes," not "four useful neighbors for each unresolved target."

Recommended correction:

- Never use globally pooled absolute latency to select multi-shape parents.
- Select specialists per shape or cluster, and generalists by incumbent-normalized regret or coverage.
- Make transfer selection per target/cluster, followed by candidate deduplication and a global cap.
- Report which target shapes caused each transfer candidate to be selected.

### B3. Multi-Shape Surrogate Activation And Acquisition Are Not Grid-Safe

Affected code:

- `evotensile/search/surrogate.py:_training_rows()`
- `evotensile/search/surrogate.py:select_surrogate_pool()`

The activation threshold counts training rows, not unique candidates or adequately covered candidate-shape evidence. On a 100-shape grid, one candidate measured on all shapes produces 100 rows and exceeds the default threshold of 24 despite providing almost no candidate-gene variation.

The multi-shape acquisition then:

- averages predicted log time across target shapes.
- adds between-shape predicted variance to tree disagreement.
- uses that combined value as candidate uncertainty.

Between-shape variance mostly reflects real shape latency scale and specialization, not epistemic model uncertainty. It can dominate the uncertainty lane on a heterogeneous grid.

A single global shortlist also hides specialists whose gain on a few shapes is valuable but whose average prediction is mediocre. `docs/plan.md` already describes the correct future direction: incumbent-normalized marginal improvement with specialist, generalist, uncertainty, and unresolved-shape lanes.

Recommended correction:

- Activate by unique candidate count, shape coverage, and feature diversity rather than raw row count.
- Separate per-shape epistemic uncertainty from between-shape performance variation.
- Score marginal improvement against each shape incumbent.
- Select a candidate set by expected grid regret reduction or coverage, not one averaged scalar.
- Support shape/cluster-specific models when evidence is sparse or heterogeneous.

## Production-Grid Search Quality And Scaling Findings

### G1. Family Archive Scoring Does Not Require Comparable Shape Coverage

Affected code:

- `evotensile/search/family.py:load_family_archive()`
- `evotensile/search/learned_linkage.py:load_candidate_evidence()`

Both systems average shape-local percentiles only over shapes where a candidate has selected evidence. Missing shapes are not penalized or represented as uncertainty.

A candidate that is excellent on one measured shape can outrank a broadly measured candidate in the same family. This can be desirable for a specialist archive, but the current API returns one undifferentiated aggregate and then uses it as general breeding evidence.

Recommended correction:

- Separate specialist score, generalist score, coverage count, and unresolved-shape uncertainty.
- Require an explicit archive objective from callers.
- Do not compare sparse and broad candidates using the same unqualified mean percentile.

### G2. Operator Credit Treats Correlated Shape Outcomes As Independent Trials

Affected code:

- `evotensile/search/operator_credit.py:_queried_child_outcomes()`

One candidate evaluated on 100 shapes contributes up to 100 binary trials. Another candidate evaluated on one shape contributes one trial. These outcomes share one proposal event, one genome, and much shared preparation cost, but UCB treats them as independent Bernoulli evidence.

The system also uses first-registration candidate lineage. If the same parameter hash is later proposed by another operator or group, only the first source receives credit.

Recommended correction:

- Persist proposal events separately from candidate identity.
- Compute event-level and shape-level reward explicitly.
- Normalize or hierarchically model correlated per-shape outcomes.
- Support workload-weighted reward for production objectives.

### G3. Screening Stabilization Is One-Shape-Specific And Its Duration Default Does Not Scale

Affected code:

- `evotensile/search/screening_stabilize.py`

The helper accepts exactly one shape and scans all artifact history for every call.

The default requests at least 100,000 microseconds of accumulated timed kernel duration but caps samples at 10. Any kernel faster than 10,000 microseconds cannot reach the duration target. Most small production-grid shapes will hit the cap without satisfying the stated reliability criterion.

Recommended correction:

- Use launch count, timer resolution, observed variance, and total runner duration as separate controls.
- Report when a duration target is unattainable under the sample cap.
- Support per-shape/cluster finalist queues rather than one global one-shape top-k.

## Performance And Operational Findings

### P1. Production Concurrency Defaults Recreate The Known Validator Failure Mode

Affected code:

- `evotensile/scheduler.py:default_prepare_workers()`
- `evotensile/scheduler.py:production_candidate_batch_size()`
- `evotensile/cli.py:_add_execution_args()`

General CLI defaults are:

- prepare workers equal to full CPU affinity.
- no validation-worker cap.
- candidate batch size chosen to create at least as many batches as prepare workers.

On the current 32-CPU environment:

```text
64 candidates, 100 shapes, shape batch 100 -> candidate batch size 2
```

This can launch 32 concurrent TensileLite/validation pipelines. The same gfx1151 system already demonstrated ROCr/KFD instability with six concurrent validators, while the one-shape campaign explicitly uses `validation_workers=1` and eight prepare workers.

Recommended correction:

- Put preparation and validation defaults in the target profile/resource profile.
- Default gfx1151 validation concurrency to one until contrary evidence exists.
- Cap prepare workers by measured memory and compiler capacity rather than CPU count alone.
- Choose batch size from measured throughput and startup cost, not only worker saturation.

### P2. Cost-Aware Preparation Order Also Becomes Benchmark Order

Affected code:

- `evotensile/scheduler.py:execute_schedule()`

When cost-aware scheduling is enabled, `planned` batches are sorted by predicted preparation weight. `executor.map()` returns results in that sorted input order, and the serial benchmark loops use the same `prepared` list.

The feature therefore does two things:

- longest-predicted-work-first preparation.
- longest-predicted-work-first serialized timing.

Only the first behavior is documented and intended. Production timing should generally prioritize expected improvement, information gain, unresolved shapes, or deadline fit.

Recommended correction:

- Preserve separate preparation order and benchmark order.
- After the barrier, reorder prepared pairs by an explicit timing allocation policy.

### P3. The Full Prepare Barrier Delays Feedback And Prevents Responsive Admission

The hard no-overlap barrier is correct for the shared APU, but one `execute_schedule()` prepares the entire planned set before any timing or coordinator decision.

For a large grid this causes:

- long time to first useful measurement.
- large disk and artifact bursts.
- inability to stop after enough evidence is found.
- inability to revise shape/candidate acquisition between waves.
- deadline overshoot from one oversized schedule.

Recommended correction:

- Keep no preparation/timing overlap, but execute bounded waves.
- Prepare one wave in parallel, drain all preparation processes, benchmark serially, return feedback, then admit the next wave.
- Make wave size a resource- and deadline-aware policy.

### P4. Proposal Feedback Repeatedly Rescans The Complete DB And Artifact Tree

One family-QD proposal can independently call:

- operator credit loading.
- semantic-group credit loading.
- donor-mode credit loading.
- family archive loading.
- learned-linkage loading.
- surrogate training-row loading.
- candidate-cost reconstruction.

The three credit loaders each rebuild child outcomes and reread run artifacts. Family/archive code performs shape-specific ranking queries. Artifact consumers now use indexed pair/candidate filters, but proposal evidence still lacks one shared snapshot.

This was acceptable at roughly 1,000 one-shape candidates but will become expensive with 100 shapes and repeated rounds.

Recommended correction:

- Build one immutable evidence snapshot per proposal call.
- Compute operator, group, donor, linkage, family, surrogate, and cost views from that snapshot.
- Push timing aggregation and coverage queries into SQL where practical.

### P5. Compile Cache Is Batch-Composition-Sensitive And Its Lock Can Hang Forever

Affected code:

- `evotensile/scheduler.py:_compile_cache_key()`
- `evotensile/scheduler.py:_compile_cache_lock()`

The cache key contains the ordered candidate hash list. The same candidate in a different batch composition gets a different cache directory and may be recompiled.

The lock is an empty directory acquired by repeated `mkdir()`. It has no owner PID, timestamp, timeout, or stale-lock recovery. A killed worker can leave a permanent lock and future runs wait indefinitely.

Recommended correction:

- Add owner metadata, bounded wait, stale-owner detection, and explicit failure reporting.
- Normalize candidate ordering in the key.
- Investigate candidate-centric or composable code-object caching while preserving authoritative library generation.

### P6. Cached Probe Screening Cannot Avoid Re-Preparation

Batch planning considers main-protocol positive/negative cache rows before preparation. Probe-screened evidence is consulted only after all planned candidates are compiled and mapped.

If a previously probe-screened candidate is proposed again without main timing, it is planned as missing and prepared again before its cached probe evidence can screen it.

Recommended correction:

- Add a pre-prepare probe-decision cache step for complete compatible probe evidence.
- Distinguish "screened for this search policy" from static invalidity and allow explicit policy/version-based retry.

### P7. ExtraTrees Uses All CPU Threads Internally

`ExtraTreesRegressor(n_jobs=-1)` consumes the complete CPU pool during every model fit. It currently runs outside preparation, so it does not violate timing serialization, but concurrent campaigns or a future grid coordinator can oversubscribe heavily.

Recommended correction:

- Make surrogate jobs a profile/campaign resource setting.
- Include proposal CPU time in campaign admission and monitoring.

## Defaults And Consistency Findings

### D1. TargetProfile Defaults Are Not Consistently Used By The CLI

`TargetProfile` contains proposal counts, rates, batch sizes, timeouts, and runner defaults. Several CLI defaults are imported once from `DEFAULT_PROFILE`, including:

- random/proposal counts.
- mutation/crossover/random-gene rates.
- transfer counts.
- shape batch size.

Selecting a future non-default profile would still receive many gfx1151 default values.

Recommended correction:

- Resolve all profile-dependent defaults after parsing `--profile`.
- Use `None` at argparse level and fill values from the selected profile.

### D3. General Production Defaults Do Not Reflect The Evaluated One-Shape Policy

The general CLI still defaults to:

- `seed-random-gomea` rather than family-QD.
- surrogate multiplier 1.
- adaptive operator/group/donor credit off.
- cost-aware scheduling off.
- validation concurrency uncapped relative to preparation workers.

Keeping experimental features opt-in is reasonable, but `docs/plan.md` describes family-QD, staged surrogate search, and repair as the intended near-term standard loop. The repository needs a named production policy/profile rather than an undocumented mixture of low-level defaults.

Recommended correction:

- Add versioned named search policies such as `gfx1151-grid-v1`.
- Keep low-level flags but record the resolved policy in metadata.
- Do not silently change generic defaults based on one one-shape result.

### D5. Target-Specific Defaults Leak Into Generic Helpers

Examples:

- `candidate_shape_mechanics()` defaults to 20 effective CUs.
- family descriptors reject profiles other than `gfx1151-nt-hhs`.

These are acceptable for the current target but should be explicit profile fields or target implementations before presenting the surrounding APIs as reusable production infrastructure.

## Dead Code And Boilerplate Candidates

These are cleanup candidates after correctness fixes, not urgent standalone changes.

### Duplicated Helpers

- `_dedupe_candidates()` exists in CLI and scheduler code. Encoding also has a related helper.
- `_rank_percentiles()` is duplicated in family and learned-linkage modules.
- `_round_up()` is duplicated in adaptive timing and screening stabilization.

Consolidation would reduce semantic drift, especially around validation and final claims.

### Redundant CLI Path

`--learned-linkage` and `--no-learned-linkage` share one destination, but learned linkage is already enabled by default. The positive flag normally restates the default and can be replaced with `BooleanOptionalAction` or only the negative override.

### Monolithic Responsibilities

- `scripts/run_blind_one_shape.py` still combines configuration, proposal orchestration, checkpoint serialization, resume, soft-budget accounting, execution, stabilization, diagnostics, and confirmation.
- `evotensile/scheduler.py` still combines proposal generation, transfer, outlier detection, batch planning, compile caching, preparation, validation, probe racing, timing, and adaptive top-ups.

The issue is not line count by itself. The current boundaries make it difficult to reuse campaign control for a grid without copying one-shape assumptions.

Recommended extraction:

- evidence snapshot and acquisition interfaces.
- preparation-wave executor.
- serialized timing allocator.

## Documentation Issues

### Documentation That Overstates Current Behavior

- `docs/database.md` says validation identity includes validator version. It contains a manually incremented protocol version, but not the structured-runner binary, hipBLASLt version, ROCm version, GPU identity, or generated-library identity.
- `docs/search_algorithms.md` describes nearest-shape transfer without documenting multi-target zero-distance tie behavior.
- `docs/search_surrogate.md` documents a 24-row threshold without warning that one candidate across 100 shapes activates the model.

### Stale Artifact References

`docs/blind_one_shape_experiment.md` still lists removed paths under its artifact section, including prior blind campaigns, aggregate analysis, and failed-attempt summaries. The chronological results can remain, but retained-artifact lists should contain only existing files or clearly label pruned paths as historical.

### Useful Existing Documentation

`docs/plan.md` already identifies several production requirements correctly:

- incumbent-normalized grid acquisition.
- specialist and generalist lanes.
- staged shape evaluation.
- shape clustering.
- cost/information-aware serialized timing.
- full-grid outlier repair and final solution-bank construction.

Those sections should be treated as required design work, not optional polish around the current multi-shape implementation.

## Test Gaps

High-value missing tests include:

- Multi-shape elite selection is incumbent-normalized rather than absolute-time biased.
- Multi-target transfer covers intended targets/clusters rather than lexical ties.
- Surrogate activation requires enough unique candidate variation.
- Specialist candidates can be proposed and measured on eligible shape subsets.
- Profile-specific defaults resolve from the selected profile.
- Compile-cache stale-lock recovery.

The current tests are strong around normal scheduler execution, one-shape mechanics, proposal accounting, restart state, resolved benchmark evidence, strict resume identity, soft-budget overrun semantics, hot confirmation, artifact verification, and guarded production export. Multi-shape acquisition still receives little or no direct coverage.

## Recommended Remediation Order

### Phase 3: Define The Production Grid Objective

- Generate candidates for explicit shape/cluster scopes rather than the all-grid intersection.
- Replace global absolute-time elites with shape-normalized specialists/generalists.
- Replace multi-target transfer ties with per-target or per-cluster transfer.
- Implement incumbent-normalized grid acquisition.
- Separate specialist, generalist, coverage, and uncertainty archive objectives.
- Normalize operator reward across correlated shape outcomes.

### Phase 4: Improve Throughput And Maintainability

- Use profile resource defaults for preparation and validation concurrency.
- Separate preparation order from timing order.
- Build one evidence snapshot per proposal.
- Index derived costs.
- Add stale compile-lock handling and improve cache composability.
- Remove confirmed dead symbols and duplicated helpers.

## Final Recommendation

Keep the current one-shape campaign as an experiment harness, not as the production grid controller. Its bookkeeping now uses explicit proposal events, active/archive diagnostics, strict configuration identity, resolved benchmark and validation state, and soft admission budgets with reported overruns. Reuse those validated components together with candidate representation, source-backed constraints, final-YAML mapping, structured validation, serial timing, probe separation, and DB ranking, but replace scalar one-shape acquisition with a grid-aware specialist/generalist objective.

The GridBased updater defaults to a no-write preview, rejects empty or incomplete profile shape sets, and transactionally stages or replaces all selected variants. The retained 100-shape timing corpus is consolidated under the current benchmark protocol, but still requires current validation evidence and registered complete artifacts before it is export-ready.
