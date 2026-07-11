# One-Shape Workflow And Production Grid Readiness Review

## Production-Grid Search Quality And Scaling Findings

### G3. Screening Stabilization Is One-Shape-Specific And Its Duration Default Does Not Scale

Affected code:
- `evotensile/search/screening_stabilize.py`

The helper accepts exactly one shape and scans all artifact history for every call.

The default requests at least 100,000 microseconds of accumulated timed kernel duration but caps samples at 10. Any kernel faster than 10,000 microseconds cannot reach the duration target. Most small production-grid shapes will hit the cap without satisfying the stated reliability criterion.

Recommended correction:
- Use launch count, timer resolution, observed variance, and total runner duration as separate controls.
- Report when a duration target is unattainable under the sample cap.
- Support per-shape/cluster finalist queues rather than one global one-shape top-k.

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

### Stale Artifact References

`docs/blind_one_shape_experiment.md` still lists removed paths under its artifact section, including prior blind campaigns, aggregate analysis, and failed-attempt summaries. The chronological results can remain, but retained-artifact lists should contain only existing files or clearly label pruned paths as historical.

## Final Recommendation

Keep the current one-shape campaign as an experiment harness, not as the production grid controller. Its bookkeeping now uses explicit proposal events, active/archive diagnostics, strict configuration identity, resolved benchmark and validation state, and soft admission budgets with reported overruns. Reuse those validated components together with scoped proposals, grid-aware acquisition, explicit archive objectives, event-level operator credit, candidate representation, source-backed constraints, final-YAML mapping, structured validation, serial timing, probe separation, and DB ranking. A production grid controller still needs the unresolved clustering, workload allocation, deployment-bank, and evaluation work in `docs/plan.md`.

The GridBased updater defaults to a no-write preview and rejects empty or incomplete profile shape sets. The retained 100-shape timing corpus is consolidated under the current benchmark protocol, but still requires current validation evidence and registered complete artifacts before it is export-ready.
