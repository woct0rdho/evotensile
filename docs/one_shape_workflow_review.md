# One-Shape Workflow And Production Grid Readiness Review

## Production-Grid Search Quality And Scaling Findings

### Monolithic Responsibilities

- `scripts/run_blind_one_shape.py` still combines configuration, proposal orchestration, checkpoint serialization, resume, soft-budget accounting, execution, stabilization, diagnostics, and confirmation.
- `evotensile/scheduler.py` still combines proposal generation, transfer, outlier detection, batch planning, compile caching, preparation, validation, probe racing, timing, and adaptive top-ups.

The issue is not line count by itself. The current boundaries make it difficult to reuse campaign control for a grid without copying one-shape assumptions.

Recommended extraction:
- evidence snapshot and acquisition interfaces.
- preparation-wave executor.
- serialized timing allocator.

## Final Recommendation

Keep the current one-shape campaign as an experiment harness, not as the production grid controller. Its bookkeeping now uses explicit proposal events, active/archive diagnostics, strict configuration identity, resolved benchmark and validation state, and soft admission budgets with reported overruns. Reuse those validated components together with scoped proposals, grid-aware acquisition, explicit archive objectives, event-level operator credit, candidate representation, source-backed constraints, final-YAML mapping, structured validation, serial timing, probe separation, and DB ranking. A production grid controller still needs the unresolved clustering, workload allocation, deployment-bank, and evaluation work in `docs/plan.md`.

The GridBased updater defaults to a no-write preview and rejects empty or incomplete profile shape sets. The retained 100-shape timing corpus is consolidated under the current benchmark protocol, but still requires current validation evidence and registered complete artifacts before it is export-ready.
