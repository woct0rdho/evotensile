# Final confirmation and deployment selection

Deployment selection is a separate boundary after campaign screening, promotion, and repair. It never treats model predictions or screening incumbents as production evidence. The output is an explicit per-shape assignment consumed by GridBased generation.

## Confidence-aware finalists

`plan_stabilization_finalists()` and `plan_confirmation_finalists()` consume calibrated contextual pair-model predictions. For each shape and posterior sample, a candidate is close when its normalized log performance is within the configured relative-loss tolerance of that sample's best candidate. The close probability is multiplied by the predicted validity probability.

The planner:
- always retains the current exact incumbent.
- adds posterior-close competitors above the configured probability floor.
- caps finalists independently per shape.
- uses the controller's normalized shape weights in exact request priority.
- emits explicit `(candidate, shape)` requests only.
- uses `STABILIZATION` evidence with a lower sample target before `CONFIRMATION` evidence with the hot sample target.

After stabilization outcomes are disclosed, the contextual model is refitted before final confirmation planning. Missing oracle pairs remain unknown in replay. Hybrid or real campaigns route genuinely absent pairs through native scheduling rather than copying neighbor performance.

## Soft confirmation admission

`run_final_confirmation()` groups exact finalist requests by candidate so one prepared artifact may cover several explicitly requested shapes. Before each candidate group, the controller checks a conservative predicted duration against the remaining soft budget. An admitted group keeps its normal build and runner timeouts and drains completely. If it overruns, the run records the overrun and admits no later group.

The run report records every admission decision, exact request key, completed group, known and unknown pair count, provenance, samples, and performance.

Production confirmation uses a dedicated `RealEvaluatorContext(ignore_cache=True)`. This deliberately ignores compatible timing and validation cache hits when planning the selected production pairs, forcing fresh validation and fresh timing under the requested confirmation protocol. Confirmation-stage native benchmark duration is charged to the controller's `confirmation` phase.

The older `hot_confirm_topk()` helper remains a diagnostic ranking tool for blind singleton runs. It reuses validation and does not insert confirmation timing into the DB, so it is not the production-export gate.

## Greedy solution bank

`select_deployment_solution_bank()` accepts positive disclosed `CONFIRMATION` outcomes only. For each required shape it first identifies the exact confirmed winner. A candidate covers a shape at tolerance `t` when:

```text
confirmed(candidate, shape) >= exact_winner(shape) * (1 - t)
```

At zero tolerance, selection bypasses consolidation and preserves the deterministic exact confirmed winner for every shape. The solution bank is the unique set of those assignments.

For nonzero tolerance, deterministic greedy set cover selects the candidate covering the largest number of uncovered shapes. Ties prefer greater workload mass, lower mean confirmed loss, and then candidate hash. A reverse redundancy pass removes selected candidates whose removal preserves complete coverage. Each shape is finally assigned its fastest qualifying selected candidate.

The result reports:
- exact winners and deployed assignments.
- uniform mean, workload-weighted mean, and worst-shape loss.
- selected solution and code-object counts.
- assigned shapes for every selected candidate.
- shapes covered by each multi-shape generalist.
- shapes requiring one-shape specialists.
- the exact shape weights used.

When artifact content identities are available, code-object count is the number of distinct selected identities. Otherwise the report conservatively counts one logical code object per selected candidate and marks the count conservative. Singleton selection always returns its confirmed winner with one solution and zero loss.

## GridBased export gate

`update_hipblaslt_gridbased_logic.py --selection-json PATH` loads a serialized `DeploymentSelection` instead of re-ranking the DB. This prevents a tolerated bank assignment from being silently replaced by the fastest per-shape DB row.

Before rendering, the exporter requires:
- exactly one assignment for every profile shape, with no missing or extra shapes.
- positive confirmation timing for every selected exact pair under the requested benchmark protocol and minimum sample count.
- latest passed validation for every selected exact pair under the requested validation identity.
- a complete, content-verified registered artifact mapping for every selected exact pair.
- a full generated solution dictionary matching every selected candidate.

Generated GridBased logic contains one solution dictionary per selected candidate and one exact row per required shape. `--allow-partial` remains development-only and is not valid for production deployment.

## Controlled replay

`out/grid100_deployment_selection_20260712.json` records three stable anchored-untuned trials. Each trial uses the frozen P12 policy for one 385-pair campaign increment, stabilizes up to three posterior-close finalists per shape, refits the contextual model, and confirms finalists under a 300-second soft confirmation budget. No native builds or external source changes were performed. Absent retained pairs remain unknown.

Across the three trials, confirmation retained complete positive coverage for all 100 shapes and selected 11-12 exact confirmed winners, with a mean of `11.33` solutions. Those candidates were already broad exact winners: requested deployment tolerances of 1%, 2%, and 5% did not reduce the bank further and therefore reported zero deployed loss. The exact bank averaged `10.33` multi-shape generalists and one specialist shape.

This result does not establish a universal preferred tolerance. Zero tolerance remains the production-safe default, while nonzero consolidation remains an explicit deployment choice whose reported loss and code-object savings must be reviewed for the actual confirmed campaign.
