# Adaptive Search Operator Portfolio

This document describes EvoTensile's semantic mutation operator and adaptive allocation of variation budget between mutation, DE, GOMEA arms, semantic groups, and donor modes. The portfolio is an optional part of `family-qd`. General proposal modes are documented in `docs/search_algorithms.md`. Measured cost scaling is documented in `docs/search_cost_model.md`.

## Motivation

A fixed operator mix wastes a practical tuning budget when some operators repeatedly produce invalid, duplicate, or slower children. At the same time, permanently disabling an operator from a short run can remove useful exploration.

The adaptive portfolio therefore:
- gives every operator a minimum allocation.
- learns only from queried child-versus-parent timing evidence.
- rewards operators that improve on their measured parents.
- retains an uncertainty bonus for under-sampled operators.
- keeps operator identity in candidate lineage for audit.

The policy contains no profile winner values or performance-derived static parameter bundles.

## Operator Arms

The adaptive arms are:

```text
semantic-mutation
de
gomea-neighborhood
gomea-mixing
```

The scheduler keeps these source names distinct when `--adaptive-operators` is enabled. Optional group credit, bounded neighborhood enumeration, and adaptive donor selection have separate flags, so existing proposal behavior remains unchanged unless explicitly enabled.

## Semantic Groups

`evotensile/search/semantics.py` defines generic mechanical groups for NT HHS:
- tile, workgroup, reduction, and GSU.
- LDS transpose and padding.
- prefetch and local-read staging.
- global/local vector widths.
- scheduling, mapping, staggering, and source swap.
- store scheduling and store vectorization.
- pointer-swap behavior.
- assertion multiples.

Singleton groups for every parameter are also included. The same group vocabulary is reused by semantic mutation and the static GOMEA neighborhood operator.

These groups describe mechanical roles. They do not prioritize a specific value inside a group.

## Semantic Mutation

`semantic_mutation_candidates()` makes deliberately small local moves:
- Choose a measured parent.
- Choose one semantic or singleton group, optionally weighted by queried group-level UCB credit.
- Change one or two mutable genes to random alternative domain values.
- Run linked repair.
- Require at least one eligible pair in the declared proposal scope.
- Reject invalid, duplicate, excluded, or unchanged-parent candidates.

Linked repair can change additional dependent fields, so final Hamming distance may exceed two. The operator is still substantially more local than independently mutating every gene with the default `0.25` probability.

## GOMEA Arm Separation

The previous single `gomea` identity is split into:
- `gomea-neighborhood`: one-parent static semantic/singleton trials around ranked elites.
- `gomea-mixing`: two-parent FOS mixing using static and learned linkage groups.

Separating the identities allows the scheduler to learn whether local refinement or donor recombination is currently more productive. Detailed mixing behavior is documented in `docs/search_gomea.md`.

## Credit Evidence

`load_operator_credits()` reads selected append-only proposal occurrences and ranked timing evidence for the active problem, benchmark protocol, and target shapes. Candidate registration remains first-writer-owned, but credit never reads source, parents, or metadata from the candidate row. A later operator appearance of an existing parameter hash is therefore independently attributable.

A proposal occurrence contributes one event-level trial only when:
- its occurrence source is one of the adaptive arms.
- compatible child timing was queried after the occurrence.
- at least one occurrence parent has compatible positive timing for the same shape.

For every comparable shape, the reference is the fastest measured occurrence parent. For parent time `t_p` and child time `t_c`:

```text
log_speedup = log(t_p / t_c)
success = t_c <= t_p * (1 - minimum_improvement_fraction)
```

The event reward is the workload-weighted mean `log_speedup` across comparable shapes. Equal shape weights are the default fixed-grid objective. Callers may provide workload weights. The default event success threshold is `0.5%`. Each arm records successes, failures, event trials, shape comparisons, cumulative event log speedup, and cumulative evaluation cost. Evaluation cost is charged once per event, not once per shape.

Only queried DB evidence after the occurrence affects credit. The latest compatible selected occurrence owns the candidate's current post-occurrence timing reward, so repeated occurrences cannot multiply one measurement into several trials. Unselected occurrences, cache-only reproposals, proposed but unmeasured children, validation failures, and hidden oracle rows do not contribute positive or negative performance credit.

The same queried child outcomes also update proposal-metadata credits:
- semantic-group credit for semantic mutation and GOMEA neighborhoods.
- donor-mode credit for quality, diverse, and random GOMEA mixing donors.

Candidate hashes remain parameter-only. Occurrence source, parents, metadata, scope, selection state, and protocol identity are append-only event data and do not change cache identity.

## UCB Allocation

`allocate_operator_budget()` starts every arm with `minimum_per_arm`, which defaults to one when the budget permits.

For arm `a`, the score is:

```text
posterior_mean_a = (successes_a + 1) / (trials_a + 2)
exploration_a = sqrt(2 * log(total_trials + 2) / (trials_a + 1))
score_a = posterior_mean_a + exploration_a
```

The remaining budget is distributed proportionally to these scores, with deterministic remainder handling. With no evidence, all arms receive an equal allocation. Successful arms gain budget, while the exploration term prevents permanent starvation. When cost-aware credit is enabled, each score is multiplied by a bounded square-root ratio between the middle sorted arm cost and that arm's smoothed average cost. Minimum exploration is assigned before this scaling.

The configured local, DE, and GOMEA counts define the total adaptive variation budget. Random exploration remains separate.

## Scheduler Integration

Enable whole-operator allocation with:

```text
--proposal family-qd
--adaptive-operators
```

Related opt-in controls are:

```text
--adaptive-group-credit
--micro-exhaustive-neighborhoods
--adaptive-donor-selection
--cost-aware-operator-credit
```

The scheduler:
- loads family and global elites.
- computes current operator credits.
- allocates the oversized variation budget.
- generates each arm independently.
- records source and parent hashes.
- passes the combined pool to optional surrogate shortlisting.

Operator allocation changes proposal counts only. It does not change validity, DB ranking, or benchmark sampling.

## Interpretation And Limitations

Observed child improvement is noisy, especially with two-sample screening. Operator credit should therefore be interpreted as a budget heuristic rather than proof that one algorithm dominates.

Current limitations:
- credit uses current DB medians without temporal decay.
- success is binary for allocation even though cumulative log speedup is recorded.
- cost attribution divides shared run duration between candidates and is not exact per-candidate profiling.
- credit aggregates selected target-shape outcomes at event level but is not yet conditioned by family or search phase.
- screening evidence remains noisy even when the blind campaign stabilizes provisional global leaders.

Campaign-level leader stabilization is documented in `docs/search_screening_stabilization.md`. It improves high-impact ranking evidence but does not make every parent/child comparison a controlled retime.
