# Adaptive Search Operator Portfolio

This document describes EvoTensile's semantic mutation operator and the adaptive allocation of variation budget between mutation, DE, and GOMEA arms. The portfolio is an optional part of `family-qd`. General proposal modes are documented in `docs/search_algorithms.md`.

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

The scheduler keeps these source names distinct when `--adaptive-operators` is enabled. Without adaptive operators, existing proposal modes and source labels retain their previous behavior.

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
- Choose one semantic or singleton group.
- Change one or two mutable genes to random alternative domain values.
- Run linked repair.
- Apply shape-dependent cheap constraints for all target shapes.
- Reject invalid, duplicate, excluded, or unchanged-parent candidates.

Linked repair can change additional dependent fields, so final Hamming distance may exceed two. The operator is still substantially more local than independently mutating every gene with the default `0.25` probability.

## GOMEA Arm Separation

The previous single `gomea` identity is split into:
- `gomea-neighborhood`: one-parent static semantic/singleton trials around ranked elites.
- `gomea-mixing`: two-parent FOS mixing using static and learned linkage groups.

Separating the identities allows the scheduler to learn whether local refinement or donor recombination is currently more productive. Detailed mixing behavior is documented in `docs/search_gomea.md`.

## Credit Evidence

`load_operator_credits()` reads ranked timing evidence for the active problem, benchmark protocol, and target shapes.

A child contributes credit only when:
- its source is one of the adaptive arms.
- it has positive timing evidence.
- at least one recorded parent has compatible positive timing evidence for the same shape.

The reference is the fastest measured parent. For parent time `t_p` and child time `t_c`:

```text
log_speedup = log(t_p / t_c)
success = t_c <= t_p * (1 - minimum_improvement_fraction)
```

The default minimum improvement is `0.5%`. Each arm records successes, failures, trials, and cumulative log speedup.

Only queried DB evidence affects credit. Proposed but unmeasured children, validation failures, and hidden oracle rows do not contribute positive or negative performance credit.

## UCB Allocation

`allocate_operator_budget()` starts every arm with `minimum_per_arm`, which defaults to one when the budget permits.

For arm `a`, the score is:

```text
posterior_mean_a = (successes_a + 1) / (trials_a + 2)
exploration_a = sqrt(2 * log(total_trials + 2) / (trials_a + 1))
score_a = posterior_mean_a + exploration_a
```

The remaining budget is distributed proportionally to these scores, with deterministic remainder handling. With no evidence, all arms receive an equal allocation. Successful arms gain budget, while the exploration term prevents permanent starvation.

The configured local, DE, and GOMEA counts define the total adaptive variation budget. Random exploration remains separate.

## Scheduler Integration

Enable the portfolio with:

```text
--proposal family-qd
--adaptive-operators
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
- credit uses final DB medians without temporal decay.
- reward does not yet include compile, validation, or proposal cost.
- success is binary for allocation even though cumulative log speedup is recorded.
- credit is pooled across the selected target shapes rather than conditioned by family or search phase.
- provisional leaders are not automatically topped up before they influence credit.

A future noise-aware portfolio can require additional samples for high-impact parent/child comparisons while retaining the same source-neutral mechanics.
