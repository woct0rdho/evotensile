# Search Algorithms Design

This document describes EvoTensile's general search loop and proposal sources. Candidate-space construction, family-QD, GOMEA, learned linkage, adaptive operator allocation, mechanical coverage, cost modeling, surrogate shortlisting, screening stabilization, noisy measurement handling, TensileLite communication, outlier repair, and database semantics have separate design docs.

## Search Unit

The search unit is a `(shape, candidate)` pair:
- A `Shape` has exact `M`, `N`, `batch`, and `K` fields plus a stable `shape_id` such as `m512_n128_b1_k256`.
- A `Candidate` is a complete TensileLite solution dictionary with a stable parameter-only `cand_...` hash. Source, parent hashes, and proposal metadata remain audit lineage outside hash identity.
- A valid observation is a timed, validation-passed DB row for that pair under the active problem and benchmark protocol hashes.

Search algorithms only propose candidates. They do not select final winners directly. Winner selection comes from DB ranking and adaptive retiming.

## Scheduler Flow

`schedule-batches` is the main search entry point:
- Resolve the target profile, shapes, benchmark protocol, and SQLite DB.
- Propose candidates with the selected proposal mode, optional adaptive operator portfolio, and optional oversized surrogate pool.
- Register candidates and shapes in the DB.
- Record immediate shape-dependent rule rejections.
- Plan missing `(shape, candidate)` observations from reusable cache status.
- Emit TensileLite YAML and manifest files for exact rectangular batches.
- Run build/map/diagnostic/correctness work in bounded parallel preparation waves, optionally ordered longest-predicted-work first.
- Cap validation-runner concurrency independently without reducing compilation parallelism.
- Join every preparation worker in the admitted wave, then run that wave's benchmark-only work in one serial queue before admitting another wave.
- When adaptive sampling is enabled, top up plausible finalists from the original prepared artifacts only.
- Write `schedule_metadata.json` with proposal, protocol, linkage, batching, and execution details.

Timing begins only after the currently admitted preparation wave drains. This wave barrier prevents benchmark overlap with compilation or correctness verification on integrated CPU/GPU systems while allowing coordinator feedback and soft-budget admission between waves.

## Proposal Modes

`propose_candidates()` supports these proposal modes:

```text
random
seed-random
local
seed-random-local
de
seed-random-de
gomea
seed-random-gomea
evolutionary
family-qd
```

The default profile proposal is `seed-random-gomea`.

Named search policies are opt-in mechanism bundles, separate from target profiles. `--search-policy gfx1151-grid-v1` selects `family-qd`, an `8x` surrogate pool, adaptive operator/group/donor allocation, micro-exhaustive neighborhoods, covering cold start, measured-cost operator credit, and cost-aware preparation. Explicit low-level flags, including `--no-*` overrides and zero proposal counts, take precedence. Metadata records the policy name and every resolved policy-controlled value. The policy does not change profile resources, validity, measurement identity, or generic defaults.

`family-qd` is the family-aware quality-diversity proposal mode. Its descriptors, archive, and stratified initialization are documented in `docs/search_family_qd.md`.

### Proposal Scope

Every proposal has an explicit global, one-shape, cluster, or shape-set scope. Generated candidates must pass global rules and have at least one eligible `(shape, candidate)` pair in a non-global scope. They do not need to satisfy every shape in a cluster or shape set. Pair-specific rejection remains authoritative in the scheduler. Scope kind and shape IDs are recorded in proposal results, generated-candidate lineage, CLI metadata, and campaign proposal events without changing candidate identity. One-shape search is the one-element special case.

### Random

Random proposal samples from the target `DOMAINS`, applies linked repairs, and rejects candidates with known invalid rules. When a shape scope is declared, random generation requires at least one eligible pair in that scope. GSU is sampled from the full domain and shape viability decides whether the candidate can proceed.

The current NT HHS generator samples the compatible TLDS0 and TLDS2 construction branches with equal probability and applies a proposal-side VALU register headroom check. Branch balance and headroom are proposal policies, not validity rules.

### Local Mutation

Local mutation starts from DB elites and independently resamples each domain gene with probability `--mutation-rate` (`0.25` by default). Each child is checked through `make_candidate()`, and invalid children are discarded.

This mode is simple and useful around measured elites, but it does not preserve multi-field couplings as strongly as GOMEA.

When `family-qd` uses `--adaptive-operators`, the broad mutation arm is replaced by semantic mutation. Semantic mutation changes one mechanical group and only one or two genes before linked repair, producing much smaller local steps. It is documented in `docs/search_operator_portfolio.md`.

### Categorical Differential Evolution

The DE-inspired operator works over categorical genomes:
- Select one target parent and three donor parents.
- Force at least one gene to cross over.
- For each crossed gene, either sample a random domain value or copy from a donor when other donors disagree.
- Convert the genome back to a repaired candidate.

This is not numeric vector arithmetic. It is a discrete recombination heuristic for TensileLite categorical parameters.

### GOMEA

GOMEA mixes linked groups of genes from elite candidates. The scheduler splits the configured GOMEA budget between:
- a compact neighborhood sweep around ranked elites.
- stochastic GOMEA mixing using static and, when available, learned FOS groups.

GOMEA mechanics are documented in `docs/search_gomea.md`. Learned linkage is documented in `docs/search_linkage_learning.md`.

### Evolutionary

`evolutionary` combines random candidates, local mutations, categorical DE, and GOMEA into one proposal set. All candidates are deduplicated by hash before scheduling.

### Family QD

`family-qd` adds a family-aware quality-diversity layer:
- It computes coarse profile-specific family descriptors from candidate structure.
- It uses descriptor-stratified random generation with repeated family occupancy and one retry for negative-only cells.
- It loads a DB-derived archive scored by validation-passed shape-local rank percentiles.
- It retains up to four quality-bounded, Hamming-diverse elites per family.
- It preserves positive family elites as parents even when they are not global leaders.
- It applies mutation, categorical DE, GOMEA neighborhoods, and donor mixing around family/global elites when evidence exists.
- With adaptive operators, GOMEA prefers within-family donors when compatible donors are available.

Family descriptors and archive entries are proposal metadata only. They do not shrink domains, change cache identity, or create validity rules. Detailed behavior is documented in `docs/search_family_qd.md`.

## Adaptive Operator Portfolio

`--adaptive-operators` is currently used with `family-qd`. It separates variation into four measured arms:

```text
semantic-mutation
de
gomea-neighborhood
gomea-mixing
```

The scheduler loads compatible event-level child-versus-parent timing outcomes and allocates the variation budget with a UCB-style score. Correlated shape comparisons are workload-weighted within one proposal-occurrence trial, and evaluation cost is charged once per occurrence. Every arm retains minimum exploration when budget permits. Optional semantic-group and donor-mode credit use append-only occurrence metadata, so later appearances of an existing candidate hash remain attributable, and optional cost-aware credit scales UCB scores from measured proposal/evaluation cost.

The portfolio changes proposal counts and proposal ordering only. It does not change validity, measurement, or ranking. See `docs/search_operator_portfolio.md` and `docs/search_cost_model.md`.

## Surrogate-Guided Oversized Pools

`--surrogate-pool-multiplier N` generates `N` times the configured random and variation budget, then shortlists the original requested measurement count.

After at least `--surrogate-min-evidence` unique varied candidates on a shape, a shape-local ExtraTrees model predicts log median time and ensemble disagreement from candidate, shape, tile, vectorization, and resource features. Multi-shape acquisition compares predictions with each shape incumbent and separates complementary specialist gain, broad generalist regret, epistemic uncertainty, unresolved-shape mechanics, family diversity, and random exploration. Before any shape has enough evidence, multi-shape selection combines per-shape mechanical priority with family-diverse round-robin sampling.

Archive and transfer parents are preserved outside the generated-candidate shortlist. Before enough model evidence exists, an opt-in one-shape covering selector can use decomposed mechanical coverage instead of the default family round-robin fallback. The default multiplier is `1`, so oversized shortlisting remains opt-in. See `docs/search_surrogate.md` and `docs/search_mechanical_coverage.md`.

## Elite And Transfer Sources

Proposal modes that need one-shape parents load that shape's validation-passed DB ranking directly. Multi-shape parent selection never pools absolute latency. Half of the available parent lane is filled round-robin from shape-local specialist rankings. The remainder prefers candidates with broad measured coverage and low mean incumbent-normalized regret. One-shape search is the one-ranking special case.

For scoped schedules, the scheduler can also seed from nearest previously tuned shapes:
- `--transfer-shapes` controls the nearest source-shape depth considered independently for each target.
- `--transfer-per-shape` controls how many top candidates are copied from each source shape.
- Shape distance uses Euclidean distance in `log2(M)`, `log2(N)`, `log2(K)`, `log2(M/N)`, `log2(K/M)`, and `log2(K/N)`.
- Per-target candidate queues are drained round-robin, deduplicated, and bounded by one global transfer cap.

Transfer candidates record the target and source shape IDs that caused selection. Target lanes are ordered by deterministic farthest-point mechanical coverage, then drained round-robin under one global cap so input or lexical shape order cannot monopolize a small transfer budget. They are inserted before random restarts so they are retained when candidate lists are truncated or batch budgets are tight.

A `cluster` proposal scope is an explicit label and shape set. It does not itself discover clusters, select medoids, stage representative-first evaluation, or promote candidates to unmeasured members. Those actions require the production grid controller tracked in `docs/plan.md`.

Installed hipBLASLt discovery creates zero-evidence planning pairs. After those exact pairs run through the normal scheduler, their measured candidates can become elites, transfer seeds, GOMEA parents, and final winners like any other candidate.

## Cache-Aware Planning

The scheduler avoids repeating known work with resolved timing and correctness state:
- Valid positive benchmark samples have precedence over every reusable benchmark negative and determine whether more samples are needed.
- When no positive timing exists, the latest `rejected` or `build_failed` row is the reusable benchmark-negative state.
- Latest compatible validation state is `passed` or `failed` under the validation-protocol hash.
- A resolved benchmark negative or latest compatible validation failure skips the pair. A different validation identity requests fresh correctness verification.

`plan_batches()` first chunks candidates and shapes, then builds exact rectangular batches only for missing observations. Shapes that have the same missing candidate subset and same required sample count are grouped together. Stable compile caching uses singleton candidate libraries so cache identity is independent of proposal cohort. Exact shape identities remain in the key because GridBased generation is shape-dependent.

## Batch Execution

Each batch writes:
- `config.yaml`: TensileLite config with candidate `Groups`.
- `config.manifest.csv`: intended shape/candidate/solution mapping.
- `run/` or a unique run directory for build and structured-runner outputs.

The production path requires `--runner-bin` unless using `--dry-run` or `--generate-only`. `--prepare-workers` controls parallel build/map/diagnostic/validation work. `--validation-workers` caps concurrent validators, `--prepare-wave-batches` bounds each admitted wave, and `--cost-aware-scheduling` orders heavier predicted preparation batches first without changing timing priority. After each wave's preparation workers exit, its benchmarks run serially. A shared/exclusive APU gate at `EVOTENSILE_APU_LOCK_PATH` protects this invariant across cooperating processes and direct runner invocations.

Without stable compile caching, the default candidate batch size is chosen by a throughput heuristic that keeps enough candidate/shape batches to saturate available workers while respecting the profile's max candidate batch size. Stable caching forces singleton candidate libraries for composable reuse. The gfx1151 profile uses 32 parallel preparation workers, so singleton libraries retain broad compile parallelism while validation remains capped at one worker.

## Adaptive Sampling

Adaptive sampling is enabled by default. After parallel compilation and one-time validation, every pair receives one probe launch. Candidates confidently slower than the shape reference by more than the default `4*` factor stop there. Provisional survivors receive two additional launches to reach the `3*1` probe target. Survivors then receive the main `--num-benchmarks` budget and existing confidence-based top-ups. Probe, main, and adaptive rounds reuse the same prepared artifacts and never recompile or revalidate.

Probe timing has a distinct protocol identity, so it cannot enter production ranking, family archives, transfer, or learned linkage. `--fixed-sampling` disables both probe racing and adaptive top-ups.

Timing-noise math and validation-gated top-up rules are documented in `docs/noisy_measurements.md`. Campaign-level provisional-leader top-ups are separate and documented in `docs/search_screening_stabilization.md`.

Use `--fixed-sampling` for deterministic fixed-budget utility runs or debugging.

## Outlier Repair

`repair-outliers` is a second-stage search command that detects shapes whose current best GFLOP/s is below a robust nearest-neighbor envelope. It reruns only those shapes with seeds from:
- the outlier's current winner.
- nearest-shape winners and near-winners.
- the selected proposal mode.

Outlier detection and repair math are documented in `docs/search_outlier_repair.md`.

## Excluded From Current Search

The implemented search stack is cache-aware random/evolutionary/family-QD generation, optional adaptive operator allocation, optional ExtraTrees shortlisting, structured measurement, and adaptive sampling.

Not currently implemented:
- family-level measurement bandits.
- family-leader-specific fidelity stages beyond per-shape grid stabilization queues.
- automatic family descriptor splitting or merging.
- general CLI-managed multi-island search beyond the blind one-shape campaign driver.
- LFBO or a persistent cross-campaign surrogate.

TensileLite prediction mechanisms such as Formocast/Origami are not hard pruning gates for this target. The profile keeps `PredictionThreshold=2.0`.
