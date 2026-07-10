# Search Algorithms Design

This document describes EvoTensile's general search loop and proposal sources. Candidate-space construction, GOMEA internals, learned linkage, noisy measurement handling, TensileLite communication, outlier repair, and database semantics have separate design docs.

## Search Unit

The search unit is a `(shape, candidate)` pair:
- A `Shape` has exact `M`, `N`, `batch`, and `K` fields plus a stable `shape_id` such as `m512_n128_b1_k256`.
- A `Candidate` is a complete TensileLite solution dictionary with a stable `cand_...` hash.
- A valid observation is a timed, validation-passed DB row for that pair under the active problem and benchmark protocol hashes.

Search algorithms only propose candidates. They do not select final winners directly. Winner selection comes from DB ranking and adaptive retiming.

## Scheduler Flow

`schedule-batches` is the main search entry point:
- Resolve the target profile, shapes, benchmark protocol, and SQLite DB.
- Propose candidates with the selected proposal mode.
- Register candidates and shapes in the DB.
- Record immediate shape-dependent rule rejections.
- Plan missing `(shape, candidate)` observations from reusable cache status.
- Emit TensileLite YAML and manifest files for exact rectangular batches.
- Run all build/map/diagnostic/correctness work in a parallel prepare queue.
- Join every prepare worker, then run benchmark-only work in one serial queue.
- When adaptive sampling is enabled, top up plausible finalists from the original prepared artifacts only.
- Write `schedule_metadata.json` with proposal, protocol, linkage, batching, and execution details.

Timing begins only after the complete prepare queue drains. This hard barrier prevents benchmark overlap with compilation or correctness verification on integrated CPU/GPU systems.

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

`family-qd` is the first family-aware quality-diversity proposal mode. The broader staged-screening roadmap is documented in `docs/family_aware_ea_screening.md`.

### Random

Random proposal samples from the target `DOMAINS`, applies linked repairs, and rejects candidates with known invalid rules. When target shapes are known, random generation also enforces shape-dependent cheap constraints before returning a candidate.

The current NT HHS generator samples the compatible TLDS0 and TLDS2 construction branches with equal probability and applies a proposal-side VALU register headroom check. Branch balance and headroom are proposal policies, not validity rules.

### Local Mutation

Local mutation starts from DB elites and independently resamples each domain gene with probability `--mutation-rate` (`0.25` by default). Each child is checked through `make_candidate()`, and invalid children are discarded.

This mode is simple and useful around measured elites, but it does not preserve multi-field couplings as strongly as GOMEA.

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

GOMEA mechanics are documented in `docs/gomea.md`. Learned linkage is documented in `docs/linkage_learning.md`.

### Evolutionary

`evolutionary` combines random candidates, local mutations, categorical DE, and GOMEA into one proposal set. All candidates are deduplicated by hash before scheduling.

### Family QD

`family-qd` adds a family-aware quality-diversity layer:
- It computes coarse profile-specific family descriptors from candidate structure.
- It loads a DB-derived family archive whose cells keep validation-passed leaders by shape-local rank percentile.
- It preserves every positive family leader as a proposal parent before random restarts.
- It uses descriptor-stratified random generation with repeated family occupancy and one retry for negative-only cells.
- It balances shape-aspect exploitation with broad-cell exploration.
- It applies local mutation, categorical DE, and GOMEA around family/global elites when evidence exists.

Family descriptors and archive entries are proposal metadata only. They do not shrink domains, change cache identity, or create validity rules.

## Elite And Transfer Sources

Proposal modes that need parents load validation-passed DB elites through `rank_evaluations()`.

For multi-shape schedules, the scheduler can also seed from nearest previously tuned shapes:
- `--transfer-shapes` controls how many nearest source shapes are considered.
- `--transfer-per-shape` controls how many top candidates are copied from each source shape.
- Shape distance uses Euclidean distance in `log2(M)`, `log2(N)`, `log2(K)`, `log2(M/N)`, `log2(K/M)`, and `log2(K/N)`.

Transfer candidates are inserted before random restarts so they are retained when candidate lists are truncated or batch budgets are tight.

Imported hipBLASLt baselines are normal DB candidates. Once imported, they can become elites, transfer seeds, GOMEA parents, and final winners like any other candidate.

## Cache-Aware Planning

The scheduler avoids repeating known work with reusable cache status:
- Positive status: `ok`.
- Reusable negative statuses: `rejected`, `validation_fail`, and `build_failed`.
- Positive sample counts determine whether more timing samples are needed.
- Negative reusable rows skip the pair entirely for the same problem/protocol/shape/candidate key.

`plan_batches()` first chunks candidates and shapes, then builds exact rectangular batches only for missing observations. Shapes that have the same missing candidate subset and same required sample count are grouped together.

## Batch Execution

Each batch writes:
- `config.yaml`: TensileLite config with candidate `Groups`.
- `config.manifest.csv`: intended shape/candidate/solution mapping.
- `run/` or a unique run directory for build and structured-runner outputs.

The production path requires `--runner-bin` unless using `--dry-run` or `--generate-only`. `--prepare-workers` controls parallel build/map/diagnostic/validation work. After all prepare workers exit, benchmarks run serially. A shared/exclusive APU gate at `EVOTENSILE_APU_LOCK_PATH` protects this invariant across cooperating processes and direct runner invocations.

The default candidate batch size is chosen by a throughput heuristic that keeps enough candidate/shape batches to saturate available workers while respecting the profile's max candidate batch size. Use `--candidate-batch-size 1` for debugging or singleton failure attribution.

## Adaptive Sampling

Adaptive sampling is enabled by default. After parallel compilation and one-time validation, every pair receives a separate `3×1` probe. Candidates confidently slower than the shape reference by more than the default `4×` factor stop there. Survivors receive the main `--num-benchmarks` budget and existing confidence-based top-ups. Probe, main, and adaptive rounds reuse the same prepared artifacts and never recompile or revalidate.

Probe timing has a distinct protocol identity, so it cannot enter production ranking, family archives, transfer, or learned linkage. `--fixed-sampling` disables both probe racing and adaptive top-ups.

Timing-noise math and validation-gated top-up rules are documented in `docs/noisy_measurements.md`.

Use `--fixed-sampling` for deterministic fixed-budget utility runs or debugging.

## Outlier Repair

`repair-outliers` is a second-stage search command that detects shapes whose current best GFLOP/s is below a robust nearest-neighbor envelope. It reruns only those shapes with seeds from:
- the outlier's current winner.
- nearest-shape winners and near-winners.
- the selected proposal mode.

Outlier detection and repair math are documented in `docs/outlier_repair.md`.

## Excluded From Current Search

The implemented search stack is cache-aware random/evolutionary/family-QD proposal generation plus structured measurement and adaptive sampling. Family-leader-specific retiming, explicit bandit allocation, and surrogate/LFBO-style proposal generation are outside the current implementation.

TensileLite prediction mechanisms such as Formocast/Origami are not hard pruning gates for this target. The profile keeps `PredictionThreshold=2.0`.
