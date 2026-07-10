# Blind One-Shape Experiment Log

This document records the blind `gfx1151-nt-hhs` search experiments for shape `8192,8192,1,8192`. It is an experiment log, not the design specification for the search algorithms. Search design lives in the search-prefixed documents under `docs/`. Replay, time accounting, and blind-run tooling are documented in `docs/blind_experiment_infrastructure.md`.

## Objective And Success Criterion

The objective is to reproduce or exceed the external known-best hot-loop median within a real or simulated 20-minute wall-time budget.

The authoritative comparison target is:

```text
45.253 TFLOP/s hot-loop median
```

Final claims use the production confirmation protocol:

```text
20 warmups
10 benchmark samples
10 enqueues per sample
1 sync per sample
```

Two-sample screening and best-pass values are exploration and diagnostic evidence, not the primary success metric.

## Blindness Rules

All campaigns in this log follow these rules:
- no external winner hash or exact parameter bundle in proposals, defaults, repairs, tests, or parent pools.
- no performance-derived static linkage group chosen because it matches the external winner.
- no imported control candidate in a proof-eligible campaign DB.
- only queried validation-passed rows may train family ranking, linkage, operator credit, or the surrogate.
- historical control measurements may answer an exact simulated query but may not guide candidate generation.
- unknown oracle candidates remain unknown.
- the external result is checked only as a final threshold.

## Early Blind Baselines

### Broad Learned-Evolution Campaign

An earlier four-generation broad campaign took about `632s` and reached only `9.749 TFLOP/s` screening median. Its superseded raw artifacts were pruned after the final experiment set was retained.

It demonstrated that broad domains and GPU utilization alone were not enough. The search needed stronger structural preservation and linked variation.

### Family-QD V2

The first clean family-QD campaign produced the following results. Its superseded raw artifacts were pruned after the final experiment set was retained.

Results:
- `125` measured candidates.
- `14.958 TFLOP/s` screening leader.
- `21.876 TFLOP/s` best hot-loop median among the top eight.

The archive could lose a coarse family after one failed representative, and the initial family descriptor was still too sparse.

### Family-QD V3

The improved blind V3 campaign produced the following result. Its superseded raw artifacts were pruned after the final experiment set was retained.

Results:
- `144` candidates.
- `12` candidates in the `MT128x128/TLDS0` coarse structural family.
- `29.014 TFLOP/s` best hot-loop media.
- `29.212 TFLOP/s` best pass.

V3 reached `64.1%` of the external target. Post-hoc, its closest generated genome remained Hamming distance `13` from the external control.

The important diagnosis was search depth and epistasis: the campaign covered the correct broad macro family but did not assemble enough complementary high-order genes.

The original V3 wrapper timeout was recovered by resuming the exact already-registered generation-zero hashes. No replacement candidates were generated.

## Search Changes Evaluated

The next experiment phase implemented:
- truncation-aware linkage evidence sizing.
- up to four quality-bounded, Hamming-diverse elites per family cell.
- semantic-group mutation.
- separate GOMEA neighborhood and donor-mixing identities.
- mostly family-local GOMEA donors.
- UCB-style adaptive operator allocation from queried parent/child outcomes.
- generic candidate/shape features and ExtraTrees shortlisting.
- oversized proposal pools with performance, uncertainty, diversity, and random quotas.
- exact-hash historical replay and explicit 20-minute accounting.
- a wall-time-driven real campaign driver and artifact-reusing hot confirmation.

The production measurement pipeline remained the normal compile, validation, three-launch probe, screening, and DB-ranking path.

## Simulated Policy Selection

Proof-eligible replay used only candidate streams from prior blind campaigns. Unknown hashes were not imputed.

Four neutral policies were compared over three seeds. The most reliable policy used:
- `24` measured candidates per feedback round.
- a `256`-candidate visible replay window.
- surrogate activation at `24` positive evidence rows.

That policy recovered the best available blind finalist in all three replay seeds. The previous `32`-candidate, `128`-window policy missed it in two seeds.

Proof-eligible replay topped out at `27.5–29.0 TFLOP/s` hot within the simulated budget and did not meet the target.

A separate directed/control pool diagnostic met the threshold quickly, but it was recorded with `proof_eligible=false`. It showed that the selector could recognize a strong candidate if visible. It did not show that blind search could generate one.

Replay summaries were used to freeze the real-run policy. Their temporary raw artifacts were pruned after the final real campaigns completed.

## Frozen Real-Run Policy

The real campaign policy used:
- `48` family-stratified cold candidates.
- subsequent feedback rounds requesting `4` random candidates plus `20` adaptively allocated variation candidates.
- an `8x` oversized proposal pool.
- `32` global/family elites.
- surrogate activation at `24` evidence rows.
- candidate batches of `8` and `8` preparation workers.
- three-launch catastrophic probes.
- two main screening samples.
- final top-eight hot-loop confirmation from existing artifacts.

The first run exposed a fixed round-cap bug. Later runs used wall-time-driven admission based on recent round duration. The final run used a `60s` confirmation reserve after measured confirmation proved substantially cheaper than the original conservative reserve.

## Real Campaign Results

| Seed | Rounds | Registered candidates | Elapsed / active time | Best hot median | Best hot pass | Target fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260710` | 21 | 496 | about `787s` active search plus confirmation | `43.403 TFLOP/s` | `44.400 TFLOP/s` | `95.91%` |
| `20260711` | 31 | 700 | `997.8s` | `44.331 TFLOP/s` | `45.106 TFLOP/s` | `97.96%` |
| `20260712` | 31 | 709 | `1118.2s` | `41.014 TFLOP/s` | `41.649 TFLOP/s` | `90.63%` |

No seed met the `45.253 TFLOP/s` hot-median target.

The best confirmed blind candidate was `cand_00b7457713836cfc` at `44.331 TFLOP/s` median. Its two-sample screening estimate was `45.762 TFLOP/s`, showing that screening noise can materially overstate a provisional leader.

Relative to V3's `29.014 TFLOP/s`, the best new median improved by about `52.8%`.

## Operator Evidence

Across compatible measured parent/child comparisons from the three real campaigns:
- semantic mutation improved on its best measured parent `35.5%` of the time.
- GOMEA neighborhood trials improved `29.9%` of the time.
- GOMEA mixing improved `28.4%` of the time.
- categorical DE improved `14.1%` of the time.

Semantic mutation and GOMEA supplied all but six aggregate top-20 slots across the campaigns. DE remained useful as minimum exploration but was the weakest measured arm.

These are noisy observational rates, not controlled operator ablations.

## Utilization Evidence

Monitoring for seeds `20260711` and `20260712` observed:
- GPU busy medians of `88%` and `93%`.
- peak package power of `137W` and `138W`.
- peak edge temperature of `82C` and `83C`.
- up to six concurrent Tensile compilers.
- up to six concurrent validation runners.
- at most one benchmark-mode runner.

The practical miss was not caused by serialized compilation or concurrent benchmark timing. Preparation remained parallel and benchmark timing remained serial.

## Interpretation

The experiment did not reproduce the external target, but it substantially narrowed the gap while remaining blind and within the wall-time budget.

The strongest conclusions are:
- linkage learning must retain enough pre-truncation evidence to remain active.
- preserving several diverse family elites helps maintain complementary building blocks.
- small semantic mutation and GOMEA are more productive than broad mutation or DE for local assembly.
- oversized surrogate pools are important for considering many structures without compiling all of them.
- short screening is adequate for exploration but too noisy for final claims or high-impact credit decisions.
- seed variance remains large enough that one global population is fragile.

## Next Experiments

The next generic experiment should keep the search blind and test:
- noise-aware top-ups for provisional archive and global leaders before they strongly influence surrogate training or operator credit.
- two independent cold-start islands with later family-local migration.
- cost-aware operator reward that includes proposal, build, validation, and timing cost.
- fixed-policy multi-seed evaluation before threshold comparison.

These changes should be evaluated by equal wall time and must not import external winner parameters.

## Artifacts

Primary real campaigns:
- `out/blind_one_shape_20min_adaptive_20260710_seed20260710/`
- `out/blind_one_shape_20min_adaptive_20260710_seed20260711/`
- `out/blind_one_shape_20min_adaptive_20260710_seed20260712/`

Best campaign:
- `out/blind_one_shape_20min_adaptive_20260710_seed20260711/campaign_summary.json`
- `out/blind_one_shape_20min_adaptive_20260710_seed20260711/hot_loop_top8/summary.json`

Aggregate operator and utilization analysis:
- `out/blind_one_shape_20min_adaptive_analysis.jsonl`
