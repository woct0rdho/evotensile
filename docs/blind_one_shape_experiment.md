# Blind One-Shape Experiment Log

This document records the blind `gfx1151-nt-hhs` search experiments for shape `8192,8192,1,8192`. It is an experiment log, not a subsystem design specification. Reusable search design lives in focused documents under `docs/`. Replay and blindness rules are in `docs/blind_experiment_infrastructure.md`, and the one-shape state machine is in `docs/blind_campaign_control.md`.

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

Proof-eligible replay topped out at `27.5-29.0 TFLOP/s` hot within the simulated budget and did not meet the target.

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

## Follow-up Campaign Series

The `2026-07-10` follow-up kept the validation, serialized-timing, hot-confirmation, and anti-hindsight rules while testing whether broader cold coverage, deeper local assembly, more reliable feedback, and explicit campaign control reduce seed variance and close the remaining performance gap.

### Policy Under Test

The combined policy enabled:
- screening-leader stabilization from `docs/search_screening_stabilization.md`.
- mechanical covering cold starts from `docs/search_mechanical_coverage.md`.
- bounded GOMEA neighborhoods and adaptive donor selection from `docs/search_gomea.md`.
- semantic-group, donor-mode, and cost-aware credit from `docs/search_operator_portfolio.md` and `docs/search_cost_model.md`.
- two isolated cold islands, later migration, checkpointing, robust round admission, and optional convergence detection from `docs/blind_campaign_control.md`.
- the staged one-launch catastrophic probe from `docs/noisy_measurements.md`.

These mechanisms were evaluated as one combined campaign policy. The run was not a component-by-component causal ablation.

### Replay Preflight

Implementation and simulation preflight completed on `2026-07-10`. Before the real campaign series, focused correctness coverage and the complete repository test suite passed.

Exact-hash replay was rerun with staged-probe accounting over seeds `20260710-20260712` and equal `1200s` budgets. The matched baseline and new policy recovered the same historical `44.331 TFLOP/s` hot finalist. Median simulated time to first exceed `40 TFLOP/s` screening improved from `168.47s` to `114.05s`, a `32.3%` reduction. The staged probe allowed a median of `984` queried candidates instead of the former `960`, although new-policy seed `20260711` regressed to `172.71s`. Replay therefore supports the timing-allocation change but not automatic early stopping or a claim of better unseen solution quality.

### Attempt 1: Concurrent Validation Failure

The historical, subsequently pruned attempt `out/blind_one_shape_next_20260710_seed20260713/` was stopped during cold validation after six concurrent validators destabilized ROCr/KFD module loading. Before timing, `48` candidates were registered, `11` validation pairs passed, `3` failed, and `2` were rejected. Two validation batches completed in about `17s`. Four concurrent batches remained active beyond `220s`.

After termination, even a singleton from a previously successful library could not validate within `30s`. A reboot restored that exact validation to `0.70s`, confirming transient loader state rather than candidate invalidity. The next policy revision retained eight preparation workers but capped validation workers at one.

### Attempt 2: Catastrophic Probe Tail

The historical, subsequently pruned attempt `out/blind_one_shape_next_v2_20260710_seed20260713/` confirmed the validation fix: all six serialized validation batches completed in about `2-28s`, at most one validator and one benchmark runner were active, and the loader failure did not recur.

This attempt was stopped after about `135s` before round zero completed. The exact-MI cold selector had chosen two `16x16` workgroups requiring about `8.7-9.0s` per launch. Their six launches consumed about `53s` of one `55.0s` probe batch. The next policy revision removed exact MI identities from coverage, added the soft dispatch-efficiency prior, and staged the probe so the slow tail received one launch before screening.

For the same seed, the revised cold pool increased mechanical-token coverage from `226` to `236`, retained `42` rather than `44` distinct MatrixInstruction values, removed all five single-instruction workgroups, reduced median WGP rounds from `740` to `276`, and raised median macro-tile area from `4608` to `12288`.

### Attempt 3: Completed Corrected Campaign

The corrected real campaign at `out/blind_one_shape_next_v3_20260710_seed20260713/` completed normally:
- round zero finished in `59.8s`, covering `42` MatrixInstruction values, `31` family cells, and `236` mechanical tokens.
- `11` catastrophic candidates were screened after the initial probe launch, while `22` received complete probe evidence.
- migration produced late gains from `29.42 TFLOP/s` in round 6 to `32.86` in round 7, `40.61` in round 16, `42.87` in round 24, and `43.39` in round 29.
- search stopped through normal round admission after `44` rounds and `1116.39s`, with `1015` unique candidates and `1143.48s` total active time including hot confirmation.
- screening leader `cand_3fa1d1f87910a88c` confirmed at `42.168 TFLOP/s` hot median and `42.465 TFLOP/s` best, a `2.81%` screening-to-hot decline.

### Final Comparison

The corrected policy improved throughput but not best confirmed quality. It measured about `0.91` candidates per search second, `29.6%` more than prior best seed `20260711`, and registered `1015` candidates versus `496`, `700`, and `709` previously. Its hot median beat prior seed `20260712` but trailed seeds `20260710` and `20260711`. It was `4.88%` below the retained best `44.331 TFLOP/s` and `6.82%` below the external `45.253 TFLOP/s` target.

Operator evidence was mixed rather than supporting a single replacement strategy. DE produced the final screening/hot leader despite only `17/105` comparable child successes. Semantic mutation and GOMEA produced `18/20` top screening candidates. Semantic mutation improved in `69/213` comparable trials, GOMEA neighborhoods in `69/197`, and GOMEA mixing in `36/149`. Diverse and random GOMEA donors had higher posterior success rates than quality donors in this seed, but all donor modes contributed top candidates.

Monitoring recorded median GPU busy `61%`, `90th` percentile `93%`, peak power `126W`, and peak edge temperature `77C`. Up to six compilers overlapped, while validation and benchmark concurrency both remained exactly one. No loader, validation-concurrency, or timing-overlap failure recurred.

### Early-Stop Observation

Automatic convergence stopping remained disabled. The corrected run improved after several earlier plateaus and did not reach its final leader until round 29. Prior best seed `20260711` improved after an eight-round gap. The final 14-round plateau could have saved about `333s` only with hindsight, while mean Hamming diversity remained high at `16.79`, so the implemented low-improvement plus low-diversity detector correctly did not trigger. More equal-time seeds or an explicit ablation are required before broadening the stop condition.

No ablation, prior, seed, linkage priority, or neighborhood order uses the external winner parameters or performance-derived hindsight.

## Artifacts

### Retained Current Artifacts

Completed follow-up campaign and analysis:
- `out/blind_one_shape_next_v3_20260710_seed20260713/campaign_summary.json`
- `out/blind_one_shape_next_v3_20260710_seed20260713/hot_loop_top8/summary.json`
- `out/blind_one_shape_next_v3_analysis.json`

Current replay evidence:
- `out/blind_one_shape_next_replay/baseline.json`
- `out/blind_one_shape_next_replay/new_policy.json`
- `out/blind_one_shape_next_replay/comparison.json`
- `out/blind_one_shape_next_replay/baseline_staged_probe.json`
- `out/blind_one_shape_next_replay/new_policy_staged_probe.json`
- `out/blind_one_shape_next_replay/comparison_staged_probe.json`

The completed campaign directory also retains its campaign database, checkpoints, round proposal records, logs, compile cache, candidate artifacts, and hot-confirmation outputs. The top-level control, campaign, and monitor logs remain beside it in `out/`.

### Historical Pruned Artifacts

The chronological results above remain authoritative experiment records, but their raw artifact paths are intentionally unavailable after output consolidation:
- prior real campaigns `out/blind_one_shape_20min_adaptive_20260710_seed20260710/`, `out/blind_one_shape_20min_adaptive_20260710_seed20260711/`, and `out/blind_one_shape_20min_adaptive_20260710_seed20260712/`.
- aggregate prior analysis `out/blind_one_shape_20min_adaptive_analysis.jsonl`.
- failed-attempt directories `out/blind_one_shape_next_20260710_seed20260713/` and `out/blind_one_shape_next_v2_20260710_seed20260713/`, including their former `aborted_summary.json` files.

These paths are historical identifiers, not retained-artifact links. Their reported measurements are preserved in this document. No current workflow should depend on the pruned files.
