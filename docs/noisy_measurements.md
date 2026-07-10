# Noisy Measurement Design

This document describes how EvoTensile handles noisy speed measurements for one `(shape, candidate)` pair and how adaptive finalist top-ups decide when a shape is resolved.

## Measurement Model

Each validation-passed candidate for one shape is treated as a noisy timing arm. Timing samples are positive kernel times in microseconds from the structured runner.

EvoTensile analyzes timing in log-time space because GPU timing noise and candidate gaps are usually multiplicative. A `2%` slowdown is represented consistently regardless of absolute problem size.

For candidate `c` with positive timing samples:

$$
t_{c,1}, \dots, t_{c,n}
$$

EvoTensile computes:

$$
y_{c,i} = \log(t_{c,i}), \qquad s_c = \mathrm{median}(y_{c,*}).
$$

The score `s_c` is median log time, so lower is better. Median time in microseconds is `exp(s_c)`.

## Robust Scale Estimate

The robust log-noise estimate is:

$$
\sigma_c = \max\left(\mathrm{stdev}(y_c),\ 1.4826\,\mathrm{MAD}(y_c),\ \frac{\mathrm{IQR}(y_c)}{1.349}\right).
$$

The constants are normal-consistency factors:
- `1.4826` converts median absolute deviation to a standard-deviation estimate under normal noise because `1 / Phi^-1(0.75) ≈ 1.4826`.
- `1.349` converts IQR to a standard-deviation estimate because `Phi^-1(0.75) - Phi^-1(0.25) ≈ 1.349`.

The approximate standard error of the median log time is:

$$
\mathrm{SE}_c = 1.2533141373155001 \frac{\sigma_c}{\sqrt{n}}.
$$

The coefficient is `sqrt(pi / 2)`, the asymptotic ratio between the standard error of a sample median and a sample mean for normally distributed noise.

`timing_stats_from_times()` also records mean log time, standard deviation, MAD, IQR, `p10`, `p90`, and high-side outlier count for audit output.

## Three-Launch Probe

Adaptive schedules first run a low-fidelity probe for every validation-passed pair:

```text
NumWarmups=0
NumBenchmarks=3
EnqueuesPerSync=1
SyncsPerBenchmark=1
```

The default probe therefore spends exactly three timed launches per candidate. Probe timing has a distinct `BenchmarkRole=probe` protocol identity and is never pooled with main timing.

Let `r` be the fastest compatible reference: the current probe leader or a faster existing main-protocol incumbent for the same shape. A probe candidate `c` is screened out only when the lower confidence bound on its log-time gap exceeds the coarse slowdown threshold:

$$
\mathrm{CI}_{c,\mathrm{low}} > \log(F)
$$

The default factor is `F=4`. The probe also applies a `5%` minimum log-noise floor because two or three observations cannot provide a reliable zero-noise estimate. At least eight probe candidates per shape survive regardless of the threshold, or all candidates when fewer than eight are available.

Probe screening is a timing-allocation decision, not invalidity or reusable negative cache evidence. Missing/incomplete probe evidence fails open: the pair remains eligible for main timing.

The probe controls are:

```text
--adaptive-probe-samples 3
--adaptive-probe-max-slowdown-factor 4.0
--adaptive-probe-confidence 0.90
--adaptive-probe-noise-floor-pct 5.0
--adaptive-probe-min-survivors 8
```

## Pairwise Plausibility

After probe survivors receive main-protocol samples, let `b` be the current best candidate by lowest median log time. For another candidate `c`, the log-time gap is:

$$
g_c = s_c - s_b.
$$

The approximate confidence interval for the gap is:

$$
\mathrm{CI}_c = g_c \pm z_\alpha \sqrt{\mathrm{SE}_c^2 + \mathrm{SE}_b^2}.
$$

`z_alpha` is derived from `--adaptive-confidence`. the CLI default is `0.90`.

A contender remains plausible if the lower bound of its gap confidence interval is within the indifference zone:

$$
\mathrm{CI}_{c,\mathrm{low}} \le \log(1 + \epsilon).
$$

`epsilon` is `--adaptive-epsilon-pct` as a fraction. The default is `2%`. This means a candidate is still retimed if current evidence cannot confidently show it is slower than the best by more than the requested tolerance.

The percent form of a log gap is:

$$
\mathrm{gap\_pct} = (\exp(g_c) - 1) \cdot 100.
$$

## Resolution States

`decide_shape_retime()` returns one of these statuses:
- `no_valid_candidates`: no validation-passed timing stats are available for the shape.
- `resolved_winner`: one candidate is available, or all non-best candidates are confidently outside the indifference zone.
- `resolved_equivalent`: all plausible candidates are mutually within the practical equivalence zone.
- `needs_retime`: multiple plausible contenders remain and the scheduler requests more samples.

A shape is practically equivalent only when every pair among candidates inside the `epsilon` zone has a confidence interval fully contained within `±epsilon`.

## Target Sample Count

When a shape remains unresolved, EvoTensile estimates the target sample count for the active plausible contenders. For contender `c`:

$$
d_c = \max\left(\left| |s_c - s_b| - \epsilon_{\log} \right|,\ \delta_{\min}\right),
$$

$$
n_c = \left\lceil\left(\frac{z_\alpha \cdot 1.2533141373155001 \sqrt{\sigma_b^2 + \sigma_c^2}}{d_c}\right)^2\right\rceil.
$$

`delta_min` is `--adaptive-min-effect-pct` converted to log space. the default is `0.5%`. It prevents the sample estimate from exploding when two candidates are nearly tied or exactly on the indifference boundary.

The scheduled target is:
- at least `--adaptive-min-samples` (`20` by default).
- at most `--adaptive-max-samples` (`80` by default).
- rounded up to `--adaptive-sample-step` (`10` by default).
- applied to at most `--adaptive-max-k` plausible candidates per shape (`8` by default).

## Adaptive Execution

`schedule-batches` uses probe racing and adaptive sampling unless `--fixed-sampling` is passed.

The execution sequence is:
- Prepare all batches in parallel: compile, map/salvage, diagnose, and validate.
- Join every prepare worker before timing starts.
- Run the serial three-launch probe queue for every validation-passed pair.
- Compare complete probe evidence by shape and screen only confidently catastrophic candidates.
- Run the main timing protocol only for probe survivors. `--num-benchmarks` is the initial main sample budget.
- Load main-protocol timing stats and decide retime groups for unresolved shapes.
- Run only missing main-protocol samples for plausible contenders from the original prepared-artifact index.
- Repeat for up to `--adaptive-max-rounds`. Default `4`.

Probe, main, and adaptive rounds never recompile or revalidate. A pair without a successful prepared artifact is ineligible for timing. Main top-ups use the same main benchmark-protocol identity because `NumBenchmarks` is an execution budget, not a timing-compatibility field.

`--fixed-sampling` bypasses both probe racing and adaptive top-ups and gives every validation-passed pair the requested main timing budget.

## Validation Gate

Correctness is handled separately from repetition count:
- Validation mode runs once per compatible `(shape, candidate, validation protocol)` and stores evidence in the `validations` table.
- Benchmark mode always sets `NumElementsToValidate=0` and cannot perform correctness work.
- Timing is admitted only for pairs in the prepared validation-passed set or compatible cached validation evidence.
- Validation failures become reusable negative evidence and are never eligible for adaptive timing.

This keeps adaptive retiming fast without recompiling or revalidating contenders and prevents unvalidated candidates from becoming winners.

## Ranking Semantics

Probe and main timing have different benchmark-protocol hashes. Production ranking, family archives, transfer, and learned linkage request the main protocol hash, so probe rows cannot become winners or training evidence accidentally.

`rank_evaluations()` ranks only `status='ok'` rows for the requested protocol and summarizes per `(shape_id, candidate_hash)`:
- `samples`: positive timing sample count.
- `median_time_us` and `best_time_us`.
- `median_gflops` and `best_gflops`, computed from shape FLOPs.

Search and update tools use median statistics for winner selection. Best-sample statistics are retained for inspection, not as the primary selection criterion.

## Screening Versus Final Confirmation

Short main-protocol screening is intended for search feedback, not final performance claims. Family archives, learned linkage, operator credit, and the surrogate can consume compatible positive screening rows, so screening must remain validation-gated and protocol-specific.

A small screening sample can still overstate a provisional leader. Blind one-shape campaigns observed a `45.762 TFLOP/s` two-sample screening leader that confirmed at `44.331 TFLOP/s` under the hot-loop protocol.

Finalists should therefore be rerun with the production confirmation protocol:

```text
NumWarmups=20
NumBenchmarks=10
EnqueuesPerSync=10
SyncsPerBenchmark=1
```

`hot_confirm_topk()` reuses generated libraries and prior compatible validation evidence. It does not recompile or revalidate finalists. Confirmation rows are reported separately from broad screening so expensive launches are reserved for a small candidate set.

The current search does not automatically top up provisional archive leaders before they influence operator credit or surrogate training. Noise-aware leader top-ups are a planned improvement. Blind experiment procedures and time-budget accounting are documented in `docs/blind_experiment_infrastructure.md`.
