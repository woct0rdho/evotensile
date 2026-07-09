# Outlier Repair Design

This document describes `repair-outliers`, the second-stage search command that spends extra budget on shapes whose current best measured candidate is locally underperforming.

## Purpose

A broad search pass can leave isolated weak shapes because of timing noise, missing transfer seeds, shape-specific validity, or discontinuities in solution selection. `repair-outliers` identifies these shapes from DB evidence and reruns only them with neighbor-heavy seeds.

Outlier repair is a search-budget heuristic. It does not prove a current winner is wrong, and it does not modify candidate validity rules.

## Input Evidence

`detect_underperforming_shapes()` uses DB-ranked winners for the active `problem_type_hash` and `benchmark_protocol_hash`.

For each shape `s`, the current performance is the median GFLOP/s of its ranked winner:

$$
P_s = \mathrm{median\ GFLOP/s}(s), \qquad p_s = \log(P_s).
$$

Only shapes with at least `--outlier-min-samples` timing samples are eligible. The CLI default is `10`, so repair uses confirmed/adaptive evidence rather than early screening noise.

## Shape Feature Space

Each shape is embedded in log/ratio feature space:

$$
x_s = [\log_2 M,\ \log_2 N,\ \log_2 K,\ \log_2(M/N),\ \log_2(K/M),\ \log_2(K/N)].
$$

Distance to another tuned shape `u` is Euclidean distance:

$$
d(s,u) = \|x_s - x_u\|_2.
$$

The same feature keys are implemented by `Shape.features()` and `_shape_distance()`.

## Neighbor Selection

For each target shape, EvoTensile takes the nearest `K` other shapes with ranked winners, where:

```text
K = --neighbor-count
```

The default is `8`.

Neighbor weights are:

$$
w_u = \frac{1}{\max(d(s,u), 0.125)}.
$$

The `0.125` floor prevents nearly duplicate shapes from receiving infinite weight. In log2 feature units, `0.125` is one eighth of an octave, so very close shapes are still strongly favored.

## Local Linear Prediction

When at least three neighbors are available, EvoTensile fits a weighted local linear model around the target shape:

$$
p_u \approx \beta_0 + \beta^T(x_u - x_s).
$$

The fitted coefficients minimize:

$$
\sum_{u \in N_K(s)} w_u \left(p_u - \beta_0 - \beta^T(x_u - x_s)\right)^2 + \lambda\|\beta\|_2^2,
$$

with:

$$
\lambda = 10^{-3} \sum_{u \in N_K(s)} w_u.
$$

The ridge term is a small numerical stabilizer for slope terms only. The intercept is not penalized.

The local-linear prediction at the target is the intercept, clipped to the observed neighbor log-performance range:

$$
\hat{p}_{s,\mathrm{lin}} = \mathrm{clip}\left(\beta_0,\ \min_{u \in N_K(s)} p_u,\ \max_{u \in N_K(s)} p_u\right).
$$

If the local system is singular or too few neighbors are available, EvoTensile skips this prediction and falls back to the envelope prediction.

## Upper-Neighborhood Envelope

EvoTensile also computes a weighted upper-neighborhood envelope:

$$
\hat{p}_{s,\mathrm{env}} = Q_q(\{p_u\}_{u \in N_K(s)},\ \{w_u\}_{u \in N_K(s)}),
$$

where:

```text
q = --envelope-quantile
```

The default is `0.75`, an upper-quartile envelope. It is more optimistic than the median but less sensitive than the maximum to one unusually fast or noisy neighbor.

## Final Prediction And Residual

The final predicted local envelope is conservative:

$$
\hat{p}_s = \min(\hat{p}_{s,\mathrm{lin}},\ \hat{p}_{s,\mathrm{env}}).
$$

If the linear prediction is unavailable, EvoTensile uses the envelope prediction alone.

The repair residual is:

$$
r_s = \hat{p}_s - p_s.
$$

A shape is selected for repair when:

$$
r_s > \log(1 + \tau),
$$

where:

```text
tau = --outlier-threshold-pct
```

The default threshold is `10%`. Selected shapes are sorted by residual descending, and `--max-outliers` can cap the set for staged repair runs.

## Repair Seeds

`repair_seed_candidates()` builds a seed set for selected outliers from:
- the outlier shape's current winner.
- each nearest neighbor's winner.
- each nearest neighbor's top candidates, controlled by `--neighbor-per-shape`.

Seed candidates are reinserted with source `repair-transfer` and parent hashes pointing to the original candidate hash.

The command then adds normal proposal-mode candidates for the repair shapes. This allows repair runs to combine neighbor transfer, random exploration, local mutation, DE, and GOMEA depending on CLI options.

## Execution Flow

`repair-outliers` uses the same scheduler and measurement pipeline as `schedule-batches`:
- Resolve profile, protocol, DB, and eligible shapes.
- Detect outliers from current DB-ranked winners.
- Build repair seed candidates.
- Propose additional candidates for only the repair shapes.
- Dedupe and optionally truncate candidates with `--max-candidates`.
- Execute cache-aware structured batches.
- Write `repair_metadata.json` with outlier diagnostics, candidate hashes, execution batches, status counts, and linkage/adaptive metadata.

If no outliers or no candidates are available, the command writes metadata and performs a dry execution plan.

## Interpretation

A selected outlier means only that the current best median GFLOP/s is below a local neighbor prediction by more than the threshold. Real GEMM behavior can legitimately have cliffs from:
- tile divisibility and edge handling.
- LDS footprint and bank behavior.
- occupancy or VGPR pressure.
- GSU workspace decisions.
- solution-selection discontinuities.
- shape-specific invalidity or validation failures.

Repair therefore allocates more measurement/search budget. It does not impose smoothness on final winner selection.
