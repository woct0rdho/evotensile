# EvoTensile

Work in progress. README is AI-generated.

EvoTensile is an external smart-search autotuner for TensileLite / hipBLASLt. It proposes complete TensileLite candidate bundles, emits them as TensileLite `Groups`, runs TensileLite as the evaluator, and stores reproducible timing/cache metadata for iterative search.

The current bundled target configuration is gfx1151 FP16 NT HHS GridBased GEMM tuning, but the core pieces are intended to stay generic: candidate hashing, shape handling, YAML emission, runner orchestration, protocol hashing, and result caching.

See `PLAN.md` for the current target-specific tuning plan and remaining work.

## Implemented Capabilities

- Candidate and shape primitives with stable canonical hashes.
- Exact-shape helpers and the current 100-shape pilot grid generator.
- Candidate generation from deterministic seeds, random valid configs, local mutations, categorical DE, and GOMEA-style linkage neighborhoods.
- TensileLite YAML generation using one `ForkParameters: Groups` list of complete candidate dictionaries.
- Hot-loop benchmark defaults for steady-state inference/training throughput.
- TensileLite subprocess runner used by the cache-aware scheduler.
- SQLite schema for candidates, shapes, runs, and evaluations.
- Manual timing-cache namespace via `--version-name`, plus problem-type and benchmark-protocol hashes.
- Cache inspection helpers for identity, status summaries, and missing candidate/shape evaluations.
- Cache-aware batch scheduler with dry-run, generate-only, compile-then-serial-benchmark, and immediate ingestion modes.
- Tolerant validation-aware CSV/log parser and `parse-csv` command for inspecting TensileLite result files.
- Candidate/shape manifest sidecars and `ingest-csv` for validation-gated SQLite evaluation records.
- Final-solution YAML mapping from TensileLite `SolutionIndex`/kernel names back to candidate hashes, including rejected/deduplicated candidate handling.
- `rank-evals` aggregation that ranks only `status=ok` validation-passed observations.

## Benchmark Protocol

The default generated YAML uses hot-loop / steady-state timing:

```yaml
NumWarmups: 10
NumBenchmarks: 10
EnqueuesPerSync: 10
SyncsPerBenchmark: 1
SleepPercent: 0
HardwareMonitor: False
```

Cold-loop behavior is intentionally not tracked during tuning because it increases tuning time and optimizes for first-run or bursty-idle effects rather than sustained throughput. Analyze cold-loop behavior later only if first-request latency becomes important.

## Quick Start

From the repo root:

```bash
python3 -m evotensile.cli summarize-space --num-random 128
```

Plan cache-aware batches without running TensileLite. By default, `schedule-batches` uses the current 100-shape first-pass proposal: `seed-random-gomea` with `64` random candidates and `64` GOMEA candidates.

```bash
python3 -m evotensile.cli schedule-batches \
  --db out/evotensile.sqlite \
  --output-dir out/scheduled \
  --version-name gfx1151_hotloop_v0 \
  --limit-shapes 100 \
  --candidate-batch-size 32 \
  --shape-batch-size 100 \
  --dry-run
```

Run planned batches with compile-only first, serial benchmarking, and immediate ingestion:

```bash
python3 -m evotensile.cli schedule-batches \
  --db out/evotensile.sqlite \
  --output-dir out/scheduled \
  --version-name gfx1151_hotloop_v0 \
  --limit-shapes 100 \
  --candidate-batch-size 32 \
  --shape-batch-size 100 \
  --compile-threads -1 \
  --benchmark-threads 1
```

Refine candidates with cached elites and GOMEA-style linkage neighborhoods:

```bash
python3 -m evotensile.cli schedule-batches \
  --db out/evotensile.sqlite \
  --output-dir out/local_refine \
  --version-name gfx1151_hotloop_v0 \
  --proposal seed-random-gomea \
  --num-random 16 \
  --elite-count 8 \
  --gomea-count 64 \
  --limit-shapes 100
```

Ingest validation-gated TensileLite CSV/log rows manually, then rank only passing observations. When given a run directory, `ingest-csv` auto-detects TensileLite `*_Final.yaml` / `*_CSVWinner.yaml` files and uses them as the source of truth for candidate mapping.

```bash
python3 -m evotensile.cli ingest-csv out/tensilelite_run_000 \
  --db out/evotensile.sqlite \
  --manifest out/pilot.manifest.csv \
  --version-name gfx1151_hotloop_v0 \
  --include-logs

python3 -m evotensile.cli rank-evals \
  --db out/evotensile.sqlite \
  --version-name gfx1151_hotloop_v0 \
  --min-samples 2
```

Additional TensileLite global parameters can be included with repeated `--global-parameter KEY=VALUE`. Benchmark-affecting global parameters are included in the benchmark-protocol hash; compile-only settings such as `CpuThreads` are intentionally excluded.

## Current Limitations

- `schedule-batches` supports seed/random, local mutation, categorical DE, and GOMEA-style proposals; surrogate proposal is still planned.
- Known rejected/unmapped candidate failures are not yet recorded as reusable negative cache entries.
- The current bundled problem type and search-space domains target gfx1151 FP16 NT HHS first.
- Keep `PredictionThreshold: 2.0` for gfx1151 unless Formocast support is added and validated.
