# EvoTensile

Work in progress. README is AI-generated.

EvoTensile is an external smart-search autotuner for TensileLite / hipBLASLt. It proposes complete TensileLite candidate bundles, emits them as TensileLite `Groups`, runs TensileLite as the evaluator, and stores reproducible timing/cache metadata for iterative search.

The current bundled target configuration is gfx1151 FP16 NT HHS GridBased GEMM tuning, but the core pieces are intended to stay generic: candidate hashing, shape handling, YAML emission, runner orchestration, protocol hashing, and result caching.

See `PLAN.md` for the current target-specific tuning plan and remaining work.

## Implemented Capabilities

- Candidate and shape primitives with stable canonical hashes.
- Exact-shape helpers and the current 100-shape pilot grid generator.
- Candidate generation from deterministic seeds plus random valid configs.
- Prototype search modules for random, local mutation, differential evolution, GOMEA-style mixing, and LFBO-style surrogate search.
- TensileLite YAML generation using one `ForkParameters: Groups` list of complete candidate dictionaries.
- Hot-loop benchmark defaults for steady-state inference/training throughput.
- TensileLite subprocess runner with `run-yaml` and compile-then-serial-benchmark `build-bench-yaml` flows.
- SQLite schema for candidates, shapes, runs, and evaluations.
- Manual timing-cache namespace via `--version-name`, plus problem-type and benchmark-protocol hashes.
- Cache inspection helpers for identity, status summaries, and missing candidate/shape evaluations.
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

Generate a TensileLite YAML:

```bash
python3 -m evotensile.cli pilot-yaml \
  --output-yaml out/pilot.yaml \
  --num-random 32 \
  --seed 1
```

Initialize a DB and register generated candidates/shapes:

```bash
python3 -m evotensile.cli init-db --db out/evotensile.sqlite
python3 -m evotensile.cli register-pilot --db out/evotensile.sqlite --num-random 64
```

Print the cache identity for a benchmark protocol:

```bash
python3 -m evotensile.cli cache-key \
  --version-name gfx1151_hotloop_v0
```

Check which generated evaluations are missing from the cache:

```bash
python3 -m evotensile.cli cache-missing \
  --db out/evotensile.sqlite \
  --version-name gfx1151_hotloop_v0 \
  --num-random 64 \
  --limit-shapes 100
```

Ingest validation-gated TensileLite CSV/log rows, then rank only passing observations. When given a run directory, `ingest-csv` auto-detects TensileLite `*_Final.yaml` / `*_CSVWinner.yaml` files and uses them as the source of truth for candidate mapping.

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

## Running TensileLite

Run an existing YAML directly:

```bash
python3 -m evotensile.cli run-yaml \
  --yaml out/pilot.yaml \
  --output-dir out/tensilelite_run_000 \
  --db out/evotensile.sqlite \
  --version-name gfx1151_hotloop_v0
```

Compile first, then benchmark serially with TensileLite cache reuse:

```bash
python3 -m evotensile.cli build-bench-yaml \
  --yaml out/pilot.yaml \
  --output-dir out/tensilelite_run_000 \
  --db out/evotensile.sqlite \
  --version-name gfx1151_hotloop_v0 \
  --compile-threads -1 \
  --benchmark-threads 1
```

Additional TensileLite global parameters can be included with repeated `--global-parameter KEY=VALUE`. Benchmark-affecting global parameters are included in the benchmark-protocol hash; compile-only settings such as `CpuThreads` are intentionally excluded.

## Current Limitations

- Cache-aware scheduling is not implemented yet; ingestion can identify valid observations, but scheduling still needs to skip measured `ok` candidate/shape pairs automatically.
- The batch scheduler for compile-only candidate batches plus serial benchmark execution is not implemented yet.
- Search modules are prototype proposal engines; they are not yet wired into a full closed-loop scheduler.
- The current bundled problem type and search-space domains target gfx1151 FP16 NT HHS first.
- Keep `PredictionThreshold: 2.0` for gfx1151 unless Formocast support is added and validated.
