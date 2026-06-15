# EvoTensile

EvoTensile is an external smart-search autotuner for TensileLite / hipBLASLt.  It generates curated candidate batches as TensileLite `Groups`, runs TensileLite as the evaluator, stores all observations, and iteratively searches the mixed categorical Tensile kernel-parameter space.

Initial target: gfx1151 FP16 NT HHS non-AuxH GridBased GEMM tuning.

See [`PLAN.md`](PLAN.md) for the full design.

## Current MVP

This initial scaffold supports:

- canonical candidate objects and stable hashes;
- the 100-shape pilot grid;
- deterministic seed candidates plus random candidates;
- YAML generation for TensileLite using `ForkParameters: Groups`;
- SQLite schema for candidates/shapes/runs/evaluations;
- a CLI dry-run flow that writes a pilot YAML without invoking TensileLite.

## Quick start

From the repo root:

```bash
python3 -m evotensile.cli pilot-yaml \
  --output-yaml out/pilot.yaml \
  --num-random 32 \
  --seed 1
```

Or, if installed editable:

```bash
pip install -e .
evotensile pilot-yaml --output-yaml out/pilot.yaml --num-random 32
```

Generate a short summary:

```bash
python3 -m evotensile.cli summarize-space --num-random 128
```

Initialize a DB and register generated candidates/shapes:

```bash
python3 -m evotensile.cli init-db --db out/evotensile.sqlite
python3 -m evotensile.cli register-pilot --db out/evotensile.sqlite --num-random 64
```

## Running TensileLite

Runner integration is intentionally conservative in the MVP.  First generate YAML, inspect it, then run TensileLite manually or via the runner once paths/protocol are confirmed.

Typical future command shape:

```bash
python3 -m evotensile.cli run-yaml \
  --yaml out/pilot.yaml \
  --output-dir out/tensile_run_000 \
  --tensile-bin /home/wd/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/bin/Tensile \
  --db out/evotensile.sqlite
```

## Notes

- Keep `PredictionThreshold: 2.0` for gfx1151 unless Formocast support is added/validated.
- Candidates are emitted as complete dictionaries inside one `Groups` list to avoid Cartesian expansion.
- This project is an orchestration layer; it should not patch TensileLite until the external loop is proven.
