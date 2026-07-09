# NT HHS Search Space Design

This document describes how EvoTensile constructs the gfx1151 FP16 NT HHS candidate space. It intentionally does not describe which search algorithm chooses candidates from that space. Random search, local search, DE, and GOMEA all consume the same candidate representation and validity layer.

## Scope

The implemented target profile is `gfx1151-nt-hhs`:
- GPU architecture: `gfx1151`.
- Operation: batched GEMM with exact shape order `[M, N, batch, K]`.
- Layout: NT, with `TransposeA=False` and `TransposeB=True`.
- Data path: FP16 inputs/outputs with FP32 accumulation (`HHS`).
- Epilogue-capable problem type: bias from `D`, `scaleAlpha_vector`, activation enabled as `hipblaslt_all`, and `UseE=False`.

The pilot profile shape grid is the 100-shape Cartesian product in `evotensile/shapes.py`:

```text
M:     [512, 640, 896, 1024]
N:     [128, 256, 512, 768, 1024]
batch: [1]
K:     [256, 512, 1024, 2048, 4096]
```

## Candidate Representation

An EvoTensile candidate is a complete TensileLite solution dictionary, not one independent knob. `Candidate.canonical_params()` JSON-canonicalizes the full parameter bundle, and `Candidate.hash` is the stable `cand_...` hash of that canonical form.

Generated YAML emits each candidate as one `Groups` entry:

```yaml
ForkParameters:
  - Groups:
    - - MatrixInstruction: [16, 16, 16, 1, 1, 4, 4, 2, 2]
        WorkGroup: [16, 16, 1]
        DepthU: 64
      - MatrixInstruction: [16, 16, 16, 1, 1, 4, 2, 2, 2]
        WorkGroup: [16, 16, 1]
        DepthU: 32
```

This keeps TensileLite from taking a Cartesian product over independent multi-valued fork parameters. The manifest records the intended `(shape_id, candidate_hash, candidate_index, problem_index, solution_index)` mapping for every rectangular batch.

## Domain Construction

`evotensile/search_space.py` defines the broad categorical domains in `DOMAINS` and target constants in `FIXED_PARAMS`.

### Matrix Instructions

Matrix instructions are generated instead of enumerated by hand:
- The fixed prefix is `(16, 16, 16, 1, 1)`.
- The variable shape is `(MIWaveTile0, MIWaveTile1, MIWaveGroup0, MIWaveGroup1)`.
- `MIWaveTile0` and `MIWaveTile1` range from `1` to `9`.
- Wave-group pairs include `(1,1)`, `(1,2)`, `(1,4)`, `(2,1)`, `(2,2)`, `(2,4)`, `(4,1)`, and `(4,2)`.
- Macro tiles above `256x256` are excluded during construction.
- Symmetric transposed macro-tile families are added when they are not already represented.

The default matrix-instruction shape is `(1, 1, 2, 2)`, which produces `MT32x32`.

### Tunable Domains

The remaining domain keys cover these solution families:
- Tile and work shape: `MatrixInstruction`, `WorkGroup`, `DepthU`, `GlobalSplitU`.
- Scheduling and prefetch: `PrefetchGlobalRead`, `PrefetchLocalRead`, `ScheduleIterAlg`, `1LDSBuffer`, `ClusterLocalRead`.
- LDS layout and padding: `TransposeLDS`, `LdsBlockSizePerPadA`, `LdsBlockSizePerPadB`, `LdsPadA`, `LdsPadB`, `SourceSwap`.
- Vectorization and stores: `VectorWidthA`, `VectorWidthB`, `GlobalReadVectorWidthA`, `GlobalReadVectorWidthB`, `StoreVectorWidth`, `StorePriorityOpt`, `StoreSyncOpt`, `GroupLoadStore`, `NumElementsPerBatchStore`.
- Shape assertions and pointer/cache behavior: `AssertFree0ElementMultiple`, `AssertFree1ElementMultiple`, `AssertSummationElementMultiple`, `WorkGroupMapping`, `StaggerU`, `StaggerUStride`, `StaggerUMapping`, `ExpandPointerSwap`.

`FIXED_PARAMS` supplies target-stable settings such as assembly kernels, wavefront size `32`, `GlobalSplitUAlgorithm=MultipleBuffer`, scheduled global/local reads, `LocalReadVectorWidth=16`, `StoreRemapVectorWidth=0`, and `MIArchVgpr=True`.

## Linked Repairs

The raw domain product contains many mechanically invalid combinations. EvoTensile keeps domains broad and repairs linked fields before creating a candidate with `make_candidate()`.

Implemented repairs include:
- Accumulator and VALU VGPR repairs that choose nearby matrix instructions when the lower-bound VGPR estimate exceeds `256`.
- Matrix-instruction dependent vector-width repairs so `VectorWidthA/B` divide `MIWaveTile0/1`.
- `TransposeLDS=1` normalization to `0` for the NT TLU/TLU path.
- `1LDSBuffer` and `ScheduleIterAlg=2` repairs that avoid `PrefetchGlobalRead=0`.
- `1LDSBuffer` with scheduled local writes repaired to a compatible `ScheduleIterAlg`.
- `GlobalSplitU>1` repaired to `DepthU>=32`.
- Global-read vector-width repairs so total global-read vectors divide the computed thread count.
- TLDS2 repairs that force `PrefetchGlobalRead=2`, `PrefetchLocalRead=0`, and `VectorWidthB=1`, then remove incompatible TLDS2 pad-block choices.
- LDS-footprint repairs that clear LDS padding and reduce `DepthU` until the footprint fits under `65536` bytes.
- Store-path repairs that keep `StoreSyncOpt`, `NumElementsPerBatchStore`, `GroupLoadStore`, and `StorePriorityOpt` in supported coupled sets.

Repairs are construction-time conveniences. They do not prove a candidate is build-valid. TensileLite remains the final authority.

## Validity Layer

`explain_invalid_nt_hhs(params, shape=None)` is the source of truth for known-disallowed combinations. It returns structured `InvalidReason` records with:
- `rule_id`: stable identifier such as `nt_hhs.tlds2.requires_pgr2_plr0`.
- `message`: concise explanation.
- `params`: involved parameter names.
- `source`: `schema`, `solutionstructs`, `kernelwriter`, `taskpredicate`, or `heuristic`.
- `shape_dependent`: whether the rule depends on the exact shape.

`cheap_constraints()` is only the boolean wrapper over this explainable API.

### Kernel-Global Rules

The implemented global rules reject:
- Non-positive or too-large macro tiles.
- Matrix-instruction/vector-width divisibility failures.
- Store-vector-width combinations rejected by the `SourceSwap` and non-`SourceSwap` store paths.
- NT `TransposeLDS=1`.
- `1LDSBuffer` / `ScheduleIterAlg=2` combinations with `PrefetchGlobalRead=0`.
- `1LDSBuffer` with scheduled local writes outside compatible schedule-iteration algorithms.
- `GlobalSplitU>1` with `DepthU<32`.
- Global-read vector counts that are not divisible by computed thread count.
- LDS footprints above `65536` bytes.
- C-accumulator or lower-bound VALU VGPR estimates above `256`.
- TLDS2 pad-block divisibility and LSP alignment failures.
- TLDS2 without the observed `PGR=2`, `PLR=0`, `VectorWidthB=1` path.
- Unsupported store-sync and grouped-load/store couplings.

### Shape-Dependent Rules

When a shape is supplied, the same validity API also rejects:
- GSU workspace above `128 MiB` for the shape.
- `K` not divisible by `AssertSummationElementMultiple`.
- `M` not divisible by `AssertFree0ElementMultiple`.
- `N` not divisible by `AssertFree1ElementMultiple`.

Shape-dependent failures are recorded as rejected observations for that `(shape, candidate)` pair, not as global invalidity for the candidate.

## Sampling Helpers

The search-space module exposes deterministic construction helpers for callers that need fresh candidates. These helpers sample broad domain values, apply linked repairs, and retry until `cheap_constraints()` passes. The helper path currently biases toward the higher-yield TLDS2 construction path and enforces proposal-side VALU headroom for generated random candidates, but explicit candidates from imports or DB parents can still use any valid domain path.

Search algorithms decide how many candidates to request, which parents to use, and how generated candidates are ranked. Those policies are documented separately in `docs/search_algorithms.md`, `docs/gomea.md`, and `docs/linkage_learning.md`.

## Maintenance Checks

Use `summarize-space` to inspect the constructed domain sizes and matrix-instruction macro tiles:

```bash
python3 -m evotensile.cli summarize-space --profile gfx1151-nt-hhs
```

Use `proposal-coverage` to check generated candidate coverage without running TensileLite. Coverage reports are used to tune construction helpers and proposal bias without shrinking the underlying domains.
