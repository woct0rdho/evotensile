# 1,135-Shape Campaign Experiment Log

This document is the live plan and historical evidence log for the gfx1151 FP16 NT HHS ComfyUI-biased 1,135-shape campaign. Stable subsystem behavior belongs in the focused design documents under `docs/`. This file owns campaign-specific decisions, commands, feedback, and convergence evidence.

## Objective

Tune the named `gfx1151-nt-hhs-comfy1135` profile until sparse general search, measured promotion, and focused outlier repair no longer produce material fresh gains. Preserve every compatible historical and future measurement in one mutable database so later work can replay the complete campaign.

## Shape Set

The deduplicated union contains 1,135 shapes with batch count 1:
- dense core: 792 shapes from `11 M * 9 N * 8 K`.
- ComfyUI feature and skinny-gradient slices: 166 shapes before overlap removal.
- local `2048`, `3072`, and `4096` neighborhoods: 16 shapes.
- workload-oriented `8192` boundary products: 184 shapes before overlap removal.

The implementation is `evotensile.shapes.comfy_nt_1135_shapes`. It is an exact superset of `pilot_100_shapes()`, spans `M/N/K=16...8192`, and was independently checked against the 9,681-shape target decomposition with zero outside points.

Provenance:
- workload rationale: `~/ComfyUI-FeatherOps/doc/input_shapes.md`.
- target decomposition: `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/shape_data/large_grid_target_union_decomposition.json`.
- known guarded 8192 winner: `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/configs/hhs_nt_scale_bias_mt128x128_tlds0_rocblas_1ldsb_vwb2_static_wgm_nepbs10_sia3_nostoreprio_probe_8192.yaml`.

## Operating Rules

- Use one mutable compatible database for imported grid100 evidence, both hipBLASLt baselines, the known 8192 candidate, all search rounds, repair, stabilization, and finalization.
- Keep exact candidate-shape causality. Artifact preparation may be shared, but only explicitly requested pairs become evidence.
- Treat EvoTensile native measurements as the search oracle. hipBLASLt heuristic queries identify seed configurations. Their query timings are provenance, not interchangeable native evidence.
- Preserve validation failures and rejected observations. Never infer family-wide invalidity from one failed pair.
- Keep normal rounds near a 300-second soft admission budget. Admitted work drains and is ingested even after the deadline.
- Prefer previously unknown pairs, parent-diverse shortlists, and measured promotion over dense restarts.
- Use deliberate seeds beginning at `12345` and obvious increments.
- Fresh finalization remains mandatory before any production deployment recommendation.

## Initial Seed Plan

- Add and validate the named 1,135-shape profile and generic profile selection in baseline, practical-round, and finalization scripts.
- Copy `out/grid100_production_search_20260712.sqlite` into the new mutable campaign database, preserving all authoritative grid100 evidence.
- Discover and natively measure current installed hipBLASLt selections over all 1,135 shapes.
- Discover and natively measure preserved untuned hipBLASLt selections over all 1,135 shapes.
- Extend the known guarded SIA3/no-store-priority candidate from its retained 100-shape evidence to all previously unknown shapes.
- Snapshot the complete initial compatible corpus and produce a fresh initial deployment checkpoint.
- Audit winner families, baseline crossovers, sparse coverage, noise, failures, and shape deficits before choosing round 1.

## Planned Search Phases

- broad incumbent-centered store and staging interactions on representative dense-core, skinny-gradient, feature-slice, and boundary shapes.
- measured promotion of successful children across parent-competitive shapes and mechanical neighbors.
- mapping, vector, and LDS trust-region probes only where prior rounds expose headroom or regime-specific behavior.
- integrated outlier repair using fresh checkpoint deficits, uncertainty, low-gain shapes, singleton specialists, and nearby measured candidates.
- stabilization of close/noisy pairs followed by convergence restarts with parent-diverse alternatives.
- full fresh confirmation of final contenders and zero-tolerance deployment selection across all 1,135 shapes.

## Seed Results

The unified mutable database is `out/grid1135_search_20260712.sqlite`. Its immutable pre-search snapshot is `out/grid1135_seed_compatible_20260712.sqlite`.
- initial copied corpus: authoritative grid100 production database with finalization-v4 evidence.
- current finalization-v4 hipBLASLt: 1,135 heuristic selections, 36 candidate families, and complete native coverage after normalizing two integral `StaggerUStride` values represented as YAML floats.
- preserved untuned hipBLASLt: 1,135 heuristic selections, 46 candidate families, complete native coverage, exact old library/database isolation, and no build failures.
- documented guarded SIA3/no-store-priority candidate `cand_07ba5e67b99df4ba`: 1,035 previously unknown pairs, all valid, with 10,350 new samples.
- seed database after imports: 618 candidates, 1,135 shapes, 26,849 events, 280,551 samples, 6,569 validations, and 5,721 artifact mappings. Integrity passes with zero foreign-key violations.

Exact native median comparison between the two hipBLASLt regimes:
- untuned wins 610 shapes, current wins 505, and 20 tie.
- when any dimension is below 128, untuned wins 534/796 shapes and current/untuned median performance is `0.880`.
- when all dimensions exceed 1024, current wins 31/31 shapes with a `1.906` median ratio.
- current is not uniformly better whenever any dimension exceeds 1024. Skinny large-axis shapes continue to favor untuned configurations.

Pooled compatible seed evidence has 88 per-shape winner candidates. The documented guarded candidate wins 100 shapes. Relative to the best exact current/untuned/guarded control on each shape, pooled retained evidence improves 43 shapes, 16 by at least 1%. The median gain is zero. Full details are in `out/grid1135_search_20260712/seed/audit.json`.

The first current baseline import preserved two failed build attempts. Root cause was checked-in integral `StaggerUStride` values encoded as `32.0` and `64.0`. Current TensileLite rejects floats. The importer now converts only integral float values for this integer parameter at the controlled logic-to-candidate boundary. A second labeled import created corrected candidates `cand_8b72e9f53672fe23` and `cand_49c624a146ff6afc`. Both passed all assigned shapes. Historical failures remain retained.

## Initial Fresh Checkpoint

`checkpoint_initial` freshly measured the top two pooled contenders for every shape:
- 2,270/2,270 valid pairs, 120 candidates, and 22,700 samples.
- `346.37 s` wall time. This broader-than-normal run was justified to establish same-session incumbents over the complete grid.
- zero-tolerance deployment: 77 solutions, 66 multi-shape generalists, and 11 singleton specialists.
- fresh ordering improved 91 shapes over the fresh original compatible control, 65 by at least 1%. Mean gain `1.185%`, median zero, maximum `111.00%`.
- historical pooled diagnostic delta: mean `-0.167%`, median `0.001%`, confirming that many large apparent fresh changes are ordering/noise effects rather than a new search result.

The checkpoint deployment `out/grid1135_search_20260712/checkpoint_initial/deployment_0.000.json` is the mandatory incumbent for the first practical round. Its report is `out/grid1135_search_20260712/checkpoint_initial/report.json`.

## Round Log

### Round 1: Broad Staging Interactions

The round completed 64 valid exact pairs across three candidates in `10.32 s`. All pairs were already retained compatible evidence, so this round screened the imported corpus rather than adding new timing events. It found 14 apparent incumbent improvements, 10 by at least 1%. `cand_0bd8ac0c7ad6b04c` won nine shapes by as much as `70.70%`, and `cand_37de88772e982ef1` won four small `K=16` shapes by as much as `15.38%`. Both children were admitted to measured promotion. The third, marginal child was not.

### Round 2: Staging-Child Promotion

The round completed 12 valid pairs for `cand_0bd8ac0c7ad6b04c` in `4.59 s`. The other child had no remaining unknown parent-competitive opportunities. Three additional shapes improved by `5.16-8.31%`, including `m1024_n128_b1_k1024`. The result justified refreshing the deployment checkpoint before opening another interaction family.

### Checkpoint After Round 2

`checkpoint_after_round02` freshly measured 2,281/2,281 valid contender pairs from 121 candidates, with 22,810 samples in `343.37 s`. Its zero-tolerance deployment uses 80 solutions. Fresh same-session selection improved 73 shapes over `checkpoint_initial`, but small-kernel volatility produced implausibly large individual ratios. Checkpoint-wide changes are selection evidence, not attributed search gains.

### Round 3: Broad Store Interactions

The round completed 64 valid pairs across six candidates in `10.34 s`, finding 31 positive comparisons and 14 gains of at least 1%. `cand_876cf1f8152a0a2a` won 11/20 assigned shapes by as much as `11.96%`. `cand_e82a17d013fae0f6` won 10/20 by as much as `1.76%`. `cand_d4288265b319d195` won 9/20 by as much as `1.08%`. and singleton `cand_b74183759aa9b574` gained `10.62%`. The two losing families were dropped. An all-core model worker emitted a read-only SQLite resource warning. `load_db_oracle_matrix()` now scopes its connection with a context manager, and pre-commit plus all 290 tests pass after the fix.

### Round 4: Store-Child Promotion

The round completed 32 valid pairs across four candidates in `7.07 s`. Ten shapes improved and four exceeded 1%. `cand_e82a17d013fae0f6` transferred to five additional `M=896` shapes, three by `1.28-1.67%`. `cand_b74183759aa9b574` added two `N=512` wins. `cand_876cf1f8152a0a2a` added three sub-percent wins. `cand_d4288265b319d195` lost all five promotion pairs and was not expanded further. The store gains are useful but regime-specific rather than broad generalists.

### Checkpoint After Round 4

`checkpoint_after_round04` freshly measured 2,284/2,284 valid contender pairs from 126 candidates, with 22,840 samples in `352.00 s`. Its zero-tolerance deployment uses 85 solutions. Fresh same-session selection improved 92 shapes over `checkpoint_after_round02`, 39 by at least 1%, with a `0.334%` mean gain. These values include expected small-kernel reordering and are not substituted for exact round evidence.

### Round 5: Staging Restart

The restart completed 64 valid pairs across seven candidates in `10.43 s`. It found one material specialization: `cand_6f0cc78a5d957f77` improved `m1024_n128_b1_k256` by `38.87%`. A competing child gained `7.19%` on the same shape, while all broad candidate families lost or remained below 1%. Only the stronger specialist was promoted.

### Round 6: Staging-Specialist Promotion

The round completed 12 valid pairs in `4.44 s`. The specialist did not generalize: it added only one `0.81%` gain at `m256_n16_b1_k32`, with no gain of at least 1%. Staging is therefore paused until other interaction families or repair expose a new parent basin.

## Adaptive Search Summary

Every successful round directory contains `plan.json` with the exact effective CLI parameters, candidate hashes, parent hashes, target-shape scopes, model fit, cost fit, acquisition scores, and selected bundles. `report.json` contains every exact outcome and comparison. The two failed admissions are also retained as empty round directories and described below.

| Round | Seed | Lane | Pairs | Wall time | Gains >=1% | Decision |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 7 | `12351` | mapping | 64 | `10.47 s` | 2 | one 1.19% assignment. No broad transfer |
| 8 | `12352` | vector + repair | 80 | `12.44 s` | 14 | opened the small `K=512` and `M=768` vector basins |
| 9 | `12353` | promotion + repair | 52 | `12.03 s` | 13 | promoted vector children and exposed stale exact deficits |
| 10 | `12354` | LDS + repair | 80 | `12.97 s` | 8 | found skinny-feature LDS child `cand_805b72a03e7e9c1f` |
| 11 | `12355` | promotion + repair | 131 | `15.35 s` | 11 | `cand_805b72a03e7e9c1f` won 58/97 promoted shapes |
| 12 | `12356` | store + repair | 80 | `16.99 s` | 3 | one 32.7% specialist and two modest store children |
| 13 | `12357` | promotion + repair | 15 | `9.87 s` | 4 | no store-child transfer. Repair found stale assignments |
| 14 | `12358` | staging + repair | 80 | `13.43 s` | 13 | opened `cand_10b5fb5bed8513bf` and `cand_d9d63caf9a2dc69d` basins |
| 15 | `12359` | promotion + repair | 121 | `16.00 s` | 17 | validated 13/97 and 10/12 staging-child transfers |
| 16 | `12360` | mapping + repair | 80 | `15.61 s` | 9 | discovered mapping generalist `cand_1a0c6fb0745fd717` |
| 17 | `12361` | promotion + repair | 130 | `13.95 s` | 50 | mapping generalist won 96/112 shapes, 43 by at least 1% |
| 18 | `12362` | vector + repair | 80 | `15.17 s` | 3 | found `ClusterLocalRead=0` child `cand_7658cacb3b7a2a43` |
| 19 | `12363` | promotion + repair | 33 | `9.63 s` | 23 | child won 21/21 promoted shapes, 17 by at least 1% |
| 20 | `12364` | targeted staging | 96 | `15.35 s` | 18 | `DepthU=64` refinement `cand_71492c73bd6f7070` won 19/22 |
| 21 | `12365` | promotion admission | 0 | n/a | 0 | no remaining parent-competitive opportunity. Empty directory retained |
| 22 | `12366` | LDS + repair | 58 | `21.36 s` | 4 | narrow LDS specialists only |
| 23 | `12367` | promotion | 77 | `8.78 s` | 3 | LDS transfer peaked at 1.59%. Broad LDS search closed |
| 24 | `12368` | staging | 80 | `15.62 s` | 14 | one new small-shape family. Direct child of `cand_7149...` lost all 22 |
| 25 | `12369` | promotion | 11 | `6.05 s` | 0 | all promotion pairs lost. Pre-blind staging closed |

### Checkpoint Sequence

Fresh checkpoints used 10 samples per contender and mandatory original/current controls. They are controller anchors, not authoritative production evidence.

| Checkpoint | Pairs | Candidates | Wall time | Zero-tolerance solutions |
| --- | ---: | ---: | ---: | ---: |
| `checkpoint_after_round09` | 2,430 | 154 | `394.54 s` | 92 |
| `checkpoint_after_round11` | 2,576 | 178 | `458.59 s` | 93 |
| `checkpoint_after_round13` | 2,600 | 176 | `459.55 s` | 93 |
| `checkpoint_after_round15` | 2,638 | 186 | `475.49 s` | 96 |
| `checkpoint_after_round17` | 2,715 | 186 | `487.46 s` | 100 |
| `checkpoint_after_round19` | 2,727 | 190 | `496.36 s` | 101 |
| `checkpoint_after_round21` | 2,721 | 186 | `506.23 s` | 105 |
| `checkpoint_after_round25` | 2,753 | 203 | `517.08 s` | 107 |

## Blind `8192^3` Evidence Import

The retained corrected blind campaign used the legacy pre-namespace schema, so it could not be consumed by `merge_compatible_databases.py`. A one-time copy-on-write conversion checked exact problem type, profile shape membership, benchmark protocol compatibility, validation protocol identity, candidate hashes, database integrity, and foreign keys. The conversion utility was removed after the authoritative import completed.

Source:
- database: `out/blind_one_shape_next_v3_20260710_seed20260713/campaign.sqlite`.
- import report: `out/grid1135_search_20260712/blind_import_report.json`.
- pre-import campaign backup: `out/grid1135_pre_blind_import_20260712.sqlite`.

Imported evidence:
- 1,015 candidates with proposal source, parent hashes, and metadata.
- 905 native runs and measured cost attribution.
- 895 validation rows.
- 2,456 grouped probe/screening events with 3,861 samples, including 117 rejections, 32 validation failures, and 3 build failures.
- eight compatible production hot-confirmation events with 80 samples.
- zero candidate collisions, zero foreign-key violations, and integrity `ok`.

The five finalist extension used `scripts/evaluate_candidates.py --shape-file out/grid1135_search_20260712/blind_8192_boundary_shapes.txt` to measure 825 previously unknown exact pairs over the 165 deduplicated shapes touching an `8192` axis. All 825 pairs were valid. Blind candidates improved 10 checkpoint shapes, with 24 candidate-pair gains above 1%. The largest were `29.77%` at `m8192_n256_b1_k8192` and `16.89%` at `m128_n8192_b1_k1024`.

`checkpoint_after_blind_import` measured 2,817 pairs in `594.30 s`. Blind finalists received 15 fresh zero-tolerance assignments, including `8192^3`. The import therefore changed the active campaign rather than serving only as historical model evidence.

## Post-Import Boundary Search

| Round | Seed | Lane | Pairs | Wall time | Gains >=1% | Decision |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 26 | `12370` | blind mapping + repair | 80 | `46.90 s` | 14 | `cand_638941cfe1b331a4` won 10/32, nine by at least 1% |
| 27 | `12371` | promotion | 12 | `7.83 s` | 0 | mapping child remained boundary-local |
| 28 | `12372` | explicit-parent staging | 0 | n/a | 0 | plan serialization failed on a valid non-incumbent parent. Empty directory retained |
| 28b | `12372` | repaired explicit-parent staging | 80 | `54.70 s` | 8 | found `cand_3c59ed613a468093` and `cand_d55ffe3af7ab7fd1` |
| 29 | `12373` | promotion | 16 | `16.52 s` | 7 | both staging children transferred strongly |
| 30 | `12374` | blind store + repair | 80 | `36.88 s` | 3 | broad store variants lost. Boundary store closed |
| 31 | `12375` | global mapping + repair | 96 | `40.77 s` | 18 | found small-shape `cand_def66c9ec7282621` and guarded child `cand_b7bbd802868b03e2` |
| 32 | `12376` | promotion | 12 | `7.60 s` | 8 | guarded child won 8/12, up to 35.30% |
| 33 | `12377` | promotion exhaustion | 8 | `8.16 s` | 2 | added two guarded-parent gains |
| 34 | `12378` | guarded mapping closure | 80 | `26.68 s` | 8 | uniquely dominant `cand_a1df8dab8704fc33` won 16/18 |
| 35 | `12379` | promotion | 1 | `5.34 s` | 0 | no transfer beyond measured scope |
| 36 | `12380` | final mapping closure | 80 | `31.22 s` | 3 | modest `cand_2e548aa37b87223c` refinement |
| 37 | `12381` | promotion | 1 | `5.20 s` | 0 | refinement failed its only remaining opportunity |
| 38 | `12382` | staging + final repair | 96 | `35.74 s` | 12 | final broad candidates `cand_c4bfe29044930d16` and `cand_7c5483aa6ed94a9c` |
| 39 | `12383` | promotion | 57 | `12.37 s` | 2 | only guarded child retained multi-shape material gains |
| 40 | `12384` | promotion exhaustion | 8 | `7.89 s` | 1 | one isolated 1.16% win. No multi-shape transfer |

Post-import checkpoints:
- `checkpoint_after_round29`: 2,811 pairs, 203 candidates, `618.10 s`, and 117 solutions.
- `checkpoint_after_round33`: 2,822 pairs, 207 candidates, `609.39 s`, and 119 solutions.
- `checkpoint_after_round37`: 2,829 pairs, 208 candidates, `605.68 s`, and 112 solutions.

## Infrastructure Feedback

Three campaign inefficiencies were fixed during execution:
- read-only oracle loading now closes SQLite connections deterministically.
- staging interaction grids now include `ClusterLocalRead=(0,1)` after repair discovered a 21/21 transferring `ClusterLocalRead=0` child that the general lane could not generate.
- explicit measured interaction parents may have zero current incumbent assignments. Plan serialization records a zero winner count instead of raising `KeyError`.
- `scripts/evaluate_candidates.py` accepts exact ordered shape files, enabling sparse boundary extension instead of dense profile-wide evaluation.
- the completed one-time legacy blind import preserved native failures, costs, validation, proposal provenance, and hot confirmation before its migration utility was removed.

All implementation changes pass pre-commit and all 295 repository tests.

## Convergence Decision

Search convergence is established after round 40:
- store, LDS, and vector closures produced no remaining transferable broad child.
- the expanded staging region exhausted the successful `ClusterLocalRead`/`DepthU` basin. Later promotion passes produced zero or isolated gains.
- the guarded mapping chain progressed from `cand_b7bb...` to `cand_a1df...` to `cand_2e548...`. Each successive promotion scope collapsed, and the final refinement lost its only remaining opportunity.
- the final integrated-repair restart produced candidates worth one promotion pass, but round 39 retained only two material transfers and round 40 retained one isolated 1.16% assignment.
- no final candidate produced a new multi-shape gain of at least 1% after promotion exhaustion.

The converged mutable database is `out/grid1135_search_20260712.sqlite`. Fresh authoritative finalization must use 30 samples per contender and must not substitute checkpoint rankings for production assignments.

## Authoritative Finalization V1

`out/grid1135_search_20260712/finalization_v1` is the authoritative production evidence for the converged campaign.

Results:
- 2,884/2,884 fresh valid exact pairs from 211 candidates.
- 86,520 fresh timing samples.
- `774.33 s` wall time.
- zero missing selected outcomes, zero selected regressions, database integrity `ok`, and zero foreign-key violations.
- zero-tolerance deployment: 125 solutions, 97 multi-shape generalists, and 28 singleton specialists.
- 0.5% deployment: 113 solutions with `0.00344%` uniform mean loss and `0.498%` worst-shape loss.
- 1% deployment: 107 solutions with `0.00810%` uniform mean loss and `0.939%` worst-shape loss.
- 2% deployment: 88 solutions with `0.07115%` uniform mean loss and `1.993%` worst-shape loss.

Fresh same-session comparison with `checkpoint_after_round37`:
- 217 improved shapes.
- 83 improvements of at least 1%.
- `0.3713%` uniform mean improvement.
- `43.546%` maximum improvement.
- zero regressions by construction.

Fresh same-session comparison with the original compatible control:
- 417 improved shapes.
- 252 improvements of at least 1%.
- `1.2585%` uniform mean improvement.
- `50.625%` maximum improvement.
- zero regressions by construction.

Representative final assignments:
- `m8192_n8192_b1_k8192`: `cand_07ba5e67b99df4ba`, `42.784 TFLOP/s`.
- `m8192_n256_b1_k8192`: `cand_6028f1fd6eb2d3a1`, `28.204 TFLOP/s`.
- `m128_n8192_b1_k1024`: `cand_88b52fcee5c07b57`, `34.264 TFLOP/s`.
- `m1024_n1024_b1_k1024`: `cand_75e7051d292480cf`, `30.608 TFLOP/s`.

Final database audit after artifact cleanup and the one-time parameter-type migration:
- 1,820 candidates and 1,135 shapes.
- Two stale imported candidates with float-valued `StaggerUStride` merged into their existing integer-typed canonical candidates.
- All 115 baseline selections, 115 benchmark events, and two run-cost rows from those candidates were preserved under the canonical candidate IDs.
- 79,625 benchmark events and 824,202 samples.
- 52,745 validations.
- 125 retained content-verified artifact bundles and 2,647 mappings, covering every selected deployment pair.
- 1,015 imported blind proposal occurrences.
- integrity `ok`. Zero foreign-key violations.

Production reporting must use `out/grid1135_search_20260712/finalization_v1/deployment_0.000.json` for maximum speed or an explicitly selected loss-bounded deployment from the same directory. Checkpoints and historical pooled rankings remain diagnostic only.

## hipBLASLt Deployment Validation

The zero-tolerance finalization was exported to all four gfx1151 GridBased variants and installed into `~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel`:
- `hhs`, `hhs_auxh`, `bbs`, and `bbs_auxb` each contain 125 solutions and 1,135 exact mappings.
- all 125 `StaggerUStride` values in each variant are YAML integer scalars.
- source files are byte-identical to the reviewed staging files under `out/gridbased_logic_finalization_1135_v1`.
- `_rocm_sdk_libraries` resolves `libhipblaslt`, `librocroller`, and `hipblaslt/library` to the rebuilt devel installation.

Installed correctness and dispatch evidence:
- the documented six-case `hipblaslt-bench --verify` gate passed 6/6 tuned and off-grid cases.
- eight representative exact tuned-grid cases passed 8/8.
- the complete 1,135-shape grid passed 1,135/1,135 production-heuristic checks in `410.35 s`. Maximum normalized error was `1.3322e-4`.
- all 1,135 runtime solution indexes matched the exported exact mapping. Installed global index minus exported local solution ID was consistently `2074`.
- upstream `hipblaslt-test --gtest_filter='*quick*'` passed all 7,606 tests in `51.48 s`.
- FeatherOps production-heuristic correctness passed TT, TN, NT, and NN for all seven tested sizes, 28/28 combinations.

Performance is reported by oracle and must not be pooled:
- EvoTensile authoritative finalization uses a 30-sample kernel hot loop and reports `30.608 TFLOP/s` at `1024^3` and `42.784 TFLOP/s` at `8192^3`.
- verified installed `hipblaslt-bench` includes bias, scale-vector, API, initialization, CPU-reference, and verification overhead. Its 10-iteration representative run reports `16.531 TFLOP/s` at `1024^3` and `39.209 TFLOP/s` at `8192^3`.
- a FeatherOps production-heuristic NT harness uses `solution_index=-1` with Triton `warmup=100` and `rep=1000`. It reports `25.962`, `31.016`, `42.863`, and `41.186 TFLOP/s` at sizes 1024, 2048, 4096, and 8192.
- the unchanged FeatherOps benchmark uses private `solution_index=-2` autotuning and reports fused NT `26.101`, `31.190`, `42.323`, and `40.397 TFLOP/s` at those sizes. It selected a different 8192 solution (`2162`) from the deployed exact mapping (`2092`), so this is an application autotune oracle rather than production dispatch evidence.
- the same FeatherOps run reports PyTorch NT `25.887`, `31.073`, `41.505`, and `41.478 TFLOP/s` at sizes 1024 through 8192.

The complete machine-readable install, hash, correctness, dispatch, and performance record is `out/gridbased_logic_finalization_1135_v1/deployment_validation_report.json`.

## StreamK 3 Extension

The search-space extension added `StreamK: [0, 3]` with the TensileLite-required linked behavior: `StreamK: 3` uses `GlobalSplitU: 0`, while normal kernels retain positive GSU. The mechanics model was corrected after inspecting TensileLite's `ContractionSolution::partialTileSize()`: StreamK partial-result workspace is `macro_tile_m * macro_tile_n * WorkspaceSizePerElemC * streamk_grid`, even when GSU is zero. EvoTensile now charges a stable one-WGP grid proxy for nonzero StreamK candidates instead of incorrectly returning zero workspace. The regression test is `test_streamk_mechanics_account_for_partial_workspace`.

The historical campaign database was migrated in place to canonicalize `StreamK: 0` on the 1,820 existing candidates. Candidate row IDs and evidence foreign keys were preserved. The pre-migration database is `out/grid1135_pre_streamk_migration_20260802.sqlite`. The migration report and hash mapping are under `out/grid1135_search_20260712/`.

### StreamK Assignment Probe

`streamk_round01_assignment_probe` evaluated the finalization assignment's normal candidate against a direct `StreamK: 3` variant for all 1,135 shapes, with 10 samples per pair. It completed 1,135 requested pairs in 116 batches and produced 115 unique StreamK variants. There were 1,118 valid comparisons, 12 build failures attributed to TensileLite's `No valid solutions found`, and five validation failures. Among valid comparisons, StreamK was faster on 214 shapes, gained at least 1% on 157, and gained at least 3% on 136. The aggregate median and mean speedups were `-3.39%` and `-3.76%`, so StreamK was not promoted globally from this probe. The complete report is `out/grid1135_search_20260712/streamk_round01_assignment_probe/analysis.json`.

### StreamK Confirmation

Two focused confirmation passes used fresh normal-kernel controls and the existing compatible campaign database:
- `streamk_round02_confirmation` remeasured the 136 first-probe cases at or above 3%. All 272 requested pairs and 2,720 samples were successful. StreamK won 126 cases, retained at least 1% on 123, and retained at least 3% on 119. The median and mean speedups were `16.20%` and `19.60%`. The range was `-27.75%` to `124.13%`.
- `streamk_round03_confirmation` remeasured the remaining 21 first-probe cases in the 1-3% band. All 42 requested pairs and 420 samples were successful. StreamK won 15 cases and retained at least 1% on 13. The median and mean speedups were `2.05%` and `2.91%`. The range was `-6.29%` to `18.59%`.

This exhausts all 157 first-probe cases with an apparent material gain of at least 1%. The confirmed StreamK assignment contains 132 shape assignments across 29 StreamK candidates. The staged selection is `out/grid1135_search_20260712/streamk_round02_confirmation/selection_candidate.json`. The four staged GridBased YAMLs contain 1,135 exact mappings and 29 `StreamK: 3` solution records each. Confirmed losses and all untested sub-1% probe cases remain on their normal assignments.

The staged export completed without a TensileLite run and without rebuilding hipBLASLt. It is not authoritative production finalization: it uses 10-sample confirmation evidence and must receive a fresh 30-sample finalization before any source update or installation. No hipBLASLt source files were modified.

### StreamK Parameter Search: Staging Interaction

The existing `scripts/run_grid100_practical_round.py` was reused with `--interaction-profile staging`, the confirmed StreamK deployment candidate as the incumbent, the same campaign database, and seed `12400`. The round directory is `out/grid1135_search_20260712/streamk_round04_staging`.

The round admitted 74 exact candidate-shape pairs and all 74 completed successfully. It found 37 incumbent improvements, including 27 at least 1%. Of those, 25 improvements came from StreamK variants. 17 were at least 1% and 14 were at least 3%. The strongest confirmed StreamK interaction children were:
- `cand_27effc4cc199acfe`: `DepthU=32`, `PrefetchGlobalRead=2`, `ClusterLocalRead=0`. It improved several `N=128`, `K=1024...8192` cases, including `m64_n128_b1_k2048` by `50.11%` in this round.
- `cand_43133e7b88b125cd`: the same main staging changes, improving `m16_n128_b1_k1024` by `21.09%` and `m64_n128_b1_k1024` by `4.43%`.
- `cand_556b4c2dbcebc9f6`: `DepthU=64`, `PrefetchGlobalRead=2`, `ClusterLocalRead=1`. It improved `m512_n16_b1_k2048` by `59.54%`.
- `cand_3dd8ccc8c577fa87`: `DepthU=64`, `PrefetchGlobalRead=2`, `ClusterLocalRead=1`. It improved `m256_n16_b1_k512` by `15.15%`.

These are exact measured comparisons, not deployment decisions. The next step is promotion against each child's measured parent and remaining parent-competitive shapes. The round report and plan retain the complete candidate, parent, shape, and sample provenance.

### StreamK Staging Promotion

The same practical-round CLI then promoted the measured staging children using their recorded parent hashes. `streamk_round05_promotion` evaluated 69 exact pairs. All completed successfully. Five children transferred to additional parent-competitive shapes, producing 14 incumbent improvements overall, 10 at least 1% and six at least 3%. Two transferring children retained `StreamK: 3`: `cand_a60199abb2f6e476` and `cand_e192c9218858d732`. The remaining transfers were normal-kernel children and are retained as compatible campaign evidence. The round report is `out/grid1135_search_20260712/streamk_round05_promotion/report.json`.

### StreamK Mapping Interaction

The existing practical-round CLI was reused again with `--interaction-profile mapping`, restricted to the 29 measured StreamK parents from the confirmed assignment. `streamk_round06_mapping` completed 80 exact pairs with no failures. It found 41 incumbent improvements, 23 at least 1% and 17 at least 3%, all from `StreamK: 3` children. The strongest result was `cand_6f2fad6edac99e62`, which improved `m16_n3840_b1_k32` by `112.82%`. Other strong cases included `m64_n128_b1_k2048` at `48.87%` and `m16_n128_b1_k1024` at `35.94%`. The dominant measured mapping pattern was `WorkGroupMapping=8` with shape-dependent `StaggerU`, `StaggerUMapping`, and occasionally `SourceSwap`. The round report and plan are under `out/grid1135_search_20260712/streamk_round06_mapping/`.

### StreamK Mapping Promotion

`streamk_round07_mapping_promotion` promoted the measured mapping children through 96 exact pairs, all successful. It retained 17 incumbent improvements, eight at least 1% and four at least 3%. The strongest transfers were `cand_63286bb7c96d1e51` on `m16_n1024_b1_k512` at `13.23%`, `cand_afd734bfa7a499a0` on `m640_n640_b1_k2048` at `7.97%`, and `cand_3bfb7928d1fa300a` on `m32_n2048_b1_k2048` at `6.93%`. The promotion report is `out/grid1135_search_20260712/streamk_round07_mapping_promotion/report.json`.

### StreamK Vector Interaction

`streamk_round08_vector` reused the practical-round CLI against the 29 StreamK parents and completed 80 requested pairs. Seventy-two pairs were valid. Eight failed validation and were retained as failures. Among valid outcomes, the round found 30 incumbent improvements, 22 at least 1% and 14 at least 3%. Strong examples included `m384_n64_b1_k2048` at `57.23%`, `m64_n128_b1_k2048` at `49.05%`, `m1024_n16_b1_k2048` at `38.54%`, and `m16_n3840_b1_k32` at `34.92%`. The report is `out/grid1135_search_20260712/streamk_round08_vector/report.json`.

The eight validation failures remain excluded from promotion. The next round enables the existing integrated repair reserve for local outlier work. In parallel, StreamK candidates that show transferable staging, mapping, or vector parameters will be compared with `StreamK: 0` counterparts before any normal-kernel promotion.

### StreamK Local Outlier Repair

`streamk_round09_local_repair` reused the CLI's integrated repair reserve with 32 deficit targets, repair weight `1.0`, and four mutations per target. It admitted 96 exact pairs: 87 valid and nine validation failures. Five repair-lane candidates were selected. Nine repair comparisons improved their incumbents by at least 1%. The largest were `cand_c9773c1b6fe0c8e4` on `m512_n16_b1_k2048` at `123.61%`, `cand_64930acbfc294c0d` on `m128_n128_b1_k1024` at `71.15%`, and the same `cand_64930acbfc294c0d` on `m384_n32_b1_k2048` at `51.17%`. The failed validation pairs remain excluded. The repair plan and evidence are in `out/grid1135_search_20260712/streamk_round09_local_repair/`.

### Repair Promotion Closure

`streamk_round10_repair_promotion` promoted the two repair children with material measured gains through eight exact parent-competitive pairs. All eight were valid, but neither child produced an additional incumbent improvement. The repair candidates therefore remain shape-local and are not generalized further. The report is `out/grid1135_search_20260712/streamk_round10_repair_promotion/report.json`.

### StreamK-to-Normal Propagation

The eight strongest StreamK configurations were converted to linked normal counterparts by changing only `StreamK: 3, GlobalSplitU: 0` to `StreamK: 0, GlobalSplitU: 1`. `streamk_round11_propagation` measured 368 counterpart pairs, all valid. On the 176 normal-incumbent comparisons, 48 gained at least 1% and 39 gained at least 3%. The best per-shape normal propagation selected 13 shapes at or above 3%. The aggregate was negative because the propagation set was intentionally broad, so only per-shape winners are eligible. The analysis and selection checkpoint are under `out/grid1135_search_20260712/streamk_round11_propagation/`.

The existing practical-round CLI then searched staging interactions around the eight normal counterparts in `streamk_round12_normal_propagation_staging`. It admitted 24 pairs: 16 valid and eight TensileLite build failures. The valid pairs produced six gains, all at least 3%. Two normal children were promising: `cand_2340cfb1045e7b83` improved `m16_n3840_b1_k32` by `161.45%`, and `cand_e197883f411a0296` improved five shapes by `18.40-48.13%`. The build failures remain attributed and excluded. No hipBLASLt rebuild was performed.

### Normal Propagation Promotion

`streamk_round13_normal_propagation_promotion` promoted the two normal staging children through eight valid pairs. It retained one additional normal-kernel improvement: `cand_2340cfb1045e7b83` improved `m384_n64_b1_k2048` by `8.14%`. The other child did not transfer further. The report is under `out/grid1135_search_20260712/streamk_round13_normal_propagation_promotion/`.

### Vector Transfer Closure

`streamk_round14_vector_promotion` promoted the vector-lane children with passed material source measurements. It admitted 68 pairs, with 67 valid and one validation failure. No new improvement reached 1%. The only positive comparisons were `0.26%` and `0.07%`. The vector transfer lane is closed at the current measured candidate set, with the validation failure retained and excluded.

### StreamK LDS Interaction

`streamk_round15_lds` reused the practical-round CLI against the 29 StreamK parents with the LDS interaction profile and 24 integrated repair targets. All 80 admitted pairs were valid. The round found 42 incumbent improvements, 22 at least 1% and 17 at least 3%. Strong LDS children improved `m64_n128_b1_k2048` by `49.08%`, `m384_n32_b1_k2048` by `47.60%`, `m16_n3840_b1_k32` by `37.29%`, and `m16_n128_b1_k1024` by `34.45%`. The round retained 24 local repair targets and one repair-lane candidate for follow-up. Its report is `out/grid1135_search_20260712/streamk_round15_lds/report.json`.

### StreamK LDS Promotion

`streamk_round16_lds_promotion` promoted the passed LDS children through 76 exact pairs, all valid. It retained 14 positive comparisons, six at least 1% and four at least 3%. The strongest transfers were `cand_9d27e25fe29863f7` on `m640_n640_b1_k2048` at `19.97%`, the same child on `m1536_n256_b1_k8192` at `4.67%`, and `cand_369807f97d269c3c` on `m128_n64_b1_k1024` at `11.42%`. The promotion report is `out/grid1135_search_20260712/streamk_round16_lds_promotion/report.json`.

### StreamK Store Interaction

`streamk_round17_store` reused the practical-round CLI with the store interaction profile and 24 integrated repair targets. All 80 admitted pairs were valid. It found 37 incumbent improvements, 20 at least 1% and 18 at least 3%. The strongest store children improved `m16_n3840_b1_k32` by `97.58%`, `m384_n64_b1_k2048` by `54.76%`, `m64_n128_b1_k2048` by `48.39%`, and `m16_n128_b1_k1024` by `28.55%`. One repair-lane candidate was selected for follow-up. The report is `out/grid1135_search_20260712/streamk_round17_store/report.json`.

### StreamK Store Promotion

`streamk_round18_store_promotion` promoted the passed store children through 67 exact pairs, all valid. It retained six positive comparisons, but only two reached 1%: `cand_66dfc96d8a95a6de` improved `m1536_n256_b1_k8192` by `4.25%` and `m640_n640_b1_k4096` by `3.29%`. The remaining four gains were below 1%, so broad store transfer is considered closed. The report is `out/grid1135_search_20260712/streamk_round18_store_promotion/report.json`.

### Consolidated Promotion Exhaustion

`streamk_round19_promotion_exhaustion` combined the passed material children from the staging, mapping, vector, repair, LDS, store, and normal-propagation lanes. It admitted 128 exact pairs. 127 were valid and one failed validation. The pass retained 30 positive comparisons, 14 at least 1% and six at least 3%. Notable remaining transfers were `m128_n16_b1_k8192` at `16.63%`, `m640_n640_b1_k2048` at `10.50%`, and `m32_n2048_b1_k3072` at `8.39%`. Because material transfers remained, the campaign is continuing with a final child-promotion pass. The report is `out/grid1135_search_20260712/streamk_round19_promotion_exhaustion/report.json`.

### Final Child Promotion

`streamk_round20_final_promotion` evaluated 68 exact pairs, all valid. It retained three gains of at least 1%: `cand_66dfc96d8a95a6de` improved `m8192_n16_b1_k2048` by `3.78%`, `cand_8a2d15b254a87f66` improved `m128_n32_b1_k4096` by `2.63%`, and `cand_afd734bfa7a499a0` improved `m1024_n256_b1_k1024` by `1.64%`. One additional comparison was above 3% only in the earlier scope. This pass found no broader family expansion. The report is `out/grid1135_search_20260712/streamk_round20_final_promotion/report.json`.

### Convergence Check

`streamk_round21_convergence_check` evaluated 26 exact pairs, all valid. It found one remaining material transfer: `cand_afd734bfa7a499a0` improved `m4096_n32_b1_k4096` by `8.58%`. No other candidate transferred. The campaign therefore continues with one single-candidate closure pass rather than reopening a broad interaction family. The report is `out/grid1135_search_20260712/streamk_round21_convergence_check/report.json`.

### Single-Candidate Closure And Checkpoint

`streamk_round22_single_candidate_closure` evaluated the final three parent-competitive pairs for `cand_afd734bfa7a499a0`. All were valid and none improved. Promotion is therefore exhausted across the measured staging, mapping, vector, LDS, store, repair, and normal-propagation children.

The consolidated candidate checkpoint is `out/grid1135_search_20260712/streamk_round22_single_candidate_closure/selection_candidate.json`. Relative to the propagated-normal base checkpoint, it changes 137 shape assignments, selects StreamK on 170 shapes, and uses 175 candidate solutions across all 1,135 mappings. A staged four-variant export succeeded with complete confirmation timing, passed validation, and 3,769 registered artifact mappings. The exporter performed no TensileLite run and no hipBLASLt rebuild. This checkpoint remains non-authoritative until fresh 30-sample finalization completes.

### Authoritative StreamK Finalization

`out/grid1135_search_20260712/finalization_streamk_v1` freshly finalized the converged normal and StreamK campaign. The preserved pre-StreamK baseline used old candidate hashes, so a one-time copy-on-write remap produced `out/grid1135_baseline_streamk_compatible_20260802.sqlite` from `out/grid1135_pre_streamk_migration_20260802.sqlite` using the retained exact hash mapping. The remapped baseline contains all 1,820 historical candidates with canonical `StreamK: 0` parameters, passes integrity and foreign-key checks, and preserves the original-control timing corpus. No migration utility remains in the repository.

Results:
- 3,535/3,535 fresh valid exact pairs from 332 candidates and 106,050 fresh timing samples.
- zero-tolerance deployment: 180 solutions, including 63 StreamK solutions assigned to 197 shapes.
- 0.5% deployment: 159 solutions with `0.00600%` uniform mean loss and `0.492%` worst-shape loss.
- 1% deployment: 142 solutions with `0.02088%` uniform mean loss and `0.968%` worst-shape loss.
- 2% deployment: 123 solutions with `0.08672%` uniform mean loss and `1.972%` worst-shape loss.
- relative to the fresh same-session round-22 incumbent, zero tolerance improves 295 shapes, 114 by at least 1%, with `1.025%` mean improvement and `70.41%` maximum improvement.
- relative to the fresh same-session original compatible winner, zero tolerance improves 395 shapes, 252 by at least 1%, with `4.166%` mean improvement and `107.48%` maximum improvement.
- database integrity is `ok`, with zero foreign-key violations and complete 1,135-shape coverage in all four deployment files.

The benchmark phase completed all 332 candidate batches before an interrupted serial report phase. Inspection found that `_timing_rankings()` rescanned every candidate-shape group for every profile shape, causing approximately 90 million Python comparisons on one CPU core after GPU work had ended. The implementation now groups timing summaries by shape in one pass and has a regression test that asserts a single pair-group scan. Finalization plans also record `fresh_started_at`, and `--resume-postprocess` can finish an interrupted report from already-ingested fresh evidence without repeating validation or GPU timing. The recovered postprocessing completed in under two seconds while monitored at zero GPU use. The authoritative report is `out/grid1135_search_20260712/finalization_streamk_v1/report.json`.

The zero-tolerance deployment was exported in staged mode to `out/grid1135_search_20260712/finalization_streamk_v1/logic_staged`. All four `hhs`, `hhs_auxh`, `bbs`, and `bbs_auxb` YAMLs contain 180 solutions and 1,135 exact mappings. Each has 63 `StreamK: 3` solution records, no unsupported StreamK value, and no violation of the linked `StreamK: 3`/`GlobalSplitU: 0` rule. The exporter resolved 3,927 registered artifact mappings. No hipBLASLt source file was modified, no TensileLite run occurred, and hipBLASLt was not rebuilt or installed at that checkpoint.

### StreamK Auxiliary-E Investigation

The four-variant preview exposed a TensileLite code-generation bug for `StreamK > 0` with auxiliary E output (`ProblemType.UseE`). `AsmStoreState` allocated the E-address VGPR only for normal `GlobalSplitU == 1/-1` kernels, leaving `addrEVgpr=None` in StreamK global writes. The correction introduces a shared `useE` condition that also covers non-workspace StreamK stores and applies it consistently to E-address allocation, accounting, setup, and gradient E data. Related StreamK-aware E-pointer conditions were corrected in `AsmAddressCalculation.py`, `Components/ComputeStoreVgprs.py`, and `Components/GlobalWriteBatch.py`. Isolated AuxH, BBS, and AuxB full-library generation then completed successfully, and the prior AuxH memory fault disappeared.

The first address-allocation repair removed the memory fault but left the original `m=512,n=16,k=2048` AuxH diagnostic at approximately `1.0` normalized error. The remaining defect was E row-address state: row-pointer initialization and advancement still used GSU-only conditions, so StreamK output rows could target stale E offsets. E itself must not be accumulated in StreamK workspace. Partial workgroups write raw accumulators, fixup adds those partials to `ValuC`, and only the tile's final owner executes the ordinary epilogue and writes E once.

The final TensileLite implementation encodes that ownership in `useEForStore(kernel, isWorkspace)`. A final-output StreamK `StoreState` allocates and advances E addressing, while StreamK partial/fixup workspace states do not enable E. `Components/GlobalWriteBatch.py` uses the same store-state predicate for E loads, conversion, stores, and issued-operation accounting. The temporary `_validateStreamKUseE` rejection was removed. `Tests/unit/test_streamk_use_e.py` covers normal, StreamK final-output, and StreamK workspace phases.

The original failing AuxH case now passes with normalized error `6.39533e-05` and an `SK3_GSU0` kernel. Initial varied matrices covered tiny, long-K, tall, wide, and partial/fixup cases. The final installed validation then exercised every one of the 197 deployed SK3 mappings and all 63 distinct SK3 solutions in both Aux variants. FP16 AuxH passed 197/197 with maximum normalized error `0.000260617`. BF16 AuxB passed 197/197 with maximum normalized error `0.00482834`. The exhaustive summaries are under `install_validation/installed_auxh_streamk_usee_full/` and `install_validation/installed_auxb_streamk_usee_full/`.

### Final Uniform Deployment

E-specific tuning is deferred. The authoritative zero-tolerance `deployment_0.000.json` assignment is copied unchanged from HHS/BBS to AuxH/AuxB. The staged logic is under `out/grid1135_search_20260712/finalization_streamk_v1/logic_streamk_usee`. Each of the four variants contains:
- 180 solutions: 117 `StreamK: 0` and 63 `StreamK: 3`.
- 1,135 unique exact mappings: 938 SK0 and 197 SK3 assignments.
- no unsupported StreamK value and no linked StreamK/GSU violation.

The staged files were copied to the corresponding gfx1151 GridBased source files in `~/rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/GridBased/`. Every source file is byte-for-byte identical to its staged `logic_streamk_usee` counterpart. This deliberately treats the non-E configurations as the Aux candidate and assignment bank. No performance claim from E-specific retuning is implied.

### hipBLASLt Build And Install

The existing gfx1151 release target was cleaned before rebuilding and installing hipBLASLt from `~/rocm-libraries/projects/hipblaslt` into `~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel`. This forced regeneration of all TensileLite code objects from the corrected source. The installed client reports hipBLASLt version `100401` and git version `95d71d4372-dirty`. The complete build/install log is `out/grid1135_search_20260712/finalization_streamk_v1/install_validation/build_hipblaslt_streamk_usee.log`.

Installed validation explicitly used:

```bash
export HIPBLASLT_TENSILE_LIBPATH="$ROCM_PATH/lib/hipblaslt/library/gfx1151"
export LD_LIBRARY_PATH="$ROCM_PATH/llvm/lib:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
```

The installed library, the eight relevant base/Aux code-object and logic assets, their timestamps, sizes, and SHA-256 digests are recorded in `out/grid1135_search_20260712/finalization_streamk_v1/install_validation/deployment_validation_report.json`.

### Installed Correctness And Dispatch Validation

The full 1,135-shape FP16 NT HHS grid was rerun against the installed library with `scripts/verify_installed_hipblaslt.py`, using one cold plus one measured iteration. All 1,135 cases passed in 418.57 seconds with zero failures and maximum normalized error `0.00013322`. Runtime dispatch exactly matched the deployed assignment: 938 SK0 cases and 197 SK3 cases, with zero StreamK-mode mismatches. Every runtime solution index was the YAML-local solution index plus 2,239, reflecting the enlarged combined library. The command and result are in `install_validation/installed_full_grid_streamk_usee_command.txt` and `install_validation/installed_full_grid_streamk_usee/summary.json`.

Installed Aux checks passed all 197 FP16 SK3 mappings and all 197 BF16 SK3 mappings. Each set covered all 63 distinct deployed SK3 solutions, every solution name contained `_SK3_`, and all results stayed within the corresponding client tolerance. The exhaustive runs completed in approximately 60.06 seconds for AuxH and 58.63 seconds for AuxB.

hipBLASLt client/gtest validation used the installed gfx1151 library explicitly. The full quick suite passed 7,609/7,609 tests. The parameterized FP16 NT GELU auxiliary filter `*matmul_bias_gelu_aux_fp16*_NT_*` passed 32/32 tests. The XML results are `install_validation/hipblaslt_test_streamk_usee_quick.xml` and `install_validation/hipblaslt_test_streamk_usee_auxh_nt.xml`.

### Application Benchmark And Final Repository Checks

`~/ComfyUI-FeatherOps/benchmark_mm_hipblaslt_fp16.py` completed successfully against the installed library. It exercised FP16 NT/NN and BF16 NT/NN square matrix multiplication from 128 through 8,192. The recorded private-auto-tuning path used solution index `-2`, so the output is an application integration and performance check rather than production-heuristic dispatch evidence. For FP16 NT, measured hipBLASLt throughput ranged from 0.583 TFLOP/s at 128 to 42.28 TFLOP/s at 4,096 and 41.39 TFLOP/s at 8,192. The complete output is `install_validation/featherops_benchmark_mm_hipblaslt_fp16.log`.

Final repository checks after deployment were:
- TensileLite targeted generation tests: 101 passed.
- TensileLite unit suite: 5,311 passed, 989 skipped, 16 expected failures, with one unrelated stale client-config golden deselected.
- EvoTensile: 307 passed in 28.03 seconds.
- EvoTensile `pre-commit --all-files`: pyupgrade, Ruff format, and ty passed, while Ruff check reported pre-existing executable-bit errors on unrelated scripts. Those modes were not changed as part of this work.

The consolidated machine-readable deployment, source/hash, installed-asset, correctness, dispatch, gtest, and test evidence is `install_validation/deployment_validation_report.json`.

### Final Status

The gfx1151 1,135-shape campaign is deployed and validated with one uniform copied configuration bank across HHS, BBS, AuxH, and AuxB. All four variants retain the 197 fresh-evidence StreamK assignments. StreamK + `UseE` is enabled through final-owner-only E output, while partial/fixup work remains raw-accumulator workspace traffic. E-specific tuning remains deferred.
