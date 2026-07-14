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
- target decomposition SHA-256: `3d9eb39eef6b754c834fad652345a32ba20784a985fb8488a1a9c53fe02f88ea`.
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
- Recover the matching untuned GridBased YAML from hipBLASLt revision `6c767e113cf31590326cc72d980586a597095cca`. Its YAML SHA-256 is `9cbf840639705192fb0a8123ef08aaa7c6ecd2bf63e0b71d8dc44e86a6187ab6` and its preserved device database SHA-256 is `900723b8a5fd64bd4024b875893fcd318fa672d6a7b5d20d4e13618b2a2317a1`.
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
- source SHA-256: `451e66d83f541f3e0ac5864f752d4be531fdb934e8582c5a79be53d6d2395768`.
- import report: `out/grid1135_search_20260712/blind_import_report.json`.
- pre-import campaign backup: `out/grid1135_pre_blind_import_20260712.sqlite`, SHA-256 `db0e34e15fdee872548be8b265c05b11203fa2c86bf5a4340de361d13347d99b`.

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
- SHA-256 `cefadd263c3f3b2bf2caaa3976bbeed597688d4321f160ae95b73da90662f7f7`.
- integrity `ok`. Zero foreign-key violations.

Production reporting must use `out/grid1135_search_20260712/finalization_v1/deployment_0.000.json` for maximum speed or an explicitly selected loss-bounded deployment from the same directory. Checkpoints and historical pooled rankings remain diagnostic only.

## hipBLASLt Deployment Validation

The zero-tolerance finalization was exported to all four gfx1151 GridBased variants and installed into `~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel`:
- `hhs`, `hhs_auxh`, `bbs`, and `bbs_auxb` each contain 125 solutions and 1,135 exact mappings.
- all 125 `StaggerUStride` values in each variant are YAML integer scalars.
- source files are byte-identical to the reviewed staging files under `out/gridbased_logic_finalization_1135_v1`.
- `libhipblaslt.so.1.4` SHA-256 is `d1bccc55bd0b9213bcc72d7e8955e407cdd378f065542c1bf5e29f5e43a7b2ce`.
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
