# Blind Campaign Control

This document describes the singleton proposal policy executed by the generic campaign runner in `evotensile/campaign/runner.py`. Shared serializable state, grid metrics, robust admission estimation, and soft-budget semantics are documented in `docs/multi_shape_campaign_control.md`. The broader blindness, replay, and artifact contract is documented in `docs/blind_experiment_infrastructure.md`. Search operators remain documented in the search-prefixed design docs.

## Implementation Ownership

The one-shape workflow is split by domain:
- `evotensile/campaign/controller.py` owns shared serializable state, grid metrics, robust admission estimation, and soft-budget accounting.
- `evotensile/campaign/models.py` owns immutable campaign and proposal records.
- `evotensile/campaign/configuration.py` owns protocol construction and binary, implementation, environment, profile, and topology identity.
- `evotensile/campaign/store.py` owns artifact paths, exact proposal serialization, strict resume checks, phase checkpoints, progress, and summaries.
- `evotensile/campaign/proposal_policy.py` owns cold/feedback allocations, island phases, restart transitions, proposal events, and deterministic round seeds.
- `evotensile/campaign/runner.py` owns generic controller restoration, soft admission, exact schedule invocation, evidence disclosure, artifact coverage, stabilization, diagnostics, confirmation, and campaign completion.
- `scripts/run_blind_one_shape.py` parses CLI options, resolves the profile and shape, constructs `CampaignRun`, and invokes `run_campaign()`.

Reusable candidate generation lives in `evotensile/search/acquisition.py`. Shared candidate-bundle scoring lives in `evotensile/campaign/acquisition.py`. Reusable population and budget helpers remain in `evotensile/search/campaign_control.py`. Campaign modules import those owners directly rather than through scheduler compatibility exports.

Tests mirror these boundaries: `tests/test_campaign_control.py` covers reusable control calculations, `tests/test_campaign_proposal_policy.py` covers acquisition and island proposal semantics, `tests/test_campaign_store.py` covers exact candidate persistence, and `tests/test_one_shape_campaign.py` covers the package state machine plus one CLI integration path.

## Scope

The active proposal profile is a fixed-policy one-shape experiment harness running through the generic controller. `CampaignRun` owns an ordered shape tuple, while the current proposal profile intentionally rejects more than one shape until clustering and multi-shape acquisition are implemented. It does not change the default behavior of ordinary `schedule-batches` runs.

Its responsibilities are:
- construct and isolate cold populations.
- select exact parents for each island phase.
- migrate to a merged archive after the isolation period.
- record proposal provenance and population diagnostics.
- trigger bounded diversity restarts.
- estimate whether another round fits before the confirmation reserve.
- checkpoint exact pending proposals and resume deterministically.
- optionally stop on a combined performance/diversity convergence signal.

## Frozen Configuration

A new campaign writes `campaign_configuration.json` before measurement. One immutable configuration owns:
- seed, shape, profile, problem identity, built-in proposal policy/counts/operator controls, linkage settings, and profile mutation rates.
- screening, validation, probe, adaptive-retiming, stabilization, and hot-confirmation protocols.
- sample limits, round limits, restart/convergence controls, soft search budget, and confirmation reserve.
- batch sizes, compile threads, preparation/validation concurrency, cache behavior, timeouts, and explicit physical-CU/WGP topology.
- absolute runner and TensileLite paths, SHA-256 content/source fingerprints, the EvoTensile implementation fingerprint, and behavior-affecting environment variables.

The current defaults remain `48` cold candidates, `24` requested feedback candidates, six isolated rounds, `16` island-local elites, `32` merged elites, an `8x` pool, `32` preparation workers, a `32`-batch preparation wave, one validation worker, and a `60s` confirmation reserve. Singleton oversized-pool selection uses information weight `0.05` after at least 24 exact positive observations.

Resume reconstructs the complete configuration from the current invocation and rejects any field, binary, implementation, or environment mismatch before reading campaign state. There is no partial identity or compatibility override.

## Singleton Pool Selection

Cold-start proposal selection remains mechanical covering because no performance model is yet identifiable. Once the campaign DB contains at least `surrogate_min_evidence` exact positive observations for the sole shape, the built-in provider fits the shared contextual pair model and selects generated singleton candidate bundles by expected improvement, calibrated information value, validity, and predicted cost. The selected P12 information weight is `0.05`.

This changes only the oversized generated-pool shortlist. Candidate generation, island isolation, exact scheduling, validation, probing, stabilization, confirmation, admission, and checkpoint behavior remain unchanged. Proposal metadata records `selection_method` as `singleton-bundle-acquisition`, `surrogate`, or `unfiltered`. The full controlled replay is documented in `docs/campaign_policy_tuning.md`.

## Proposal Events And Occurrences

The scheduler returns explicit preserved, novel-generated, and selected candidate sets. Each proposal call persists shared namespace, scope, arguments, island/restart identity, duration, and timestamp in `proposal_events`. Child `proposal_candidates` rows own source, parent keys, operator metadata, generated/preserved state, and selected state. Proposal cost is derived from event duration across distinct generated candidates, while preserved candidates receive none.

Campaign checkpoints additionally persist the exact call sets in `round_NN/proposals.json`. `load_island_elites()` ranks compatible validation-passed timing evidence, restores parameter-only candidates, then reconstructs selected occurrence lineage and island metadata from SQLite.

The general scheduler's `parent_candidates` override prevents DB-global elites or transfer candidates from leaking into an isolated island proposal call.

## Island Phases

Round zero splits the cold measurement budget across independently seeded islands. The second island receives the first island's already covered mechanical tokens so the two pools favor complementary coverage.

For rounds `1` through `island_isolation_rounds`:
- each island loads only its own validation-passed elites.
- learned linkage is disabled between islands.
- operator budgets are split deterministically.
- no candidate from another island can become a parent through normal DB-global ranking.

After isolation, proposal calls use the merged family/global archive with learned linkage enabled. Migration is therefore explicit and phase-based rather than continuous.

## Diversity Restarts

`population_diagnostics()` reports deduplicated candidate count, family-cell count, distinct complete MatrixInstruction count for audit, mechanical-token count, and mean/minimum categorical-genome Hamming distance. Hamming calculations sample at most the first `64` deduplicated candidates to bound diagnostic cost.

The campaign records three separate diagnostic scopes:
- active population: selected novel candidates from the current proposal events. Restart and convergence policy use this scope.
- measured-new population: candidates represented in newly planned schedule batches.
- archive: selected preserved parents, reported for historical coverage only.

Carried archive entries never inflate active diversity or restart/convergence decisions.

A restart requires both:
- a best-history plateau under the configured patience and minimum improvement fraction.
- mean Hamming distance no greater than the configured restart threshold.

During isolation, the affected island can receive a fresh covering allocation. After migration, one part of a feedback round can become a separately tagged restart island while the other part continues merged feedback. Campaign state stores one restart epoch counter per island plus one merged counter. A counter increments only when a restart transition is admitted. Ordinary rounds retain the current epoch without incrementing it. Restarts never shrink `DOMAINS` or turn family failures into validity rules.

## Round Admission

The one-shape driver converts up to six recent completed rounds into duration and exact requested-pair observations. `estimate_admission_duration_s()` in the shared campaign controller computes:

```text
seconds_per_requested_pair = round_duration_s / requested_pairs
```

It then uses the median plus median absolute deviation as a robust per-pair estimate, scales by expected requested pairs, applies a `1.15` margin, adds `5s`, and enforces a minimum estimate.

The driver treats `--time-budget` as a soft admission budget. It admits a new round only when the round estimate fits before the search admission deadline. Once admitted, the schedule runs normally in bounded prepare->serial-time waves: each admitted wave drains all preparation before timing, and build/runner subprocesses retain their configured timeouts. The current one-shape round normally fits in one profile-sized wave. A production coordinator can inspect feedback and the soft budget before admitting a later wave. An admitted wave may finish after the soft deadline. No later round or stabilization group is admitted afterward.

The confirmation reserve is the greater of the configured floor and a launch-cost estimate for the currently ranked finalists. The estimate uses each finalist's measured median kernel time, the complete hot protocol launch count, per-finalist startup allowance, and a duration margin. Search admission recalculates this reserve before every round, so slow finalists can reserve more than the default `60s`. Hot finalists are likewise admitted only before the total soft deadline, but an admitted finalist receives the full runner timeout and may complete afterward.

An external process timeout is an operational guard, not the campaign budget. Set it reasonably above `--time-budget` so one admitted round, hot confirmation, artifact writes, and process cleanup can finish. For the default `1200s` campaign, use at least `1800s` unless measured worst-case round and confirmation costs justify a larger value.

## Checkpoints

`CampaignStore` writes `campaign_checkpoint.json` atomically at three phase boundaries:
- `proposed`: exact candidate dictionaries and hashes are already materialized in `round_NN/proposals.json`.
- `completed`: the round record and DB evidence are durable.
- `finished`: campaign summary and hot-confirmation attempt are complete.

Checkpoint metadata nests the complete `CampaignControllerState`, which owns phase, round index, elapsed accounting, reserves, exact pair state, incumbents, artifact coverage, singleton mechanical clustering, staged-round state, phase costs, and trace. Round seed, exact pending hashes, configuration hash, and restart counters remain beside it. Candidate dictionaries are reconstructed on resume and every hash is verified before scheduling. The current code consumes the current internal layout directly and provides no compatibility coercion for earlier development layouts.

When resuming a `proposed` round, cache-aware planning reuses any build, validation, probe, or timing evidence already inserted before interruption. It never generates replacement candidates for the pending round.

An abrupt termination during an active round can occur between elapsed-time checkpoints. Exact proposals and DB evidence remain recoverable, but uncheckpointed wall time may require an explicit audit amendment if strict total-time accounting is required. The harness does not silently invent that elapsed interval.

## Determinism

Round seeds are derived from the campaign seed and round index. Island proposal calls use deterministic seed offsets. Proposal files store full candidate dictionaries, source, parent hashes, proposal metadata, and call parameters.

The surrogate is refit from the checkpointed DB, and operator/group/donor credit is recomputed from queried DB evidence. No opaque model state is required for exact pending-proposal resume because pending candidates are loaded from disk rather than regenerated.

## Convergence Stop

`--early-stop-on-convergence` is opt-in. The detector requires both:
- no sufficient best-so-far improvement over the configured patience window.
- low mean Hamming diversity.

The helper defaults are eight rounds, `0.25%` minimum improvement, and mean Hamming distance at most `4.0`. The campaign's restart thresholds are separate policy values.

The current experiment log keeps convergence stopping disabled because observed campaigns made important late improvements while population diversity remained high. Evidence and conclusions belong in `docs/experiment_blind_one_shape.md`.

## Replay Relationship

Historical replay models covering selection, hash-partitioned islands, stabilization costs, population diagnostics, and optional convergence stopping. It remains an approximation:
- it can expose only historical exact hashes.
- replay island partitioning does not reproduce real generated-parent genealogy.
- preparation uses a calibrated wave cost rather than executing TensileLite.

See `docs/blind_experiment_infrastructure.md` for proof eligibility and query-causal oracle rules.

## Artifacts

Campaign-control artifacts include:
- `campaign_configuration.json`.
- `campaign_checkpoint.json`.
- `campaign_progress.json`.
- `round_NN/proposals.json` with typed proposal events and explicit active/archive hashes.
- per-round active, measured-new, archive, leader, schedule, and stabilization summaries.
- `campaign_summary.json`.

## Limitations

- The controller and runner are shared package code, but the current proposal profile still supports only the singleton degeneration.
- Island identity is proposal metadata rather than a dedicated database table.
- Restart and convergence thresholds have not received a full equal-time multi-seed ablation.
- Hard process termination cannot atomically checkpoint in-flight subprocess duration.
- The campaign reconstructs adaptive state from SQLite rather than persisting model objects.
