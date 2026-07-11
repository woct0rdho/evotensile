# Blind Campaign Control

This document describes the campaign-specific state machine implemented by `scripts/run_blind_one_shape.py` and helpers in `evotensile/search/campaign_control.py`. The broader blindness, replay, and artifact contract is documented in `docs/blind_experiment_infrastructure.md`. Search operators remain documented in the search-prefixed design docs.

## Scope

The campaign driver is a fixed-policy one-shape experiment harness. It is not a general multi-island CLI framework and does not change the default behavior of ordinary `schedule-batches` runs.

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

A new campaign writes `campaign_configuration.json` before measurement. One immutable versioned configuration owns:
- seed, shape, profile, problem identity, proposal mode/counts/operator controls, linkage settings, and profile mutation rates.
- screening, validation, probe, adaptive-retiming, stabilization, and hot-confirmation protocols.
- sample limits, round limits, restart/convergence controls, soft search budget, and confirmation reserve.
- batch sizes, compile threads, preparation/validation concurrency, cache behavior, timeouts, and explicit physical-CU/WGP topology.
- absolute runner and TensileLite paths, SHA-256 content/source fingerprints, the EvoTensile implementation fingerprint, and behavior-affecting environment variables.

The current defaults remain `48` cold candidates, `24` requested feedback candidates, six isolated rounds, `16` island-local elites, `32` merged elites, an `8x` pool, `32` preparation workers, a `32`-batch preparation wave, one validation worker, and a `60s` confirmation reserve.

Resume reconstructs the complete configuration from the current invocation and rejects any field, binary, implementation, or environment mismatch before reading campaign state. There is no partial identity or compatibility override.

## Proposal Events And Candidate Metadata

The scheduler returns explicit preserved, novel-generated, and selected candidate sets. `tag_generated_proposals()` adds `island_id`, `restart_index`, and per-generated-hash proposal cost only to selected novel candidates. Preserved archive parents and previously registered duplicate hashes are returned unchanged.

Each proposal call also creates an immutable event containing breeding-parent hashes, preserved hashes, all generated hashes before shortlisting, selected hashes, duration, proposal arguments, island, and restart identity. The event is persisted in `round_NN/proposals.json` independently from candidate identity.

Parent hashes, semantic-group metadata, and GOMEA donor metadata remain candidate-origin metadata. `load_island_elites()` ranks compatible validation-passed timing evidence, restores candidates from SQLite, and returns only candidates originally generated for the requested island.

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

`estimate_next_round_duration_s()` uses up to six recent completed rounds. For each usable round it computes:

```text
seconds_per_missing_pair = round_duration_s / missing_pairs
```

It then uses the median plus median absolute deviation as a robust per-pair estimate, scales by expected missing pairs, applies a `1.15` margin, adds `5s`, and enforces a minimum estimate.

The driver treats `--time-budget` as a soft admission budget. It admits a new round only when the round estimate fits before the search admission deadline. Once admitted, the schedule runs normally in bounded prepare→serial-time waves: each admitted wave drains all preparation before timing, and build/runner subprocesses retain their configured timeouts. The current one-shape round normally fits in one profile-sized wave. A production coordinator can inspect feedback and the soft budget before admitting a later wave. An admitted wave may finish after the soft deadline. No later round or stabilization group is admitted afterward.

The confirmation reserve is the greater of the configured floor and a launch-cost estimate for the currently ranked finalists. The estimate uses each finalist's measured median kernel time, the complete hot protocol launch count, per-finalist startup allowance, and a duration margin. Search admission recalculates this reserve before every round, so slow finalists can reserve more than the default `60s`. Hot finalists are likewise admitted only before the total soft deadline, but an admitted finalist receives the full runner timeout and may complete afterward.

An external process timeout is an operational guard, not the campaign budget. Set it reasonably above `--time-budget` so one admitted round, hot confirmation, artifact writes, and process cleanup can finish. For the default `1200s` campaign, use at least `1800s` unless measured worst-case round and confirmation costs justify a larger value.

## Checkpoints

The driver writes `campaign_checkpoint.json` atomically at three phase boundaries:
- `proposed`: exact candidate dictionaries and hashes are already materialized in `round_NN/proposals.json`.
- `completed`: the round record and DB evidence are durable.
- `finished`: campaign summary and hot-confirmation attempt are complete.

Checkpoint metadata includes round index, round seed, exact pending hashes, the complete configuration hash, restart counters, and elapsed accounting. Candidate dictionaries are reconstructed on resume and every hash is verified before scheduling.

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

The current experiment log keeps convergence stopping disabled because observed campaigns made important late improvements while population diversity remained high. Evidence and conclusions belong in `docs/blind_one_shape_experiment.md`.

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

- Multi-island control exists only in the one-shape driver.
- Island identity is proposal metadata rather than a dedicated database table.
- Restart and convergence thresholds have not received a full equal-time multi-seed ablation.
- Hard process termination cannot atomically checkpoint in-flight subprocess duration.
- The campaign reconstructs adaptive state from SQLite rather than persisting model objects.
