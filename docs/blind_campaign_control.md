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

## Frozen Policy

A new campaign writes `frozen_policy.json` before measurement. The current one-shape policy includes:
- `48` cold candidates from two coordinated covering pools.
- `24` requested feedback candidates per round.
- six island-isolated feedback rounds.
- `16` island-local elites and `32` merged elites.
- an `8x` proposal pool.
- leader stabilization and a `60s` hot-confirmation reserve.
- eight preparation workers, one validation worker, and serial benchmarking.

CLI overrides for budget, reserve, round safety limit, and leader stabilization are incorporated into the frozen record. Resume checks seed, shape, and profile identity before using an existing root.

## Proposal Metadata

`tag_proposals()` copies candidates with parameter identity unchanged and persists:
- `island_id`.
- `restart_index`.
- per-generated-candidate proposal wall-time share.

Parent hashes, semantic-group metadata, and GOMEA donor metadata are preserved. `load_island_elites()` ranks compatible validation-passed timing evidence, restores candidates from SQLite, and returns only candidates tagged with the requested island.

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

`population_diagnostics()` reports:
- deduplicated candidate count.
- family-cell count.
- distinct complete MatrixInstruction count for audit.
- mechanical-token count.
- mean and minimum categorical-genome Hamming distance.

Hamming calculations sample at most the first `64` deduplicated candidates to bound diagnostic cost.

A restart requires both:
- a best-history plateau under the configured patience and minimum improvement fraction.
- mean Hamming distance no greater than the configured restart threshold.

During isolation, the affected island can receive a fresh covering allocation. After migration, one part of a feedback round can become a separately tagged restart island while the other part continues merged feedback. Restarts never shrink `DOMAINS` or turn family failures into validity rules.

## Round Admission

`estimate_next_round_duration_s()` uses up to six recent completed rounds. For each usable round it computes:

```text
seconds_per_missing_pair = round_duration_s / missing_pairs
```

It then uses the median plus median absolute deviation as a robust per-pair estimate, scales by expected missing pairs, applies a `1.15` margin, adds `5s`, and enforces a minimum estimate.

The driver admits a new round only when that estimate fits before the search deadline. Search and hot confirmation have separate deadlines so feedback cannot consume the reserved final measurement budget.

## Checkpoints

The driver writes `campaign_checkpoint.json` atomically at three phase boundaries:
- `proposed`: exact candidate dictionaries and hashes are already materialized in `round_NN/proposals.json`.
- `completed`: the round record and DB evidence are durable.
- `finished`: campaign summary and hot-confirmation attempt are complete.

Checkpoint metadata includes round index, round seed, exact pending hashes, frozen policy, and elapsed accounting. Candidate dictionaries are reconstructed on resume and every hash is verified before scheduling.

When resuming a `proposed` round, cache-aware planning reuses any build, validation, probe, or timing evidence already inserted before interruption. It never generates replacement candidates for the pending round.

Resume currently verifies seed, shape, profile, and every pending candidate hash. It does not yet reject changed time budget, maximum-round limit, runner/build timeouts, binary paths, or early-stop activation. Strict continuation therefore requires reusing the original invocation recorded by the external control log until those fields become enforced resume identity.

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
- `frozen_policy.json`.
- `campaign_checkpoint.json`.
- `campaign_progress.json`.
- `round_NN/proposals.json`.
- per-round population, leader, proposal-call, schedule, and stabilization summaries.
- `campaign_summary.json`.

## Limitations

- Multi-island control exists only in the one-shape driver.
- Island identity is proposal metadata rather than a dedicated database table.
- Restart and convergence thresholds have not received a full equal-time multi-seed ablation.
- Hard process termination cannot atomically checkpoint in-flight subprocess duration.
- Resume identity does not yet enforce every operational CLI option. Control logs remain part of strict audit provenance.
- Early-stop activation is not currently persisted as a frozen-policy field unless it triggers the recorded `stop_reason`.
- The campaign reconstructs adaptive state from SQLite rather than persisting model objects.
