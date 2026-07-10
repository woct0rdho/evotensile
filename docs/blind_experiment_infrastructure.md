# Blind Experiment Infrastructure

This document describes the infrastructure for blind EvoTensile search experiments, including real wall-time campaigns, exact-hash historical replay, simulated time accounting, finalist confirmation, and audit artifacts. The one-shape campaign state machine is documented in `docs/blind_campaign_control.md`. Individual experiment outcomes belong in logs such as `docs/blind_one_shape_experiment.md`.

## Blindness Contract

A blind experiment must not use an external winner's hash, exact parameter bundle, or hindsight-derived interaction groups in:
- production proposal code.
- search defaults or repair ordering.
- initial candidates or imported parents.
- tests used to define search behavior.
- surrogate or linkage training before an exact query.
- campaign bookkeeping that changes proposals.

Allowed information includes:
- broad parameter domains.
- exact source-backed validity rules.
- generic mechanical semantic groups.
- performance rows produced by candidates already queried in the active campaign.
- an external performance number used only as a final threshold.
- historical measurements returned by an exact-hash simulated oracle after the query.

Unknown oracle candidates remain unknown. Simulation must not impute their performance from a hidden control candidate.

## Real Campaign Driver

`scripts/run_blind_one_shape.py` runs the current one-shape policy against an empty campaign DB.

The driver provides:
- a fixed seed and recorded frozen policy.
- two independently seeded, mechanically covering cold populations.
- island-local feedback rounds followed by explicit population migration.
- repeated family-QD feedback rounds with optional low-diversity restarts.
- adaptive operator allocation and surrogate oversized pools.
- normal cache-aware compile, validation, probe, and screening execution.
- cost-aware round admission using robust recent duration per missing pair.
- a reserved finalist-confirmation budget.
- per-round proposal lineage and execution summaries.
- atomic exact-proposal checkpoints and hash-verified `--resume` under the original invocation contract.
- a final campaign summary.

The current policy uses `48` measured cold candidates selected from two complementary `8x` island pools, then requests `24` generated candidates per feedback round. The first six feedback rounds keep parents island-local. Later rounds use the merged archive. Existing parents may also appear in the proposal list, but cache-aware planning measures only missing pairs. Preparation uses eight workers while GPU validation is capped at one worker after concurrent validation exposed a reproducible ROCr/KFD operational failure.

The driver uses a staged catastrophic probe: one launch for every validated pair, followed by two more only for provisional survivors, then the two-sample screening protocol. It also stabilizes a bounded set of provisional main-protocol leaders between rounds. Probe math is documented in `docs/noisy_measurements.md`. Stabilization is documented in `docs/search_screening_stabilization.md`. Neither changes TensileLite validity or validation behavior.

## Exact-Hash Replay Oracle

`evotensile/search/replay.py` loads historical evidence into `OracleRecord` objects keyed by canonical candidate hash.

Sources can include:
- EvoTensile SQLite databases filtered by shape and benchmark protocol.
- CSV files with candidate parameter JSON and measured performance.
- hot-loop summaries keyed by candidate hash or candidate label.

`merge_oracle_records()` deduplicates exact hashes and attaches hot-loop measurements when available.

The simulator discloses a row only after the search queries that exact candidate hash. If the hash is absent from the oracle, the query is recorded as unknown and no timing evidence is inserted into the simulated campaign DB.

Historical directed/control candidates may be used as hidden exact-query responses. Their candidate sequence must not be exposed as a proof-eligible proposal stream.

## Proof-Eligible And Diagnostic Streams

`scripts/simulate_blind_search.py` distinguishes two replay uses:
- Proof-eligible replay: the visible stream comes from blind historical campaigns. The algorithm may reorder or shortlist only candidates that those blind campaigns generated.
- Diagnostic pool: the visible stream may contain directed or control candidates. This can test whether selection would recognize a good candidate if proposed, but it is not evidence that blind search can generate that candidate.

CSV candidate streams require `--diagnostic-pool`. Output records `proof_eligible=false` for those runs.

## Simulated Time Model

`ReplayCostModel` accounts for:
- parallel preparation waves.
- one initial probe launch per candidate plus two additional launches for provisional survivors.
- main screening launches for probe survivors.
- optional provisional-leader stabilization samples.
- a reserved hot-confirmation budget.
- final hot-loop launches.

Preparation wall time is modeled as:

```text
ceil(selected_candidates / prepare_workers) * prepare_seconds_per_candidate
```

Launch cost is derived from shape FLOPs and the exact measured GFLOP/s of the queried candidate.

The simulator applies the same coarse probe policy used by production search:
- screen candidates outside the configured slowdown factor after the initial launch.
- retain a minimum survivor count.
- charge remaining probe launches only to provisional survivors.
- keep probe-only rows out of the main training DB.
- insert main screening evidence only for complete probe survivors.

Optional replay controls model covering cold selection, deterministic hash-partitioned islands, isolation duration, leader-stabilization cost, population diagnostics, and convergence stopping. These reproduce policy/accounting behavior over a fixed historical stream. They do not reproduce the real generator's parent genealogy.

Search stops before the confirmation reserve. Finalists are ranked from queried main-protocol evidence and hot-confirmed only when an exact hot measurement exists and simulated time remains.

## Query-Causal Search State

Each replay seed uses a fresh temporary SQLite database. Proposal shortlisting can consume only rows inserted earlier in that replay.

This preserves causal ordering for:
- surrogate training.
- family archive ranking.
- leader stabilization.
- any credit or diagnostics derived from inserted replay evidence.
- best-so-far traces.

The oracle object itself can contain future answers, but proposal code receives only candidates and the simulated DB evidence already disclosed by prior queries.

## Hot-Loop Confirmation

`hot_confirm_topk()` and `scripts/hot_confirm_topk.py` confirm validation-passed screening finalists without recompiling or revalidating them.

The helper:
- ranks candidates under the screening protocol.
- requires compatible passed validation evidence.
- finds the generated library and mapped pair from run metadata.
- reuses the existing library artifact.
- runs `20` warmups and `10` samples with `10` enqueues per sample.
- writes JSON and CSV rankings.

Benchmark mode sets validation extent to zero and relies on prior validation evidence. Confirmation is for final performance claims. Two-sample screening is only exploration evidence.

## Audit Artifacts

Real campaigns write:
- `frozen_policy.json`.
- `campaign.sqlite`.
- `round_NN/proposals.json` with exact parameters, source, parent hashes, proposal metadata, and proposal-call settings.
- per-round build, validation, probe, and benchmark artifacts.
- `campaign_progress.json`.
- `campaign_checkpoint.json` with phase, exact pending hashes, seeds, policy, and elapsed accounting.
- `campaign_summary.json`.
- `hot_loop_top8/summary.json` and `ranked.csv`.

Replay writes:
- cost-model and enabled-policy parameters.
- proof eligibility.
- query, unknown, screened, and survivor counts.
- simulated elapsed time and stop reason.
- per-round best-so-far, stabilization, and population-diagnostic traces.
- hot-confirmed result and threshold outcome when configured.

Long real runs should additionally capture GPU busy, power, temperature, memory, and relevant process counts. Monitoring is external to the proposal algorithm and must not change the candidate sequence.

## Example Commands

Proof-eligible replay:

```bash
python scripts/simulate_blind_search.py \
  --oracle-db <blind-campaign.sqlite> \
  --stream-db <blind-campaign.sqlite> \
  --hot-summary <hot-summary.json> \
  --protocol-hash <benchmark-protocol-hash> \
  --seed 1 --seed 2 --seed 3 \
  --time-budget 1200 \
  --output out/replay/results.json
```

Real campaign:

```bash
python scripts/run_blind_one_shape.py \
  --output out/blind-seed-1 \
  --shape 8192,8192,1,8192 \
  --seed 1 \
  --time-budget 1200 \
  --runner-bin build/evotensile-structured-runner
```

## Limitations

- Exact replay can score only historical candidate hashes.
- A historical candidate stream evaluates selection policy, not the ability of a new generator to produce unseen candidates.
- Simulated preparation cost is configurable and should be calibrated from comparable real campaigns.
- Screening evidence can be noisy. Simulated replay cannot reconstruct unrecorded timing distributions.
- A successful diagnostic pool is not a successful blind experiment.
- Repeated real runs after observing a threshold result must be labeled transparently if policy or budget allocation changed.
- Exact pending proposals survive resume, but an abrupt mid-round termination can leave an uncheckpointed wall-time interval. `docs/blind_campaign_control.md` documents the required audit treatment.
