import csv
import json
import math
import sqlite3
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from evotensile.adaptive_retime import load_timing_stats
from evotensile.candidate import Candidate, Shape
from evotensile.database import BenchmarkEventInsert, EvoTensileDB, ValidationInsert
from evotensile.metrics import gflops_from_us
from evotensile.profile import DEFAULT_PROFILE
from evotensile.protocol import BenchmarkProtocol
from evotensile.search.campaign_control import convergence_detected, population_diagnostics, split_budget
from evotensile.search.evidence import load_proposal_evidence_snapshot
from evotensile.search.screening_stabilize import ScreeningStabilizationPolicy, plan_screening_stabilization
from evotensile.search.surrogate import select_surrogate_pool


@dataclass(frozen=True)
class OracleRecord:
    candidate: Candidate
    status: str
    screening_gflops: float | None = None
    hot_gflops: float | None = None
    order: float = 0.0
    source_artifact: str = ""


@dataclass(frozen=True)
class ReplayCostModel:
    time_budget_s: float = 1200.0
    prepare_workers: int = 4
    prepare_seconds_per_candidate: float = 8.0
    probe_launches: int = 3
    initial_probe_launches: int = 1
    screening_launches: int = 3
    hot_launches: int = 120
    hot_reserve_s: float = 60.0
    probe_max_slowdown_factor: float = 4.0
    probe_min_survivors: int = 8
    stabilization_protocol: BenchmarkProtocol = field(
        default_factory=lambda: BenchmarkProtocol(
            num_warmups=1,
            num_benchmarks=2,
            enqueues_per_sync=1,
            syncs_per_benchmark=1,
        )
    )
    stabilization_policy: ScreeningStabilizationPolicy = field(default_factory=ScreeningStabilizationPolicy)


@dataclass
class ReplayResult:
    seed: int
    queried: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    screened: list[str] = field(default_factory=list)
    screening_survivors: list[str] = field(default_factory=list)
    simulated_time_s: float = 0.0
    best_screening_hash: str | None = None
    best_screening_gflops: float | None = None
    best_hot_hash: str | None = None
    best_hot_gflops: float | None = None
    reached_target: bool = False
    stop_reason: str | None = None
    trace: list[dict[str, object]] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "queried": len(self.queried),
            "unknown": len(self.unknown),
            "screened": len(self.screened),
            "screening_survivors": len(self.screening_survivors),
            "simulated_time_s": self.simulated_time_s,
            "best_screening_hash": self.best_screening_hash,
            "best_screening_gflops": self.best_screening_gflops,
            "best_hot_hash": self.best_hot_hash,
            "best_hot_gflops": self.best_hot_gflops,
            "reached_target": self.reached_target,
            "stop_reason": self.stop_reason,
            "trace": self.trace,
        }


def load_db_oracle(
    path: str | Path,
    *,
    shape: Shape,
    benchmark_protocol_hash: str | None = None,
) -> list[OracleRecord]:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    protocol_clause = "AND bp.benchmark_protocol_hash = ?" if benchmark_protocol_hash is not None else ""
    params: list[str] = [shape.id]
    if benchmark_protocol_hash is not None:
        params.append(benchmark_protocol_hash)
    rows = con.execute(
        f"""
        SELECT c.params_json, c.created_at, c.candidate_hash, be.status, bs.time_us
        FROM benchmark_events AS be
        LEFT JOIN benchmark_samples AS bs USING (event_id)
        JOIN candidates AS c USING (candidate_id)
        JOIN shapes AS s USING (shape_key)
        JOIN benchmark_namespaces AS bn USING (benchmark_namespace_id)
        JOIN benchmark_protocols AS bp USING (benchmark_protocol_id)
        WHERE s.shape_id = ? {protocol_clause}
        ORDER BY c.created_at, be.event_id, bs.sample_index
        """,
        params,
    ).fetchall()
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    payloads: dict[str, str] = {}
    order: dict[str, float] = {}
    for row in rows:
        candidate_hash = str(row["candidate_hash"])
        grouped[candidate_hash].append(row)
        payloads[candidate_hash] = str(row["params_json"])
        order[candidate_hash] = float(row["created_at"])
    records = []
    for candidate_hash, candidate_rows in grouped.items():
        times = [float(row["time_us"]) for row in candidate_rows if row["status"] == "ok" and row["time_us"]]
        if times:
            screening_gflops = gflops_from_us(shape, statistics.median(times))
            status = "ok"
        else:
            screening_gflops = None
            statuses = [str(row["status"]) for row in candidate_rows]
            status = statuses[-1]
        records.append(
            OracleRecord(
                candidate=Candidate(params=dict(json.loads(payloads[candidate_hash])), source="historical_replay"),
                status=status,
                screening_gflops=screening_gflops,
                order=order[candidate_hash],
                source_artifact=str(path),
            )
        )
    con.close()
    return records


def load_csv_oracle(path: str | Path, *, order_offset: float = 0.0) -> list[OracleRecord]:
    records = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            params_json = row.get("params_json")
            if not params_json:
                continue
            candidate = Candidate(
                params=json.loads(params_json),
                source=row.get("source") or "oracle",
                parent_hashes=(),
            )
            value = row.get("median_gflops")
            screening_gflops = None if value is None or value in {"", "-1.0"} else float(value)
            status = "ok" if screening_gflops is not None and screening_gflops > 0.0 else row.get("status", "unknown")
            records.append(
                OracleRecord(
                    candidate=candidate,
                    status=status,
                    screening_gflops=screening_gflops,
                    order=order_offset + index,
                    source_artifact=str(path),
                )
            )
    return records


def load_hot_summary(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("ranked", payload) if isinstance(payload, dict) else payload
    measurements = {}
    for row in rows:
        candidate_hash = row.get("candidate_hash")
        if not candidate_hash and str(row.get("label", "")).startswith("cand_"):
            candidate_hash = str(row["label"]).split("_MT", 1)[0]
        if candidate_hash and row.get("median_gflops") is not None:
            measurements[str(candidate_hash)] = float(row["median_gflops"])
    return measurements


def merge_oracle_records(
    record_groups: Sequence[Sequence[OracleRecord]],
    *,
    hot_measurements: dict[str, float] | None = None,
) -> dict[str, OracleRecord]:
    merged: dict[str, OracleRecord] = {}
    hot = hot_measurements or {}
    for records in record_groups:
        for record in records:
            existing = merged.get(record.candidate.hash)
            screening = record.screening_gflops
            if existing is not None and existing.screening_gflops is not None:
                screening = (
                    existing.screening_gflops if screening is None else max(existing.screening_gflops, screening)
                )
            merged[record.candidate.hash] = OracleRecord(
                candidate=record.candidate,
                status="ok" if screening is not None else record.status,
                screening_gflops=screening,
                hot_gflops=hot.get(record.candidate.hash, existing.hot_gflops if existing else None),
                order=min(record.order, existing.order) if existing else record.order,
                source_artifact=record.source_artifact,
            )
    for candidate_hash, hot_gflops in hot.items():
        if candidate_hash in merged:
            record = merged[candidate_hash]
            merged[candidate_hash] = OracleRecord(
                candidate=record.candidate,
                status=record.status,
                screening_gflops=record.screening_gflops,
                hot_gflops=hot_gflops,
                order=record.order,
                source_artifact=record.source_artifact,
            )
    return merged


def _launch_seconds(shape: Shape, gflops: float) -> float:
    return 2.0 * shape.m * shape.n * shape.batch * shape.k / (gflops * 1e9)


def _insert_screening_evidence(
    db: EvoTensileDB,
    *,
    shape: Shape,
    record: OracleRecord,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    source_ref: str,
    samples: int = 2,
) -> None:
    if record.screening_gflops is None or record.screening_gflops <= 0.0:
        return
    time_us = 2.0 * shape.m * shape.n * shape.batch * shape.k / (record.screening_gflops * 1e3)
    validation_protocol_hash = DEFAULT_PROFILE.default_protocol.validation_protocol_hash()
    db.insert_validations(
        [
            ValidationInsert(
                shape_id=shape.id,
                candidate_hash=record.candidate.hash,
                run_id=source_ref,
                status="passed",
                problem_type_hash=problem_type_hash,
                validation_protocol_hash=validation_protocol_hash,
                source_kind="replay",
            )
        ]
    )
    db.insert_benchmark_events(
        [
            BenchmarkEventInsert(
                shape_id=shape.id,
                candidate_hash=record.candidate.hash,
                run_id=source_ref,
                status="ok",
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                source_kind="replay",
                samples_us=(time_us,) * samples,
                validation_protocol_hash=validation_protocol_hash,
            )
        ]
    )


def _candidate_island(candidate: Candidate, island_count: int) -> int:
    return int(candidate.hash.removeprefix("cand_")[:8], 16) % max(1, island_count)


def _select_replay_batch(
    pending: Sequence[Candidate],
    *,
    db: EvoTensileDB,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    shape: Shape,
    count: int,
    seed: int,
    min_evidence: int,
    covering_cold_start: bool,
    island_count: int,
    isolated: bool,
) -> list[Candidate]:
    evidence = load_proposal_evidence_snapshot(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        shapes=[shape],
    )
    if not isolated or island_count <= 1:
        return select_surrogate_pool(
            pending,
            evidence=evidence,
            shapes=[shape],
            count=min(count, len(pending)),
            seed=seed,
            min_evidence=min_evidence,
            covering_cold_start=covering_cold_start,
            surrogate_jobs=1,
            workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
        )
    selected: list[Candidate] = []
    budgets = split_budget(count, island_count)
    for island_index, budget in enumerate(budgets):
        pool = [candidate for candidate in pending if _candidate_island(candidate, island_count) == island_index]
        if not pool or budget <= 0:
            continue
        selected.extend(
            select_surrogate_pool(
                pool,
                evidence=evidence,
                shapes=[shape],
                count=min(budget, len(pool)),
                seed=seed + island_index * 1_000_003,
                min_evidence=min_evidence,
                covering_cold_start=covering_cold_start,
                surrogate_jobs=1,
                workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
            )
        )
    if len(selected) < min(count, len(pending)):
        remaining = [candidate for candidate in pending if candidate.hash not in {item.hash for item in selected}]
        selected.extend(remaining[: min(count, len(pending)) - len(selected)])
    return selected


def simulate_candidate_stream(
    stream: Sequence[Candidate],
    *,
    oracle: Mapping[str, OracleRecord],
    shape: Shape,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    cost: ReplayCostModel,
    seed: int,
    batch_size: int = 32,
    pool_window: int = 128,
    surrogate_min_evidence: int = 24,
    hot_finalists: int = 8,
    target_hot_gflops: float | None = None,
    covering_cold_start: bool = False,
    island_count: int = 1,
    island_isolation_rounds: int = 0,
    leader_stabilization: bool = False,
    early_stop_on_convergence: bool = False,
) -> ReplayResult:
    result = ReplayResult(seed=seed)
    with tempfile.TemporaryDirectory(prefix="evotensile-replay-") as directory:
        db = EvoTensileDB.connect(
            Path(directory) / "replay.sqlite",
            environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
        )
        db.init()
        db.register_shapes([shape])
        replay_source_ref = f"simulated_stream_seed_{seed}"
        pending: dict[str, Candidate] = {}
        stream_index = 0
        queried: set[str] = set()
        round_index = 0
        incumbent_gflops: float | None = None
        best_history: list[float] = []
        search_time_limit = max(0.0, cost.time_budget_s - max(0.0, cost.hot_reserve_s))
        budget_exhausted = False
        while result.simulated_time_s < search_time_limit and (stream_index < len(stream) or pending):
            while stream_index < len(stream) and len(pending) < pool_window:
                candidate = stream[stream_index]
                stream_index += 1
                if candidate.hash not in queried:
                    pending.setdefault(candidate.hash, candidate)
            if not pending:
                break
            selected = _select_replay_batch(
                list(pending.values()),
                db=db,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                shape=shape,
                count=min(batch_size, len(pending)),
                seed=seed + round_index,
                min_evidence=surrogate_min_evidence,
                covering_cold_start=covering_cold_start and round_index == 0,
                island_count=island_count,
                isolated=round_index <= island_isolation_rounds,
            )
            preparation_cost = (
                math.ceil(len(selected) / max(1, cost.prepare_workers)) * cost.prepare_seconds_per_candidate
            )
            if result.simulated_time_s + preparation_cost > search_time_limit:
                result.stop_reason = "insufficient_prepare_budget"
                break
            result.simulated_time_s += preparation_cost
            known_ok = []
            for candidate in selected:
                pending.pop(candidate.hash, None)
                queried.add(candidate.hash)
                result.queried.append(candidate.hash)
                db.register_candidates([candidate])
                record = oracle.get(candidate.hash)
                if record is None:
                    result.unknown.append(candidate.hash)
                    continue
                if record.screening_gflops is not None and record.screening_gflops > 0.0:
                    initial_probe_launches = min(
                        max(0, cost.initial_probe_launches),
                        max(0, cost.probe_launches),
                    )
                    initial_probe_cost = initial_probe_launches * _launch_seconds(shape, record.screening_gflops)
                    if result.simulated_time_s + initial_probe_cost > search_time_limit:
                        budget_exhausted = True
                        break
                    result.simulated_time_s += initial_probe_cost
                    known_ok.append(record)
            if known_ok:
                reference = max(
                    [record.screening_gflops or 0.0 for record in known_ok]
                    + ([incumbent_gflops] if incumbent_gflops is not None else [])
                )
                threshold = reference / cost.probe_max_slowdown_factor
                ranked = sorted(known_ok, key=lambda record: record.screening_gflops or 0.0, reverse=True)
                minimum_hashes = {record.candidate.hash for record in ranked[: cost.probe_min_survivors]}
                provisional_survivors = [
                    record
                    for record in known_ok
                    if (record.screening_gflops or 0.0) >= threshold or record.candidate.hash in minimum_hashes
                ]
                provisional_hashes = {record.candidate.hash for record in provisional_survivors}
                for record in known_ok:
                    if record.candidate.hash not in provisional_hashes:
                        result.screened.append(record.candidate.hash)
                survivors = []
                additional_probe_launches = max(
                    0,
                    cost.probe_launches - min(max(0, cost.initial_probe_launches), max(0, cost.probe_launches)),
                )
                for record in provisional_survivors:
                    additional_probe_cost = additional_probe_launches * _launch_seconds(
                        shape,
                        record.screening_gflops or 0.0,
                    )
                    if result.simulated_time_s + additional_probe_cost > search_time_limit:
                        budget_exhausted = True
                        break
                    result.simulated_time_s += additional_probe_cost
                    survivors.append(record)
                for record in survivors:
                    if record.screening_gflops is None:
                        continue
                    screening_cost = cost.screening_launches * _launch_seconds(shape, record.screening_gflops)
                    if result.simulated_time_s + screening_cost > search_time_limit:
                        budget_exhausted = True
                        break
                    result.simulated_time_s += screening_cost
                    result.screening_survivors.append(record.candidate.hash)
                    _insert_screening_evidence(
                        db,
                        shape=shape,
                        record=record,
                        problem_type_hash=problem_type_hash,
                        benchmark_protocol_hash=benchmark_protocol_hash,
                        source_ref=replay_source_ref,
                    )
                    if incumbent_gflops is None or record.screening_gflops > incumbent_gflops:
                        incumbent_gflops = record.screening_gflops
                        result.best_screening_hash = record.candidate.hash
                        result.best_screening_gflops = record.screening_gflops
            stabilized_candidates = 0
            stabilization_samples = 0
            if leader_stabilization:
                stats_by_shape = load_timing_stats(
                    db,
                    problem_type_hash=problem_type_hash,
                    benchmark_protocol_hashes=[benchmark_protocol_hash],
                    min_samples=1,
                    shape_ids={shape.id},
                )
                stabilization_plan = plan_screening_stabilization(
                    stats_by_shape,
                    shapes=[shape],
                    protocol=cost.stabilization_protocol,
                    policy=cost.stabilization_policy,
                )
                for request in stabilization_plan.requests:
                    record = oracle.get(request.candidate_hash)
                    if record is None or record.screening_gflops is None or record.screening_gflops <= 0.0:
                        continue
                    launch_seconds = _launch_seconds(shape, record.screening_gflops)
                    topup_cost = (request.remaining_samples + 1) * launch_seconds
                    if result.simulated_time_s + topup_cost > search_time_limit:
                        continue
                    result.simulated_time_s += topup_cost
                    _insert_screening_evidence(
                        db,
                        shape=shape,
                        record=record,
                        problem_type_hash=problem_type_hash,
                        benchmark_protocol_hash=benchmark_protocol_hash,
                        source_ref=replay_source_ref,
                        samples=request.remaining_samples,
                    )
                    stabilized_candidates += 1
                    stabilization_samples += request.remaining_samples
            if result.best_screening_gflops is not None:
                best_history.append(result.best_screening_gflops)
            diagnostics = population_diagnostics(
                selected,
                shape,
                workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
            )
            result.trace.append(
                {
                    "round": round_index,
                    "queried": len(result.queried),
                    "unknown": len(result.unknown),
                    "screened": len(result.screened),
                    "simulated_time_s": result.simulated_time_s,
                    "best_screening_hash": result.best_screening_hash,
                    "best_screening_gflops": result.best_screening_gflops,
                    "stabilized_candidates": stabilized_candidates,
                    "stabilization_samples": stabilization_samples,
                    "population_diagnostics": diagnostics.to_dict(),
                }
            )
            round_index += 1
            if budget_exhausted:
                result.stop_reason = "timing_budget_exhausted"
                break
            if early_stop_on_convergence and convergence_detected(best_history, diagnostics):
                result.stop_reason = "converged"
                break

        if result.stop_reason is None:
            result.stop_reason = "stream_exhausted" if stream_index >= len(stream) and not pending else "search_budget"

        ranked_hashes = [
            summary.candidate_hash
            for summary in db.rank_benchmarks(
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                shape_id=shape.id,
                min_samples=2,
                limit=hot_finalists,
            )
        ]
        for candidate_hash in ranked_hashes:
            record = oracle.get(candidate_hash)
            if record is None or record.hot_gflops is None or record.hot_gflops <= 0.0:
                continue
            hot_cost = cost.hot_launches * _launch_seconds(shape, record.hot_gflops)
            if result.simulated_time_s + hot_cost > cost.time_budget_s:
                break
            result.simulated_time_s += hot_cost
            if result.best_hot_gflops is None or record.hot_gflops > result.best_hot_gflops:
                result.best_hot_hash = candidate_hash
                result.best_hot_gflops = record.hot_gflops
        result.reached_target = bool(
            target_hot_gflops is not None
            and result.best_hot_gflops is not None
            and result.best_hot_gflops >= target_hot_gflops
        )
    return result
