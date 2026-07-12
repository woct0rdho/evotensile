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
from evotensile.campaign.controller import CampaignControllerState, estimate_admission_duration_s
from evotensile.campaign.protocols import CAMPAIGN_HOT_PROTOCOL, CAMPAIGN_SCREENING_PROTOCOL, protocol_launches
from evotensile.candidate import Candidate, Shape
from evotensile.database import BenchmarkEventInsert, EvoTensileDB, ValidationInsert
from evotensile.metrics import gflops_from_us
from evotensile.profile import TargetProfile
from evotensile.protocol import BenchmarkProtocol
from evotensile.search.campaign_control import convergence_detected, population_diagnostics, split_budget
from evotensile.search.evidence import ProposalEvidenceSnapshot, load_proposal_evidence_snapshot
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
class ReplayPairQuery:
    shape_id: str
    candidate_hash: str
    record: OracleRecord | None
    first_query: bool

    @property
    def known(self) -> bool:
        return self.record is not None


@dataclass(frozen=True)
class ReplayShapeState:
    shape_id: str
    queried_pairs: int
    known_pairs: int
    unknown_pairs: int
    successful_pairs: int
    incumbent_hash: str | None
    incumbent_gflops: float | None

    @property
    def resolved(self) -> bool:
        return self.incumbent_hash is not None


class ExactOracleReplayState:
    def __init__(
        self,
        *,
        db: EvoTensileDB,
        shapes: Sequence[Shape],
        oracle: Mapping[tuple[str, str], OracleRecord],
        profile: TargetProfile,
        screening_protocol: BenchmarkProtocol = CAMPAIGN_SCREENING_PROTOCOL,
        source_ref: str = "multi_shape_exact_oracle",
    ) -> None:
        self.db = db
        self.profile = profile
        self.screening_protocol = screening_protocol
        self.source_ref = source_ref
        self._shapes = {shape.id: shape for shape in shapes}
        if not self._shapes:
            raise ValueError("replay state requires at least one shape")
        if len(self._shapes) != len(shapes):
            raise ValueError("replay shapes must have unique shape IDs")
        self._oracle = dict(oracle)
        self._candidates: dict[str, Candidate] = {}
        for (shape_id, candidate_hash), record in self._oracle.items():
            if shape_id not in self._shapes:
                raise ValueError(f"oracle pair references an unregistered shape: {shape_id}")
            if candidate_hash != record.candidate.hash:
                raise ValueError(
                    f"oracle candidate hash mismatch for {shape_id}: expected {candidate_hash}, "
                    f"got {record.candidate.hash}"
                )
            self._register_candidate_identity(record.candidate)
        self._queried_pairs: set[tuple[str, str]] = set()
        self._unknown_pairs: set[tuple[str, str]] = set()
        self._disclosed_pairs: set[tuple[str, str]] = set()
        self._screening_samples: dict[tuple[str, str], int] = {}
        self._query_order: list[tuple[str, str]] = []
        self._prepared_candidates: set[str] = set()
        self._preparation_time_s = 0.0
        self._pair_time_s = 0.0
        self._pair_times_s: dict[tuple[str, str], float] = defaultdict(float)
        self.db.init()
        self.db.register_shapes(list(self._shapes.values()))

    def _register_candidate_identity(self, candidate: Candidate) -> None:
        existing = self._candidates.get(candidate.hash)
        if existing is not None and existing.canonical_params() != candidate.canonical_params():
            raise ValueError(f"conflicting candidate parameters for hash {candidate.hash}")
        self._candidates.setdefault(candidate.hash, candidate)

    def _pair_key(self, shape: Shape, candidate_hash: str) -> tuple[str, str]:
        registered = self._shapes.get(shape.id)
        if registered is None or registered != shape:
            raise ValueError(f"shape is not registered in replay state: {shape.id}")
        return shape.id, candidate_hash

    @property
    def candidate_catalog(self) -> dict[str, Candidate]:
        return dict(self._candidates)

    def has_oracle_pair(self, shape: Shape, candidate_hash: str) -> bool:
        return self._pair_key(shape, candidate_hash) in self._oracle

    def oracle_record(self, shape: Shape, candidate_hash: str) -> OracleRecord | None:
        return self._oracle.get(self._pair_key(shape, candidate_hash))

    @property
    def queried_pairs(self) -> frozenset[tuple[str, str]]:
        return frozenset(self._queried_pairs)

    @property
    def unknown_pairs(self) -> frozenset[tuple[str, str]]:
        return frozenset(self._unknown_pairs)

    @property
    def disclosed_pairs(self) -> frozenset[tuple[str, str]]:
        return frozenset(self._disclosed_pairs)

    @property
    def query_order(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._query_order)

    @property
    def prepared_candidate_hashes(self) -> frozenset[str]:
        return frozenset(self._prepared_candidates)

    @property
    def preparation_time_s(self) -> float:
        return self._preparation_time_s

    @property
    def pair_time_s(self) -> float:
        return self._pair_time_s

    @property
    def simulated_time_s(self) -> float:
        return self._preparation_time_s + self._pair_time_s

    def preparation_cost(
        self,
        candidates: Sequence[Candidate],
        *,
        workers: int,
        seconds_per_candidate: float,
    ) -> float:
        if workers <= 0:
            raise ValueError("replay preparation workers must be positive")
        if not math.isfinite(seconds_per_candidate) or seconds_per_candidate < 0.0:
            raise ValueError("replay preparation seconds must be finite and nonnegative")
        new_hashes = {candidate.hash for candidate in candidates} - self._prepared_candidates
        return math.ceil(len(new_hashes) / workers) * seconds_per_candidate

    def prepare_candidates(
        self,
        candidates: Sequence[Candidate],
        *,
        workers: int,
        seconds_per_candidate: float,
    ) -> float:
        duration_s = self.preparation_cost(
            candidates,
            workers=workers,
            seconds_per_candidate=seconds_per_candidate,
        )
        unique = {candidate.hash: candidate for candidate in candidates}
        for candidate in unique.values():
            self._register_candidate_identity(candidate)
        new_candidates = [
            candidate for candidate_hash, candidate in unique.items() if candidate_hash not in self._prepared_candidates
        ]
        if not new_candidates:
            return 0.0
        self.db.register_candidates(new_candidates)
        self._prepared_candidates.update(candidate.hash for candidate in new_candidates)
        self._preparation_time_s += duration_s
        return duration_s

    def query_pair(
        self,
        shape: Shape,
        candidate: Candidate,
        *,
        disclose: bool = True,
        samples: int | None = None,
    ) -> ReplayPairQuery:
        self._register_candidate_identity(candidate)
        key = self._pair_key(shape, candidate.hash)
        first_query = key not in self._queried_pairs
        if first_query:
            self.db.register_candidates([candidate])
            self._queried_pairs.add(key)
            self._query_order.append(key)
            if key not in self._oracle:
                self._unknown_pairs.add(key)
        if disclose:
            if key not in self._disclosed_pairs:
                self.disclose_pair(shape, candidate.hash, samples=samples)
            elif samples is not None:
                self.ensure_screening_samples(shape, candidate.hash, target_samples=samples)
        return ReplayPairQuery(
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            record=self._oracle.get(key),
            first_query=first_query,
        )

    def disclose_pair(self, shape: Shape, candidate_hash: str, *, samples: int | None = None) -> bool:
        if samples is not None and samples <= 0:
            raise ValueError("replay screening samples must be positive")
        key = self._pair_key(shape, candidate_hash)
        if key not in self._queried_pairs:
            raise ValueError(f"oracle evidence cannot be disclosed before exact query: {shape.id}/{candidate_hash}")
        if key in self._disclosed_pairs:
            return False
        record = self._oracle.get(key)
        if record is None:
            return False
        if record.screening_gflops is not None and record.screening_gflops > 0.0:
            inserted_samples = self.screening_protocol.num_benchmarks if samples is None else samples
            _insert_screening_evidence(
                self.db,
                shape=shape,
                record=record,
                profile=self.profile,
                screening_protocol=self.screening_protocol,
                source_ref=self.source_ref,
                samples=inserted_samples,
            )
            self._screening_samples[key] = inserted_samples
        elif record.status in {"validation_fail", "validation_failed", "failed_validation"}:
            self.db.insert_validations(
                [
                    ValidationInsert(
                        shape_id=shape.id,
                        candidate_hash=candidate_hash,
                        run_id=self.source_ref,
                        status="failed",
                        problem_type_hash=self.profile.problem_type_hash,
                        validation_protocol_hash=self.screening_protocol.validation_protocol_hash(),
                        source_kind="replay",
                        detail=f"exact oracle status: {record.status}",
                    )
                ]
            )
        elif record.status not in {"ok", "unknown"}:
            self.db.insert_benchmark_events(
                [
                    BenchmarkEventInsert(
                        shape_id=shape.id,
                        candidate_hash=candidate_hash,
                        run_id=self.source_ref,
                        status=record.status,
                        problem_type_hash=self.profile.problem_type_hash,
                        benchmark_protocol_hash=self.profile.benchmark_protocol_hash(self.screening_protocol),
                        source_kind="replay",
                    )
                ]
            )
        self._disclosed_pairs.add(key)
        return True

    def add_screening_samples(self, shape: Shape, candidate_hash: str, *, samples: int) -> None:
        if samples <= 0:
            raise ValueError("replay screening top-up samples must be positive")
        key = self._pair_key(shape, candidate_hash)
        if key not in self._queried_pairs:
            raise ValueError(f"oracle evidence cannot be disclosed before exact query: {shape.id}/{candidate_hash}")
        record = self._oracle.get(key)
        if record is None or record.screening_gflops is None or record.screening_gflops <= 0.0:
            raise ValueError(f"exact oracle pair has no positive screening evidence: {shape.id}/{candidate_hash}")
        _insert_screening_evidence(
            self.db,
            shape=shape,
            record=record,
            profile=self.profile,
            screening_protocol=self.screening_protocol,
            source_ref=self.source_ref,
            samples=samples,
        )
        self._screening_samples[key] = self._screening_samples.get(key, 0) + samples
        self._disclosed_pairs.add(key)

    def screening_samples(self, shape: Shape, candidate_hash: str) -> int:
        return self._screening_samples.get(self._pair_key(shape, candidate_hash), 0)

    def ensure_screening_samples(self, shape: Shape, candidate_hash: str, *, target_samples: int) -> int:
        if target_samples <= 0:
            raise ValueError("replay screening sample target must be positive")
        current = self.screening_samples(shape, candidate_hash)
        added = max(0, target_samples - current)
        if added > 0:
            self.add_screening_samples(shape, candidate_hash, samples=added)
        return added

    def record_pair_time(self, shape: Shape, candidate_hash: str, duration_s: float) -> None:
        if not math.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError("replay pair time must be finite and nonnegative")
        key = self._pair_key(shape, candidate_hash)
        if key not in self._queried_pairs:
            raise ValueError(f"pair time cannot be charged before exact query: {shape.id}/{candidate_hash}")
        self._pair_times_s[key] += duration_s
        self._pair_time_s += duration_s

    def pair_time(self, shape: Shape, candidate_hash: str) -> float:
        return self._pair_times_s.get(self._pair_key(shape, candidate_hash), 0.0)

    def shape_state(self, shape: Shape) -> ReplayShapeState:
        self._pair_key(shape, "")
        queried = [key for key in self._queried_pairs if key[0] == shape.id]
        known = [key for key in queried if key in self._oracle]
        successful = [
            key
            for key in known
            if key in self._disclosed_pairs
            and self._oracle[key].screening_gflops is not None
            and (self._oracle[key].screening_gflops or 0.0) > 0.0
        ]
        incumbent_hash = None
        incumbent_gflops = None
        if successful:
            incumbent_key = max(successful, key=lambda key: self._oracle[key].screening_gflops or 0.0)
            incumbent_hash = incumbent_key[1]
            incumbent_gflops = self._oracle[incumbent_key].screening_gflops
        return ReplayShapeState(
            shape_id=shape.id,
            queried_pairs=len(queried),
            known_pairs=len(known),
            unknown_pairs=len([key for key in queried if key in self._unknown_pairs]),
            successful_pairs=len(successful),
            incumbent_hash=incumbent_hash,
            incumbent_gflops=incumbent_gflops,
        )

    def queried_shape_ids(self, candidate_hash: str, *, successful_only: bool = False) -> tuple[str, ...]:
        shape_ids = []
        for shape_id, queried_hash in self._query_order:
            if queried_hash != candidate_hash or shape_id in shape_ids:
                continue
            record = self._oracle.get((shape_id, candidate_hash))
            if successful_only and (
                (shape_id, candidate_hash) not in self._disclosed_pairs
                or record is None
                or record.screening_gflops is None
                or record.screening_gflops <= 0.0
            ):
                continue
            shape_ids.append(shape_id)
        return tuple(shape_ids)

    def evidence_snapshot(self, *, shapes: Sequence[Shape] | None = None) -> ProposalEvidenceSnapshot:
        selected_shapes = list(self._shapes.values()) if shapes is None else list(shapes)
        for shape in selected_shapes:
            self._pair_key(shape, "")
        return load_proposal_evidence_snapshot(
            self.db,
            problem_type_hash=self.profile.problem_type_hash,
            benchmark_protocol_hash=self.profile.benchmark_protocol_hash(self.screening_protocol),
            shapes=selected_shapes,
        )

    def summary(self) -> dict[str, object]:
        shape_states = [self.shape_state(shape) for shape in self._shapes.values()]
        return {
            "shapes": len(self._shapes),
            "oracle_pairs": len(self._oracle),
            "catalog_candidates": len(self._candidates),
            "queried_pairs": len(self._queried_pairs),
            "known_pairs": len(self._queried_pairs - self._unknown_pairs),
            "unknown_pairs": len(self._unknown_pairs),
            "disclosed_pairs": len(self._disclosed_pairs),
            "prepared_candidates": len(self._prepared_candidates),
            "preparation_time_s": self._preparation_time_s,
            "pair_time_s": self._pair_time_s,
            "simulated_time_s": self.simulated_time_s,
            "unresolved_shape_ids": [state.shape_id for state in shape_states if not state.resolved],
            "shape_states": [
                {
                    "shape_id": state.shape_id,
                    "queried_pairs": state.queried_pairs,
                    "known_pairs": state.known_pairs,
                    "unknown_pairs": state.unknown_pairs,
                    "successful_pairs": state.successful_pairs,
                    "incumbent_hash": state.incumbent_hash,
                    "incumbent_gflops": state.incumbent_gflops,
                    "resolved": state.resolved,
                }
                for state in shape_states
            ],
        }


@dataclass(frozen=True)
class ReplayCostModel:
    time_budget_s: float = 1200.0
    prepare_workers: int = 4
    prepare_seconds_per_candidate: float = 8.0
    probe_launches: int = 3
    initial_probe_launches: int = 1
    hot_reserve_s: float = 60.0
    probe_max_slowdown_factor: float = 4.0
    probe_min_survivors: int = 8
    screening_protocol: BenchmarkProtocol = CAMPAIGN_SCREENING_PROTOCOL
    hot_protocol: BenchmarkProtocol = CAMPAIGN_HOT_PROTOCOL
    stabilization_policy: ScreeningStabilizationPolicy = field(default_factory=ScreeningStabilizationPolicy)

    @property
    def screening_launches(self) -> int:
        return protocol_launches(self.screening_protocol)

    @property
    def hot_launches(self) -> int:
        return protocol_launches(self.hot_protocol)


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
    controller_summary: Mapping[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "queried": len(self.queried),
            "unknown": len(self.unknown),
            "screened": len(self.screened),
            "screening_survivors": len(self.screening_survivors),
            "simulated_time_s": self.simulated_time_s,
            "budget_overrun_s": self.controller_summary.get("budget_overrun_s", 0.0),
            "best_screening_hash": self.best_screening_hash,
            "best_screening_gflops": self.best_screening_gflops,
            "best_hot_hash": self.best_hot_hash,
            "best_hot_gflops": self.best_hot_gflops,
            "reached_target": self.reached_target,
            "stop_reason": self.stop_reason,
            "trace": self.trace,
            "controller": self.controller_summary,
        }


@dataclass(frozen=True)
class IndependentReplayResult:
    seed: int
    shape_results: dict[str, ReplayResult]

    def summary(self) -> dict[str, object]:
        per_shape_log_regret: dict[str, float | None] = {}
        for shape_id, result in self.shape_results.items():
            controller_metrics = result.controller_summary.get("grid_metrics")
            regret = None
            if isinstance(controller_metrics, Mapping):
                values = controller_metrics.get("per_shape_log_regret")
                if isinstance(values, Mapping):
                    value = values.get(shape_id)
                    regret = float(value) if isinstance(value, (int, float)) else None
            per_shape_log_regret[shape_id] = regret
        resolved = [value for value in per_shape_log_regret.values() if value is not None]
        prepared_candidates = [
            value
            for result in self.shape_results.values()
            if isinstance((value := result.controller_summary.get("prepared_candidates")), int)
        ]
        return {
            "seed": self.seed,
            "shapes": len(self.shape_results),
            "shape_results": {shape_id: result.summary() for shape_id, result in self.shape_results.items()},
            "total_simulated_time_s": sum(result.simulated_time_s for result in self.shape_results.values()),
            "total_queries": sum(len(result.queried) for result in self.shape_results.values()),
            "total_prepared_candidates": sum(prepared_candidates),
            "resolved_shapes": len(resolved),
            "unresolved_shapes": len(self.shape_results) - len(resolved),
            "mean_log_regret": statistics.fmean(resolved) if resolved else None,
            "worst_log_regret": max(resolved) if resolved else None,
            "per_shape_log_regret": per_shape_log_regret,
        }


def load_db_oracle_matrix(
    path: str | Path,
    *,
    shapes: Sequence[Shape],
    benchmark_protocol_hash: str | None = None,
) -> dict[tuple[str, str], OracleRecord]:
    shapes_by_id = {shape.id: shape for shape in shapes}
    if len(shapes_by_id) != len(shapes):
        raise ValueError("oracle shapes must have unique shape IDs")
    if not shapes_by_id:
        return {}
    con = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    shape_placeholders = ",".join("?" for _ in shapes_by_id)
    protocol_clause = "AND bp.benchmark_protocol_hash = ?" if benchmark_protocol_hash is not None else ""
    params = list(shapes_by_id)
    if benchmark_protocol_hash is not None:
        params.append(benchmark_protocol_hash)
    rows = con.execute(
        f"""
        SELECT s.shape_id, c.params_json, c.created_at, c.candidate_hash, be.status, bs.time_us
        FROM benchmark_events AS be
        LEFT JOIN benchmark_samples AS bs USING (event_id)
        JOIN candidates AS c USING (candidate_id)
        JOIN shapes AS s USING (shape_key)
        JOIN benchmark_namespaces AS bn USING (benchmark_namespace_id)
        JOIN benchmark_protocols AS bp USING (benchmark_protocol_id)
        WHERE s.shape_id IN ({shape_placeholders}) {protocol_clause}
        ORDER BY c.created_at, s.shape_id, be.event_id, bs.sample_index
        """,
        params,
    ).fetchall()
    con.close()
    grouped: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
    payloads: dict[str, str] = {}
    order: dict[str, float] = {}
    for row in rows:
        candidate_hash = str(row["candidate_hash"])
        key = str(row["shape_id"]), candidate_hash
        grouped[key].append(row)
        payloads[candidate_hash] = str(row["params_json"])
        order[candidate_hash] = float(row["created_at"])
    records: dict[tuple[str, str], OracleRecord] = {}
    for key in sorted(grouped, key=lambda item: (order[item[1]], item[0], item[1])):
        shape_id, candidate_hash = key
        candidate_rows = grouped[key]
        times = [float(row["time_us"]) for row in candidate_rows if row["status"] == "ok" and row["time_us"]]
        if times:
            screening_gflops = gflops_from_us(shapes_by_id[shape_id], statistics.median(times))
            status = "ok"
        else:
            screening_gflops = None
            statuses = [str(row["status"]) for row in candidate_rows]
            status = statuses[-1]
        records[key] = OracleRecord(
            candidate=Candidate(params=dict(json.loads(payloads[candidate_hash])), source="historical_replay"),
            status=status,
            screening_gflops=screening_gflops,
            order=order[candidate_hash],
            source_artifact=str(path),
        )
    return records


def load_db_oracle(
    path: str | Path,
    *,
    shape: Shape,
    benchmark_protocol_hash: str | None = None,
) -> list[OracleRecord]:
    matrix = load_db_oracle_matrix(
        path,
        shapes=[shape],
        benchmark_protocol_hash=benchmark_protocol_hash,
    )
    return list(matrix.values())


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
    profile: TargetProfile,
    screening_protocol: BenchmarkProtocol,
    source_ref: str,
    samples: int | None = None,
) -> None:
    if record.screening_gflops is None or record.screening_gflops <= 0.0:
        return
    time_us = 2.0 * shape.m * shape.n * shape.batch * shape.k / (record.screening_gflops * 1e3)
    validation_protocol_hash = screening_protocol.validation_protocol_hash()
    samples = screening_protocol.num_benchmarks if samples is None else samples
    db.insert_validations(
        [
            ValidationInsert(
                shape_id=shape.id,
                candidate_hash=record.candidate.hash,
                run_id=source_ref,
                status="passed",
                problem_type_hash=profile.problem_type_hash,
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
                problem_type_hash=profile.problem_type_hash,
                benchmark_protocol_hash=profile.benchmark_protocol_hash(screening_protocol),
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
    profile: TargetProfile,
    screening_protocol: BenchmarkProtocol,
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
        problem_type_hash=profile.problem_type_hash,
        benchmark_protocol_hash=profile.benchmark_protocol_hash(screening_protocol),
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
            workgroup_processor_count=profile.workgroup_processor_count,
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
                seed=seed + island_index,
                min_evidence=min_evidence,
                covering_cold_start=covering_cold_start,
                surrogate_jobs=1,
                workgroup_processor_count=profile.workgroup_processor_count,
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
    profile: TargetProfile,
    cost: ReplayCostModel,
    seed: int,
    surrogate_min_evidence: int,
    batch_size: int = 32,
    pool_window: int = 128,
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
            environment_compatibility_tag=profile.environment_compatibility_tag,
        )
        replay_source_ref = f"simulated_stream_seed_{seed}"
        state = ExactOracleReplayState(
            db=db,
            shapes=[shape],
            oracle={(shape.id, candidate_hash): record for candidate_hash, record in oracle.items()},
            profile=profile,
            screening_protocol=cost.screening_protocol,
            source_ref=replay_source_ref,
        )
        controller = CampaignControllerState(
            shape_ids=(shape.id,),
            time_budget_s=cost.time_budget_s,
            session_started_at=0.0,
        )
        controller.set_reserve("confirmation", max(0.0, cost.hot_reserve_s))
        round_cost_observations: list[tuple[float, int]] = []
        pending: dict[str, Candidate] = {}
        stream_index = 0
        queried: set[str] = set()
        round_index = 0
        incumbent_gflops: float | None = None
        best_history: list[float] = []
        while stream_index < len(stream) or pending:
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
                profile=profile,
                screening_protocol=cost.screening_protocol,
                shape=shape,
                count=min(batch_size, len(pending)),
                seed=seed + round_index,
                min_evidence=surrogate_min_evidence,
                covering_cold_start=covering_cold_start and round_index == 0,
                island_count=island_count,
                isolated=round_index <= island_isolation_rounds,
            )
            preparation_cost = state.preparation_cost(
                selected,
                workers=max(1, cost.prepare_workers),
                seconds_per_candidate=cost.prepare_seconds_per_candidate,
            )
            predicted_round_s = estimate_admission_duration_s(
                round_cost_observations,
                expected_units=len(selected),
                minimum_s=preparation_cost,
                fixed_overhead_s=0.0,
                default_s=preparation_cost,
            )
            admission = controller.decide_admission(
                predicted_duration_s=predicted_round_s,
                reserve_s=max(0.0, cost.hot_reserve_s),
                now=state.simulated_time_s,
            )
            if not admission.admitted:
                result.stop_reason = admission.reason
                break
            round_started_s = state.simulated_time_s
            prepared_shape_ids = [shape.id] if selected else []
            state.prepare_candidates(
                selected,
                workers=max(1, cost.prepare_workers),
                seconds_per_candidate=cost.prepare_seconds_per_candidate,
            )
            for candidate in selected:
                controller.record_prepared(candidate.hash, prepared_shape_ids)
            controller.record_phase_time("preparation", state.simulated_time_s - round_started_s)
            result.simulated_time_s = state.simulated_time_s
            known_ok = []
            for candidate in selected:
                pending.pop(candidate.hash, None)
                queried.add(candidate.hash)
                result.queried.append(candidate.hash)
                query = state.query_pair(shape, candidate, disclose=False)
                controller.record_query(shape.id, candidate.hash, known=query.known)
                record = query.record
                if record is None:
                    result.unknown.append(candidate.hash)
                    continue
                if record.screening_gflops is not None and record.screening_gflops > 0.0:
                    initial_probe_launches = min(
                        max(0, cost.initial_probe_launches),
                        max(0, cost.probe_launches),
                    )
                    initial_probe_cost = initial_probe_launches * _launch_seconds(shape, record.screening_gflops)
                    state.record_pair_time(shape, record.candidate.hash, initial_probe_cost)
                    controller.record_phase_time("probe", initial_probe_cost)
                    result.simulated_time_s = state.simulated_time_s
                    known_ok.append(record)
                else:
                    state.disclose_pair(shape, candidate.hash)
                    controller.disclose(shape.id, candidate.hash)
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
                    state.record_pair_time(shape, record.candidate.hash, additional_probe_cost)
                    controller.record_phase_time("probe", additional_probe_cost)
                    result.simulated_time_s = state.simulated_time_s
                    survivors.append(record)
                for record in survivors:
                    if record.screening_gflops is None:
                        continue
                    screening_cost = cost.screening_launches * _launch_seconds(shape, record.screening_gflops)
                    state.record_pair_time(shape, record.candidate.hash, screening_cost)
                    controller.record_phase_time("screening", screening_cost)
                    result.simulated_time_s = state.simulated_time_s
                    result.screening_survivors.append(record.candidate.hash)
                    state.disclose_pair(shape, record.candidate.hash)
                    controller.disclose(
                        shape.id,
                        record.candidate.hash,
                        performance=record.screening_gflops,
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
                    problem_type_hash=profile.problem_type_hash,
                    benchmark_protocol_hashes=[profile.benchmark_protocol_hash(cost.screening_protocol)],
                    min_samples=1,
                    shape_ids={shape.id},
                )
                stabilization_plan = plan_screening_stabilization(
                    stats_by_shape,
                    shapes=[shape],
                    protocol=cost.screening_protocol,
                    policy=cost.stabilization_policy,
                )
                for request in stabilization_plan.requests:
                    record = oracle.get(request.candidate_hash)
                    if record is None or record.screening_gflops is None or record.screening_gflops <= 0.0:
                        continue
                    launch_seconds = _launch_seconds(shape, record.screening_gflops)
                    topup_cost = (
                        cost.screening_protocol.num_warmups
                        + request.remaining_samples
                        * cost.screening_protocol.enqueues_per_sync
                        * cost.screening_protocol.syncs_per_benchmark
                    ) * launch_seconds
                    state.record_pair_time(shape, record.candidate.hash, topup_cost)
                    controller.record_phase_time("stabilization", topup_cost)
                    result.simulated_time_s = state.simulated_time_s
                    state.add_screening_samples(
                        shape,
                        record.candidate.hash,
                        samples=request.remaining_samples,
                    )
                    stabilized_candidates += 1
                    stabilization_samples += request.remaining_samples
            if result.best_screening_gflops is not None:
                best_history.append(result.best_screening_gflops)
            diagnostics = population_diagnostics(
                selected,
                shape,
                workgroup_processor_count=profile.workgroup_processor_count,
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
            round_cost_observations.append((state.simulated_time_s - round_started_s, len(selected)))
            round_index += 1
            if early_stop_on_convergence and convergence_detected(best_history, diagnostics):
                result.stop_reason = "converged"
                break

        if result.stop_reason is None:
            result.stop_reason = "stream_exhausted" if stream_index >= len(stream) and not pending else "search_budget"

        ranked_hashes = [
            summary.candidate_hash
            for summary in db.rank_benchmarks(
                problem_type_hash=profile.problem_type_hash,
                benchmark_protocol_hash=profile.benchmark_protocol_hash(cost.screening_protocol),
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
            admission = controller.decide_admission(
                predicted_duration_s=hot_cost,
                now=state.simulated_time_s,
            )
            if not admission.admitted:
                break
            state.record_pair_time(shape, candidate_hash, hot_cost)
            controller.record_phase_time("confirmation", hot_cost)
            result.simulated_time_s = state.simulated_time_s
            if result.best_hot_gflops is None or record.hot_gflops > result.best_hot_gflops:
                result.best_hot_hash = candidate_hash
                result.best_hot_gflops = record.hot_gflops
        result.reached_target = bool(
            target_hot_gflops is not None
            and result.best_hot_gflops is not None
            and result.best_hot_gflops >= target_hot_gflops
        )
        oracle_best = max(
            (record.screening_gflops or 0.0 for record in oracle.values()),
            default=0.0,
        )
        result.controller_summary = controller.summary(
            oracle_best_by_shape={shape.id: oracle_best} if oracle_best > 0.0 else None,
            now=state.simulated_time_s,
        )
    return result


def simulate_independent_shape_baseline(
    stream: Sequence[Candidate],
    *,
    oracle: Mapping[tuple[str, str], OracleRecord],
    shapes: Sequence[Shape],
    profile: TargetProfile,
    cost: ReplayCostModel,
    seed: int,
    surrogate_min_evidence: int,
    batch_size: int = 32,
    pool_window: int = 128,
    hot_finalists: int = 8,
    covering_cold_start: bool = False,
    island_count: int = 1,
    island_isolation_rounds: int = 0,
    leader_stabilization: bool = False,
    early_stop_on_convergence: bool = False,
) -> IndependentReplayResult:
    shape_ids = [shape.id for shape in shapes]
    if not shape_ids or len(set(shape_ids)) != len(shape_ids):
        raise ValueError("independent replay shapes must be non-empty and unique")
    results = {}
    for shape in shapes:
        shape_oracle = {
            candidate_hash: record for (shape_id, candidate_hash), record in oracle.items() if shape_id == shape.id
        }
        results[shape.id] = simulate_candidate_stream(
            stream,
            oracle=shape_oracle,
            shape=shape,
            profile=profile,
            cost=cost,
            seed=seed,
            surrogate_min_evidence=surrogate_min_evidence,
            batch_size=batch_size,
            pool_window=pool_window,
            hot_finalists=hot_finalists,
            covering_cold_start=covering_cold_start,
            island_count=island_count,
            island_isolation_rounds=island_isolation_rounds,
            leader_stabilization=leader_stabilization,
            early_stop_on_convergence=early_stop_on_convergence,
        )
    return IndependentReplayResult(seed=seed, shape_results=results)
