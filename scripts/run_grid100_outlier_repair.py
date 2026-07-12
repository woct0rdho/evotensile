#!/usr/bin/env python3

import argparse
import json
import math
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

from evotensile.adaptive_retime import AdaptivePolicy, ProbePolicy
from evotensile.artifacts import load_artifact_mappings
from evotensile.campaign.acquisition import BundleAcquisitionPolicy, BundleCostModel
from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.evaluator import PairEvaluationOutcome, RealEvaluator, RealEvaluatorContext
from evotensile.campaign.repair import (
    RepairPolicy,
    ShapeRepairDeficit,
    build_repair_candidate_pool,
    plan_repair_acquisition,
)
from evotensile.candidate import Shape
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.scheduling.models import EvidenceStage, PairRequest
from evotensile.search.evidence import load_proposal_evidence_snapshot
from evotensile.search.pair_model import ContextualPairModel, PairModelConfiguration
from evotensile.search.replay import OracleRecord, load_db_oracle_matrix
from evotensile.search.shape_clustering import ShapeClusteringConfiguration, cluster_shapes
from evotensile.shapes import pilot_100_shapes

DEFAULT_DB = Path("out/grid100_production_search_20260712.sqlite")
DEFAULT_CAMPAIGN_ROOT = Path("out/grid100_production_search_20260712")
DEFAULT_FINALIZATION = DEFAULT_CAMPAIGN_ROOT / "finalization_v2"


class DeploymentPayload(TypedDict):
    assignments: dict[str, str]
    confirmed_performance: dict[str, float]


class FinalizationReportPayload(TypedDict):
    fresh_started_at: float
    zero_tolerance_improvement: dict[str, object]
    historical_reference_delta: dict[str, object]


class TargetEvidencePayload(TypedDict):
    shape_id: str
    candidate_hash: str
    confirmed_performance: float
    fresh_gain_fraction: float
    historical_delta_fraction: float
    fresh_relative_mad: float
    assignment_coverage: int
    target_score: float
    headroom_fraction: float | None


class RepairOutcomePayload(TypedDict):
    shape_id: str
    candidate_hash: str
    parent_hashes: list[str]
    status: str
    samples: int
    performance: float | None
    baseline_candidate_hash: str
    baseline_performance: float
    improvement_fraction: float | None
    source_ref: str


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _oracle_outcomes(
    oracle: Mapping[tuple[str, str], OracleRecord],
    *,
    shape_by_id: Mapping[str, Shape],
) -> tuple[PairEvaluationOutcome, ...]:
    return tuple(
        PairEvaluationOutcome(
            request=PairRequest(record.candidate, shape_by_id[shape_id], evidence_stage=EvidenceStage.SCREENING),
            provenance="compatible-db",
            source_ref=record.source_artifact,
            status=record.status,
            known=True,
            disclosed=True,
            samples=1 if record.screening_gflops is not None else 0,
            performance=record.screening_gflops,
        )
        for (shape_id, _), record in sorted(oracle.items())
    )


def _fresh_relative_mad(path: Path, *, created_after: float) -> dict[tuple[str, str], float]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT shape.shape_id, candidate.candidate_hash, sample.time_us
        FROM benchmark_samples AS sample
        JOIN benchmark_events AS event USING (event_id)
        JOIN benchmark_namespaces AS namespace USING (benchmark_namespace_id)
        JOIN benchmark_protocols AS protocol USING (benchmark_protocol_id)
        JOIN shapes AS shape USING (shape_key)
        JOIN candidates AS candidate USING (candidate_id)
        WHERE event.status = 'ok'
          AND protocol.benchmark_protocol_hash = ?
          AND event.created_at >= ?
        ORDER BY event.event_id, sample.sample_index
        """,
        (DEFAULT_PROFILE.benchmark_protocol_hash(), created_after),
    ).fetchall()
    connection.close()
    times_by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        times_by_pair[(str(row["shape_id"]), str(row["candidate_hash"]))].append(float(row["time_us"]))
    relative_mad = {}
    for key, times in times_by_pair.items():
        median = statistics.median(times)
        relative_mad[key] = statistics.median(abs(value - median) for value in times) / median
    return relative_mad


def _repair_deficits(
    deployment: DeploymentPayload,
    report: FinalizationReportPayload,
    *,
    relative_mad: Mapping[tuple[str, str], float],
    target_count: int,
) -> tuple[dict[str, ShapeRepairDeficit], list[TargetEvidencePayload]]:
    assignments = deployment["assignments"]
    performance = deployment["confirmed_performance"]
    coverage = Counter(assignments.values())
    gains = cast(dict[str, float], report["zero_tolerance_improvement"]["per_shape"])
    historical_deltas = cast(dict[str, float], report["historical_reference_delta"]["per_shape"])
    evidence: list[TargetEvidencePayload] = []
    for shape_id, candidate_hash in assignments.items():
        mad = relative_mad[(shape_id, candidate_hash)]
        historical_shortfall = max(0.0, -historical_deltas[shape_id])
        singleton = coverage[candidate_hash] == 1
        eligible = historical_shortfall > 0.0 or mad >= 0.005 or singleton
        score = historical_shortfall + 2.0 * mad - gains[shape_id]
        if eligible:
            row: TargetEvidencePayload = {
                "shape_id": shape_id,
                "candidate_hash": candidate_hash,
                "confirmed_performance": performance[shape_id],
                "fresh_gain_fraction": gains[shape_id],
                "historical_delta_fraction": historical_deltas[shape_id],
                "fresh_relative_mad": mad,
                "assignment_coverage": coverage[candidate_hash],
                "target_score": score,
                "headroom_fraction": None,
            }
            evidence.append(row)
    evidence.sort(key=lambda row: (-float(row["target_score"]), str(row["shape_id"])))
    selected = evidence[:target_count]
    deficits = {}
    for row in selected:
        shape_id = str(row["shape_id"])
        incumbent_performance = float(row["confirmed_performance"])
        headroom_fraction = min(
            0.20,
            max(
                0.05,
                max(0.0, -float(row["historical_delta_fraction"])) + 2.0 * float(row["fresh_relative_mad"]),
            ),
        )
        deficit_log = math.log1p(headroom_fraction)
        deficits[shape_id] = ShapeRepairDeficit(
            shape_id=shape_id,
            incumbent_candidate_hash=str(row["candidate_hash"]),
            incumbent_performance=incumbent_performance,
            reference_target=incumbent_performance * (1.0 + headroom_fraction),
            neighbor_target=None,
            cluster_target=None,
            uncertainty_log=0.0,
            evidence_target=incumbent_performance * (1.0 + headroom_fraction),
            raw_deficit_log=deficit_log,
            capped_deficit_log=deficit_log,
        )
        row["headroom_fraction"] = headroom_fraction
    return deficits, selected


def _controller(
    oracle: Mapping[tuple[str, str], OracleRecord],
    deployment: DeploymentPayload,
    *,
    shapes: Sequence[Shape],
    time_budget_s: float,
) -> CampaignControllerState:
    controller = CampaignControllerState(
        shape_ids=tuple(shape.id for shape in shapes),
        time_budget_s=time_budget_s,
        session_started_at=time.monotonic(),
    )
    for shape_id, candidate_hash in oracle:
        controller.record_query(shape_id, candidate_hash, known=True)
    for shape_id, candidate_hash in deployment["assignments"].items():
        controller.disclose(shape_id, candidate_hash, performance=deployment["confirmed_performance"][shape_id])
    return controller


def _mark_registered_artifacts(
    db: EvoTensileDB,
    controller: CampaignControllerState,
    *,
    candidate_hashes: Sequence[str],
    shape_ids: Sequence[str],
) -> None:
    mappings = load_artifact_mappings(
        db,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        candidate_hashes=list(candidate_hashes),
        shape_ids=list(shape_ids),
    )
    by_candidate: dict[str, list[str]] = defaultdict(list)
    for shape_id, candidate_hash in mappings:
        by_candidate[candidate_hash].append(shape_id)
    for candidate_hash, prepared_shape_ids in by_candidate.items():
        controller.record_prepared(candidate_hash, prepared_shape_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--finalization-dir", type=Path, default=DEFAULT_FINALIZATION)
    parser.add_argument("--round-id", default="round26_outlier_repair")
    parser.add_argument("--target-count", type=int, default=12)
    parser.add_argument("--max-pairs", type=int, default=48)
    parser.add_argument("--max-bundles", type=int, default=16)
    parser.add_argument("--soft-budget-s", type=float, default=300.0)
    parser.add_argument("--surrogate-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=12370)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.target_count <= 0 or args.max_pairs <= 0 or args.max_bundles <= 0 or args.soft_budget_s <= 0.0:
        raise ValueError("repair targets, capacities, and budget must be positive")
    if args.surrogate_jobs == 0:
        raise ValueError("surrogate jobs cannot be zero")
    round_dir = args.campaign_root / args.round_id
    if round_dir.exists():
        raise FileExistsError(round_dir)
    round_dir.mkdir(parents=True)

    deployment = cast(
        DeploymentPayload,
        json.loads((args.finalization_dir / "deployment_0.000.json").read_text(encoding="utf-8")),
    )
    finalization_report = cast(
        FinalizationReportPayload,
        json.loads((args.finalization_dir / "report.json").read_text(encoding="utf-8")),
    )
    shapes = pilot_100_shapes()
    shape_by_id = {shape.id: shape for shape in shapes}
    oracle = load_db_oracle_matrix(
        args.db,
        shapes=shapes,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
    )
    outcomes = _oracle_outcomes(oracle, shape_by_id=shape_by_id)
    candidate_by_hash = {record.candidate.hash: record.candidate for record in oracle.values()}
    controller = _controller(oracle, deployment, shapes=shapes, time_budget_s=args.soft_budget_s)
    clustering = cluster_shapes(
        shapes,
        ShapeClusteringConfiguration(
            workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
            cluster_count=16,
        ),
    )
    controller.set_clustering(clustering.to_dict())
    relative_mad = _fresh_relative_mad(args.db, created_after=finalization_report["fresh_started_at"])
    deficits, target_evidence = _repair_deficits(
        deployment,
        finalization_report,
        relative_mad=relative_mad,
        target_count=args.target_count,
    )
    repair_policy = RepairPolicy(
        neighbor_count=8,
        neighbor_candidates_per_shape=3,
        cluster_candidates=4,
        mutation_candidates_per_shape=6,
        mutation_max_changed_genes=2,
        uncertainty_weight=0.0,
        minimum_close_probability=0.10,
        seed=args.seed,
    )
    pool = build_repair_candidate_pool(
        controller,
        shapes=shapes,
        clustering=clustering,
        deficits=deficits,
        observations=outcomes,
        candidate_catalog=candidate_by_hash,
        policy=repair_policy,
    )
    model = ContextualPairModel(
        workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
        configuration=PairModelConfiguration(
            n_estimators=192,
            min_performance_rows=24,
            seed=args.seed,
            jobs=args.surrogate_jobs,
        ),
    )
    fit_summary = model.fit(outcomes)
    predictions = model.predict(pool.prediction_requests(shapes))

    db = EvoTensileDB.connect(
        args.db,
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    _mark_registered_artifacts(
        db,
        controller,
        candidate_hashes=tuple(candidate_by_hash),
        shape_ids=tuple(shape_by_id),
    )
    evidence = load_proposal_evidence_snapshot(
        db,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        shapes=shapes,
    )
    shapes_by_candidate: dict[str, list[Shape]] = defaultdict(list)
    for shape_id, candidate_hash in oracle:
        shapes_by_candidate[candidate_hash].append(shape_by_id[shape_id])
    cost_model = BundleCostModel(
        workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
        fallback_preparation_s=8.0,
        fallback_validation_s=0.15,
        fallback_timing_s=0.05,
        seed=args.seed + 1,
        jobs=args.surrogate_jobs,
    )
    cost_fit = cost_model.fit(
        candidates=candidate_by_hash,
        shapes_by_candidate=shapes_by_candidate,
        measured_costs=evidence.candidate_costs,
    )
    repair = plan_repair_acquisition(
        controller,
        candidates=pool.candidates,
        shapes=shapes,
        deficits=deficits,
        predictions=predictions,
        cost_model=cost_model,
        acquisition_policy=BundleAcquisitionPolicy(
            improvement_weight=0.25,
            coverage_weight=0.0,
            information_weight=0.02,
            repair_weight=1.0,
            bundle_sizes=(1, 2, 4),
            max_pairs=args.max_pairs,
            max_bundles=args.max_bundles,
            max_predicted_cost_s=args.soft_budget_s * 0.85,
            min_samples=DEFAULT_PROFILE.default_protocol.num_benchmarks,
            evidence_stage=EvidenceStage.SCREENING,
        ),
        repair_policy=repair_policy,
    )
    plan = {
        "round_id": args.round_id,
        "database": str(args.db),
        "finalization_directory": str(args.finalization_dir),
        "parameters": {
            "target_count": args.target_count,
            "max_pairs": args.max_pairs,
            "max_bundles": args.max_bundles,
            "soft_budget_s": args.soft_budget_s,
            "surrogate_jobs": args.surrogate_jobs,
            "seed": args.seed,
        },
        "target_evidence": target_evidence,
        "candidate_pool": pool.to_dict(),
        "prediction_request_count": len(predictions),
        "model_fit": fit_summary.to_dict(),
        "cost_fit": cost_fit.to_dict(),
        "repair": repair.to_dict(),
    }
    _write_json(round_dir / "plan.json", plan)
    if args.plan_only or not repair.plan.timing_requests:
        print(round_dir / "plan.json")
        return

    started = time.monotonic()
    evaluator = RealEvaluator(
        RealEvaluatorContext(
            db=db,
            output_root=round_dir,
            target_profile=DEFAULT_PROFILE,
            protocol=DEFAULT_PROFILE.default_protocol,
            runner_bin=DEFAULT_PROFILE.default_runner_bin,
            candidate_batch_size=1,
            shape_batch_size=DEFAULT_PROFILE.default_shape_batch_size,
            build_timeout_s=DEFAULT_PROFILE.default_build_timeout_s,
            runner_timeout_s=DEFAULT_PROFILE.default_runner_timeout_s,
            prepare_workers=DEFAULT_PROFILE.default_prepare_workers,
            prepare_wave_batches=min(DEFAULT_PROFILE.default_prepare_wave_batches, args.max_bundles),
            validation_workers=DEFAULT_PROFILE.default_validation_workers,
            compile_cache_root=args.campaign_root / "compile_cache",
            cost_aware_scheduling=True,
            adaptive_policy=AdaptivePolicy(),
            probe_policy=ProbePolicy(),
        ),
        source_ref=f"grid100-practical:{args.round_id}",
    )
    result = evaluator.evaluate(
        repair.plan.timing_requests,
        artifact_shapes_by_candidate=repair.plan.artifact_shapes_by_candidate,
    )
    wall_time_s = time.monotonic() - started
    baseline = deployment["confirmed_performance"]
    outcomes_payload: list[RepairOutcomePayload] = []
    improvements: list[RepairOutcomePayload] = []
    for outcome in result.outcomes:
        baseline_performance = baseline[outcome.request.shape.id]
        improvement = None if outcome.performance is None else outcome.performance / baseline_performance - 1.0
        row: RepairOutcomePayload = {
            "shape_id": outcome.request.shape.id,
            "candidate_hash": outcome.request.candidate.hash,
            "parent_hashes": list(outcome.request.candidate.parent_hashes),
            "status": outcome.status,
            "samples": outcome.samples,
            "performance": outcome.performance,
            "baseline_candidate_hash": deployment["assignments"][outcome.request.shape.id],
            "baseline_performance": baseline_performance,
            "improvement_fraction": improvement,
            "source_ref": outcome.source_ref,
        }
        outcomes_payload.append(row)
        if improvement is not None and improvement > 0.0:
            improvements.append(row)
    improvements.sort(key=lambda row: -float(row["improvement_fraction"]))
    report = {
        "round_id": args.round_id,
        "database": str(args.db),
        "strategy": "post-finalization-evidence-backed-outlier-repair",
        "wall_time_s": wall_time_s,
        "phase_time_s": result.phase_time_s,
        "requested_pairs": len(repair.plan.timing_requests),
        "requested_candidates": len({request.candidate.hash for request in repair.plan.timing_requests}),
        "known_pairs": result.known_pairs,
        "unknown_pairs": result.unknown_pairs,
        "status_counts": dict(Counter(outcome.status for outcome in result.outcomes)),
        "improvements_over_fresh_finalization": improvements,
        "significant_improvements": [
            row
            for row in improvements
            if row["improvement_fraction"] is not None and row["improvement_fraction"] >= 0.01
        ],
        "outcomes": outcomes_payload,
        "plan": str(round_dir / "plan.json"),
    }
    _write_json(round_dir / "report.json", report)
    print(round_dir / "report.json")


if __name__ == "__main__":
    main()
