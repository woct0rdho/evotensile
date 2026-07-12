#!/usr/bin/env python3

import argparse
import json
import sqlite3
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import TypedDict, cast

from evotensile.campaign.evaluator import RealEvaluator, RealEvaluatorContext
from evotensile.candidate import Shape
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.scheduling.models import EvidenceStage, PairRequest
from evotensile.shapes import pilot_100_shapes

DEFAULT_DB = Path("out/grid100_production_search_20260712.sqlite")
DEFAULT_CAMPAIGN_ROOT = Path("out/grid100_production_search_20260712")
DEFAULT_DEPLOYMENT = DEFAULT_CAMPAIGN_ROOT / "finalization_v3/deployment_0.000.json"


class DeploymentPayload(TypedDict):
    assignments: dict[str, str]
    confirmed_performance: dict[str, float]


class ImprovementPayload(TypedDict):
    shape_id: str
    after_candidate_hash: str
    improvement_fraction: float


class OutcomePayload(TypedDict):
    candidate_hash: str
    improvement_fraction: float | None
    shape_id: str
    status: str


class RoundReportPayload(TypedDict):
    round_id: str
    incumbent_improvements: list[ImprovementPayload]
    outcomes: list[OutcomePayload]


def _screening_contenders(
    report: RoundReportPayload,
    *,
    minimum_gain: float,
) -> dict[str, set[str]]:
    contenders: dict[str, set[str]] = defaultdict(set)
    for outcome in report["outcomes"]:
        gain = outcome["improvement_fraction"]
        if outcome["status"] == "ok" and gain is not None and gain >= minimum_gain:
            contenders[outcome["shape_id"]].add(outcome["candidate_hash"])
    return contenders


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fresh_performance(
    path: Path,
    *,
    created_after: float,
) -> dict[tuple[str, str], tuple[float, int]]:
    shape_by_id = {shape.id: shape for shape in pilot_100_shapes()}
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
    return {
        key: (
            2.0
            * shape_by_id[key[0]].m
            * shape_by_id[key[0]].n
            * shape_by_id[key[0]].batch
            * shape_by_id[key[0]].k
            / (statistics.median(times) * 1e3),
            len(times),
        )
        for key, times in times_by_pair.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--round-report", type=Path, required=True)
    parser.add_argument("--incumbent-deployment", type=Path, default=DEFAULT_DEPLOYMENT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-screening-gain", type=float, default=0.01)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.minimum_screening_gain < 0.0 or args.samples <= 0:
        raise ValueError("confirmation gain and samples must be nonnegative and positive")
    args.output_dir.mkdir(parents=True)

    round_report = cast(RoundReportPayload, json.loads(args.round_report.read_text(encoding="utf-8")))
    deployment = cast(DeploymentPayload, json.loads(args.incumbent_deployment.read_text(encoding="utf-8")))
    contenders = _screening_contenders(
        round_report,
        minimum_gain=args.minimum_screening_gain,
    )
    for shape_id in contenders:
        contenders[shape_id].add(deployment["assignments"][shape_id])
    if not contenders:
        _write_json(
            args.output_dir / "report.json",
            {
                "round_id": round_report["round_id"],
                "requested_pairs": 0,
                "comparisons": [],
                "confirmed_winners": [],
                "checkpoint_deployment": str(args.incumbent_deployment),
            },
        )
        print(args.output_dir / "report.json")
        return

    shape_by_id = {shape.id: shape for shape in pilot_100_shapes()}
    candidate_hashes = sorted({candidate_hash for values in contenders.values() for candidate_hash in values})
    db = EvoTensileDB.connect(
        args.db,
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    candidate_by_hash = {candidate.hash: candidate for candidate in db.get_candidates(candidate_hashes)}
    requests = tuple(
        PairRequest(
            candidate_by_hash[candidate_hash],
            shape_by_id[shape_id],
            evidence_stage=EvidenceStage.CONFIRMATION,
            min_samples=args.samples,
        )
        for shape_id in sorted(contenders)
        for candidate_hash in sorted(contenders[shape_id])
    )
    artifact_shapes: dict[str, list[Shape]] = defaultdict(list)
    for request in requests:
        artifact_shapes[request.candidate.hash].append(request.shape)
    _write_json(
        args.output_dir / "plan.json",
        {
            "round_id": round_report["round_id"],
            "incumbent_deployment": str(args.incumbent_deployment),
            "samples": args.samples,
            "pairs": [list(request.key) for request in requests],
        },
    )

    fresh_started_at = time.time()
    wall_started_at = time.monotonic()
    evaluator = RealEvaluator(
        RealEvaluatorContext(
            db=db,
            output_root=args.output_dir,
            target_profile=DEFAULT_PROFILE,
            protocol=DEFAULT_PROFILE.default_protocol,
            runner_bin=DEFAULT_PROFILE.default_runner_bin,
            candidate_batch_size=1,
            shape_batch_size=DEFAULT_PROFILE.default_shape_batch_size,
            build_timeout_s=DEFAULT_PROFILE.default_build_timeout_s,
            runner_timeout_s=DEFAULT_PROFILE.default_runner_timeout_s,
            prepare_workers=DEFAULT_PROFILE.default_prepare_workers,
            prepare_wave_batches=DEFAULT_PROFILE.default_prepare_wave_batches,
            validation_workers=DEFAULT_PROFILE.default_validation_workers,
            compile_cache_root=args.output_dir.parent.parent / "compile_cache",
            cost_aware_scheduling=True,
            ignore_cache=True,
        ),
        source_ref=f"grid100-practical:{round_report['round_id']}:confirmation",
    )
    result = evaluator.evaluate(
        requests,
        artifact_shapes_by_candidate={
            candidate_hash: tuple({shape.id: shape for shape in candidate_shapes}.values())
            for candidate_hash, candidate_shapes in artifact_shapes.items()
        },
    )
    performance = _fresh_performance(args.db, created_after=fresh_started_at)
    missing = sorted(request.key for request in requests if request.key not in performance)
    if result.unknown_pairs or missing:
        raise ValueError(f"focused confirmation has missing pairs: {missing}")

    assignments = dict(deployment["assignments"])
    confirmed_performance = dict(deployment["confirmed_performance"])
    comparisons = []
    confirmed_winners = []
    for shape_id, candidate_values in sorted(contenders.items()):
        incumbent_hash = deployment["assignments"][shape_id]
        incumbent_performance, incumbent_samples = performance[(shape_id, incumbent_hash)]
        winner_hash = max(candidate_values, key=lambda candidate_hash: performance[(shape_id, candidate_hash)][0])
        winner_performance, winner_samples = performance[(shape_id, winner_hash)]
        for candidate_hash in sorted(candidate_values - {incumbent_hash}):
            candidate_performance, samples = performance[(shape_id, candidate_hash)]
            comparisons.append(
                {
                    "shape_id": shape_id,
                    "candidate_hash": candidate_hash,
                    "incumbent_candidate_hash": incumbent_hash,
                    "candidate_performance": candidate_performance,
                    "incumbent_performance": incumbent_performance,
                    "gain_fraction": candidate_performance / incumbent_performance - 1.0,
                    "samples": samples,
                    "incumbent_samples": incumbent_samples,
                }
            )
        if winner_hash != incumbent_hash:
            assignments[shape_id] = winner_hash
            confirmed_performance[shape_id] = winner_performance
            confirmed_winners.append(
                {
                    "shape_id": shape_id,
                    "candidate_hash": winner_hash,
                    "incumbent_candidate_hash": incumbent_hash,
                    "gain_fraction": winner_performance / incumbent_performance - 1.0,
                    "samples": winner_samples,
                }
            )
    checkpoint_path = args.output_dir / "checkpoint_deployment.json"
    _write_json(
        checkpoint_path,
        {
            "assignments": assignments,
            "confirmed_performance": confirmed_performance,
            "source_deployment": str(args.incumbent_deployment),
            "confirmation_round": round_report["round_id"],
        },
    )
    _write_json(
        args.output_dir / "report.json",
        {
            "round_id": round_report["round_id"],
            "fresh_started_at": fresh_started_at,
            "wall_time_s": time.monotonic() - wall_started_at,
            "requested_pairs": len(requests),
            "known_pairs": result.known_pairs,
            "unknown_pairs": result.unknown_pairs,
            "phase_time_s": result.phase_time_s,
            "comparisons": comparisons,
            "confirmed_winners": confirmed_winners,
            "checkpoint_deployment": str(checkpoint_path),
        },
    )
    print(args.output_dir / "report.json")


if __name__ == "__main__":
    main()
