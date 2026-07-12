#!/usr/bin/env python3

import argparse
import json
import sqlite3
import statistics
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from evotensile.artifacts import load_artifact_mappings
from evotensile.campaign.deployment import select_deployment_solution_bank
from evotensile.campaign.evaluator import PairEvaluationOutcome, RealEvaluator, RealEvaluatorContext
from evotensile.candidate import Shape
from evotensile.database import EvoTensileDB
from evotensile.metrics import gflops_from_us
from evotensile.profile import DEFAULT_PROFILE, PROFILES, get_profile
from evotensile.scheduling.models import EvidenceStage, PairRequest

DEFAULT_DB = Path("out/grid100_production_search_20260712.sqlite")
DEFAULT_OUTPUT_DIR = Path("out/grid100_production_search_20260712/finalization")
DEFAULT_BASELINE_DB = Path("out/grid100_compatible_20260712.sqlite")


class DeploymentAssignmentsPayload(TypedDict):
    assignments: dict[str, str]


@dataclass(frozen=True)
class TimingSummary:
    shape_id: str
    candidate_hash: str
    samples: int
    median_time_us: float
    median_gflops: float
    relative_mad: float

    def to_dict(self) -> dict[str, object]:
        return {
            "shape_id": self.shape_id,
            "candidate_hash": self.candidate_hash,
            "samples": self.samples,
            "median_time_us": self.median_time_us,
            "median_gflops": self.median_gflops,
            "relative_mad": self.relative_mad,
        }


def _timing_rankings(
    path: Path,
    *,
    shapes: Sequence[Shape],
    created_after: float | None = None,
) -> dict[str, tuple[TimingSummary, ...]]:
    shape_by_id = {shape.id: shape for shape in shapes}
    with closing(sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        created_clause = "AND event.created_at >= ?" if created_after is not None else ""
        parameters: list[object] = [DEFAULT_PROFILE.benchmark_protocol_hash()]
        if created_after is not None:
            parameters.append(created_after)
        rows = connection.execute(
            f"""
            SELECT s.shape_id, c.candidate_hash, sample.time_us
            FROM benchmark_samples AS sample
            JOIN benchmark_events AS event USING (event_id)
            JOIN benchmark_namespaces AS namespace USING (benchmark_namespace_id)
            JOIN benchmark_protocols AS protocol USING (benchmark_protocol_id)
            JOIN shapes AS s USING (shape_key)
            JOIN candidates AS c USING (candidate_id)
            WHERE event.status = 'ok'
              AND protocol.benchmark_protocol_hash = ?
              {created_clause}
            ORDER BY s.shape_id, c.candidate_hash, event.event_id, sample.sample_index
            """,
            parameters,
        ).fetchall()
    times_by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        times_by_pair[(str(row["shape_id"]), str(row["candidate_hash"]))].append(float(row["time_us"]))
    rankings: dict[str, tuple[TimingSummary, ...]] = {}
    for shape_id, shape in shape_by_id.items():
        summaries = []
        for (pair_shape_id, candidate_hash), times in times_by_pair.items():
            if pair_shape_id != shape_id:
                continue
            median_time_us = statistics.median(times)
            mad_us = statistics.median(abs(value - median_time_us) for value in times)
            summaries.append(
                TimingSummary(
                    shape_id=shape_id,
                    candidate_hash=candidate_hash,
                    samples=len(times),
                    median_time_us=median_time_us,
                    median_gflops=gflops_from_us(shape, median_time_us),
                    relative_mad=mad_us / median_time_us,
                )
            )
        rankings[shape_id] = tuple(
            sorted(summaries, key=lambda summary: (summary.median_time_us, summary.candidate_hash))
        )
    return rankings


def _latest_fresh_validation_states(
    path: Path,
    *,
    created_after: float,
) -> dict[tuple[str, str], str]:
    with closing(sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT s.shape_id, c.candidate_hash, validation.status,
                   validation.created_at, validation.validation_id
            FROM validations AS validation
            JOIN validation_namespaces AS namespace USING (validation_namespace_id)
            JOIN validation_protocols AS protocol USING (validation_protocol_id)
            JOIN shapes AS s USING (shape_key)
            JOIN candidates AS c USING (candidate_id)
            WHERE protocol.validation_protocol_hash = ?
              AND validation.created_at >= ?
            ORDER BY validation.created_at, validation.validation_id
            """,
            (DEFAULT_PROFILE.default_protocol.validation_protocol_hash(), created_after),
        ).fetchall()
    return {(str(row["shape_id"]), str(row["candidate_hash"])): str(row["status"]) for row in rows}


def _select_contenders(
    rankings: Mapping[str, Sequence[TimingSummary]],
    *,
    maximum_contenders: int,
    relative_tolerance: float,
    mandatory_candidate_by_shape: Mapping[str, str] | None = None,
    additional_mandatory_candidates_by_shape: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, tuple[str, ...]]:
    selected = {}
    for shape_id, rows in rankings.items():
        if not rows:
            raise ValueError(f"timing database has no positive candidate for {shape_id}")
        winner = rows[0]
        contenders = [winner.candidate_hash]
        for row in rows[1:]:
            if len(contenders) >= maximum_contenders:
                break
            if len(contenders) < 2 or row.median_gflops >= winner.median_gflops * (1.0 - relative_tolerance):
                contenders.append(row.candidate_hash)
        mandatory_candidates = []
        mandatory_candidate = (mandatory_candidate_by_shape or {}).get(shape_id)
        if mandatory_candidate is not None:
            mandatory_candidates.append(mandatory_candidate)
        mandatory_candidates.extend((additional_mandatory_candidates_by_shape or {}).get(shape_id, ()))
        for candidate_hash in mandatory_candidates:
            if candidate_hash not in contenders:
                contenders.append(candidate_hash)
        selected[shape_id] = tuple(contenders)
    return selected


def _fresh_outcomes(
    requests: Sequence[PairRequest],
    *,
    rankings: Mapping[str, Sequence[TimingSummary]],
    validation_states: Mapping[tuple[str, str], str],
    source_ref: str,
) -> tuple[PairEvaluationOutcome, ...]:
    summary_by_key = {
        (summary.shape_id, summary.candidate_hash): summary
        for shape_rankings in rankings.values()
        for summary in shape_rankings
    }
    outcomes = []
    for request in requests:
        summary = summary_by_key.get(request.key)
        validation_status = validation_states.get(request.key)
        passed = validation_status == "passed"
        outcomes.append(
            PairEvaluationOutcome(
                request=request,
                provenance="native-fresh-confirmation",
                source_ref=source_ref,
                status="ok" if passed and summary is not None else f"validation_{validation_status or 'unknown'}",
                known=validation_status is not None or summary is not None,
                disclosed=validation_status is not None or summary is not None,
                samples=0 if summary is None else summary.samples,
                performance=summary.median_gflops if passed and summary is not None else None,
            )
        )
    return tuple(outcomes)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--profile", choices=sorted(PROFILES), default=DEFAULT_PROFILE.name)
    parser.add_argument("--baseline-db", type=Path, default=DEFAULT_BASELINE_DB)
    parser.add_argument(
        "--incumbent-deployment",
        type=Path,
        help="Deployment artifact whose incumbent remains mandatory for every shape",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--maximum-contenders", type=int, default=3)
    parser.add_argument("--contender-tolerance", type=float, default=0.02)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    profile = get_profile(args.profile)
    if args.maximum_contenders < 2 or args.samples <= 0:
        raise ValueError("finalization requires at least two contenders and positive samples")
    if not 0.0 <= args.contender_tolerance < 1.0:
        raise ValueError("contender tolerance must be in [0, 1)")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    shapes = profile.shapes()
    shape_by_id = {shape.id: shape for shape in shapes}
    before_rankings = _timing_rankings(args.db, shapes=shapes)
    baseline_rankings = _timing_rankings(args.baseline_db, shapes=shapes)
    baseline_candidate_by_shape = {
        shape_id: rankings[0].candidate_hash for shape_id, rankings in baseline_rankings.items()
    }
    incumbent_candidate_by_shape: dict[str, str] = {}
    if args.incumbent_deployment is not None:
        incumbent_payload = cast(
            DeploymentAssignmentsPayload,
            json.loads(args.incumbent_deployment.read_text(encoding="utf-8")),
        )
        incumbent_candidate_by_shape = dict(incumbent_payload["assignments"])
        if set(incumbent_candidate_by_shape) != set(shape_by_id):
            raise ValueError("incumbent deployment assignments must cover the finalization shapes")
    contenders = _select_contenders(
        before_rankings,
        maximum_contenders=args.maximum_contenders,
        relative_tolerance=args.contender_tolerance,
        mandatory_candidate_by_shape=baseline_candidate_by_shape,
        additional_mandatory_candidates_by_shape={
            shape_id: (candidate_hash,) for shape_id, candidate_hash in incumbent_candidate_by_shape.items()
        },
    )
    candidate_hashes = sorted({candidate_hash for values in contenders.values() for candidate_hash in values})
    db = EvoTensileDB.connect(
        args.db,
        environment_compatibility_tag=profile.environment_compatibility_tag,
    )
    candidate_by_hash = {candidate.hash: candidate for candidate in db.get_candidates(candidate_hashes)}
    missing_candidates = sorted(set(candidate_hashes) - set(candidate_by_hash))
    if missing_candidates:
        raise ValueError(f"finalization candidates are missing from the database: {missing_candidates}")
    requests = tuple(
        PairRequest(
            candidate_by_hash[candidate_hash],
            shape_by_id[shape_id],
            evidence_stage=EvidenceStage.CONFIRMATION,
            min_samples=args.samples,
            priority=before_rankings[shape_id][0].median_gflops
            / next(row.median_gflops for row in before_rankings[shape_id] if row.candidate_hash == candidate_hash),
        )
        for shape_id in sorted(contenders)
        for candidate_hash in contenders[shape_id]
    )
    artifact_shapes_by_candidate: dict[str, list[Shape]] = defaultdict(list)
    for request in requests:
        artifact_shapes_by_candidate[request.candidate.hash].append(request.shape)
    plan = {
        "database": str(args.db),
        "baseline_database": str(args.baseline_db),
        "profile": profile.name,
        "incumbent_deployment": None if args.incumbent_deployment is None else str(args.incumbent_deployment),
        "maximum_contenders": args.maximum_contenders,
        "contender_tolerance": args.contender_tolerance,
        "samples": args.samples,
        "shape_count": len(shapes),
        "candidate_count": len(candidate_hashes),
        "pair_count": len(requests),
        "contenders": {shape_id: list(values) for shape_id, values in sorted(contenders.items())},
    }
    _write_json(args.output_dir / "plan.json", plan)

    source_ref = f"{profile.name}-production-final-confirmation"
    fresh_started_at = time.time()
    wall_started_at = time.monotonic()
    evaluator = RealEvaluator(
        RealEvaluatorContext(
            db=db,
            output_root=args.output_dir,
            target_profile=profile,
            protocol=profile.default_protocol,
            runner_bin=profile.default_runner_bin,
            candidate_batch_size=1,
            shape_batch_size=profile.default_shape_batch_size,
            build_timeout_s=profile.default_build_timeout_s,
            runner_timeout_s=profile.default_runner_timeout_s,
            prepare_workers=profile.default_prepare_workers,
            prepare_wave_batches=profile.default_prepare_wave_batches,
            validation_workers=profile.default_validation_workers,
            compile_cache_root=args.output_dir.parent / "compile_cache",
            cost_aware_scheduling=True,
            ignore_cache=True,
        ),
        source_ref=source_ref,
    )
    evaluation = evaluator.evaluate(
        requests,
        artifact_shapes_by_candidate={
            candidate_hash: tuple({shape.id: shape for shape in candidate_shapes}.values())
            for candidate_hash, candidate_shapes in artifact_shapes_by_candidate.items()
        },
    )
    wall_time_s = time.monotonic() - wall_started_at
    fresh_rankings = _timing_rankings(args.db, shapes=shapes, created_after=fresh_started_at)
    validation_states = _latest_fresh_validation_states(args.db, created_after=fresh_started_at)
    fresh_outcomes = _fresh_outcomes(
        requests,
        rankings=fresh_rankings,
        validation_states=validation_states,
        source_ref=source_ref,
    )
    if any(outcome.status != "ok" for outcome in fresh_outcomes):
        failures = [outcome.key for outcome in fresh_outcomes if outcome.status != "ok"]
        raise ValueError(f"fresh confirmation contains failed or missing pairs: {failures}")

    tolerances = (0.0, 0.005, 0.01, 0.02)
    selections = {
        f"{tolerance:.3f}": select_deployment_solution_bank(
            fresh_outcomes,
            shape_ids=[shape.id for shape in shapes],
            tolerance_fraction=tolerance,
        )
        for tolerance in tolerances
    }
    mappings = load_artifact_mappings(
        db,
        problem_type_hash=profile.problem_type_hash,
        candidate_hashes=candidate_hashes,
        shape_ids=[shape.id for shape in shapes],
    )
    selection_payloads = {}
    for key, selection in selections.items():
        missing_artifacts = sorted(
            (shape_id, candidate_hash)
            for shape_id, candidate_hash in selection.assignments.items()
            if (shape_id, candidate_hash) not in mappings
        )
        if missing_artifacts:
            raise ValueError(f"deployment selection {key} lacks registered artifacts: {missing_artifacts}")
        selection_payloads[key] = selection.to_dict()
        _write_json(args.output_dir / f"deployment_{key}.json", selection.to_dict())

    zero_selection = selections["0.000"]
    fresh_performance_by_key = {
        outcome.key: outcome.performance for outcome in fresh_outcomes if outcome.performance is not None
    }
    fresh_baseline_performance = {
        shape_id: fresh_performance_by_key[(shape_id, baseline_candidate_by_shape[shape_id])]
        for shape_id in zero_selection.shape_ids
    }
    improvement_by_shape = {
        shape_id: zero_selection.confirmed_performance[shape_id] / fresh_baseline_performance[shape_id] - 1.0
        for shape_id in zero_selection.shape_ids
    }
    current_incumbent_improvement = None
    if incumbent_candidate_by_shape:
        fresh_incumbent_performance = {
            shape_id: fresh_performance_by_key[(shape_id, incumbent_candidate_by_shape[shape_id])]
            for shape_id in zero_selection.shape_ids
        }
        incumbent_improvement_by_shape = {
            shape_id: zero_selection.confirmed_performance[shape_id] / fresh_incumbent_performance[shape_id] - 1.0
            for shape_id in zero_selection.shape_ids
        }
        current_incumbent_improvement = {
            "comparison": "fresh same-session current deployment incumbent",
            "improved_shapes": sum(value > 0.0 for value in incumbent_improvement_by_shape.values()),
            "improved_over_one_percent": sum(value >= 0.01 for value in incumbent_improvement_by_shape.values()),
            "mean_improvement_fraction": statistics.fmean(incumbent_improvement_by_shape.values()),
            "median_improvement_fraction": statistics.median(incumbent_improvement_by_shape.values()),
            "maximum_improvement_fraction": max(incumbent_improvement_by_shape.values()),
            "minimum_improvement_fraction": min(incumbent_improvement_by_shape.values()),
            "per_shape": dict(sorted(incumbent_improvement_by_shape.items())),
        }
    historical_reference_performance = {
        shape_id: rankings[0].median_gflops for shape_id, rankings in baseline_rankings.items()
    }
    historical_reference_delta = {
        shape_id: zero_selection.confirmed_performance[shape_id] / historical_reference_performance[shape_id] - 1.0
        for shape_id in zero_selection.shape_ids
    }
    report = {
        "database": str(args.db),
        "baseline_database": str(args.baseline_db),
        "profile": profile.name,
        "plan": str(args.output_dir / "plan.json"),
        "fresh_started_at": fresh_started_at,
        "wall_time_s": wall_time_s,
        "phase_time_s": evaluation.phase_time_s,
        "requested_pairs": len(requests),
        "requested_candidates": len(candidate_hashes),
        "fresh_ok_pairs": sum(outcome.status == "ok" for outcome in fresh_outcomes),
        "fresh_sample_count": sum(outcome.samples for outcome in fresh_outcomes),
        "selection_summary": {
            key: {
                "solution_count": selection.solution_count,
                "generalist_count": len(selection.generalist_coverage),
                "specialist_shape_count": len(selection.specialist_shape_ids),
                "uniform_mean_loss_fraction": selection.uniform_mean_loss_fraction,
                "worst_shape_loss_fraction": selection.worst_shape_loss_fraction,
            }
            for key, selection in selections.items()
        },
        "zero_tolerance_improvement": {
            "comparison": "fresh same-session original compatible winner",
            "improved_shapes": sum(value > 0.0 for value in improvement_by_shape.values()),
            "improved_over_one_percent": sum(value >= 0.01 for value in improvement_by_shape.values()),
            "mean_improvement_fraction": statistics.fmean(improvement_by_shape.values()),
            "median_improvement_fraction": statistics.median(improvement_by_shape.values()),
            "maximum_improvement_fraction": max(improvement_by_shape.values()),
            "minimum_improvement_fraction": min(improvement_by_shape.values()),
            "per_shape": dict(sorted(improvement_by_shape.items())),
        },
        "current_incumbent_improvement": current_incumbent_improvement,
        "historical_reference_delta": {
            "comparison": "cross-session historical pooled median; diagnostic only",
            "mean_fraction": statistics.fmean(historical_reference_delta.values()),
            "median_fraction": statistics.median(historical_reference_delta.values()),
            "per_shape": dict(sorted(historical_reference_delta.items())),
        },
        "selections": selection_payloads,
        "fresh_outcomes": [
            {
                "shape_id": outcome.request.shape.id,
                "candidate_hash": outcome.request.candidate.hash,
                "status": outcome.status,
                "samples": outcome.samples,
                "performance": outcome.performance,
            }
            for outcome in fresh_outcomes
        ],
    }
    _write_json(args.output_dir / "report.json", report)
    print(args.output_dir / "report.json")


if __name__ == "__main__":
    main()
