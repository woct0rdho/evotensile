#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from evotensile.campaign.evaluator import (
    HybridEvaluator,
    RealEvaluator,
    RealEvaluatorContext,
    ReplayEvaluator,
)
from evotensile.candidate import Candidate
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.scheduling.models import EvidenceStage, PairRequest
from evotensile.search.replay import ExactOracleReplayState, OracleRecord, load_db_oracle_matrix
from evotensile.shapes import pilot_100_shapes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tuning",
        type=Path,
        default=Path("out/grid100_policy_tuning_20260712.json"),
    )
    parser.add_argument(
        "--retained-db",
        type=Path,
        default=Path("out/grid100_full_20260618_repaired.sqlite"),
    )
    parser.add_argument(
        "--untuned-baseline-db",
        type=Path,
        default=Path("out/grid100_untuned_hipblaslt_baseline_20260712.sqlite"),
    )
    parser.add_argument(
        "--tuned-baseline-db",
        type=Path,
        default=Path("out/grid100_tuned_hipblaslt_baseline_20260712.sqlite"),
    )
    parser.add_argument(
        "--overlay-db",
        type=Path,
        default=Path("out/grid100_policy_hybrid_20260712.sqlite"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/grid100_policy_hybrid_20260712"),
    )
    parser.add_argument("--profiles", nargs="*", default=("anchored-untuned", "anchored-tuned"))
    parser.add_argument("--pairs-per-profile", type=int, default=16)
    args = parser.parse_args()
    if args.pairs_per_profile <= 0:
        raise ValueError("hybrid finalist pair limit must be positive")
    payload = json.loads(args.tuning.read_text(encoding="utf-8"))
    shapes = pilot_100_shapes()
    shape_by_id = {shape.id: shape for shape in shapes}
    source_paths = (args.retained_db, args.untuned_baseline_db, args.tuned_baseline_db)
    oracle: dict[tuple[str, str], OracleRecord] = {}
    candidate_by_hash: dict[str, Candidate] = {}
    for path in source_paths:
        records = load_db_oracle_matrix(path, shapes=shapes)
        oracle.update(records)
        candidate_by_hash.update((record.candidate.hash, record.candidate) for record in records.values())
    selected_pairs = []
    selected_by_profile = {}
    seen = set()
    for profile in args.profiles:
        finalist = payload["hybrid_finalists"][profile]
        profile_pairs = []
        for shape_id, candidate_hash in finalist["remaining_native_pairs"]:
            key = shape_id, candidate_hash
            if key in seen or key in oracle:
                continue
            if shape_id not in shape_by_id or candidate_hash not in candidate_by_hash:
                raise ValueError(f"hybrid finalist pair lacks catalog identity: {key}")
            seen.add(key)
            profile_pairs.append(key)
            selected_pairs.append(key)
            if len(profile_pairs) >= args.pairs_per_profile:
                break
        selected_by_profile[profile] = profile_pairs
    overlay_db = EvoTensileDB.connect(
        args.overlay_db,
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    replay = ReplayEvaluator(
        ExactOracleReplayState(
            db=overlay_db,
            shapes=shapes,
            oracle=oracle,
            profile=DEFAULT_PROFILE,
            source_ref="policy-tuning-retained-oracle",
        ),
        prepare_workers=DEFAULT_PROFILE.default_prepare_workers,
    )
    real = RealEvaluator(
        RealEvaluatorContext(
            db=overlay_db,
            output_root=args.output_dir,
            target_profile=DEFAULT_PROFILE,
            protocol=DEFAULT_PROFILE.default_protocol,
            runner_bin=DEFAULT_PROFILE.default_runner_bin,
            candidate_batch_size=4,
            shape_batch_size=16,
            prepare_workers=DEFAULT_PROFILE.default_prepare_workers,
            validation_workers=DEFAULT_PROFILE.default_validation_workers,
            compile_cache_root=args.output_dir / "compile_cache",
            cost_aware_scheduling=True,
        ),
        source_ref="policy-tuning-hybrid-native",
    )
    evaluator = HybridEvaluator(replay, real)
    requests = [
        PairRequest(
            candidate_by_hash[candidate_hash],
            shape_by_id[shape_id],
            evidence_stage=EvidenceStage.SCREENING,
            min_samples=DEFAULT_PROFILE.default_protocol.num_benchmarks,
        )
        for shape_id, candidate_hash in selected_pairs
    ]
    result = evaluator.evaluate(requests)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "tuning": str(args.tuning),
        "overlay_db": str(args.overlay_db),
        "selected_by_profile": {
            profile: [list(pair) for pair in pairs] for profile, pairs in selected_by_profile.items()
        },
        "requested_pairs": len(requests),
        "known_pairs": result.known_pairs,
        "unknown_pairs": result.unknown_pairs,
        "phase_time_s": result.phase_time_s,
        "outcomes": [
            {
                "shape_id": outcome.request.shape.id,
                "candidate_hash": outcome.request.candidate.hash,
                "status": outcome.status,
                "known": outcome.known,
                "disclosed": outcome.disclosed,
                "samples": outcome.samples,
                "performance": outcome.performance,
                "provenance": outcome.provenance,
                "source_ref": outcome.source_ref,
            }
            for outcome in result.outcomes
        ],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
