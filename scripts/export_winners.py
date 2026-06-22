#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

from evotensile.database import EvoTensileDB
from evotensile.profile import PROFILES, TargetProfile, get_profile
from evotensile.protocol import BenchmarkProtocol
from evotensile.shapes import shape_from_id
from evotensile.yaml_writer import write_tensilelite_yaml


def _load_winners(
    db: EvoTensileDB,
    *,
    profile: TargetProfile,
    protocol: BenchmarkProtocol,
    min_samples: int,
) -> list:
    summaries = db.rank_evaluations(
        problem_type_hash=profile.problem_type_hash,
        benchmark_protocol_hash=profile.benchmark_protocol_hash(protocol),
        min_samples=min_samples,
    )
    winners_by_shape = {}
    for summary in summaries:
        winners_by_shape.setdefault(summary.shape_id, summary)
    return [winners_by_shape[shape_id] for shape_id in sorted(winners_by_shape)]


def _protocol_from_args(args: argparse.Namespace, profile: TargetProfile) -> BenchmarkProtocol:
    return profile.default_protocol.with_overrides(
        num_warmups=args.num_warmups,
        num_benchmarks=args.num_benchmarks,
        enqueues_per_sync=args.enqueues_per_sync,
        syncs_per_benchmark=args.syncs_per_benchmark,
        num_elements_to_validate=args.num_elements_to_validate,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export one best validation-passed EvoTensile candidate per shape")
    parser.add_argument("--db", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--num-warmups", type=int, default=None)
    parser.add_argument("--num-benchmarks", type=int, default=None)
    parser.add_argument("--enqueues-per-sync", type=int, default=None)
    parser.add_argument("--syncs-per-benchmark", type=int, default=None)
    parser.add_argument("--num-elements-to-validate", type=int, default=None)
    args = parser.parse_args()

    profile = get_profile(args.profile)
    protocol = _protocol_from_args(args, profile)
    db = EvoTensileDB.connect(args.db)
    output_dir = Path(args.output_dir)
    yaml_dir = output_dir / "per_shape_yaml"
    json_dir = output_dir / "candidates_json"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    winners = _load_winners(
        db,
        profile=profile,
        protocol=protocol,
        min_samples=args.min_samples,
    )
    candidates = {
        candidate.hash: candidate for candidate in db.get_candidates([winner.candidate_hash for winner in winners])
    }

    csv_path = output_dir / "winners.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "shape_id",
                "candidate_hash",
                "samples",
                "median_gflops",
                "best_gflops",
                "median_time_us",
                "best_time_us",
                "yaml_path",
                "candidate_json_path",
            ]
        )
        for winner in winners:
            candidate = candidates.get(winner.candidate_hash)
            if candidate is None:
                continue
            shape = shape_from_id(winner.shape_id)
            yaml_path = yaml_dir / f"{winner.shape_id}_{candidate.hash}.yaml"
            json_path = json_dir / f"{candidate.hash}.json"
            write_tensilelite_yaml(
                yaml_path,
                [candidate],
                [shape],
                global_parameters=profile.global_parameters(protocol),
                library_logic=profile.library_logic,
                problem_type=profile.problem_type,
            )
            json_path.write_text(candidate.to_json() + "\n", encoding="utf-8")
            writer.writerow(
                [
                    winner.shape_id,
                    winner.candidate_hash,
                    winner.samples,
                    winner.median_gflops if winner.median_gflops is not None else "",
                    winner.best_gflops if winner.best_gflops is not None else "",
                    winner.median_time_us if winner.median_time_us is not None else "",
                    winner.best_time_us if winner.best_time_us is not None else "",
                    yaml_path,
                    json_path,
                ]
            )

    metadata = {
        "db": str(args.db),
        "profile": profile.name,
        "problem_type_hash": profile.problem_type_hash,
        "benchmark_protocol_hash": profile.benchmark_protocol_hash(protocol),
        "protocol": protocol.global_parameters(),
        "min_samples": args.min_samples,
        "winner_count": len(winners),
        "winners_csv": str(csv_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
