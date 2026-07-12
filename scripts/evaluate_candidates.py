#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from evotensile.candidate import Shape
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE, PROFILES, get_profile
from evotensile.protocol import apply_benchmark_protocol_overrides
from evotensile.scheduler import execute_schedule
from evotensile.scheduling.models import EvidenceStage, PairRequest
from evotensile.search.replay import load_db_oracle_matrix


def _selected_shapes(shapes: Sequence[Shape], shape_file: Path | None) -> list[Shape]:
    if shape_file is None:
        return list(shapes)
    requested = [
        line.split("#", 1)[0].strip()
        for line in shape_file.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]
    if len(requested) != len(set(requested)):
        raise ValueError("shape file contains duplicate shape IDs")
    shapes_by_id = {shape.id: shape for shape in shapes}
    missing = sorted(set(requested) - set(shapes_by_id))
    if missing:
        raise ValueError(f"shape file contains IDs outside the profile: {missing}")
    return [shapes_by_id[shape_id] for shape_id in requested]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate named database candidates on a profile shape set")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default=DEFAULT_PROFILE.name)
    parser.add_argument("--candidate", action="append", required=True, dest="candidate_hashes")
    parser.add_argument("--num-benchmarks", type=int, default=None)
    parser.add_argument("--shape-file", type=Path)
    parser.add_argument("--unknown-only", action="store_true")
    parser.add_argument("--ignore-cache", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    profile = get_profile(args.profile)
    protocol = apply_benchmark_protocol_overrides(profile.default_protocol, vars(args))
    shapes = _selected_shapes(profile.shapes(), args.shape_file)
    db = EvoTensileDB.connect(args.db, environment_compatibility_tag=profile.environment_compatibility_tag)
    candidates = db.get_candidates(list(dict.fromkeys(args.candidate_hashes)))
    candidate_by_hash = {candidate.hash: candidate for candidate in candidates}
    missing = sorted(set(args.candidate_hashes) - set(candidate_by_hash))
    if missing:
        raise ValueError(f"database candidates are unavailable: {missing}")

    known_pairs: set[tuple[str, str]] = set()
    if args.unknown_only:
        known_pairs = set(
            load_db_oracle_matrix(
                args.db,
                shapes=shapes,
                benchmark_protocol_hash=profile.benchmark_protocol_hash(protocol),
            )
        )
    requests = tuple(
        PairRequest(
            candidate=candidate_by_hash[candidate_hash],
            shape=shape,
            evidence_stage=EvidenceStage.SCREENING,
            min_samples=protocol.num_benchmarks,
        )
        for candidate_hash in args.candidate_hashes
        for shape in shapes
        if (shape.id, candidate_hash) not in known_pairs
    )
    args.output_dir.mkdir(parents=True)
    result = execute_schedule(
        db,
        requests=requests,
        output_root=args.output_dir,
        target_profile=profile,
        protocol=protocol,
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
        ignore_cache=args.ignore_cache,
    )
    status_counts: Counter[str] = Counter()
    for batch in result.executed_batches:
        if batch.ingest is not None:
            status_counts.update(batch.ingest.status_counts)
    report = {
        "database": str(args.db),
        "profile": profile.name,
        "shape_count": len(shapes),
        "candidate_hashes": list(args.candidate_hashes),
        "unknown_only": args.unknown_only,
        "requested_pairs": len(requests),
        "planned_batches": len(result.planned_batches),
        "executed_batches": len(result.executed_batches),
        "status_counts": dict(status_counts),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
