#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from evotensile.profile import get_profile
from evotensile.protocol import BenchmarkProtocol
from evotensile.search.hot_confirm import hot_confirm_topk
from evotensile.shapes import parse_shape


def main() -> int:
    parser = argparse.ArgumentParser(description="Hot-loop confirm validation-passed screening finalists")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner-bin", type=Path, required=True)
    parser.add_argument("--shape", default="8192,8192,1,8192")
    parser.add_argument("--profile", default="gfx1151-nt-hhs")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--runner-timeout", type=float, default=300.0)
    args = parser.parse_args()

    profile = get_profile(args.profile)
    shape = parse_shape(args.shape)
    screening_protocol = BenchmarkProtocol(
        num_warmups=1,
        num_benchmarks=2,
        enqueues_per_sync=1,
        syncs_per_benchmark=1,
    )
    records = hot_confirm_topk(
        db_path=args.db,
        output_dir=args.output,
        runner_bin=args.runner_bin,
        shape_id=shape.id,
        problem_type_hash=profile.problem_type_hash,
        screening_protocol_hash=profile.benchmark_protocol_hash(screening_protocol),
        validation_protocol_hash=screening_protocol.validation_protocol_hash(),
        top_k=args.top_k,
        runner_timeout_s=args.runner_timeout,
    )
    print(json.dumps(records, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
