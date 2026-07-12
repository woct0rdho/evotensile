#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from evotensile.campaign.protocols import CAMPAIGN_HOT_PROTOCOL, CAMPAIGN_SCREENING_PROTOCOL
from evotensile.profile import PROFILES, get_profile
from evotensile.search.hot_confirm import hot_confirm_topk
from evotensile.shapes import parse_shape


def main() -> int:
    parser = argparse.ArgumentParser(description="Hot-loop confirm validation-passed screening finalists")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner-bin", type=Path, default=None)
    parser.add_argument("--shape", default="8192,8192,1,8192")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--runner-timeout", type=float, default=None)
    args = parser.parse_args()

    profile = get_profile(args.profile)
    shape = parse_shape(args.shape)
    runner_timeout_s = profile.default_runner_timeout_s if args.runner_timeout is None else args.runner_timeout
    if runner_timeout_s is None:
        raise ValueError("hot confirmation requires a positive runner timeout")
    records = hot_confirm_topk(
        db_path=args.db,
        environment_compatibility_tag=profile.environment_compatibility_tag,
        output_dir=args.output,
        runner_bin=args.runner_bin or profile.default_runner_bin,
        shape_id=shape.id,
        problem_type_hash=profile.problem_type_hash,
        screening_protocol=CAMPAIGN_SCREENING_PROTOCOL,
        hot_protocol=CAMPAIGN_HOT_PROTOCOL,
        top_k=args.top_k,
        runner_timeout_s=runner_timeout_s,
    )
    print(json.dumps(records, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
