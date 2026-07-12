#!/usr/bin/env python3

import argparse
from pathlib import Path

from evotensile.campaign.configuration import CampaignConfigurationRequest, build_campaign_configuration
from evotensile.campaign.runner import CampaignRun, run_campaign
from evotensile.campaign.store import CampaignStore
from evotensile.profile import PROFILES, get_profile
from evotensile.runner import DEFAULT_TENSILELITE_BIN
from evotensile.shapes import parse_shape

DEFAULT_HOT_RESERVE_S = 60.0
DEFAULT_MAX_FEEDBACK_ROUNDS = 100


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the blind one-shape 20-minute search policy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", default="8192,8192,1,8192")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--time-budget", type=float, default=1200.0)
    parser.add_argument("--hot-reserve", type=float, default=DEFAULT_HOT_RESERVE_S)
    parser.add_argument("--max-feedback-rounds", type=int, default=DEFAULT_MAX_FEEDBACK_ROUNDS)
    parser.add_argument("--no-leader-stabilization", action="store_true")
    parser.add_argument("--early-stop-on-convergence", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--runner-bin", type=Path, default=None)
    parser.add_argument("--tensilelite-bin", type=Path, default=Path(DEFAULT_TENSILELITE_BIN))
    parser.add_argument("--build-timeout", type=float, default=None)
    parser.add_argument("--runner-timeout", type=float, default=None)
    args = parser.parse_args()

    profile = get_profile(args.profile)
    shape = parse_shape(args.shape)
    runner_bin = Path(profile.default_runner_bin) if args.runner_bin is None else args.runner_bin
    build_timeout_s = profile.default_build_timeout_s if args.build_timeout is None else args.build_timeout
    runner_timeout_s = profile.default_runner_timeout_s if args.runner_timeout is None else args.runner_timeout
    if build_timeout_s is None or runner_timeout_s is None:
        raise ValueError("blind campaigns require positive build and runner timeouts")
    return run_campaign(
        CampaignRun(
            configuration=build_campaign_configuration(
                CampaignConfigurationRequest(
                    runner_bin=runner_bin,
                    tensilelite_bin=args.tensilelite_bin,
                    seed=args.seed,
                    time_budget_s=args.time_budget,
                    hot_reserve_s=args.hot_reserve,
                    max_feedback_rounds=args.max_feedback_rounds,
                    early_stop_on_convergence=args.early_stop_on_convergence,
                    build_timeout_s=build_timeout_s,
                    runner_timeout_s=runner_timeout_s,
                    leader_stabilization=not args.no_leader_stabilization,
                ),
                profile=profile,
                shape=shape,
            ),
            profile=profile,
            shapes=(shape,),
            store=CampaignStore(args.output),
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
