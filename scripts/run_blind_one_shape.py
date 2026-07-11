#!/usr/bin/env python3
import argparse
from pathlib import Path

from evotensile.campaign.configuration import CampaignConfigurationRequest, build_campaign_configuration
from evotensile.campaign.one_shape import OneShapeCampaign, run_one_shape_campaign
from evotensile.campaign.store import CampaignStore
from evotensile.profile import get_profile
from evotensile.runner import DEFAULT_TENSILELITE_BIN
from evotensile.shapes import parse_shape

DEFAULT_HOT_RESERVE_S = 60.0
DEFAULT_MAX_FEEDBACK_ROUNDS = 100


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the blind one-shape 20-minute search policy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", default="8192,8192,1,8192")
    parser.add_argument("--profile", default="gfx1151-nt-hhs")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--time-budget", type=float, default=1200.0)
    parser.add_argument("--hot-reserve", type=float, default=DEFAULT_HOT_RESERVE_S)
    parser.add_argument("--max-feedback-rounds", type=int, default=DEFAULT_MAX_FEEDBACK_ROUNDS)
    parser.add_argument("--no-leader-stabilization", action="store_true")
    parser.add_argument("--early-stop-on-convergence", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--runner-bin", type=Path, default=Path("build/evotensile-structured-runner"))
    parser.add_argument("--tensilelite-bin", type=Path, default=Path(DEFAULT_TENSILELITE_BIN))
    parser.add_argument("--build-timeout", type=float, default=300.0)
    parser.add_argument("--runner-timeout", type=float, default=300.0)
    args = parser.parse_args()

    profile = get_profile(args.profile)
    shape = parse_shape(args.shape)
    return run_one_shape_campaign(
        OneShapeCampaign(
            configuration=build_campaign_configuration(
                CampaignConfigurationRequest(
                    runner_bin=args.runner_bin,
                    tensilelite_bin=args.tensilelite_bin,
                    seed=args.seed,
                    time_budget_s=args.time_budget,
                    hot_reserve_s=args.hot_reserve,
                    max_feedback_rounds=args.max_feedback_rounds,
                    early_stop_on_convergence=args.early_stop_on_convergence,
                    build_timeout_s=args.build_timeout,
                    runner_timeout_s=args.runner_timeout,
                    leader_stabilization=not args.no_leader_stabilization,
                ),
                profile=profile,
                shape=shape,
            ),
            profile=profile,
            shape=shape,
            store=CampaignStore(args.output),
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
