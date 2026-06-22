from evotensile.adaptive_retime import AdaptivePolicy, decide_shape_retime, timing_stats_from_times


def _stats(candidate_hash: str, samples: list[float]):
    return timing_stats_from_times("m1_n1_b1_k1", candidate_hash, samples)


def test_adaptive_retime_resolves_clear_winner():
    policy = AdaptivePolicy(epsilon_pct=2.0, min_retime_samples=20, max_retime_samples=80, max_k=4)
    decision = decide_shape_retime(
        "m1_n1_b1_k1",
        [
            _stats("best", [100.0, 100.1, 99.9, 100.0, 100.1, 99.9]),
            _stats("slow", [115.0, 115.1, 114.9, 115.0, 115.1, 114.9]),
        ],
        policy=policy,
    )

    assert decision.status == "resolved_winner"
    assert decision.winner_hash == "best"
    assert decision.retime_candidate_hashes == ()


def test_adaptive_retime_resolves_practical_equivalence():
    policy = AdaptivePolicy(epsilon_pct=2.0, min_retime_samples=20, max_retime_samples=80, max_k=4)
    decision = decide_shape_retime(
        "m1_n1_b1_k1",
        [
            _stats("left", [100.0, 100.1, 99.9, 100.0, 100.1, 99.9]),
            _stats("right", [100.5, 100.6, 100.4, 100.5, 100.6, 100.4]),
        ],
        policy=policy,
    )

    assert decision.status == "resolved_equivalent"
    assert decision.winner_hash == "left"
    assert decision.retime_candidate_hashes == ()


def test_adaptive_retime_selects_plausible_noisy_contenders():
    policy = AdaptivePolicy(
        epsilon_pct=1.0,
        min_retime_samples=20,
        max_retime_samples=80,
        sample_step=10,
        max_k=3,
        min_effect_pct=0.5,
    )
    decision = decide_shape_retime(
        "m1_n1_b1_k1",
        [
            _stats("best", [100.0, 98.0, 102.0, 99.0, 101.0, 103.0, 97.0, 100.5, 99.5, 100.0]),
            _stats("near", [101.0, 99.0, 103.0, 100.0, 102.0, 104.0, 98.0, 101.5, 100.5, 101.0]),
            _stats("also_near", [102.0, 100.0, 104.0, 101.0, 103.0, 105.0, 99.0, 102.5, 101.5, 102.0]),
            _stats("slow", [130.0, 131.0, 129.0, 130.5, 129.5, 130.0, 131.5, 128.5, 130.2, 129.8]),
        ],
        policy=policy,
    )

    assert decision.status == "needs_retime"
    assert decision.retime_candidate_hashes == ("best", "near", "also_near")
    assert decision.target_samples >= policy.min_retime_samples
    assert decision.target_samples <= policy.max_retime_samples
    assert decision.target_samples % policy.sample_step == 0
