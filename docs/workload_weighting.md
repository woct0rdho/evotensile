# Workload-weighted campaign allocation

Uniform fixed-grid allocation remains the default. An experiment may opt into workload weighting only by supplying one exact workload entry for every registered campaign shape.

## Input and provenance

A workload entry contains:
- `shape_id`: the exact registered shape identifier.
- `call_count`: the expected number of calls in the deployment workload.
- `baseline_latency_us`: the latency of the declared baseline implementation under the campaign benchmark protocol.

The JSON input accepted by `load_workload_weights()` has this form:

```json
{
  "provenance": {
    "call_count_source": "deployment-trace-20260712.json",
    "baseline_label": "anchored-untuned",
    "baseline_source": "out/baseline.sqlite",
    "benchmark_protocol_hash": "bproto_9f4055f5f13232a3",
    "environment_compatibility_tag": "gfx1151-nt-hhs-v1"
  },
  "shapes": [
    {
      "shape_id": "m1024_n1024_b1_k1024",
      "call_count": 42,
      "baseline_latency_us": 17.5
    }
  ]
}
```

The input must cover the exact ordered campaign shape set without duplicates or additional shapes. Call counts must be finite and nonnegative, baseline latencies must be finite and positive, and the total baseline-time contribution must be positive. Provenance must include nonempty `call_count_source`, `baseline_label`, `baseline_source`, `benchmark_protocol_hash`, and `environment_compatibility_tag` values. Those fields are validated, checkpointed, and reported with the resolved workload.

For shape `s`, the unnormalized contribution is:

```text
contribution(s) = call_count(s) * baseline_latency_us(s)
```

Weights normalize to the number of campaign shapes:

```text
weight(s) = contribution(s) * shape_count / sum(contribution)
```

This keeps the uniform mean scale unchanged. A singleton workload always resolves to weight one regardless of its raw contribution.

## Controller state

`CampaignControllerState` owns the resolved workload. `set_workload()` rejects weights for a different shape set. Checkpoints and summaries persist the mode, ordered shape IDs, raw entries, normalized weights, total call count, and total baseline time. With no explicit workload, the controller resolves uniform weight one for every shape.

Workload state is restored before pending staged-round work. Resume therefore preserves acquisition scores and exact timing priorities rather than silently reverting to uniform allocation.

## Allocation boundaries

The same normalized weights flow through:
- contextual bundle acquisition, including improvement, coverage, information, and repair utility.
- exact timing-request priority and staged admission order.
- family generalist and coverage archives.
- representative ordering and multi-shape elite ranking.
- generation and parameter-group operator credit.
- controller, replay, and campaign summaries.

Archive specialist quality remains the unweighted worst observed percentile, and reports retain unweighted unresolved counts and worst-shape regret. Workload weighting can prioritize high-contribution shapes but cannot make a weak measured shape disappear from specialist or worst-shape reporting.

Uniform mode assigns every required shape positive equal mass and remains the complete-grid experiment mode. Workload mode may assign zero weight to a zero-contribution shape and may defer it when the finite round budget is exhausted. That behavior is only permitted after explicit workload selection.

## Reporting

Campaign grid summaries always include:
- unweighted mean, median, p95, and worst log regret.
- workload-weighted mean log regret.
- resolved and unresolved shape counts.
- the complete per-shape regret vector.
- serialized workload mode, entries, weights, and contribution totals.

Proposal metadata records whether shape weighting was active and the exact weights used. Family archive entries record whether their aggregate score was shape-weighted.

## Controlled replay

`out/grid100_workload_weighting_20260712.json` compares uniform and workload allocation from the frozen anchored-untuned initialization and selected P12 policy. The workload contains one call per shape, so its contribution is proportional to the exact untuned hipBLASLt baseline latency. Three stable seeds use the same 385-pair round budget and 300-second predicted-cost admission cap.

Mean results were:

| metric | uniform | workload |
| --- | ---: | ---: |
| workload-weighted mean log regret | 0.26605 | 0.26502 |
| unweighted mean log regret | 0.19079 | 0.18810 |
| unweighted p95 log regret | 0.53466 | 0.52816 |
| worst log regret | 0.65584 | 0.65584 |
| unknown pairs | 148.7 | 128.7 |
| prepared candidates | 11.7 | 12.7 |
| top workload-quartile pair fraction | 22.7% | 26.6% |
| unresolved shapes | 0 | 0 |

The workload-aware policy moved allocation toward high-contribution shapes and modestly improved both weighted and unweighted aggregate regret without worsening the observed tail maximum. The effect is not large enough to justify changing the default. Uniform weighting remains the frozen default, while workload weighting is available as an explicit deployment-specific mode.
