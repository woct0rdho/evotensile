import json
import math
import random
from collections import Counter
from collections.abc import Sequence

from evotensile.candidate import Candidate, Shape
from evotensile.search.encoding import PARAM_NAMES
from evotensile.search.family import family_descriptor
from evotensile.search_space import (
    NT_HHS_MAX_GSU_WORKSPACE_BYTES,
    NT_HHS_MAX_LDS_BYTES,
    NT_HHS_MAX_VGPR,
    _gsu_workspace_bytes,
    _nt_hhs_lds_bytes,
    _valu_vgpr_lower_bound,
    macro_tile,
)

GFX1151_EFFECTIVE_CU_COUNT = 20


def candidate_shape_mechanics(
    candidate: Candidate,
    shape: Shape,
    *,
    effective_cu_count: int = GFX1151_EFFECTIVE_CU_COUNT,
) -> dict[str, float]:
    if effective_cu_count <= 0:
        raise ValueError("effective CU count must be positive")
    params = candidate.canonical_params()
    instruction = params["MatrixInstruction"]
    macro_tile0, macro_tile1 = macro_tile(instruction)
    tiles_m = math.ceil(shape.m / macro_tile0)
    tiles_n = math.ceil(shape.n / macro_tile1)
    output_tiles = tiles_m * tiles_n * shape.batch
    workgroups = output_tiles * params["GlobalSplitU"]
    tiles_per_cu = workgroups / effective_cu_count
    cu_rounds = max(1, math.ceil(tiles_per_cu))
    cu_granularity = tiles_per_cu / cu_rounds
    depth_per_split = max(1, params["DepthU"] * params["GlobalSplitU"])
    reduction_iterations = math.ceil(shape.k / depth_per_split)
    covered_k = reduction_iterations * depth_per_split
    k_fill = shape.k / covered_k
    workgroup_threads = math.prod(params["WorkGroup"])
    wavefront_size = int(params.get("WavefrontSize", 32))
    instruction_tile_area = instruction[0] * instruction[1]
    macro_tile_area = macro_tile0 * macro_tile1
    workgroup_tile_multiple = macro_tile_area / instruction_tile_area
    dispatch_efficiency = 1.0 - 1.0 / math.sqrt(max(1.0, workgroup_tile_multiple))
    waves_per_workgroup = workgroup_threads / wavefront_size
    lds_bytes = _nt_hhs_lds_bytes(params)
    valu_vgprs = _valu_vgpr_lower_bound(params)
    workspace_bytes = _gsu_workspace_bytes(params, shape)
    input_output_bytes = (
        2 * shape.batch * shape.m * shape.k + 2 * shape.batch * shape.n * shape.k + 4 * shape.batch * shape.m * shape.n
    )
    flops = 2.0 * shape.m * shape.n * shape.batch * shape.k
    return {
        "tile_fill_m": shape.m / (tiles_m * macro_tile0),
        "tile_fill_n": shape.n / (tiles_n * macro_tile1),
        "output_tiles": float(output_tiles),
        "workgroups": float(workgroups),
        "tiles_per_cu": tiles_per_cu,
        "cu_rounds": float(cu_rounds),
        "cu_granularity": cu_granularity,
        "waves_per_workgroup": waves_per_workgroup,
        "macro_tile_area": float(macro_tile_area),
        "instruction_tile_area": float(instruction_tile_area),
        "workgroup_tile_multiple": workgroup_tile_multiple,
        "dispatch_efficiency": dispatch_efficiency,
        "wave_tile_area": float(instruction[5] * instruction[6]),
        "wave_group_size": float(instruction[7] * instruction[8]),
        "reduction_iterations": float(reduction_iterations),
        "k_fill": k_fill,
        "lds_bytes": float(lds_bytes),
        "lds_fraction": lds_bytes / NT_HHS_MAX_LDS_BYTES,
        "valu_vgpr_lower_bound": float(valu_vgprs),
        "valu_vgpr_fraction": valu_vgprs / NT_HHS_MAX_VGPR,
        "workspace_bytes": float(workspace_bytes),
        "workspace_fraction": workspace_bytes / NT_HHS_MAX_GSU_WORKSPACE_BYTES,
        "arithmetic_intensity": flops / max(input_output_bytes, 1),
    }


def mechanical_prior_score(
    candidate: Candidate,
    shape: Shape,
    *,
    effective_cu_count: int = GFX1151_EFFECTIVE_CU_COUNT,
) -> float:
    mechanics = candidate_shape_mechanics(candidate, shape, effective_cu_count=effective_cu_count)
    utilization = (
        mechanics["tile_fill_m"]
        * mechanics["tile_fill_n"]
        * mechanics["cu_granularity"]
        * min(1.0, mechanics["tiles_per_cu"] / 2.0)
        * mechanics["k_fill"]
        * mechanics["dispatch_efficiency"]
    )
    resource_headroom = 0.10 * max(0.0, 1.0 - mechanics["valu_vgpr_fraction"])
    resource_headroom += 0.05 * max(0.0, 1.0 - mechanics["lds_fraction"])
    return utilization + resource_headroom


def _fraction_bucket(value: float, *, buckets: int = 20) -> int:
    return min(buckets, max(0, int(math.floor(value * buckets))))


def mechanical_coverage_tokens(
    candidate: Candidate,
    shape: Shape,
    *,
    effective_cu_count: int = GFX1151_EFFECTIVE_CU_COUNT,
) -> frozenset[str]:
    params = candidate.canonical_params()
    instruction = params["MatrixInstruction"]
    macro_tile0, macro_tile1 = macro_tile(instruction)
    mechanics = candidate_shape_mechanics(candidate, shape, effective_cu_count=effective_cu_count)
    tokens = {
        f"family:{family_descriptor(candidate).key}",
        f"mi-wave-tile:{instruction[5]}x{instruction[6]}",
        f"mi-wave-group:{instruction[7]}x{instruction[8]}",
        f"macro-area-log2:{int(math.floor(math.log2(macro_tile0 * macro_tile1)))}",
        f"macro-aspect-log2:{int(round(math.log2(macro_tile0 / macro_tile1)))}",
        f"cu-round-log2:{int(math.floor(math.log2(max(mechanics['cu_rounds'], 1.0))))}",
        f"cu-granularity:{_fraction_bucket(mechanics['cu_granularity'])}",
        f"wave-count:{mechanics['waves_per_workgroup']:g}",
        f"k-fill:{_fraction_bucket(mechanics['k_fill'])}",
        f"lds-fraction:{_fraction_bucket(mechanics['lds_fraction'], buckets=8)}",
        f"vgpr-fraction:{_fraction_bucket(mechanics['valu_vgpr_fraction'], buckets=8)}",
    }
    for name in PARAM_NAMES:
        if name != "MatrixInstruction":
            tokens.add(f"gene:{name}:{json.dumps(params[name], sort_keys=True, separators=(',', ':'))}")
    return frozenset(tokens)


def select_covering_cold_pool(
    candidates: Sequence[Candidate],
    *,
    shape: Shape,
    count: int,
    seed: int,
    effective_cu_count: int = GFX1151_EFFECTIVE_CU_COUNT,
    coverage_fraction: float = 0.80,
    prior_fraction: float = 0.10,
    precovered_tokens: set[str] | None = None,
) -> list[Candidate]:
    deduped = list({candidate.hash: candidate for candidate in candidates}.values())
    if count <= 0:
        return []
    if len(deduped) <= count:
        return deduped
    if coverage_fraction < 0.0 or prior_fraction < 0.0 or coverage_fraction + prior_fraction > 1.0:
        raise ValueError("cold-pool fractions must be non-negative and sum to at most one")

    rng = random.Random(seed)
    tokens = {
        candidate.hash: mechanical_coverage_tokens(
            candidate,
            shape,
            effective_cu_count=effective_cu_count,
        )
        for candidate in deduped
    }
    token_counts = Counter(token for candidate_tokens in tokens.values() for token in candidate_tokens)

    def token_priority(token: str) -> float:
        if token.startswith("mi-wave-") or token.startswith("macro-"):
            return 2.0
        if token.startswith("family:"):
            return 1.5
        return 1.0

    token_weights = {token: token_priority(token) / math.sqrt(frequency) for token, frequency in token_counts.items()}
    priors = {
        candidate.hash: mechanical_prior_score(
            candidate,
            shape,
            effective_cu_count=effective_cu_count,
        )
        for candidate in deduped
    }
    prior_values = list(priors.values())
    prior_min = min(prior_values)
    prior_span = max(prior_values) - prior_min
    normalized_priors = {
        candidate_hash: (score - prior_min) / prior_span if prior_span > 0.0 else 1.0
        for candidate_hash, score in priors.items()
    }
    tie_break = {candidate.hash: rng.random() for candidate in deduped}
    selected: list[Candidate] = []
    remaining = list(deduped)
    covered: set[str] = set(precovered_tokens or ())
    coverage_target = min(count, max(1, int(round(count * coverage_fraction))))
    prior_target = min(count - coverage_target, int(round(count * prior_fraction)))

    while remaining and len(selected) < coverage_target:

        def coverage_key(candidate: Candidate) -> tuple[float, float, float]:
            marginal = sum(token_weights[token] for token in tokens[candidate.hash] if token not in covered)
            quality_weight = 0.35 + 0.65 * normalized_priors[candidate.hash]
            return (marginal * quality_weight, priors[candidate.hash], tie_break[candidate.hash])

        chosen = max(remaining, key=coverage_key)
        selected.append(chosen)
        covered.update(tokens[chosen.hash])
        remaining.remove(chosen)

    prior_candidates = sorted(
        remaining,
        key=lambda candidate: (-priors[candidate.hash], tie_break[candidate.hash], candidate.hash),
    )
    for candidate in prior_candidates[:prior_target]:
        selected.append(candidate)
        remaining.remove(candidate)

    rng.shuffle(remaining)
    selected.extend(remaining[: count - len(selected)])
    return selected[:count]
