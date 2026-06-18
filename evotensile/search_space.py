import random
from collections.abc import Iterable
from typing import Any

from .candidate import Candidate

# MatrixInstruction comments indicate approximate MacroTile when WorkGroup=[16,16,1].
MATRIX_INSTRUCTIONS: list[list[int]] = [
    [16, 16, 16, 1, 1, 1, 1, 2, 2],  # MT32x32
    [16, 16, 16, 1, 1, 1, 1, 1, 4],  # MT16x64
    [16, 16, 16, 1, 1, 1, 1, 4, 1],  # MT64x16
    [16, 16, 16, 1, 1, 2, 1, 2, 2],  # MT64x32
    [16, 16, 16, 1, 1, 1, 2, 2, 2],  # MT32x64
    [16, 16, 16, 1, 1, 2, 2, 2, 2],  # MT64x64
    [16, 16, 16, 1, 1, 2, 3, 2, 2],  # MT64x96
    [16, 16, 16, 1, 1, 3, 2, 2, 2],  # MT96x64
    [16, 16, 16, 1, 1, 3, 3, 2, 2],  # MT96x96
    [16, 16, 16, 1, 1, 4, 2, 2, 2],  # MT128x64
    [16, 16, 16, 1, 1, 2, 4, 2, 2],  # MT64x128
    [16, 16, 16, 1, 1, 4, 3, 2, 2],  # MT128x96
    [16, 16, 16, 1, 1, 3, 4, 2, 2],  # MT96x128
    [16, 16, 16, 1, 1, 4, 4, 2, 2],  # MT128x128
    [16, 16, 16, 1, 1, 4, 4, 4, 1],  # MT256x64
    [16, 16, 16, 1, 1, 4, 4, 2, 4],  # MT128x256
    [16, 16, 16, 1, 1, 4, 4, 4, 2],  # MT256x128
]

DOMAINS: dict[str, list[Any]] = {
    "MatrixInstruction": MATRIX_INSTRUCTIONS,
    "WorkGroup": [[16, 16, 1]],
    "DepthU": [16, 32, 64],
    "GlobalSplitU": [1, 2, 4],
    "PrefetchGlobalRead": [1, 2, 0],
    "PrefetchLocalRead": [1, 0],
    "ScheduleIterAlg": [2, 3, 1],
    "WorkGroupMapping": [8, 5],
    "StaggerU": [32, 8, 0],
    "StaggerUMapping": [0, 1],
    "SourceSwap": [1, 0],
    "1LDSBuffer": [1, 0],
    "ClusterLocalRead": [0, 1],
    "VectorWidthB": [2, 1],
    "GlobalReadVectorWidthA": [8, 4, 2, 1],
    "GlobalReadVectorWidthB": [8, 4, 2, 1],
    "StorePriorityOpt": [True, False],
    "NumElementsPerBatchStore": [4, 8, 10, 12, 16],
}

FIXED_PARAMS: dict[str, Any] = {
    "KernelLanguage": "Assembly",
    "WavefrontSize": 32,
    "GlobalSplitUAlgorithm": "MultipleBuffer",
    "ScheduleGlobalRead": 1,
    "ScheduleLocalWrite": 1,
    "StaggerUStride": 256,
    "TransposeLDS": 0,
    "VectorWidthA": 1,
    "LocalReadVectorWidth": 16,
    "StoreVectorWidth": -1,
    "StoreRemapVectorWidth": 0,
    "StoreSyncOpt": 0,
    "GroupLoadStore": False,
    "MIArchVgpr": 1,
    "ExpandPointerSwap": 0,
    "LdsBlockSizePerPadA": 0,
    "LdsBlockSizePerPadB": 0,
    "LdsPadA": 0,
    "LdsPadB": 0,
    "AssertFree0ElementMultiple": 8,
    "AssertFree1ElementMultiple": 8,
    "AssertSummationElementMultiple": 16,
}


def macro_tile(mi: list[int]) -> tuple[int, int]:
    """Compute approximate macro tile dimensions from 9-field MatrixInstruction."""
    return (mi[0] * mi[5] * mi[7], mi[1] * mi[6] * mi[8])


def cheap_constraints(params: dict[str, Any]) -> bool:
    """Cheap pre-Tensile constraints. Tensile still performs authoritative validation."""
    mi = params["MatrixInstruction"]
    mt0, mt1 = macro_tile(mi)
    if mt0 <= 0 or mt1 <= 0:
        return False
    if mt0 > 256 or mt1 > 256:
        return False

    # Keep broad-grid candidates conservative for now.
    if params["GlobalSplitU"] > 1 and params["DepthU"] < 32:
        return False

    # TensileLite rejects this combination: PGR0 already uses the single-LDS-buffer path.
    if params["PrefetchGlobalRead"] == 0 and params["1LDSBuffer"] == 0:
        return False

    # Very small vector reads with DU64 are usually not worth early random budget.
    if params["DepthU"] == 64 and params["GlobalReadVectorWidthA"] == 1 and params["GlobalReadVectorWidthB"] == 1:
        return False

    return True


def make_candidate(overrides: dict[str, Any], *, source: str, parents: Iterable[str] = ()) -> Candidate:
    params = dict(FIXED_PARAMS)
    params.update(overrides)
    if not cheap_constraints(params):
        raise ValueError(f"candidate failed cheap constraints: {params}")
    return Candidate(params=params, source=source, parent_hashes=tuple(parents))


def known_seed_candidates() -> list[Candidate]:
    """Initial deterministic seeds. These are deliberately conservative."""
    seeds: list[dict[str, Any]] = [
        # 8192^3-center-like rocBLAS baseline family.
        {
            "MatrixInstruction": [16, 16, 16, 1, 1, 4, 4, 2, 2],
            "WorkGroup": [16, 16, 1],
            "DepthU": 16,
            "GlobalSplitU": 1,
            "PrefetchGlobalRead": 1,
            "PrefetchLocalRead": 1,
            "ScheduleIterAlg": 2,
            "WorkGroupMapping": 8,
            "StaggerU": 32,
            "StaggerUMapping": 0,
            "SourceSwap": 1,
            "1LDSBuffer": 1,
            "ClusterLocalRead": 0,
            "VectorWidthB": 2,
            "GlobalReadVectorWidthA": 8,
            "GlobalReadVectorWidthB": 8,
            "StorePriorityOpt": True,
            "NumElementsPerBatchStore": 10,
        },
        # Slightly smaller K/pipeline variant.
        {
            "MatrixInstruction": [16, 16, 16, 1, 1, 4, 4, 2, 2],
            "WorkGroup": [16, 16, 1],
            "DepthU": 32,
            "GlobalSplitU": 1,
            "PrefetchGlobalRead": 1,
            "PrefetchLocalRead": 1,
            "ScheduleIterAlg": 3,
            "WorkGroupMapping": 8,
            "StaggerU": 8,
            "StaggerUMapping": 0,
            "SourceSwap": 1,
            "1LDSBuffer": 1,
            "ClusterLocalRead": 0,
            "VectorWidthB": 2,
            "GlobalReadVectorWidthA": 8,
            "GlobalReadVectorWidthB": 8,
        },
        # M-wide.
        {
            "MatrixInstruction": [16, 16, 16, 1, 1, 4, 2, 2, 2],
            "WorkGroup": [16, 16, 1],
            "DepthU": 32,
            "GlobalSplitU": 1,
            "PrefetchGlobalRead": 1,
            "PrefetchLocalRead": 1,
            "ScheduleIterAlg": 2,
            "WorkGroupMapping": 8,
            "StaggerU": 32,
            "StaggerUMapping": 0,
            "SourceSwap": 1,
            "1LDSBuffer": 1,
            "ClusterLocalRead": 0,
            "VectorWidthB": 2,
            "GlobalReadVectorWidthA": 8,
            "GlobalReadVectorWidthB": 8,
        },
        # N-wide / N-heavy.
        {
            "MatrixInstruction": [16, 16, 16, 1, 1, 2, 4, 2, 2],
            "WorkGroup": [16, 16, 1],
            "DepthU": 32,
            "GlobalSplitU": 1,
            "PrefetchGlobalRead": 1,
            "PrefetchLocalRead": 1,
            "ScheduleIterAlg": 2,
            "WorkGroupMapping": 8,
            "StaggerU": 32,
            "StaggerUMapping": 0,
            "SourceSwap": 1,
            "1LDSBuffer": 1,
            "ClusterLocalRead": 0,
            "VectorWidthB": 2,
            "GlobalReadVectorWidthA": 8,
            "GlobalReadVectorWidthB": 8,
        },
        # GSU probe.
        {
            "MatrixInstruction": [16, 16, 16, 1, 1, 3, 3, 2, 2],
            "WorkGroup": [16, 16, 1],
            "DepthU": 32,
            "GlobalSplitU": 2,
            "PrefetchGlobalRead": 1,
            "PrefetchLocalRead": 1,
            "ScheduleIterAlg": 2,
            "WorkGroupMapping": 5,
            "StaggerU": 8,
            "StaggerUMapping": 0,
            "SourceSwap": 1,
            "1LDSBuffer": 1,
            "ClusterLocalRead": 0,
            "VectorWidthB": 2,
            "GlobalReadVectorWidthA": 8,
            "GlobalReadVectorWidthB": 8,
        },
    ]
    return [make_candidate(seed, source="seed") for seed in seeds]


def documented_winner_candidate() -> Candidate:
    """Current documented 8192^3 FP16 NT HHS winner, for ground-truth checks only. May change in future."""
    return make_candidate(
        {
            "MatrixInstruction": [16, 16, 16, 1, 1, 4, 4, 2, 2],
            "WorkGroup": [16, 16, 1],
            "DepthU": 16,
            "GlobalSplitU": 1,
            "PrefetchGlobalRead": 1,
            "PrefetchLocalRead": 1,
            "ScheduleIterAlg": 3,
            "WorkGroupMapping": 8,
            "StaggerU": 32,
            "StaggerUMapping": 0,
            "SourceSwap": 1,
            "1LDSBuffer": 1,
            "ClusterLocalRead": 0,
            "VectorWidthB": 2,
            "GlobalReadVectorWidthA": 8,
            "GlobalReadVectorWidthB": 8,
            "StorePriorityOpt": False,
            "NumElementsPerBatchStore": 10,
        },
        source="documented_winner",
    )


def random_candidate(rng: random.Random, *, source: str = "random") -> Candidate:
    for _ in range(1000):
        overrides = {name: rng.choice(values) for name, values in DOMAINS.items()}
        try:
            return make_candidate(overrides, source=source)
        except ValueError:
            continue
    raise RuntimeError("failed to generate a valid random candidate")


def random_candidates(count: int, *, seed: int = 1) -> list[Candidate]:
    rng = random.Random(seed)
    out: dict[str, Candidate] = {}
    while len(out) < count:
        cand = random_candidate(rng)
        out[cand.hash] = cand
    return list(out.values())


def seed_and_random_candidates(num_random: int, *, seed: int = 1) -> list[Candidate]:
    out: dict[str, Candidate] = {}
    for cand in known_seed_candidates() + random_candidates(num_random, seed=seed):
        out[cand.hash] = cand
    return list(out.values())
