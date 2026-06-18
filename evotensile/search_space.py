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
    [16, 16, 16, 1, 1, 2, 6, 4, 1],  # MT128x96 artifact wave shape
    [16, 16, 16, 1, 1, 3, 4, 2, 2],  # MT96x128
    [16, 16, 16, 1, 1, 6, 2, 1, 4],  # MT96x128 artifact wave shape
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
    "WorkGroupMapping": [8, 16, 5, 4],
    "StaggerU": [32, 16, 8, 64, 0],
    "StaggerUMapping": [0, 1],
    "SourceSwap": [1, 0],
    "1LDSBuffer": [1, 0],
    "ClusterLocalRead": [0, 1],
    "TransposeLDS": [0, 2],
    "VectorWidthB": [2, 1],
    "GlobalReadVectorWidthA": [8, 4, 2, 1],
    "GlobalReadVectorWidthB": [8, 4, 2, 1],
    "StoreVectorWidth": [-1, 1],
    "StorePriorityOpt": [True, False],
    # 0 means TensileLite auto-selects the store batch size; nonzero values cap it explicitly.
    "NumElementsPerBatchStore": [10, 8, 12, 16, 4, 0, 14, 20, 24, 32, 1, 2, 6],
    "StoreSyncOpt": [0, 1, 2, 4],
    "GroupLoadStore": [False, True],
    "LdsBlockSizePerPadA": [0, 128, 256, 512, 1024, 2048],
    "LdsBlockSizePerPadB": [0, 128, 256, 512, 1024, 2048],
    "LdsPadA": [0, 8, 16, 4],
    "LdsPadB": [0, 8, 16, 4],
}

LDS_PAD_PROFILES: set[tuple[int, int, int, int, int]] = {
    (0, 0, 0, 0, 0),
    (0, 128, 128, 8, 8),
    (0, 512, 2048, 16, 16),
    (0, 1024, 1024, 16, 16),
    (0, 2048, 512, 16, 16),
    (2, 0, 0, 0, 0),
    (2, 128, 128, 4, 4),
    (2, 128, 128, 8, 8),
    (2, 128, 128, 16, 16),
    (2, 256, 256, 8, 8),
    (2, 256, 256, 16, 16),
    (2, 128, 0, 8, 0),
    (2, 0, 128, 0, 8),
}

FIXED_PARAMS: dict[str, Any] = {
    "KernelLanguage": "Assembly",
    "WavefrontSize": 32,
    "GlobalSplitUAlgorithm": "MultipleBuffer",
    "ScheduleGlobalRead": 1,
    "ScheduleLocalWrite": 1,
    "StaggerUStride": 256,
    "VectorWidthA": 1,
    "LocalReadVectorWidth": 16,
    "StoreRemapVectorWidth": 0,
    "MIArchVgpr": 1,
    "ExpandPointerSwap": 0,
    "AssertFree0ElementMultiple": 8,
    "AssertFree1ElementMultiple": 8,
    "AssertSummationElementMultiple": 16,
}


def macro_tile(mi: list[int]) -> tuple[int, int]:
    """Compute approximate macro tile dimensions from 9-field MatrixInstruction."""
    return (mi[0] * mi[5] * mi[7], mi[1] * mi[6] * mi[8])


def cheap_constraints(params: dict[str, Any]) -> bool:
    """Cheap pre-TensileLite constraints. TensileLite still performs authoritative validation."""
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

    lds_profile = (
        params["TransposeLDS"],
        params["LdsBlockSizePerPadA"],
        params["LdsBlockSizePerPadB"],
        params["LdsPadA"],
        params["LdsPadB"],
    )
    if lds_profile not in LDS_PAD_PROFILES:
        return False

    # The observed TLDS2 NT HHS configs use the PGR2/PLR0 path; broader TLDS2 mixing is left to later data.
    if params["TransposeLDS"] == 2:
        if params["PrefetchGlobalRead"] != 2 or params["PrefetchLocalRead"] != 0:
            return False
        if params["VectorWidthB"] != 1:
            return False

    # Store sync/load-store grouping are real codegen paths, but prior artifacts only exercised them with explicit batches.
    if params["StoreSyncOpt"] and params["NumElementsPerBatchStore"] == 0:
        return False
    if params["GroupLoadStore"] and (
        not params["StorePriorityOpt"] or params["NumElementsPerBatchStore"] not in {8, 10, 12}
    ):
        return False

    return True


def defaulted_params(overrides: dict[str, Any]) -> dict[str, Any]:
    params = dict(FIXED_PARAMS)
    for name, values in DOMAINS.items():
        params.setdefault(name, values[0])
    params.update(overrides)
    return params


def repair_linked_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Repair linked categorical genes that are invalid when mixed independently."""
    params = defaulted_params(overrides)
    lds_profile = (
        params["TransposeLDS"],
        params["LdsBlockSizePerPadA"],
        params["LdsBlockSizePerPadB"],
        params["LdsPadA"],
        params["LdsPadB"],
    )
    if lds_profile not in LDS_PAD_PROFILES:
        candidates = sorted(profile for profile in LDS_PAD_PROFILES if profile[0] == params["TransposeLDS"])
        transpose_lds, block_a, block_b, pad_a, pad_b = next(
            (profile for profile in candidates if profile[1:] == (0, 0, 0, 0)),
            candidates[0],
        )
        params.update(
            {
                "TransposeLDS": transpose_lds,
                "LdsBlockSizePerPadA": block_a,
                "LdsBlockSizePerPadB": block_b,
                "LdsPadA": pad_a,
                "LdsPadB": pad_b,
            }
        )
    if params["TransposeLDS"] == 2:
        params["PrefetchGlobalRead"] = 2
        params["PrefetchLocalRead"] = 0
        params["VectorWidthB"] = 1
    if params["PrefetchGlobalRead"] == 0 and params["1LDSBuffer"] == 0:
        params["1LDSBuffer"] = 1
    if params["DepthU"] == 64 and params["GlobalReadVectorWidthA"] == 1 and params["GlobalReadVectorWidthB"] == 1:
        params["GlobalReadVectorWidthA"] = 2
    if params["StoreSyncOpt"] and params["NumElementsPerBatchStore"] == 0:
        params["NumElementsPerBatchStore"] = 10
    if params["GroupLoadStore"]:
        params["StorePriorityOpt"] = True
        if params["NumElementsPerBatchStore"] not in {8, 10, 12}:
            params["NumElementsPerBatchStore"] = 10
    return params


def make_candidate(overrides: dict[str, Any], *, source: str, parents: Iterable[str] = ()) -> Candidate:
    params = defaulted_params(overrides)
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
        # TLDS2/padded LDS family seen in prior NT artifacts.
        {
            "MatrixInstruction": [16, 16, 16, 1, 1, 4, 4, 2, 2],
            "WorkGroup": [16, 16, 1],
            "DepthU": 16,
            "GlobalSplitU": 1,
            "PrefetchGlobalRead": 2,
            "PrefetchLocalRead": 0,
            "ScheduleIterAlg": 3,
            "WorkGroupMapping": 16,
            "StaggerU": 32,
            "StaggerUMapping": 0,
            "SourceSwap": 1,
            "1LDSBuffer": 0,
            "ClusterLocalRead": 0,
            "TransposeLDS": 2,
            "VectorWidthB": 1,
            "GlobalReadVectorWidthA": 8,
            "GlobalReadVectorWidthB": 8,
            "StoreVectorWidth": -1,
            "StorePriorityOpt": True,
            "NumElementsPerBatchStore": 16,
            "LdsBlockSizePerPadA": 128,
            "LdsBlockSizePerPadB": 128,
            "LdsPadA": 8,
            "LdsPadB": 8,
        },
        # Small square checked-in-style seed for the low-M/N pilot shapes.
        {
            "MatrixInstruction": [16, 16, 16, 1, 1, 1, 1, 2, 2],
            "WorkGroup": [16, 16, 1],
            "DepthU": 32,
            "GlobalSplitU": 1,
            "PrefetchGlobalRead": 2,
            "PrefetchLocalRead": 1,
            "ScheduleIterAlg": 3,
            "WorkGroupMapping": 8,
            "StaggerU": 32,
            "StaggerUMapping": 1,
            "SourceSwap": 0,
            "1LDSBuffer": 0,
            "ClusterLocalRead": 1,
            "TransposeLDS": 0,
            "VectorWidthB": 1,
            "GlobalReadVectorWidthA": 2,
            "GlobalReadVectorWidthB": 1,
            "StoreVectorWidth": 1,
            "StorePriorityOpt": False,
            "NumElementsPerBatchStore": 0,
            "LdsBlockSizePerPadA": 1024,
            "LdsBlockSizePerPadB": 1024,
            "LdsPadA": 16,
            "LdsPadB": 16,
        },
        # N-skinny / M-skinny checked-in-style seeds.
        {
            "MatrixInstruction": [16, 16, 16, 1, 1, 1, 1, 1, 4],
            "WorkGroup": [16, 16, 1],
            "DepthU": 32,
            "GlobalSplitU": 1,
            "PrefetchGlobalRead": 2,
            "PrefetchLocalRead": 1,
            "ScheduleIterAlg": 3,
            "WorkGroupMapping": 8,
            "StaggerU": 32,
            "StaggerUMapping": 1,
            "SourceSwap": 0,
            "1LDSBuffer": 0,
            "ClusterLocalRead": 1,
            "TransposeLDS": 0,
            "VectorWidthB": 1,
            "GlobalReadVectorWidthA": 2,
            "GlobalReadVectorWidthB": 8,
            "StoreVectorWidth": 1,
            "StorePriorityOpt": False,
            "NumElementsPerBatchStore": 0,
            "LdsBlockSizePerPadA": 512,
            "LdsBlockSizePerPadB": 2048,
            "LdsPadA": 16,
            "LdsPadB": 16,
        },
        {
            "MatrixInstruction": [16, 16, 16, 1, 1, 1, 1, 4, 1],
            "WorkGroup": [16, 16, 1],
            "DepthU": 32,
            "GlobalSplitU": 1,
            "PrefetchGlobalRead": 2,
            "PrefetchLocalRead": 1,
            "ScheduleIterAlg": 3,
            "WorkGroupMapping": 8,
            "StaggerU": 32,
            "StaggerUMapping": 1,
            "SourceSwap": 1,
            "1LDSBuffer": 0,
            "ClusterLocalRead": 1,
            "TransposeLDS": 0,
            "VectorWidthB": 1,
            "GlobalReadVectorWidthA": 8,
            "GlobalReadVectorWidthB": 2,
            "StoreVectorWidth": 1,
            "StorePriorityOpt": False,
            "NumElementsPerBatchStore": 0,
            "LdsBlockSizePerPadA": 2048,
            "LdsBlockSizePerPadB": 512,
            "LdsPadA": 16,
            "LdsPadB": 16,
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


def _random_linked_overrides(rng: random.Random) -> dict[str, Any]:
    overrides = {name: rng.choice(values) for name, values in DOMAINS.items()}
    transpose_lds, block_a, block_b, pad_a, pad_b = rng.choice(tuple(LDS_PAD_PROFILES))
    overrides.update(
        {
            "TransposeLDS": transpose_lds,
            "LdsBlockSizePerPadA": block_a,
            "LdsBlockSizePerPadB": block_b,
            "LdsPadA": pad_a,
            "LdsPadB": pad_b,
        }
    )
    if transpose_lds == 2:
        overrides["PrefetchGlobalRead"] = 2
        overrides["PrefetchLocalRead"] = 0
        overrides["VectorWidthB"] = 1
    if overrides["PrefetchGlobalRead"] == 0 and overrides["1LDSBuffer"] == 0:
        overrides["1LDSBuffer"] = 1
    if (
        overrides["DepthU"] == 64
        and overrides["GlobalReadVectorWidthA"] == 1
        and overrides["GlobalReadVectorWidthB"] == 1
    ):
        overrides[rng.choice(("GlobalReadVectorWidthA", "GlobalReadVectorWidthB"))] = 2
    if overrides["StoreSyncOpt"] and overrides["NumElementsPerBatchStore"] == 0:
        overrides["NumElementsPerBatchStore"] = rng.choice([8, 10, 12, 16])
    if overrides["GroupLoadStore"]:
        overrides["StorePriorityOpt"] = True
        overrides["NumElementsPerBatchStore"] = rng.choice([8, 10, 12])
    return overrides


def random_candidate(rng: random.Random, *, source: str = "random") -> Candidate:
    for _ in range(1000):
        try:
            return make_candidate(_random_linked_overrides(rng), source=source)
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
