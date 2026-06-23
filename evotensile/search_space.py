import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .candidate import Candidate, Shape


def _matrix_instruction_macro_tile(instruction: Sequence[int]) -> tuple[int, int]:
    return (instruction[0] * instruction[5] * instruction[7], instruction[1] * instruction[6] * instruction[8])


def _transpose_matrix_instruction(instruction: Sequence[int]) -> tuple[int, ...]:
    return (
        instruction[1],
        instruction[0],
        instruction[2],
        instruction[3],
        instruction[4],
        instruction[6],
        instruction[5],
        instruction[8],
        instruction[7],
    )


def _with_symmetric_macro_tiles(base_instructions: Sequence[Sequence[int]]) -> list[list[int]]:
    instructions: list[tuple[int, ...]] = []
    seen_instructions: set[tuple[int, ...]] = set()
    seen_macro_tiles: set[tuple[int, int]] = set()
    for instruction in base_instructions:
        item = tuple(instruction)
        if item in seen_instructions:
            continue
        instructions.append(item)
        seen_instructions.add(item)
        seen_macro_tiles.add(_matrix_instruction_macro_tile(item))

    for instruction in tuple(instructions):
        transposed = _transpose_matrix_instruction(instruction)
        transposed_macro_tile = _matrix_instruction_macro_tile(transposed)
        if transposed_macro_tile in seen_macro_tiles or transposed in seen_instructions:
            continue
        instructions.append(transposed)
        seen_instructions.add(transposed)
        seen_macro_tiles.add(transposed_macro_tile)

    return [list(instruction) for instruction in instructions]


_MATRIX_INSTRUCTION_PREFIX = (16, 16, 16, 1, 1)


def _matrix_instruction(shape: Sequence[int]) -> tuple[int, ...]:
    return (*_MATRIX_INSTRUCTION_PREFIX, *shape)


# MatrixInstruction shapes are (MIWaveTile0, MIWaveTile1, MIWaveGroup0, MIWaveGroup1).
_DEFAULT_MATRIX_INSTRUCTION_SHAPE = (1, 1, 2, 2)  # MT32x32
_MI_WAVE_TILE0_VALUES = (1, 2, 3, 4, 5, 6, 7, 8, 9)
_MI_WAVE_TILE1_VALUES = (1, 2, 3, 4, 5, 6, 7, 8, 9)
_MI_WAVE_GROUPS = ((1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (2, 4), (4, 1), (4, 2))


def _matrix_instruction_shape_macro_tile(shape: Sequence[int]) -> tuple[int, int]:
    return (_MATRIX_INSTRUCTION_PREFIX[0] * shape[0] * shape[2], _MATRIX_INSTRUCTION_PREFIX[1] * shape[1] * shape[3])


def _base_matrix_instruction_shapes() -> tuple[tuple[int, int, int, int], ...]:
    shapes = [_DEFAULT_MATRIX_INSTRUCTION_SHAPE]
    seen = {_DEFAULT_MATRIX_INSTRUCTION_SHAPE}
    for wave_tile0 in _MI_WAVE_TILE0_VALUES:
        for wave_tile1 in _MI_WAVE_TILE1_VALUES:
            for wave_group0, wave_group1 in _MI_WAVE_GROUPS:
                shape = (wave_tile0, wave_tile1, wave_group0, wave_group1)
                macro_tile0, macro_tile1 = _matrix_instruction_shape_macro_tile(shape)
                if shape in seen or macro_tile0 > 256 or macro_tile1 > 256:
                    continue
                shapes.append(shape)
                seen.add(shape)
    return tuple(shapes)


_BASE_MATRIX_INSTRUCTION_SHAPES = _base_matrix_instruction_shapes()
_BASE_MATRIX_INSTRUCTIONS = tuple(_matrix_instruction(shape) for shape in _BASE_MATRIX_INSTRUCTION_SHAPES)
MATRIX_INSTRUCTIONS: list[list[int]] = _with_symmetric_macro_tiles(_BASE_MATRIX_INSTRUCTIONS)

# First value is default
DOMAINS: dict[str, list[Any]] = {
    "MatrixInstruction": MATRIX_INSTRUCTIONS,
    "WorkGroup": [[16, 16, 1], [16, 2, 1], [16, 4, 1], [16, 8, 1], [32, 2, 1], [32, 4, 1], [64, 2, 1], [64, 4, 1]],
    "DepthU": [16, 32, 64, 128],
    "GlobalSplitU": [1, 2, 4],
    "PrefetchGlobalRead": [1, 0, 2],
    "PrefetchLocalRead": [1, 0],
    "ScheduleIterAlg": [2, 1, 3],
    "WorkGroupMapping": [8, 4, 5, 16],
    "StaggerU": [32, 0, 8, 16, 64],
    "StaggerUMapping": [0, 1],
    "SourceSwap": [True, False],
    "1LDSBuffer": [1, 0],
    "ClusterLocalRead": [0, 1],
    "TransposeLDS": [0, 1, 2],
    "VectorWidthA": [1, 2, 4, 8],
    "VectorWidthB": [2, 1, 4, 8],
    "GlobalReadVectorWidthA": [8, 1, 2, 4],
    "GlobalReadVectorWidthB": [8, 1, 2, 4],
    "StoreVectorWidth": [-1, 1, 2, 4, 8],
    "StaggerUStride": [256, 0, 64, 128],
    "ExpandPointerSwap": [False, True],
    "AssertFree0ElementMultiple": [8, 1],
    "AssertFree1ElementMultiple": [8, 1],
    "AssertSummationElementMultiple": [16, 1],
    "StorePriorityOpt": [True, False],
    # 0 means TensileLite auto-selects the store batch size; nonzero values cap it explicitly.
    "NumElementsPerBatchStore": [10, 0, 1, 2, 4, 6, 8, 12, 14, 16, 20, 24, 32],
    "StoreSyncOpt": [0, 1, 2, 4],
    "GroupLoadStore": [False, True],
    "LdsBlockSizePerPadA": [0, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192],
    "LdsBlockSizePerPadB": [0, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192],
    "LdsPadA": [0, 4, 8, 16],
    "LdsPadB": [0, 4, 8, 16],
}

FIXED_PARAMS: dict[str, Any] = {
    "KernelLanguage": "Assembly",
    "WavefrontSize": 32,
    "GlobalSplitUAlgorithm": "MultipleBuffer",
    "ScheduleGlobalRead": 1,
    "ScheduleLocalWrite": 1,
    "LocalReadVectorWidth": 16,
    "StoreRemapVectorWidth": 0,
    "MIArchVgpr": True,
}

macro_tile = _matrix_instruction_macro_tile

STORE_SYNC_OPT_CHOICES = (0, 2, 4)
STORE_SYNC_BATCH_CHOICES = (10, 12)
GROUP_LOAD_STORE_BATCH_CHOICES = (12,)
NT_HHS_WMMA_V1_OUTPUT_VECTOR_WIDTH = 1


@dataclass(frozen=True)
class InvalidReason:
    rule_id: str
    message: str
    params: tuple[str, ...]
    source: str
    shape_dependent: bool = False


def _largest_divisor_choice(value: int, choices: Sequence[int]) -> int:
    valid = [choice for choice in choices if choice > 0 and value % choice == 0]
    return max(valid) if valid else 1


def _repair_linked_params(params: dict[str, Any], *, rng: random.Random | None = None) -> dict[str, Any]:
    params = dict(params)
    mi_wave_tile0 = params["MatrixInstruction"][5]
    mi_wave_tile1 = params["MatrixInstruction"][6]
    if mi_wave_tile0 % params["VectorWidthA"] != 0:
        params["VectorWidthA"] = _largest_divisor_choice(mi_wave_tile0, DOMAINS["VectorWidthA"])
    if mi_wave_tile1 % params["VectorWidthB"] != 0:
        params["VectorWidthB"] = _largest_divisor_choice(mi_wave_tile1, DOMAINS["VectorWidthB"])
    if params["StoreVectorWidth"] != -1:
        if params["SourceSwap"] and params["VectorWidthA"] % params["StoreVectorWidth"] != 0:
            params["StoreVectorWidth"] = -1
        elif (
            not params["SourceSwap"]
            and params["VectorWidthA"] * NT_HHS_WMMA_V1_OUTPUT_VECTOR_WIDTH % params["StoreVectorWidth"] != 0
        ):
            params["StoreVectorWidth"] = -1
    if params["TransposeLDS"] == 1:
        params["TransposeLDS"] = 0
    if params["1LDSBuffer"]:
        if params["PrefetchGlobalRead"] == 0:
            params["PrefetchGlobalRead"] = 1
        if params["ScheduleIterAlg"] == 1 and params["ScheduleLocalWrite"]:
            params["ScheduleIterAlg"] = 2
    if params["GlobalSplitU"] > 1 and params["DepthU"] < 32:
        params["DepthU"] = 32
    if params["TransposeLDS"] == 2:
        lds_block_multiple = params["DepthU"] * 2
        for suffix in ("A", "B"):
            key = f"LdsBlockSizePerPad{suffix}"
            if params[key] and params[key] % lds_block_multiple != 0:
                params[key] = 0
        params["PrefetchGlobalRead"] = 2
        params["PrefetchLocalRead"] = 0
        params["VectorWidthB"] = 1
    if params["StoreSyncOpt"] not in STORE_SYNC_OPT_CHOICES:
        params["StoreSyncOpt"] = rng.choice(STORE_SYNC_OPT_CHOICES) if rng else 0
    if params["StoreSyncOpt"] and params["NumElementsPerBatchStore"] not in STORE_SYNC_BATCH_CHOICES:
        params["NumElementsPerBatchStore"] = rng.choice(STORE_SYNC_BATCH_CHOICES) if rng else 10
    if params["GroupLoadStore"]:
        params["StoreSyncOpt"] = 4
        params["StorePriorityOpt"] = True
        if params["NumElementsPerBatchStore"] not in GROUP_LOAD_STORE_BATCH_CHOICES:
            params["NumElementsPerBatchStore"] = rng.choice(GROUP_LOAD_STORE_BATCH_CHOICES) if rng else 12
    return params


def explain_invalid_nt_hhs(params: dict[str, Any], *, shape: Shape | None = None) -> list[InvalidReason]:
    """Explain known-disallowed NT HHS parameter combinations.

    This is a broad-domain negative rule layer. TensileLite remains authoritative for
    final acceptance, rejection, and normalization.
    """
    reasons: list[InvalidReason] = []
    mt0, mt1 = macro_tile(params["MatrixInstruction"])
    if mt0 <= 0 or mt1 <= 0:
        reasons.append(
            InvalidReason(
                "nt_hhs.macro_tile.non_positive",
                "Macro tile dimensions must be positive.",
                ("MatrixInstruction",),
                source="solutionstructs",
            )
        )
    if mt0 > 256 or mt1 > 256:
        reasons.append(
            InvalidReason(
                "nt_hhs.macro_tile.exceeds_256",
                "Macro tile dimensions above 256 are currently rejected before TensileLite build.",
                ("MatrixInstruction",),
                source="heuristic",
            )
        )

    mi_wave_tile0 = params["MatrixInstruction"][5]
    mi_wave_tile1 = params["MatrixInstruction"][6]
    if mi_wave_tile0 % params["VectorWidthA"] != 0:
        reasons.append(
            InvalidReason(
                "nt_hhs.mi_wave_tile0.requires_vector_width_a_divisor",
                "TensileLite rejects MatrixInstruction MIWaveTile0 values that are not multiples of VectorWidthA.",
                ("MatrixInstruction", "VectorWidthA"),
                source="solutionstructs",
            )
        )
    if mi_wave_tile1 % params["VectorWidthB"] != 0:
        reasons.append(
            InvalidReason(
                "nt_hhs.mi_wave_tile1.requires_vector_width_b_divisor",
                "TensileLite rejects MatrixInstruction MIWaveTile1 values that are not multiples of VectorWidthB.",
                ("MatrixInstruction", "VectorWidthB"),
                source="solutionstructs",
            )
        )
    if params["StoreVectorWidth"] != -1:
        if params["SourceSwap"] and params["VectorWidthA"] % params["StoreVectorWidth"] != 0:
            reasons.append(
                InvalidReason(
                    "nt_hhs.source_swap.requires_store_vector_width_divides_vector_width_a",
                    "TensileLite rejects SourceSwap store vector widths that do not divide VectorWidthA.",
                    ("SourceSwap", "VectorWidthA", "StoreVectorWidth"),
                    source="solutionstructs",
                )
            )
        if (
            not params["SourceSwap"]
            and params["VectorWidthA"] * NT_HHS_WMMA_V1_OUTPUT_VECTOR_WIDTH % params["StoreVectorWidth"] != 0
        ):
            reasons.append(
                InvalidReason(
                    "nt_hhs.non_source_swap.requires_store_vector_width_divides_miovw",
                    "TensileLite rejects non-SourceSwap store vector widths that do not divide VectorWidthA * MIOutputVectorWidth.",
                    ("SourceSwap", "VectorWidthA", "StoreVectorWidth"),
                    source="solutionstructs",
                )
            )
    if params["TransposeLDS"] == 1:
        reasons.append(
            InvalidReason(
                "nt_hhs.tlds1.rejects_nt_tlua_tlub",
                "TensileLite rejects TransposeLDS=1 when both NT operands use TLU layout.",
                ("TransposeLDS",),
                source="solutionstructs",
            )
        )
    if params["1LDSBuffer"] and params["PrefetchGlobalRead"] == 0:
        reasons.append(
            InvalidReason(
                "nt_hhs.one_lds_buffer.rejects_pgr0",
                "TensileLite rejects 1LDSBuffer with PrefetchGlobalRead=0 because PGR=0 already uses one LDS buffer.",
                ("1LDSBuffer", "PrefetchGlobalRead"),
                source="solutionstructs",
            )
        )
    if params["1LDSBuffer"] and params["ScheduleLocalWrite"] and params["ScheduleIterAlg"] not in {2, 3}:
        reasons.append(
            InvalidReason(
                "nt_hhs.one_lds_buffer.requires_sia2_or_sia3_with_slw",
                "TensileLite rejects 1LDSBuffer with scheduled local writes unless ScheduleIterAlg is 2 or 3.",
                ("1LDSBuffer", "ScheduleIterAlg", "ScheduleLocalWrite"),
                source="solutionstructs",
            )
        )

    if params["GlobalSplitU"] > 1 and params["DepthU"] < 32:
        reasons.append(
            InvalidReason(
                "nt_hhs.gsu.requires_depthu_ge_32",
                "GlobalSplitU greater than 1 is kept to DepthU >= 32.",
                ("GlobalSplitU", "DepthU"),
                source="heuristic",
            )
        )

    if params["TransposeLDS"] == 2:
        lds_block_multiple = params["DepthU"] * 2
        for suffix in ("A", "B"):
            value = params[f"LdsBlockSizePerPad{suffix}"]
            if value and value % lds_block_multiple != 0:
                reasons.append(
                    InvalidReason(
                        f"nt_hhs.lds.tlds2_block_size_{suffix.lower()}_must_divide_depthu_bytes",
                        "TensileLite rejects TLDS2 block-size-per-pad values that are not multiples of DepthU times 2-byte FP16 LDS elements.",
                        ("TransposeLDS", "DepthU", f"LdsBlockSizePerPad{suffix}"),
                        source="solutionstructs",
                    )
                )
        if params["PrefetchGlobalRead"] != 2 or params["PrefetchLocalRead"] != 0:
            reasons.append(
                InvalidReason(
                    "nt_hhs.tlds2.requires_pgr2_plr0",
                    "Observed NT TLDS2 path requires PrefetchGlobalRead=2 and PrefetchLocalRead=0.",
                    ("TransposeLDS", "PrefetchGlobalRead", "PrefetchLocalRead"),
                    source="solutionstructs",
                )
            )
        if params["VectorWidthB"] != 1:
            reasons.append(
                InvalidReason(
                    "nt_hhs.tlds2.requires_vector_width_b_1",
                    "Observed NT TLDS2 path requires VectorWidthB=1.",
                    ("TransposeLDS", "VectorWidthB"),
                    source="solutionstructs",
                )
            )

    if params["StoreSyncOpt"] not in STORE_SYNC_OPT_CHOICES:
        reasons.append(
            InvalidReason(
                "nt_hhs.store_sync.unsupported_opt",
                "StoreSyncOpt is outside the supported NT HHS store-sync set.",
                ("StoreSyncOpt",),
                source="solutionstructs",
            )
        )
    if params["StoreSyncOpt"] and params["NumElementsPerBatchStore"] not in STORE_SYNC_BATCH_CHOICES:
        reasons.append(
            InvalidReason(
                "nt_hhs.store_sync.requires_batch_choice",
                "StoreSyncOpt requires an observed explicit NumElementsPerBatchStore value.",
                ("StoreSyncOpt", "NumElementsPerBatchStore"),
                source="solutionstructs",
            )
        )
    if params["GroupLoadStore"] and params["StoreSyncOpt"] != 4:
        reasons.append(
            InvalidReason(
                "nt_hhs.group_load_store.requires_store_sync_4",
                "GroupLoadStore requires StoreSyncOpt=4 in observed NT HHS configs.",
                ("GroupLoadStore", "StoreSyncOpt"),
                source="solutionstructs",
            )
        )
    if params["GroupLoadStore"] and not params["StorePriorityOpt"]:
        reasons.append(
            InvalidReason(
                "nt_hhs.group_load_store.requires_store_priority",
                "GroupLoadStore requires StorePriorityOpt=True in observed NT HHS configs.",
                ("GroupLoadStore", "StorePriorityOpt"),
                source="solutionstructs",
            )
        )
    if params["GroupLoadStore"] and params["NumElementsPerBatchStore"] not in GROUP_LOAD_STORE_BATCH_CHOICES:
        reasons.append(
            InvalidReason(
                "nt_hhs.group_load_store.requires_batch_choice",
                "GroupLoadStore requires the observed explicit batch-store value.",
                ("GroupLoadStore", "NumElementsPerBatchStore"),
                source="solutionstructs",
            )
        )

    if shape is not None:
        if shape.k % params["AssertSummationElementMultiple"] != 0:
            reasons.append(
                InvalidReason(
                    "nt_hhs.shape.assert_summation_multiple",
                    "K is not divisible by AssertSummationElementMultiple.",
                    ("AssertSummationElementMultiple",),
                    source="schema",
                    shape_dependent=True,
                )
            )
        if shape.m % params["AssertFree0ElementMultiple"] != 0:
            reasons.append(
                InvalidReason(
                    "nt_hhs.shape.assert_free0_multiple",
                    "M is not divisible by AssertFree0ElementMultiple.",
                    ("AssertFree0ElementMultiple",),
                    source="schema",
                    shape_dependent=True,
                )
            )
        if shape.n % params["AssertFree1ElementMultiple"] != 0:
            reasons.append(
                InvalidReason(
                    "nt_hhs.shape.assert_free1_multiple",
                    "N is not divisible by AssertFree1ElementMultiple.",
                    ("AssertFree1ElementMultiple",),
                    source="schema",
                    shape_dependent=True,
                )
            )

    return reasons


def cheap_constraints(params: dict[str, Any]) -> bool:
    """Cheap pre-TensileLite constraints. TensileLite still performs authoritative validation."""
    return not explain_invalid_nt_hhs(params)


def defaulted_params(overrides: dict[str, Any]) -> dict[str, Any]:
    params = dict(FIXED_PARAMS)
    for name, values in DOMAINS.items():
        params.setdefault(name, values[0])
    params.update(overrides)
    return params


def repair_linked_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Repair linked categorical genes that are invalid when mixed independently."""
    return _repair_linked_params(defaulted_params(overrides))


def make_candidate(overrides: dict[str, Any], *, source: str, parents: Iterable[str] = ()) -> Candidate:
    params = defaulted_params(overrides)
    if not cheap_constraints(params):
        raise ValueError(f"candidate failed cheap constraints: {params}")
    return Candidate(params=params, source=source, parent_hashes=tuple(parents))


def _random_domain_overrides(rng: random.Random) -> dict[str, Any]:
    return {name: rng.choice(values) for name, values in DOMAINS.items()}


def _random_mechanical_overrides(rng: random.Random) -> dict[str, Any]:
    return _repair_linked_params(defaulted_params(_random_domain_overrides(rng)), rng=rng)


def random_candidate(rng: random.Random, *, source: str = "random") -> Candidate:
    for _ in range(1000):
        try:
            return make_candidate(_random_mechanical_overrides(rng), source=source)
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
