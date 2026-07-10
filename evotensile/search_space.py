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
    "NumElementsPerBatchStore": [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32],
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
NT_HHS_MAX_VGPR = 256
NT_HHS_MAX_LDS_BYTES = 65536
NT_HHS_RANDOM_VALU_VGPR_HEADROOM = 192
NT_HHS_WORKSPACE_SIZE_PER_ELEM_C = 4
NT_HHS_MAX_GSU_WORKSPACE_BYTES = 128 * 1024 * 1024
_RANDOM_TLDS2_HEADROOM_MATRIX_INSTRUCTIONS: tuple[list[int], ...] = tuple(
    instruction
    for instruction in MATRIX_INSTRUCTIONS
    if _matrix_instruction_macro_tile(instruction)[0] <= 256
    and _matrix_instruction_macro_tile(instruction)[1] <= 256
    and instruction[5] * instruction[6] <= NT_HHS_RANDOM_VALU_VGPR_HEADROOM
)


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


def _num_threads(params: dict[str, Any]) -> int:
    instruction = params["MatrixInstruction"]
    work_group = params["WorkGroup"]
    wavefront_size = params["WavefrontSize"]
    matrix_inst_n = instruction[1]
    matrix_inst_bn = instruction[4]
    mi_wave_group0 = instruction[7]
    mi_wave_group1 = instruction[8]
    return (
        (wavefront_size // matrix_inst_n)
        * mi_wave_group0
        * matrix_inst_n
        * matrix_inst_bn
        * mi_wave_group1
        * work_group[2]
    )


def _total_global_read_vectors(params: dict[str, Any], suffix: str) -> int:
    macro_tile_value = macro_tile(params["MatrixInstruction"])[0 if suffix == "A" else 1]
    return macro_tile_value * params["DepthU"] // params[f"GlobalReadVectorWidth{suffix}"]


def _repair_global_read_vector_width(params: dict[str, Any], suffix: str) -> None:
    total_elements = macro_tile(params["MatrixInstruction"])[0 if suffix == "A" else 1] * params["DepthU"]
    valid = [
        choice for choice in DOMAINS[f"GlobalReadVectorWidth{suffix}"] if choice > 0 and total_elements % choice == 0
    ]
    valid = [choice for choice in valid if (total_elements // choice) % _num_threads(params) == 0]
    if valid:
        params[f"GlobalReadVectorWidth{suffix}"] = max(valid)


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length() if value > 0 else 0


def _lds_num_bytes_ab(params: dict[str, Any], suffix: str) -> tuple[int, int]:
    macro_tile_value = macro_tile(params["MatrixInstruction"])[0 if suffix == "A" else 1]
    depth = params["DepthU"]
    pad = params[f"LdsPad{suffix}"]
    block_size_per_pad = params[f"LdsBlockSizePerPad{suffix}"]
    unroll_major = params["TransposeLDS"] == 2
    if unroll_major:
        bytes_used = int((depth + pad) * macro_tile_value * 2)
    else:
        bytes_used = int(depth * (macro_tile_value + pad) * 2)
    if block_size_per_pad:
        bytes_used = int((depth * macro_tile_value * 2) / block_size_per_pad * (block_size_per_pad + pad * 2))
    return bytes_used, _round_up_to_multiple(bytes_used, 128)


def _uses_one_lds_buffer(params: dict[str, Any]) -> bool:
    return bool(params["1LDSBuffer"]) or params["ScheduleIterAlg"] == 2


def _nt_hhs_lsp(params: dict[str, Any], suffix: str) -> int:
    macro_tile_value = macro_tile(params["MatrixInstruction"])[0 if suffix == "A" else 1]
    grvw = params[f"GlobalReadVectorWidth{suffix}"]
    total_vectors = macro_tile_value * params["DepthU"] // grvw
    num_loads = total_vectors // _num_threads(params)
    total_vectors_coalesced = macro_tile_value // grvw
    total_elements_perpendicular = params["DepthU"]
    for num_loads_coalesced in range(1, num_loads + 1):
        num_loads_perpendicular = num_loads // num_loads_coalesced
        if (
            num_loads % num_loads_coalesced == 0
            and total_vectors_coalesced % num_loads_coalesced == 0
            and total_elements_perpendicular % num_loads_perpendicular == 0
        ):
            return (total_elements_perpendicular + num_loads_perpendicular - 1) // num_loads_perpendicular
    return 0


def _invalid_unroll_major_lds_pad_block(params: dict[str, Any], suffix: str) -> bool:
    block_size_per_pad = params[f"LdsBlockSizePerPad{suffix}"]
    if params["TransposeLDS"] != 2 or not block_size_per_pad:
        return False
    depth_bytes = params["DepthU"] * 2
    if block_size_per_pad % depth_bytes != 0:
        return False
    pad_period = block_size_per_pad // depth_bytes
    lsp = _nt_hhs_lsp(params, suffix)
    return bool(lsp and pad_period % lsp != 0 and lsp % pad_period != 0)


def _nt_hhs_lds_bytes(params: dict[str, Any]) -> int:
    bytes_a, aligned_a = _lds_num_bytes_ab(params, "A")
    bytes_b, aligned_b = _lds_num_bytes_ab(params, "B")
    if params["PrefetchGlobalRead"] >= 3:
        num_lds_blocks = params["PrefetchGlobalRead"]
        one_lds_buffer = False
    else:
        one_lds_buffer = _uses_one_lds_buffer(params)
        num_lds_blocks = 1 if one_lds_buffer else 2

    if not params["PrefetchGlobalRead"]:
        return aligned_a + bytes_b
    if one_lds_buffer:
        return aligned_a + aligned_b

    offset_block = aligned_a + aligned_b
    rounded_offset_block = _next_power_of_two(offset_block)
    store_swap_addr = offset_block > 0 and offset_block + rounded_offset_block > NT_HHS_MAX_LDS_BYTES
    if offset_block > 0 and not store_swap_addr and num_lds_blocks == 2:
        offset_block = rounded_offset_block
    lds_offset_b_block = offset_block + aligned_a
    return (num_lds_blocks - 2) * offset_block + lds_offset_b_block + bytes_b


def _is_tlds2(params: dict[str, Any]) -> bool:
    return params["TransposeLDS"] == 2


def _apply_tlds2_required_params(params: dict[str, Any]) -> None:
    params["TransposeLDS"] = 2
    params["PrefetchGlobalRead"] = 2
    params["PrefetchLocalRead"] = 0
    params["VectorWidthB"] = 1


def _tlds2_requires_pgr2_plr0(params: dict[str, Any]) -> bool:
    return params["PrefetchGlobalRead"] != 2 or params["PrefetchLocalRead"] != 0


def _tlds2_requires_vector_width_b_1(params: dict[str, Any]) -> bool:
    return params["VectorWidthB"] != 1


def _repair_tlds2_lds_pad_blocks(params: dict[str, Any]) -> None:
    if not _is_tlds2(params):
        return
    lds_block_multiple = params["DepthU"] * 2
    for suffix in ("A", "B"):
        key = f"LdsBlockSizePerPad{suffix}"
        if params[key] and (
            params[key] % lds_block_multiple != 0 or _invalid_unroll_major_lds_pad_block(params, suffix)
        ):
            params[key] = 0


def _valid_tlds2_lds_pad_block_choices(params: dict[str, Any], suffix: str) -> list[int]:
    return [
        value
        for value in DOMAINS[f"LdsBlockSizePerPad{suffix}"]
        if not value
        or (
            value % (params["DepthU"] * 2) == 0
            and not _invalid_unroll_major_lds_pad_block({**params, f"LdsBlockSizePerPad{suffix}": value}, suffix)
        )
    ]


def _repair_lds_footprint(params: dict[str, Any]) -> None:
    for key in ("LdsPadA", "LdsPadB", "LdsBlockSizePerPadA", "LdsBlockSizePerPadB"):
        if params[key]:
            params[key] = 0
    for depth in sorted((value for value in DOMAINS["DepthU"] if value <= params["DepthU"]), reverse=True):
        params["DepthU"] = depth
        if _nt_hhs_lds_bytes(params) <= NT_HHS_MAX_LDS_BYTES:
            break


def _thread_tile(params: dict[str, Any]) -> tuple[int, int]:
    instruction = params["MatrixInstruction"]
    matrix_inst_m = instruction[0]
    matrix_inst_n = instruction[1]
    matrix_inst_bm = instruction[3]
    matrix_inst_bn = instruction[4]
    mi_wave_tile0 = instruction[5]
    mi_wave_tile1 = instruction[6]
    if matrix_inst_m == 4:
        return mi_wave_tile0 * NT_HHS_WMMA_V1_OUTPUT_VECTOR_WIDTH, mi_wave_tile1
    outputs_per_wave = matrix_inst_m * matrix_inst_n // params["WavefrontSize"]
    return matrix_inst_bm * mi_wave_tile0 * outputs_per_wave, matrix_inst_bn * mi_wave_tile1


def _c_accumulator_vgprs(params: dict[str, Any]) -> int:
    thread_tile0, thread_tile1 = _thread_tile(params)
    return thread_tile0 * thread_tile1


def _ranked_valid_matrix_instructions(
    params: dict[str, Any], predicate: Any, *, rng: random.Random | None = None
) -> list[list[int]]:
    current_macro_tile = macro_tile(params["MatrixInstruction"])
    valid_instructions = [
        instruction for instruction in MATRIX_INSTRUCTIONS if predicate({**params, "MatrixInstruction": instruction})
    ]
    ranked = sorted(
        valid_instructions,
        key=lambda instruction: (
            abs(macro_tile(instruction)[0] - current_macro_tile[0])
            + abs(macro_tile(instruction)[1] - current_macro_tile[1]),
            -macro_tile(instruction)[0] * macro_tile(instruction)[1],
        ),
    )
    return ranked[: min(len(ranked), 8)] if rng else ranked


def _repair_c_accumulator_vgprs(params: dict[str, Any], *, rng: random.Random | None = None) -> None:
    if _c_accumulator_vgprs(params) <= NT_HHS_MAX_VGPR:
        return

    ranked = _ranked_valid_matrix_instructions(
        params,
        lambda candidate_params: _c_accumulator_vgprs(candidate_params) <= NT_HHS_MAX_VGPR,
        rng=rng,
    )
    if ranked:
        params["MatrixInstruction"] = rng.choice(ranked) if rng else ranked[0]


def _effective_inner_unroll(params: dict[str, Any]) -> int:
    if params["ScheduleIterAlg"] == 2:
        return max(1, params["DepthU"] // params["MatrixInstruction"][2])
    return 1


def _loop_iters(params: dict[str, Any]) -> int:
    loop_unroll = params["DepthU"] // params["WorkGroup"][2]
    loop_unroll //= _effective_inner_unroll(params)
    return max(1, loop_unroll // params["MatrixInstruction"][2])


def _effective_prefetch_local_read(params: dict[str, Any]) -> int:
    return 1 if params["ScheduleIterAlg"] == 2 else params["PrefetchLocalRead"]


def _num_iters_plr(params: dict[str, Any]) -> int:
    loop_iters = _loop_iters(params)
    local_read_vector_width = params.get("LocalReadVectorWidthA", params["LocalReadVectorWidth"])
    mi_input_per_thread = params["MatrixInstruction"][2]
    prefetch_local_read = _effective_prefetch_local_read(params)
    if local_read_vector_width >= mi_input_per_thread:
        wider_local_read = max(local_read_vector_width // mi_input_per_thread, 1)
        divisor = loop_iters // wider_local_read
        return prefetch_local_read % divisor if divisor else 0
    return prefetch_local_read % loop_iters


def _num_vgpr_buffer(params: dict[str, Any]) -> int:
    loop_iters = _loop_iters(params)
    prefetch_local_read = _effective_prefetch_local_read(params)
    if params["ClusterLocalRead"]:
        return loop_iters
    if loop_iters == 1 and prefetch_local_read >= loop_iters and _num_iters_plr(params) == 0:
        return 1
    return prefetch_local_read + 1


def _lrvw_tile(params: dict[str, Any], suffix: str) -> int:
    return 1 if params["TransposeLDS"] == 2 else params[f"VectorWidth{suffix}"]


def _ab_valu_vgprs(params: dict[str, Any], suffix: str) -> int:
    instruction = params["MatrixInstruction"]
    mi_wave_tile = instruction[5 if suffix == "A" else 6]
    mi_input_per_thread = instruction[2]
    vgprs_per_block = mi_wave_tile * mi_input_per_thread * 2 // 4
    inner_unroll = _effective_inner_unroll(params)
    vgprs = vgprs_per_block * _num_vgpr_buffer(params) * inner_unroll
    if _lrvw_tile(params, suffix) > 1:
        vgprs = vgprs_per_block * inner_unroll
    return int(vgprs)


def _valu_vgpr_lower_bound(params: dict[str, Any]) -> int:
    return _c_accumulator_vgprs(params) + _ab_valu_vgprs(params, "A") + _ab_valu_vgprs(params, "B")


def _gsu_workspace_bytes(params: dict[str, Any], shape: Shape) -> int:
    if params["GlobalSplitU"] <= 1:
        return 0
    return shape.m * shape.n * shape.batch * NT_HHS_WORKSPACE_SIZE_PER_ELEM_C * params["GlobalSplitU"]


def _repair_valu_vgpr_lower_bound(params: dict[str, Any], *, rng: random.Random | None = None) -> None:
    if _valu_vgpr_lower_bound(params) <= NT_HHS_MAX_VGPR:
        return

    ranked = _ranked_valid_matrix_instructions(
        params,
        lambda candidate_params: _valu_vgpr_lower_bound(candidate_params) <= NT_HHS_MAX_VGPR,
        rng=rng,
    )
    if ranked:
        params["MatrixInstruction"] = rng.choice(ranked) if rng else ranked[0]


def _unique_choices(values: Sequence[Any]) -> tuple[Any, ...]:
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return tuple(out)


def _repair_random_valu_vgpr_headroom(params: dict[str, Any], *, rng: random.Random) -> None:
    if _valu_vgpr_lower_bound(params) <= NT_HHS_RANDOM_VALU_VGPR_HEADROOM:
        return

    repair_genes = (
        "DepthU",
        "ScheduleIterAlg",
        "PrefetchLocalRead",
        "ClusterLocalRead",
        "TransposeLDS",
        "VectorWidthA",
        "VectorWidthB",
    )
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for name in repair_genes:
        for value in DOMAINS[name]:
            if value == params[name]:
                continue
            variant = _repair_linked_params({**params, name: value})
            lower_bound = _valu_vgpr_lower_bound(variant)
            candidates.append((int(lower_bound > NT_HHS_RANDOM_VALU_VGPR_HEADROOM), lower_bound, variant))

    ranked_matrices = _ranked_valid_matrix_instructions(
        params,
        lambda candidate_params: _valu_vgpr_lower_bound(candidate_params) <= NT_HHS_RANDOM_VALU_VGPR_HEADROOM,
        rng=None,
    )
    for instruction in ranked_matrices[:16]:
        variant = _repair_linked_params({**params, "MatrixInstruction": instruction})
        lower_bound = _valu_vgpr_lower_bound(variant)
        candidates.append((int(lower_bound > NT_HHS_RANDOM_VALU_VGPR_HEADROOM), lower_bound, variant))

    if not candidates:
        return
    candidates.sort(key=lambda item: (item[0], item[1]))
    best_class = candidates[0][:2]
    tied = [variant for invalid, lower_bound, variant in candidates if (invalid, lower_bound) == best_class]
    params.update(rng.choice(tied))


def _repair_matrix_instruction_dependent_params(params: dict[str, Any]) -> None:
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


def _repair_linked_params(params: dict[str, Any], *, rng: random.Random | None = None) -> dict[str, Any]:
    params = dict(params)
    _repair_c_accumulator_vgprs(params, rng=rng)
    _repair_valu_vgpr_lower_bound(params, rng=rng)
    _repair_matrix_instruction_dependent_params(params)
    if params["TransposeLDS"] == 1:
        params["TransposeLDS"] = 0
    if params["1LDSBuffer"] or params["ScheduleIterAlg"] == 2:
        if params["PrefetchGlobalRead"] == 0:
            params["PrefetchGlobalRead"] = 1
    if params["1LDSBuffer"] and params["ScheduleIterAlg"] == 1 and params["ScheduleLocalWrite"]:
        params["ScheduleIterAlg"] = 2
    if params["GlobalSplitU"] > 1 and params["DepthU"] < 32:
        params["DepthU"] = 32
    for suffix in ("A", "B"):
        if _total_global_read_vectors(params, suffix) % _num_threads(params) != 0:
            _repair_global_read_vector_width(params, suffix)
    if _is_tlds2(params):
        _apply_tlds2_required_params(params)
        _repair_tlds2_lds_pad_blocks(params)
    if _nt_hhs_lds_bytes(params) > NT_HHS_MAX_LDS_BYTES:
        _repair_lds_footprint(params)
    if _valu_vgpr_lower_bound(params) > NT_HHS_MAX_VGPR:
        _repair_valu_vgpr_lower_bound(params, rng=rng)
        _repair_matrix_instruction_dependent_params(params)
        if _nt_hhs_lds_bytes(params) > NT_HHS_MAX_LDS_BYTES:
            _repair_lds_footprint(params)
    for suffix in ("A", "B"):
        if _total_global_read_vectors(params, suffix) % _num_threads(params) != 0:
            _repair_global_read_vector_width(params, suffix)
    _repair_tlds2_lds_pad_blocks(params)
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
    if params["ScheduleIterAlg"] == 2 and params["PrefetchGlobalRead"] == 0:
        reasons.append(
            InvalidReason(
                "nt_hhs.sia2_forces_one_lds_buffer.rejects_pgr0",
                "TensileLite ScheduleIterAlg=2 forces 1LDSBuffer=1 before rejecting PrefetchGlobalRead=0.",
                ("ScheduleIterAlg", "PrefetchGlobalRead"),
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

    for suffix in ("A", "B"):
        if _total_global_read_vectors(params, suffix) % _num_threads(params) != 0:
            reasons.append(
                InvalidReason(
                    f"nt_hhs.global_read_vectors_{suffix.lower()}.requires_num_threads_divisor",
                    "TensileLite rejects GlobalReadVectorWidth values when total global-read vectors are not divisible by NumThreads.",
                    ("MatrixInstruction", "WorkGroup", "DepthU", f"GlobalReadVectorWidth{suffix}"),
                    source="solutionstructs",
                )
            )

    lds_bytes = _nt_hhs_lds_bytes(params)
    if lds_bytes > NT_HHS_MAX_LDS_BYTES:
        reasons.append(
            InvalidReason(
                "nt_hhs.lds.footprint_exceeds_max_lds",
                "TensileLite rejects kernels whose calculated LDS footprint exceeds MaxLDS=65536 bytes.",
                (
                    "MatrixInstruction",
                    "DepthU",
                    "PrefetchGlobalRead",
                    "1LDSBuffer",
                    "ScheduleIterAlg",
                    "TransposeLDS",
                    "LdsPadA",
                    "LdsPadB",
                    "LdsBlockSizePerPadA",
                    "LdsBlockSizePerPadB",
                ),
                source="solutionstructs",
            )
        )

    c_accumulator_vgprs = _c_accumulator_vgprs(params)
    if c_accumulator_vgprs > NT_HHS_MAX_VGPR:
        reasons.append(
            InvalidReason(
                "nt_hhs.vgpr.c_accumulators_exceed_max_vgpr",
                "TensileLite rejects kernels when C accumulator VGPRs alone exceed gfx1151 MaxVgpr=256.",
                ("MatrixInstruction", "WavefrontSize"),
                source="kernelwriter",
            )
        )
    elif _valu_vgpr_lower_bound(params) > NT_HHS_MAX_VGPR:
        reasons.append(
            InvalidReason(
                "nt_hhs.vgpr.valu_lower_bound_exceeds_max_vgpr",
                "TensileLite KernelWriter allocates C accumulators plus mandatory A/B VALU registers before enforcing gfx1151 MaxVgpr=256.",
                (
                    "MatrixInstruction",
                    "DepthU",
                    "ScheduleIterAlg",
                    "PrefetchLocalRead",
                    "ClusterLocalRead",
                    "VectorWidthA",
                    "VectorWidthB",
                    "TransposeLDS",
                    "WavefrontSize",
                ),
                source="kernelwriter",
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
            if _invalid_unroll_major_lds_pad_block(params, suffix):
                reasons.append(
                    InvalidReason(
                        f"nt_hhs.lds.tlds2_block_size_{suffix.lower()}_must_align_lsp",
                        "TensileLite rejects TLDS2 padding when LdsBlockSizePerPad/(DepthU*2) and LSP are not mutually divisible.",
                        (
                            "MatrixInstruction",
                            "WorkGroup",
                            "DepthU",
                            "TransposeLDS",
                            f"GlobalReadVectorWidth{suffix}",
                            f"LdsBlockSizePerPad{suffix}",
                        ),
                        source="solutionstructs",
                    )
                )
        if _tlds2_requires_pgr2_plr0(params):
            reasons.append(
                InvalidReason(
                    "nt_hhs.tlds2.requires_pgr2_plr0",
                    "Observed NT TLDS2 path requires PrefetchGlobalRead=2 and PrefetchLocalRead=0.",
                    ("TransposeLDS", "PrefetchGlobalRead", "PrefetchLocalRead"),
                    source="solutionstructs",
                )
            )
        if _tlds2_requires_vector_width_b_1(params):
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
        if _gsu_workspace_bytes(params, shape) > NT_HHS_MAX_GSU_WORKSPACE_BYTES:
            reasons.append(
                InvalidReason(
                    "nt_hhs.shape.gsu_workspace_exceeds_max",
                    "TensileLite WorkspaceCheck rejects GSU kernels whose C workspace exceeds 128 MiB.",
                    ("GlobalSplitU",),
                    source="taskpredicate",
                    shape_dependent=True,
                )
            )
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


def cheap_constraints(params: dict[str, Any], *, shape: Shape | None = None) -> bool:
    """Cheap pre-TensileLite constraints. TensileLite still performs authoritative validation."""
    return not explain_invalid_nt_hhs(params, shape=shape)


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


def _random_tlds0_direct_overrides(
    rng: random.Random, *, target_shapes: Sequence[Shape] | None = None
) -> dict[str, Any]:
    params = defaulted_params(_random_domain_overrides(rng))
    params["TransposeLDS"] = 0
    params["GlobalSplitU"] = 1 if target_shapes else rng.choice(DOMAINS["GlobalSplitU"])
    params = _repair_linked_params(params, rng=rng)
    if _valu_vgpr_lower_bound(params) > NT_HHS_RANDOM_VALU_VGPR_HEADROOM:
        _repair_random_valu_vgpr_headroom(params, rng=rng)
    params["TransposeLDS"] = 0
    return _repair_linked_params(params, rng=rng)


def _random_tlds2_direct_overrides(
    rng: random.Random, *, target_shapes: Sequence[Shape] | None = None
) -> dict[str, Any]:
    params = defaulted_params(_random_domain_overrides(rng))
    params["MatrixInstruction"] = rng.choice(_RANDOM_TLDS2_HEADROOM_MATRIX_INSTRUCTIONS)
    params["DepthU"] = rng.choice(DOMAINS["DepthU"])
    params["GlobalSplitU"] = 1 if target_shapes else rng.choice(DOMAINS["GlobalSplitU"])
    _apply_tlds2_required_params(params)
    params["1LDSBuffer"] = 1
    params["ClusterLocalRead"] = rng.choice(DOMAINS["ClusterLocalRead"])
    params["ScheduleIterAlg"] = rng.choice(DOMAINS["ScheduleIterAlg"])
    params["WorkGroup"] = rng.choice(DOMAINS["WorkGroup"])
    _repair_matrix_instruction_dependent_params(params)
    for suffix in ("A", "B"):
        params[f"GlobalReadVectorWidth{suffix}"] = rng.choice(DOMAINS[f"GlobalReadVectorWidth{suffix}"])
    params = _repair_linked_params(params, rng=rng)
    for suffix in ("A", "B"):
        choices = _valid_tlds2_lds_pad_block_choices(params, suffix)
        params[f"LdsBlockSizePerPad{suffix}"] = rng.choice(choices) if choices else 0
    if _nt_hhs_lds_bytes(params) > NT_HHS_MAX_LDS_BYTES:
        _repair_lds_footprint(params)
    if _valu_vgpr_lower_bound(params) > NT_HHS_RANDOM_VALU_VGPR_HEADROOM:
        _repair_random_valu_vgpr_headroom(params, rng=rng)
    return _repair_linked_params(params, rng=rng)


def random_candidate(
    rng: random.Random,
    *,
    source: str = "random",
    target_shapes: Sequence[Shape] | None = None,
    transpose_lds: int | None = None,
) -> Candidate:
    if transpose_lds not in {None, 0, 2}:
        raise ValueError("transpose_lds must be one of: 0, 2, or None")
    for _ in range(1000):
        try:
            if transpose_lds == 0:
                params = _random_tlds0_direct_overrides(rng, target_shapes=target_shapes)
            elif transpose_lds == 2:
                params = _random_tlds2_direct_overrides(rng, target_shapes=target_shapes)
            else:
                branch = rng.choice((0, 2))
                params = (
                    _random_tlds0_direct_overrides(rng, target_shapes=target_shapes)
                    if branch == 0
                    else _random_tlds2_direct_overrides(rng, target_shapes=target_shapes)
                )
            candidate = make_candidate(params, source=source)
            if target_shapes and not all(
                cheap_constraints(candidate.canonical_params(), shape=shape) for shape in target_shapes
            ):
                continue
            return candidate
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
