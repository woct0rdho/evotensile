import math
from collections.abc import Sequence

from evotensile.candidate import Candidate, Shape
from evotensile.search.mechanics import candidate_shape_mechanics


def predicted_candidate_prepare_weight(
    candidate: Candidate,
    shape: Shape,
    *,
    effective_cu_count: int,
) -> float:
    mechanics = candidate_shape_mechanics(candidate, shape, effective_cu_count=effective_cu_count)
    return (
        1.0
        + 0.40 * mechanics["valu_vgpr_fraction"]
        + 0.25 * mechanics["lds_fraction"]
        + 0.10 * math.log2(max(1.0, mechanics["wave_tile_area"]))
        + 0.05 * math.log2(max(1.0, mechanics["wave_group_size"]))
    )


def predicted_batch_prepare_weight(
    candidates: Sequence[Candidate],
    shapes: Sequence[Shape],
    *,
    effective_cu_count: int,
) -> float:
    if not candidates or not shapes:
        return 0.0
    return sum(
        sum(
            predicted_candidate_prepare_weight(
                candidate,
                shape,
                effective_cu_count=effective_cu_count,
            )
            for shape in shapes
        )
        / len(shapes)
        for candidate in candidates
    )
