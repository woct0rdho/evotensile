import math
from collections.abc import Sequence

from evotensile.candidate import Shape

_SHAPE_DISTANCE_KEYS = (
    "log2_m",
    "log2_n",
    "log2_k",
    "log2_m_over_n",
    "log2_k_over_m",
    "log2_k_over_n",
)


def shape_distance(left: Shape, right: Shape) -> float:
    left_features = left.features()
    right_features = right.features()
    return math.sqrt(sum((left_features[key] - right_features[key]) ** 2 for key in _SHAPE_DISTANCE_KEYS))


def shape_feature_delta(target: Shape, other: Shape) -> list[float]:
    target_features = target.features()
    other_features = other.features()
    return [other_features[key] - target_features[key] for key in _SHAPE_DISTANCE_KEYS]


def representative_shape_order(shapes: Sequence[Shape]) -> list[Shape]:
    remaining = {shape.id: shape for shape in shapes}
    if not remaining:
        return []
    first = max(
        remaining.values(),
        key=lambda shape: (
            sum(shape_distance(shape, other) for other in remaining.values()),
            shape.id,
        ),
    )
    ordered = [first]
    del remaining[first.id]
    while remaining:
        chosen = max(
            remaining.values(),
            key=lambda shape: (
                min(shape_distance(shape, selected) for selected in ordered),
                shape.id,
            ),
        )
        ordered.append(chosen)
        del remaining[chosen.id]
    return ordered
