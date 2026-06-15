from collections.abc import Iterable

from .candidate import Shape

PILOT_M = [512, 640, 896, 1024]
PILOT_N = [128, 256, 512, 768, 1024]
PILOT_BATCH = [1]
PILOT_K = [256, 512, 1024, 2048, 4096]


def pilot_100_shapes() -> list[Shape]:
    return [
        Shape(m=m, n=n, batch=batch, k=k) for m in PILOT_M for n in PILOT_N for batch in PILOT_BATCH for k in PILOT_K
    ]


def parse_shape(text: str) -> Shape:
    """Parse M,N,B,K or MxNxBxK."""
    cleaned = text.lower().replace("x", ",").replace(" ", "")
    parts = [int(p) for p in cleaned.split(",") if p]
    if len(parts) != 4:
        raise ValueError(f"shape must have 4 fields M,N,batch,K: {text!r}")
    return Shape(m=parts[0], n=parts[1], batch=parts[2], k=parts[3])


def shape_bucket(shape: Shape) -> str:
    """Simple initial bucket label for batching/search policy."""
    aspect = shape.m / shape.n
    if aspect >= 4:
        aspect_label = "m_wide"
    elif aspect <= 0.25:
        aspect_label = "n_wide"
    else:
        aspect_label = "squareish"

    if shape.k <= 512:
        k_label = "k_small"
    elif shape.k >= 2048:
        k_label = "k_large"
    else:
        k_label = "k_mid"

    return f"{aspect_label}_{k_label}"


def group_by_bucket(shapes: Iterable[Shape]) -> dict[str, list[Shape]]:
    buckets: dict[str, list[Shape]] = {}
    for shape in shapes:
        buckets.setdefault(shape_bucket(shape), []).append(shape)
    return buckets
