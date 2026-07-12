import re
from itertools import product

from .candidate import Shape

_SHAPE_ID_RE = re.compile(r"^m(?P<m>\d+)_n(?P<n>\d+)_b(?P<batch>\d+)_k(?P<k>\d+)$")

PILOT_M = [512, 640, 896, 1024]
PILOT_N = [128, 256, 512, 768, 1024]
PILOT_BATCH = [1]
PILOT_K = [256, 512, 1024, 2048, 4096]

COMFY_1135_DENSE_M = [16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
COMFY_1135_DENSE_N = [16, 32, 64, 128, 256, 512, 640, 768, 1024]
COMFY_1135_CORE_K = [16, 32, 64, 256, 512, 1024, 2048, 4096]
COMFY_1135_MID_N = [1280, 1536, 1792, 2304, 2560, 2816, 3072, 3328, 3584, 3840]
COMFY_1135_FULL_K = [16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 8192]
COMFY_1135_TAIL_M = [16, 32, 64, 128, 256, 384, 768, 1536, 3072, 8192]
COMFY_1135_TAIL_N = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
COMFY_1135_TAIL_K = [16, 512, 1024, 2048, 4096, 8192]


def pilot_100_shapes() -> list[Shape]:
    return [
        Shape(m=m, n=n, batch=batch, k=k) for m in PILOT_M for n in PILOT_N for batch in PILOT_BATCH for k in PILOT_K
    ]


def comfy_nt_1135_shapes() -> list[Shape]:
    blocks = (
        product(COMFY_1135_DENSE_M, COMFY_1135_DENSE_N, [1], COMFY_1135_CORE_K),
        product([16, 32, 3072], COMFY_1135_MID_N, [1], [16, 32, 3072]),
        product([16, 32], [2048, 4096], [1], COMFY_1135_FULL_K),
        product([3072, 4096], [16, 32], [1], COMFY_1135_CORE_K),
        product([2048], [16, 32, 2048], [1], [16, 32, 2048]),
        product([4096], [4096], [1], [16, 32, 1024, 4096]),
        product([3072], [768], [1], [16, 32, 3072]),
        product([8192], [16, 2048, 4096, 8192], [1], COMFY_1135_TAIL_K),
        product(COMFY_1135_TAIL_M, [8192], [1], COMFY_1135_TAIL_K),
        product(COMFY_1135_TAIL_M, COMFY_1135_TAIL_N, [1], [8192]),
    )
    coordinates = {coordinate for block in blocks for coordinate in block}
    return [Shape(m=m, n=n, batch=batch, k=k) for m, n, batch, k in sorted(coordinates)]


def parse_shape(text: str) -> Shape:
    """Parse M,N,B,K or MxNxBxK."""
    cleaned = text.lower().replace("x", ",").replace(" ", "")
    parts = [int(p) for p in cleaned.split(",") if p]
    if len(parts) != 4:
        raise ValueError(f"shape must have 4 fields M,N,batch,K: {text!r}")
    return Shape(m=parts[0], n=parts[1], batch=parts[2], k=parts[3])


def shape_from_id(shape_id: str) -> Shape:
    match = _SHAPE_ID_RE.match(shape_id)
    if not match:
        raise ValueError(f"invalid EvoTensile shape id: {shape_id!r}")
    return Shape(
        m=int(match.group("m")),
        n=int(match.group("n")),
        batch=int(match.group("batch")),
        k=int(match.group("k")),
    )
