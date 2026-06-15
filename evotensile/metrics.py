from .candidate import Shape


def gemm_flops(shape: Shape) -> int:
    return 2 * shape.m * shape.n * shape.k * shape.batch


def tflops_from_us(shape: Shape, time_us: float) -> float:
    if time_us <= 0:
        return 0.0
    return gemm_flops(shape) / (time_us * 1e-6) / 1e12


def gflops_from_us(shape: Shape, time_us: float) -> float:
    return tflops_from_us(shape, time_us) * 1000.0
