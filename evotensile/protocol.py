from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from .candidate import stable_hash

VALIDATION_PROTOCOL_VERSION = 1

BENCHMARK_PROTOCOL_OVERRIDE_FIELDS = (
    "num_warmups",
    "num_benchmarks",
    "enqueues_per_sync",
    "syncs_per_benchmark",
    "num_elements_to_validate",
    "validation_backend",
)

BENCHMARK_PROTOCOL_KEYS = (
    "KernelTime",
    "PreciseKernelTime",
    "NumWarmups",
    "EnqueuesPerSync",
    "SyncsPerBenchmark",
    "SleepPercent",
    "HardwareMonitor",
    "DataInitTypeA",
    "DataInitTypeB",
    "DataInitTypeC",
    "DataInitTypeD",
    "DataInitTypeAlpha",
    "DataInitTypeBeta",
    "DataInitTypeBias",
    "DataInitTypeScaleAlphaVec",
    "CEqualD",
    "PredictionThreshold",
    "GranularityThreshold",
    "SkipSlowSolutionRatio",
    "ParallelGpuExecution",
)


@dataclass(frozen=True)
class BenchmarkProtocol:
    role: Literal["main", "probe"] = "main"
    kernel_time: bool = True
    precise_kernel_time: bool = True
    num_warmups: int = 10
    num_benchmarks: int = 10
    enqueues_per_sync: int = 10
    syncs_per_benchmark: int = 1
    sleep_percent: int = 0
    hardware_monitor: bool = False
    num_elements_to_validate: int = -1
    data_init_type_a: int = 3
    data_init_type_b: int = 3
    data_init_type_c: int = 3
    data_init_type_d: int = 0
    data_init_type_alpha: int = 2
    data_init_type_beta: int = 2
    data_init_type_bias: int = 3
    data_init_type_scale_alpha_vec: int = 3
    c_equal_d: bool = False
    prediction_threshold: float = 2.0
    granularity_threshold: float = 0.0
    skip_slow_solution_ratio: float = 0.0
    parallel_gpu_execution: int = 1
    validation_backend: str = "hipblaslt"

    def __post_init__(self) -> None:
        if self.role not in {"main", "probe"}:
            raise ValueError("role must be one of: main, probe")
        if self.num_warmups < 0:
            raise ValueError("num_warmups must be non-negative")
        if self.num_benchmarks <= 0:
            raise ValueError("num_benchmarks must be positive")
        if self.enqueues_per_sync <= 0:
            raise ValueError("enqueues_per_sync must be positive")
        if self.syncs_per_benchmark <= 0:
            raise ValueError("syncs_per_benchmark must be positive")
        if self.sleep_percent != 0:
            raise ValueError("SleepPercent is not supported by the structured runner")
        if self.hardware_monitor:
            raise ValueError("HardwareMonitor is not supported by the structured runner")
        if self.parallel_gpu_execution != 1:
            raise ValueError("ParallelGpuExecution must be 1 for serial structured benchmarking")
        if self.validation_backend not in {"cpu", "hipblaslt"}:
            raise ValueError("validation_backend must be one of: cpu, hipblaslt")

    def with_overrides(self, **overrides: Any) -> "BenchmarkProtocol":
        clean = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **clean)

    def global_parameters(self) -> dict[str, Any]:
        return {
            "KernelTime": self.kernel_time,
            "PreciseKernelTime": self.precise_kernel_time,
            "NumWarmups": self.num_warmups,
            "NumBenchmarks": self.num_benchmarks,
            "EnqueuesPerSync": self.enqueues_per_sync,
            "SyncsPerBenchmark": self.syncs_per_benchmark,
            "SleepPercent": self.sleep_percent,
            "HardwareMonitor": self.hardware_monitor,
            "NumElementsToValidate": self.num_elements_to_validate,
            "DataInitTypeA": self.data_init_type_a,
            "DataInitTypeB": self.data_init_type_b,
            "DataInitTypeC": self.data_init_type_c,
            "DataInitTypeD": self.data_init_type_d,
            "DataInitTypeAlpha": self.data_init_type_alpha,
            "DataInitTypeBeta": self.data_init_type_beta,
            "DataInitTypeBias": self.data_init_type_bias,
            "DataInitTypeScaleAlphaVec": self.data_init_type_scale_alpha_vec,
            "CEqualD": self.c_equal_d,
            "PredictionThreshold": self.prediction_threshold,
            "GranularityThreshold": self.granularity_threshold,
            "SkipSlowSolutionRatio": self.skip_slow_solution_ratio,
            "ParallelGpuExecution": self.parallel_gpu_execution,
        }

    def runner_parameters(self) -> dict[str, Any]:
        return {"ValidationBackend": self.validation_backend}

    def identity_parameters(self) -> dict[str, Any]:
        return {
            "BenchmarkRole": self.role,
            **{key: value for key, value in self.global_parameters().items() if key in BENCHMARK_PROTOCOL_KEYS},
        }

    def validation_identity_parameters(self) -> dict[str, Any]:
        return {
            "ValidationProtocolVersion": VALIDATION_PROTOCOL_VERSION,
            "ValidationBackend": self.validation_backend,
            "NumElementsToValidate": self.num_elements_to_validate,
            "DataInitTypeA": self.data_init_type_a,
            "DataInitTypeB": self.data_init_type_b,
            "DataInitTypeC": self.data_init_type_c,
            "DataInitTypeD": self.data_init_type_d,
            "DataInitTypeAlpha": self.data_init_type_alpha,
            "DataInitTypeBeta": self.data_init_type_beta,
            "DataInitTypeBias": self.data_init_type_bias,
            "DataInitTypeScaleAlphaVec": self.data_init_type_scale_alpha_vec,
            "CEqualD": self.c_equal_d,
        }

    def protocol_hash(self) -> str:
        return stable_hash(self.identity_parameters(), prefix="bproto_")[:23]

    def validation_protocol_hash(self) -> str:
        return stable_hash(self.validation_identity_parameters(), prefix="vproto_")[:23]


def apply_benchmark_protocol_overrides(
    protocol: BenchmarkProtocol,
    overrides: Mapping[str, Any],
) -> BenchmarkProtocol:
    return protocol.with_overrides(**{field: overrides.get(field) for field in BENCHMARK_PROTOCOL_OVERRIDE_FIELDS})


def _format_global_value(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return repr(value)
    return str(value)


def global_parameter_items(values: Mapping[str, object]) -> list[str]:
    return [f"{key}={_format_global_value(value)}" for key, value in values.items()]


DEFAULT_BENCHMARK_PROTOCOL = BenchmarkProtocol()
