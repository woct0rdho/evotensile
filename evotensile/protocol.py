from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .candidate import stable_hash

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
        if self.validation_backend not in {"cpu", "hipblaslt", "none"}:
            raise ValueError("validation_backend must be one of: cpu, hipblaslt, none")

    @property
    def samples_per_pair(self) -> int:
        return self.num_benchmarks

    @property
    def launches_per_sample(self) -> int:
        return self.enqueues_per_sync * self.syncs_per_benchmark

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
        values = {**self.global_parameters(), **self.runner_parameters()}
        return {key: value for key, value in values.items() if key in BENCHMARK_PROTOCOL_KEYS}

    def protocol_hash(self) -> str:
        return stable_hash(self.identity_parameters(), prefix="bproto_")[:23]


def benchmark_protocol_hash(protocol: BenchmarkProtocol) -> str:
    return protocol.protocol_hash()


def _format_global_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return repr(value)
    return str(value)


def global_parameter_items(values: Mapping[str, Any]) -> list[str]:
    return [f"{key}={_format_global_value(value)}" for key, value in values.items()]


DEFAULT_BENCHMARK_PROTOCOL = BenchmarkProtocol()
