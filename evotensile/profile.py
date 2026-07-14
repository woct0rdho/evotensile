from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .candidate import Shape, stable_hash
from .protocol import (
    DEFAULT_BENCHMARK_PROTOCOL,
    BenchmarkProtocol,
    ProtocolParameterValue,
    global_parameter_items,
)
from .shapes import comfy_nt_1135_shapes, pilot_100_shapes
from .yaml_writer import FP16_NT_HHS_PROBLEM_TYPE, LIBRARY_LOGIC_GRIDBASED_GFX1151

BASE_GLOBAL_PARAMETERS: dict[str, ProtocolParameterValue] = {
    "MinimumRequiredVersion": "5.0.0",
    "RuntimeLanguage": "HIP",
    "ValidationMaxToPrint": 4,
    "ValidationPrintValids": False,
    "ForceRedoBenchmarkProblems": True,
    "ForceRedoLibraryLogic": True,
    "LibraryFormat": "yaml",
    "LogicFormat": "yaml",
}


@dataclass(frozen=True)
class TargetProfile:
    name: str
    problem_type: dict[str, Any]
    library_logic: dict[str, Any]
    default_protocol: BenchmarkProtocol
    shapes_fn: Callable[[], list[Shape]]
    environment_compatibility_tag: str
    compute_unit_count: int
    workgroup_processor_count: int
    compute_units_per_workgroup_processor: int
    max_no_cache_candidate_batch_size: int = 32
    default_shape_batch_size: int = 100
    default_prepare_workers: int = 32
    default_prepare_wave_batches: int = 32
    default_validation_workers: int = 1
    default_surrogate_jobs: int = 1
    default_runner_bin: str = "./build/evotensile-structured-runner"
    default_build_timeout_s: float | None = 1800.0
    default_runner_timeout_s: float | None = 600.0

    def __post_init__(self) -> None:
        if not self.environment_compatibility_tag.strip():
            raise ValueError("environment compatibility tag must not be empty")
        if self.compute_unit_count <= 0 or self.workgroup_processor_count <= 0:
            raise ValueError("hardware execution-unit counts must be positive")
        if self.compute_units_per_workgroup_processor <= 0:
            raise ValueError("compute units per work-group processor must be positive")
        if self.compute_unit_count != (self.workgroup_processor_count * self.compute_units_per_workgroup_processor):
            raise ValueError("compute-unit and work-group-processor topology is inconsistent")

    @property
    def problem_type_hash(self) -> str:
        return stable_hash(self.problem_type, prefix="ptype_")[:22]

    def benchmark_protocol_hash(self, protocol: BenchmarkProtocol | None = None) -> str:
        return (protocol or self.default_protocol).protocol_hash()

    def global_parameters(self, protocol: BenchmarkProtocol | None = None) -> dict[str, ProtocolParameterValue]:
        return {**BASE_GLOBAL_PARAMETERS, **(protocol or self.default_protocol).global_parameters()}

    def global_parameter_items(self, protocol: BenchmarkProtocol | None = None) -> list[str]:
        return global_parameter_items(self.global_parameters(protocol))

    def shapes(self) -> list[Shape]:
        return self.shapes_fn()


GFX1151_NT_HHS = TargetProfile(
    name="gfx1151-nt-hhs",
    problem_type=FP16_NT_HHS_PROBLEM_TYPE,
    library_logic=LIBRARY_LOGIC_GRIDBASED_GFX1151,
    default_protocol=DEFAULT_BENCHMARK_PROTOCOL,
    shapes_fn=pilot_100_shapes,
    environment_compatibility_tag="gfx1151-nt-hhs-v1",
    compute_unit_count=40,
    workgroup_processor_count=20,
    compute_units_per_workgroup_processor=2,
)

GFX1151_NT_HHS_COMFY1135 = TargetProfile(
    name="gfx1151-nt-hhs-comfy1135",
    problem_type=FP16_NT_HHS_PROBLEM_TYPE,
    library_logic=LIBRARY_LOGIC_GRIDBASED_GFX1151,
    default_protocol=DEFAULT_BENCHMARK_PROTOCOL,
    shapes_fn=comfy_nt_1135_shapes,
    environment_compatibility_tag="gfx1151-nt-hhs-v1",
    compute_unit_count=40,
    workgroup_processor_count=20,
    compute_units_per_workgroup_processor=2,
)

PROFILES = {
    GFX1151_NT_HHS.name: GFX1151_NT_HHS,
    GFX1151_NT_HHS_COMFY1135.name: GFX1151_NT_HHS_COMFY1135,
}
DEFAULT_PROFILE = GFX1151_NT_HHS


def get_profile(name: str | None) -> TargetProfile:
    if not name:
        return DEFAULT_PROFILE
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown profile: {name}") from exc
