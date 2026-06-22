from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .candidate import Shape, stable_hash
from .protocol import DEFAULT_BENCHMARK_PROTOCOL, BenchmarkProtocol, global_parameter_items
from .shapes import pilot_100_shapes
from .yaml_writer import FP16_NT_HHS_PROBLEM_TYPE, LIBRARY_LOGIC_GRIDBASED_GFX1151

BASE_GLOBAL_PARAMETERS: dict[str, Any] = {
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
    default_proposal: str = "seed-random-gomea"
    default_num_random: int = 64
    default_elite_count: int = 8
    default_local_count: int = 32
    default_de_count: int = 32
    default_gomea_count: int = 64
    default_transfer_shapes: int = 4
    default_transfer_per_shape: int = 2
    default_mutation_rate: float = 0.25
    default_crossover_rate: float = 0.8
    default_random_gene_rate: float = 0.1
    default_candidate_batch_size: int = 32
    default_shape_batch_size: int = 100
    default_runner_bin: str = "./build/evotensile-structured-runner"
    default_build_timeout_s: float | None = 1800.0
    default_runner_timeout_s: float | None = 600.0
    structured_runner_build_command: tuple[str, ...] = ("scripts/build_structured_runner.sh",)

    @property
    def problem_type_hash(self) -> str:
        return stable_hash(self.problem_type, prefix="ptype_")[:22]

    def benchmark_protocol_hash(self, protocol: BenchmarkProtocol | None = None) -> str:
        return (protocol or self.default_protocol).protocol_hash()

    def global_parameters(self, protocol: BenchmarkProtocol | None = None) -> dict[str, Any]:
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
)

PROFILES = {GFX1151_NT_HHS.name: GFX1151_NT_HHS}
DEFAULT_PROFILE = GFX1151_NT_HHS


def get_profile(name: str | None) -> TargetProfile:
    if not name:
        return DEFAULT_PROFILE
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown profile: {name}") from exc
