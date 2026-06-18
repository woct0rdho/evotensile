import ast
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .candidate import Candidate, Shape, stable_hash
from .yaml_writer import DEFAULT_GLOBAL_PARAMETERS, FP16_NT_HHS_PROBLEM_TYPE

DEFAULT_VERSION_NAME = "unversioned"

# Keep this explicit so changing unrelated YAML fields does not invalidate timing data.
BENCHMARK_PROTOCOL_KEYS = [
    "KernelTime",
    "PreciseKernelTime",
    "NumWarmups",
    "NumBenchmarks",
    "EnqueuesPerSync",
    "SyncsPerBenchmark",
    "MaxEnqueuesPerSync",
    "MinFlopsPerSync",
    "SleepPercent",
    "HardwareMonitor",
    "NumElementsToValidate",
    "NumElementsToValidateWinner",
    "DataInitTypeA",
    "DataInitTypeB",
    "DataInitTypeC",
    "DataInitTypeD",
    "DataInitTypeAlpha",
    "DataInitTypeBeta",
    "DataInitTypeBias",
    "DataInitTypeScaleAlphaVec",
    "DataInitSeed",
    "CEqualD",
    "PredictionThreshold",
    "GranularityThreshold",
    "SkipSlowSolutionRatio",
    "ParallelGpuExecution",
]


@dataclass(frozen=True)
class CacheKey:
    version_name: str
    problem_type_hash: str
    benchmark_protocol_hash: str
    shape_id: str
    candidate_hash: str


def normalize_version_name(version_name: str | None) -> str:
    if version_name is None or not version_name.strip():
        return DEFAULT_VERSION_NAME
    return version_name.strip()


def problem_type_hash(problem_type: dict[str, Any] | None = None) -> str:
    ptype = dict(FP16_NT_HHS_PROBLEM_TYPE if problem_type is None else problem_type)
    return stable_hash(ptype, prefix="ptype_")[:22]


def parse_global_parameter_items(items: Iterable[str] | None) -> dict[str, Any]:
    """Parse KEY=VALUE strings accepted by Tensile's --global-parameters."""
    parsed: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"global parameter must be KEY=VALUE: {item!r}")
        key, value = item.split("=", 1)
        try:
            parsed[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            if value == "True":
                parsed[key] = True
            elif value == "False":
                parsed[key] = False
            elif value == "None":
                parsed[key] = None
            else:
                parsed[key] = value
    return parsed


def benchmark_protocol_dict(global_parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(DEFAULT_GLOBAL_PARAMETERS)
    if global_parameters:
        params.update(global_parameters)
    return {key: params.get(key) for key in BENCHMARK_PROTOCOL_KEYS if key in params}


def benchmark_protocol_hash(global_parameters: dict[str, Any] | None = None) -> str:
    return stable_hash(benchmark_protocol_dict(global_parameters), prefix="bproto_")[:23]


def benchmark_protocol_hash_from_items(items: Iterable[str] | None = None) -> str:
    return benchmark_protocol_hash(parse_global_parameter_items(items))


def cache_keys(
    shapes: Iterable[Shape],
    candidates: Iterable[Candidate],
    *,
    version_name: str | None,
    problem_hash: str,
    protocol_hash: str,
) -> list[CacheKey]:
    version = normalize_version_name(version_name)
    return [
        CacheKey(
            version_name=version,
            problem_type_hash=problem_hash,
            benchmark_protocol_hash=protocol_hash,
            shape_id=shape.id,
            candidate_hash=candidate.hash,
        )
        for shape in shapes
        for candidate in candidates
    ]
