from pathlib import Path
from typing import Any

import yaml

from .candidate import Candidate, Shape
from .protocol import DEFAULT_BENCHMARK_PROTOCOL


class FlowList(list):
    """Marker for YAML flow-style lists."""


def _flow_representer(dumper: yaml.Dumper, value: FlowList):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=True)


def _none_representer(dumper: yaml.Dumper, _value: None):
    return dumper.represent_scalar("tag:yaml.org,2002:null", "")


class EvoTensileDumper(yaml.SafeDumper):
    pass


EvoTensileDumper.add_representer(FlowList, _flow_representer)
EvoTensileDumper.add_representer(type(None), _none_representer)


def flow_lists(value: Any) -> Any:
    if isinstance(value, list):
        return FlowList(flow_lists(v) for v in value)
    if isinstance(value, dict):
        return {k: flow_lists(v) for k, v in value.items()}
    return value


FP16_NT_HHS_PROBLEM_TYPE: dict[str, Any] = {
    "OperationType": "GEMM",
    "DataType": "H",
    "DataTypeA": "H",
    "DataTypeB": "H",
    "DestDataType": "H",
    "ComputeDataType": "S",
    "HighPrecisionAccumulate": True,
    "TransposeA": False,
    "TransposeB": True,
    "UseBeta": True,
    "Batched": True,
    "BiasSrc": "D",
    "UseBias": 1,
    "BiasDataTypeList": ["h"],
    "UseE": False,
    "UseScaleAlphaVec": 1,
    "UseScaleAB": "",
    "UseScaleCD": False,
    "Activation": True,
    "ActivationType": "hipblaslt_all",
    "Gradient": False,
    "GroupedGemm": False,
    "Sparse": 0,
    "SupportUserArgs": True,
}


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


DEFAULT_GLOBAL_PARAMETERS: dict[str, Any] = {
    **BASE_GLOBAL_PARAMETERS,
    **DEFAULT_BENCHMARK_PROTOCOL.global_parameters(),
}


LIBRARY_LOGIC_GRIDBASED_GFX1151: dict[str, Any] = {
    "ScheduleName": "gfx1151",
    "DeviceNames": ["Device 150e", "Device 150f", "Device 1510", "Device 1511"],
    # Keep CUCount out of the generated logic predicate. gfx1151 has 40 physical
    # CUs, but HIP in RDNA WGP mode reports 20 multi-processors (one per 2-CU WGP).
    # A predicate tied to either ambiguous count can produce WRONG_HARDWARE/nan.
    "ArchitectureName": "gfx1151",
    "LibraryType": "GridBased",
}


def benchmark_problem(
    candidates: list[Candidate],
    shapes: list[Shape],
    *,
    problem_type: dict[str, Any] | None = None,
) -> list[Any]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    if not shapes:
        raise ValueError("at least one shape is required")

    ptype = dict(FP16_NT_HHS_PROBLEM_TYPE if problem_type is None else problem_type)

    group_entries = [c.canonical_params() for c in candidates]
    problem_sizes = [{"Exact": FlowList(shape.exact_list())} for shape in shapes]

    size_group = {
        "BenchmarkCommonParameters": None,
        "ForkParameters": [
            {"Groups": [group_entries]},
        ],
        "BenchmarkFinalParameters": [
            {"ProblemSizes": problem_sizes},
            {"BiasTypeArgs": FlowList(["h"])},
            {"FactorDimArgs": FlowList([0])},
            {"ActivationArgs": [FlowList([{"Enum": "none"}])]},
        ],
    }
    return [flow_lists(ptype), size_group]


def tensilelite_config(
    candidates: list[Candidate],
    shapes: list[Shape],
    *,
    global_parameters: dict[str, Any] | None = None,
    library_logic: dict[str, Any] | None = None,
    problem_type: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gp = dict(DEFAULT_GLOBAL_PARAMETERS)
    if global_parameters:
        gp.update(global_parameters)

    ll = dict(LIBRARY_LOGIC_GRIDBASED_GFX1151)
    if library_logic:
        ll.update(library_logic)

    return {
        "GlobalParameters": flow_lists(gp),
        "BenchmarkProblems": [benchmark_problem(candidates, shapes, problem_type=problem_type)],
        "LibraryLogic": flow_lists(ll),
    }


def write_tensilelite_yaml(
    path: str | Path,
    candidates: list[Candidate],
    shapes: list[Shape],
    *,
    global_parameters: dict[str, Any] | None = None,
    library_logic: dict[str, Any] | None = None,
    problem_type: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = tensilelite_config(
        candidates,
        shapes,
        global_parameters=global_parameters,
        library_logic=library_logic,
        problem_type=problem_type,
    )
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, Dumper=EvoTensileDumper, sort_keys=False, width=160)
    return path
