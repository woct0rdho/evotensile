from pathlib import Path
from typing import Any

import yaml

from .candidate import Candidate, Shape


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


HOT_LOOP_BENCHMARK_PARAMETERS: dict[str, Any] = {
    "KernelTime": True,
    "PreciseKernelTime": True,
    "NumWarmups": 10,
    "NumBenchmarks": 10,
    "EnqueuesPerSync": 10,
    "SyncsPerBenchmark": 1,
    "SleepPercent": 0,
    "HardwareMonitor": False,
}


DEFAULT_GLOBAL_PARAMETERS: dict[str, Any] = {
    "MinimumRequiredVersion": "0.0.0",
    "RuntimeLanguage": "HIP",
    **HOT_LOOP_BENCHMARK_PARAMETERS,
    "DataInitTypeA": 3,
    "DataInitTypeB": 3,
    "DataInitTypeC": 3,
    "DataInitTypeD": 0,
    "DataInitTypeAlpha": 2,
    "DataInitTypeBeta": 2,
    "DataInitTypeBias": 3,
    "DataInitTypeScaleAlphaVec": 3,
    "NumElementsToValidate": 128,
    "ValidationMaxToPrint": 4,
    "ValidationPrintValids": False,
    "ForceRedoBenchmarkProblems": True,
    "ForceRedoLibraryLogic": True,
    "ForceRedoLibraryClient": True,
    "CSVExportWinner": True,
    "CSVMergeSameProblemID": False,
    "PrintWinnersOnly": False,
    "PredictionThreshold": 2.0,
    "GranularityThreshold": 0.0,
    "SkipSlowSolutionRatio": 0.0,
    "CEqualD": False,
    "LibraryFormat": "yaml",
    "LogicFormat": "yaml",
}


LIBRARY_LOGIC_GRIDBASED_GFX1151: dict[str, Any] = {
    "ScheduleName": "gfx1151",
    "DeviceNames": ["Device 150e", "Device 150f", "Device 1510", "Device 1511"],
    "ArchitectureName": {"Architecture": "gfx1151", "CUCount": 16},
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

    group_entries = [c.group_entry() for c in candidates]
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
        "LibraryClient": None,
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
