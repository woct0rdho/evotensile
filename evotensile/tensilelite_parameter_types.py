from collections.abc import Mapping
from typing import Any

_INTEGER_PARAMETERS = frozenset(
    {
        "1LDSBuffer",
        "AssertFree0ElementMultiple",
        "AssertFree1ElementMultiple",
        "AssertSummationElementMultiple",
        "ClusterLocalRead",
        "DepthU",
        "GlobalReadVectorWidthA",
        "GlobalReadVectorWidthB",
        "GlobalSplitU",
        "LdsBlockSizePerPadA",
        "LdsBlockSizePerPadB",
        "LdsPadA",
        "LdsPadB",
        "LocalReadVectorWidth",
        "NumElementsPerBatchStore",
        "PrefetchGlobalRead",
        "PrefetchLocalRead",
        "ScheduleGlobalRead",
        "ScheduleIterAlg",
        "ScheduleLocalWrite",
        "StaggerU",
        "StaggerUMapping",
        "StaggerUStride",
        "StoreRemapVectorWidth",
        "StoreSyncOpt",
        "StoreVectorWidth",
        "TransposeLDS",
        "VectorWidthA",
        "VectorWidthB",
        "WavefrontSize",
        "WorkGroupMapping",
    }
)
_BOOLEAN_PARAMETERS = frozenset(
    {
        "ExpandPointerSwap",
        "GroupLoadStore",
        "MIArchVgpr",
        "SourceSwap",
        "StorePriorityOpt",
    }
)
_STRING_PARAMETERS = frozenset({"GlobalSplitUAlgorithm", "KernelLanguage"})
_INTEGER_LIST_PARAMETERS = frozenset({"MatrixInstruction", "WorkGroup"})

TENSILELITE_PARAMETER_TYPES: dict[str, type[object]] = {
    **{name: int for name in _INTEGER_PARAMETERS},
    **{name: bool for name in _BOOLEAN_PARAMETERS},
    **{name: str for name in _STRING_PARAMETERS},
    **{name: list for name in _INTEGER_LIST_PARAMETERS},
}
TENSILELITE_PARAMETER_LIST_ITEM_TYPES: dict[str, type[object]] = {name: int for name in _INTEGER_LIST_PARAMETERS}


def validate_tensilelite_parameter_types(parameters: Mapping[str, object]) -> None:
    for name, value in parameters.items():
        expected_type = TENSILELITE_PARAMETER_TYPES.get(name)
        if expected_type is None:
            continue
        if expected_type is list:
            if not isinstance(value, list) or type(value) is not list:
                raise TypeError(
                    f"TensileLite parameter {name} must be {expected_type.__name__}, not {type(value).__name__}"
                )
            item_type = TENSILELITE_PARAMETER_LIST_ITEM_TYPES[name]
            for index, item in enumerate(value):
                if type(item) is not item_type:
                    raise TypeError(
                        f"TensileLite parameter {name}[{index}] must be {item_type.__name__}, not {type(item).__name__}"
                    )
            continue
        if type(value) is not expected_type:
            raise TypeError(
                f"TensileLite parameter {name} must be {expected_type.__name__}, not {type(value).__name__}"
            )


def normalize_imported_solution_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(parameters)
    for name, value in parameters.items():
        expected_type = TENSILELITE_PARAMETER_TYPES.get(name)
        if expected_type is None:
            continue
        if expected_type is int:
            if type(value) is int:
                continue
            if type(value) is float and value.is_integer():
                normalized[name] = int(value)
                continue
        elif expected_type is bool:
            if type(value) is bool:
                continue
            if type(value) is int and value in (0, 1):
                normalized[name] = bool(value)
                continue
        elif expected_type is str:
            if type(value) is str:
                continue
        elif expected_type is list and type(value) is list:
            item_type = TENSILELITE_PARAMETER_LIST_ITEM_TYPES[name]
            normalized_items = []
            for index, item in enumerate(value):
                if item_type is int and type(item) is int:
                    normalized_items.append(item)
                elif item_type is int and type(item) is float and item.is_integer():
                    normalized_items.append(int(item))
                else:
                    raise TypeError(
                        f"imported TensileLite parameter {name}[{index}] cannot be normalized "
                        f"from {type(item).__name__} to {item_type.__name__}"
                    )
            normalized[name] = normalized_items
            continue
        raise TypeError(
            f"imported TensileLite parameter {name} cannot be normalized "
            f"from {type(value).__name__} to {expected_type.__name__}"
        )
    validate_tensilelite_parameter_types(normalized)
    return normalized
