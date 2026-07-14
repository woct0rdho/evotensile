import pytest

from evotensile.candidate import Candidate
from evotensile.search_space import DOMAINS, FIXED_PARAMS
from evotensile.tensilelite_parameter_types import (
    TENSILELITE_PARAMETER_LIST_ITEM_TYPES,
    TENSILELITE_PARAMETER_TYPES,
    normalize_imported_solution_parameters,
    validate_tensilelite_parameter_types,
)
from tests.helpers import REFERENCE_CANDIDATE


def test_parameter_type_schema_matches_search_space():
    expected_values = {name: values for name, values in DOMAINS.items()}
    for name, value in FIXED_PARAMS.items():
        expected_values.setdefault(name, [value])

    assert set(TENSILELITE_PARAMETER_TYPES) == set(expected_values)
    for name, values in expected_values.items():
        assert {type(value) for value in values} == {TENSILELITE_PARAMETER_TYPES[name]}
        if TENSILELITE_PARAMETER_TYPES[name] is list:
            assert {type(item) for value in values for item in value} == {TENSILELITE_PARAMETER_LIST_ITEM_TYPES[name]}


def test_import_normalization_applies_schema_to_every_known_parameter():
    expected = REFERENCE_CANDIDATE.canonical_params()
    imported = dict(expected)
    for name, expected_type in TENSILELITE_PARAMETER_TYPES.items():
        value = imported[name]
        if expected_type is int:
            imported[name] = float(value)
        elif expected_type is bool:
            imported[name] = int(value)
        elif expected_type is list:
            imported[name] = [float(item) for item in value]

    normalized = normalize_imported_solution_parameters(imported)

    assert normalized == expected
    validate_tensilelite_parameter_types(normalized)
    for name, expected_type in TENSILELITE_PARAMETER_TYPES.items():
        assert type(normalized[name]) is expected_type
        if expected_type is list:
            item_type = TENSILELITE_PARAMETER_LIST_ITEM_TYPES[name]
            assert all(type(item) is item_type for item in normalized[name])


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"DepthU": True}, "DepthU.*bool.*int"),
        ({"SourceSwap": 2}, "SourceSwap.*int.*bool"),
        ({"StaggerUStride": 32.5}, "StaggerUStride.*float.*int"),
        ({"MatrixInstruction": [16, 16.5]}, r"MatrixInstruction\[1\].*float.*int"),
    ],
)
def test_import_normalization_rejects_noncanonical_values(parameters, message):
    with pytest.raises(TypeError, match=message):
        normalize_imported_solution_parameters(parameters)


def test_candidate_requires_canonical_known_parameter_types():
    with pytest.raises(TypeError, match="DepthU must be int, not float"):
        Candidate(params={"DepthU": 32.0})
    with pytest.raises(TypeError, match="SourceSwap must be bool, not int"):
        Candidate(params={"SourceSwap": 1})

    candidate = Candidate(params={"external_metadata": 1.0})

    assert candidate.params == {"external_metadata": 1.0}
