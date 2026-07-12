import pytest

from evotensile.candidate import Shape
from evotensile.search.trust_region import interaction_grid_candidates
from evotensile.search_space import make_candidate, repair_linked_overrides


def _parent():
    return make_candidate(repair_linked_overrides({}), source="parent")


def test_interaction_grid_enumerates_cross_parameter_variants():
    parent = _parent()

    candidates = interaction_grid_candidates(
        parent,
        parameter_values={
            "ScheduleIterAlg": (2, 3),
            "StorePriorityOpt": (True, False),
        },
    )

    assert len(candidates) == 3
    assert {candidate.parent_hashes for candidate in candidates} == {(parent.hash,)}
    assert {
        (candidate.params["ScheduleIterAlg"], candidate.params["StorePriorityOpt"]) for candidate in candidates
    } == {(2, False), (3, True), (3, False)}
    assert all(
        candidate.proposal_metadata["interaction_parameters"] == ["ScheduleIterAlg", "StorePriorityOpt"]
        for candidate in candidates
    )


def test_interaction_grid_applies_changed_gene_limit_and_exclusion():
    parent = _parent()
    one_gene = interaction_grid_candidates(
        parent,
        parameter_values={
            "ScheduleIterAlg": (2, 3),
            "StorePriorityOpt": (True, False),
        },
        max_changed_genes=1,
    )
    assert len(one_gene) == 2

    remaining = interaction_grid_candidates(
        parent,
        parameter_values={
            "ScheduleIterAlg": (2, 3),
            "StorePriorityOpt": (True, False),
        },
        max_changed_genes=1,
        exclude={one_gene[0].hash},
    )
    assert len(remaining) == 1


def test_interaction_grid_keeps_only_shape_eligible_variants():
    parent = _parent()
    shape = Shape(m=510, n=128, batch=1, k=256)

    candidates = interaction_grid_candidates(
        parent,
        parameter_values={"AssertFree0ElementMultiple": (8, 1)},
        target_shapes=(shape,),
    )

    assert len(candidates) == 1
    assert candidates[0].params["AssertFree0ElementMultiple"] == 1


@pytest.mark.parametrize(
    ("parameter_values", "message"),
    [
        ({}, "at least one parameter"),
        ({"Unknown": (1,)}, "not searchable"),
        ({"ScheduleIterAlg": ()}, "has no values"),
        ({"ScheduleIterAlg": (99,)}, "outside the domain"),
    ],
)
def test_interaction_grid_rejects_invalid_definitions(parameter_values, message):
    parent = _parent()
    with pytest.raises(ValueError, match=message):
        interaction_grid_candidates(parent, parameter_values=parameter_values)
