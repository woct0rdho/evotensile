import itertools
from collections.abc import Mapping, Sequence

from evotensile.candidate import Candidate, Shape, canonical_json
from evotensile.search_space import DOMAINS, eligible_for_shape_scope, make_candidate, repair_linked_overrides


def interaction_grid_candidates(
    parent: Candidate,
    *,
    parameter_values: Mapping[str, Sequence[object]],
    target_shapes: Sequence[Shape] | None = None,
    max_changed_genes: int | None = None,
    repair_linked: bool = False,
    exclude: set[str] | None = None,
    source: str = "interaction-grid",
) -> tuple[Candidate, ...]:
    """Enumerate a bounded parameter interaction grid around one complete parent."""
    if not parameter_values:
        raise ValueError("interaction grid requires at least one parameter")
    if max_changed_genes is not None and max_changed_genes <= 0:
        raise ValueError("interaction-grid changed-gene limit must be positive")
    names = tuple(parameter_values)
    value_lists = []
    for name in names:
        if name not in DOMAINS:
            raise ValueError(f"interaction-grid parameter is not searchable: {name}")
        values = tuple(parameter_values[name])
        if not values:
            raise ValueError(f"interaction-grid parameter has no values: {name}")
        domain_keys = {canonical_json(value) for value in DOMAINS[name]}
        invalid_values = [value for value in values if canonical_json(value) not in domain_keys]
        if invalid_values:
            raise ValueError(f"interaction-grid values are outside the domain for {name}: {invalid_values!r}")
        value_lists.append(values)

    parent_params = parent.canonical_params()
    excluded = set(exclude or ()) | {parent.hash}
    candidates: dict[str, Candidate] = {}
    for combination in itertools.product(*value_lists):
        params = dict(parent_params)
        requested_transitions = {}
        for name, value in zip(names, combination, strict=True):
            if canonical_json(params[name]) != canonical_json(value):
                requested_transitions[name] = {"from": params[name], "to": value}
                params[name] = value
        if not requested_transitions:
            continue
        if repair_linked:
            params = repair_linked_overrides(params)
        changed_genes = tuple(
            name for name in DOMAINS if canonical_json(params[name]) != canonical_json(parent_params[name])
        )
        if max_changed_genes is not None and len(changed_genes) > max_changed_genes:
            continue
        if not eligible_for_shape_scope(params, target_shapes):
            continue
        try:
            candidate = make_candidate(
                params,
                source=source,
                parents=(parent.hash,),
                proposal_metadata={
                    "interaction_parameters": list(names),
                    "requested_transitions": requested_transitions,
                    "changed_genes": list(changed_genes),
                },
            )
        except ValueError:
            continue
        if candidate.hash in excluded:
            continue
        candidates.setdefault(candidate.hash, candidate)
    return tuple(candidates.values())
