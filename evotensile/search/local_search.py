import random
from collections.abc import Mapping, Sequence
from typing import Any

from evotensile.candidate import Candidate, Shape
from evotensile.search.semantics import semantic_group_key, semantic_group_names
from evotensile.search_space import DOMAINS, eligible_for_shape_scope, make_candidate, repair_linked_overrides


def mutate_candidate(
    candidate: Candidate,
    *,
    seed: int | None = None,
    mutation_rate: float = 0.25,
    target_shapes: Sequence[Shape] | None = None,
) -> Candidate:
    """Simple categorical mutation around an existing candidate."""
    rng = random.Random(seed)
    params: dict[str, Any] = dict(candidate.canonical_params())
    for name, values in DOMAINS.items():
        if rng.random() < mutation_rate:
            params[name] = rng.choice(values)
    params = repair_linked_overrides(params)
    if not eligible_for_shape_scope(params, target_shapes):
        raise ValueError("mutated candidate has no eligible shape in proposal scope")
    return make_candidate(params, source="mutation", parents=[candidate.hash])


def mutate_elites(
    elites: list[Candidate],
    *,
    count: int,
    seed: int = 12345,
    mutation_rate: float = 0.25,
    target_shapes: Sequence[Shape] | None = None,
    max_attempts: int | None = None,
) -> list[Candidate]:
    rng = random.Random(seed)
    out: dict[str, Candidate] = {}
    attempts = 0
    limit = max_attempts if max_attempts is not None else max(100, count * 50)
    while len(out) < count and elites and attempts < limit:
        attempts += 1
        parent = rng.choice(elites)
        try:
            child = mutate_candidate(
                parent,
                seed=rng.randrange(1 << 30),
                mutation_rate=mutation_rate,
                target_shapes=target_shapes,
            )
        except ValueError:
            continue
        out[child.hash] = child
    return list(out.values())


def _weighted_group_choice(
    rng: random.Random,
    groups: tuple[tuple[str, ...], ...],
    weights: Mapping[str, float] | None,
) -> tuple[str, ...]:
    if not weights:
        return rng.choice(groups)
    resolved = [max(0.0, float(weights.get(semantic_group_key(group), 1.0))) for group in groups]
    total = sum(resolved)
    if total <= 0.0:
        return rng.choice(groups)
    threshold = rng.random() * total
    cumulative = 0.0
    for group, weight in zip(groups, resolved, strict=True):
        cumulative += weight
        if cumulative >= threshold:
            return group
    return groups[-1]


def semantic_mutation_candidates(
    parents: Sequence[Candidate],
    *,
    count: int,
    seed: int = 12345,
    target_shapes: Sequence[Shape] | None = None,
    max_changed_genes: int = 2,
    exclude: set[str] | None = None,
    group_weights: Mapping[str, float] | None = None,
) -> list[Candidate]:
    """Mutate one semantic group while keeping the step deliberately small."""
    if count <= 0 or not parents:
        return []
    rng = random.Random(seed)
    groups = semantic_group_names()
    seen = set(exclude or ()) | {parent.hash for parent in parents}
    out: list[Candidate] = []
    attempts = 0
    max_attempts = max(400, count * 300)
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        parent = rng.choice(list(parents))
        parent_params = parent.canonical_params()
        params = dict(parent_params)
        group = _weighted_group_choice(rng, groups, group_weights)
        mutable_names = [name for name in group if len(DOMAINS[name]) > 1]
        if not mutable_names:
            continue
        changed_count = rng.randint(1, min(max(1, max_changed_genes), len(mutable_names)))
        requested_transitions: dict[str, dict[str, Any]] = {}
        for name in rng.sample(mutable_names, changed_count):
            alternatives = [value for value in DOMAINS[name] if value != params.get(name)]
            if alternatives:
                selected = rng.choice(alternatives)
                requested_transitions[name] = {"from": params.get(name), "to": selected}
                params[name] = selected
        params = repair_linked_overrides(params)
        if not eligible_for_shape_scope(params, target_shapes):
            continue
        try:
            child = make_candidate(
                params,
                source="semantic-mutation",
                parents=[parent.hash],
                proposal_metadata={
                    "semantic_group": semantic_group_key(group),
                    "requested_transitions": requested_transitions,
                    "changed_genes": sorted(name for name in DOMAINS if params[name] != parent_params[name]),
                },
            )
        except ValueError:
            continue
        if child.hash in seen:
            continue
        seen.add(child.hash)
        out.append(child)
    return out
