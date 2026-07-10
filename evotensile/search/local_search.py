import random
from collections.abc import Sequence
from typing import Any

from evotensile.candidate import Candidate, Shape
from evotensile.search.semantics import semantic_group_names
from evotensile.search_space import DOMAINS, cheap_constraints, make_candidate, repair_linked_overrides


def mutate_candidate(candidate: Candidate, *, seed: int | None = None, mutation_rate: float = 0.25) -> Candidate:
    """Simple categorical mutation around an existing candidate."""
    rng = random.Random(seed)
    params: dict[str, Any] = dict(candidate.canonical_params())
    for name, values in DOMAINS.items():
        if rng.random() < mutation_rate:
            params[name] = rng.choice(values)
    return make_candidate(params, source="mutation", parents=[candidate.hash])


def mutate_elites(
    elites: list[Candidate],
    *,
    count: int,
    seed: int = 1,
    mutation_rate: float = 0.25,
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
            child = mutate_candidate(parent, seed=rng.randrange(1 << 30), mutation_rate=mutation_rate)
        except ValueError:
            continue
        out[child.hash] = child
    return list(out.values())


def semantic_mutation_candidates(
    parents: Sequence[Candidate],
    *,
    count: int,
    seed: int = 1,
    target_shapes: Sequence[Shape] | None = None,
    max_changed_genes: int = 2,
    exclude: set[str] | None = None,
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
        params = dict(parent.canonical_params())
        group = rng.choice(groups)
        mutable_names = [name for name in group if len(DOMAINS[name]) > 1]
        if not mutable_names:
            continue
        changed_count = rng.randint(1, min(max(1, max_changed_genes), len(mutable_names)))
        for name in rng.sample(mutable_names, changed_count):
            alternatives = [value for value in DOMAINS[name] if value != params.get(name)]
            if alternatives:
                params[name] = rng.choice(alternatives)
        params = repair_linked_overrides(params)
        if target_shapes and not all(cheap_constraints(params, shape=shape) for shape in target_shapes):
            continue
        try:
            child = make_candidate(params, source="semantic-mutation", parents=[parent.hash])
        except ValueError:
            continue
        if child.hash in seen:
            continue
        seen.add(child.hash)
        out.append(child)
    return out
