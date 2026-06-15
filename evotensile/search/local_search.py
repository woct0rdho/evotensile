import random
from typing import Any

from evotensile.candidate import Candidate
from evotensile.search_space import DOMAINS, make_candidate


def mutate_candidate(candidate: Candidate, *, seed: int | None = None, mutation_rate: float = 0.25) -> Candidate:
    """Simple categorical mutation around an existing candidate.

    This is intentionally small and will be replaced/extended by shape-aware local search.
    """
    rng = random.Random(seed)
    params: dict[str, Any] = dict(candidate.canonical_params())
    for name, values in DOMAINS.items():
        if rng.random() < mutation_rate:
            params[name] = rng.choice(values)
    return make_candidate(params, source="mutation", parents=[candidate.hash])


def mutate_elites(elites: list[Candidate], *, count: int, seed: int = 1) -> list[Candidate]:
    rng = random.Random(seed)
    out: dict[str, Candidate] = {}
    while len(out) < count and elites:
        parent = rng.choice(elites)
        try:
            child = mutate_candidate(parent, seed=rng.randrange(1 << 30))
        except ValueError:
            continue
        out[child.hash] = child
    return list(out.values())
