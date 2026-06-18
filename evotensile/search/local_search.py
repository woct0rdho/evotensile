import random
from typing import Any

from evotensile.candidate import Candidate
from evotensile.search_space import DOMAINS, make_candidate


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
