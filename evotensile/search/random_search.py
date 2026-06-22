from evotensile.candidate import Candidate
from evotensile.search_space import random_candidates


def initial_random_batch(num_random: int, *, seed: int = 1) -> list[Candidate]:
    """Return deterministic random valid candidates."""
    return random_candidates(num_random, seed=seed)
