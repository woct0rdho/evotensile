from evotensile.candidate import Candidate
from evotensile.search_space import seed_and_random_candidates


def initial_random_batch(num_random: int, *, seed: int = 1) -> list[Candidate]:
    """Return deterministic seeds plus random valid candidates."""
    return seed_and_random_candidates(num_random, seed=seed)
