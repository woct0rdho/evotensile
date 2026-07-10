from evotensile.candidate import Candidate
from evotensile.search_space import make_candidate, random_candidates, repair_linked_overrides


def sample_candidates(count: int, *, seed: int = 1151) -> list[Candidate]:
    return random_candidates(count, seed=seed)


def sample_candidate(*, seed: int = 1151) -> Candidate:
    return sample_candidates(1, seed=seed)[0]


REFERENCE_CANDIDATE = make_candidate(repair_linked_overrides({}), source="reference")
