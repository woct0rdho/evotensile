import math
from collections.abc import Iterable

from .candidate import Candidate


def dedupe_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    by_hash: dict[str, Candidate] = {}
    for candidate in candidates:
        by_hash.setdefault(candidate.hash, candidate)
    return list(by_hash.values())


def round_up(value: int, step: int) -> int:
    if step <= 1:
        return value
    return int(math.ceil(value / step) * step)
