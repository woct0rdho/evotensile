import random
from collections.abc import Iterable, Sequence
from typing import Any

from evotensile.candidate import Candidate, canonical_json
from evotensile.search_space import DOMAINS, make_candidate, repair_linked_overrides

PARAM_NAMES = tuple(DOMAINS.keys())
_VALUE_TO_INDEX: dict[str, dict[str, int]] = {
    name: {canonical_json(value): idx for idx, value in enumerate(values)} for name, values in DOMAINS.items()
}


def ordered_domain_values(name: str, current: Any | None = None) -> list[Any]:
    values = list(DOMAINS[name])
    if current is None:
        return values
    key = canonical_json(current)
    for idx, value in enumerate(values):
        if canonical_json(value) == key:
            return [value, *values[:idx], *values[idx + 1 :]]
    return values


def candidate_to_genome(candidate: Candidate) -> tuple[int, ...]:
    params = candidate.canonical_params()
    genome: list[int] = []
    for name in PARAM_NAMES:
        value = params.get(name, DOMAINS[name][0])
        try:
            genome.append(_VALUE_TO_INDEX[name][canonical_json(value)])
        except KeyError as exc:
            raise ValueError(f"candidate has value outside domain for {name}: {value!r}") from exc
    return tuple(genome)


def genome_to_overrides(genome: Sequence[int]) -> dict[str, Any]:
    if len(genome) != len(PARAM_NAMES):
        raise ValueError(f"genome has {len(genome)} genes, expected {len(PARAM_NAMES)}")
    return {name: DOMAINS[name][idx] for name, idx in zip(PARAM_NAMES, genome, strict=True)}


def genome_to_candidate(
    genome: Sequence[int], *, source: str, parents: Iterable[str] = (), repair: bool = True
) -> Candidate:
    overrides = genome_to_overrides(genome)
    if repair:
        overrides = repair_linked_overrides(overrides)
    return make_candidate(overrides, source=source, parents=parents)


def random_genome(rng: random.Random) -> tuple[int, ...]:
    return tuple(rng.randrange(len(DOMAINS[name])) for name in PARAM_NAMES)


def hamming_distance(left: Sequence[int], right: Sequence[int]) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def dedupe_candidates(candidates: Iterable[Candidate], *, exclude: set[str] | None = None) -> list[Candidate]:
    excluded = exclude or set()
    out: dict[str, Candidate] = {}
    for candidate in candidates:
        if candidate.hash not in excluded:
            out.setdefault(candidate.hash, candidate)
    return list(out.values())
