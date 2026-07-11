import random

from evotensile.candidate import Candidate, Shape
from evotensile.search.encoding import (
    PARAM_NAMES,
    candidate_to_genome,
    dedupe_candidates,
    genome_to_candidate,
)
from evotensile.search_space import DOMAINS, eligible_for_shape_scope


def _mutate_gene(
    rng: random.Random,
    target: tuple[int, ...],
    a: tuple[int, ...],
    b: tuple[int, ...],
    c: tuple[int, ...],
    idx: int,
    *,
    crossover_rate: float,
    random_gene_rate: float,
) -> int:
    if rng.random() >= crossover_rate:
        return target[idx]
    if rng.random() < random_gene_rate:
        return rng.randrange(len(DOMAINS[PARAM_NAMES[idx]]))
    if b[idx] != c[idx]:
        return a[idx]
    return rng.choice((a[idx], b[idx], target[idx]))


def differential_evolution_candidates(
    parents: list[Candidate],
    *,
    count: int,
    seed: int = 1,
    crossover_rate: float = 0.8,
    random_gene_rate: float = 0.1,
    exclude: set[str] | None = None,
    target_shapes: list[Shape] | None = None,
) -> list[Candidate]:
    """Generate categorical-DE candidates from parent configs.

    The mutation is DE-inspired rather than numeric vector arithmetic: when two
    donor genomes disagree on a gene, that gene is treated as volatile and copied
    from a third donor or randomly re-sampled. This keeps candidates in the
    discrete TensileLite domains while still mixing multi-gene parent structure.
    """
    if count <= 0 or len(parents) < 4:
        return []
    rng = random.Random(seed)
    pool = list(parents)
    genomes = [candidate_to_genome(candidate) for candidate in pool]

    out: list[Candidate] = []
    attempts = 0
    max_attempts = max(200, count * 100)
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        target_idx = rng.randrange(len(genomes))
        target = genomes[target_idx]
        donor_indices = [idx for idx in range(len(genomes)) if idx != target_idx]
        a_idx, b_idx, c_idx = rng.sample(donor_indices, 3)
        a, b, c = genomes[a_idx], genomes[b_idx], genomes[c_idx]
        forced_idx = rng.randrange(len(PARAM_NAMES))
        child = tuple(
            _mutate_gene(
                rng,
                target,
                a,
                b,
                c,
                idx,
                crossover_rate=1.0 if idx == forced_idx else crossover_rate,
                random_gene_rate=random_gene_rate,
            )
            for idx in range(len(PARAM_NAMES))
        )
        parent_hashes = tuple(pool[idx].hash for idx in (target_idx, a_idx, b_idx, c_idx) if idx < len(pool))
        try:
            candidate = genome_to_candidate(child, source="de", parents=parent_hashes)
        except ValueError:
            continue
        if eligible_for_shape_scope(candidate.canonical_params(), target_shapes):
            out.append(candidate)
    return dedupe_candidates(out, exclude=exclude)[:count]
