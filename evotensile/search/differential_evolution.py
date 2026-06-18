import random

from evotensile.candidate import Candidate
from evotensile.search.encoding import (
    PARAM_NAMES,
    candidate_to_genome,
    dedupe_candidates,
    genome_to_candidate,
    random_genome,
)
from evotensile.search_space import DOMAINS


def _random_valid_genome(rng: random.Random) -> tuple[int, ...]:
    for _ in range(1000):
        genome = random_genome(rng)
        try:
            genome_to_candidate(genome, source="de_probe")
        except ValueError:
            continue
        return genome
    raise RuntimeError("failed to sample a valid DE genome")


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
    random_parent_count: int = 32,
    exclude: set[str] | None = None,
) -> list[Candidate]:
    """Generate categorical-DE candidates from parent configs.

    The mutation is DE-inspired rather than numeric vector arithmetic: when two
    donor genomes disagree on a gene, that gene is treated as volatile and copied
    from a third donor or randomly re-sampled. This keeps candidates in the
    discrete TensileLite domains while still mixing multi-gene parent structure.
    """
    if count <= 0:
        return []
    rng = random.Random(seed)
    pool = list(parents)
    if len(pool) < 4:
        from evotensile.search.random_search import initial_random_batch

        pool.extend(initial_random_batch(random_parent_count, seed=seed + 17))
    genomes = [candidate_to_genome(candidate) for candidate in pool]
    while len(genomes) < 4:
        genomes.append(_random_valid_genome(rng))

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
            out.append(genome_to_candidate(child, source="de", parents=parent_hashes))
        except ValueError:
            continue
    return dedupe_candidates(out, exclude=exclude)[:count]
