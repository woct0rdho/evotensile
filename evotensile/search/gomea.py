import itertools
import math
import random
from collections import Counter
from collections.abc import Sequence

from evotensile.candidate import Candidate
from evotensile.search.encoding import (
    PARAM_NAMES,
    candidate_to_genome,
    dedupe_candidates,
    genome_to_candidate,
    hamming_distance,
    ordered_domain_values,
)
from evotensile.search_space import (
    NT_HHS_RANDOM_VALU_VGPR_HEADROOM,
    _valu_vgpr_lower_bound,
    cheap_constraints,
    explain_invalid_nt_hhs,
    make_candidate,
    repair_linked_overrides,
)
from evotensile.shapes import Shape


def _ranked_genomes(parents: list[Candidate]) -> list[tuple[Candidate, tuple[int, ...]]]:
    return [(candidate, candidate_to_genome(candidate)) for candidate in parents]


def _pairwise_mutual_information(genomes: Sequence[tuple[int, ...]]) -> list[list[float]]:
    n_genes = len(PARAM_NAMES)
    if len(genomes) < 2:
        return [[0.0] * n_genes for _ in range(n_genes)]
    n = float(len(genomes))
    matrix = [[0.0] * n_genes for _ in range(n_genes)]
    for i in range(n_genes):
        counts_i = Counter(genome[i] for genome in genomes)
        for j in range(i + 1, n_genes):
            counts_j = Counter(genome[j] for genome in genomes)
            counts_ij = Counter((genome[i], genome[j]) for genome in genomes)
            mi = 0.0
            for (left, right), count in counts_ij.items():
                p_ij = count / n
                p_i = counts_i[left] / n
                p_j = counts_j[right] / n
                mi += p_ij * math.log(p_ij / (p_i * p_j))
            matrix[i][j] = mi
            matrix[j][i] = mi
    return matrix


def linkage_tree(genomes: Sequence[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Build a small UPGMA linkage tree from top parent genomes."""
    n_genes = len(PARAM_NAMES)
    fos: list[tuple[int, ...]] = [(idx,) for idx in range(n_genes)]
    if len(genomes) < 2:
        return fos
    mi = _pairwise_mutual_information(genomes)
    active: dict[int, tuple[int, ...]] = {idx: (idx,) for idx in range(n_genes)}
    next_id = n_genes
    while len(active) > 1:
        ids = list(active)
        best_pair: tuple[int, int] | None = None
        best_score = 0.0
        for left_idx, left_id in enumerate(ids):
            for right_id in ids[left_idx + 1 :]:
                left = active[left_id]
                right = active[right_id]
                score = sum(mi[i][j] for i in left for j in right) / (len(left) * len(right))
                if score > best_score:
                    best_score = score
                    best_pair = (left_id, right_id)
        if best_pair is None or best_score <= 1e-12:
            break
        left_id, right_id = best_pair
        merged = tuple(sorted((*active[left_id], *active[right_id])))
        del active[left_id]
        del active[right_id]
        active[next_id] = merged
        next_id += 1
        if len(merged) < n_genes:
            fos.append(merged)
    return fos


def _nearest_elite(genome: tuple[int, ...], elites: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
    return min(elites, key=lambda elite: hamming_distance(genome, elite))


NT_HHS_LINKAGE_GROUPS = (
    ("MatrixInstruction", "WorkGroup", "DepthU", "GlobalSplitU"),
    ("TransposeLDS", "LdsBlockSizePerPadA", "LdsBlockSizePerPadB", "LdsPadA", "LdsPadB"),
    ("PrefetchGlobalRead", "PrefetchLocalRead", "1LDSBuffer", "ClusterLocalRead", "VectorWidthB"),
    ("GlobalReadVectorWidthA", "GlobalReadVectorWidthB", "VectorWidthA", "VectorWidthB"),
    ("ScheduleIterAlg", "WorkGroupMapping", "StaggerU", "StaggerUStride", "StaggerUMapping", "SourceSwap"),
    ("StorePriorityOpt", "NumElementsPerBatchStore", "StoreSyncOpt", "GroupLoadStore", "StoreVectorWidth"),
    ("ExpandPointerSwap",),
    ("AssertFree0ElementMultiple", "AssertFree1ElementMultiple", "AssertSummationElementMultiple"),
)
_PRIORITY_GROUPS = (
    ("ScheduleIterAlg", "StorePriorityOpt"),
    ("ScheduleIterAlg", "StorePriorityOpt", "NumElementsPerBatchStore"),
    ("StorePriorityOpt", "NumElementsPerBatchStore", "StoreSyncOpt", "GroupLoadStore"),
    ("PrefetchGlobalRead", "PrefetchLocalRead", "1LDSBuffer", "TransposeLDS"),
    ("SourceSwap", "ClusterLocalRead", "TransposeLDS"),
    ("MatrixInstruction", "DepthU"),
    ("GlobalSplitU", "DepthU"),
    *NT_HHS_LINKAGE_GROUPS,
)


def neighborhood_group_names() -> tuple[tuple[str, ...], ...]:
    groups = [*_PRIORITY_GROUPS, *tuple((name,) for group in _PRIORITY_GROUPS for name in group)]
    seen_groups: list[tuple[str, ...]] = []
    for group in groups:
        if group not in seen_groups:
            seen_groups.append(group)
    return tuple(seen_groups)


def _group_value_products(names: tuple[str, ...], current: dict[str, object]):
    value_lists = [ordered_domain_values(name, current.get(name)) for name in names]
    return itertools.product(*value_lists)


def gomea_neighborhood_candidates(
    parents: list[Candidate],
    *,
    count: int,
    max_elites: int | None = 4,
    exclude: set[str] | None = None,
    beam_width: int = 16,
) -> list[Candidate]:
    """Sweep and compose compact FOS groups around ranked elites."""
    if count <= 0:
        return []
    out: list[Candidate] = []
    seen_hashes = set(exclude or ())
    seen_groups = neighborhood_group_names()
    parent_pool = parents if max_elites is None else parents[:max_elites]
    for parent in parent_pool:
        frontier = [dict(parent.canonical_params())]
        for group in seen_groups:
            next_frontier: list[dict[str, object]] = []
            for base in frontier:
                for values in _group_value_products(group, base):
                    if all(base.get(name) == value for name, value in zip(group, values, strict=True)):
                        continue
                    trial = dict(base)
                    trial.update(dict(zip(group, values, strict=True)))
                    try:
                        candidate = make_candidate(trial, source="gomea", parents=[parent.hash])
                    except ValueError:
                        continue
                    if candidate.hash in seen_hashes:
                        continue
                    seen_hashes.add(candidate.hash)
                    out.append(candidate)
                    next_frontier.append(candidate.canonical_params())
                    if len(out) >= count:
                        return out
            frontier = [*frontier[:1], *next_frontier[:beam_width]]
    return out[:count]


def _genome_with_group(base: tuple[int, ...], donor: tuple[int, ...], group: tuple[int, ...]) -> tuple[int, ...]:
    genes = list(base)
    for idx in group:
        genes[idx] = donor[idx]
    return tuple(genes)


def _gomea_candidate_ok(params: dict[str, object], *, target_shapes: Sequence[Shape] | None = None) -> bool:
    if explain_invalid_nt_hhs(params):
        return False
    if target_shapes and not all(cheap_constraints(params, shape=shape) for shape in target_shapes):
        return False
    return _valu_vgpr_lower_bound(params) <= NT_HHS_RANDOM_VALU_VGPR_HEADROOM


def _is_rule_valid_genome(genome: tuple[int, ...], *, target_shapes: Sequence[Shape] | None = None) -> bool:
    candidate = genome_to_candidate(genome, source="gomea", repair=True)
    return _gomea_candidate_ok(candidate.canonical_params(), target_shapes=target_shapes)


def _rule_valid_candidate(
    genome: tuple[int, ...], *, source: str, parents: tuple[str, ...], target_shapes: Sequence[Shape] | None = None
) -> Candidate:
    params = genome_to_candidate(genome, source=source, parents=parents, repair=True).canonical_params()
    params = repair_linked_overrides(params)
    if not _gomea_candidate_ok(params, target_shapes=target_shapes):
        raise ValueError("candidate failed NT HHS GOMEA proposal checks")
    return make_candidate(params, source=source, parents=parents)


def gomea_candidates(
    parents: list[Candidate],
    *,
    count: int,
    seed: int = 1,
    elite_count: int = 8,
    include_forced_improvement: bool = True,
    exclude: set[str] | None = None,
    target_shapes: Sequence[Shape] | None = None,
) -> list[Candidate]:
    """Generate linkage-aware categorical candidates from ranked parents.

    `parents` should be ordered best-first. The operator mixes Family-of-Subsets
    groups learned from the top parents, preserving linked knob bundles instead
    of mutating each field independently.
    """
    if count <= 0 or not parents:
        return []
    rng = random.Random(seed)
    ranked = _ranked_genomes(parents)
    elites = ranked[: max(1, min(elite_count, len(ranked)))]
    elite_genomes = [genome for _, genome in elites]
    rule_groups = [tuple(PARAM_NAMES.index(name) for name in group) for group in NT_HHS_LINKAGE_GROUPS]
    fos = [*rule_groups, *linkage_tree(elite_genomes)]
    out: list[Candidate] = []
    attempts = 0
    max_attempts = max(200, count * 100)
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        base_candidate, base_genome = rng.choice(ranked)
        donor_candidate, donor_genome = rng.choice(ranked)
        genes = base_genome
        groups = list(fos)
        rng.shuffle(groups)
        changed = False
        for group in groups[: rng.randint(1, max(1, min(4, len(groups))))]:
            trial = _genome_with_group(genes, donor_genome, group)
            try:
                if _is_rule_valid_genome(trial, target_shapes=target_shapes):
                    genes = trial
                    changed = True
            except ValueError:
                continue
        if include_forced_improvement and (not changed or genes == base_genome):
            elite = _nearest_elite(base_genome, elite_genomes)
            group = rng.choice(fos)
            trial = _genome_with_group(genes, elite, group)
            try:
                if _is_rule_valid_genome(trial, target_shapes=target_shapes):
                    genes = trial
            except ValueError:
                pass
        try:
            out.append(
                _rule_valid_candidate(
                    genes,
                    source="gomea",
                    parents=(base_candidate.hash, donor_candidate.hash),
                    target_shapes=target_shapes,
                )
            )
        except ValueError:
            continue
    return dedupe_candidates(out, exclude=exclude)[:count]
