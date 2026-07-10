import random
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
from evotensile.search.learned_linkage import LinkageModel, fos_from_genomes, nearest_linkage_model
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


def neighborhood_group_names() -> tuple[tuple[str, ...], ...]:
    groups = [*NT_HHS_LINKAGE_GROUPS, *tuple((name,) for name in PARAM_NAMES)]
    seen_groups: list[tuple[str, ...]] = []
    for group in groups:
        if group not in seen_groups:
            seen_groups.append(group)
    return tuple(seen_groups)


def _random_group_trial(
    group: tuple[str, ...],
    current: dict[str, object],
    *,
    rng: random.Random,
) -> dict[str, object]:
    trial = dict(current)
    changed_name = rng.choice(group)
    for name in group:
        values = ordered_domain_values(name, current.get(name))
        alternatives = values[1:]
        if name == changed_name and alternatives:
            trial[name] = rng.choice(alternatives)
        elif alternatives and rng.random() < 0.5:
            trial[name] = rng.choice(alternatives)
    return trial


def gomea_neighborhood_candidates(
    parents: list[Candidate],
    *,
    count: int,
    max_elites: int | None = 4,
    exclude: set[str] | None = None,
    beam_width: int = 16,
    seed: int = 0,
) -> list[Candidate]:
    """Sample static and univariate neighborhoods fairly across ranked elites."""
    if count <= 0:
        return []
    out: list[Candidate] = []
    seen_hashes = set(exclude or ())
    rng = random.Random(seed)
    parent_pool = parents if max_elites is None else parents[:max_elites]
    slots = [(parent, group) for parent in parent_pool for group in neighborhood_group_names()]
    rng.shuffle(slots)
    max_attempts_per_slot = max(4, beam_width)
    for parent, group in slots:
        base = dict(parent.canonical_params())
        for _ in range(max_attempts_per_slot):
            trial = _random_group_trial(group, base, rng=rng)
            try:
                candidate = make_candidate(trial, source="gomea", parents=[parent.hash])
            except ValueError:
                continue
            if candidate.hash in seen_hashes:
                continue
            seen_hashes.add(candidate.hash)
            out.append(candidate)
            break
        if len(out) >= count:
            break
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
    linkage_models: Sequence[LinkageModel] | None = None,
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
    fallback_fos = [*rule_groups, *fos_from_genomes(elite_genomes)]
    out: list[Candidate] = []
    attempts = 0
    max_attempts = max(200, count * 100)
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        base_candidate, base_genome = rng.choice(ranked)
        donor_candidate, donor_genome = rng.choice(ranked)
        genes = base_genome
        model = nearest_linkage_model(base_genome, linkage_models or [])
        model_groups = list(model.fos_groups) if model is not None else []
        groups = [*rule_groups, *model_groups] if model_groups else list(fallback_fos)
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
            group = rng.choice(groups)
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
