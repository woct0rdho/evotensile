import itertools
import math
import random
from collections.abc import Mapping, Sequence

from evotensile.candidate import Candidate
from evotensile.search.encoding import (
    PARAM_NAMES,
    candidate_to_genome,
    dedupe_candidates,
    genome_to_candidate,
    hamming_distance,
    ordered_domain_values,
)
from evotensile.search.family import family_descriptor
from evotensile.search.learned_linkage import LinkageModel, fos_from_genomes, nearest_linkage_model
from evotensile.search.operator_credit import DONOR_MODES
from evotensile.search.semantics import NT_HHS_SEMANTIC_GROUPS, semantic_group_key, semantic_group_names
from evotensile.search_space import (
    DOMAINS,
    NT_HHS_RANDOM_VALU_VGPR_HEADROOM,
    _valu_vgpr_lower_bound,
    eligible_for_shape_scope,
    explain_invalid_nt_hhs,
    make_candidate,
    repair_linked_overrides,
)
from evotensile.shapes import Shape


def _ranked_genomes(parents: list[Candidate]) -> list[tuple[Candidate, tuple[int, ...]]]:
    return [(candidate, candidate_to_genome(candidate)) for candidate in parents]


def _nearest_elite(genome: tuple[int, ...], elites: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
    return min(elites, key=lambda elite: hamming_distance(genome, elite))


NT_HHS_LINKAGE_GROUPS = NT_HHS_SEMANTIC_GROUPS


def neighborhood_group_names() -> tuple[tuple[str, ...], ...]:
    return semantic_group_names()


def _bounded_group_trials(
    group: tuple[str, ...],
    current: Mapping[str, object],
    *,
    rng: random.Random,
    max_variants: int,
) -> list[dict[str, object]]:
    if max_variants <= 0:
        return []
    mutable = [name for name in group if len(DOMAINS[name]) > 1]
    if not mutable:
        return []
    values_by_name = {name: ordered_domain_values(name, current.get(name)) for name in mutable}
    total_product = math.prod(len(values_by_name[name]) for name in mutable)
    trials: list[dict[str, object]] = []
    if total_product - 1 <= max_variants:
        for values in itertools.product(*(values_by_name[name] for name in mutable)):
            trial = dict(current)
            for name, value in zip(mutable, values, strict=True):
                trial[name] = value
            if any(trial[name] != current.get(name) for name in mutable):
                trials.append(trial)
        rng.shuffle(trials)
        return trials

    singles = [(name, value) for name in mutable for value in values_by_name[name] if value != current.get(name)]
    rng.shuffle(singles)
    for name, value in singles:
        trial = dict(current)
        trial[name] = value
        trials.append(trial)
        if len(trials) >= max_variants:
            return trials

    pairs = list(itertools.combinations(mutable, 2))
    rng.shuffle(pairs)
    for left, right in pairs:
        left_values = [value for value in values_by_name[left] if value != current.get(left)]
        right_values = [value for value in values_by_name[right] if value != current.get(right)]
        rng.shuffle(left_values)
        rng.shuffle(right_values)
        for left_value in left_values:
            for right_value in right_values:
                trial = dict(current)
                trial[left] = left_value
                trial[right] = right_value
                trials.append(trial)
                if len(trials) >= max_variants:
                    return trials
    return trials


def _random_group_trial(
    group: tuple[str, ...],
    current: Mapping[str, object],
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


def _weighted_slot_order(
    slots: list[tuple[Candidate, tuple[str, ...]]],
    *,
    rng: random.Random,
    group_weights: Mapping[str, float] | None,
) -> list[tuple[Candidate, tuple[str, ...]]]:
    rng.shuffle(slots)
    if not group_weights:
        return slots

    def priority(slot: tuple[Candidate, tuple[str, ...]]) -> float:
        weight = max(1e-9, float(group_weights.get(semantic_group_key(slot[1]), 1.0)))
        return -math.log(max(rng.random(), 1e-12)) / weight

    slots.sort(key=priority)
    return slots


def gomea_neighborhood_candidates(
    parents: list[Candidate],
    *,
    count: int,
    max_elites: int | None = 4,
    exclude: set[str] | None = None,
    beam_width: int = 16,
    seed: int = 0,
    source: str = "gomea",
    target_shapes: Sequence[Shape] | None = None,
    group_weights: Mapping[str, float] | None = None,
    micro_exhaustive: bool = False,
) -> list[Candidate]:
    """Sample or enumerate bounded semantic neighborhoods across ranked elites."""
    if count <= 0 or not parents:
        return []
    rng = random.Random(seed)
    parent_pool = parents if max_elites is None else parents[:max_elites]
    seen_hashes = set(exclude or ()) | {parent.hash for parent in parent_pool}
    slots = _weighted_slot_order(
        [(parent, group) for parent in parent_pool for group in neighborhood_group_names()],
        rng=rng,
        group_weights=group_weights,
    )
    if not micro_exhaustive:
        out: list[Candidate] = []
        for parent, group in slots:
            parent_params = parent.canonical_params()
            for _ in range(max(4, beam_width)):
                trial = _random_group_trial(group, parent_params, rng=rng)
                repaired = repair_linked_overrides(trial)
                if not _gomea_candidate_ok(repaired, target_shapes=target_shapes):
                    continue
                try:
                    candidate = make_candidate(
                        repaired,
                        source=source,
                        parents=[parent.hash],
                        proposal_metadata={
                            "semantic_group": semantic_group_key(group),
                            "changed_genes": sorted(name for name in DOMAINS if repaired[name] != parent_params[name]),
                            "enumerated_neighborhood": False,
                        },
                    )
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

    slot_limit = min(len(slots), max(4, min(12, math.ceil(count / 8))))
    variants_per_slot = max(1, min(beam_width, math.ceil(count / max(slot_limit, 1))))
    queues: list[list[Candidate]] = []
    slot_index = 0
    generated = 0
    while slot_index < len(slots) and (slot_index < slot_limit or generated < count):
        parent, group = slots[slot_index]
        slot_index += 1
        parent_params = parent.canonical_params()
        queue: list[Candidate] = []
        for trial in _bounded_group_trials(
            group,
            parent_params,
            rng=rng,
            max_variants=variants_per_slot,
        ):
            requested_transitions = {
                name: {"from": parent_params[name], "to": trial[name]}
                for name in group
                if trial.get(name) != parent_params.get(name)
            }
            repaired = repair_linked_overrides(trial)
            if not _gomea_candidate_ok(repaired, target_shapes=target_shapes):
                continue
            try:
                candidate = make_candidate(
                    repaired,
                    source=source,
                    parents=[parent.hash],
                    proposal_metadata={
                        "semantic_group": semantic_group_key(group),
                        "requested_transitions": requested_transitions,
                        "changed_genes": sorted(name for name in DOMAINS if repaired[name] != parent_params[name]),
                        "enumerated_neighborhood": True,
                    },
                )
            except ValueError:
                continue
            if candidate.hash in seen_hashes:
                continue
            seen_hashes.add(candidate.hash)
            queue.append(candidate)
        if queue:
            queues.append(queue)
            generated += len(queue)

    out: list[Candidate] = []
    while queues and len(out) < count:
        next_queues = []
        for queue in queues:
            if queue and len(out) < count:
                out.append(queue.pop(0))
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return out[:count]


def _genome_with_group(base: tuple[int, ...], donor: tuple[int, ...], group: tuple[int, ...]) -> tuple[int, ...]:
    genes = list(base)
    for idx in group:
        genes[idx] = donor[idx]
    return tuple(genes)


def _gomea_candidate_ok(params: dict[str, object], *, target_shapes: Sequence[Shape] | None = None) -> bool:
    if explain_invalid_nt_hhs(params):
        return False
    if not eligible_for_shape_scope(params, target_shapes):
        return False
    return _valu_vgpr_lower_bound(params) <= NT_HHS_RANDOM_VALU_VGPR_HEADROOM


def _is_rule_valid_genome(genome: tuple[int, ...], *, target_shapes: Sequence[Shape] | None = None) -> bool:
    candidate = genome_to_candidate(genome, source="gomea", repair=True)
    return _gomea_candidate_ok(candidate.canonical_params(), target_shapes=target_shapes)


def _rule_valid_candidate(
    genome: tuple[int, ...],
    *,
    source: str,
    parents: tuple[str, ...],
    target_shapes: Sequence[Shape] | None = None,
    proposal_metadata: Mapping[str, object] | None = None,
) -> Candidate:
    params = genome_to_candidate(genome, source=source, parents=parents, repair=True).canonical_params()
    params = repair_linked_overrides(params)
    if not _gomea_candidate_ok(params, target_shapes=target_shapes):
        raise ValueError("candidate failed NT HHS GOMEA proposal checks")
    return make_candidate(
        params,
        source=source,
        parents=parents,
        proposal_metadata=proposal_metadata,
    )


def _weighted_mode_choice(
    rng: random.Random,
    weights: Mapping[str, float] | None,
) -> str:
    base = {"quality": 0.5, "diverse": 0.3, "random": 0.2}
    resolved = {mode: base[mode] * max(0.0, float(weights.get(mode, 1.0)) if weights else 1.0) for mode in DONOR_MODES}
    total = sum(resolved.values())
    if total <= 0.0:
        return "random"
    threshold = rng.random() * total
    cumulative = 0.0
    for mode in DONOR_MODES:
        cumulative += resolved[mode]
        if cumulative >= threshold:
            return mode
    return DONOR_MODES[-1]


def _choose_donor(
    base_genome: tuple[int, ...],
    donor_pool: list[tuple[Candidate, tuple[int, ...]]],
    *,
    rng: random.Random,
    mode_weights: Mapping[str, float] | None,
    adaptive_selection: bool,
) -> tuple[Candidate, tuple[int, ...], str]:
    mode = _weighted_mode_choice(rng, mode_weights) if adaptive_selection else "random"
    if mode == "quality":
        quality_pool = donor_pool[: max(1, math.ceil(len(donor_pool) * 0.25))]
        candidate, genome = rng.choice(quality_pool)
        return candidate, genome, mode
    if mode == "diverse":
        maximum_distance = max(hamming_distance(base_genome, genome) for _, genome in donor_pool)
        diverse_pool = [item for item in donor_pool if hamming_distance(base_genome, item[1]) == maximum_distance]
        candidate, genome = rng.choice(diverse_pool)
        return candidate, genome, mode
    candidate, genome = rng.choice(donor_pool)
    return candidate, genome, mode


def _group_label(group: tuple[int, ...]) -> str:
    return "|".join(PARAM_NAMES[index] for index in group)


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
    family_local_probability: float = 0.0,
    source: str = "gomea",
    donor_mode_weights: Mapping[str, float] | None = None,
    adaptive_donor_selection: bool = False,
) -> list[Candidate]:
    """Generate linkage-aware categorical candidates from ranked parents."""
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
        donor_pool = [item for item in ranked if item[0].hash != base_candidate.hash]
        family_local = False
        if family_local_probability > 0.0 and rng.random() < family_local_probability:
            base_family = family_descriptor(base_candidate)
            local_pool = [item for item in donor_pool if family_descriptor(item[0]) == base_family]
            if local_pool:
                donor_pool = local_pool
                family_local = True
        if not donor_pool:
            continue
        donor_candidate, donor_genome, donor_mode = _choose_donor(
            base_genome,
            donor_pool,
            rng=rng,
            mode_weights=donor_mode_weights,
            adaptive_selection=adaptive_donor_selection,
        )
        genes = base_genome
        model = nearest_linkage_model(base_genome, linkage_models or [])
        model_groups = list(model.fos_groups) if model is not None else []
        groups = [*rule_groups, *model_groups] if model_groups else list(fallback_fos)
        rng.shuffle(groups)
        changed = False
        applied_groups: list[tuple[int, ...]] = []
        for group in groups[: rng.randint(1, max(1, min(4, len(groups))))]:
            trial = _genome_with_group(genes, donor_genome, group)
            try:
                if _is_rule_valid_genome(trial, target_shapes=target_shapes):
                    genes = trial
                    changed = True
                    applied_groups.append(group)
            except ValueError:
                continue
        if include_forced_improvement and (not changed or genes == base_genome):
            elite = _nearest_elite(base_genome, elite_genomes)
            group = rng.choice(groups)
            trial = _genome_with_group(genes, elite, group)
            try:
                if _is_rule_valid_genome(trial, target_shapes=target_shapes):
                    genes = trial
                    applied_groups.append(group)
            except ValueError:
                pass
        try:
            child = _rule_valid_candidate(
                genes,
                source=source,
                parents=(base_candidate.hash, donor_candidate.hash),
                target_shapes=target_shapes,
                proposal_metadata={
                    "donor_mode": donor_mode,
                    "family_local": family_local,
                    "donor_distance": hamming_distance(base_genome, donor_genome),
                    "mixed_groups": [_group_label(group) for group in applied_groups],
                    "changed_genes": [
                        PARAM_NAMES[index]
                        for index, (before, after) in enumerate(zip(base_genome, genes, strict=True))
                        if before != after
                    ],
                },
            )
        except ValueError:
            continue
        out.append(child)
    return dedupe_candidates(out, exclude=exclude)[:count]
