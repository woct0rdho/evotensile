import random

from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.search.encoding import candidate_to_genome, hamming_distance
from evotensile.search.family import family_descriptor
from evotensile.search.gomea import gomea_candidates, gomea_neighborhood_candidates
from evotensile.search.local_search import semantic_mutation_candidates
from evotensile.search.operator_credit import (
    OperatorCredit,
    allocate_operator_budget,
    credit_ucb_scores,
    load_donor_mode_credits,
    load_operator_credits,
    load_semantic_group_credits,
)
from evotensile.search.semantics import semantic_group_key, semantic_group_names
from evotensile.search_space import DOMAINS, make_candidate, random_candidate
from evotensile.shapes import Shape
from tests.helpers import REFERENCE_CANDIDATE


def _family_variants(base, count: int):
    out = []
    for index, stagger_u in enumerate(DOMAINS["StaggerU"]):
        params = base.canonical_params()
        params["StaggerU"] = stagger_u
        params["StaggerUMapping"] = index % 2
        candidate = make_candidate(params, source="parent")
        if candidate.hash not in {item.hash for item in out}:
            out.append(candidate)
        if len(out) >= count:
            return out
    raise AssertionError(f"failed to create {count} family variants")


def test_semantic_mutation_uses_small_valid_steps():
    shape = Shape(8192, 8192, 1, 8192)
    parents = [random_candidate(random.Random(seed), target_shapes=[shape]) for seed in range(20260710, 20260718)]
    parent_by_hash = {candidate.hash: candidate for candidate in parents}

    children = semantic_mutation_candidates(parents, count=16, seed=20260711, target_shapes=[shape])

    assert len(children) == 16
    assert {candidate.source for candidate in children} == {"semantic-mutation"}
    distances = [
        hamming_distance(
            candidate_to_genome(child),
            candidate_to_genome(parent_by_hash[child.parent_hashes[0]]),
        )
        for child in children
    ]
    assert min(distances) == 1
    assert sorted(distances)[len(distances) // 2] <= 3


def test_semantic_mutation_records_and_uses_group_credit():
    shape = Shape(8192, 8192, 1, 8192)
    parents = [random_candidate(random.Random(seed), target_shapes=[shape]) for seed in range(20260720, 20260732)]
    target_group = ("StaggerU",)
    weights = {semantic_group_key(group): 0.0 for group in semantic_group_names()}
    weights[semantic_group_key(target_group)] = 100.0

    children = semantic_mutation_candidates(
        parents,
        count=8,
        seed=20260713,
        target_shapes=[shape],
        max_changed_genes=1,
        group_weights=weights,
    )

    assert len(children) == 8
    assert {child.proposal_metadata["semantic_group"] for child in children} == {"StaggerU"}
    assert all("StaggerU" in child.proposal_metadata["requested_transitions"] for child in children)


def test_micro_exhaustive_neighborhood_enumerates_small_group():
    parent = REFERENCE_CANDIDATE
    target_group = ("StaggerU",)
    weights = {semantic_group_key(group): 0.0 for group in semantic_group_names()}
    weights[semantic_group_key(target_group)] = 100.0

    children = gomea_neighborhood_candidates(
        [parent],
        count=16,
        max_elites=None,
        seed=20260714,
        source="gomea-neighborhood",
        group_weights=weights,
        micro_exhaustive=True,
    )

    target_children = [child for child in children if child.proposal_metadata["semantic_group"] == "StaggerU"]
    target_values = {child.canonical_params()["StaggerU"] for child in target_children}
    assert target_values == set(DOMAINS["StaggerU"]) - {parent.canonical_params()["StaggerU"]}
    assert all(child.proposal_metadata["enumerated_neighborhood"] for child in children)


def test_gomea_family_local_donors_stay_in_the_base_family():
    first_family = _family_variants(REFERENCE_CANDIDATE, 4)
    rng = random.Random(20260710)
    other = random_candidate(rng)
    while family_descriptor(other) == family_descriptor(REFERENCE_CANDIDATE):
        other = random_candidate(rng)
    second_family = _family_variants(other, 4)
    parents = [*first_family, *second_family]
    parent_by_hash = {candidate.hash: candidate for candidate in parents}

    children = gomea_candidates(
        parents,
        count=12,
        seed=20260712,
        family_local_probability=1.0,
        source="gomea-mixing",
    )

    assert children
    assert {candidate.source for candidate in children} == {"gomea-mixing"}
    for child in children:
        assert len(child.parent_hashes) == 2
        left, right = (parent_by_hash[parent_hash] for parent_hash in child.parent_hashes)
        assert family_descriptor(left) == family_descriptor(right)


def test_operator_credit_and_ucb_allocation_use_only_queried_parent_comparisons(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "operators.sqlite")
    db.init()
    shape = Shape(8192, 8192, 1, 8192)
    parents = _family_variants(REFERENCE_CANDIDATE, 4)
    arms = ("semantic-mutation", "de", "gomea-neighborhood", "gomea-mixing")
    children = []
    for index, (arm, parent) in enumerate(zip(arms, parents, strict=True)):
        params = parent.canonical_params()
        params["NumElementsPerBatchStore"] = DOMAINS["NumElementsPerBatchStore"][index + 1]
        children.append(make_candidate(params, source=arm, parents=[parent.hash]))
    db.register_candidates([*parents, *children])
    db.record_proposal_occurrences(
        children,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        scope_kind="shape",
        scope_shape_ids=(shape.id,),
        selected_candidates=children,
    )
    db.register_shapes([shape])
    problem_hash = DEFAULT_PROFILE.problem_type_hash
    protocol_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    for parent in parents:
        db.insert_evaluation(
            shape_id=shape.id,
            candidate_hash=parent.hash,
            run_id="parent",
            status="ok",
            problem_type_hash=problem_hash,
            benchmark_protocol_hash=protocol_hash,
            time_us=100.0,
            validation="PASSED",
        )
    for index, child in enumerate(children):
        db.insert_evaluation(
            shape_id=shape.id,
            candidate_hash=child.hash,
            run_id="child",
            status="ok",
            problem_type_hash=problem_hash,
            benchmark_protocol_hash=protocol_hash,
            time_us=90.0 if index == 0 else 110.0,
            validation="PASSED",
        )

    credits = load_operator_credits(
        db,
        problem_type_hash=problem_hash,
        benchmark_protocol_hash=protocol_hash,
        shapes=[shape],
    )

    assert credits["semantic-mutation"].successes == 1
    assert credits["semantic-mutation"].failures == 0
    assert credits["semantic-mutation"].shape_comparisons == 1
    assert all(credits[arm].failures == 1 for arm in arms[1:])
    assert credits["semantic-mutation"].posterior_mean > max(credits[arm].posterior_mean for arm in arms[1:])


def test_group_and_donor_credits_use_persisted_proposal_metadata(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "metadata-credits.sqlite")
    db.init()
    shape = Shape(8192, 8192, 1, 8192)
    parent = REFERENCE_CANDIDATE
    semantic_params = parent.canonical_params()
    semantic_params["StaggerU"] = DOMAINS["StaggerU"][1]
    semantic_child = make_candidate(
        semantic_params,
        source="semantic-mutation",
        parents=[parent.hash],
        proposal_metadata={"semantic_group": "StaggerU"},
    )
    donor_params = parent.canonical_params()
    donor_params["WorkGroupMapping"] = DOMAINS["WorkGroupMapping"][1]
    donor_child = make_candidate(
        donor_params,
        source="gomea-mixing",
        parents=[parent.hash],
        proposal_metadata={"donor_mode": "diverse", "donor_distance": 4},
    )
    db.register_candidates([parent, semantic_child, donor_child])
    db.record_proposal_occurrences(
        [semantic_child, donor_child],
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        scope_kind="shape",
        scope_shape_ids=(shape.id,),
        selected_candidates=[semantic_child, donor_child],
    )
    db.register_shapes([shape])
    problem_hash = DEFAULT_PROFILE.problem_type_hash
    protocol_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    for candidate, time_us in ((parent, 100.0), (semantic_child, 90.0), (donor_child, 110.0)):
        db.insert_evaluation(
            shape_id=shape.id,
            candidate_hash=candidate.hash,
            run_id="queried",
            status="ok",
            problem_type_hash=problem_hash,
            benchmark_protocol_hash=protocol_hash,
            time_us=time_us,
            validation="PASSED",
        )

    semantic_credits = load_semantic_group_credits(
        db,
        problem_type_hash=problem_hash,
        benchmark_protocol_hash=protocol_hash,
        shapes=[shape],
    )
    donor_credits = load_donor_mode_credits(
        db,
        problem_type_hash=problem_hash,
        benchmark_protocol_hash=protocol_hash,
        shapes=[shape],
    )
    restored = db.get_candidates([semantic_child.hash, donor_child.hash])

    assert semantic_credits["StaggerU"].successes == 1
    assert donor_credits["diverse"].failures == 1
    assert restored[0].proposal_metadata["semantic_group"] == "StaggerU"
    assert restored[1].proposal_metadata["donor_mode"] == "diverse"
    assert credit_ucb_scores(semantic_credits)["StaggerU"] > 0.0


def test_operator_credit_aggregates_correlated_shapes_per_occurrence_and_credits_reproposal(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "event-credits.sqlite")
    db.init()
    shapes = [Shape(512, 128, 1, 256), Shape(1024, 1024, 1, 1024)]
    parent = REFERENCE_CANDIDATE
    child_params = parent.canonical_params()
    child_params["StaggerU"] = DOMAINS["StaggerU"][1]
    first_registration = make_candidate(child_params, source="random")
    reproposal = make_candidate(child_params, source="de", parents=[parent.hash])
    db.register_candidates([parent, first_registration])
    db.record_proposal_occurrences(
        [reproposal],
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        scope_kind="shape-set",
        scope_shape_ids=tuple(shape.id for shape in shapes),
        selected_candidates=[reproposal],
    )
    db.record_proposal_occurrences(
        [reproposal],
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        scope_kind="shape-set",
        scope_shape_ids=tuple(shape.id for shape in shapes),
        selected_candidates=[reproposal],
    )
    db.register_shapes(shapes)
    problem_hash = DEFAULT_PROFILE.problem_type_hash
    protocol_hash = DEFAULT_PROFILE.benchmark_protocol_hash()
    for shape, child_time in zip(shapes, (80.0, 125.0), strict=True):
        for candidate_hash, time_us in ((parent.hash, 100.0), (reproposal.hash, child_time)):
            db.insert_evaluation(
                shape_id=shape.id,
                candidate_hash=candidate_hash,
                run_id="queried",
                status="ok",
                problem_type_hash=problem_hash,
                benchmark_protocol_hash=protocol_hash,
                time_us=time_us,
                validation="PASSED",
            )

    equal_credits = load_operator_credits(
        db,
        problem_type_hash=problem_hash,
        benchmark_protocol_hash=protocol_hash,
        shapes=shapes,
    )
    weighted_credits = load_operator_credits(
        db,
        problem_type_hash=problem_hash,
        benchmark_protocol_hash=protocol_hash,
        shapes=shapes,
        shape_weights={shapes[0].id: 4.0, shapes[1].id: 1.0},
    )
    restored = db.get_candidates([reproposal.hash])[0]

    assert equal_credits["de"].trials == 1
    assert equal_credits["de"].failures == 1
    assert equal_credits["de"].shape_comparisons == 2
    assert weighted_credits["de"].trials == 1
    assert weighted_credits["de"].successes == 1
    assert restored.source == "random"


def test_adaptive_donor_selection_records_quality_diversity_strategy():
    parents = _family_variants(REFERENCE_CANDIDATE, 4)

    children = gomea_candidates(
        parents,
        count=8,
        seed=20260715,
        source="gomea-mixing",
        adaptive_donor_selection=True,
        donor_mode_weights={"quality": 100.0, "diverse": 0.0, "random": 0.0},
    )

    assert children
    assert {child.proposal_metadata["donor_mode"] for child in children} == {"quality"}
    assert all(int(child.proposal_metadata["donor_distance"]) > 0 for child in children)


def test_operator_budget_is_neutral_then_rewards_repeated_success():
    arms = ("semantic-mutation", "de", "gomea-neighborhood", "gomea-mixing")
    neutral = {arm: OperatorCredit(arm=arm) for arm in arms}

    neutral_allocation = allocate_operator_budget(20, neutral)
    informed = dict(neutral)
    informed["gomea-neighborhood"] = OperatorCredit(arm="gomea-neighborhood", successes=6, failures=1)
    informed["de"] = OperatorCredit(arm="de", successes=1, failures=6)
    informed_allocation = allocate_operator_budget(20, informed)

    assert set(neutral_allocation.values()) == {5}
    assert informed_allocation["gomea-neighborhood"] > informed_allocation["de"]
    assert sum(informed_allocation.values()) == 20
