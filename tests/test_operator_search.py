import random

from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.search.encoding import candidate_to_genome, hamming_distance
from evotensile.search.family import family_descriptor
from evotensile.search.gomea import gomea_candidates
from evotensile.search.local_search import semantic_mutation_candidates
from evotensile.search.operator_credit import OperatorCredit, allocate_operator_budget, load_operator_credits
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
    assert all(credits[arm].failures == 1 for arm in arms[1:])
    assert credits["semantic-mutation"].posterior_mean > max(credits[arm].posterior_mean for arm in arms[1:])


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
