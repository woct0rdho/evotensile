from evotensile.search_space import known_seed_candidates, random_candidates
from evotensile.shapes import pilot_100_shapes
from evotensile.yaml_writer import tensile_config


def test_pilot_shape_count():
    assert len(pilot_100_shapes()) == 100


def test_candidate_hash_stable():
    a = known_seed_candidates()[0]
    b = known_seed_candidates()[0]
    assert a.hash == b.hash
    assert a.hash.startswith("cand_")


def test_random_candidate_count():
    cands = random_candidates(8, seed=123)
    assert len(cands) == 8
    assert len({c.hash for c in cands}) == 8


def test_yaml_shape():
    cands = known_seed_candidates()[:2]
    shapes = pilot_100_shapes()[:3]
    data = tensile_config(cands, shapes)
    assert "BenchmarkProblems" in data
    group = data["BenchmarkProblems"][0][1]["ForkParameters"][0]["Groups"][0]
    assert len(group) == 2
    sizes = data["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"][0]["ProblemSizes"]
    assert len(sizes) == 3
