from evotensile.search.differential_evolution import differential_evolution_candidates
from evotensile.search.encoding import candidate_to_genome, genome_to_candidate, ordered_domain_values
from evotensile.search.gomea import gomea_neighborhood_candidates
from evotensile.search.random_search import initial_random_batch
from evotensile.search_space import documented_winner_candidate, known_seed_candidates


def test_encoding_round_trips_complete_candidate():
    candidate = known_seed_candidates()[0]
    genome = candidate_to_genome(candidate)
    decoded = genome_to_candidate(genome, source="roundtrip")

    assert decoded.hash == candidate.hash
    assert ordered_domain_values("ScheduleIterAlg", 3)[0] == 3


def test_differential_evolution_generates_valid_candidates():
    parents = initial_random_batch(4, seed=7)
    proposed = differential_evolution_candidates(parents, count=8, seed=11, exclude={parent.hash for parent in parents})

    assert proposed
    assert len(proposed) <= 8
    assert {candidate.source for candidate in proposed} == {"de"}
    assert {candidate.hash for candidate in proposed}.isdisjoint({parent.hash for parent in parents})


def test_gomea_neighborhood_can_compose_linked_knobs():
    parents = initial_random_batch(0, seed=1151)
    proposed = gomea_neighborhood_candidates(
        parents, count=16, max_elites=None, exclude={parent.hash for parent in parents}
    )

    assert documented_winner_candidate().hash in {candidate.hash for candidate in proposed}
    assert {candidate.source for candidate in proposed} == {"gomea"}
