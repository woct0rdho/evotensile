from evotensile.candidate import Candidate, Shape
from evotensile.search_space import make_candidate, repair_linked_overrides
from scripts.run_grid100_practical_round import (
    _interaction_pool,
    _parent_diverse_selection,
    _parent_winner_count_payload,
)


def _candidate(name: str, parent: str | None) -> Candidate:
    return Candidate(
        params={"name": name},
        source="test",
        parent_hashes=() if parent is None else (parent,),
    )


def test_staging_interactions_include_cluster_local_read():
    parent = make_candidate(repair_linked_overrides({"ClusterLocalRead": 1}), source="parent")

    candidates = _interaction_pool(
        (parent,),
        {parent.hash: (Shape(64, 64, 1, 256),)},
        profile="staging",
        store_batch_values=(0, 8),
        known_candidate_hashes=set(),
    )

    assert any(candidate.params["ClusterLocalRead"] == 0 for candidate in candidates)


def test_parent_winner_count_payload_supports_explicit_nonincumbents():
    parent = _candidate("explicit", None)

    assert _parent_winner_count_payload((parent,), {}) == {parent.hash: 0}


def test_parent_diverse_selection_reserves_general_parents_only():
    dominant_one = _candidate("dominant-one", "parent-dominant")
    dominant_two = _candidate("dominant-two", "parent-dominant")
    repair = _candidate("repair", "parent-repair")
    singleton = _candidate("singleton", "parent-singleton")
    ranked = (dominant_one, dominant_two, repair, singleton)

    selected = _parent_diverse_selection(
        ranked,
        general_candidate_hashes={dominant_one.hash, dominant_two.hash, singleton.hash},
        limit=3,
    )

    assert selected == (dominant_one, dominant_two, singleton)
