from evotensile.candidate import Candidate
from scripts.run_grid100_practical_round import _parent_diverse_selection


def _candidate(name: str, parent: str | None) -> Candidate:
    return Candidate(
        params={"name": name},
        source="test",
        parent_hashes=() if parent is None else (parent,),
    )


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
