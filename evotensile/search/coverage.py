from collections import Counter
from collections.abc import Iterable
from typing import Any, TypedDict

from evotensile.candidate import Candidate
from evotensile.search.encoding import PARAM_NAMES
from evotensile.search_space import explain_invalid_nt_hhs


class CandidateCoverage(TypedDict):
    candidates: int
    unique_candidate_hashes: int
    unique_values: dict[str, int]
    invalid_reason_counts: dict[str, int]


def _freeze(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ",".join(_freeze(item) for item in value) + "]"
    if isinstance(value, tuple):
        return "(" + ",".join(_freeze(item) for item in value) + ")"
    return repr(value)


def candidate_coverage(candidates: Iterable[Candidate]) -> CandidateCoverage:
    items = list(candidates)
    invalid_reason_counts: Counter[str] = Counter()
    for candidate in items:
        params = candidate.canonical_params()
        for reason in explain_invalid_nt_hhs(params):
            invalid_reason_counts[reason.rule_id] += 1
    unique_values: dict[str, int] = {}
    for name in PARAM_NAMES:
        unique_values[name] = len({_freeze(candidate.canonical_params().get(name)) for candidate in items})
    return {
        "candidates": len(items),
        "unique_candidate_hashes": len({candidate.hash for candidate in items}),
        "unique_values": unique_values,
        "invalid_reason_counts": dict(invalid_reason_counts),
    }
