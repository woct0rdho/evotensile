from dataclasses import dataclass
from typing import Any

from .candidate import stable_hash
from .profile import DEFAULT_PROFILE
from .protocol import DEFAULT_BENCHMARK_PROTOCOL, BenchmarkProtocol

POSITIVE_CACHE_STATUSES = ("ok",)
NEGATIVE_CACHE_STATUSES = ("rejected", "validation_fail", "build_failed")
REUSABLE_CACHE_STATUSES = (*POSITIVE_CACHE_STATUSES, *NEGATIVE_CACHE_STATUSES)


@dataclass(frozen=True)
class CacheKey:
    problem_type_hash: str
    benchmark_protocol_hash: str
    shape_id: str
    candidate_hash: str


def problem_type_hash(problem_type: dict[str, Any] | None = None) -> str:
    return stable_hash(problem_type or DEFAULT_PROFILE.problem_type, prefix="ptype_")[:22]


def benchmark_protocol_hash(protocol: BenchmarkProtocol | None = None) -> str:
    return (protocol or DEFAULT_BENCHMARK_PROTOCOL).protocol_hash()
