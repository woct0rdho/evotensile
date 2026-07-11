from dataclasses import dataclass

POSITIVE_CACHE_STATUSES = ("ok",)
NEGATIVE_CACHE_STATUSES = ("rejected", "build_failed")
REUSABLE_CACHE_STATUSES = (*POSITIVE_CACHE_STATUSES, *NEGATIVE_CACHE_STATUSES)


@dataclass(frozen=True)
class CacheKey:
    problem_type_hash: str
    benchmark_protocol_hash: str
    shape_id: str
    candidate_hash: str
