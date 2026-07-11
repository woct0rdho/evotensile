import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def canonicalize(value: Any) -> Any:
    """Return a JSON-stable form for nested candidate/config values."""
    if isinstance(value, Mapping):
        return {str(k): canonicalize(value[k]) for k in sorted(value)}
    if isinstance(value, tuple):
        return [canonicalize(v) for v in value]
    if isinstance(value, list):
        return [canonicalize(v) for v in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(canonicalize(value), sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any, *, prefix: str = "") -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}{digest}" if prefix else digest


@dataclass(frozen=True)
class Candidate:
    """A complete TensileLite candidate bundle emitted as one Groups entry."""

    params: Mapping[str, Any]
    source: str = "unknown"
    parent_hashes: tuple[str, ...] = field(default_factory=tuple)
    proposal_metadata: Mapping[str, Any] = field(default_factory=dict)

    def canonical_params(self) -> dict[str, Any]:
        return canonicalize(dict(self.params))

    @property
    def hash(self) -> str:
        return stable_hash(self.canonical_params(), prefix="cand_")[:21]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Candidate":
        params = payload.get("params")
        parent_hashes = payload.get("parent_hashes", [])
        proposal_metadata = payload.get("proposal_metadata", {})
        if not isinstance(params, Mapping):
            raise ValueError("candidate params must be a mapping")
        if not isinstance(parent_hashes, (list, tuple)):
            raise ValueError("candidate parent hashes must be a sequence")
        if not isinstance(proposal_metadata, Mapping):
            raise ValueError("candidate proposal metadata must be a mapping")
        candidate = cls(
            params={str(key): value for key, value in params.items()},
            source=str(payload.get("source", "unknown")),
            parent_hashes=tuple(str(value) for value in parent_hashes),
            proposal_metadata={str(key): value for key, value in proposal_metadata.items()},
        )
        expected_hash = payload.get("candidate_hash", payload.get("hash"))
        if expected_hash is not None and candidate.hash != str(expected_hash):
            raise ValueError(f"candidate hash mismatch: expected {expected_hash}, got {candidate.hash}")
        return candidate

    def to_mapping(self, *, hash_key: str = "hash") -> dict[str, Any]:
        return {
            hash_key: self.hash,
            "source": self.source,
            "parent_hashes": list(self.parent_hashes),
            "proposal_metadata": canonicalize(dict(self.proposal_metadata)),
            "params": self.canonical_params(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), sort_keys=True, indent=2)


@dataclass(frozen=True)
class Shape:
    m: int
    n: int
    batch: int
    k: int

    @property
    def id(self) -> str:
        return f"m{self.m}_n{self.n}_b{self.batch}_k{self.k}"

    def exact_list(self) -> list[int]:
        # TensileLite batched GEMM order for this problem type is [M, N, batch, K].
        return [self.m, self.n, self.batch, self.k]

    def features(self) -> dict[str, float]:
        m = float(self.m)
        n = float(self.n)
        k = float(self.k)
        return {
            "log2_m": math.log2(m),
            "log2_n": math.log2(n),
            "log2_k": math.log2(k),
            "log2_m_over_n": math.log2(m / n),
            "log2_k_over_m": math.log2(k / m),
            "log2_k_over_n": math.log2(k / n),
        }
