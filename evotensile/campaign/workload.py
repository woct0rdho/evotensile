import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict, cast

WorkloadMode = Literal["uniform", "workload"]


class ShapeWorkloadPayload(TypedDict):
    shape_id: str
    call_count: float
    baseline_latency_us: float
    baseline_time_contribution_us: float


class ResolvedWorkloadPayload(TypedDict):
    mode: WorkloadMode
    shape_ids: list[str]
    weights: dict[str, float]
    entries: list[ShapeWorkloadPayload]
    provenance: dict[str, str]
    total_call_count: float | None
    total_baseline_time_us: float | None


class WorkloadFilePayload(TypedDict):
    provenance: dict[str, str]
    shapes: list[ShapeWorkloadPayload]


_REQUIRED_PROVENANCE_FIELDS = frozenset(
    {
        "call_count_source",
        "baseline_label",
        "baseline_source",
        "benchmark_protocol_hash",
        "environment_compatibility_tag",
    }
)


@dataclass(frozen=True)
class ShapeWorkload:
    shape_id: str
    call_count: float
    baseline_latency_us: float

    def __post_init__(self) -> None:
        if not self.shape_id:
            raise ValueError("workload shape ID must be non-empty")
        if not math.isfinite(self.call_count) or self.call_count < 0.0:
            raise ValueError("workload call count must be finite and nonnegative")
        if not math.isfinite(self.baseline_latency_us) or self.baseline_latency_us <= 0.0:
            raise ValueError("workload baseline latency must be finite and positive")

    @property
    def baseline_time_contribution_us(self) -> float:
        return self.call_count * self.baseline_latency_us

    def to_dict(self) -> ShapeWorkloadPayload:
        return {
            "shape_id": self.shape_id,
            "call_count": self.call_count,
            "baseline_latency_us": self.baseline_latency_us,
            "baseline_time_contribution_us": self.baseline_time_contribution_us,
        }


@dataclass(frozen=True)
class ResolvedWorkloadWeights:
    mode: WorkloadMode
    shape_ids: tuple[str, ...]
    weights: dict[str, float]
    entries: tuple[ShapeWorkload, ...] = ()
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"uniform", "workload"}:
            raise ValueError("workload mode must be uniform or workload")
        if not self.shape_ids or len(set(self.shape_ids)) != len(self.shape_ids):
            raise ValueError("workload shape IDs must be non-empty and unique")
        if set(self.weights) != set(self.shape_ids):
            raise ValueError("workload weights must cover the exact shape set")
        if any(not math.isfinite(value) or value < 0.0 for value in self.weights.values()):
            raise ValueError("workload weights must be finite and nonnegative")
        if not math.isclose(
            sum(self.weights.values()),
            float(len(self.shape_ids)),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("workload weights must normalize to the shape count")
        entry_ids = {entry.shape_id for entry in self.entries}
        if any(not key or not value for key, value in self.provenance.items()):
            raise ValueError("workload provenance keys and values must be non-empty")
        if self.mode == "uniform" and (self.entries or self.provenance):
            raise ValueError("uniform workload mode cannot carry entries or provenance")
        if self.mode == "workload" and entry_ids != set(self.shape_ids):
            raise ValueError("workload entries must cover the exact shape set")
        if self.mode == "workload" and not _REQUIRED_PROVENANCE_FIELDS.issubset(self.provenance):
            missing = sorted(_REQUIRED_PROVENANCE_FIELDS - self.provenance.keys())
            raise ValueError(f"workload provenance is missing required fields: {missing}")

    @classmethod
    def uniform(cls, shape_ids: Sequence[str]) -> "ResolvedWorkloadWeights":
        ordered = tuple(str(shape_id) for shape_id in shape_ids)
        return cls(
            mode="uniform",
            shape_ids=ordered,
            weights={shape_id: 1.0 for shape_id in ordered},
        )

    @classmethod
    def workload(
        cls,
        shape_ids: Sequence[str],
        entries: Sequence[ShapeWorkload],
        *,
        provenance: Mapping[str, str],
    ) -> "ResolvedWorkloadWeights":
        ordered = tuple(str(shape_id) for shape_id in shape_ids)
        by_shape = {entry.shape_id: entry for entry in entries}
        if len(by_shape) != len(entries) or set(by_shape) != set(ordered):
            raise ValueError("workload entries must uniquely cover the exact shape set")
        total = sum(entry.baseline_time_contribution_us for entry in by_shape.values())
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("workload baseline-time contribution must be positive")
        scale = len(ordered) / total
        return cls(
            mode="workload",
            shape_ids=ordered,
            weights={shape_id: by_shape[shape_id].baseline_time_contribution_us * scale for shape_id in ordered},
            entries=tuple(by_shape[shape_id] for shape_id in ordered),
            provenance={str(key): str(value) for key, value in provenance.items()},
        )

    @classmethod
    def from_dict(cls, payload: ResolvedWorkloadPayload) -> "ResolvedWorkloadWeights":
        entries = [
            ShapeWorkload(
                shape_id=entry["shape_id"],
                call_count=float(entry["call_count"]),
                baseline_latency_us=float(entry["baseline_latency_us"]),
            )
            for entry in payload["entries"]
        ]
        mode = payload["mode"]
        if mode not in {"uniform", "workload"}:
            raise ValueError("workload mode must be uniform or workload")
        return cls(
            mode=mode,
            shape_ids=tuple(payload["shape_ids"]),
            weights=dict(payload["weights"]),
            entries=tuple(entries),
            provenance=dict(payload["provenance"]),
        )

    @property
    def total_call_count(self) -> float | None:
        if self.mode == "uniform":
            return None
        return sum(entry.call_count for entry in self.entries)

    @property
    def total_baseline_time_us(self) -> float | None:
        if self.mode == "uniform":
            return None
        return sum(entry.baseline_time_contribution_us for entry in self.entries)

    def to_dict(self) -> ResolvedWorkloadPayload:
        return {
            "mode": self.mode,
            "shape_ids": list(self.shape_ids),
            "weights": dict(sorted(self.weights.items())),
            "entries": [entry.to_dict() for entry in self.entries],
            "provenance": dict(sorted(self.provenance.items())),
            "total_call_count": self.total_call_count,
            "total_baseline_time_us": self.total_baseline_time_us,
        }


def load_workload_weights(
    path: str | Path,
    *,
    shape_ids: Sequence[str],
) -> ResolvedWorkloadWeights:
    payload = cast(
        WorkloadFilePayload,
        json.loads(Path(path).read_text(encoding="utf-8")),
    )
    entries = [
        ShapeWorkload(
            shape_id=entry["shape_id"],
            call_count=float(entry["call_count"]),
            baseline_latency_us=float(entry["baseline_latency_us"]),
        )
        for entry in payload["shapes"]
    ]
    return ResolvedWorkloadWeights.workload(
        shape_ids,
        entries,
        provenance=payload["provenance"],
    )
