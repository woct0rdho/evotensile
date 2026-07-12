import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from evotensile.candidate import Candidate, Shape
from evotensile.database import BenchmarkEventInsert
from evotensile.runner import RunResult
from evotensile.structured_runner import RunnablePair, StructuredRunOutput


class EvidenceStage(str, Enum):
    PROBE = "probe"
    SCREENING = "screening"
    STABILIZATION = "stabilization"
    CONFIRMATION = "confirmation"


@dataclass(frozen=True)
class PairRequest:
    candidate: Candidate
    shape: Shape
    evidence_stage: EvidenceStage = EvidenceStage.SCREENING
    min_samples: int = 1
    priority: float = 0.0

    def __post_init__(self) -> None:
        if self.min_samples <= 0:
            raise ValueError("pair-request minimum samples must be positive")
        if not math.isfinite(self.priority):
            raise ValueError("pair-request priority must be finite")

    @property
    def key(self) -> tuple[str, str]:
        return self.shape.id, self.candidate.hash


@dataclass(frozen=True)
class PlannedPair:
    request: PairRequest
    samples_to_collect: int
    requires_validation: bool

    def __post_init__(self) -> None:
        if self.samples_to_collect <= 0:
            raise ValueError("planned pair samples must be positive")

    @property
    def key(self) -> tuple[str, str]:
        return self.request.key


@dataclass(frozen=True)
class PlannedBatch:
    batch_index: int
    pairs: tuple[PlannedPair, ...]
    artifact_candidates: tuple[Candidate, ...]
    artifact_shapes: tuple[Shape, ...]
    evidence_stage: EvidenceStage

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ValueError("planned batch requires at least one exact pair")
        if not self.artifact_candidates:
            raise ValueError("planned batch requires at least one artifact candidate")
        if not self.artifact_shapes:
            raise ValueError("planned batch requires at least one artifact shape")
        candidate_hashes = {candidate.hash for candidate in self.artifact_candidates}
        shape_ids = {shape.id for shape in self.artifact_shapes}
        keys = [pair.key for pair in self.pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("planned batch exact pairs must be unique")
        for pair in self.pairs:
            if pair.request.evidence_stage != self.evidence_stage:
                raise ValueError("planned batch evidence stages must match")
            if pair.request.candidate.hash not in candidate_hashes or pair.request.shape.id not in shape_ids:
                raise ValueError("planned pair must be covered by the artifact scope")

    @property
    def requested_pairs(self) -> int:
        return len(self.pairs)

    @property
    def requested_samples(self) -> int:
        return sum(pair.samples_to_collect for pair in self.pairs)

    @property
    def requires_validation(self) -> bool:
        return any(pair.requires_validation for pair in self.pairs)

    @property
    def priority(self) -> float:
        return max(pair.request.priority for pair in self.pairs)

    @property
    def pair_keys(self) -> set[tuple[str, str]]:
        return {pair.key for pair in self.pairs}

    def planned_pair(self, key: tuple[str, str]) -> PlannedPair:
        for pair in self.pairs:
            if pair.key == key:
                return pair
        raise KeyError(key)


@dataclass(frozen=True)
class BatchIngestResult:
    inserted: int
    unmapped: int
    status_counts: dict[str, int]
    rejected: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class PreparedBatch:
    planned: PlannedBatch
    yaml_path: Path
    manifest_path: Path
    output_dir: Path
    build_output_dir: Path
    build_result: RunResult
    library_dir: Path | None
    validated_pairs: list[RunnablePair]
    preparation_inserts: list[BenchmarkEventInsert]
    validation_result: StructuredRunOutput | None = None
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutedBatch:
    planned: PlannedBatch
    yaml_path: Path
    manifest_path: Path
    output_dir: Path
    build_returncode: int | None = None
    validation_returncode: int | None = None
    runner_returncode: int | None = None
    ingest: BatchIngestResult | None = None
    build_output_dir: Path | None = None
    phase: str = "initial"


@dataclass(frozen=True)
class ScheduleResult:
    planned_batches: list[PlannedBatch]
    executed_batches: list[ExecutedBatch] = field(default_factory=list)
    build_timeout_s: float | None = None
    runner_timeout_s: float | None = None
    candidate_batch_size: int = 1
    shape_batch_size: int = 1
    prepare_workers: int = 1
    prepare_wave_batches: int = 1
    validation_workers: int = 1
    runner_bin: str | None = None
    completed_waves: int = 0
    adaptive_rounds: int = 0
    probe_protocol_hash: str | None = None
    probe_policy_hash: str | None = None
    probe_survivor_pairs: int = 0
    probe_screened_pairs: int = 0
    probe_preprepare_screened_pairs: int = 0

    @property
    def requested_pairs(self) -> int:
        return sum(batch.requested_pairs for batch in self.planned_batches)

    @property
    def requested_samples(self) -> int:
        return sum(batch.requested_samples for batch in self.planned_batches)
