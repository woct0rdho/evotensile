from dataclasses import dataclass, field
from pathlib import Path

from evotensile.candidate import Candidate, Shape
from evotensile.database import BenchmarkEventInsert
from evotensile.runner import RunResult
from evotensile.structured_runner import RunnablePair, StructuredRunOutput


@dataclass(frozen=True)
class PlannedBatch:
    batch_index: int
    candidates: list[Candidate]
    shapes: list[Shape]
    missing_pairs: int
    nominal_pairs: int
    samples_per_pair: int
    requires_validation: bool = True

    @property
    def extra_pairs(self) -> int:
        return self.nominal_pairs - self.missing_pairs

    @property
    def missing_samples(self) -> int:
        return self.missing_pairs * self.samples_per_pair

    @property
    def nominal_samples(self) -> int:
        return self.nominal_pairs * self.samples_per_pair


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
    completed_waves: int = 0
    adaptive_rounds: int = 0
    probe_protocol_hash: str | None = None
    probe_policy_hash: str | None = None
    probe_survivor_pairs: int = 0
    probe_screened_pairs: int = 0
    probe_preprepare_screened_pairs: int = 0

    @property
    def missing_pairs(self) -> int:
        return sum(batch.missing_pairs for batch in self.planned_batches)

    @property
    def nominal_pairs(self) -> int:
        return sum(batch.nominal_pairs for batch in self.planned_batches)
