import json
import math
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .candidate import Shape
from .database import EvaluationInsert, ValidationInsert, validation_token
from .manifest import manifest_by_shape_candidate, read_manifest
from .protocol import BenchmarkProtocol
from .runner import _merged_env
from .solution_mapping import build_solution_candidate_mapper
from .subprocess_utils import run_logged_process

RunMode = Literal["validate", "benchmark"]


@dataclass(frozen=True)
class RunnablePair:
    shape_id: str
    candidate_hash: str
    problem_index: int
    requested_solution_index: int
    library_solution_index: int
    manifest_solution_index: int | None


@dataclass(frozen=True)
class StructuredSample:
    shape_id: str
    candidate_hash: str
    status: str
    sample_index: int | None = None
    time_us: float | None = None
    validation: str | None = None
    solution_index: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuredRunOutput:
    mode: RunMode
    run_id: str
    returncode: int
    samples: list[StructuredSample]
    stdout_path: Path
    stderr_path: Path
    results_path: Path
    duration_s: float
    command: list[str]
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@dataclass(frozen=True)
class ValidationOutcome:
    passed_pairs: list[RunnablePair]
    validations: list[ValidationInsert]
    negative_evaluations: list[EvaluationInsert]


def _finite_positive(value: Any) -> bool:
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _sample_from_json(value: dict[str, Any]) -> StructuredSample:
    validation = value.get("validation")
    detail = value.get("validation_detail")
    if detail not in (None, ""):
        validation = detail
    solution_index = value.get("solution_index")
    return StructuredSample(
        shape_id=str(value["shape_id"]),
        candidate_hash=str(value["candidate_hash"]),
        status=str(value.get("status") or "ok"),
        sample_index=int(value["sample_index"]) if value.get("sample_index") not in (None, "") else None,
        time_us=float(value["time_us"]) if value.get("time_us") not in (None, "") else None,
        validation=str(validation) if validation is not None else None,
        solution_index=int(solution_index) if solution_index not in (None, "") else None,
        raw=value,
    )


def read_structured_results(path: str | Path) -> list[StructuredSample]:
    path = Path(path)
    samples: list[StructuredSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            samples.append(_sample_from_json(value))
    return samples


def _write_pairs(path: Path, pairs: list[RunnablePair], shapes: dict[str, Shape], protocol: BenchmarkProtocol) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            shape = shapes[pair.shape_id]
            handle.write(
                json.dumps(
                    {
                        "shape_id": pair.shape_id,
                        "candidate_hash": pair.candidate_hash,
                        "m": shape.m,
                        "n": shape.n,
                        "batch": shape.batch,
                        "k": shape.k,
                        "problem_index": pair.problem_index,
                        "requested_solution_index": pair.requested_solution_index,
                        "library_solution_index": pair.library_solution_index,
                        "manifest_solution_index": pair.manifest_solution_index,
                        "num_warmups": protocol.num_warmups,
                        "num_benchmarks": protocol.num_benchmarks,
                        "enqueues_per_sync": protocol.enqueues_per_sync,
                        "syncs_per_benchmark": protocol.syncs_per_benchmark,
                        "num_elements_to_validate": protocol.num_elements_to_validate,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def library_dir_from_build(build_dir: Path) -> Path | None:
    patterns = (
        "4_LibraryClient/library/gfx*",
        "1_BenchmarkProblems/**/source/library/gfx*",
        "**/source/library/gfx*",
    )
    for pattern in patterns:
        candidates = sorted(path for path in build_dir.glob(pattern) if path.is_dir())
        if candidates:
            return candidates[0]
    return None


def build_runnable_pairs(
    *,
    manifest_path: str | Path,
    solution_yaml_paths: Sequence[str | Path],
    planned_pairs: set[tuple[str, str]],
) -> tuple[list[RunnablePair], list[EvaluationInsert]]:
    entries = read_manifest(manifest_path)
    by_shape_candidate = manifest_by_shape_candidate(entries)
    mapper = build_solution_candidate_mapper(entries, solution_yaml_paths)
    runnable: list[RunnablePair] = []
    accepted_pairs: set[tuple[str, str]] = set()

    for (shape_id, solution_index), mapped_entries in sorted(mapper.by_shape_solution.items()):
        for entry in mapped_entries:
            key = (entry.shape_id, entry.candidate_hash)
            if key not in planned_pairs or key in accepted_pairs:
                continue
            runnable.append(
                RunnablePair(
                    shape_id=entry.shape_id,
                    candidate_hash=entry.candidate_hash,
                    problem_index=entry.problem_index,
                    requested_solution_index=solution_index,
                    library_solution_index=solution_index,
                    manifest_solution_index=entry.solution_index,
                )
            )
            accepted_pairs.add(key)

    negative = [
        EvaluationInsert(
            shape_id=shape_id,
            candidate_hash=candidate_hash,
            run_id=None,
            status="rejected" if (shape_id, candidate_hash) in by_shape_candidate else "unmapped",
        )
        for shape_id, candidate_hash in sorted(planned_pairs - accepted_pairs)
    ]
    runnable.sort(key=lambda item: (item.shape_id, item.candidate_hash, item.requested_solution_index))
    return runnable, negative


def _group_samples(
    samples: list[StructuredSample], runnable_pairs: list[RunnablePair]
) -> tuple[dict[tuple[str, str], RunnablePair], dict[tuple[str, str], list[StructuredSample]]]:
    allowed = {(pair.shape_id, pair.candidate_hash): pair for pair in runnable_pairs}
    grouped: dict[tuple[str, str], list[StructuredSample]] = {key: [] for key in allowed}
    for sample in samples:
        key = (sample.shape_id, sample.candidate_hash)
        pair = allowed.get(key)
        if pair is None:
            raise ValueError(f"structured runner emitted unexpected pair {key}")
        if sample.solution_index != pair.library_solution_index:
            raise ValueError(
                "structured runner emitted wrong solution_index for "
                f"{key}: expected {pair.library_solution_index}, got {sample.solution_index}"
            )
        grouped[key].append(sample)
    return allowed, grouped


def validate_validation_samples(
    samples: list[StructuredSample],
    *,
    runnable_pairs: list[RunnablePair],
    problem_type_hash: str,
    validation_protocol_hash: str,
    benchmark_protocol_hash: str,
    run_id: str,
    runner_returncode: int = 0,
) -> ValidationOutcome:
    allowed, grouped = _group_samples(samples, runnable_pairs)
    passed_pairs: list[RunnablePair] = []
    validations: list[ValidationInsert] = []
    negative: list[EvaluationInsert] = []
    positive_seen = False

    for key, pair_samples in grouped.items():
        if len(pair_samples) != 1:
            raise ValueError(f"validation runner emitted {len(pair_samples)} rows for {key}; expected 1")
        sample = pair_samples[0]
        if sample.time_us is not None:
            raise ValueError(f"validation runner emitted timing for {key}")
        token = validation_token(sample.validation)
        passed = sample.status == "ok" and token in {"PASSED", "OK", "VALID"}
        validations.append(
            ValidationInsert(
                shape_id=sample.shape_id,
                candidate_hash=sample.candidate_hash,
                run_id=run_id,
                status="passed" if passed else "failed",
                problem_type_hash=problem_type_hash,
                validation_protocol_hash=validation_protocol_hash,
                detail=sample.validation,
                solution_index=sample.solution_index,
            )
        )
        if passed:
            positive_seen = True
            passed_pairs.append(allowed[key])
            continue
        status = sample.status if sample.status not in {"ok", "invalid"} else "validation_fail"
        negative.append(
            EvaluationInsert(
                shape_id=sample.shape_id,
                candidate_hash=sample.candidate_hash,
                run_id=run_id,
                status=status,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                validation=sample.validation,
                solution_index=sample.solution_index,
            )
        )

    if runner_returncode != 0 and positive_seen:
        raise ValueError(f"validation runner returned {runner_returncode} with positive result rows")
    return ValidationOutcome(passed_pairs=passed_pairs, validations=validations, negative_evaluations=negative)


def validate_benchmark_samples(
    samples: list[StructuredSample],
    *,
    runnable_pairs: list[RunnablePair],
    protocol: BenchmarkProtocol,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    run_id: str,
    runner_returncode: int = 0,
) -> list[EvaluationInsert]:
    _, grouped = _group_samples(samples, runnable_pairs)
    expected_indices = set(range(protocol.num_benchmarks))
    inserts: list[EvaluationInsert] = []
    positive_seen = False

    for key, pair_samples in grouped.items():
        negative = [sample for sample in pair_samples if sample.status != "ok"]
        if negative:
            if len(negative) != len(pair_samples):
                raise ValueError(f"benchmark runner emitted mixed positive and negative rows for {key}")
            sample = negative[0]
            inserts.append(
                EvaluationInsert(
                    shape_id=sample.shape_id,
                    candidate_hash=sample.candidate_hash,
                    run_id=run_id,
                    status=sample.status,
                    problem_type_hash=problem_type_hash,
                    benchmark_protocol_hash=benchmark_protocol_hash,
                    validation=sample.validation,
                    solution_index=sample.solution_index,
                )
            )
            continue

        indices = [sample.sample_index for sample in pair_samples]
        if any(index is None for index in indices):
            raise ValueError(f"benchmark runner emitted missing sample_index for {key}")
        if len(indices) != len(set(indices)):
            raise ValueError(f"benchmark runner emitted duplicate sample_index for {key}: {indices}")
        if set(indices) != expected_indices:
            raise ValueError(
                f"benchmark runner emitted incomplete sample set for {key}: "
                f"expected {sorted(expected_indices)}, got {sorted(indices)}"
            )
        for sample in sorted(pair_samples, key=lambda item: item.sample_index or 0):
            if validation_token(sample.validation) != "NO_CHECK":
                raise ValueError(f"benchmark runner performed validation for {key}")
            if not _finite_positive(sample.time_us):
                raise ValueError(f"benchmark runner emitted invalid time for {key}: {sample.time_us}")
            positive_seen = True
            inserts.append(
                EvaluationInsert(
                    shape_id=sample.shape_id,
                    candidate_hash=sample.candidate_hash,
                    run_id=run_id,
                    status="ok",
                    problem_type_hash=problem_type_hash,
                    benchmark_protocol_hash=benchmark_protocol_hash,
                    time_us=sample.time_us,
                    validation="PASSED prior_validation",
                    solution_index=sample.solution_index,
                )
            )

    if runner_returncode != 0 and positive_seen:
        raise ValueError(f"benchmark runner returned {runner_returncode} with positive result rows")
    return inserts


def run_structured_phase(
    *,
    mode: RunMode,
    run_dir: Path,
    pairs: list[RunnablePair],
    shapes: list[Shape],
    protocol: BenchmarkProtocol,
    runner_bin: str | Path,
    library_dir: str | Path,
    env: dict[str, str] | None = None,
    timeout_s: float | None = None,
) -> StructuredRunOutput:
    if mode == "validate":
        if protocol.num_elements_to_validate == 0:
            raise ValueError("validate mode requires correctness verification")
    elif protocol.num_elements_to_validate != 0:
        raise ValueError("benchmark mode requires num_elements_to_validate=0")

    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{mode}_{uuid.uuid4().hex[:12]}"
    stdout_path = run_dir / f"{run_id}.stdout.log"
    stderr_path = run_dir / f"{run_id}.stderr.log"
    pairs_path = run_dir / f"{run_id}.pairs.jsonl"
    results_path = run_dir / f"{run_id}.results.jsonl"
    _write_pairs(pairs_path, pairs, {shape.id: shape for shape in shapes}, protocol)
    command = [
        str(runner_bin),
        "--mode",
        mode,
        "--pairs",
        str(pairs_path),
        "--output",
        str(results_path),
        "--validation-backend",
        protocol.validation_backend,
        "--library-dir",
        str(library_dir),
    ]

    start = time.perf_counter()
    timed_out = False
    returncode = 0
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        returncode, timed_out = run_logged_process(
            command,
            stdout=stdout,
            stderr=stderr,
            env=_merged_env(env),
            timeout_s=timeout_s,
        )
        if timed_out:
            stderr.write(f"\nStructured {mode} phase timed out after {timeout_s} seconds\n")

    return StructuredRunOutput(
        mode=mode,
        run_id=run_id,
        returncode=returncode,
        samples=read_structured_results(results_path) if results_path.exists() else [],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        results_path=results_path,
        duration_s=time.perf_counter() - start,
        command=command,
        timed_out=timed_out,
    )
