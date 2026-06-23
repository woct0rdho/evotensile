import json
import math
import subprocess
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .candidate import Candidate, Shape
from .database import EvaluationInsert, EvoTensileDB
from .manifest import manifest_by_shape_candidate, read_manifest
from .profile import TargetProfile
from .protocol import BenchmarkProtocol
from .runner import RunResult, _merged_env, run_tensilelite
from .solution_mapping import build_solution_candidate_mapper, find_solution_yamls


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


def _finite_positive(value: Any) -> bool:
    if value is None:
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and f > 0.0


def _sample_from_json(value: dict[str, Any]) -> StructuredSample:
    shape_id = str(value["shape_id"])
    candidate_hash = str(value["candidate_hash"])
    status = str(value.get("status") or "ok")
    time_us = float(value["time_us"]) if value.get("time_us") not in (None, "") else None
    validation = value.get("validation")
    if validation is not None:
        validation = str(validation)
    solution_index = value.get("solution_index")
    return StructuredSample(
        shape_id=shape_id,
        candidate_hash=candidate_hash,
        status=status,
        sample_index=int(value["sample_index"]) if value.get("sample_index") not in (None, "") else None,
        time_us=time_us,
        validation=validation,
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


def _library_dir_from_run(run_dir: Path) -> Path | None:
    patterns = (
        "4_LibraryClient/library/gfx*",
        "1_BenchmarkProblems/**/source/library/gfx*",
    )
    for pattern in patterns:
        candidates = sorted(path for path in run_dir.glob(pattern) if path.is_dir())
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

    negative: list[EvaluationInsert] = []
    for key in sorted(planned_pairs):
        if key in accepted_pairs:
            continue
        entry = by_shape_candidate.get(key)
        negative.append(
            EvaluationInsert(
                shape_id=key[0],
                candidate_hash=key[1],
                run_id=None,
                status="rejected" if entry is not None else "unmapped",
            )
        )
    runnable.sort(key=lambda item: (item.shape_id, item.candidate_hash, item.requested_solution_index))
    return runnable, negative


def _normalized_sample_status(sample: StructuredSample, *, allow_no_check: bool = False) -> str:
    status = sample.status
    if status == "ok":
        if sample.validation is None:
            return "validation_unknown"
        validation = sample.validation.upper()
        if validation == "NO_CHECK":
            if not allow_no_check:
                return "validation_unknown"
        elif validation not in {"PASSED", "OK", "VALID"}:
            return "validation_fail"
        if not _finite_positive(sample.time_us):
            return "invalid"
    return status


def validate_structured_samples(
    samples: list[StructuredSample],
    *,
    runnable_pairs: list[RunnablePair],
    protocol: BenchmarkProtocol,
    runner_returncode: int = 0,
    allow_no_check: bool = False,
) -> list[EvaluationInsert]:
    allowed = {(pair.shape_id, pair.candidate_hash): pair for pair in runnable_pairs}
    grouped: dict[tuple[str, str], list[StructuredSample]] = {key: [] for key in allowed}
    expected_indices = set(range(protocol.num_benchmarks))

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
        status = _normalized_sample_status(sample, allow_no_check=allow_no_check)
        if status != "ok" and sample.sample_index is None:
            grouped[key].append(sample)
            continue
        if sample.sample_index is None:
            raise ValueError(f"structured runner emitted missing sample_index for {key}")
        if sample.sample_index not in expected_indices:
            raise ValueError(
                f"structured runner emitted out-of-range sample_index for {key}: "
                f"expected 0..{protocol.num_benchmarks - 1}, got {sample.sample_index}"
            )
        grouped[key].append(sample)

    inserts: list[EvaluationInsert] = []
    positive_status_seen = False
    for key, pair_samples in grouped.items():
        negative_samples = [
            sample
            for sample in pair_samples
            if _normalized_sample_status(sample, allow_no_check=allow_no_check) != "ok"
        ]
        if negative_samples:
            if len(pair_samples) != len(negative_samples):
                raise ValueError(f"structured runner emitted mixed positive and negative samples for {key}")
        else:
            actual_indices = [sample.sample_index for sample in pair_samples]
            if len(actual_indices) != len(set(actual_indices)):
                raise ValueError(f"structured runner emitted duplicate sample_index for {key}: {actual_indices}")
            if set(actual_indices) != expected_indices:
                raise ValueError(
                    f"structured runner emitted incomplete sample set for {key}: "
                    f"expected {sorted(expected_indices)}, got {sorted(actual_indices)}"
                )
        for sample in sorted(pair_samples, key=lambda item: item.sample_index if item.sample_index is not None else -1):
            status = _normalized_sample_status(sample, allow_no_check=allow_no_check)
            if status == "ok":
                positive_status_seen = True
            inserts.append(
                EvaluationInsert(
                    shape_id=sample.shape_id,
                    candidate_hash=sample.candidate_hash,
                    run_id=None,
                    status=status,
                    time_us=sample.time_us,
                    validation=sample.validation,
                    solution_index=sample.solution_index,
                )
            )

    if runner_returncode != 0 and positive_status_seen:
        raise ValueError(f"structured runner returned {runner_returncode} with positive result rows")
    return inserts


def run_structured_backend(
    *,
    run_dir: Path,
    pairs: list[RunnablePair],
    shapes: list[Shape],
    protocol: BenchmarkProtocol,
    runner_bin: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout_s: float | None = None,
) -> StructuredRunOutput:
    run_id = f"structured_{uuid.uuid4().hex[:12]}"
    stdout_path = run_dir / f"{run_id}.stdout.log"
    stderr_path = run_dir / f"{run_id}.stderr.log"
    pairs_path = run_dir / f"{run_id}.pairs.jsonl"
    results_path = run_dir / f"{run_id}.results.jsonl"
    shape_map = {shape.id: shape for shape in shapes}
    _write_pairs(pairs_path, pairs, shape_map, protocol)
    if runner_bin is None:
        raise ValueError("runner_bin is required for the structured runner")

    start = time.perf_counter()
    library_dir = _library_dir_from_run(run_dir)
    command = [str(runner_bin), "--pairs", str(pairs_path), "--output", str(results_path)]
    if library_dir is not None:
        command.extend(["--library-dir", str(library_dir)])
    timed_out = False
    returncode = 0
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            proc = subprocess.run(
                command,
                text=True,
                stdout=stdout,
                stderr=stderr,
                env=_merged_env(env),
                check=False,
                timeout=timeout_s,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            stderr.write(f"\nStructured runner timed out after {exc.timeout} seconds\n")

    duration_s = time.perf_counter() - start
    samples = read_structured_results(results_path) if results_path.exists() else []
    return StructuredRunOutput(
        returncode=returncode,
        samples=samples,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        results_path=results_path,
        duration_s=duration_s,
        command=command,
        timed_out=timed_out,
    )


def build_then_structured_benchmark(
    yaml_path: str | Path,
    manifest_path: str | Path,
    run_dir: str | Path,
    *,
    shapes: list[Shape],
    candidates: list[Candidate],
    db: EvoTensileDB,
    tensilelite_bin: str | Path,
    compile_threads: int | None,
    target_profile: TargetProfile,
    protocol: BenchmarkProtocol,
    runner_bin: str | Path | None = None,
    env: dict[str, str] | None = None,
    trust_prior_validation: bool = False,
    build_timeout_s: float | None = None,
    runner_timeout_s: float | None = None,
) -> tuple[RunResult, StructuredRunOutput | None, list[EvaluationInsert], list[str]]:
    run_dir = Path(run_dir)
    problem_type_hash = target_profile.problem_type_hash
    benchmark_protocol_hash = target_profile.benchmark_protocol_hash(protocol)
    build_globals = target_profile.global_parameter_items(protocol)
    build_result = run_tensilelite(
        yaml_path,
        run_dir,
        tensilelite_bin=tensilelite_bin,
        db=db,
        build_only=True,
        cpu_threads=compile_threads,
        global_parameters=build_globals,
        env=env,
        timeout_s=build_timeout_s,
    )
    planned_pairs = {(shape.id, candidate.hash) for shape in shapes for candidate in candidates}
    if not build_result.ok and not build_result.output_dir.exists():
        return build_result, None, [], []

    solution_yamls = [str(path) for path in find_solution_yamls([run_dir])]
    runnable, negative = build_runnable_pairs(
        manifest_path=manifest_path,
        solution_yaml_paths=solution_yamls,
        planned_pairs=planned_pairs,
    )
    if not runnable:
        if not build_result.ok:
            return build_result, None, [], []
        for idx, item in enumerate(negative):
            negative[idx] = EvaluationInsert(
                shape_id=item.shape_id,
                candidate_hash=item.candidate_hash,
                run_id=build_result.run_id,
                status=item.status,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
            )
        return build_result, None, negative, []
    for idx, item in enumerate(negative):
        negative[idx] = EvaluationInsert(
            shape_id=item.shape_id,
            candidate_hash=item.candidate_hash,
            run_id=build_result.run_id,
            status=item.status,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
        )

    structured = run_structured_backend(
        run_dir=run_dir,
        pairs=runnable,
        shapes=shapes,
        protocol=protocol,
        runner_bin=runner_bin,
        env=env,
        timeout_s=runner_timeout_s,
    )
    structured_run_id = f"run_{uuid.uuid4().hex[:12]}"
    db.insert_run(
        structured_run_id,
        yaml_path=str(yaml_path),
        output_dir=str(run_dir),
        status="timeout" if structured.timed_out else "ok" if structured.ok else "failed",
        returncode=structured.returncode,
        metadata_json=json.dumps(
            {
                "command": structured.command,
                "results_path": str(structured.results_path),
                "stdout_path": str(structured.stdout_path),
                "stderr_path": str(structured.stderr_path),
                "duration_s": structured.duration_s,
                "timed_out": structured.timed_out,
                "runnable_pairs": len(runnable),
                "negative_pairs": len(negative),
                "solution_yamls": solution_yamls,
            },
            sort_keys=True,
        ),
    )
    try:
        inserts = validate_structured_samples(
            structured.samples,
            runnable_pairs=runnable,
            protocol=protocol,
            runner_returncode=structured.returncode,
            allow_no_check=trust_prior_validation,
        )
    except Exception as exc:
        return build_result, structured, negative, [str(exc)]
    inserts = [
        EvaluationInsert(
            shape_id=item.shape_id,
            candidate_hash=item.candidate_hash,
            run_id=structured_run_id,
            status=item.status,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            time_us=item.time_us,
            validation=item.validation,
            solution_index=item.solution_index,
        )
        for item in inserts
    ]
    return build_result, structured, [*negative, *inserts], []
