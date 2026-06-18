import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .database import EvoTensileDB
from .manifest import manifest_by_problem_solution, read_manifest
from .parser import evaluation_status, find_result_csvs, parse_tensilelite_csv
from .solution_mapping import build_solution_candidate_mapper, find_solution_yamls


@dataclass(frozen=True)
class IngestResult:
    inserted: int
    unmapped: int
    status_counts: dict[str, int]
    solution_yamls: list[Path]
    unmatched_final_solutions: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def csv_paths(items: list[str | Path], *, include_logs: bool = False) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            paths.extend(find_result_csvs(p, include_logs=include_logs))
        else:
            paths.append(p)
    return sorted(set(paths))


def ingest_results(
    *,
    db: EvoTensileDB,
    paths: list[str | Path],
    manifest_path: str | Path,
    version_name: str,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    run_id: str | None = None,
    include_logs: bool = False,
    solutions_yaml: list[str | Path] | None = None,
    allow_manifest_order_fallback: bool = False,
    allow_unknown_validation: bool = False,
) -> IngestResult:
    manifest_entries = read_manifest(manifest_path)
    solution_yamls = [Path(p) for p in solutions_yaml] if solutions_yaml else find_solution_yamls(paths)

    mapper = None
    by_problem_solution = {}
    errors: list[str] = []
    if solution_yamls:
        mapper = build_solution_candidate_mapper(manifest_entries, solution_yamls)
    elif allow_manifest_order_fallback:
        by_problem_solution = manifest_by_problem_solution(manifest_entries)
    else:
        errors.append("no TensileLite final solution YAML found; pass --solutions-yaml or ingest a run directory")
        return IngestResult(
            inserted=0,
            unmapped=0,
            status_counts={},
            solution_yamls=[],
            errors=errors,
        )

    inserted = 0
    unmapped = 0
    status_counts: dict[str, int] = {}
    for path in csv_paths(paths, include_logs=include_logs):
        for row in parse_tensilelite_csv(path):
            if mapper is not None:
                entries = mapper.entries_for(
                    shape_id=row.shape_id,
                    solution_index=row.solution_index,
                    solution_name=row.solution_name,
                )
            else:
                entry = None
                if row.problem_index is not None and row.solution_index is not None:
                    entry = by_problem_solution.get((row.problem_index, row.solution_index))
                entries = [entry] if entry is not None else []
            if not entries:
                unmapped += 1
                continue
            status = evaluation_status(row, require_validation=not allow_unknown_validation)
            for entry in entries:
                status_counts[status] = status_counts.get(status, 0) + 1
                db.insert_evaluation(
                    shape_id=entry.shape_id,
                    candidate_hash=entry.candidate_hash,
                    run_id=run_id,
                    status=status,
                    version_name=version_name,
                    problem_type_hash=problem_type_hash,
                    benchmark_protocol_hash=benchmark_protocol_hash,
                    time_us=row.time_us,
                    gflops=row.gflops,
                    validation=row.validation,
                    solution_index=row.solution_index,
                    raw_csv_row=json.dumps(row.raw, sort_keys=True),
                )
                inserted += 1

    return IngestResult(
        inserted=inserted,
        unmapped=unmapped,
        status_counts=status_counts,
        solution_yamls=solution_yamls,
        unmatched_final_solutions=len(mapper.unmatched_solutions) if mapper is not None else 0,
    )


def print_ingest_result(result: IngestResult, *, db_path: str | Path, manifest_path: str | Path) -> None:
    print(f"db: {db_path}")
    print(f"manifest: {manifest_path}")
    print(f"solution_yamls: {len(result.solution_yamls)}")
    if result.unmatched_final_solutions:
        print(f"unmatched final solutions: {result.unmatched_final_solutions}")
    print(f"inserted evaluations: {result.inserted}")
    print(f"unmapped rows: {result.unmapped}")
    print("status counts:")
    for status in sorted(result.status_counts):
        print(f"  {status}: {result.status_counts[status]}")
    for error in result.errors:
        print(f"error: {error}", file=sys.stderr)
