import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .candidate import Candidate, Shape


@dataclass(frozen=True)
class ManifestEntry:
    candidate_hash: str
    shape_id: str
    candidate_index: int
    problem_index: int
    solution_index: int | None
    params: dict[str, Any]


def write_manifest(path: str | Path, candidates: list[Candidate], shapes: list[Shape]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_hash",
        "shape_id",
        "candidate_index",
        "problem_index",
        "solution_index",
        "params_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for problem_index, shape in enumerate(shapes):
            for candidate_index, candidate in enumerate(candidates):
                writer.writerow(
                    {
                        "candidate_hash": candidate.hash,
                        "shape_id": shape.id,
                        "candidate_index": candidate_index,
                        "problem_index": problem_index,
                        # One YAML group per candidate, so solution order follows group order.
                        "solution_index": candidate_index,
                        "params_json": json.dumps(candidate.canonical_params(), sort_keys=True, separators=(",", ":")),
                    }
                )
    return path


def read_manifest(path: str | Path) -> list[ManifestEntry]:
    path = Path(path)
    out: list[ManifestEntry] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(
                ManifestEntry(
                    candidate_hash=row["candidate_hash"],
                    shape_id=row["shape_id"],
                    candidate_index=int(row.get("candidate_index") or 0),
                    problem_index=int(row.get("problem_index") or 0),
                    solution_index=int(row["solution_index"]) if row.get("solution_index") not in (None, "") else None,
                    params=json.loads(row.get("params_json") or "{}"),
                )
            )
    return out


def manifest_by_problem_solution(entries: list[ManifestEntry]) -> dict[tuple[int, int], ManifestEntry]:
    return {(entry.problem_index, entry.solution_index): entry for entry in entries if entry.solution_index is not None}


def manifest_by_shape_solution(entries: list[ManifestEntry]) -> dict[tuple[str, int], ManifestEntry]:
    return {(entry.shape_id, entry.solution_index): entry for entry in entries if entry.solution_index is not None}


def manifest_by_shape_candidate(entries: list[ManifestEntry]) -> dict[tuple[str, str], ManifestEntry]:
    return {(entry.shape_id, entry.candidate_hash): entry for entry in entries}
