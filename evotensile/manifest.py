import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .candidate import Candidate, Shape


@dataclass(frozen=True)
class ManifestPair:
    candidate: Candidate
    shape: Shape

    @property
    def key(self) -> tuple[str, str]:
        return self.shape.id, self.candidate.hash


@dataclass(frozen=True)
class ManifestEntry:
    candidate_hash: str
    shape_id: str
    candidate_index: int
    problem_index: int
    solution_index: int | None
    params: dict[str, Any]


def write_manifest(
    path: str | Path,
    pairs: Sequence[ManifestPair],
    *,
    artifact_candidates: Sequence[Candidate],
    artifact_shapes: Sequence[Shape],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate_indices = {candidate.hash: index for index, candidate in enumerate(artifact_candidates)}
    problem_indices = {shape.id: index for index, shape in enumerate(artifact_shapes)}
    if len(candidate_indices) != len(artifact_candidates):
        raise ValueError("artifact candidates must be unique")
    if len(problem_indices) != len(artifact_shapes):
        raise ValueError("artifact shapes must be unique")
    pair_by_key: dict[tuple[str, str], ManifestPair] = {}
    for pair in pairs:
        if pair.candidate.hash not in candidate_indices or pair.shape.id not in problem_indices:
            raise ValueError(f"manifest pair is outside artifact scope: {pair.key}")
        pair_by_key.setdefault(pair.key, pair)
    if len(pair_by_key) != len(pairs):
        raise ValueError("manifest pairs must be unique")

    fieldnames = [
        "candidate_hash",
        "shape_id",
        "candidate_index",
        "problem_index",
        "solution_index",
        "params_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in sorted(
            pair_by_key.values(),
            key=lambda item: (problem_indices[item.shape.id], candidate_indices[item.candidate.hash]),
        ):
            candidate_index = candidate_indices[pair.candidate.hash]
            writer.writerow(
                {
                    "candidate_hash": pair.candidate.hash,
                    "shape_id": pair.shape.id,
                    "candidate_index": candidate_index,
                    "problem_index": problem_indices[pair.shape.id],
                    "solution_index": candidate_index,
                    "params_json": json.dumps(pair.candidate.canonical_params(), sort_keys=True, separators=(",", ":")),
                }
            )
    return path


def read_manifest(path: str | Path) -> list[ManifestEntry]:
    path = Path(path)
    out: list[ManifestEntry] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
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


def manifest_by_shape_candidate(entries: list[ManifestEntry]) -> dict[tuple[str, str], ManifestEntry]:
    return {(entry.shape_id, entry.candidate_hash): entry for entry in entries}
