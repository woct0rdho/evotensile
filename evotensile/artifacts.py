import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .database import CandidateArtifactInsert, EvoTensileDB
from .structured_runner import RunnablePair


@dataclass(frozen=True)
class CandidateArtifact:
    runnable_pair: RunnablePair
    build_run_id: str
    build_output_dir: Path
    library_dir: Path
    solution_yaml_paths: tuple[Path, ...]
    manifest_path: Path | None
    code_object_identity: str


def library_content_identity(library_dir: str | Path) -> str:
    root = Path(library_dir).resolve(strict=True)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"artifact library directory is empty: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"artifact_{digest.hexdigest()[:24]}"


def register_candidate_artifacts(
    db: EvoTensileDB,
    *,
    problem_type_hash: str,
    runnable_pairs: list[RunnablePair],
    build_run_id: str,
    build_output_dir: str | Path,
    library_dir: str | Path,
    solution_yaml_paths: Sequence[str | Path],
    manifest_path: str | Path | None,
) -> list[CandidateArtifactInsert]:
    if not runnable_pairs:
        return []
    build_root = Path(build_output_dir).resolve(strict=True)
    library_root = Path(library_dir).resolve(strict=True)
    solution_paths = tuple(sorted({Path(path).resolve(strict=True) for path in solution_yaml_paths}))
    if not solution_paths:
        raise ValueError("candidate artifact registration requires generated solution YAML")
    resolved_manifest = Path(manifest_path).resolve(strict=True) if manifest_path is not None else None
    identity = library_content_identity(library_root)
    encoded_solution_paths = json.dumps([str(path) for path in solution_paths], sort_keys=True)
    inserts = [
        CandidateArtifactInsert(
            problem_type_hash=problem_type_hash,
            shape_id=pair.shape_id,
            candidate_hash=pair.candidate_hash,
            problem_index=pair.problem_index,
            requested_solution_index=pair.requested_solution_index,
            library_solution_index=pair.library_solution_index,
            manifest_solution_index=pair.manifest_solution_index,
            build_run_id=build_run_id,
            build_output_dir=str(build_root),
            library_dir=str(library_root),
            solution_yaml_paths_json=encoded_solution_paths,
            manifest_path=str(resolved_manifest) if resolved_manifest is not None else None,
            code_object_identity=identity,
        )
        for pair in runnable_pairs
    ]
    db.insert_candidate_artifacts(inserts)
    return inserts


def load_candidate_artifacts(
    db: EvoTensileDB,
    *,
    problem_type_hash: str,
    shape_ids: list[str] | None = None,
    candidate_hashes: list[str] | None = None,
) -> dict[tuple[str, str], CandidateArtifact]:
    records = db.candidate_artifact_records(
        problem_type_hash=problem_type_hash,
        shape_ids=shape_ids,
        candidate_hashes=candidate_hashes,
    )
    found: dict[tuple[str, str], CandidateArtifact] = {}
    identity_cache: dict[Path, str | None] = {}
    for record in records:
        key = (record.shape_id, record.candidate_hash)
        if key in found:
            continue
        build_output_dir = Path(record.build_output_dir)
        library_dir = Path(record.library_dir)
        try:
            raw_solution_paths = json.loads(record.solution_yaml_paths_json)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw_solution_paths, list) or not all(isinstance(path, str) for path in raw_solution_paths):
            continue
        solution_yaml_paths = tuple(Path(path) for path in raw_solution_paths)
        manifest_path = Path(record.manifest_path) if record.manifest_path is not None else None
        if not build_output_dir.is_dir() or not library_dir.is_dir():
            continue
        if not solution_yaml_paths or any(not path.is_file() for path in solution_yaml_paths):
            continue
        if manifest_path is not None and not manifest_path.is_file():
            continue
        if library_dir not in identity_cache:
            try:
                identity_cache[library_dir] = library_content_identity(library_dir)
            except (OSError, ValueError):
                identity_cache[library_dir] = None
        if identity_cache[library_dir] != record.code_object_identity:
            continue
        found[key] = CandidateArtifact(
            runnable_pair=RunnablePair(
                shape_id=record.shape_id,
                candidate_hash=record.candidate_hash,
                problem_index=record.problem_index,
                requested_solution_index=record.requested_solution_index,
                library_solution_index=record.library_solution_index,
                manifest_solution_index=record.manifest_solution_index,
            ),
            build_run_id=record.build_run_id,
            build_output_dir=build_output_dir,
            library_dir=library_dir,
            solution_yaml_paths=solution_yaml_paths,
            manifest_path=manifest_path,
            code_object_identity=record.code_object_identity,
        )
    return found
