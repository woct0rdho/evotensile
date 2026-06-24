from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .candidate import canonicalize
from .manifest import ManifestEntry
from .shapes import Shape
from .tensilelite_keys import (
    DIRECT_SOLUTION_MATCH_KEYS,
    EXACT_KEY,
    KERNEL_NAME_MIN_KEY,
    MATRIX_INSTRUCTION_KEY,
    MI_WAVE_GROUP_KEY,
    MI_WAVE_TILE_KEY,
    PROBLEM_SIZES_KEY,
    SOLUTION_INDEX_KEY,
    SOLUTION_NAME_MIN_KEY,
    SOLUTION_YAML_GLOBS,
    WORK_GROUP_KEY,
)


@dataclass(frozen=True)
class SolutionRecord:
    path: Path
    solution_index: int
    solution_name: str | None
    shape_ids: tuple[str, ...]
    solution: dict[str, Any]


@dataclass
class SolutionCandidateMapper:
    by_shape_solution: dict[tuple[str, int], list[ManifestEntry]]
    by_shape_solution_name: dict[tuple[str, str], list[ManifestEntry]]
    unmatched_solutions: list[SolutionRecord]
    solution_yaml_paths: list[Path]

    def entries_for(
        self,
        *,
        shape_id: str | None,
        solution_index: int | None,
        solution_name: str | None = None,
    ) -> list[ManifestEntry]:
        out: list[ManifestEntry] = []
        if shape_id is not None and solution_index is not None:
            out.extend(self.by_shape_solution.get((shape_id, solution_index), []))
        if not out and shape_id is not None and solution_name:
            out.extend(self.by_shape_solution_name.get((shape_id, solution_name), []))
        return _unique_entries(out)


def _unique_entries(entries: list[ManifestEntry]) -> list[ManifestEntry]:
    seen: set[tuple[str, str]] = set()
    out: list[ManifestEntry] = []
    for entry in entries:
        key = (entry.shape_id, entry.candidate_hash)
        if key not in seen:
            seen.add(key)
            out.append(entry)
    return out


def _value_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if isinstance(expected, (bool, int)) and isinstance(actual, (bool, int)):
            return int(expected) == int(actual)
    return canonicalize(expected) == canonicalize(actual)


def _matrix_instruction_matches(candidate_mi: Any, solution: dict[str, Any]) -> bool:
    if not isinstance(candidate_mi, list) or len(candidate_mi) < 4:
        return False
    if not _value_equal(candidate_mi[:4], solution.get(MATRIX_INSTRUCTION_KEY)):
        return False
    if len(candidate_mi) >= 7 and MI_WAVE_TILE_KEY in solution:
        if not _value_equal([candidate_mi[5], candidate_mi[6]], solution.get(MI_WAVE_TILE_KEY)):
            return False
    if len(candidate_mi) >= 9 and MI_WAVE_GROUP_KEY in solution:
        if not _value_equal([candidate_mi[7], candidate_mi[8]], solution.get(MI_WAVE_GROUP_KEY)):
            return False
    return True


_INACTIVE_STAGGER_DERIVED_KEYS = frozenset({"StaggerUMapping", "StaggerUStride"})


def _inactive_stagger_derived_key(key: str, solution: dict[str, Any], candidate_params: dict[str, Any]) -> bool:
    if key not in _INACTIVE_STAGGER_DERIVED_KEYS:
        return False
    return _value_equal(candidate_params.get("StaggerU"), 0) and _value_equal(solution.get("StaggerU"), 0)


def _inactive_lds_pad_block_size_key(key: str, solution: dict[str, Any], candidate_params: dict[str, Any]) -> bool:
    if key not in {"LdsBlockSizePerPadA", "LdsBlockSizePerPadB"}:
        return False
    suffix = key[-1]
    return _value_equal(candidate_params.get(f"LdsPad{suffix}"), 0) and _value_equal(solution.get(f"LdsPad{suffix}"), 0)


def _candidate_loop_iters(candidate_params: dict[str, Any]) -> int | None:
    matrix_instruction = candidate_params.get(MATRIX_INSTRUCTION_KEY)
    work_group = candidate_params.get(WORK_GROUP_KEY)
    depth = candidate_params.get("DepthU")
    if depth is None:
        return None
    if not isinstance(matrix_instruction, list) or len(matrix_instruction) < 3 or not isinstance(work_group, list):
        return None
    try:
        matrix_inst_k = int(matrix_instruction[2])
        local_split_u = int(work_group[2])
        depth_u = int(depth)
    except (TypeError, ValueError, IndexError):
        return None
    if matrix_inst_k <= 0 or local_split_u <= 0:
        return None
    inner_unroll = max(1, depth_u // matrix_inst_k) if _value_equal(candidate_params.get("ScheduleIterAlg"), 2) else 1
    loop_unroll = depth_u // local_split_u
    loop_unroll //= inner_unroll
    return max(1, loop_unroll // matrix_inst_k)


def _normalized_cluster_local_read_key(key: str, solution: dict[str, Any], candidate_params: dict[str, Any]) -> bool:
    if key not in {"ClusterLocalRead", "PrefetchLocalRead"}:
        return False
    if not _value_equal(candidate_params.get("ClusterLocalRead"), 1) or not _value_equal(
        solution.get("ClusterLocalRead"), 0
    ):
        return False
    if not _value_equal(solution.get("PrefetchLocalRead"), 0):
        return False
    if _value_equal(candidate_params.get("ScheduleIterAlg"), 2):
        return False
    loop_iters = _candidate_loop_iters(candidate_params)
    return loop_iters is not None and int(candidate_params.get("PrefetchLocalRead", 0)) >= loop_iters


def solution_matches_candidate(solution: dict[str, Any], candidate_params: dict[str, Any]) -> bool:
    """Return True if a final TensileLite solution came from a candidate.

    TensileLite assigns derived parameters and may reject or deduplicate solutions
    before writing `*_Final.yaml`. Matching against that final YAML is the stable
    source of truth; input group order is only a fallback/debug aid.
    """
    if MATRIX_INSTRUCTION_KEY in candidate_params:
        if not _matrix_instruction_matches(candidate_params[MATRIX_INSTRUCTION_KEY], solution):
            return False
    for key in sorted(DIRECT_SOLUTION_MATCH_KEYS):
        if key not in candidate_params:
            continue
        if _inactive_stagger_derived_key(key, solution, candidate_params):
            # TensileLite may normalize StaggerUMapping/StaggerUStride even when
            # StaggerU=0 disables staggering, so those inactive fields are not
            # reliable evidence that the final solution came from another input.
            continue
        if _inactive_lds_pad_block_size_key(key, solution, candidate_params):
            # TensileLite normalizes LdsBlockSizePerPad* to 0 when the matching
            # LdsPad* is 0, because the block-size value is inert without padding.
            continue
        if _normalized_cluster_local_read_key(key, solution, candidate_params):
            # TensileLite rewrites ClusterLocalRead and PrefetchLocalRead to 0
            # when PLR already covers the loop iterations outside SIA2.
            continue
        if key not in solution:
            return False
        if not _value_equal(candidate_params[key], solution[key]):
            return False
    return True


def _shape_id_from_exact(exact: Any) -> str | None:
    if not isinstance(exact, list) or len(exact) < 4:
        return None
    try:
        return Shape(int(exact[0]), int(exact[1]), int(exact[2]), int(exact[3])).id
    except (TypeError, ValueError):
        return None


def _shape_ids_from_solution_yaml(data: Any) -> tuple[str, ...]:
    if not isinstance(data, list):
        return ()
    shape_ids: list[str] = []
    for item in data:
        if not isinstance(item, dict) or PROBLEM_SIZES_KEY not in item:
            continue
        for size in item.get(PROBLEM_SIZES_KEY) or []:
            if isinstance(size, dict) and EXACT_KEY in size:
                shape_id = _shape_id_from_exact(size[EXACT_KEY])
                if shape_id is not None:
                    shape_ids.append(shape_id)
    return tuple(dict.fromkeys(shape_ids))


def read_solution_records(path: str | Path) -> list[SolutionRecord]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    shape_ids = _shape_ids_from_solution_yaml(data)
    solutions = [
        item for item in data if isinstance(item, dict) and (SOLUTION_INDEX_KEY in item or KERNEL_NAME_MIN_KEY in item)
    ]
    records: list[SolutionRecord] = []
    for ordinal, solution in enumerate(solutions):
        solution_index = int(solution.get(SOLUTION_INDEX_KEY, ordinal))
        solution_name = solution.get(KERNEL_NAME_MIN_KEY) or solution.get(SOLUTION_NAME_MIN_KEY)
        records.append(
            SolutionRecord(
                path=path,
                solution_index=solution_index,
                solution_name=str(solution_name) if solution_name else None,
                shape_ids=shape_ids,
                solution=solution,
            )
        )
    return records


def find_solution_yamls(paths: list[str | Path]) -> list[Path]:
    found: set[Path] = set()
    for item in paths:
        path = Path(item)
        if path.is_dir():
            roots = [path]
        elif path.suffix in {".yaml", ".yml"}:
            roots = []
            found.add(path)
        else:
            roots = [path.parent]
        for root in roots:
            for pattern in SOLUTION_YAML_GLOBS:
                found.update(p for p in root.glob(pattern) if p.is_file())
    return sorted(found)


def build_solution_candidate_mapper(
    manifest_entries: list[ManifestEntry],
    solution_yaml_paths: Sequence[str | Path],
) -> SolutionCandidateMapper:
    entries_by_candidate: dict[str, list[ManifestEntry]] = {}
    params_by_candidate: dict[str, dict[str, Any]] = {}
    shapes_in_manifest = tuple(dict.fromkeys(entry.shape_id for entry in manifest_entries))
    for entry in manifest_entries:
        entries_by_candidate.setdefault(entry.candidate_hash, []).append(entry)
        params_by_candidate.setdefault(entry.candidate_hash, entry.params)

    by_shape_solution: dict[tuple[str, int], list[ManifestEntry]] = {}
    by_shape_solution_name: dict[tuple[str, str], list[ManifestEntry]] = {}
    unmatched_solutions: list[SolutionRecord] = []
    resolved_paths = [Path(p) for p in solution_yaml_paths]

    for path in resolved_paths:
        for record in read_solution_records(path):
            matched_hashes = [
                candidate_hash
                for candidate_hash, params in params_by_candidate.items()
                if solution_matches_candidate(record.solution, params)
            ]
            if not matched_hashes:
                unmatched_solutions.append(record)
                continue
            shape_ids = record.shape_ids or shapes_in_manifest
            for shape_id in shape_ids:
                for candidate_hash in matched_hashes:
                    for entry in entries_by_candidate[candidate_hash]:
                        if entry.shape_id != shape_id:
                            continue
                        by_shape_solution.setdefault((shape_id, record.solution_index), []).append(entry)
                        if record.solution_name:
                            by_shape_solution_name.setdefault((shape_id, record.solution_name), []).append(entry)

    for key, entries in list(by_shape_solution.items()):
        by_shape_solution[key] = _unique_entries(entries)
    for key, entries in list(by_shape_solution_name.items()):
        by_shape_solution_name[key] = _unique_entries(entries)

    return SolutionCandidateMapper(
        by_shape_solution=by_shape_solution,
        by_shape_solution_name=by_shape_solution_name,
        unmatched_solutions=unmatched_solutions,
        solution_yaml_paths=resolved_paths,
    )
