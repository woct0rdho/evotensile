#!/usr/bin/env python3

import argparse
import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from evotensile.candidate import Candidate
from evotensile.solution_mapping import solution_matches_candidate

DEFAULT_EXPORT_DIR = Path("out/grid100_full_20260618_hybrid_best_export")
DEFAULT_RETIME_DIR = Path("out/grid100_full_20260618_top4_retime")
DEFAULT_LOGIC_DIR = (
    Path.home()
    / "rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/GridBased"
)
REFERENCE_SCHEMA_FILES = (
    "gfx1151_Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml",
    "gfx1151_Cijk_Alik_Bljk_HHS_BH_Bias_AuxH_HAS_SAV_UserArgs.yaml",
    "gfx1151_Cijk_Alik_Bljk_BBS_BH_Bias_HAS_SAV_UserArgs.yaml",
    "gfx1151_Cijk_Alik_Bljk_BBS_BH_Bias_AuxB_HAS_SAV_UserArgs.yaml",
)


@dataclass(frozen=True)
class Variant:
    name: str
    filename: str
    solution_name_prefix: str


VARIANTS: dict[str, Variant] = {
    "hhs": Variant(
        name="hhs",
        filename="gfx1151_Cijk_Ailk_Bjlk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml",
        # The checked-in gfx1151 HHS and AuxH files already share AuxH-named solutions.
        solution_name_prefix="Cijk_Ailk_Bjlk_HHS_BH_Bias_AuxH_HAS_SAV_UserArgs",
    ),
    "hhs_auxh": Variant(
        name="hhs_auxh",
        filename="gfx1151_Cijk_Ailk_Bjlk_HHS_BH_Bias_AuxH_HAS_SAV_UserArgs.yaml",
        solution_name_prefix="Cijk_Ailk_Bjlk_HHS_BH_Bias_AuxH_HAS_SAV_UserArgs",
    ),
    "bbs": Variant(
        name="bbs",
        filename="gfx1151_Cijk_Ailk_Bjlk_BBS_BH_Bias_HAS_SAV_UserArgs.yaml",
        solution_name_prefix="Cijk_Ailk_Bjlk_BBS_BH_Bias_AuxB_HAS_SAV_UserArgs",
    ),
    "bbs_auxb": Variant(
        name="bbs_auxb",
        filename="gfx1151_Cijk_Ailk_Bjlk_BBS_BH_Bias_AuxB_HAS_SAV_UserArgs.yaml",
        solution_name_prefix="Cijk_Ailk_Bjlk_BBS_BH_Bias_AuxB_HAS_SAV_UserArgs",
    ),
}

NAME_KEYS = ("BaseName", "CustomKernelName", "KernelName", "KernelNameMin", "SolutionName", "SolutionNameMin")


@dataclass(frozen=True)
class Winner:
    shape_id: str
    candidate_hash: str
    candidate_json_path: Path
    selected_gflops: float
    logic_solution_index: int | None
    logic_solution_name: str | None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _with_problem_header_bool_style(text: str) -> str:
    # Nearby checked-in NT logic uses Python-style bools only in the top-level
    # ProblemType Transpose fields; solution dictionaries use normal YAML bools.
    return text.replace("  TransposeA: false\n", "  TransposeA: False\n").replace(
        "  TransposeB: true\n", "  TransposeB: True\n"
    )


def _write_yaml(path: Path, data: Any) -> None:
    # Match TensileLite merge/update tools: compact simple lists/dicts such as
    # [Device 1536] and {MinimumRequiredVersion: 5.0.0}, with stable key order.
    text = yaml.safe_dump(data, default_flow_style=None, sort_keys=False)
    path.write_text(_with_problem_header_bool_style(text), encoding="utf-8")


def _resolve_export_path(value: str, *, export_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = export_dir / path.name
    if candidate.exists():
        return candidate
    return path


def _parse_float(value: str) -> float:
    if value == "":
        return 0.0
    return float(value)


def _parse_int(value: str) -> int | None:
    if value == "":
        return None
    return int(value)


def _load_winners(export_dir: Path) -> list[Winner]:
    rows = _read_csv(export_dir / "winners.csv")
    winners: list[Winner] = []
    for row in rows:
        candidate_hash = row.get("selected_candidate_hash") or row.get("candidate_hash")
        if not candidate_hash:
            raise ValueError(f"winner row has no candidate hash: {row}")
        selected_gflops = row.get("selected_gflops") or row.get("median_gflops") or row.get("evotensile_median_gflops")
        if selected_gflops is None:
            raise ValueError(f"winner row has no selected throughput: {row}")
        winners.append(
            Winner(
                shape_id=row["shape_id"],
                candidate_hash=candidate_hash,
                candidate_json_path=_resolve_export_path(row["candidate_json_path"], export_dir=export_dir),
                selected_gflops=_parse_float(selected_gflops),
                logic_solution_index=_parse_int(row.get("logic_solution_index", "")),
                logic_solution_name=row.get("logic_solution_name") or None,
            )
        )
    return winners


def _load_candidate(path: Path) -> Candidate:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Candidate(
        params=payload["params"],
        source=payload.get("source", "hybrid_export"),
        parent_hashes=tuple(payload.get("parent_hashes", ())),
    )


def _candidate_params_by_hash(winners: list[Winner]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for winner in winners:
        out.setdefault(winner.candidate_hash, _load_candidate(winner.candidate_json_path).canonical_params())
    return out


def _solution_key(solution: dict[str, Any]) -> str:
    for key in ("SolutionNameMin", "KernelNameMin", "BaseName"):
        value = solution.get(key)
        if value:
            return str(value)
    return json.dumps(solution, sort_keys=True, default=str)


def _solution_records_from_logic(path: Path) -> list[dict[str, Any]]:
    data = _load_yaml(path)
    if not isinstance(data, list) or len(data) < 6 or not isinstance(data[5], list):
        raise ValueError(f"unsupported logic YAML layout: {path}")
    return [solution for solution in data[5] if isinstance(solution, dict)]


def _solution_records_from_final_yaml(path: Path) -> list[dict[str, Any]]:
    data = _load_yaml(path)
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict) and "SolutionIndex" in item]


def _collect_retime_solutions(retime_dir: Path) -> list[dict[str, Any]]:
    solutions: dict[str, dict[str, Any]] = {}
    for path in sorted(retime_dir.glob("group_*/batch_*/run_*/3_LibraryLogic/gfx1151_*.yaml")):
        for solution in _solution_records_from_logic(path):
            solutions.setdefault(_solution_key(solution), solution)
    for path in sorted(retime_dir.glob("group_*/batch_*/run_*/1_BenchmarkProblems/**/Data/00_Final.yaml")):
        for solution in _solution_records_from_final_yaml(path):
            # Data/00_Final.yaml is often the only place that keeps all accepted
            # per-candidate final solutions from a group; 3_LibraryLogic may only
            # contain the group winner for each exact size.
            solutions.setdefault(_solution_key(solution), solution)
    return list(solutions.values())


def _source_solutions_by_index(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for solution in _solution_records_from_logic(path):
        if "SolutionIndex" in solution:
            out[int(solution["SolutionIndex"])] = solution
    return out


def _find_matching_solution(candidate_params: dict[str, Any], solutions: list[dict[str, Any]]) -> dict[str, Any] | None:
    for solution in solutions:
        if solution_matches_candidate(solution, candidate_params):
            return solution
    return None


def _build_base_solutions(
    *,
    winners: list[Winner],
    candidate_params: dict[str, dict[str, Any]],
    retime_solutions: list[dict[str, Any]],
    source_hhs_solutions: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_candidate: dict[str, dict[str, Any]] = {}
    for candidate_hash, params in candidate_params.items():
        solution = _find_matching_solution(params, retime_solutions)
        if solution is not None:
            by_candidate[candidate_hash] = solution

    for winner in winners:
        if winner.candidate_hash in by_candidate:
            continue
        if winner.logic_solution_index is not None and winner.logic_solution_index in source_hhs_solutions:
            by_candidate[winner.candidate_hash] = source_hhs_solutions[winner.logic_solution_index]
            continue
        solution = _find_matching_solution(candidate_params[winner.candidate_hash], list(source_hhs_solutions.values()))
        if solution is not None:
            by_candidate[winner.candidate_hash] = solution
            continue
        raise ValueError(f"could not find a full solution dictionary for {winner.shape_id} {winner.candidate_hash}")
    return by_candidate


def _retarget_name(value: str, name_prefix: str) -> str:
    marker = value.find("_MT")
    if marker >= 0:
        return f"{name_prefix}{value[marker:]}"
    marker = value.find("UserArgs")
    if marker >= 0:
        return f"{name_prefix}{value[marker + len('UserArgs') :]}"
    return value


def _normalize_solution_scalars(solution: dict[str, Any]) -> None:
    solution.pop("ProblemType", None)
    for key in ("ExpandPointerSwap", "SourceSwap"):
        if key in solution and isinstance(solution[key], int):
            solution[key] = bool(solution[key])
    if "GlobalReadPerMfma" in solution and isinstance(solution["GlobalReadPerMfma"], int):
        solution["GlobalReadPerMfma"] = float(solution["GlobalReadPerMfma"])
    code_object_version = solution.get("CodeObjectVersion")
    if isinstance(code_object_version, str) and code_object_version.isdigit():
        solution["CodeObjectVersion"] = int(code_object_version)


def _retarget_solution(
    solution: dict[str, Any],
    *,
    new_index: int,
    name_prefix: str,
    solution_key_order: list[str],
    use_e: bool,
) -> dict[str, Any]:
    out = {key: copy.deepcopy(solution[key]) for key in solution_key_order if key in solution}
    out["SolutionIndex"] = new_index
    if use_e:
        # TensileLite rejects library logic with UseE and grouped load/store.
        out["GroupLoadStore"] = False
    for key in NAME_KEYS:
        value = out.get(key)
        if isinstance(value, str) and value:
            out[key] = _retarget_name(value, name_prefix)
    _normalize_solution_scalars(out)
    return out


def _exact_list_from_shape_id(shape_id: str) -> list[int]:
    parts = shape_id.split("_")
    if len(parts) != 4:
        raise ValueError(f"invalid shape id: {shape_id}")
    return [int(parts[0][1:]), int(parts[1][1:]), int(parts[2][1:]), int(parts[3][1:])]


def _update_logic_data(
    *,
    template_data: list[Any],
    variant: Variant,
    winners: list[Winner],
    base_solutions: dict[str, dict[str, Any]],
    solution_key_order: list[str],
) -> tuple[list[Any], int, int]:
    data = copy.deepcopy(template_data)
    solution_index_by_hash: dict[str, int] = {}
    new_solutions: list[dict[str, Any]] = []

    for candidate_hash in dict.fromkeys(winner.candidate_hash for winner in winners):
        solution_index_by_hash[candidate_hash] = len(new_solutions)
        new_solutions.append(
            _retarget_solution(
                base_solutions[candidate_hash],
                new_index=len(new_solutions),
                name_prefix=variant.solution_name_prefix,
                solution_key_order=solution_key_order,
                use_e=bool(data[4].get("UseE")),
            )
        )

    exact_rows = []
    for winner in sorted(winners, key=lambda item: _exact_list_from_shape_id(item.shape_id)):
        exact_rows.append(
            [
                _exact_list_from_shape_id(winner.shape_id),
                [solution_index_by_hash[winner.candidate_hash], winner.selected_gflops],
            ]
        )

    data[5] = new_solutions
    data[7] = exact_rows
    return data, len(new_solutions), len(exact_rows)


def _reference_solution_key_order(logic_dir: Path) -> list[str]:
    key_order: list[str] = []
    for path in [logic_dir / name for name in REFERENCE_SCHEMA_FILES]:
        if not path.exists():
            continue
        for solution in _solution_records_from_logic(path):
            for key in solution:
                if key not in key_order:
                    key_order.append(key)
    if "SolutionIndex" not in key_order:
        key_order.append("SolutionIndex")
    return key_order


def update_logic_files(
    *,
    export_dir: Path,
    retime_dir: Path,
    logic_dir: Path,
    variant_names: list[str],
    dry_run: bool,
) -> dict[str, Any]:
    export_dir = export_dir.resolve()
    retime_dir = retime_dir.resolve()
    logic_dir = logic_dir.resolve()
    unknown = sorted(set(variant_names) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {', '.join(unknown)}")

    winners = _load_winners(export_dir)
    candidate_params = _candidate_params_by_hash(winners)
    retime_solutions = _collect_retime_solutions(retime_dir)
    solution_key_order = _reference_solution_key_order(logic_dir)
    source_hhs_path = logic_dir / VARIANTS["hhs"].filename
    source_hhs_solutions = _source_solutions_by_index(source_hhs_path)
    base_solutions = _build_base_solutions(
        winners=winners,
        candidate_params=candidate_params,
        retime_solutions=retime_solutions,
        source_hhs_solutions=source_hhs_solutions,
    )

    files: dict[str, Any] = {}
    for variant_name in variant_names:
        variant = VARIANTS[variant_name]
        path = logic_dir / variant.filename
        template_data = _load_yaml(path)
        if not isinstance(template_data, list) or len(template_data) < 8:
            raise ValueError(f"unsupported logic YAML layout: {path}")
        updated, solution_count, exact_count = _update_logic_data(
            template_data=template_data,
            variant=variant,
            winners=winners,
            base_solutions=base_solutions,
            solution_key_order=solution_key_order,
        )
        if not dry_run:
            _write_yaml(path, updated)
        files[variant_name] = {
            "path": str(path),
            "solution_count": solution_count,
            "exact_mapping_count": exact_count,
            "written": not dry_run,
        }

    return {
        "export_dir": str(export_dir),
        "retime_dir": str(retime_dir),
        "logic_dir": str(logic_dir),
        "shape_count": len(winners),
        "candidate_count": len(candidate_params),
        "retime_solution_pool_count": len(retime_solutions),
        "reference_solution_key_count": len(solution_key_order),
        "files": files,
        "dry_run": dry_run,
        "note": "No TensileLite run or hipBLASLt rebuild was performed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--retime-dir", type=Path, default=DEFAULT_RETIME_DIR)
    parser.add_argument("--logic-dir", type=Path, default=DEFAULT_LOGIC_DIR)
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS), default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    variant_names = args.variant or ["hhs", "hhs_auxh", "bbs", "bbs_auxb"]
    result = update_logic_files(
        export_dir=args.export_dir,
        retime_dir=args.retime_dir,
        logic_dir=args.logic_dir,
        variant_names=variant_names,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
