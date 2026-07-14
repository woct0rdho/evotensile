#!/usr/bin/env python3

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from evotensile.artifacts import load_artifact_mappings
from evotensile.campaign.deployment import DeploymentSelection, DeploymentSelectionPayload
from evotensile.database import BenchmarkSummary, EvoTensileDB
from evotensile.profile import PROFILES, TargetProfile, get_profile
from evotensile.protocol import BenchmarkProtocol, apply_benchmark_protocol_overrides
from evotensile.solution_mapping import find_solution_yamls, solution_matches_candidate
from evotensile.tensilelite_parameter_types import normalize_imported_solution_parameters

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
    median_gflops: float


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _with_problem_header_bool_style(text: str) -> str:
    # Nearby checked-in NT logic uses Python-style bools only in the top-level
    # ProblemType Transpose fields; solution dictionaries use normal YAML bools.
    return text.replace("  TransposeA: false\n", "  TransposeA: False\n").replace(
        "  TransposeB: true\n", "  TransposeB: True\n"
    )


def _render_yaml(data: Any) -> str:
    text = yaml.safe_dump(data, default_flow_style=None, sort_keys=False)
    return _with_problem_header_bool_style(text)


def _protocol_from_args(args: argparse.Namespace, profile: TargetProfile) -> BenchmarkProtocol:
    return apply_benchmark_protocol_overrides(profile.default_protocol, vars(args))


def _winner_summaries(
    db: EvoTensileDB,
    *,
    profile: TargetProfile,
    protocol: BenchmarkProtocol,
    min_samples: int,
) -> list[BenchmarkSummary]:
    summaries = db.rank_benchmarks(
        problem_type_hash=profile.problem_type_hash,
        benchmark_protocol_hash=profile.benchmark_protocol_hash(protocol),
        min_samples=min_samples,
    )
    winners_by_shape: dict[str, BenchmarkSummary] = {}
    for summary in summaries:
        winners_by_shape.setdefault(summary.shape_id, summary)
    return [winners_by_shape[shape_id] for shape_id in sorted(winners_by_shape)]


def _load_winners_from_db(
    db: EvoTensileDB,
    *,
    profile: TargetProfile,
    protocol: BenchmarkProtocol,
    min_samples: int,
) -> list[Winner]:
    winners = []
    for summary in _winner_summaries(db, profile=profile, protocol=protocol, min_samples=min_samples):
        if summary.median_gflops is None:
            raise ValueError(f"winner row has no median throughput: {summary}")
        winners.append(
            Winner(
                shape_id=summary.shape_id,
                candidate_hash=summary.candidate_hash,
                median_gflops=summary.median_gflops,
            )
        )
    validated = db.validated_cache_entries(
        problem_type_hash=profile.problem_type_hash,
        validation_protocol_hash=protocol.validation_protocol_hash(),
        shape_ids=[winner.shape_id for winner in winners],
        candidate_hashes=[winner.candidate_hash for winner in winners],
    )
    missing = sorted(
        f"{winner.shape_id}:{winner.candidate_hash}"
        for winner in winners
        if (winner.shape_id, winner.candidate_hash) not in validated
    )
    if missing:
        raise ValueError(f"winners lack current passed validation evidence: {', '.join(missing)}")
    return winners


def _load_winners_from_assignments(
    db: EvoTensileDB,
    *,
    assignments: dict[str, str],
    profile: TargetProfile,
    protocol: BenchmarkProtocol,
    min_samples: int,
) -> list[Winner]:
    summaries = {
        (summary.shape_id, summary.candidate_hash): summary
        for summary in db.rank_benchmarks(
            problem_type_hash=profile.problem_type_hash,
            benchmark_protocol_hash=profile.benchmark_protocol_hash(protocol),
            min_samples=min_samples,
        )
    }
    missing = sorted(
        f"{shape_id}:{candidate_hash}"
        for shape_id, candidate_hash in assignments.items()
        if (shape_id, candidate_hash) not in summaries
    )
    if missing:
        raise ValueError("selected production pairs lack complete confirmation timing: " + ", ".join(missing))
    validated = db.validated_cache_entries(
        problem_type_hash=profile.problem_type_hash,
        validation_protocol_hash=protocol.validation_protocol_hash(),
        shape_ids=list(assignments),
        candidate_hashes=list(dict.fromkeys(assignments.values())),
    )
    unvalidated = sorted(
        f"{shape_id}:{candidate_hash}"
        for shape_id, candidate_hash in assignments.items()
        if (shape_id, candidate_hash) not in validated
    )
    if unvalidated:
        raise ValueError("selected production pairs lack current passed validation: " + ", ".join(unvalidated))
    winners = []
    for shape_id, candidate_hash in sorted(assignments.items()):
        median_gflops = summaries[(shape_id, candidate_hash)].median_gflops
        if median_gflops is None or median_gflops <= 0.0:
            raise ValueError(
                f"selected production pair lacks positive confirmation throughput: {shape_id}:{candidate_hash}"
            )
        winners.append(Winner(shape_id, candidate_hash, median_gflops))
    return winners


def _validate_winner_shape_set(
    winners: list[Winner],
    *,
    profile: TargetProfile,
    allow_partial: bool,
) -> dict[str, list[str]]:
    winner_shape_ids = [winner.shape_id for winner in winners]
    duplicate_shape_ids = sorted(shape_id for shape_id in set(winner_shape_ids) if winner_shape_ids.count(shape_id) > 1)
    if duplicate_shape_ids:
        raise ValueError(f"duplicate winner shapes: {', '.join(duplicate_shape_ids)}")
    if not winner_shape_ids:
        raise ValueError("refusing to generate logic without winners")

    expected = {shape.id for shape in profile.shapes()}
    actual = set(winner_shape_ids)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if extra:
        raise ValueError(f"winner set contains shapes outside profile {profile.name}: {', '.join(extra)}")
    if missing and not allow_partial:
        raise ValueError(
            f"winner set is incomplete for profile {profile.name}: missing {len(missing)} of {len(expected)} shapes"
        )
    return {"expected": sorted(expected), "actual": sorted(actual), "missing": missing}


def _candidate_params_by_hash(db: EvoTensileDB, winners: list[Winner]) -> dict[str, dict[str, Any]]:
    candidate_hashes = list(dict.fromkeys(winner.candidate_hash for winner in winners))
    candidates = db.get_candidates(candidate_hashes)
    out: dict[str, dict[str, Any]] = {candidate.hash: candidate.canonical_params() for candidate in candidates}
    missing = sorted(set(candidate_hashes) - set(out))
    if missing:
        raise ValueError(f"DB is missing candidate JSON for winner hashes: {', '.join(missing)}")
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
        return []
    return [normalize_imported_solution_parameters(solution) for solution in data[5] if isinstance(solution, dict)]


def _solution_records_from_final_yaml(path: Path) -> list[dict[str, Any]]:
    data = _load_yaml(path)
    if not isinstance(data, list):
        return []
    return [
        normalize_imported_solution_parameters(item)
        for item in data
        if isinstance(item, dict) and "SolutionIndex" in item
    ]


def _collect_solution_pool(paths: list[Path]) -> list[dict[str, Any]]:
    solutions: dict[str, dict[str, Any]] = {}
    solution_roots: list[str | Path] = [path for path in paths if path.exists()]
    for path in sorted(find_solution_yamls(solution_roots)):
        for solution in _solution_records_from_logic(path):
            solutions.setdefault(_solution_key(solution), solution)
        for solution in _solution_records_from_final_yaml(path):
            # Data/00_Final.yaml is often the only place that keeps all accepted
            # per-candidate final solutions from a group; 3_LibraryLogic may only
            # contain the group winner for each exact size.
            solutions.setdefault(_solution_key(solution), solution)
    return list(solutions.values())


def _find_matching_solution(candidate_params: dict[str, Any], solutions: list[dict[str, Any]]) -> dict[str, Any] | None:
    for solution in solutions:
        if solution_matches_candidate(solution, candidate_params):
            return solution
    return None


def _build_base_solutions(
    *,
    winners: list[Winner],
    candidate_params: dict[str, dict[str, Any]],
    artifact_solutions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_candidate: dict[str, dict[str, Any]] = {}
    for candidate_hash, params in candidate_params.items():
        solution = _find_matching_solution(params, artifact_solutions)
        if solution is not None:
            by_candidate[candidate_hash] = solution

    missing = sorted({winner.candidate_hash for winner in winners} - set(by_candidate))
    if missing:
        raise ValueError(f"could not find full solution dictionaries for winner hashes: {', '.join(missing)}")
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
                [solution_index_by_hash[winner.candidate_hash], winner.median_gflops],
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
    db_path: Path,
    profile: TargetProfile,
    protocol: BenchmarkProtocol,
    min_samples: int,
    logic_dir: Path,
    variant_names: list[str],
    destination_dir: Path | None = None,
    allow_partial: bool = False,
    winner_assignments: dict[str, str] | None = None,
) -> dict[str, Any]:
    db = EvoTensileDB.connect(
        db_path,
        environment_compatibility_tag=profile.environment_compatibility_tag,
    )
    logic_dir = logic_dir.resolve()
    destination_dir = destination_dir.resolve() if destination_dir is not None else None
    unknown = sorted(set(variant_names) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {', '.join(unknown)}")

    winners = (
        _load_winners_from_db(db, profile=profile, protocol=protocol, min_samples=min_samples)
        if winner_assignments is None
        else _load_winners_from_assignments(
            db,
            assignments=winner_assignments,
            profile=profile,
            protocol=protocol,
            min_samples=min_samples,
        )
    )
    shape_set = _validate_winner_shape_set(winners, profile=profile, allow_partial=allow_partial)
    candidate_params = _candidate_params_by_hash(db, winners)
    candidate_hashes = list(candidate_params)
    artifacts = load_artifact_mappings(
        db,
        problem_type_hash=profile.problem_type_hash,
        shape_ids=[winner.shape_id for winner in winners],
        candidate_hashes=candidate_hashes,
    )
    missing_artifacts = sorted(
        f"{winner.shape_id}:{winner.candidate_hash}"
        for winner in winners
        if (winner.shape_id, winner.candidate_hash) not in artifacts
    )
    if missing_artifacts:
        raise ValueError(f"winners lack complete registered artifacts: {', '.join(missing_artifacts)}")
    artifact_solution_paths = sorted({path for artifact in artifacts.values() for path in artifact.solution_yaml_paths})
    artifact_solutions = _collect_solution_pool(artifact_solution_paths)
    solution_key_order = _reference_solution_key_order(logic_dir)
    base_solutions = _build_base_solutions(
        winners=winners,
        candidate_params=candidate_params,
        artifact_solutions=artifact_solutions,
    )

    rendered_files: dict[Path, str] = {}
    files: dict[str, Any] = {}
    for variant_name in variant_names:
        variant = VARIANTS[variant_name]
        template_path = logic_dir / variant.filename
        template_data = _load_yaml(template_path)
        if not isinstance(template_data, list) or len(template_data) < 8:
            raise ValueError(f"unsupported logic YAML layout: {template_path}")
        updated, solution_count, exact_count = _update_logic_data(
            template_data=template_data,
            variant=variant,
            winners=winners,
            base_solutions=base_solutions,
            solution_key_order=solution_key_order,
        )
        output_path = (destination_dir / variant.filename) if destination_dir is not None else template_path
        rendered_files[output_path] = _render_yaml(updated)
        files[variant_name] = {
            "template_path": str(template_path),
            "output_path": str(output_path),
            "solution_count": solution_count,
            "exact_mapping_count": exact_count,
            "written": destination_dir is not None,
        }

    if destination_dir is not None:
        for output_path, text in rendered_files.items():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")

    return {
        "db": str(db_path),
        "profile": profile.name,
        "problem_type_hash": profile.problem_type_hash,
        "benchmark_protocol_hash": profile.benchmark_protocol_hash(protocol),
        "protocol": protocol.global_parameters(),
        "min_samples": min_samples,
        "logic_dir": str(logic_dir),
        "destination_dir": str(destination_dir) if destination_dir is not None else None,
        "write_mode": "staged"
        if destination_dir is not None and destination_dir != logic_dir
        else ("source" if destination_dir is not None else "preview"),
        "allow_partial": allow_partial,
        "winner_source": "database-rank" if winner_assignments is None else "deployment-selection",
        "expected_shape_count": len(shape_set["expected"]),
        "missing_shape_ids": shape_set["missing"],
        "shape_count": len(winners),
        "candidate_count": len(candidate_params),
        "registered_artifact_count": len(artifacts),
        "artifact_identities": sorted({artifact.code_object_identity for artifact in artifacts.values()}),
        "artifact_solution_paths": [str(path) for path in artifact_solution_paths],
        "artifact_solution_pool_count": len(artifact_solutions),
        "reference_solution_key_count": len(solution_key_order),
        "files": files,
        "note": "No TensileLite run or hipBLASLt rebuild was performed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--num-warmups", type=int, default=None)
    parser.add_argument("--num-benchmarks", type=int, default=None)
    parser.add_argument("--enqueues-per-sync", type=int, default=None)
    parser.add_argument("--syncs-per-benchmark", type=int, default=None)
    parser.add_argument("--num-elements-to-validate", type=int, default=None)
    parser.add_argument("--logic-dir", type=Path, default=DEFAULT_LOGIC_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--write-source", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--selection-json", type=Path)
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS), default=[])
    args = parser.parse_args()

    if not args.db.exists():
        raise FileNotFoundError(args.db)
    if args.output_dir is not None and args.write_source:
        parser.error("--output-dir and --write-source are mutually exclusive")
    profile = get_profile(args.profile)
    protocol = _protocol_from_args(args, profile)
    variant_names = args.variant or ["hhs", "hhs_auxh", "bbs", "bbs_auxb"]
    winner_assignments = None
    if args.selection_json is not None:
        payload = json.loads(args.selection_json.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("deployment selection JSON must contain one object")
        selection_payload = payload.get("selection", payload)
        if not isinstance(selection_payload, dict):
            raise ValueError("deployment selection payload must contain one object")
        winner_assignments = DeploymentSelection.from_dict(
            cast(DeploymentSelectionPayload, selection_payload)
        ).assignments
    result = update_logic_files(
        db_path=args.db,
        profile=profile,
        protocol=protocol,
        min_samples=args.min_samples,
        logic_dir=args.logic_dir,
        variant_names=variant_names,
        destination_dir=args.logic_dir if args.write_source else args.output_dir,
        allow_partial=args.allow_partial,
        winner_assignments=winner_assignments,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
