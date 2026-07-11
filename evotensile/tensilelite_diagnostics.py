import csv
import importlib
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .activity import apu_activity_lock
from .database import EvaluationInsert, EvoTensileDB
from .profile import TargetProfile
from .protocol import BenchmarkProtocol
from .runner import DEFAULT_TENSILELITE_BIN, _merged_env
from .subprocess_utils import run_logged_process


@dataclass(frozen=True)
class DiagnosticRecord:
    candidate_hash: str
    candidate_index: int | None
    status: str
    phase: str
    reason: str | None = None
    shape_ids: tuple[str, ...] = ()
    errcode: int | None = None
    kernel_name: str | None = None
    solution_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiagnosticRunResult:
    run_id: str
    returncode: int
    records: list[DiagnosticRecord]
    results_path: Path
    stdout_path: Path
    stderr_path: Path
    command: list[str]
    duration_s: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def read_diagnostic_records(path: str | Path) -> list[DiagnosticRecord]:
    path = Path(path)
    records: list[DiagnosticRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            shape_ids = value.get("shape_ids") or []
            if not isinstance(shape_ids, list):
                shape_ids = []
            metadata = value.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            records.append(
                DiagnosticRecord(
                    candidate_hash=str(value["candidate_hash"]),
                    candidate_index=int(value["candidate_index"]) if value.get("candidate_index") is not None else None,
                    status=str(value.get("status") or "unknown"),
                    phase=str(value.get("phase") or "unknown"),
                    reason=str(value["reason"]) if value.get("reason") not in (None, "") else None,
                    shape_ids=tuple(str(item) for item in shape_ids),
                    errcode=int(value["errcode"]) if value.get("errcode") is not None else None,
                    kernel_name=str(value["kernel_name"]) if value.get("kernel_name") not in (None, "") else None,
                    solution_index=int(value["solution_index"]) if value.get("solution_index") is not None else None,
                    metadata=metadata,
                )
            )
    return records


def run_tensilelite_diagnostics(
    yaml_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    tensilelite_bin: str | Path = DEFAULT_TENSILELITE_BIN,
    db: EvoTensileDB | None = None,
    target_profile: TargetProfile,
    protocol: BenchmarkProtocol,
    env: dict[str, str] | None = None,
    timeout_s: float | None = None,
    candidate_hashes: list[str] | None = None,
) -> DiagnosticRunResult:
    yaml_path = Path(yaml_path)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"diagnostics_{uuid.uuid4().hex[:12]}"
    results_path = output_dir / f"{run_id}.diagnostics.jsonl"
    stdout_path = output_dir / f"{run_id}.stdout.log"
    stderr_path = output_dir / f"{run_id}.stderr.log"

    command = [
        sys.executable,
        "-m",
        "evotensile.tensilelite_diagnostics",
        "--config",
        str(yaml_path),
        "--manifest",
        str(manifest_path),
        "--output",
        str(results_path),
        "--tensilelite-bin",
        str(tensilelite_bin),
    ]
    start = time.perf_counter()
    timed_out = False
    returncode = 0
    with apu_activity_lock(exclusive=False):
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            returncode, timed_out = run_logged_process(
                command,
                stdout=stdout,
                stderr=stderr,
                env=_merged_env(env),
                timeout_s=timeout_s,
            )
            if timed_out:
                stderr.write(f"\nTensileLite diagnostics timed out after {timeout_s} seconds\n")
    duration_s = time.perf_counter() - start
    records = read_diagnostic_records(results_path) if results_path.exists() else []
    result = DiagnosticRunResult(
        run_id=run_id,
        returncode=returncode,
        records=records,
        results_path=results_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        command=command,
        duration_s=duration_s,
        timed_out=timed_out,
    )
    if db is not None:
        db.insert_run(
            run_id,
            yaml_path=str(yaml_path),
            output_dir=str(output_dir),
            status="timeout" if timed_out else "ok" if returncode == 0 else "failed",
            returncode=returncode,
            candidate_hashes=candidate_hashes,
            cost_phase="prepare",
            duration_s=duration_s,
            metadata_json=json.dumps(
                {
                    "command": command,
                    "duration_s": duration_s,
                    "results_path": str(results_path),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "timed_out": timed_out,
                    "records": len(records),
                },
                sort_keys=True,
            ),
        )
    return result


def attribution_inserts_from_diagnostics(
    records: list[DiagnosticRecord],
    *,
    planned_shape_ids: list[str],
    failed_candidate_hashes: set[str],
    run_id: str | None,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    unattributed_status: str = "build_failed_unattributed",
) -> list[EvaluationInsert]:
    records_by_hash: dict[str, DiagnosticRecord] = {}
    for record in records:
        if record.candidate_hash not in failed_candidate_hashes:
            continue
        if record.status in {"solutionstructs_rejected", "kernelwriter_failed", "codegen_failed", "build_failed"}:
            records_by_hash[record.candidate_hash] = record
    inserts: list[EvaluationInsert] = []
    for candidate_hash in sorted(failed_candidate_hashes):
        record = records_by_hash.get(candidate_hash)
        status = "build_failed" if record is not None else unattributed_status
        shape_ids = record.shape_ids if record is not None and record.shape_ids else tuple(planned_shape_ids)
        for shape_id in shape_ids:
            inserts.append(
                EvaluationInsert(
                    shape_id=shape_id,
                    candidate_hash=candidate_hash,
                    run_id=run_id,
                    status=status,
                    problem_type_hash=problem_type_hash,
                    benchmark_protocol_hash=benchmark_protocol_hash,
                )
            )
    return inserts


def _manifest_rows(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            candidate_index = int(row.get("candidate_index") or 0)
            out.setdefault(
                candidate_index,
                {
                    "candidate_hash": row["candidate_hash"],
                    "candidate_index": candidate_index,
                    "shape_ids": [],
                },
            )
            out[candidate_index]["shape_ids"].append(row["shape_id"])
    for row in out.values():
        row["shape_ids"] = list(dict.fromkeys(row["shape_ids"]))
    return out


def _write_record(handle: Any, **record: Any) -> None:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


def _architecture_name(config: dict[str, Any]) -> str:
    library_logic = config.get("LibraryLogic") or {}
    architecture = library_logic.get("ArchitectureName") or library_logic.get("ScheduleName") or "gfx1151"
    return str(architecture)


def _add_tensile_import_path(tensilelite_bin: Path) -> None:
    parentdir = tensilelite_bin.resolve().parents[2]
    if str(parentdir) not in sys.path:
        sys.path.insert(0, str(parentdir))


def _diagnose_with_tensilelite(config_path: Path, manifest_path: Path, output_path: Path, tensilelite_bin: Path) -> int:
    _add_tensile_import_path(tensilelite_bin)

    import rocisa
    import Tensile.SolutionStructs.Utilities as solution_utilities
    from Tensile.BenchmarkProblems import _generate_single_solution
    from Tensile.BenchmarkStructs import BenchmarkProcess, constructForkPermutations
    from Tensile.Common import makeDebugConfig, setVerbosity
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Common.GlobalParameters import assignGlobalParameters, globalParameters, restoreDefaultGlobalParameters
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.TensileCreateLibrary.Run import processKernelSource
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import ToolchainDefaults

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"expected TensileLite config object: {config_path}")
    manifest = _manifest_rows(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    restoreDefaultGlobalParameters()
    globalParameters["ConfigPath"] = [str(config_path)]
    globalParameters["CpuThreads"] = 0
    globalParameters["PrintSolutionRejectionReason"] = False
    globalParameters["ForceRedoBenchmarkProblems"] = True
    globalParameters["GenerateSourcesAndExit"] = True
    architecture = _architecture_name(config)
    isa = gfxToIsa(architecture)
    if isa is None:
        raise ValueError(f"unsupported TensileLite architecture: {architecture}")
    cxx_compiler = str(ToolchainDefaults.CXX_COMPILER)
    isa_info_map = makeIsaInfoMap([isa], cxx_compiler)
    assignGlobalParameters(config.get("GlobalParameters", {}), isa_info_map)
    setVerbosity(0)

    asm_toolchain = makeAssemblyToolchain(
        cxx_compiler,
        str(ToolchainDefaults.OFFLOAD_BUNDLER),
        globalParameters.get("CodeObjectVersion", "4"),
    )
    debug_config = makeDebugConfig({**config.get("GlobalParameters", {}), "PrintSolutionRejectionReason": False})
    debug_config = debug_config._replace(printSolutionRejectionReason=False)
    kernel_writer = KernelWriterAssembly(asm_toolchain.assembler, debug_config)
    ti = rocisa.rocIsa.getInstance()  # ty: ignore[unresolved-attribute]
    ti.init(isa, cxx_compiler, False)
    out_options = ti.getOutputOptions()
    split_gsu = debug_config.splitGSU

    solution_module = importlib.import_module("Tensile.SolutionStructs.Solution")
    captured_rejections: dict[int, list[str]] = {}
    original_utility_reject = solution_utilities.reject
    original_solution_reject = getattr(solution_module, "reject")

    def capture_reject(state, print_solution_rejection_reason=True, *args):
        candidate_index = None
        if state is not None:
            candidate_index = state.get("_EvoTensileCandidateIndex")
        if args:
            reason_args = args
        elif isinstance(print_solution_rejection_reason, str):
            reason_args = (print_solution_rejection_reason,)
        else:
            reason_args = ()
        reason = " ".join(str(arg) for arg in reason_args) if reason_args else None
        if candidate_index is not None and reason:
            captured_rejections.setdefault(int(candidate_index), []).append(reason)
        return original_utility_reject(state, False, *reason_args)

    solution_utilities.reject = capture_reject
    setattr(solution_module, "reject", capture_reject)
    try:
        with output_path.open("w", encoding="utf-8") as output:
            benchmark_problems = config.get("BenchmarkProblems") or []
            if not benchmark_problems:
                raise ValueError("TensileLite config has no BenchmarkProblems")
            solution_index = 0
            for outer_index, benchmark_problem_type_config in enumerate(benchmark_problems):
                problem_type_config = benchmark_problem_type_config[0]
                problem_size_groups = benchmark_problem_type_config[1:] or [{}]
                for group_index, size_group_config in enumerate(problem_size_groups):
                    benchmark_process = BenchmarkProcess(
                        problem_type_config,
                        size_group_config,
                        debug_config.printIndexAssignmentInfo,
                        keyPathPrefix=f"BenchmarkProblems[{outer_index}][{1 + group_index}]",
                        srcFile=str(config_path),
                    )
                    for benchmark_step in benchmark_process.benchmarkSteps:
                        fork_permutations = constructForkPermutations(
                            benchmark_step.forkParams,
                            benchmark_step.paramGroups,
                        )
                        for candidate_index, permutation in enumerate(fork_permutations):
                            manifest_row = manifest.get(candidate_index)
                            if manifest_row is None:
                                continue
                            permutation = dict(permutation)
                            permutation["_EvoTensileCandidateIndex"] = candidate_index
                            captured_rejections.pop(candidate_index, None)
                            solution = _generate_single_solution(
                                permutation,
                                benchmark_process.problemType,
                                benchmark_step.constantParams,
                                asm_toolchain.assembler,
                                debug_config,
                                isa_info_map,
                            )
                            base = {
                                "candidate_hash": manifest_row["candidate_hash"],
                                "candidate_index": candidate_index,
                                "shape_ids": manifest_row["shape_ids"],
                            }
                            if solution is None:
                                reasons = captured_rejections.get(candidate_index) or []
                                _write_record(
                                    output,
                                    **base,
                                    status="solutionstructs_rejected",
                                    phase="solutionstructs",
                                    reason=reasons[-1] if reasons else None,
                                    metadata={"reasons": reasons},
                                )
                                continue
                            solution["SolutionIndex"] = solution_index
                            solution_index += 1
                            kernels = solution.getKernels()
                            failed = False
                            for kernel in kernels:
                                try:
                                    result = processKernelSource(
                                        kernel_writer,
                                        ti.getData(),
                                        out_options,
                                        split_gsu,
                                        kernel,
                                        compress=False,
                                    )
                                except Exception as exc:
                                    _write_record(
                                        output,
                                        **base,
                                        status="codegen_failed",
                                        phase="kernelwriter",
                                        reason=str(exc),
                                        solution_index=solution.get("SolutionIndex"),
                                        kernel_name=str(
                                            kernel.get("KernelNameMin") or kernel.get("SolutionNameMin") or ""
                                        ),
                                    )
                                    failed = True
                                    break
                                overflowed_resources = int(getattr(kernel_writer.states, "overflowedResources", 0) or 0)
                                if result.err != 0 or overflowed_resources:
                                    reason = (
                                        f"KernelWriter overflowed resources {overflowed_resources}"
                                        if overflowed_resources
                                        else f"KernelWriter returned errcode {result.err}"
                                    )
                                    _write_record(
                                        output,
                                        **base,
                                        status="kernelwriter_failed",
                                        phase="kernelwriter",
                                        reason=reason,
                                        errcode=result.err,
                                        kernel_name=str(result.name),
                                        solution_index=solution.get("SolutionIndex"),
                                        metadata={
                                            "cu_occupancy": result.cuoccupancy,
                                            "overflowed_resources": overflowed_resources,
                                            "prefetch_global_read": result.pgr,
                                        },
                                    )
                                    failed = True
                                    break
                            if not failed:
                                _write_record(
                                    output,
                                    **base,
                                    status="ok",
                                    phase="kernelwriter",
                                    solution_index=solution.get("SolutionIndex"),
                                )
    finally:
        solution_utilities.reject = original_utility_reject
        setattr(solution_module, "reject", original_solution_reject)
    return 0


def _parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Emit structured TensileLite candidate diagnostics")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensilelite-bin", type=Path, default=Path(DEFAULT_TENSILELITE_BIN))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return _diagnose_with_tensilelite(args.config, args.manifest, args.output, args.tensilelite_bin)
    except Exception as exc:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as output:
            _write_record(
                output,
                candidate_hash="<diagnostics>",
                candidate_index=None,
                status="diagnostics_failed",
                phase="diagnostics",
                reason=str(exc),
            )
        print(f"TensileLite diagnostics failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
