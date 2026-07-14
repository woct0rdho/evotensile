import importlib
import inspect
import json
import sys
from pathlib import Path

from evotensile.runner import DEFAULT_TENSILELITE_BIN
from evotensile.tensilelite_diagnostics import (
    DiagnosticRecord,
    _add_tensilelite_import_path,
    attribution_inserts_from_diagnostics,
    read_diagnostic_records,
)


def test_read_diagnostic_records_loads_jsonl(tmp_path: Path):
    path = tmp_path / "diagnostics.jsonl"
    path.write_text(
        json.dumps(
            {
                "candidate_hash": "cand_a",
                "candidate_index": 1,
                "status": "kernelwriter_failed",
                "phase": "kernelwriter",
                "reason": "KernelWriter returned errcode -2",
                "shape_ids": ["m1_n2_b1_k3"],
                "errcode": -2,
                "kernel_name": "Kernel0",
                "solution_index": 4,
                "metadata": {"cu_occupancy": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = read_diagnostic_records(path)

    assert records == [
        DiagnosticRecord(
            candidate_hash="cand_a",
            candidate_index=1,
            status="kernelwriter_failed",
            phase="kernelwriter",
            reason="KernelWriter returned errcode -2",
            shape_ids=("m1_n2_b1_k3",),
            errcode=-2,
            kernel_name="Kernel0",
            solution_index=4,
            metadata={"cu_occupancy": 0},
        )
    ]


def test_attribution_inserts_marks_only_structured_failures_reusable():
    inserts = attribution_inserts_from_diagnostics(
        [
            DiagnosticRecord(
                candidate_hash="cand_a",
                candidate_index=0,
                status="kernelwriter_failed",
                phase="kernelwriter",
                shape_ids=("m1_n2_b1_k3",),
            ),
            DiagnosticRecord(
                candidate_hash="cand_ok",
                candidate_index=2,
                status="ok",
                phase="kernelwriter",
                shape_ids=("m1_n2_b1_k3",),
            ),
        ],
        planned_pairs={("m1_n2_b1_k3", "cand_a"), ("m1_n2_b1_k3", "cand_b")},
        failed_candidate_hashes={"cand_a", "cand_b"},
        run_id="diag_run",
        problem_type_hash="ptype",
        benchmark_protocol_hash="proto",
        unattributed_status="build_timeout_unattributed",
    )

    assert [(item.candidate_hash, item.status) for item in inserts] == [
        ("cand_a", "build_failed"),
        ("cand_b", "build_timeout_unattributed"),
    ]


def test_tensilelite_internal_diagnostics_api_contract():
    tensilelite_bin = Path(DEFAULT_TENSILELITE_BIN)
    if not tensilelite_bin.exists():
        return
    _add_tensilelite_import_path(tensilelite_bin)

    import Tensile.SolutionStructs.Utilities as solution_utilities
    from Tensile.BenchmarkProblems import _generate_single_solution
    from Tensile.BenchmarkStructs import BenchmarkProcess, constructForkPermutations
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.TensileCreateLibrary.Run import processKernelSource

    solution_module = importlib.import_module("Tensile.SolutionStructs.Solution")
    assert "perm" in inspect.signature(_generate_single_solution).parameters
    assert "problemSizeGroupConfig" in inspect.signature(BenchmarkProcess).parameters
    assert hasattr(constructForkPermutations, "__iter__")
    assert "kernel" in inspect.signature(processKernelSource).parameters
    assert "debugConfig" in inspect.signature(KernelWriterAssembly).parameters
    assert callable(solution_utilities.reject)
    assert getattr(solution_module, "reject") is solution_utilities.reject
    assert str(tensilelite_bin.resolve().parents[2]) in sys.path
