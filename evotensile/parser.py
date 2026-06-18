import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CsvEvaluation:
    problem_index: int | None
    solution_index: int | None
    time_us: float | None
    gflops: float | None
    validation: str | None
    raw: dict[str, Any]
    solution_name: str | None = None
    m: int | None = None
    n: int | None = None
    batch: int | None = None
    k: int | None = None

    @property
    def shape_id(self) -> str | None:
        if None in (self.m, self.n, self.batch, self.k):
            return None
        return f"m{self.m}_n{self.n}_b{self.batch}_k{self.k}"


def _normalize_name(name: str | None) -> str:
    return "" if name is None else name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _first_present(row: dict[str, str], names: list[str]) -> str | None:
    normalized = {_normalize_name(k): k for k in row.keys()}
    for name in names:
        key = normalized.get(_normalize_name(name))
        if key is not None:
            value = row.get(key)
            if value not in (None, ""):
                return value.strip() if isinstance(value, str) else value
    return None


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if "/" in text:
        text = text.split("/", 1)[0]
    try:
        return int(float(text))
    except ValueError:
        return None


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value.strip())
    except ValueError:
        return None
    if parsed != parsed:
        return None
    return parsed


def _csv_fields(line: str) -> list[str]:
    try:
        return next(csv.reader([line], skipinitialspace=True))
    except csv.Error:
        return []


def _is_header(fields: list[str]) -> bool:
    names = {_normalize_name(field) for field in fields}
    if "validation" in names and ("timeus" in names or "gflops" in names):
        return True
    if "totalflops" in names and (
        "winnergflops" in names or any(field.strip().startswith("Cijk_") for field in fields)
    ):
        return True
    return False


def _row_dict(header: list[str], fields: list[str]) -> dict[str, str] | None:
    if len(fields) < min(4, len(header)):
        return None
    padded = fields + [""] * max(0, len(header) - len(fields))
    return {header[i].strip(): padded[i].strip() for i in range(len(header))}


def _shape_from_row(row: dict[str, str]) -> tuple[int | None, int | None, int | None, int | None]:
    m = _to_int(_first_present(row, ["SizeI", "M"]))
    n = _to_int(_first_present(row, ["SizeJ", "N"]))
    batch = _to_int(_first_present(row, ["SizeK", "Batch", "BatchSize"]))
    k = _to_int(_first_present(row, ["SizeL", "K"]))
    if None not in (m, n, batch, k):
        return m, n, batch, k

    problem_sizes = _first_present(row, ["problem-sizes", "ProblemSizes"])
    if problem_sizes:
        text = problem_sizes.strip().strip("()").strip("[]")
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if len(parts) >= 4:
            parsed = [_to_int(part) for part in parts[:4]]
            if all(value is not None for value in parsed):
                return parsed[0], parsed[1], parsed[2], parsed[3]
    return m, n, batch, k


def _last_nonzero_perf_column(row: dict[str, str]) -> float | None:
    keys = list(row.keys())
    total_idx = next((i for i, key in enumerate(keys) if _normalize_name(key) == "totalflops"), None)
    if total_idx is None:
        return None
    values: list[float] = []
    for key in keys[total_idx + 1 :]:
        if _normalize_name(key) in {
            "tilespercu",
            "totalgranularity",
            "winnergflops",
            "winnertimeus",
            "winneridx",
            "winnername",
        }:
            continue
        value = _to_float(row.get(key))
        if value is not None and value != 0:
            values.append(value)
    return values[-1] if values else None


def _evaluation_from_row(row: dict[str, str]) -> CsvEvaluation | None:
    if _first_present(row, ["run"]) is not None and _to_int(_first_present(row, ["run"])) is None:
        return None
    problem_index = _to_int(
        _first_present(row, ["ProblemIdx", "ProblemIndex", "problem-index", "Problem", "problem-progress"])
    )
    solution_index = _to_int(
        _first_present(row, ["SolutionIndex", "solution-index", "WinnerIdx", "SolIdx", "solution-progress", "Solution"])
    )
    time_us = _to_float(_first_present(row, ["WinnerTimeUS", "TimeUS", "time-us", "Time", "us"]))
    gflops = _to_float(_first_present(row, ["SpeedGFlops", "WinnerGFlops", "gflops", "GFlops"]))
    if gflops == 0 or gflops is None:
        gflops = _last_nonzero_perf_column(row)
    validation = _first_present(row, ["Validation", "validation"])
    solution_name = _first_present(row, ["WinnerName", "SolutionName", "KernelName", "solution"])
    m, n, batch, k = _shape_from_row(row)

    if solution_index is None and problem_index is None and None in (m, n, batch, k):
        return None
    if time_us is None and gflops is None:
        validation_text = str(validation or "").strip().lower()
        known_validation = any(token in validation_text for token in ["pass", "fail", "invalid", "mismatch", "error"])
        if not known_validation:
            return None
    return CsvEvaluation(
        problem_index=problem_index,
        solution_index=solution_index,
        time_us=time_us,
        gflops=gflops,
        validation=validation,
        raw=dict(row),
        solution_name=solution_name,
        m=m,
        n=n,
        batch=batch,
        k=k,
    )


def _parse_blocks(lines: list[str]) -> list[CsvEvaluation]:
    out: list[CsvEvaluation] = []
    active_header: list[str] | None = None
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = _csv_fields(line)
        if not fields:
            continue
        if _is_header(fields):
            active_header = [field.strip() for field in fields]
            continue
        if active_header is None:
            continue
        row = _row_dict(active_header, fields)
        if row is None:
            continue
        evaluation = _evaluation_from_row(row)
        if evaluation is not None:
            out.append(evaluation)
    return out


def parse_tensilelite_csv(path: str | Path) -> list[CsvEvaluation]:
    """Parse TensileLite CSV files or stdout logs with embedded CSV blocks."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return _parse_blocks(lines)


def validation_status(validation: str | None, *, require_validation: bool = True) -> str:
    """Map a TensileLite validation field to an EvoTensile evaluation status."""
    if validation is None or not str(validation).strip():
        return "validation_unknown" if require_validation else "ok"
    value = str(validation).strip().lower()
    if value in {"pass", "passed", "valid", "ok", "true"}:
        return "ok"
    if value in {"fail", "failed", "invalid", "mismatch", "mismatched", "false"}:
        return "validation_fail"
    if "pass" in value and "fail" not in value and "invalid" not in value:
        return "ok"
    if any(token in value for token in ["fail", "invalid", "mismatch", "error"]):
        return "validation_fail"
    return "validation_unknown" if require_validation else "ok"


def evaluation_status(evaluation: CsvEvaluation, *, require_validation: bool = True) -> str:
    status = validation_status(evaluation.validation, require_validation=require_validation)
    if status != "ok":
        return status
    if evaluation.time_us is None and evaluation.gflops is None:
        return "parse_fail"
    return "ok"


def find_result_csvs(output_dir: str | Path, *, include_logs: bool = False) -> list[Path]:
    output_dir = Path(output_dir)
    patterns = ["**/*.csv"]
    if include_logs:
        patterns.extend(["**/*.log", "**/*.stdout"])
    out: list[Path] = []
    for pattern in patterns:
        out.extend(output_dir.glob(pattern))
    return sorted(set(out))
