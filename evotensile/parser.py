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


def _first_present(row: dict[str, str], names: list[str]) -> str | None:
    lower_map = {k.lower(): k for k in row.keys()}
    for name in names:
        key = lower_map.get(name.lower())
        if key is not None:
            value = row.get(key)
            if value not in (None, ""):
                return value
    return None


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_tensile_csv(path: str | Path) -> list[CsvEvaluation]:
    """Parse a TensileLite benchmark CSV with tolerant column names.

    Tensile result column names have changed across versions.  This parser extracts
    the fields EvoTensile needs if present and preserves the raw row for debugging.
    """
    path = Path(path)
    out: list[CsvEvaluation] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(line for line in f if line.strip() and not line.startswith("#"))
        for row in reader:
            problem_index = _to_int(_first_present(row, ["ProblemIdx", "ProblemIndex", "problem-index", "Problem"]))
            solution_index = _to_int(_first_present(row, ["SolutionIndex", "solution-index", "Solution", "SolIdx"]))
            time_us = _to_float(_first_present(row, ["TimeUS", "time-us", "Time", "us"]))
            gflops = _to_float(_first_present(row, ["SpeedGFlops", "GFlops", "gflops", "WinnerGFlops"]))
            validation = _first_present(row, ["Validation", "validation"])
            out.append(
                CsvEvaluation(
                    problem_index=problem_index,
                    solution_index=solution_index,
                    time_us=time_us,
                    gflops=gflops,
                    validation=validation,
                    raw=dict(row),
                )
            )
    return out


def find_result_csvs(output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    return sorted(output_dir.glob("**/*.csv"))
