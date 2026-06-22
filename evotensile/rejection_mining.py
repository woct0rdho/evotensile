import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

_ACTUAL_SOLUTIONS_RE = re.compile(r"Actual Solutions:\s*(\d+)\s*/\s*(\d+)\s+after\s+(\S+)")
_CONFIG_TYPE_RE = re.compile(r"ConfigTypeError:\s*(.*)")
_VGPR_RE = re.compile(r"total\s+vgpr:\s*([^\n]+)", re.IGNORECASE)
_FATAL_RE = re.compile(r"Tensile::FATAL:\s*(.*)")
_TRACEBACK_RE = re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE)
_REJECTION_HINT_RE = re.compile(r"reject(?:ed|ion)?[^\n]*", re.IGNORECASE)


@dataclass(frozen=True)
class RejectionLogSummary:
    path: str
    classification: str
    actual_solutions: int | None = None
    total_solutions: int | None = None
    solution_stage: str | None = None
    messages: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _unique_messages(messages: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for message in messages:
        normalized = " ".join(message.strip().split())
        if normalized and normalized not in seen:
            out.append(normalized)
            seen.add(normalized)
    return tuple(out)


def classify_tensilelite_log(text: str, *, path: str = "") -> RejectionLogSummary:
    messages: list[str] = []
    actual_solutions: int | None = None
    total_solutions: int | None = None
    solution_stage: str | None = None

    for match in _CONFIG_TYPE_RE.finditer(text):
        messages.append(match.group(1))
    for match in _VGPR_RE.finditer(text):
        messages.append(f"total vgpr: {match.group(1)}")
    for match in _FATAL_RE.finditer(text):
        messages.append(match.group(1))
    for match in _REJECTION_HINT_RE.finditer(text):
        messages.append(match.group(0))

    for match in _ACTUAL_SOLUTIONS_RE.finditer(text):
        actual_solutions = int(match.group(1))
        total_solutions = int(match.group(2))
        solution_stage = match.group(3)

    if _CONFIG_TYPE_RE.search(text):
        classification = "schema"
    elif _VGPR_RE.search(text):
        classification = "kernelwriter_resource"
    elif actual_solutions == 0 and total_solutions:
        classification = "solutionstructs_zero"
    elif actual_solutions is not None and total_solutions is not None and actual_solutions < total_solutions:
        classification = "solutionstructs_partial"
    elif "Validation" in text and "FAILED" in text:
        classification = "runtime_validation"
    elif _TRACEBACK_RE.search(text) or _FATAL_RE.search(text):
        classification = "kernelwriter_bug_or_unknown"
    elif actual_solutions is not None:
        classification = "accepted"
    else:
        classification = "unknown"

    return RejectionLogSummary(
        path=path,
        classification=classification,
        actual_solutions=actual_solutions,
        total_solutions=total_solutions,
        solution_stage=solution_stage,
        messages=_unique_messages(messages),
    )


def classify_log_file(path: str | Path) -> RejectionLogSummary:
    log_path = Path(path)
    return classify_tensilelite_log(log_path.read_text(errors="ignore"), path=str(log_path))


def summarize_rejection_logs(paths: list[str | Path]) -> list[RejectionLogSummary]:
    log_paths: list[Path] = []
    for path in paths:
        item = Path(path)
        if item.is_dir():
            log_paths.extend(sorted(item.rglob("*.stdout.log")))
            log_paths.extend(sorted(item.rglob("*.stderr.log")))
        elif item.exists():
            log_paths.append(item)
    return [classify_log_file(path) for path in log_paths]


def classification_counts(summaries: list[RejectionLogSummary]) -> dict[str, int]:
    return dict(Counter(summary.classification for summary in summaries))
