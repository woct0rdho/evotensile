import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from evotensile.candidate import Candidate, Shape
from evotensile.database import EvoTensileDB
from evotensile.search.mechanics import candidate_shape_mechanics


@dataclass(frozen=True)
class CandidateEvaluationCost:
    proposal_s: float = 0.0
    prepare_s: float = 0.0
    validation_s: float = 0.0
    probe_s: float = 0.0
    screening_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.proposal_s + self.prepare_s + self.validation_s + self.probe_s + self.screening_s


def _manifest_candidate_hashes(yaml_path: str | None) -> set[str]:
    if not yaml_path:
        return set()
    manifest_path = Path(yaml_path).with_name("config.manifest.csv")
    if not manifest_path.exists():
        return set()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return {str(row["candidate_hash"]) for row in csv.DictReader(handle) if row.get("candidate_hash")}


def _pair_candidate_hashes(command: list[object]) -> tuple[set[str], dict[str, object] | None]:
    values = [str(item) for item in command]
    if "--pairs" not in values:
        return set(), None
    pairs_path = Path(values[values.index("--pairs") + 1])
    if not pairs_path.exists():
        return set(), None
    hashes: set[str] = set()
    first: dict[str, object] | None = None
    for line in pairs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pair = json.loads(line)
        first = first or pair
        hashes.add(str(pair["candidate_hash"]))
    return hashes, first


def load_candidate_evaluation_costs(db: EvoTensileDB) -> dict[str, CandidateEvaluationCost]:
    with db.connection() as con:
        run_rows = con.execute(
            "SELECT yaml_path, metadata_json FROM runs WHERE metadata_json IS NOT NULL ORDER BY timestamp"
        ).fetchall()
        candidate_rows = con.execute("SELECT candidate_hash, candidate_json FROM candidates").fetchall()
    mutable: dict[str, dict[str, float]] = {}
    for row in candidate_rows:
        payload = json.loads(row["candidate_json"])
        metadata = payload.get("proposal_metadata", {})
        proposal_cost = float(metadata.get("proposal_cost_s", 0.0) or 0.0)
        mutable[str(row["candidate_hash"])] = {
            "proposal_s": max(0.0, proposal_cost),
            "prepare_s": 0.0,
            "validation_s": 0.0,
            "probe_s": 0.0,
            "screening_s": 0.0,
        }

    for row in run_rows:
        metadata = json.loads(row["metadata_json"])
        duration = max(0.0, float(metadata.get("duration_s", 0.0) or 0.0))
        command = metadata.get("command") or []
        hashes, first_pair = _pair_candidate_hashes(command)
        mode = str(metadata.get("mode") or "")
        if not hashes:
            hashes = _manifest_candidate_hashes(row["yaml_path"])
            phase = "prepare_s"
        elif mode == "validate":
            phase = "validation_s"
        elif first_pair is not None and first_pair.get("num_warmups") == 0:
            phase = "probe_s"
        else:
            phase = "screening_s"
        if not hashes:
            continue
        share = duration / len(hashes)
        for candidate_hash in hashes:
            bucket = mutable.setdefault(
                candidate_hash,
                {
                    "proposal_s": 0.0,
                    "prepare_s": 0.0,
                    "validation_s": 0.0,
                    "probe_s": 0.0,
                    "screening_s": 0.0,
                },
            )
            bucket[phase] += share

    return {candidate_hash: CandidateEvaluationCost(**values) for candidate_hash, values in mutable.items()}


def predicted_candidate_prepare_weight(candidate: Candidate, shape: Shape) -> float:
    mechanics = candidate_shape_mechanics(candidate, shape)
    return (
        1.0
        + 0.40 * mechanics["valu_vgpr_fraction"]
        + 0.25 * mechanics["lds_fraction"]
        + 0.10 * math.log2(max(1.0, mechanics["wave_tile_area"]))
        + 0.05 * math.log2(max(1.0, mechanics["wave_group_size"]))
    )


def predicted_batch_prepare_weight(
    candidates: Sequence[Candidate],
    shapes: Sequence[Shape],
) -> float:
    if not candidates or not shapes:
        return 0.0
    return sum(
        sum(predicted_candidate_prepare_weight(candidate, shape) for shape in shapes) / len(shapes)
        for candidate in candidates
    )
