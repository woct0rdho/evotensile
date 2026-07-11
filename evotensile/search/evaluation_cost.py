import json
from dataclasses import dataclass

from evotensile.database import EvoTensileDB


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


def load_candidate_evaluation_costs(db: EvoTensileDB) -> dict[str, CandidateEvaluationCost]:
    with db.connection() as con:
        cost_rows = con.execute(
            """
            SELECT candidate_hash, phase, SUM(duration_s) AS duration_s
            FROM run_candidate_costs
            GROUP BY candidate_hash, phase
            """
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

    phase_fields = {
        "prepare": "prepare_s",
        "validation": "validation_s",
        "probe": "probe_s",
        "screening": "screening_s",
    }
    for row in cost_rows:
        field = phase_fields.get(str(row["phase"]))
        if field is None:
            continue
        candidate_hash = str(row["candidate_hash"])
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
        bucket[field] += max(0.0, float(row["duration_s"]))

    return {candidate_hash: CandidateEvaluationCost(**values) for candidate_hash, values in mutable.items()}
