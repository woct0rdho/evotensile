from dataclasses import dataclass

from evotensile.database import EvoTensileDB


@dataclass(frozen=True)
class CandidateMeasuredCost:
    proposal_s: float = 0.0
    prepare_s: float = 0.0
    validation_s: float = 0.0
    probe_s: float = 0.0
    screening_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.proposal_s + self.prepare_s + self.validation_s + self.probe_s + self.screening_s


def load_candidate_measured_costs(db: EvoTensileDB) -> dict[str, CandidateMeasuredCost]:
    with db.connection() as con:
        cost_rows = con.execute(
            """
            SELECT c.candidate_hash, rcc.phase, SUM(rcc.duration_s) AS duration_s
            FROM run_candidate_costs AS rcc
            JOIN candidates AS c USING (candidate_id)
            GROUP BY rcc.candidate_id, rcc.phase
            """
        ).fetchall()
        proposal_rows = con.execute(
            """
            WITH generated_counts AS (
              SELECT proposal_event_id, COUNT(DISTINCT candidate_id) AS generated_count
              FROM proposal_candidates WHERE state = 'generated' GROUP BY proposal_event_id
            )
            SELECT c.candidate_hash,
                   SUM(pe.duration_s / generated_counts.generated_count) AS duration_s
            FROM proposal_candidates AS pc
            JOIN proposal_events AS pe USING (proposal_event_id)
            JOIN generated_counts USING (proposal_event_id)
            JOIN candidates AS c USING (candidate_id)
            WHERE pc.state = 'generated'
            GROUP BY pc.candidate_id
            """
        ).fetchall()
        candidate_rows = con.execute("SELECT candidate_hash FROM candidates").fetchall()
    proposal_costs = {str(row["candidate_hash"]): max(0.0, float(row["duration_s"])) for row in proposal_rows}
    mutable: dict[str, dict[str, float]] = {}
    for row in candidate_rows:
        candidate_hash = str(row["candidate_hash"])
        mutable[candidate_hash] = {
            "proposal_s": proposal_costs.get(candidate_hash, 0.0),
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

    return {candidate_hash: CandidateMeasuredCost(**values) for candidate_hash, values in mutable.items()}
