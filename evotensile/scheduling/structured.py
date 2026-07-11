from collections.abc import Sequence

from evotensile.database import EvoTensileDB
from evotensile.structured_runner import RunnablePair, StructuredRunOutput


def record_structured_run(
    db: EvoTensileDB,
    output: StructuredRunOutput,
    *,
    pairs: Sequence[RunnablePair],
    cost_phase: str,
) -> None:
    db.insert_run(
        output.run_id,
        phase=cost_phase,
        status="timeout" if output.timed_out else "ok" if output.ok else "failed",
        duration_s=output.duration_s,
        returncode=output.returncode,
        candidate_hashes=[pair.candidate_hash for pair in pairs],
    )
