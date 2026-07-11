import json
from collections.abc import Sequence
from pathlib import Path

from evotensile.database import EvoTensileDB
from evotensile.structured_runner import RunnablePair, StructuredRunOutput


def record_structured_run(
    db: EvoTensileDB,
    output: StructuredRunOutput,
    *,
    yaml_path: Path,
    output_dir: Path,
    pairs: Sequence[RunnablePair],
    cost_phase: str,
) -> None:
    db.insert_run(
        output.run_id,
        yaml_path=str(yaml_path),
        output_dir=str(output_dir),
        status="timeout" if output.timed_out else "ok" if output.ok else "failed",
        returncode=output.returncode,
        candidate_hashes=[pair.candidate_hash for pair in pairs],
        cost_phase=cost_phase,
        duration_s=output.duration_s,
        metadata_json=json.dumps(
            {
                "command": output.command,
                "duration_s": output.duration_s,
                "mode": output.mode,
                "pair_count": len(pairs),
                "results_path": str(output.results_path),
                "stderr_path": str(output.stderr_path),
                "stdout_path": str(output.stdout_path),
                "timed_out": output.timed_out,
            },
            sort_keys=True,
        ),
    )
