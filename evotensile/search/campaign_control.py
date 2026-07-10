import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evotensile.candidate import Candidate, Shape
from evotensile.database import EvoTensileDB
from evotensile.search.encoding import candidate_to_genome, hamming_distance
from evotensile.search.family import family_descriptor_counts
from evotensile.search.mechanics import mechanical_coverage_tokens


@dataclass(frozen=True)
class PopulationDiagnostics:
    candidates: int
    family_cells: int
    matrix_instructions: int
    mechanical_tokens: int
    mean_pairwise_hamming: float
    minimum_pairwise_hamming: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "candidates": self.candidates,
            "family_cells": self.family_cells,
            "matrix_instructions": self.matrix_instructions,
            "mechanical_tokens": self.mechanical_tokens,
            "mean_pairwise_hamming": self.mean_pairwise_hamming,
            "minimum_pairwise_hamming": self.minimum_pairwise_hamming,
        }


def tag_proposals(
    candidates: Sequence[Candidate],
    *,
    island_id: str,
    parent_hashes: set[str],
    proposal_duration_s: float,
    restart_index: int = 0,
) -> list[Candidate]:
    generated_count = sum(candidate.hash not in parent_hashes for candidate in candidates)
    proposal_cost_s = proposal_duration_s / max(generated_count, 1)
    tagged = []
    for candidate in candidates:
        metadata = dict(candidate.proposal_metadata)
        metadata["island_id"] = island_id
        metadata["restart_index"] = restart_index
        if candidate.hash not in parent_hashes:
            metadata["proposal_cost_s"] = proposal_cost_s
        tagged.append(
            Candidate(
                params=candidate.canonical_params(),
                source=candidate.source,
                parent_hashes=candidate.parent_hashes,
                proposal_metadata=metadata,
            )
        )
    return tagged


def load_island_elites(
    db: EvoTensileDB,
    *,
    island_id: str,
    shape_id: str,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    limit: int,
) -> list[Candidate]:
    summaries = db.rank_evaluations(
        shape_id=shape_id,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        min_samples=1,
        limit=None,
    )
    candidates = db.get_candidates([summary.candidate_hash for summary in summaries])
    return [
        candidate for candidate in candidates if str(candidate.proposal_metadata.get("island_id", "")) == island_id
    ][:limit]


def population_diagnostics(candidates: Sequence[Candidate], shape: Shape) -> PopulationDiagnostics:
    deduped = list({candidate.hash: candidate for candidate in candidates}.values())
    if not deduped:
        return PopulationDiagnostics(0, 0, 0, 0, 0.0, 0)
    sampled = deduped[:64]
    genomes = [candidate_to_genome(candidate) for candidate in sampled]
    distances = [hamming_distance(left, right) for index, left in enumerate(genomes) for right in genomes[index + 1 :]]
    tokens = set().union(*(mechanical_coverage_tokens(candidate, shape) for candidate in deduped))
    instructions = {tuple(candidate.canonical_params()["MatrixInstruction"]) for candidate in deduped}
    return PopulationDiagnostics(
        candidates=len(deduped),
        family_cells=len(family_descriptor_counts(deduped)),
        matrix_instructions=len(instructions),
        mechanical_tokens=len(tokens),
        mean_pairwise_hamming=statistics.fmean(distances) if distances else 0.0,
        minimum_pairwise_hamming=min(distances) if distances else 0,
    )


def plateau_detected(
    best_history: Sequence[float],
    *,
    patience: int,
    minimum_improvement_fraction: float,
) -> bool:
    if patience <= 0 or len(best_history) <= patience:
        return False
    previous_best = max(best_history[:-patience])
    recent_best = max(best_history[-patience:])
    return recent_best <= previous_best * (1.0 + max(0.0, minimum_improvement_fraction))


def estimate_next_round_duration_s(
    rounds: Sequence[Mapping[str, object]],
    *,
    expected_missing_pairs: int,
    minimum_s: float = 20.0,
) -> float:
    usable = []
    for item in rounds[-6:]:
        schedule = item.get("schedule")
        if not isinstance(schedule, Mapping):
            continue
        missing_value = schedule.get("missing_pairs", 0)
        duration_value = item.get("duration_s", 0.0)
        missing = int(missing_value) if isinstance(missing_value, (int, float, str)) else 0
        duration = float(duration_value) if isinstance(duration_value, (int, float, str)) else 0.0
        if missing > 0 and duration > 0.0:
            usable.append(duration / missing)
    if not usable:
        return max(minimum_s, 30.0)
    median_per_pair = statistics.median(usable)
    deviations = [abs(value - median_per_pair) for value in usable]
    robust_margin = statistics.median(deviations) if deviations else 0.0
    estimate = (median_per_pair + robust_margin) * max(1, expected_missing_pairs)
    return max(minimum_s, estimate * 1.15 + 5.0)


def convergence_detected(
    best_history: Sequence[float],
    diagnostics: PopulationDiagnostics,
    *,
    patience: int = 8,
    minimum_improvement_fraction: float = 0.0025,
    maximum_mean_hamming: float = 4.0,
) -> bool:
    return (
        plateau_detected(
            best_history,
            patience=patience,
            minimum_improvement_fraction=minimum_improvement_fraction,
        )
        and diagnostics.mean_pairwise_hamming <= maximum_mean_hamming
    )


def split_budget(total: int, parts: int) -> list[int]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    base, remainder = divmod(max(0, total), parts)
    return [base + int(index < remainder) for index in range(parts)]


def scaled_count(value: int, numerator: int, denominator: int) -> int:
    if value <= 0 or numerator <= 0:
        return 0
    return max(1, math.floor(value * numerator / denominator))
