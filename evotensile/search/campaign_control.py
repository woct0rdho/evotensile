import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evotensile.candidate import Candidate, Shape
from evotensile.database import EvoTensileDB
from evotensile.protocol import BenchmarkProtocol
from evotensile.search.encoding import candidate_to_genome, hamming_distance
from evotensile.search.family import family_descriptor_counts
from evotensile.search.mechanics import mechanical_coverage_tokens


@dataclass(frozen=True)
class ProposalEvent:
    island_id: str
    seed: int
    restart_index: int
    learned_linkage: bool
    scope_kind: str
    scope_shape_ids: tuple[str, ...]
    parent_hashes: tuple[str, ...]
    preserved_hashes: tuple[str, ...]
    generated_hashes: tuple[str, ...]
    selected_hashes: tuple[str, ...]
    duration_s: float
    proposal_cost_s: float
    proposal_args: Mapping[str, object]

    @property
    def selected_generated_hashes(self) -> tuple[str, ...]:
        generated = set(self.generated_hashes)
        return tuple(candidate_hash for candidate_hash in self.selected_hashes if candidate_hash in generated)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ProposalEvent":
        def hashes(key: str) -> tuple[str, ...]:
            values = payload[key]
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError(f"proposal event {key} must be a sequence")
            return tuple(str(value) for value in values)

        def number(key: str) -> float:
            value = payload[key]
            if not isinstance(value, (int, float, str)):
                raise ValueError(f"proposal event {key} must be numeric")
            return float(value)

        proposal_args = payload["proposal_args"]
        if not isinstance(proposal_args, Mapping):
            raise ValueError("proposal event arguments must be a mapping")
        return cls(
            island_id=str(payload["island_id"]),
            seed=int(number("seed")),
            restart_index=int(number("restart_index")),
            learned_linkage=bool(payload["learned_linkage"]),
            scope_kind=str(payload["scope_kind"]),
            scope_shape_ids=hashes("scope_shape_ids"),
            parent_hashes=hashes("parent_hashes"),
            preserved_hashes=hashes("preserved_hashes"),
            generated_hashes=hashes("generated_hashes"),
            selected_hashes=hashes("selected_hashes"),
            duration_s=number("duration_s"),
            proposal_cost_s=number("proposal_cost_s"),
            proposal_args={str(key): value for key, value in proposal_args.items()},
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "island_id": self.island_id,
            "seed": self.seed,
            "restart_index": self.restart_index,
            "learned_linkage": self.learned_linkage,
            "scope_kind": self.scope_kind,
            "scope_shape_ids": list(self.scope_shape_ids),
            "parent_hashes": list(self.parent_hashes),
            "preserved_hashes": list(self.preserved_hashes),
            "generated_hashes": list(self.generated_hashes),
            "selected_hashes": list(self.selected_hashes),
            "selected_generated_hashes": list(self.selected_generated_hashes),
            "duration_s": self.duration_s,
            "proposal_cost_s": self.proposal_cost_s,
            "proposal_args": dict(self.proposal_args),
        }


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


def tag_generated_proposals(
    candidates: Sequence[Candidate],
    *,
    generated_hashes: set[str],
    island_id: str,
    proposal_cost_s: float,
    restart_index: int = 0,
) -> list[Candidate]:
    tagged = []
    for candidate in candidates:
        if candidate.hash not in generated_hashes:
            tagged.append(candidate)
            continue
        metadata = {
            **candidate.proposal_metadata,
            "island_id": island_id,
            "restart_index": restart_index,
            "proposal_cost_s": proposal_cost_s,
        }
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


def population_diagnostics(
    candidates: Sequence[Candidate],
    shape: Shape,
    *,
    effective_cu_count: int,
) -> PopulationDiagnostics:
    deduped = list({candidate.hash: candidate for candidate in candidates}.values())
    if not deduped:
        return PopulationDiagnostics(0, 0, 0, 0, 0.0, 0)
    sampled = deduped[:64]
    genomes = [candidate_to_genome(candidate) for candidate in sampled]
    distances = [hamming_distance(left, right) for index, left in enumerate(genomes) for right in genomes[index + 1 :]]
    tokens = set().union(
        *(
            mechanical_coverage_tokens(
                candidate,
                shape,
                effective_cu_count=effective_cu_count,
            )
            for candidate in deduped
        )
    )
    instructions = {tuple(candidate.canonical_params()["MatrixInstruction"]) for candidate in deduped}
    return PopulationDiagnostics(
        candidates=len(deduped),
        family_cells=len(family_descriptor_counts(deduped)),
        matrix_instructions=len(instructions),
        mechanical_tokens=len(tokens),
        mean_pairwise_hamming=statistics.fmean(distances) if distances else 0.0,
        minimum_pairwise_hamming=min(distances) if distances else 0,
    )


def restart_epoch(
    counters: dict[str, int],
    *,
    scope: str,
    transition: bool,
) -> int:
    current = counters.get(scope, 0)
    if transition:
        current += 1
        counters[scope] = current
    return current


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


def estimate_confirmation_reserve_s(
    finalist_median_times_us: Sequence[float],
    *,
    protocol: BenchmarkProtocol,
    top_k: int,
    minimum_reserve_s: float,
    per_finalist_overhead_s: float = 1.0,
    duration_margin: float = 1.5,
) -> float:
    launches = protocol.num_warmups + (
        protocol.num_benchmarks * protocol.enqueues_per_sync * protocol.syncs_per_benchmark
    )
    estimate = 0.0
    for median_time_us in finalist_median_times_us[: max(0, top_k)]:
        timed_duration_s = max(0.0, median_time_us) * launches / 1_000_000.0
        estimate += per_finalist_overhead_s + timed_duration_s * duration_margin
    return max(minimum_reserve_s, estimate)


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
