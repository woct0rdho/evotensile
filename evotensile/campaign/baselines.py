import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.evaluator import EvaluationResult, PairEvaluator
from evotensile.candidate import Candidate, Shape
from evotensile.scheduling.models import EvidenceStage, PairRequest
from evotensile.search.replay import OracleRecord
from evotensile.search.shape_clustering import ShapeClustering


@dataclass(frozen=True)
class BaselineEvaluation:
    policy: str
    requested_pairs: int
    result: EvaluationResult


@dataclass(frozen=True)
class RepresentativeDiagnostics:
    tolerance_fraction: float
    medoid_pairs: int
    known_medoid_pairs: int
    assessed_cluster_pairs: int
    precise_promotions: int
    missed_specialists: int
    unavailable_promotions: int
    promotion_precision: float | None
    missed_specialist_rate: float | None
    median_regret_fraction: float | None
    worst_regret_fraction: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "tolerance_fraction": self.tolerance_fraction,
            "medoid_pairs": self.medoid_pairs,
            "known_medoid_pairs": self.known_medoid_pairs,
            "assessed_cluster_pairs": self.assessed_cluster_pairs,
            "precise_promotions": self.precise_promotions,
            "missed_specialists": self.missed_specialists,
            "unavailable_promotions": self.unavailable_promotions,
            "promotion_precision": self.promotion_precision,
            "missed_specialist_rate": self.missed_specialist_rate,
            "median_regret_fraction": self.median_regret_fraction,
            "worst_regret_fraction": self.worst_regret_fraction,
        }


def evaluate_global_candidate_dense_baseline(
    evaluator: PairEvaluator,
    controller: CampaignControllerState,
    *,
    candidates: Sequence[Candidate],
    shapes: Sequence[Shape],
    evidence_stage: EvidenceStage = EvidenceStage.SCREENING,
    min_samples: int = 1,
) -> BaselineEvaluation:
    requests = [
        PairRequest(candidate, shape, evidence_stage=evidence_stage, min_samples=min_samples)
        for candidate in candidates
        for shape in shapes
    ]
    result = evaluator.evaluate(requests)
    result.apply(controller)
    return BaselineEvaluation("global_candidate_dense", len(requests), result)


def evaluate_representative_first_baseline(
    evaluator: PairEvaluator,
    controller: CampaignControllerState,
    *,
    candidates: Sequence[Candidate],
    shapes: Sequence[Shape],
    clustering: ShapeClustering,
    evidence_stage: EvidenceStage = EvidenceStage.SCREENING,
    min_samples: int = 1,
) -> BaselineEvaluation:
    shape_by_id = {shape.id: shape for shape in shapes}
    if set(shape_by_id) != set(clustering.shape_ids):
        raise ValueError("representative baseline clustering must cover the requested shapes")
    representatives = [shape_by_id[shape_id] for shape_id in clustering.medoid_shape_ids]
    requests = [
        PairRequest(candidate, shape, evidence_stage=evidence_stage, min_samples=min_samples)
        for candidate in candidates
        for shape in representatives
    ]
    result = evaluator.evaluate(requests)
    result.apply(controller)
    return BaselineEvaluation("representative_first_no_transfer", len(requests), result)


def characterize_representative_promotions(
    clustering: ShapeClustering,
    oracle: Mapping[tuple[str, str], OracleRecord],
    *,
    candidate_hashes: Sequence[str],
    tolerance_fraction: float = 0.05,
) -> RepresentativeDiagnostics:
    if tolerance_fraction < 0.0:
        raise ValueError("representative tolerance must be nonnegative")
    unique_candidate_hashes = tuple(dict.fromkeys(candidate_hashes))
    best_by_shape: dict[str, float] = {}
    for shape_id in clustering.shape_ids:
        performances = [
            record.screening_gflops
            for candidate_hash in unique_candidate_hashes
            if (record := oracle.get((shape_id, candidate_hash))) is not None
            and record.screening_gflops is not None
            and record.screening_gflops > 0.0
        ]
        if performances:
            best_by_shape[shape_id] = max(performances)

    known_medoid_pairs = 0
    assessed = 0
    precise = 0
    missed = 0
    unavailable = 0
    regrets = []
    for cluster in clustering.clusters:
        medoid_records = [
            record
            for candidate_hash in unique_candidate_hashes
            if (record := oracle.get((cluster.medoid_shape_id, candidate_hash))) is not None
            and record.screening_gflops is not None
            and record.screening_gflops > 0.0
        ]
        known_medoid_pairs += len(medoid_records)
        if not medoid_records:
            unavailable += len(cluster.shape_ids)
            continue
        winner = max(medoid_records, key=lambda record: (record.screening_gflops or 0.0, record.candidate.hash))
        for shape_id in cluster.shape_ids:
            best = best_by_shape.get(shape_id)
            promoted = oracle.get((shape_id, winner.candidate.hash))
            if (
                best is None
                or promoted is None
                or promoted.screening_gflops is None
                or promoted.screening_gflops <= 0.0
            ):
                unavailable += 1
                continue
            regret = max(0.0, best / promoted.screening_gflops - 1.0)
            regrets.append(regret)
            assessed += 1
            if regret <= tolerance_fraction:
                precise += 1
            else:
                missed += 1
    return RepresentativeDiagnostics(
        tolerance_fraction=tolerance_fraction,
        medoid_pairs=len(unique_candidate_hashes) * len(clustering.clusters),
        known_medoid_pairs=known_medoid_pairs,
        assessed_cluster_pairs=assessed,
        precise_promotions=precise,
        missed_specialists=missed,
        unavailable_promotions=unavailable,
        promotion_precision=(precise / assessed if assessed else None),
        missed_specialist_rate=(missed / assessed if assessed else None),
        median_regret_fraction=(statistics.median(regrets) if regrets else None),
        worst_regret_fraction=(max(regrets) if regrets else None),
    )
