import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from evotensile.candidate import Candidate, Shape
from evotensile.database import BenchmarkSummary
from evotensile.search.encoding import PARAM_NAMES, candidate_to_genome, hamming_distance
from evotensile.search.evidence import ProposalEvidenceSnapshot
from evotensile.search.grid_evidence import candidate_grid_scores

DEFAULT_TRUNCATION_TAU = 0.5
DEFAULT_MIN_LINKAGE_SAMPLES = 8
DEFAULT_MAX_CLUSTERS = 8
DEFAULT_ORDINAL_BINS = 4
DEFAULT_MI_FLOOR = 1e-6

DEFAULT_ORDINAL_PARAM_NAMES = frozenset(
    {
        "DepthU",
        "GlobalSplitU",
        "VectorWidthA",
        "VectorWidthB",
        "GlobalReadVectorWidthA",
        "GlobalReadVectorWidthB",
        "StoreVectorWidth",
        "WorkGroupMapping",
        "StaggerU",
        "StaggerUStride",
        "AssertFree0ElementMultiple",
        "AssertFree1ElementMultiple",
        "AssertSummationElementMultiple",
    }
)


@dataclass(frozen=True)
class CandidateEvidence:
    candidate: Candidate
    aggregate_score: float
    coverage_fraction: float
    unresolved_shape_count: int
    samples: int


@dataclass(frozen=True)
class ScoredGenome:
    genome: tuple[int, ...]
    score: float
    candidate_hash: str | None = None
    samples: int = 1


@dataclass(frozen=True)
class LinkageModel:
    leader_genome: tuple[int, ...]
    leader_candidate_hash: str | None
    fos_groups: tuple[tuple[int, ...], ...]
    cluster_size: int
    evidence_count: int
    mi_floor: float = DEFAULT_MI_FLOOR

    def summary(self) -> dict[str, object]:
        return {
            "leader_candidate_hash": self.leader_candidate_hash,
            "cluster_size": self.cluster_size,
            "evidence_count": self.evidence_count,
            "fos_group_count": len(self.fos_groups),
            "max_fos_group_size": max((len(group) for group in self.fos_groups), default=0),
            "mi_floor": self.mi_floor,
        }


@dataclass(frozen=True)
class LinkageLearningSummary:
    enabled: bool
    model_count: int
    evidence_count: int
    selected_count: int
    fallback_reason: str | None = None


def ordinal_gene_indices(param_names: Iterable[str] = DEFAULT_ORDINAL_PARAM_NAMES) -> frozenset[int]:
    names = set(param_names)
    return frozenset(index for index, name in enumerate(PARAM_NAMES) if name in names)


def load_candidate_evidence(
    snapshot: ProposalEvidenceSnapshot,
    *,
    shapes: Sequence[Shape] | None = None,
    min_samples: int = 1,
    elite_per_shape: int = 8,
    limit: int | None = None,
) -> list[CandidateEvidence]:
    target_shape_ids = [shape.id for shape in shapes] if shapes is not None else []
    summaries_by_shape: dict[str, list[BenchmarkSummary]] = {}
    for summary in snapshot.summaries:
        if summary.samples < min_samples:
            continue
        if target_shape_ids and summary.shape_id not in target_shape_ids:
            continue
        summaries_by_shape.setdefault(summary.shape_id, []).append(summary)

    scores = candidate_grid_scores(
        summaries_by_shape,
        target_shape_ids=target_shape_ids or sorted(summaries_by_shape),
        elite_per_shape=elite_per_shape,
    )
    candidate_hashes = sorted(scores)
    candidates = {
        candidate_hash: snapshot.candidates[candidate_hash]
        for candidate_hash in candidate_hashes
        if candidate_hash in snapshot.candidates
    }
    evidence: list[CandidateEvidence] = []
    for candidate_hash, score in scores.items():
        candidate = candidates.get(candidate_hash)
        if candidate is None:
            continue
        evidence.append(
            CandidateEvidence(
                candidate=candidate,
                aggregate_score=score.generalist_score,
                coverage_fraction=score.coverage_fraction,
                unresolved_shape_count=score.unresolved_shape_count,
                samples=score.samples,
            )
        )
    evidence.sort(key=lambda item: (item.aggregate_score, -item.samples, item.candidate.hash))
    return evidence[:limit] if limit is not None else evidence


def evidence_to_scored_genomes(evidence: Sequence[CandidateEvidence]) -> list[ScoredGenome]:
    return [
        ScoredGenome(
            genome=candidate_to_genome(item.candidate),
            score=item.aggregate_score,
            candidate_hash=item.candidate.hash,
            samples=item.samples,
        )
        for item in evidence
    ]


def _sort_key(item: ScoredGenome) -> tuple[float, int, str]:
    return (item.score, -item.samples, item.candidate_hash or "")


def select_truncation_pool(
    scored_genomes: Sequence[ScoredGenome],
    *,
    truncation_tau: float = DEFAULT_TRUNCATION_TAU,
    min_samples: int = DEFAULT_MIN_LINKAGE_SAMPLES,
) -> list[ScoredGenome]:
    if truncation_tau <= 0.0 or truncation_tau > 1.0:
        raise ValueError("truncation_tau must be in (0, 1]")
    finite = [item for item in scored_genomes if math.isfinite(item.score)]
    if not finite:
        return []
    selected_count = max(1, int(len(finite) * truncation_tau))
    selected = sorted(finite, key=_sort_key)[:selected_count]
    return selected if len(selected) >= min_samples else []


def leader_clusters(
    scored_genomes: Sequence[ScoredGenome],
    *,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    hamming_threshold: int | None = None,
) -> list[list[ScoredGenome]]:
    if max_clusters <= 0:
        raise ValueError("max_clusters must be positive")
    ordered = sorted(scored_genomes, key=_sort_key)
    if not ordered:
        return []
    n_genes = len(ordered[0].genome)
    threshold = max(2, int(n_genes * 0.3)) if hamming_threshold is None else hamming_threshold
    clusters: list[list[ScoredGenome]] = []
    leaders: list[ScoredGenome] = []
    for item in ordered:
        matching_indices = [
            index for index, leader in enumerate(leaders) if hamming_distance(item.genome, leader.genome) <= threshold
        ]
        if matching_indices:
            cluster_index = min(matching_indices, key=lambda index: len(clusters[index]))
            clusters[cluster_index].append(item)
        elif len(clusters) < max_clusters:
            leaders.append(item)
            clusters.append([item])
        else:
            cluster_index = min(
                range(len(leaders)),
                key=lambda index: (hamming_distance(item.genome, leaders[index].genome), len(clusters[index])),
            )
            clusters[cluster_index].append(item)
    return clusters


def _rank_binned_values(values: Sequence[int], bins: int) -> tuple[int, ...]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    ordered_values = sorted(set(values))
    rank_by_value = {value: index / max(len(ordered_values) - 1, 1) for index, value in enumerate(ordered_values)}
    return tuple(min(int(rank_by_value[value] * bins), bins - 1) for value in values)


def _mutual_information(left_values: Sequence[int], right_values: Sequence[int]) -> float:
    if len(left_values) != len(right_values):
        raise ValueError("MI inputs must have the same length")
    n = len(left_values)
    if n == 0:
        return 0.0
    counts_left = Counter(left_values)
    counts_right = Counter(right_values)
    counts_pair = Counter(zip(left_values, right_values, strict=True))
    mi = 0.0
    for (left, right), count in counts_pair.items():
        p_pair = count / n
        p_left = counts_left[left] / n
        p_right = counts_right[right] / n
        if p_left > 0.0 and p_right > 0.0:
            mi += p_pair * math.log(p_pair / (p_left * p_right))
    return mi


def hybrid_mi_matrix(
    genomes: Sequence[tuple[int, ...]],
    *,
    ordinal_indices: frozenset[int] | None = None,
    ordinal_bins: int = DEFAULT_ORDINAL_BINS,
) -> list[list[float]]:
    if not genomes:
        return []
    n_genes = len(genomes[0])
    if any(len(genome) != n_genes for genome in genomes):
        raise ValueError("all genomes must have the same length")
    ordinal = ordinal_gene_indices() if ordinal_indices is None else ordinal_indices
    columns: list[tuple[int, ...]] = []
    for gene_index in range(n_genes):
        values = tuple(genome[gene_index] for genome in genomes)
        columns.append(_rank_binned_values(values, ordinal_bins) if gene_index in ordinal else values)
    matrix = [[0.0] * n_genes for _ in range(n_genes)]
    for left in range(n_genes):
        for right in range(left + 1, n_genes):
            mi = _mutual_information(columns[left], columns[right])
            matrix[left][right] = mi
            matrix[right][left] = mi
    return matrix


def fos_from_genomes(
    genomes: Sequence[tuple[int, ...]],
    *,
    ordinal_indices: frozenset[int] | None = None,
    ordinal_bins: int = DEFAULT_ORDINAL_BINS,
    mi_floor: float = DEFAULT_MI_FLOOR,
) -> list[tuple[int, ...]]:
    if not genomes:
        return []
    return upgma_fos(
        hybrid_mi_matrix(genomes, ordinal_indices=ordinal_indices, ordinal_bins=ordinal_bins),
        mi_floor=mi_floor,
    )


def upgma_fos(mi_matrix: Sequence[Sequence[float]], *, mi_floor: float = DEFAULT_MI_FLOOR) -> list[tuple[int, ...]]:
    n_genes = len(mi_matrix)
    fos: list[tuple[int, ...]] = [(index,) for index in range(n_genes)]
    if n_genes < 2:
        return fos
    active: dict[int, tuple[int, ...]] = {index: (index,) for index in range(n_genes)}
    next_id = n_genes
    while len(active) > 1:
        active_ids = list(active)
        best_pair: tuple[int, int] | None = None
        best_score = -float("inf")
        for left_pos, left_id in enumerate(active_ids):
            for right_id in active_ids[left_pos + 1 :]:
                left_group = active[left_id]
                right_group = active[right_id]
                score = sum(mi_matrix[left][right] for left in left_group for right in right_group) / (
                    len(left_group) * len(right_group)
                )
                if score > best_score:
                    best_score = score
                    best_pair = (left_id, right_id)
        if best_pair is None or best_score <= mi_floor:
            break
        left_id, right_id = best_pair
        merged = tuple(sorted((*active[left_id], *active[right_id])))
        del active[left_id]
        del active[right_id]
        active[next_id] = merged
        next_id += 1
        if len(merged) < n_genes:
            fos.append(merged)
    return fos


def _dedupe_groups(groups: Iterable[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    for group in groups:
        normalized = tuple(dict.fromkeys(group))
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return tuple(out)


def learn_linkage_models(
    scored_genomes: Sequence[ScoredGenome],
    *,
    truncation_tau: float = DEFAULT_TRUNCATION_TAU,
    min_samples: int = DEFAULT_MIN_LINKAGE_SAMPLES,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    hamming_threshold: int | None = None,
    ordinal_indices: frozenset[int] | None = None,
    ordinal_bins: int = DEFAULT_ORDINAL_BINS,
    mi_floor: float = DEFAULT_MI_FLOOR,
) -> tuple[list[LinkageModel], LinkageLearningSummary]:
    selected = select_truncation_pool(
        scored_genomes,
        truncation_tau=truncation_tau,
        min_samples=min_samples,
    )
    if not selected:
        return [], LinkageLearningSummary(
            enabled=False,
            model_count=0,
            evidence_count=len(scored_genomes),
            selected_count=0,
            fallback_reason="insufficient_validated_evidence",
        )

    models: list[LinkageModel] = []
    for cluster in leader_clusters(selected, max_clusters=max_clusters, hamming_threshold=hamming_threshold):
        leader = sorted(cluster, key=_sort_key)[0]
        genomes = [item.genome for item in cluster]
        if len(cluster) < 2:
            fos = [(index,) for index in range(len(leader.genome))]
        else:
            fos = fos_from_genomes(
                genomes,
                ordinal_indices=ordinal_indices,
                ordinal_bins=ordinal_bins,
                mi_floor=mi_floor,
            )
        models.append(
            LinkageModel(
                leader_genome=leader.genome,
                leader_candidate_hash=leader.candidate_hash,
                fos_groups=_dedupe_groups(fos),
                cluster_size=len(cluster),
                evidence_count=len(scored_genomes),
                mi_floor=mi_floor,
            )
        )

    return models, LinkageLearningSummary(
        enabled=bool(models),
        model_count=len(models),
        evidence_count=len(scored_genomes),
        selected_count=len(selected),
        fallback_reason=None if models else "no_linkage_clusters",
    )


def minimum_evidence_for_truncation(*, truncation_tau: float, min_samples: int) -> int:
    if truncation_tau <= 0.0 or truncation_tau > 1.0:
        raise ValueError("truncation_tau must be in (0, 1]")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    required = max(1, math.ceil(min_samples / truncation_tau))
    while int(required * truncation_tau) < min_samples:
        required += 1
    return required


def learn_linkage_models_from_snapshot(
    snapshot: ProposalEvidenceSnapshot,
    *,
    shapes: Sequence[Shape] | None = None,
    evidence_min_samples: int = 1,
    elite_per_shape: int | None = None,
    truncation_tau: float = DEFAULT_TRUNCATION_TAU,
    min_samples: int = DEFAULT_MIN_LINKAGE_SAMPLES,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    ordinal_bins: int = DEFAULT_ORDINAL_BINS,
) -> tuple[list[LinkageModel], LinkageLearningSummary]:
    evidence_limit = (
        minimum_evidence_for_truncation(truncation_tau=truncation_tau, min_samples=min_samples)
        if elite_per_shape is None
        else elite_per_shape
    )
    evidence = load_candidate_evidence(
        snapshot,
        shapes=shapes,
        min_samples=evidence_min_samples,
        elite_per_shape=evidence_limit,
    )
    return learn_linkage_models(
        evidence_to_scored_genomes(evidence),
        truncation_tau=truncation_tau,
        min_samples=min_samples,
        max_clusters=max_clusters,
        ordinal_bins=ordinal_bins,
    )


def nearest_linkage_model(genome: tuple[int, ...], models: Sequence[LinkageModel]) -> LinkageModel | None:
    if not models:
        return None
    return min(
        models, key=lambda model: (hamming_distance(genome, model.leader_genome), model.leader_candidate_hash or "")
    )
