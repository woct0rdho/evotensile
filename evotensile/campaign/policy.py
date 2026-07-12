import math
from dataclasses import dataclass, field
from typing import Literal

from evotensile.campaign.acquisition import BundleAcquisitionPolicy
from evotensile.campaign.promotion import PromotionPolicy
from evotensile.campaign.repair import RepairPolicy
from evotensile.campaign.round_controller import StagedRoundConfiguration
from evotensile.candidate import stable_hash

ArtifactScopePolicy = Literal["requested", "cluster"]
InitializationProfile = Literal["blind", "anchored"]
InitializationRegime = Literal["blind", "anchored-untuned", "anchored-tuned"]


@dataclass(frozen=True)
class CampaignPolicyConfiguration:
    name: str
    initialization_profile: InitializationProfile = "blind"
    initialization_label: str | None = None
    cluster_count: int = 16
    calibration_candidate_count: int = 0
    artifact_scope: ArtifactScopePolicy = "requested"
    round: StagedRoundConfiguration = field(default_factory=StagedRoundConfiguration)
    acquisition: BundleAcquisitionPolicy = field(default_factory=BundleAcquisitionPolicy)
    promotion: PromotionPolicy = field(default_factory=PromotionPolicy)
    repair: RepairPolicy = field(default_factory=RepairPolicy)
    singleton_acquisition_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("campaign policy name must be non-empty")
        if self.initialization_profile not in {"blind", "anchored"}:
            raise ValueError("campaign initialization profile must be blind or anchored")
        if self.initialization_profile == "anchored" and not self.initialization_label:
            raise ValueError("anchored campaign policy requires an initialization label")
        if self.initialization_profile == "blind" and self.initialization_label is not None:
            raise ValueError("blind campaign policy cannot declare an initialization label")
        if self.cluster_count <= 0:
            raise ValueError("campaign policy cluster count must be positive")
        if self.calibration_candidate_count < 0:
            raise ValueError("campaign calibration candidate count must be nonnegative")
        if self.artifact_scope not in {"requested", "cluster"}:
            raise ValueError("campaign policy artifact scope must be requested or cluster")

    @property
    def identity_hash(self) -> str:
        return stable_hash(self.to_dict(), prefix="campaign_policy_")[:24]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "initialization_profile": self.initialization_profile,
            "initialization_label": self.initialization_label,
            "cluster_count": self.cluster_count,
            "calibration_candidate_count": self.calibration_candidate_count,
            "artifact_scope": self.artifact_scope,
            "round": self.round.to_dict(),
            "acquisition": {
                "improvement_weight": self.acquisition.improvement_weight,
                "coverage_weight": self.acquisition.coverage_weight,
                "information_weight": self.acquisition.information_weight,
                "repair_weight": self.acquisition.repair_weight,
                "bundle_sizes": list(self.acquisition.bundle_sizes),
                "max_pairs": self.acquisition.max_pairs,
                "max_bundles": self.acquisition.max_bundles,
                "max_predicted_cost_s": self.acquisition.max_predicted_cost_s,
                "min_utility_per_s": self.acquisition.min_utility_per_s,
                "min_samples": self.acquisition.min_samples,
                "evidence_stage": self.acquisition.evidence_stage.value,
            },
            "promotion": {
                "neighbor_depth": self.promotion.neighbor_depth,
                "representative_finalist_count": self.promotion.representative_finalist_count,
                "max_promotions_per_shape": self.promotion.max_promotions_per_shape,
                "specialist_slots": self.promotion.specialist_slots,
                "survivor_floor": self.promotion.survivor_floor,
                "broad_candidate_slots": self.promotion.broad_candidate_slots,
                "broad_candidate_min_shapes": self.promotion.broad_candidate_min_shapes,
                "adjacent_cluster_depth": self.promotion.adjacent_cluster_depth,
                "source_near_winner_fraction": self.promotion.source_near_winner_fraction,
                "probe_survivor_regret_fraction": self.promotion.probe_survivor_regret_fraction,
                "stop_regret_fraction": self.promotion.stop_regret_fraction,
                "probe_samples": self.promotion.probe_samples,
                "main_samples": self.promotion.main_samples,
            },
            "repair": {
                "neighbor_count": self.repair.neighbor_count,
                "neighbor_quantile": self.repair.neighbor_quantile,
                "cluster_quantile": self.repair.cluster_quantile,
                "uncertainty_weight": self.repair.uncertainty_weight,
                "minimum_deficit_fraction": self.repair.minimum_deficit_fraction,
                "maximum_deficit_fraction": self.repair.maximum_deficit_fraction,
                "useful_close_fraction": self.repair.useful_close_fraction,
                "minimum_close_probability": self.repair.minimum_close_probability,
                "neighbor_candidates_per_shape": self.repair.neighbor_candidates_per_shape,
                "cluster_candidates": self.repair.cluster_candidates,
                "mutation_candidates_per_shape": self.repair.mutation_candidates_per_shape,
                "mutation_max_changed_genes": self.repair.mutation_max_changed_genes,
                "seed": self.repair.seed,
            },
            "singleton_acquisition_enabled": self.singleton_acquisition_enabled,
        }


@dataclass(frozen=True)
class CampaignRoundSchedule:
    name: str
    pair_budget_fractions: tuple[float, ...]
    repair_enabled: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.pair_budget_fractions:
            raise ValueError("campaign round schedule identity and fractions are required")
        if len(self.pair_budget_fractions) != len(self.repair_enabled):
            raise ValueError("campaign round schedule roles must match budget fractions")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.pair_budget_fractions):
            raise ValueError("campaign round budget fractions must be finite and positive")
        if not math.isclose(sum(self.pair_budget_fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("campaign round budget fractions must sum to one")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "pair_budget_fractions": list(self.pair_budget_fractions),
            "repair_enabled": list(self.repair_enabled),
        }


def _selected_round_configuration(
    *,
    repair_pairs: int,
    total_pairs: int,
    guard_s: float,
) -> StagedRoundConfiguration:
    repair_fraction = repair_pairs / total_pairs
    return StagedRoundConfiguration(
        phase_fractions=(
            ("broad", 0.35),
            ("promotion", 0.45 - repair_fraction),
            ("repair", repair_fraction),
            ("stabilization", 0.10),
            ("confirmation", 0.10),
        ),
        no_new_preparation_guard_s=guard_s,
    )


def selected_campaign_policy(
    regime: InitializationRegime,
    *,
    pair_budget: int = 385,
) -> CampaignPolicyConfiguration:
    if pair_budget < 36:
        raise ValueError("selected campaign policy pair budget must be at least 36")
    if regime == "blind":
        return CampaignPolicyConfiguration(
            name="balanced-16-requested",
            cluster_count=16,
            artifact_scope="requested",
            round=_selected_round_configuration(
                repair_pairs=12,
                total_pairs=pair_budget,
                guard_s=30.0,
            ),
            acquisition=_selected_acquisition_policy(
                coverage_weight=0.50,
                information_weight=0.10,
                pair_budget=pair_budget,
            ),
            repair=RepairPolicy(uncertainty_weight=0.0),
        )
    if regime == "anchored-untuned":
        return CampaignPolicyConfiguration(
            name="tail-16-cluster",
            initialization_profile="anchored",
            initialization_label=regime,
            cluster_count=16,
            calibration_candidate_count=4,
            artifact_scope="cluster",
            round=_selected_round_configuration(
                repair_pairs=16,
                total_pairs=pair_budget,
                guard_s=20.0,
            ),
            acquisition=_selected_acquisition_policy(
                coverage_weight=0.35,
                information_weight=0.05,
                pair_budget=pair_budget,
            ),
            repair=RepairPolicy(
                uncertainty_weight=0.0,
                maximum_deficit_fraction=0.20,
                minimum_close_probability=0.20,
            ),
        )
    if regime == "anchored-tuned":
        return CampaignPolicyConfiguration(
            name="information-20-requested",
            initialization_profile="anchored",
            initialization_label=regime,
            cluster_count=20,
            calibration_candidate_count=8,
            artifact_scope="requested",
            round=_selected_round_configuration(
                repair_pairs=8,
                total_pairs=pair_budget,
                guard_s=20.0,
            ),
            acquisition=_selected_acquisition_policy(
                coverage_weight=0.50,
                information_weight=0.25,
                pair_budget=pair_budget,
            ),
            repair=RepairPolicy(uncertainty_weight=0.10),
        )
    raise ValueError(f"unknown campaign initialization regime: {regime}")


def _selected_acquisition_policy(
    *,
    coverage_weight: float,
    information_weight: float,
    pair_budget: int,
) -> BundleAcquisitionPolicy:
    return BundleAcquisitionPolicy(
        improvement_weight=1.0,
        coverage_weight=coverage_weight,
        information_weight=information_weight,
        bundle_sizes=(1, 2, 4, 8, 16),
        max_pairs=pair_budget,
        max_bundles=96,
        max_predicted_cost_s=300.0,
    )


def selected_campaign_round_schedule(regime: InitializationRegime) -> CampaignRoundSchedule:
    if regime == "blind":
        return CampaignRoundSchedule(
            name="fixed",
            pair_budget_fractions=(0.5, 0.5),
            repair_enabled=(True, True),
        )
    if regime in {"anchored-untuned", "anchored-tuned"}:
        return CampaignRoundSchedule(
            name="role-specialized",
            pair_budget_fractions=(0.6, 0.4),
            repair_enabled=(False, True),
        )
    raise ValueError(f"unknown campaign initialization regime: {regime}")
