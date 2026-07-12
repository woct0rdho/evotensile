import json
import math
from dataclasses import asdict, dataclass, field
from typing import TypedDict

from evotensile.adaptive_retime import AdaptivePolicy, ProbePolicy
from evotensile.candidate import Candidate, stable_hash
from evotensile.proposal import FamilyQDPolicy
from evotensile.protocol import BenchmarkProtocol
from evotensile.search.campaign_control import ProposalEvent
from evotensile.search.screening_stabilize import ScreeningStabilizationPolicy

CAMPAIGN_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "EVOTENSILE_APU_LOCK_PATH",
    "HIPBLASLT_TENSILE_LIBPATH",
    "HIP_PATH",
    "HIP_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "LD_LIBRARY_PATH",
    "OPENBLAS_NUM_THREADS",
    "PATH",
    "PYTHONHASHSEED",
    "ROCM_PATH",
    "ROCR_VISIBLE_DEVICES",
    "TENSILELITE_ROOT",
)


@dataclass(frozen=True)
class CampaignConfiguration:
    seed: int
    shape_id: str
    profile_name: str
    problem_type_hash: str
    runner_bin: str
    runner_fingerprint: str
    tensilelite_bin: str
    tensilelite_fingerprint: str
    implementation_fingerprint: str
    environment: tuple[tuple[str, str], ...]
    time_budget_s: float
    hot_reserve_s: float
    max_feedback_rounds: int
    early_stop_on_convergence: bool
    build_timeout_s: float
    runner_timeout_s: float
    screening_protocol: BenchmarkProtocol
    hot_protocol: BenchmarkProtocol
    prepare_workers: int
    prepare_wave_batches: int
    validation_workers: int
    surrogate_jobs: int
    compute_unit_count: int
    workgroup_processor_count: int
    compute_units_per_workgroup_processor: int
    adaptive_policy: AdaptivePolicy = field(default_factory=lambda: AdaptivePolicy(max_rounds=0))
    probe_policy: ProbePolicy = field(default_factory=ProbePolicy)
    stabilization_policy: ScreeningStabilizationPolicy = field(default_factory=ScreeningStabilizationPolicy)
    cold_candidates: int = 48
    cold_pool_multiplier: int = 8
    feedback_candidates: int = 24
    feedback_random: int = 4
    feedback_semantic_mutation: int = 6
    feedback_de: int = 4
    feedback_gomea: int = 10
    feedback_pool_multiplier: int = 8
    surrogate_min_evidence: int = 24
    elite_count: int = 32
    candidate_batch_size: int = 1
    shape_batch_size: int = 1
    min_samples: int = 2
    compile_threads: int = 1
    keep_going: bool = True
    compile_cache: bool = True
    cost_aware_scheduling: bool = True
    leader_stabilization: bool = True
    island_count: int = 2
    island_isolation_rounds: int = 6
    island_elites: int = 16
    plateau_patience: int = 3
    plateau_min_improvement_fraction: float = 0.005
    restart_max_mean_hamming: float = 5.0
    convergence_patience: int = 8
    convergence_minimum_improvement_fraction: float = 0.0025
    convergence_maximum_mean_hamming: float = 4.0
    hot_top_k: int = 8
    mutation_rate: float = FamilyQDPolicy.mutation_rate
    crossover_rate: float = FamilyQDPolicy.crossover_rate
    random_gene_rate: float = FamilyQDPolicy.random_gene_rate
    linkage_truncation_tau: float = FamilyQDPolicy.linkage_truncation_tau
    linkage_min_samples: int = FamilyQDPolicy.linkage_min_samples
    linkage_max_clusters: int = FamilyQDPolicy.linkage_max_clusters
    linkage_ordinal_bins: int = FamilyQDPolicy.linkage_ordinal_bins
    transfer_shape_count: int = 0
    transfer_per_shape: int = 0
    adaptive_operators: bool = True
    adaptive_group_credit: bool = True
    micro_exhaustive_neighborhoods: bool = True
    adaptive_donor_selection: bool = True
    cost_aware_operator_credit: bool = True
    covering_cold_start: bool = True
    singleton_acquisition_enabled: bool = True
    singleton_information_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.time_budget_s <= 0.0:
            raise ValueError("campaign time budget must be positive")
        if not 0.0 <= self.hot_reserve_s < self.time_budget_s:
            raise ValueError("hot reserve must be non-negative and smaller than the time budget")
        if self.max_feedback_rounds < 0:
            raise ValueError("maximum feedback rounds must be non-negative")
        if self.feedback_candidates != (
            self.feedback_random + self.feedback_semantic_mutation + self.feedback_de + self.feedback_gomea
        ):
            raise ValueError("feedback candidate count must equal the operator budget sum")
        if self.island_count <= 0:
            raise ValueError("island count must be positive")
        if not 0 < self.island_elites <= self.elite_count:
            raise ValueError("island elites must be positive and no greater than merged elites")
        if self.candidate_batch_size <= 0 or self.shape_batch_size <= 0:
            raise ValueError("campaign batch sizes must be positive")
        if (
            self.prepare_workers <= 0
            or self.prepare_wave_batches <= 0
            or self.validation_workers <= 0
            or self.surrogate_jobs <= 0
        ):
            raise ValueError("campaign worker counts must be positive")
        if self.compute_unit_count <= 0 or self.workgroup_processor_count <= 0:
            raise ValueError("campaign hardware execution-unit counts must be positive")
        if self.compute_units_per_workgroup_processor <= 0:
            raise ValueError("campaign compute units per work-group processor must be positive")
        if self.compute_unit_count != (self.workgroup_processor_count * self.compute_units_per_workgroup_processor):
            raise ValueError("campaign compute-unit and work-group-processor topology is inconsistent")
        if self.compile_threads <= 0:
            raise ValueError("campaign compile threads must be positive")
        if self.build_timeout_s <= 0.0 or self.runner_timeout_s <= 0.0:
            raise ValueError("campaign subprocess timeouts must be positive")
        if self.screening_protocol.role != "main" or self.hot_protocol.role != "main":
            raise ValueError("screening and hot protocols must use the main benchmark role")
        if self.screening_protocol.num_elements_to_validate == 0:
            raise ValueError("screening protocol must retain correctness validation")
        if self.hot_protocol.num_elements_to_validate != 0:
            raise ValueError("hot protocol must disable repeated validation")
        if self.hot_top_k <= 0:
            raise ValueError("hot finalist count must be positive")
        if not math.isfinite(self.singleton_information_weight) or self.singleton_information_weight < 0.0:
            raise ValueError("singleton information weight must be finite and nonnegative")

    def to_dict(self) -> dict[str, object]:
        return json.loads(json.dumps(asdict(self), sort_keys=True))

    @property
    def identity_hash(self) -> str:
        return stable_hash(self.to_dict(), prefix="campaign_")[:25]


@dataclass(frozen=True)
class RoundProposal:
    selected: tuple[Candidate, ...]
    active: tuple[Candidate, ...]
    archive: tuple[Candidate, ...]
    events: tuple[ProposalEvent, ...]


class ProposalArgs(TypedDict):
    num_random: int
    elite_count: int
    local_count: int
    de_count: int
    gomea_count: int
    adaptive_operators: bool
    surrogate_pool_multiplier: int
    covering_cold_start: bool
    adaptive_group_credit: bool
    micro_exhaustive_neighborhoods: bool
    adaptive_donor_selection: bool
    cost_aware_operator_credit: bool
    surrogate_min_evidence: int
