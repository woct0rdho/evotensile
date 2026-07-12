#!/usr/bin/env python3

import argparse
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

from evotensile.adaptive_retime import AdaptivePolicy, ProbePolicy
from evotensile.artifacts import load_artifact_mappings
from evotensile.campaign.acquisition import BundleAcquisitionPolicy, BundleCostModel, plan_candidate_bundles
from evotensile.campaign.controller import CampaignControllerState
from evotensile.campaign.evaluator import PairEvaluationOutcome, RealEvaluator, RealEvaluatorContext
from evotensile.campaign.repair import (
    RepairPolicy,
    ShapeRepairDeficit,
    build_repair_candidate_pool,
    repair_pair_close_probabilities,
)
from evotensile.candidate import Candidate, Shape
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.scheduling.models import EvidenceStage, PairRequest
from evotensile.search.evidence import load_proposal_evidence_snapshot
from evotensile.search.pair_model import ContextualPairModel, PairModelConfiguration, PairPrediction
from evotensile.search.replay import OracleRecord, load_db_oracle_matrix
from evotensile.search.shape_clustering import ShapeClusteringConfiguration, cluster_shapes
from evotensile.search.trust_region import interaction_grid_candidates
from evotensile.shapes import pilot_100_shapes

DEFAULT_DB = Path("out/grid100_production_search_20260712.sqlite")
DEFAULT_INITIAL_DB = Path("out/grid100_compatible_20260712.sqlite")
DEFAULT_CAMPAIGN_ROOT = Path("out/grid100_production_search_20260712")
DEFAULT_INCUMBENT_DEPLOYMENT = DEFAULT_CAMPAIGN_ROOT / "finalization_v3/deployment_0.000.json"
DEFAULT_INCUMBENT_REPORT = DEFAULT_CAMPAIGN_ROOT / "finalization_v3/report.json"


class DeploymentPayload(TypedDict):
    assignments: dict[str, str]
    confirmed_performance: dict[str, float]


class FinalizationReportPayload(TypedDict):
    zero_tolerance_improvement: dict[str, object]
    historical_reference_delta: dict[str, object]


class IntegratedRepairEvidencePayload(TypedDict):
    shape_id: str
    candidate_hash: str
    score: float
    headroom_fraction: float
    historical_shortfall_fraction: float
    singleton_bonus: float
    low_gain_bonus: float


def _initialize_database(path: Path, source: Path) -> bool:
    if path.exists():
        return False
    if not source.exists():
        raise FileNotFoundError(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, path)
    return True


def _oracle_outcomes(
    oracle: Mapping[tuple[str, str], OracleRecord],
    *,
    shape_by_id: Mapping[str, Shape],
) -> tuple[PairEvaluationOutcome, ...]:
    outcomes = []
    for (shape_id, _), record in sorted(oracle.items()):
        outcomes.append(
            PairEvaluationOutcome(
                request=PairRequest(
                    record.candidate,
                    shape_by_id[shape_id],
                    evidence_stage=EvidenceStage.SCREENING,
                    min_samples=1,
                ),
                provenance="compatible-db",
                source_ref=record.source_artifact,
                status=record.status,
                known=True,
                disclosed=True,
                samples=1 if record.screening_gflops is not None else 0,
                performance=record.screening_gflops,
            )
        )
    return tuple(outcomes)


def _seed_controller(
    outcomes: Sequence[PairEvaluationOutcome],
    *,
    shapes: Sequence[Shape],
    time_budget_s: float,
    deployment: DeploymentPayload | None = None,
) -> CampaignControllerState:
    controller = CampaignControllerState(
        shape_ids=tuple(shape.id for shape in shapes),
        time_budget_s=time_budget_s,
        session_started_at=time.monotonic(),
    )
    for outcome in outcomes:
        controller.record_query(outcome.request.shape.id, outcome.request.candidate.hash, known=True)
        if deployment is None:
            controller.disclose(
                outcome.request.shape.id,
                outcome.request.candidate.hash,
                performance=outcome.performance,
            )
    if deployment is not None:
        if set(deployment["assignments"]) != set(controller.shape_ids):
            raise ValueError("incumbent deployment must cover the practical-round shapes")
        for shape_id, candidate_hash in deployment["assignments"].items():
            controller.disclose(
                shape_id,
                candidate_hash,
                performance=deployment["confirmed_performance"][shape_id],
            )
    return controller


def _integrated_repair_deficits(
    deployment: DeploymentPayload,
    report: FinalizationReportPayload,
    *,
    target_count: int,
) -> tuple[dict[str, ShapeRepairDeficit], tuple[IntegratedRepairEvidencePayload, ...]]:
    if target_count <= 0:
        return {}, ()
    assignments = deployment["assignments"]
    performance = deployment["confirmed_performance"]
    coverage = Counter(assignments.values())
    fresh_gains = cast(dict[str, float], report["zero_tolerance_improvement"]["per_shape"])
    historical_deltas = cast(dict[str, float], report["historical_reference_delta"]["per_shape"])
    ranked = []
    for shape_id, candidate_hash in assignments.items():
        historical_shortfall = max(0.0, -historical_deltas[shape_id])
        singleton_bonus = 0.03 if coverage[candidate_hash] == 1 else 0.0
        low_gain_bonus = max(0.0, 0.01 - fresh_gains[shape_id])
        score = historical_shortfall + singleton_bonus + low_gain_bonus
        if score > 0.0:
            ranked.append((score, shape_id, candidate_hash, historical_shortfall, singleton_bonus, low_gain_bonus))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    deficits = {}
    evidence: list[IntegratedRepairEvidencePayload] = []
    for score, shape_id, candidate_hash, historical_shortfall, singleton_bonus, low_gain_bonus in ranked[:target_count]:
        headroom_fraction = min(0.15, max(0.03, score))
        incumbent_performance = performance[shape_id]
        deficit_log = math.log1p(headroom_fraction)
        deficits[shape_id] = ShapeRepairDeficit(
            shape_id=shape_id,
            incumbent_candidate_hash=candidate_hash,
            incumbent_performance=incumbent_performance,
            reference_target=incumbent_performance * (1.0 + headroom_fraction),
            neighbor_target=None,
            cluster_target=None,
            uncertainty_log=0.0,
            evidence_target=incumbent_performance * (1.0 + headroom_fraction),
            raw_deficit_log=deficit_log,
            capped_deficit_log=deficit_log,
        )
        evidence.append(
            {
                "shape_id": shape_id,
                "candidate_hash": candidate_hash,
                "score": score,
                "headroom_fraction": headroom_fraction,
                "historical_shortfall_fraction": historical_shortfall,
                "singleton_bonus": singleton_bonus,
                "low_gain_bonus": low_gain_bonus,
            }
        )
    return deficits, tuple(evidence)


def _mark_registered_artifacts(
    db: EvoTensileDB,
    controller: CampaignControllerState,
    *,
    candidate_hashes: Sequence[str],
    shape_ids: Sequence[str],
) -> None:
    mappings = load_artifact_mappings(
        db,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        candidate_hashes=list(candidate_hashes),
        shape_ids=list(shape_ids),
    )
    by_candidate: dict[str, list[str]] = defaultdict(list)
    for shape_id, candidate_hash in mappings:
        by_candidate[candidate_hash].append(shape_id)
    for candidate_hash, prepared_shape_ids in by_candidate.items():
        controller.record_prepared(candidate_hash, prepared_shape_ids)


def _candidate_targets(
    controller: CampaignControllerState,
    outcomes: Sequence[PairEvaluationOutcome],
    *,
    shapes: Sequence[Shape],
    parent_count: int,
    near_incumbent_fraction: float,
    max_target_shapes: int,
    requested_parent_hashes: Sequence[str] = (),
) -> tuple[list[Candidate], dict[str, tuple[Shape, ...]], dict[str, int]]:
    performance_by_pair = {
        outcome.key: outcome.performance
        for outcome in outcomes
        if outcome.performance is not None and outcome.performance > 0.0
    }
    candidate_by_hash = {outcome.request.candidate.hash: outcome.request.candidate for outcome in outcomes}
    winner_counts = Counter(incumbent.candidate_hash for incumbent in controller.incumbents.values())
    if requested_parent_hashes:
        parent_hashes = list(dict.fromkeys(requested_parent_hashes))
        missing = [candidate_hash for candidate_hash in parent_hashes if candidate_hash not in candidate_by_hash]
        if missing:
            raise ValueError(f"interaction parents are unavailable: {missing}")
    else:
        parent_hashes = sorted(
            winner_counts,
            key=lambda candidate_hash: (-winner_counts[candidate_hash], candidate_hash),
        )[:parent_count]
    parents = [candidate_by_hash[candidate_hash] for candidate_hash in parent_hashes]
    targets = {}
    for parent in parents:
        rows = []
        for shape in shapes:
            incumbent = controller.incumbents[shape.id]
            parent_performance = (
                incumbent.performance
                if incumbent.candidate_hash == parent.hash
                else performance_by_pair.get((shape.id, parent.hash))
            )
            if parent_performance is None:
                continue
            ratio = parent_performance / incumbent.performance
            if ratio + 1e-12 < 1.0 - near_incumbent_fraction:
                continue
            rows.append(
                (
                    int(incumbent.candidate_hash == parent.hash),
                    ratio,
                    shape.id,
                    shape,
                )
            )
        rows.sort(key=lambda row: (-row[0], -row[1], row[2]))
        selected = tuple(row[3] for row in rows[:max_target_shapes])
        if selected:
            targets[parent.hash] = selected
    return [parent for parent in parents if parent.hash in targets], targets, dict(winner_counts)


def _interaction_pool(
    parents: Sequence[Candidate],
    targets_by_parent: Mapping[str, Sequence[Shape]],
    *,
    profile: str,
    store_batch_values: Sequence[int],
    known_candidate_hashes: set[str],
) -> tuple[Candidate, ...]:
    if profile == "store":
        parameter_values: Mapping[str, Sequence[object]] = {
            "ScheduleIterAlg": (2, 3),
            "StorePriorityOpt": (True, False),
            "NumElementsPerBatchStore": tuple(store_batch_values),
            "StoreVectorWidth": (-1, 1, 2),
        }
        repair_linked = False
        max_changed_genes = 4
    elif profile == "staging":
        parameter_values = {
            "DepthU": (16, 32, 64),
            "PrefetchGlobalRead": (1, 2),
            "PrefetchLocalRead": (0, 1),
            "1LDSBuffer": (0, 1),
        }
        repair_linked = True
        max_changed_genes = 8
    elif profile == "mapping":
        parameter_values = {
            "WorkGroupMapping": (4, 5, 8, 16),
            "StaggerU": (0, 8, 16, 32, 64),
            "StaggerUMapping": (0, 1),
            "SourceSwap": (True, False),
        }
        repair_linked = False
        max_changed_genes = 4
    elif profile == "vector":
        parameter_values = {
            "VectorWidthA": (1, 2),
            "VectorWidthB": (1, 2, 4),
            "GlobalReadVectorWidthA": (4, 8),
            "GlobalReadVectorWidthB": (2, 4, 8),
        }
        repair_linked = True
        max_changed_genes = 8
    elif profile == "lds":
        parameter_values = {
            "LdsPadA": (0, 4, 8, 16),
            "LdsPadB": (0, 4, 8, 16),
        }
        repair_linked = False
        max_changed_genes = 2
    else:
        raise ValueError(f"unknown interaction profile: {profile}")

    candidates: dict[str, Candidate] = {}
    for parent in parents:
        generated = interaction_grid_candidates(
            parent,
            parameter_values=parameter_values,
            target_shapes=targets_by_parent[parent.hash],
            max_changed_genes=max_changed_genes,
            repair_linked=repair_linked,
            exclude=known_candidate_hashes | set(candidates),
            source=f"grid100-{profile}-interaction",
        )
        candidates.update((candidate.hash, candidate) for candidate in generated)
    return tuple(candidates.values())


def _promotion_pool(
    specifications: Sequence[str],
    *,
    candidate_by_hash: Mapping[str, Candidate],
    controller: CampaignControllerState,
    outcomes: Sequence[PairEvaluationOutcome],
    shapes: Sequence[Shape],
    parent_floor: float,
    max_target_shapes: int,
) -> tuple[tuple[Candidate, ...], dict[str, tuple[Shape, ...]]]:
    performance_by_pair = {
        outcome.key: outcome.performance
        for outcome in outcomes
        if outcome.performance is not None and outcome.performance > 0.0
    }
    candidates = []
    targets_by_candidate = {}
    for specification in specifications:
        try:
            candidate_hash, parent_hash = specification.split(":", 1)
        except ValueError as exc:
            raise ValueError("promotion specifications must use CANDIDATE_HASH:PARENT_HASH") from exc
        if candidate_hash not in candidate_by_hash:
            raise ValueError(f"promotion candidate is unavailable: {candidate_hash}")
        if parent_hash not in candidate_by_hash:
            raise ValueError(f"promotion parent is unavailable: {parent_hash}")
        candidate = Candidate(
            params=candidate_by_hash[candidate_hash].canonical_params(),
            source="measured-promotion",
            parent_hashes=(parent_hash,),
            proposal_metadata={"promotion_parent_hash": parent_hash},
        )
        rows = []
        for shape in shapes:
            if (shape.id, candidate_hash) in controller.queried_pairs:
                continue
            incumbent = controller.incumbents[shape.id]
            parent_performance = (
                incumbent.performance
                if incumbent.candidate_hash == parent_hash
                else performance_by_pair.get((shape.id, parent_hash))
            )
            if parent_performance is None:
                continue
            ratio = parent_performance / incumbent.performance
            if ratio + 1e-12 < parent_floor:
                continue
            rows.append((ratio, shape.id, shape))
        rows.sort(key=lambda row: (-row[0], row[1]))
        target_shapes = tuple(row[2] for row in rows[:max_target_shapes])
        if not target_shapes:
            continue
        candidates.append(candidate)
        targets_by_candidate[candidate.hash] = target_shapes
    return tuple(candidates), targets_by_candidate


def _candidate_prediction_score(
    predictions: Sequence[PairPrediction],
    *,
    controller: CampaignControllerState,
    minimum_gain_fraction: float,
    repair_utility_by_pair: Mapping[tuple[str, str], float] | None = None,
) -> tuple[float, float, float, float]:
    probabilities = []
    predicted_gains = []
    for prediction in predictions:
        incumbent = controller.incumbents[prediction.shape_id].performance
        probabilities.append(
            prediction.probability_of_improvement(
                incumbent_performance=incumbent,
                minimum_gain_fraction=minimum_gain_fraction,
            )
        )
        predicted = prediction.predicted_performance
        predicted_gains.append(-math.inf if predicted is None else predicted / incumbent - 1.0)
    ranked_probabilities = sorted(probabilities, reverse=True)
    repair_utility = repair_utility_by_pair or {}
    return (
        ranked_probabilities[0],
        sum(ranked_probabilities[: min(3, len(ranked_probabilities))]),
        max(predicted_gains),
        max(
            (repair_utility.get((prediction.shape_id, prediction.candidate_hash), 0.0) for prediction in predictions),
            default=0.0,
        ),
    )


def _parent_diverse_selection(
    ranked_candidates: Sequence[Candidate],
    *,
    general_candidate_hashes: set[str],
    limit: int,
) -> tuple[Candidate, ...]:
    reserved_hashes: set[str] = set()
    reserved_parents: set[str] = set()
    for candidate in ranked_candidates:
        if candidate.hash not in general_candidate_hashes or not candidate.parent_hashes:
            continue
        parent_hash = candidate.parent_hashes[0]
        if parent_hash in reserved_parents:
            continue
        reserved_hashes.add(candidate.hash)
        reserved_parents.add(parent_hash)
        if len(reserved_hashes) == limit:
            break
    selected_hashes = set(reserved_hashes)
    for candidate in ranked_candidates:
        if len(selected_hashes) == limit:
            break
        selected_hashes.add(candidate.hash)
    return tuple(candidate for candidate in ranked_candidates if candidate.hash in selected_hashes)


def _shortlist_candidates(
    candidates: Sequence[Candidate],
    predictions: Sequence[PairPrediction],
    *,
    controller: CampaignControllerState,
    limit: int,
    minimum_gain_fraction: float,
    general_candidate_hashes: set[str],
    repair_utility_by_pair: Mapping[tuple[str, str], float] | None = None,
) -> tuple[tuple[Candidate, ...], tuple[PairPrediction, ...], dict[str, tuple[float, float, float, float]]]:
    predictions_by_candidate: dict[str, list[PairPrediction]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_candidate[prediction.candidate_hash].append(prediction)
    scores = {
        candidate.hash: _candidate_prediction_score(
            predictions_by_candidate[candidate.hash],
            controller=controller,
            minimum_gain_fraction=minimum_gain_fraction,
            repair_utility_by_pair=repair_utility_by_pair,
        )
        for candidate in candidates
    }
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -(scores[candidate.hash][0] + scores[candidate.hash][3]),
            -scores[candidate.hash][3],
            -scores[candidate.hash][1],
            -scores[candidate.hash][2],
            candidate.hash,
        ),
    )
    selected = _parent_diverse_selection(
        ranked,
        general_candidate_hashes=general_candidate_hashes,
        limit=limit,
    )
    selected_hashes = {candidate.hash for candidate in selected}
    return (
        tuple(selected),
        tuple(prediction for prediction in predictions if prediction.candidate_hash in selected_hashes),
        scores,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _update_manifest(
    path: Path,
    *,
    database: Path,
    initial_database: Path,
    initialized: bool,
    round_summary: Mapping[str, object] | None = None,
) -> None:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("campaign manifest must contain one JSON object")
    else:
        payload = {
            "database": str(database),
            "initial_database": str(initial_database),
            "created_at": time.time(),
            "rounds": [],
        }
    if initialized:
        payload["initialized_at"] = time.time()
    if round_summary is not None:
        rounds = payload.setdefault("rounds", [])
        if not isinstance(rounds, list):
            raise ValueError("campaign manifest rounds must be a list")
        rounds.append(dict(round_summary))
    _write_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--initialize-from", type=Path, default=DEFAULT_INITIAL_DB)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--incumbent-deployment", type=Path, default=DEFAULT_INCUMBENT_DEPLOYMENT)
    parser.add_argument("--incumbent-report", type=Path, default=DEFAULT_INCUMBENT_REPORT)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--strategy", choices=("interaction", "promotion"), default="interaction")
    parser.add_argument(
        "--promote",
        action="append",
        default=[],
        metavar="CANDIDATE_HASH:PARENT_HASH",
        help="Measured candidate and its comparison parent for promotion rounds",
    )
    parser.add_argument("--promotion-parent-floor", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--parent-count", type=int, default=16)
    parser.add_argument(
        "--interaction-parent",
        action="append",
        default=[],
        metavar="CANDIDATE_HASH",
        help="Restrict interaction generation to named measured parents",
    )
    parser.add_argument(
        "--interaction-profile",
        choices=("store", "staging", "mapping", "vector", "lds"),
        default="store",
    )
    parser.add_argument(
        "--store-batch-values",
        type=int,
        nargs="+",
        default=(0, 8, 10, 12, 14, 16, 20, 24, 32),
    )
    parser.add_argument("--near-incumbent-fraction", type=float, default=0.03)
    parser.add_argument("--max-target-shapes", type=int, default=12)
    parser.add_argument("--candidate-pool-limit", type=int, default=128)
    parser.add_argument("--max-pairs", type=int, default=48)
    parser.add_argument("--max-bundles", type=int, default=16)
    parser.add_argument("--soft-budget-s", type=float, default=300.0)
    parser.add_argument("--minimum-model-gain", type=float, default=0.005)
    parser.add_argument("--information-weight", type=float, default=0.03)
    parser.add_argument("--integrated-repair-targets", type=int, default=0)
    parser.add_argument("--integrated-repair-weight", type=float, default=0.0)
    parser.add_argument("--integrated-repair-mutations", type=int, default=4)
    parser.add_argument(
        "--surrogate-jobs",
        type=int,
        default=DEFAULT_PROFILE.default_surrogate_jobs,
        help="Parallel model jobs; use -1 for all available CPUs",
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.parent_count <= 0 or args.max_target_shapes <= 0 or args.candidate_pool_limit <= 0:
        raise ValueError("parent, target-shape, and candidate-pool limits must be positive")
    if not args.store_batch_values:
        raise ValueError("store-batch interaction grid must not be empty")
    if args.max_pairs <= 0 or args.max_bundles <= 0 or args.soft_budget_s <= 0.0:
        raise ValueError("round capacities and soft budget must be positive")
    if not 0.0 <= args.near_incumbent_fraction < 1.0:
        raise ValueError("near-incumbent fraction must be in [0, 1)")
    if not 0.0 <= args.promotion_parent_floor <= 1.0:
        raise ValueError("promotion parent floor must be in [0, 1]")
    if args.strategy == "promotion" and not args.promote:
        raise ValueError("promotion strategy requires at least one --promote specification")
    if args.strategy == "promotion" and args.interaction_parent:
        raise ValueError("--interaction-parent is only valid with the interaction strategy")
    if args.strategy == "interaction" and args.promote:
        raise ValueError("--promote is only valid with the promotion strategy")
    if args.minimum_model_gain < 0.0 or args.information_weight < 0.0:
        raise ValueError("model gain and information weight must be nonnegative")
    if args.integrated_repair_targets < 0 or args.integrated_repair_mutations < 0:
        raise ValueError("integrated repair counts must be nonnegative")
    if args.integrated_repair_weight < 0.0 or args.surrogate_jobs == 0:
        raise ValueError("integrated repair weight must be nonnegative and surrogate jobs nonzero")

    round_dir = args.campaign_root / args.round_id
    if round_dir.exists():
        raise FileExistsError(round_dir)
    round_dir.mkdir(parents=True)
    manifest_path = args.db.with_name(f"{args.db.stem}_manifest.json")
    initialized = _initialize_database(args.db, args.initialize_from)
    _update_manifest(
        manifest_path,
        database=args.db,
        initial_database=args.initialize_from,
        initialized=initialized,
    )

    shapes = pilot_100_shapes()
    shape_by_id = {shape.id: shape for shape in shapes}
    oracle = load_db_oracle_matrix(
        args.db,
        shapes=shapes,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
    )
    outcomes = _oracle_outcomes(oracle, shape_by_id=shape_by_id)
    deployment = cast(
        DeploymentPayload,
        json.loads(args.incumbent_deployment.read_text(encoding="utf-8")),
    )
    incumbent_report = cast(
        FinalizationReportPayload,
        json.loads(args.incumbent_report.read_text(encoding="utf-8")),
    )
    controller = _seed_controller(
        outcomes,
        shapes=shapes,
        time_budget_s=args.soft_budget_s,
        deployment=deployment,
    )
    before_incumbents = {
        shape_id: {
            "candidate_hash": incumbent.candidate_hash,
            "performance": incumbent.performance,
        }
        for shape_id, incumbent in controller.incumbents.items()
    }
    candidate_by_hash = {record.candidate.hash: record.candidate for record in oracle.values()}
    db = EvoTensileDB.connect(
        args.db,
        environment_compatibility_tag=DEFAULT_PROFILE.environment_compatibility_tag,
    )
    promotion_hashes = {
        candidate_hash
        for specification in args.promote
        for candidate_hash in specification.split(":", 1)
        if ":" in specification
    }
    if promotion_hashes:
        candidate_by_hash.update(
            (candidate.hash, candidate) for candidate in db.get_candidates(sorted(promotion_hashes))
        )
    _mark_registered_artifacts(
        db,
        controller,
        candidate_hashes=tuple(candidate_by_hash),
        shape_ids=tuple(shape_by_id),
    )
    parents: list[Candidate] = []
    winner_counts: dict[str, int] = {}
    generated_candidate_count = 0
    if args.strategy == "interaction":
        parents, targets_by_parent, winner_counts = _candidate_targets(
            controller,
            outcomes,
            shapes=shapes,
            parent_count=args.parent_count,
            near_incumbent_fraction=args.near_incumbent_fraction,
            max_target_shapes=args.max_target_shapes,
            requested_parent_hashes=args.interaction_parent,
        )
        pool = _interaction_pool(
            parents,
            targets_by_parent,
            profile=args.interaction_profile,
            store_batch_values=args.store_batch_values,
            known_candidate_hashes=set(candidate_by_hash),
        )
        targets_by_candidate = {
            candidate.hash: tuple(targets_by_parent[candidate.parent_hashes[0]]) for candidate in pool
        }
        generated_candidate_count = len(pool)
        strategy_label = f"incumbent-{args.interaction_profile}-interaction-trust-region"
    else:
        pool, targets_by_candidate = _promotion_pool(
            args.promote,
            candidate_by_hash=candidate_by_hash,
            controller=controller,
            outcomes=outcomes,
            shapes=shapes,
            parent_floor=args.promotion_parent_floor,
            max_target_shapes=args.max_target_shapes,
        )
        strategy_label = "measured-candidate-parent-competitive-promotion"
    if not pool:
        raise ValueError(f"{args.strategy} strategy produced no candidate-shape opportunities")

    repair_deficits, repair_evidence = _integrated_repair_deficits(
        deployment,
        incumbent_report,
        target_count=args.integrated_repair_targets,
    )
    repair_policy = RepairPolicy(
        neighbor_count=8,
        neighbor_candidates_per_shape=3,
        cluster_candidates=4,
        mutation_candidates_per_shape=args.integrated_repair_mutations,
        mutation_max_changed_genes=2,
        uncertainty_weight=0.0,
        minimum_close_probability=0.10,
        seed=args.seed + 2,
    )
    repair_pool = None
    repair_candidate_hashes: set[str] = set()
    general_candidate_hashes = {candidate.hash for candidate in pool}
    if repair_deficits:
        clustering = cluster_shapes(
            shapes,
            ShapeClusteringConfiguration(
                workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
                cluster_count=16,
            ),
        )
        repair_pool = build_repair_candidate_pool(
            controller,
            shapes=shapes,
            clustering=clustering,
            deficits=repair_deficits,
            observations=outcomes,
            candidate_catalog=candidate_by_hash,
            policy=repair_policy,
        )
        repair_candidate_hashes = {origin.candidate_hash for origin in repair_pool.origins}
        merged_pool = {candidate.hash: candidate for candidate in pool}
        merged_targets = {
            candidate_hash: {shape.id: shape for shape in target_shapes}
            for candidate_hash, target_shapes in targets_by_candidate.items()
        }
        for candidate, shape in repair_pool.prediction_requests(shapes):
            merged_pool.setdefault(candidate.hash, candidate)
            merged_targets.setdefault(candidate.hash, {})[shape.id] = shape
        pool = tuple(merged_pool.values())
        targets_by_candidate = {
            candidate_hash: tuple(targets.values()) for candidate_hash, targets in merged_targets.items()
        }

    model = ContextualPairModel(
        workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
        configuration=PairModelConfiguration(
            n_estimators=192,
            min_performance_rows=24,
            seed=args.seed,
            jobs=args.surrogate_jobs,
        ),
    )
    fit_summary = model.fit(outcomes)
    prediction_requests = [(candidate, shape) for candidate in pool for shape in targets_by_candidate[candidate.hash]]
    all_predictions = model.predict(prediction_requests)
    repair_probabilities = repair_pair_close_probabilities(
        controller,
        deficits=repair_deficits,
        predictions=all_predictions,
        policy=repair_policy,
    )
    repair_utility_by_pair = {
        (shape_id, candidate_hash): repair_deficits[shape_id].capped_deficit_log * probability
        for (shape_id, candidate_hash), probability in repair_probabilities.items()
    }
    candidates, predictions, prediction_scores = _shortlist_candidates(
        pool,
        all_predictions,
        controller=controller,
        limit=args.candidate_pool_limit,
        minimum_gain_fraction=args.minimum_model_gain,
        general_candidate_hashes=general_candidate_hashes,
        repair_utility_by_pair=repair_utility_by_pair,
    )

    evidence = load_proposal_evidence_snapshot(
        db,
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        shapes=shapes,
    )
    shapes_by_candidate: dict[str, list[Shape]] = defaultdict(list)
    for shape_id, candidate_hash in oracle:
        shapes_by_candidate[candidate_hash].append(shape_by_id[shape_id])
    cost_model = BundleCostModel(
        workgroup_processor_count=DEFAULT_PROFILE.workgroup_processor_count,
        fallback_preparation_s=8.0,
        fallback_validation_s=0.15,
        fallback_timing_s=0.05,
        seed=args.seed + 1,
        jobs=args.surrogate_jobs,
    )
    cost_fit = cost_model.fit(
        candidates=candidate_by_hash,
        shapes_by_candidate=shapes_by_candidate,
        measured_costs=evidence.candidate_costs,
    )
    shortlisted_prediction_keys = {(prediction.shape_id, prediction.candidate_hash) for prediction in predictions}
    acquisition = plan_candidate_bundles(
        controller,
        candidates=candidates,
        shapes=shapes,
        predictions=predictions,
        cost_model=cost_model,
        policy=BundleAcquisitionPolicy(
            improvement_weight=1.0,
            coverage_weight=0.0,
            information_weight=args.information_weight,
            repair_weight=args.integrated_repair_weight,
            bundle_sizes=(1, 2, 4, 8, 12),
            max_pairs=args.max_pairs,
            max_bundles=args.max_bundles,
            max_predicted_cost_s=args.soft_budget_s * 0.85,
            min_samples=DEFAULT_PROFILE.default_protocol.num_benchmarks,
            evidence_stage=EvidenceStage.SCREENING,
        ),
        repair_values={
            shape.id: repair_deficits[shape.id].capped_deficit_log if shape.id in repair_deficits else 0.0
            for shape in shapes
        },
        repair_probabilities={
            key: value for key, value in repair_probabilities.items() if key in shortlisted_prediction_keys
        },
    )
    plan_payload = {
        "round_id": args.round_id,
        "database": str(args.db),
        "initial_database": str(args.initialize_from),
        "initialized_database": initialized,
        "strategy": strategy_label,
        "parameters": {
            "strategy": args.strategy,
            "promote": list(args.promote),
            "promotion_parent_floor": args.promotion_parent_floor,
            "parent_count": args.parent_count,
            "interaction_parents": list(args.interaction_parent),
            "interaction_profile": args.interaction_profile,
            "store_batch_values": list(args.store_batch_values),
            "near_incumbent_fraction": args.near_incumbent_fraction,
            "max_target_shapes": args.max_target_shapes,
            "candidate_pool_limit": args.candidate_pool_limit,
            "max_pairs": args.max_pairs,
            "max_bundles": args.max_bundles,
            "soft_budget_s": args.soft_budget_s,
            "minimum_model_gain": args.minimum_model_gain,
            "information_weight": args.information_weight,
            "integrated_repair_targets": args.integrated_repair_targets,
            "integrated_repair_weight": args.integrated_repair_weight,
            "integrated_repair_mutations": args.integrated_repair_mutations,
            "surrogate_jobs": args.surrogate_jobs,
            "seed": args.seed,
        },
        "compatible_oracle_pairs": len(oracle),
        "positive_oracle_pairs": sum(outcome.performance is not None for outcome in outcomes),
        "parent_winner_counts": {parent.hash: winner_counts[parent.hash] for parent in parents},
        "candidate_target_shapes": {
            candidate_hash: [shape.id for shape in target_shapes]
            for candidate_hash, target_shapes in targets_by_candidate.items()
        },
        "generated_candidate_count": generated_candidate_count,
        "candidate_pool_count": len(pool),
        "shortlisted_candidate_count": len(candidates),
        "incumbent_deployment": str(args.incumbent_deployment),
        "incumbent_report": str(args.incumbent_report),
        "integrated_repair_evidence": list(repair_evidence),
        "integrated_repair_candidate_pool": None if repair_pool is None else repair_pool.to_dict(),
        "general_candidate_hashes": sorted(general_candidate_hashes),
        "model_fit": fit_summary.to_dict(),
        "cost_fit": cost_fit.to_dict(),
        "acquisition": acquisition.to_dict(),
        "selected_candidate_predictions": {
            score.bundle.candidate.hash: {
                "parent_hashes": list(score.bundle.candidate.parent_hashes),
                "proposal_lanes": [
                    *(["general"] if score.bundle.candidate.hash in general_candidate_hashes else []),
                    *(["integrated-repair"] if score.bundle.candidate.hash in repair_candidate_hashes else []),
                ],
                "model_score": list(prediction_scores[score.bundle.candidate.hash]),
                "params": score.bundle.candidate.canonical_params(),
            }
            for score in acquisition.selected
        },
    }
    _write_json(round_dir / "plan.json", plan_payload)
    if args.plan_only or not acquisition.timing_requests:
        print(round_dir / "plan.json")
        return

    started = time.monotonic()
    evaluator = RealEvaluator(
        RealEvaluatorContext(
            db=db,
            output_root=round_dir,
            target_profile=DEFAULT_PROFILE,
            protocol=DEFAULT_PROFILE.default_protocol,
            runner_bin=DEFAULT_PROFILE.default_runner_bin,
            candidate_batch_size=1,
            shape_batch_size=DEFAULT_PROFILE.default_shape_batch_size,
            build_timeout_s=DEFAULT_PROFILE.default_build_timeout_s,
            runner_timeout_s=DEFAULT_PROFILE.default_runner_timeout_s,
            prepare_workers=DEFAULT_PROFILE.default_prepare_workers,
            prepare_wave_batches=min(DEFAULT_PROFILE.default_prepare_wave_batches, args.max_bundles),
            validation_workers=DEFAULT_PROFILE.default_validation_workers,
            compile_cache_root=args.campaign_root / "compile_cache",
            cost_aware_scheduling=True,
            adaptive_policy=AdaptivePolicy(),
            probe_policy=ProbePolicy(),
        ),
        source_ref=f"grid100-practical:{args.round_id}",
    )
    result = evaluator.evaluate(
        acquisition.timing_requests,
        artifact_shapes_by_candidate=acquisition.artifact_shapes_by_candidate,
    )
    wall_s = time.monotonic() - started
    result.apply(controller)
    incumbent_improvements = []
    for shape_id, before in before_incumbents.items():
        after = controller.incumbents[shape_id]
        before_performance = float(before["performance"])
        improvement = after.performance / before_performance - 1.0
        if improvement > 0.0:
            incumbent_improvements.append(
                {
                    "shape_id": shape_id,
                    "before_candidate_hash": before["candidate_hash"],
                    "after_candidate_hash": after.candidate_hash,
                    "before_performance": before_performance,
                    "after_performance": after.performance,
                    "improvement_fraction": improvement,
                }
            )
    incumbent_improvements.sort(key=lambda row: -float(row["improvement_fraction"]))
    report = {
        "round_id": args.round_id,
        "database": str(args.db),
        "strategy": plan_payload["strategy"],
        "wall_time_s": wall_s,
        "phase_time_s": result.phase_time_s,
        "requested_pairs": len(acquisition.timing_requests),
        "requested_candidates": len({request.candidate.hash for request in acquisition.timing_requests}),
        "known_pairs": result.known_pairs,
        "unknown_pairs": result.unknown_pairs,
        "status_counts": dict(Counter(outcome.status for outcome in result.outcomes)),
        "incumbent_improvements": incumbent_improvements,
        "significant_improvements": [
            row for row in incumbent_improvements if float(row["improvement_fraction"]) >= 0.01
        ],
        "outcomes": [
            {
                "shape_id": outcome.request.shape.id,
                "candidate_hash": outcome.request.candidate.hash,
                "parent_hashes": list(outcome.request.candidate.parent_hashes),
                "status": outcome.status,
                "samples": outcome.samples,
                "performance": outcome.performance,
                "baseline_performance": before_incumbents[outcome.request.shape.id]["performance"],
                "improvement_fraction": None
                if outcome.performance is None
                else outcome.performance / float(before_incumbents[outcome.request.shape.id]["performance"]) - 1.0,
                "source_ref": outcome.source_ref,
            }
            for outcome in result.outcomes
        ],
        "plan": str(round_dir / "plan.json"),
    }
    report_path = round_dir / "report.json"
    _write_json(report_path, report)
    _update_manifest(
        manifest_path,
        database=args.db,
        initial_database=args.initialize_from,
        initialized=False,
        round_summary={
            "round_id": args.round_id,
            "strategy": plan_payload["strategy"],
            "report": str(report_path),
            "wall_time_s": wall_s,
            "requested_pairs": len(acquisition.timing_requests),
            "requested_candidates": len({request.candidate.hash for request in acquisition.timing_requests}),
            "improved_shapes": len(incumbent_improvements),
            "significantly_improved_shapes": len(report["significant_improvements"]),
        },
    )
    print(report_path)


if __name__ == "__main__":
    main()
