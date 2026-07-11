import hashlib
import importlib.util
import json
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, TypeAlias

from evotensile.candidate import Candidate, Shape, canonical_json
from evotensile.database import EvoTensileDB
from evotensile.profile import TargetProfile
from evotensile.search.evidence import ProposalEvidenceSnapshot
from evotensile.search.learned_linkage import (
    DEFAULT_MAX_CLUSTERS,
    DEFAULT_MIN_LINKAGE_SAMPLES,
    DEFAULT_ORDINAL_BINS,
    DEFAULT_TRUNCATION_TAU,
)
from evotensile.search.surrogate import DEFAULT_SURROGATE_MIN_EVIDENCE
from evotensile.search_space import DOMAINS, FIXED_PARAMS, cheap_constraints, eligible_for_shape_scope

BUILTIN_PROPOSAL_VERSION = "gfx1151-grid-v1"


@dataclass(frozen=True)
class ProposalScope:
    kind: str
    shape_ids: tuple[str, ...]


@dataclass(frozen=True)
class FamilyQDPolicy:
    version: str = BUILTIN_PROPOSAL_VERSION
    num_random: int = 64
    elite_count: int = 8
    local_count: int = 32
    de_count: int = 32
    gomea_count: int = 64
    transfer_shape_count: int = 4
    transfer_per_shape: int = 2
    mutation_rate: float = 0.25
    crossover_rate: float = 0.8
    random_gene_rate: float = 0.1
    learned_linkage: bool = True
    linkage_truncation_tau: float = DEFAULT_TRUNCATION_TAU
    linkage_min_samples: int = DEFAULT_MIN_LINKAGE_SAMPLES
    linkage_max_clusters: int = DEFAULT_MAX_CLUSTERS
    linkage_ordinal_bins: int = DEFAULT_ORDINAL_BINS
    adaptive_operators: bool = True
    surrogate_pool_multiplier: int = 8
    surrogate_min_evidence: int = DEFAULT_SURROGATE_MIN_EVIDENCE
    covering_cold_start: bool = True
    adaptive_group_credit: bool = True
    micro_exhaustive_neighborhoods: bool = True
    adaptive_donor_selection: bool = True
    cost_aware_operator_credit: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ProposalContext:
    target_profile: TargetProfile
    shapes: tuple[Shape, ...]
    scope: ProposalScope
    seed: int
    evidence: ProposalEvidenceSnapshot
    config: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    family_qd_policy: FamilyQDPolicy | None = None
    shape_id: str | None = None
    parent_candidates: tuple[Candidate, ...] | None = None
    cold_start_precovered_tokens: frozenset[str] = frozenset()
    island_id: str | None = None
    restart_index: int = 0


@dataclass(frozen=True)
class ProposalOutput:
    candidates: tuple[Candidate, ...]
    selected_candidate_hashes: tuple[str, ...] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


SimpleProposalOutput: TypeAlias = Iterable[Candidate]
ProviderReturn: TypeAlias = ProposalOutput | SimpleProposalOutput


class ProposalProvider(Protocol):
    def __call__(self, context: ProposalContext) -> ProviderReturn: ...


@dataclass(frozen=True)
class ProviderProvenance:
    identity: str
    script_path: str | None = None
    script_sha256: str | None = None
    declared_name: str | None = None
    declared_version: str | None = None

    def to_dict(self, *, environment_compatibility_tag: str) -> dict[str, object]:
        try:
            package_version = version("evotensile")
        except PackageNotFoundError:
            package_version = None
        return {
            "identity": self.identity,
            "script_path": self.script_path,
            "script_sha256": self.script_sha256,
            "declared_name": self.declared_name,
            "declared_version": self.declared_version,
            "evotensile_version": package_version,
            "environment_compatibility_tag": environment_compatibility_tag,
        }


@dataclass(frozen=True)
class ProposalResult:
    scope: ProposalScope
    preserved: tuple[Candidate, ...]
    generated: tuple[Candidate, ...]
    selected: tuple[Candidate, ...]
    provider: Mapping[str, object]
    metadata: Mapping[str, object]


def proposal_scope(shapes: tuple[Shape, ...], kind: str | None) -> ProposalScope:
    shape_ids = tuple(shape.id for shape in shapes)
    inferred = "global" if not shape_ids else ("shape" if len(shape_ids) == 1 else "shape-set")
    resolved = kind or inferred
    if resolved not in {"global", "shape", "cluster", "shape-set"}:
        raise ValueError(f"unknown proposal scope kind: {resolved}")
    if resolved == "global" and shape_ids:
        raise ValueError("global proposal scope cannot contain shapes")
    if resolved != "global" and not shape_ids:
        raise ValueError(f"{resolved} proposal scope requires at least one shape")
    if resolved == "shape" and len(shape_ids) != 1:
        raise ValueError("shape proposal scope requires exactly one shape")
    return ProposalScope(resolved, shape_ids)


def load_proposal_config(path: str | Path | None) -> Mapping[str, object]:
    if path is None:
        return MappingProxyType({})
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("proposal config must contain one JSON object")
    canonical_json(payload)
    return MappingProxyType(payload)


def load_proposal_script(path: str | Path) -> tuple[ProposalProvider, ProviderProvenance]:
    resolved = Path(path).resolve(strict=True)
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    module_name = f"_evotensile_proposal_{digest[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load proposal script: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    provider = getattr(module, "propose", None)
    if not callable(provider):
        raise ValueError(f"proposal script must export callable propose(context): {resolved}")
    provenance = ProviderProvenance(
        identity=f"script:{resolved}",
        script_path=str(resolved),
        script_sha256=digest,
        declared_name=_optional_module_string(module, "PROVIDER_NAME"),
        declared_version=_optional_module_string(module, "PROVIDER_VERSION"),
    )
    return provider, provenance


def _optional_module_string(module: object, name: str) -> str | None:
    value = getattr(module, name, None)
    return None if value is None else str(value)


def normalize_proposal_output(value: ProviderReturn) -> ProposalOutput:
    if isinstance(value, ProposalOutput):
        canonical_json(dict(value.metadata))
        return value
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError("proposal provider must return candidates or ProposalOutput")
    candidates = tuple(value)
    if not all(isinstance(candidate, Candidate) for candidate in candidates):
        raise TypeError("proposal provider returned a non-Candidate value")
    return ProposalOutput(candidates=candidates)


def finalize_proposal(
    db: EvoTensileDB,
    *,
    context: ProposalContext,
    output: ProposalOutput,
    provenance: ProviderProvenance,
    duration_s: float,
) -> ProposalResult:
    expected_keys = set(FIXED_PARAMS) | set(DOMAINS)
    selected_hashes = (
        {candidate.hash for candidate in output.candidates}
        if output.selected_candidate_hashes is None
        else set(output.selected_candidate_hashes)
    )
    pool_hashes = {candidate.hash for candidate in output.candidates}
    unknown_selected = selected_hashes - pool_hashes
    if unknown_selected:
        raise ValueError(f"selected candidate hashes are not in the provider pool: {sorted(unknown_selected)}")

    deduped: dict[str, Candidate] = {}
    for candidate in output.candidates:
        params = candidate.canonical_params()
        missing = expected_keys - set(params)
        unexpected = set(params) - expected_keys
        if missing or unexpected:
            raise ValueError(
                f"provider candidate {candidate.hash} has incomplete parameters; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        if not cheap_constraints(params):
            raise ValueError(f"provider candidate {candidate.hash} failed global cheap constraints")
        if not eligible_for_shape_scope(params, context.shapes):
            raise ValueError(f"provider candidate {candidate.hash} is ineligible for its proposal scope")
        if candidate.hash not in deduped or candidate.hash in selected_hashes:
            deduped[candidate.hash] = candidate

    parent_hashes = {parent for candidate in deduped.values() for parent in candidate.parent_hashes}
    available_parent_hashes = set(deduped) | {
        candidate.hash for candidate in db.get_candidates(sorted(parent_hashes - set(deduped)))
    }
    unknown_parents = parent_hashes - available_parent_hashes
    if unknown_parents:
        raise ValueError(f"provider candidates reference unknown parents: {sorted(unknown_parents)}")

    known_hashes = {candidate.hash for candidate in db.get_candidates(sorted(deduped))}
    preserved = tuple(candidate for candidate in deduped.values() if candidate.hash in known_hashes)
    generated = tuple(
        Candidate(
            params=candidate.canonical_params(),
            source=candidate.source,
            parent_hashes=candidate.parent_hashes,
            proposal_metadata={
                **candidate.proposal_metadata,
                "proposal_scope_kind": context.scope.kind,
                "proposal_scope_shape_ids": list(context.scope.shape_ids),
            },
        )
        for candidate in deduped.values()
        if candidate.hash not in known_hashes
    )
    by_hash = {candidate.hash: candidate for candidate in (*preserved, *generated)}
    selected = tuple(by_hash[candidate_hash] for candidate_hash in deduped if candidate_hash in selected_hashes)
    provider_metadata = provenance.to_dict(
        environment_compatibility_tag=context.target_profile.environment_compatibility_tag
    )
    event_args: dict[str, object] = {
        "provider": provider_metadata,
        "provider_config": dict(context.config),
        "seed": context.seed,
        "selected_count": len(selected),
        "generated_count": len(generated),
        "provider_metadata": dict(output.metadata),
    }
    db.record_proposal_event(
        [*preserved, *generated],
        problem_type_hash=context.evidence.problem_type_hash,
        benchmark_protocol_hash=context.evidence.benchmark_protocol_hash,
        scope_kind=context.scope.kind,
        scope_shape_ids=context.scope.shape_ids,
        generated_hashes={candidate.hash for candidate in generated},
        selected_candidates=list(selected),
        proposal_args=event_args,
        island_id=context.island_id,
        restart_index=context.restart_index,
        duration_s=duration_s,
    )
    return ProposalResult(
        scope=context.scope,
        preserved=preserved,
        generated=generated,
        selected=selected,
        provider=MappingProxyType(provider_metadata),
        metadata=MappingProxyType(dict(output.metadata)),
    )


def execute_proposal_provider(
    db: EvoTensileDB,
    *,
    context: ProposalContext,
    provider: ProposalProvider,
    provenance: ProviderProvenance,
) -> ProposalResult:
    started = time.perf_counter()
    output = normalize_proposal_output(provider(context))
    return finalize_proposal(
        db,
        context=context,
        output=output,
        provenance=provenance,
        duration_s=time.perf_counter() - started,
    )
