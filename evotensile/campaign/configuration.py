import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from evotensile.campaign.models import CAMPAIGN_ENVIRONMENT_KEYS, CampaignConfiguration
from evotensile.campaign.protocols import CAMPAIGN_HOT_PROTOCOL, CAMPAIGN_SCREENING_PROTOCOL
from evotensile.candidate import Shape
from evotensile.profile import TargetProfile


def _content_fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve(strict=True) for item in paths}, key=str):
        if not path.is_file():
            continue
        digest.update(str(path).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _binary_identity(path: Path, *, include_python_tree: bool = False) -> tuple[str, str]:
    resolved = path.resolve(strict=True)
    files = [resolved]
    if include_python_tree and resolved.parent.name == "bin":
        files.extend(resolved.parent.parent.rglob("*.py"))
    return str(resolved), _content_fingerprint(files)


def _implementation_fingerprint() -> str:
    package_root = Path(__file__).resolve().parents[1]
    repository_root = package_root.parent
    files = [repository_root / "scripts" / "run_blind_one_shape.py", *package_root.rglob("*.py")]
    return _content_fingerprint(files)


@dataclass(frozen=True)
class CampaignConfigurationRequest:
    runner_bin: Path
    tensilelite_bin: Path
    seed: int
    time_budget_s: float
    hot_reserve_s: float
    max_feedback_rounds: int
    early_stop_on_convergence: bool
    build_timeout_s: float
    runner_timeout_s: float
    leader_stabilization: bool


def build_campaign_configuration(
    request: CampaignConfigurationRequest,
    *,
    profile: TargetProfile,
    shape: Shape,
) -> CampaignConfiguration:
    runner_bin, runner_fingerprint = _binary_identity(request.runner_bin)
    tensilelite_bin, tensilelite_fingerprint = _binary_identity(request.tensilelite_bin, include_python_tree=True)
    return CampaignConfiguration(
        seed=request.seed,
        shape_id=shape.id,
        profile_name=profile.name,
        problem_type_hash=profile.problem_type_hash,
        runner_bin=runner_bin,
        runner_fingerprint=runner_fingerprint,
        tensilelite_bin=tensilelite_bin,
        tensilelite_fingerprint=tensilelite_fingerprint,
        implementation_fingerprint=_implementation_fingerprint(),
        environment=tuple((key, os.environ.get(key, "")) for key in CAMPAIGN_ENVIRONMENT_KEYS),
        time_budget_s=request.time_budget_s,
        hot_reserve_s=request.hot_reserve_s,
        max_feedback_rounds=request.max_feedback_rounds,
        early_stop_on_convergence=request.early_stop_on_convergence,
        build_timeout_s=request.build_timeout_s,
        runner_timeout_s=request.runner_timeout_s,
        screening_protocol=CAMPAIGN_SCREENING_PROTOCOL,
        hot_protocol=CAMPAIGN_HOT_PROTOCOL,
        leader_stabilization=request.leader_stabilization,
        prepare_workers=profile.default_prepare_workers,
        prepare_wave_batches=profile.default_prepare_wave_batches,
        validation_workers=profile.default_validation_workers,
        surrogate_jobs=profile.default_surrogate_jobs,
        compute_unit_count=profile.compute_unit_count,
        workgroup_processor_count=profile.workgroup_processor_count,
        compute_units_per_workgroup_processor=profile.compute_units_per_workgroup_processor,
    )
