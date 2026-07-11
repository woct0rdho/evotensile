import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .activity import apu_activity_lock
from .database import EvoTensileDB
from .subprocess_utils import run_logged_process

DEFAULT_TENSILELITE_BIN = os.path.expanduser("~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/bin/Tensile")


@dataclass
class RunResult:
    run_id: str
    returncode: int
    stdout_path: Path
    stderr_path: Path
    output_dir: Path
    command: list[str]
    duration_s: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _merged_env(env: dict[str, str] | None) -> dict[str, str] | None:
    if env is None:
        return None
    merged = os.environ.copy()
    merged.update(env)
    return merged


def _without_global_parameter(global_parameters: list[str] | None, key: str) -> list[str]:
    key_prefix = f"{key}="
    return [item for item in global_parameters or [] if not item.strip().startswith(key_prefix)]


def _global_parameter_args(
    global_parameters: list[str] | None,
    *,
    cpu_threads: int | None,
) -> list[str]:
    params = _without_global_parameter(global_parameters, "CpuThreads")
    if cpu_threads is not None:
        params.append(f"CpuThreads={cpu_threads}")
    if not params:
        return []
    return ["--global-parameters", *params]


def run_tensilelite(
    yaml_path: str | Path,
    output_dir: str | Path,
    *,
    tensilelite_bin: str | Path = DEFAULT_TENSILELITE_BIN,
    db: EvoTensileDB | None = None,
    build_only: bool = False,
    cpu_threads: int | None = None,
    global_parameters: list[str] | None = None,
    env: dict[str, str] | None = None,
    timeout_s: float | None = None,
    use_cache: bool = False,
    candidate_hashes: list[str] | None = None,
) -> RunResult:
    yaml_path = Path(yaml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    stdout_path = output_dir / f"{run_id}.stdout.log"
    stderr_path = output_dir / f"{run_id}.stderr.log"

    cmd = [str(tensilelite_bin), str(yaml_path), str(output_dir)]
    if use_cache:
        cmd.append("--use-cache")
    if build_only:
        cmd.append("--build-only")
    cmd.extend(_global_parameter_args(global_parameters, cpu_threads=cpu_threads))

    start = time.perf_counter()
    timed_out = False
    returncode = 0
    with apu_activity_lock(exclusive=False):
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            returncode, timed_out = run_logged_process(
                cmd,
                stdout=stdout,
                stderr=stderr,
                env=_merged_env(env),
                timeout_s=timeout_s,
            )
            if timed_out:
                stderr.write(f"\nTensileLite build timed out after {timeout_s} seconds\n")
    duration_s = time.perf_counter() - start

    result = RunResult(
        run_id=run_id,
        returncode=returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        output_dir=output_dir,
        command=cmd,
        duration_s=duration_s,
        timed_out=timed_out,
    )
    if db is not None:
        db.insert_run(
            run_id,
            phase="prepare",
            status="timeout" if result.timed_out else "ok" if result.ok else "failed",
            duration_s=duration_s,
            returncode=result.returncode,
            candidate_hashes=candidate_hashes,
        )
    return result
