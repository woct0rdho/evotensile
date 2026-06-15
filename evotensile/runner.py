import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from .database import EvoTensileDB

DEFAULT_TENSILE_BIN = "/home/wd/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/bin/Tensile"


@dataclass
class RunResult:
    run_id: str
    returncode: int
    stdout_path: Path
    stderr_path: Path
    output_dir: Path
    command: list[str]

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _merged_env(env: dict[str, str] | None) -> dict[str, str] | None:
    if env is None:
        return None
    merged = os.environ.copy()
    merged.update(env)
    return merged


def _global_parameter_args(
    global_parameters: list[str] | None,
    *,
    cpu_threads: int | None,
) -> list[str]:
    params = list(global_parameters or [])
    if cpu_threads is not None:
        params.append(f"CpuThreads={cpu_threads}")
    if not params:
        return []
    return ["--global-parameters", *params]


def run_tensile(
    yaml_path: str | Path,
    output_dir: str | Path,
    *,
    tensile_bin: str | Path = DEFAULT_TENSILE_BIN,
    db: EvoTensileDB | None = None,
    use_cache: bool = False,
    build_only: bool = False,
    cpu_threads: int | None = None,
    global_parameters: list[str] | None = None,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> RunResult:
    yaml_path = Path(yaml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    stdout_path = output_dir / f"{run_id}.stdout.log"
    stderr_path = output_dir / f"{run_id}.stderr.log"

    cmd = [str(tensile_bin), str(yaml_path), str(output_dir)]
    if use_cache:
        cmd.append("--use-cache")
    if build_only:
        cmd.append("--build-only")
    cmd.extend(_global_parameter_args(global_parameters, cpu_threads=cpu_threads))
    if extra_args:
        cmd.extend(extra_args)

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(
            cmd,
            stdout=stdout,
            stderr=stderr,
            text=True,
            env=_merged_env(env),
            check=False,
        )

    result = RunResult(
        run_id=run_id,
        returncode=proc.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        output_dir=output_dir,
        command=cmd,
    )
    if db is not None:
        db.insert_run(
            run_id,
            yaml_path=str(yaml_path),
            output_dir=str(output_dir),
            tensile_bin=str(tensile_bin),
            status="ok" if result.ok else "failed",
            returncode=result.returncode,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            metadata_json=json.dumps({"command": cmd}),
        )
    return result


def build_then_benchmark(
    yaml_path: str | Path,
    output_dir: str | Path,
    *,
    tensile_bin: str | Path = DEFAULT_TENSILE_BIN,
    db: EvoTensileDB | None = None,
    compile_threads: int | None = -1,
    benchmark_threads: int | None = 1,
    global_parameters: list[str] | None = None,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> tuple[RunResult, RunResult | None]:
    """Compile with --build-only, then benchmark serially with --use-cache."""
    build_result = run_tensile(
        yaml_path,
        output_dir,
        tensile_bin=tensile_bin,
        db=db,
        build_only=True,
        cpu_threads=compile_threads,
        global_parameters=global_parameters,
        extra_args=extra_args,
        env=env,
    )
    if not build_result.ok:
        return build_result, None

    benchmark_globals = list(global_parameters or [])
    benchmark_globals.append("ParallelGpuExecution=1")
    bench_result = run_tensile(
        yaml_path,
        output_dir,
        tensile_bin=tensile_bin,
        db=db,
        use_cache=True,
        cpu_threads=benchmark_threads,
        global_parameters=benchmark_globals,
        extra_args=extra_args,
        env=env,
    )
    return build_result, bench_result
