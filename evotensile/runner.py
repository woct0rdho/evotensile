import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from .cache import (
    benchmark_protocol_hash_from_items,
    normalize_version_name,
)
from .cache import (
    problem_type_hash as default_problem_type_hash,
)
from .database import EvoTensileDB

DEFAULT_TENSILELITE_BIN = "/home/wd/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/bin/Tensile"


@dataclass
class RunResult:
    run_id: str
    returncode: int
    stdout_path: Path
    stderr_path: Path
    output_dir: Path
    command: list[str]
    version_name: str
    problem_type_hash: str
    benchmark_protocol_hash: str

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


def _effective_protocol_hash(
    global_parameters: list[str] | None,
    *,
    cpu_threads: int | None,
    benchmark_protocol_hash: str | None,
) -> str:
    if benchmark_protocol_hash:
        return benchmark_protocol_hash
    # CpuThreads affects compilation parallelism, not timings, so exclude it from the protocol hash.
    return benchmark_protocol_hash_from_items(global_parameters)


def run_tensilelite(
    yaml_path: str | Path,
    output_dir: str | Path,
    *,
    tensilelite_bin: str | Path = DEFAULT_TENSILELITE_BIN,
    db: EvoTensileDB | None = None,
    use_cache: bool = False,
    build_only: bool = False,
    cpu_threads: int | None = None,
    global_parameters: list[str] | None = None,
    version_name: str | None = None,
    problem_type_hash: str | None = None,
    benchmark_protocol_hash: str | None = None,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> RunResult:
    yaml_path = Path(yaml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    stdout_path = output_dir / f"{run_id}.stdout.log"
    stderr_path = output_dir / f"{run_id}.stderr.log"

    version = normalize_version_name(version_name)
    ptype_hash = problem_type_hash or default_problem_type_hash()
    proto_hash = _effective_protocol_hash(
        global_parameters, cpu_threads=cpu_threads, benchmark_protocol_hash=benchmark_protocol_hash
    )

    cmd = [str(tensilelite_bin), str(yaml_path), str(output_dir)]
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
        version_name=version,
        problem_type_hash=ptype_hash,
        benchmark_protocol_hash=proto_hash,
    )
    if db is not None:
        db.insert_run(
            run_id,
            yaml_path=str(yaml_path),
            output_dir=str(output_dir),
            tensilelite_bin=str(tensilelite_bin),
            status="ok" if result.ok else "failed",
            version_name=version,
            problem_type_hash=ptype_hash,
            benchmark_protocol_hash=proto_hash,
            returncode=result.returncode,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            metadata_json=json.dumps(
                {
                    "command": cmd,
                    "version_name": version,
                    "problem_type_hash": ptype_hash,
                    "benchmark_protocol_hash": proto_hash,
                },
                sort_keys=True,
            ),
        )
    return result


def build_then_benchmark(
    yaml_path: str | Path,
    output_dir: str | Path,
    *,
    tensilelite_bin: str | Path = DEFAULT_TENSILELITE_BIN,
    db: EvoTensileDB | None = None,
    compile_threads: int | None = -1,
    benchmark_threads: int | None = 1,
    global_parameters: list[str] | None = None,
    version_name: str | None = None,
    problem_type_hash: str | None = None,
    benchmark_protocol_hash: str | None = None,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> tuple[RunResult, RunResult | None]:
    """Compile with --build-only, then benchmark serially with --use-cache."""
    build_result = run_tensilelite(
        yaml_path,
        output_dir,
        tensilelite_bin=tensilelite_bin,
        db=db,
        build_only=True,
        cpu_threads=compile_threads,
        global_parameters=global_parameters,
        version_name=version_name,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        extra_args=extra_args,
        env=env,
    )
    if not build_result.ok:
        return build_result, None

    benchmark_globals = list(global_parameters or [])
    benchmark_globals.append("ParallelGpuExecution=1")
    bench_result = run_tensilelite(
        yaml_path,
        output_dir,
        tensilelite_bin=tensilelite_bin,
        db=db,
        use_cache=True,
        cpu_threads=benchmark_threads,
        global_parameters=benchmark_globals,
        version_name=version_name,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        extra_args=extra_args,
        env=env,
    )
    return build_result, bench_result
