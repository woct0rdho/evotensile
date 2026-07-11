import contextlib
import fcntl
import json
import os
import socket
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

from evotensile.candidate import Candidate, Shape, stable_hash
from evotensile.profile import TargetProfile
from evotensile.protocol import BenchmarkProtocol
from evotensile.scheduling.models import PlannedBatch

_COMPILE_CACHE_LOCK_POLL_S = 0.1

_COMPILE_CACHE_LOCK_WAIT_S = 3600.0
_SUCCESS_MARKER = ".evotensile_compile_cache_ok"


def _compile_cache_global_parameters(target_profile: TargetProfile, protocol: BenchmarkProtocol) -> dict[str, object]:
    protocol_keys = set(protocol.global_parameters())
    return {
        key: value
        for key, value in target_profile.global_parameters(protocol).items()
        if key not in protocol_keys
        and key
        not in {
            "ForceRedoBenchmarkProblems",
            "ForceRedoLibraryLogic",
            "ValidationMaxToPrint",
            "ValidationPrintValids",
        }
    }


def _compile_cache_key(
    candidates: list[Candidate],
    shapes: list[Shape],
    *,
    target_profile: TargetProfile,
    protocol: BenchmarkProtocol,
) -> str:
    payload = {
        "candidates": sorted(candidate.hash for candidate in candidates),
        "shapes": sorted(shape.id for shape in shapes),
        "global_parameters": _compile_cache_global_parameters(target_profile, protocol),
        "library_logic": target_profile.library_logic,
        "problem_type_hash": target_profile.problem_type_hash,
    }
    return stable_hash(payload, prefix="ccache_")[:22]


def compile_cache_dir(
    compile_cache_root: str | Path | None,
    current: PlannedBatch,
    *,
    target_profile: TargetProfile,
    protocol: BenchmarkProtocol,
) -> Path | None:
    if compile_cache_root is None:
        return None
    return Path(compile_cache_root) / _compile_cache_key(
        current.candidates,
        current.shapes,
        target_profile=target_profile,
        protocol=protocol,
    )


def has_tensilelite_cache(path: Path) -> bool:
    return (path / _SUCCESS_MARKER).exists() and any(path.glob("**/caches/*/cache.yaml"))


@contextlib.contextmanager
def compile_cache_lock(
    path: Path,
    *,
    wait_timeout_s: float = _COMPILE_CACHE_LOCK_WAIT_S,
) -> Iterator[None]:
    if wait_timeout_s < 0:
        raise ValueError("compile-cache lock timeout must be non-negative")
    lock_path = path.parent / f".{path.name}.lock"
    owner = {
        "created_at": time.time(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "token": uuid.uuid4().hex,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait_timeout_s
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out after {wait_timeout_s:g}s waiting for compile-cache lock {lock_path}"
                    )
                time.sleep(min(_COMPILE_CACHE_LOCK_POLL_S, max(0.0, deadline - time.monotonic())))
        lock_file.seek(0)
        lock_file.truncate()
        json.dump(owner, lock_file, sort_keys=True)
        lock_file.write("\n")
        lock_file.flush()
        try:
            yield
        finally:
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.flush()
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
