import fcntl
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def apu_activity_lock(*, exclusive: bool) -> Iterator[None]:
    lock_path = Path(os.environ.get("EVOTENSILE_APU_LOCK_PATH", Path(tempfile.gettempdir()) / "evotensile-apu.lock"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
