import os
import signal
import subprocess
from typing import IO


def run_logged_process(
    command: list[str],
    *,
    stdout: IO[str],
    stderr: IO[str],
    env: dict[str, str] | None,
    timeout_s: float | None,
) -> tuple[int, bool]:
    process = subprocess.Popen(
        command,
        text=True,
        stdout=stdout,
        stderr=stderr,
        env=env,
        start_new_session=True,
    )
    try:
        process.wait(timeout=timeout_s)
        return process.returncode, False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        return 124, True
