# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Directory cooldown claims and detached, globally serialized Claude extraction."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from chrys.foundation.platform.files import (
    atomic_write_owner_only_bytes,
    secure_open_owner_only_append,
    secure_open_owner_only_binary,
    strip_inherited_directory_acl,
)
from chrys.foundation.util.lock import FileLock

INTERVAL_SECONDS = 4 * 60 * 60
BATCH_CAP = 8
RUN_LOCK_TIMEOUT_SECONDS = 2 * 60 * 60
STATE_LOCK_TIMEOUT_SECONDS = 1


def _directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink():
        raise OSError("Automatic extraction directory must not be a link.")
    path.chmod(0o700)
    strip_inherited_directory_acl(path)


def _lock(path: Path, timeout: float) -> FileLock:
    # Pre-create and verify the lock with the same owner-only gates as the state.
    os.close(secure_open_owner_only_append(path).fd)
    return FileLock(path, timeout=timeout)


def _state(path: Path) -> dict[str, float]:
    try:
        with secure_open_owner_only_binary(path) as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return {}
        if any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for key, value in data.items()
        ):
            return {}
        return data
    except OSError, ValueError, TypeError, RecursionError, OverflowError:
        return {}


def maybe_start(memory_root: Path, source_dir: Path, project_dir: Path, selector: str) -> None:
    """Claim one source directory only after a detached process successfully starts."""
    root = memory_root / "auto-extract"
    _directory(root)
    lock = _lock(root / "state.lock", STATE_LOCK_TIMEOUT_SECONDS)
    try:
        lock.acquire()
    except TimeoutError:
        return
    try:
        source = os.path.normcase(str(source_dir.resolve()))
        state = _state(root / "state.json")
        now = time.time()
        previous = state.get(source)
        if previous is not None and 0 <= now - previous < INTERVAL_SECONDS:
            return
        logs = root / "logs"
        _directory(logs)
        digest = hashlib.sha256(os.fsencode(source)).hexdigest()
        handle = secure_open_owner_only_append(logs / f"{digest}.log")
        with os.fdopen(handle.fd, "wb") as log:
            os.ftruncate(log.fileno(), 0)
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            kwargs: dict[str, Any] = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            subprocess.Popen(  # noqa: S603 - fixed interpreter/module; paths and selector are separate argv entries.
                [
                    sys.executable,
                    "-m",
                    "chrys.app.memory_hosts.extract",
                    "-C",
                    str(project_dir),
                    "--claude-source",
                    str(source_dir),
                    "--model-profile",
                    selector,
                    "--batch-cap",
                    str(BATCH_CAP),
                    "--auto-extract-lock",
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                env=env,
                **kwargs,
            )
        state[source] = time.time()
        atomic_write_owner_only_bytes(root / "state.json", json.dumps(state, ensure_ascii=True).encode("utf-8"))
    finally:
        lock.release()


def run_locked(memory_root: Path, extract: Callable[[], int]) -> int:
    """Wait a bounded time for the single automatic extractor, then release on exit."""
    root = memory_root / "auto-extract"
    _directory(root)
    lock = _lock(root / "run.lock", RUN_LOCK_TIMEOUT_SECONDS)
    try:
        lock.acquire()
    except TimeoutError:
        sys.stderr.write("Memory auto-extraction timed out waiting for the run lock.\n")
        return 1
    try:
        return extract()
    finally:
        lock.release()
