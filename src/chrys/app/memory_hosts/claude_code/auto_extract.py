# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Directory cooldown claims and detached, globally serialized host extraction."""

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


def _launch(root: Path, key: str, host: str, source_dir: Path, project_dir: Path, selector: str) -> None:
    logs = root / "logs"
    _directory(logs)
    digest = hashlib.sha256(os.fsencode(key)).hexdigest()
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
                f"--{host}-source",
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


def maybe_start(memory_root: Path, source_dir: Path, project_dir: Path, selector: str, project_id: str) -> None:
    """Process independent Claude and Codex claims under the same state lock."""
    root = memory_root / "auto-extract"
    _directory(root)
    lock = _lock(root / "state.lock", STATE_LOCK_TIMEOUT_SECONDS)
    try:
        lock.acquire()
    except TimeoutError:
        return
    try:
        source = os.path.normcase(str(source_dir.resolve()))
        claims = [(source, "claude", source_dir)]
        if sys.platform != "win32":
            codex_home = os.environ.get("CODEX_HOME")
            codex_source = (Path(codex_home) if codex_home else Path.home() / ".codex") / "sessions"
            if codex_source.is_dir():
                normalized = os.path.normcase(str(codex_source.resolve()))
                claims.append((f"codex|{normalized}|{project_id}", "codex", codex_source))
        state = _state(root / "state.json")
        error: Exception | None = None
        for key, host, directory in claims:
            previous = state.get(key)
            if previous is not None and 0 <= time.time() - previous < INTERVAL_SECONDS:
                continue
            try:
                _launch(root, key, host, directory, project_dir, selector)
                state[key] = time.time()
                atomic_write_owner_only_bytes(root / "state.json", json.dumps(state, ensure_ascii=True).encode("utf-8"))
            except Exception as failure:
                # A failed host must not prevent admission of the other host.
                error = error or failure
        if error is not None:
            raise error
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
