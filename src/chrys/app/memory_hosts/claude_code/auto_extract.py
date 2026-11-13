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

from chrys.app.memory_hosts.runtime_files import ensure_private_directory, private_file_lock
from chrys.foundation.platform.files import (
    atomic_write_owner_only_bytes,
    secure_open_owner_only_append,
    secure_open_owner_only_binary,
)

INTERVAL_SECONDS = 4 * 60 * 60
BATCH_CAP = 8
RUN_LOCK_TIMEOUT_SECONDS = 2 * 60 * 60
STATE_LOCK_TIMEOUT_SECONDS = 1
REVIEW_CLAIM_SECONDS = 60 * 60
REVIEW_INTERVAL_SECONDS = 24 * 60 * 60


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
    ensure_private_directory(logs)
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
        argv = (
            [
                sys.executable,
                "-m",
                "chrys.app.memory_hosts.review",
                "--project",
                str(project_dir),
                "--memory-dir",
                str(root.parent),
                "--model-profile",
                selector,
                "--automatic",
            ]
            if host == "review"
            else [
                sys.executable,
                "-m",
                "chrys.app.memory_hosts.extract",
                "-C",
                str(project_dir),
                {"claude": "--claude-source", "codex": "--codex-source"}[host],
                str(source_dir),
                "--model-profile",
                selector,
                "--batch-cap",
                str(BATCH_CAP),
                "--auto-extract-lock",
            ]
        )
        subprocess.Popen(  # noqa: S603 - fixed interpreter and modules; values are separate argv entries.
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            env=env,
            **kwargs,
        )


def _review_due(memory_root: Path, project_id: str, launched_at: float | None) -> bool:
    """A paused review still launches once per interval: only the child can tell whether the store changed."""
    from chrys.app.memory_hosts.review_state import load_state

    state = load_state(memory_root, project_id)
    since = launched_at if state.paused else state.success_at
    return not since or not 0 <= time.time() - since < REVIEW_INTERVAL_SECONDS


def maybe_start(memory_root: Path, source_dir: Path, project_dir: Path, selector: str, project_id: str) -> None:
    """Process independent Claude and Codex claims under the same state lock."""
    root = memory_root / "auto-extract"
    ensure_private_directory(root)
    lock = private_file_lock(root / "state.lock", STATE_LOCK_TIMEOUT_SECONDS)
    try:
        lock.acquire()
    except TimeoutError:
        return
    try:
        source = os.path.normcase(str(source_dir.resolve()))
        claims = [(source, "claude", source_dir)]
        if sys.platform != "win32":
            codex_home = os.environ.get("CODEX_HOME")
            base = Path(codex_home) if codex_home else Path.home() / ".codex"
            # Codex moves archived threads out of sessions; each directory is its own source.
            for codex_source in (base / "sessions", base / "archived_sessions"):
                if codex_source.is_dir():
                    normalized = os.path.normcase(str(codex_source.resolve()))
                    claims.append((f"codex|{normalized}|{project_id}", "codex", codex_source))
        claims.append((f"review|{project_id}", "review", memory_root))
        state = _state(root / "state.json")
        error: Exception | None = None
        for key, host, directory in claims:
            previous = state.get(key)
            interval = REVIEW_CLAIM_SECONDS if host == "review" else INTERVAL_SECONDS
            if previous is not None and 0 <= time.time() - previous < interval:
                continue
            try:
                if host == "review" and not _review_due(memory_root, project_id, previous):
                    continue
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
    ensure_private_directory(root)
    lock = private_file_lock(root / "run.lock", RUN_LOCK_TIMEOUT_SECONDS)
    try:
        lock.acquire()
    except TimeoutError:
        sys.stderr.write("Memory auto-extraction timed out waiting for the run lock.\n")
        return 1
    try:
        return extract()
    finally:
        lock.release()
