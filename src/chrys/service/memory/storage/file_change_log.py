# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Owner-only metadata logs for evaluation, separate from model-visible memories.

Each scope's entry lock serializes encoding, rollover and appending across
processes. The active ``changes.jsonl`` rolls before an append would exceed
1 MiB, retaining every numbered archive. A single oversized record stays whole.
Readers skip invalid JSON lines: an incomplete tail left by a hard process kill
is separated from the next record. Ordinary write failures truncate back to the
pre-append size so they cannot damage a later record.

The complete transaction runs in one worker thread. Cancelling its awaiting
task does not interrupt that worker or release the worker-owned lock early.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Final

from chrys.foundation.platform.files import secure_open_owner_only_append
from chrys.foundation.util.lock import FileLock

from .layout import scope_directory

if TYPE_CHECKING:
    from chrys.service.memory.change_log import MemoryChange

DEFAULT_MAX_BYTES: Final = 1024 * 1024
_LOCK_TIMEOUT_SECONDS: Final = 5.0
_ARCHIVE_NAME: Final = re.compile(r"changes\.([1-9][0-9]*)\.jsonl")


class FileChangeLog:
    """Append metadata under the memory root without adding any store methods."""

    def __init__(self, root: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        if max_bytes < 1:
            raise ValueError("The memory change-log size limit must be positive.")
        self._root = root
        self._max_bytes = max_bytes

    async def append(self, change: MemoryChange) -> None:
        """Run the whole scope transaction independently of caller cancellation."""
        await asyncio.to_thread(self._append, change)

    def _append(self, change: MemoryChange) -> None:
        directory = scope_directory(self._root, change.scope)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with FileLock(directory / ".lock", timeout=_LOCK_TIMEOUT_SECONDS):
            scope: dict[str, str | None] = {"scope": change.scope.scope}
            if change.scope.scope == "project":
                scope["project_id"] = change.scope.project_id
            payload = (
                json.dumps(
                    {
                        "time": change.time,
                        "scope": scope,
                        "ref": change.ref,
                        "type": change.type,
                        "operation": change.operation,
                        "written_by": change.written_by,
                        "source": change.source,
                        "before_version": change.before_version,
                        "after_version": change.after_version,
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            path = directory / "changes.jsonl"
            fd = secure_open_owner_only_append(path).fd
            try:
                size = os.fstat(fd).st_size
                separator = b""
                if size:
                    os.lseek(fd, -1, os.SEEK_END)
                    if os.read(fd, 1) != b"\n":
                        separator = b"\n"
                if size and size + len(separator) + len(payload) > self._max_bytes:
                    os.close(fd)
                    fd = -1
                    path.rename(_next_archive(directory))
                    fd = secure_open_owner_only_append(path).fd
                    size = 0
                    separator = b""
                try:
                    _write_all(fd, separator + payload)
                    os.fsync(fd)
                except BaseException:
                    os.ftruncate(fd, size)
                    raise
            finally:
                if fd != -1:
                    os.close(fd)


def _next_archive(directory: Path) -> Path:
    largest = 0
    for path in directory.iterdir():
        match = _ARCHIVE_NAME.fullmatch(path.name)
        if match is not None:
            largest = max(largest, int(match.group(1)))
    return directory / f"changes.{largest + 1}.jsonl"


def _write_all(fd: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(fd, remaining)
        if written <= 0:
            raise OSError("The memory change log could not be written.")
        remaining = remaining[written:]
