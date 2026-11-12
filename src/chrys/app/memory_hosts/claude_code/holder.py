# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Read Claude's native process registrations without changing their lifecycle."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from chrys.foundation.platform import get_platform
from chrys.foundation.platform.files import secure_open_regular_binary

Activity = Literal["inactive", "active", "unknown"]
_MONTHS = {
    name: index
    for index, name in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
    )
}
_START = re.compile(r"[A-Z][a-z]{2} ([A-Z][a-z]{2}) +([0-9]{1,2}) ([0-9]{2}):([0-9]{2}):([0-9]{2}) ([0-9]{4})")


class HolderProcesses(Protocol):
    """Return UTC process creation seconds, None for a dead PID; errors stay unknown."""

    def started_seconds(self, pid: int) -> int | None: ...


class SystemHolderProcesses:
    """Use process creation identity rather than the later registration timestamp."""

    def started_seconds(self, pid: int) -> int | None:
        import psutil

        try:
            return int(psutil.Process(pid).create_time())
        except psutil.NoSuchProcess:
            return None
        except psutil.Error:
            raise OSError("Process identity unavailable.") from None


class ClaudeHolder(Protocol):
    """Check whether a complete Claude session is still owned by a live process."""

    def check(self, session: str) -> Activity: ...


def proc_start_seconds(value: object) -> int | None:
    """Parse the native LC_ALL=C, TZ=UTC ps token independently of local locale."""
    match = _START.fullmatch(value) if isinstance(value, str) else None
    if match is None or match[1] not in _MONTHS:
        return None
    try:
        return int(
            datetime(
                int(match[6]), _MONTHS[match[1]], int(match[2]), int(match[3]), int(match[4]), int(match[5]), tzinfo=UTC
            ).timestamp()
        )
    except ValueError, OverflowError:
        return None


class NativeClaudeHolder:
    """Inactive means no valid native registration matches a live process identity."""

    def __init__(
        self, sessions_dir: Path, *, processes: HolderProcesses | None = None, pid_domain: str | None = None
    ) -> None:
        self.sessions_dir = sessions_dir
        self.processes = processes or SystemHolderProcesses()
        self.pid_domain = (
            pid_domain or {"macos": "darwin", "linux": "linux", "windows": "win32"}[get_platform().os_name]
        )

    def check(self, session: str) -> Activity:
        try:
            self.sessions_dir.stat()
            with os.scandir(self.sessions_dir) as entries:
                paths = tuple(Path(entry.path) for entry in entries if entry.name.endswith(".json"))
        except FileNotFoundError:
            return "inactive"
        except OSError:
            return "unknown"
        unknown = False
        for path in paths:
            try:
                with secure_open_regular_binary(path) as handle:
                    payload = handle.read(64 * 1024 + 1)
                record = json.loads(payload) if len(payload) <= 64 * 1024 else None
                if not isinstance(record, dict):
                    unknown = True
                    continue
                if not isinstance(record.get("sessionId"), str) or not record["sessionId"]:
                    unknown = True
                    continue
                if record.get("sessionId") != session:
                    continue
                pid = record.get("pid")
                started = proc_start_seconds(record.get("procStart"))
                if (
                    record.get("pidDomain") != self.pid_domain
                    or type(pid) is not int
                    or pid <= 0
                    or started is None
                    or path.stem != str(pid)
                ):
                    unknown = True
                    continue
                actual = self.processes.started_seconds(pid)
                if actual == started:
                    return "active"
            except OSError, ValueError, UnicodeError, RecursionError:
                unknown = True
        return "unknown" if unknown else "inactive"
