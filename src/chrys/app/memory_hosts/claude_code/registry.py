# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Claude host identities, locator registrations and catalog delivery state."""

from __future__ import annotations

import errno
import hashlib
import json
import ntpath
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from chrys.app.memory_hosts.context import SOURCE_UNAVAILABLE
from chrys.foundation.platform.files import (
    atomic_write_owner_only_bytes,
    read_owner_verified_bounded,
    secure_open_owner_verified_binary,
    secure_unlink_owner_verified,
)
from chrys.service.memory.storage.store import MemoryOperationError

WORKSPACE_UNAVAILABLE = (
    "Claude Code's session workspace is unavailable. Submit a new prompt after restoring the workspace; "
    "the shell cwd is not used as a fallback."
)
REGISTRATION_LIMIT = 16 * 1024
DELIVERY_LIMIT = 64 * 1024


@dataclass(frozen=True, slots=True)
class HostKey:
    """A process identity that cannot survive PID reuse."""

    pid: int
    started_us: int


@dataclass(frozen=True, slots=True)
class Registration:
    """Only the current host's source locator and session workspace."""

    session_id: str
    transcript_path: str
    project_dir: str | None


@dataclass(frozen=True, slots=True)
class ProcessInfo:
    """The executable and real parent of one inspected process."""

    pid: int
    ppid: int
    exe: str
    started_us: int
    argv: tuple[str, ...] | None


class ProcessTree(Protocol):
    """Inspect a live process, returning None when its identity is unavailable."""

    def inspect(self, pid: int) -> ProcessInfo | None: ...


class SystemProcessTree:
    """Read process facts from psutil without trusting argv to identify a host."""

    def inspect(self, pid: int) -> ProcessInfo | None:
        import psutil

        try:
            process = psutil.Process(pid)
            with process.oneshot():
                started_us = int(process.create_time() * 1_000_000)
                exe = process.exe()
                ppid = process.ppid()
                try:
                    argv = tuple(process.cmdline())
                except psutil.AccessDenied:
                    argv = None
            if not process.is_running():
                return None
            return ProcessInfo(pid, ppid, exe, started_us, argv)
        except psutil.Error:
            return None


def _is_claude(exe: str) -> bool:
    return ntpath.basename(exe).casefold() in {"claude", "claude.exe"}


def find_host(tree: ProcessTree, pid: int | None = None) -> HostKey | None:
    """Find the nearest executable Claude ancestor, including across wrappers."""
    current = tree.inspect(os.getpid() if pid is None else pid)
    seen: set[int] = set()
    while current is not None:
        seen.add(current.pid)
        if current.ppid <= 0 or current.ppid in seen:
            return None
        current = tree.inspect(current.ppid)
        if current is not None and _is_claude(current.exe):
            return HostKey(current.pid, current.started_us)
    return None


def verify_host(tree: ProcessTree, key: HostKey) -> bool:
    """Check the frozen host directly, without choosing another ancestor."""
    process = tree.inspect(key.pid)
    return process is not None and process.started_us == key.started_us and _is_claude(process.exe)


def project_source_supported(tree: ProcessTree, key: HostKey) -> bool:
    """Reject unverified argv and the unsupported project-config-root override."""
    process = tree.inspect(key.pid)
    if process is None or process.started_us != key.started_us or not _is_claude(process.exe):
        return False
    return process.argv is not None and not any(
        arg == "--project-config-root" or arg.startswith("--project-config-root=") for arg in process.argv
    )


def project_directory(tree: ProcessTree, key: HostKey, environ: Mapping[str, str]) -> str | None:
    """Use only a presently valid CLAUDE_PROJECT_DIR from this hook invocation."""
    if not project_source_supported(tree, key):
        return None
    directory = environ.get("CLAUDE_PROJECT_DIR")
    if not directory or not os.path.isabs(directory) or not os.path.isdir(directory):
        return None
    return directory


@dataclass(frozen=True, slots=True)
class DeliveryState:
    """The last rendered catalog, independent of the redelivery flag."""

    session_id: str
    last_text: str | None
    ever_nonempty: bool
    valid: bool = True


class Registry:
    """Owner-verified atomic state below the memory root; no locks or ordering."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def registration_path(self, key: HostKey) -> Path:
        return self.root / "hosts" / "claude-code" / "registrations" / f"{key.pid}-{key.started_us}.json"

    def delivery_path(self, key: HostKey, session_id: str) -> Path:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return self.root / "hosts" / "claude-code" / "delivery" / f"{key.pid}-{key.started_us}" / f"{digest}.json"

    def redeliver_path(self, key: HostKey, session_id: str) -> Path:
        return self.delivery_path(key, session_id).with_suffix(".redeliver")

    def read_registration(self, key: HostKey) -> Registration | None:
        try:
            payload = read_owner_verified_bounded(self.registration_path(key), max_bytes=REGISTRATION_LIMIT)
            data = json.loads(payload)
            if not isinstance(data, dict) or set(data) != {"session_id", "transcript_path", "project_dir"}:
                raise ValueError
            session = data["session_id"]
            transcript = data["transcript_path"]
            project = data["project_dir"]
            if not isinstance(session, str) or not session or not isinstance(transcript, str) or not transcript:
                raise ValueError
            if project is not None and not isinstance(project, str):
                raise ValueError
            return Registration(session, transcript, project)
        except OSError as error:
            if error.errno == errno.ENOENT:
                return None
            raise MemoryOperationError("memory_store_unavailable", SOURCE_UNAVAILABLE) from error
        except (ValueError, TypeError, RecursionError) as error:
            raise MemoryOperationError("memory_store_unavailable", SOURCE_UNAVAILABLE) from error

    def write_registration(self, key: HostKey, registration: Registration) -> None:
        payload = _encode(
            {
                "session_id": registration.session_id,
                "transcript_path": registration.transcript_path,
                "project_dir": registration.project_dir,
            }
        )
        if not registration.session_id or not registration.transcript_path or len(payload) > REGISTRATION_LIMIT:
            raise MemoryOperationError("memory_store_unavailable", SOURCE_UNAVAILABLE)
        atomic_write_owner_only_bytes(self.registration_path(key), payload)

    def read_delivery(self, key: HostKey, session_id: str) -> DeliveryState | None:
        try:
            data = json.loads(
                read_owner_verified_bounded(self.delivery_path(key, session_id), max_bytes=DELIVERY_LIMIT)
            )
            if not isinstance(data, dict) or data.get("session_id") != session_id:
                return None
            ever = data.get("ever_nonempty")
            if not isinstance(ever, bool):
                return None
            text = data.get("last_text")
            valid = set(data) == {"session_id", "last_text", "ever_nonempty"} and (
                text is None or isinstance(text, str)
            )
            return DeliveryState(session_id, text if isinstance(text, str) else None, ever, valid)
        except OSError, ValueError, TypeError, RecursionError:
            return None

    def write_delivery(self, key: HostKey, state: DeliveryState) -> None:
        payload = _encode(
            {
                "session_id": state.session_id,
                "last_text": state.last_text,
                "ever_nonempty": state.ever_nonempty,
            }
        )
        if len(payload) > DELIVERY_LIMIT:
            raise MemoryOperationError("memory_store_unavailable", "Memory catalog delivery state is unavailable.")
        atomic_write_owner_only_bytes(self.delivery_path(key, state.session_id), payload)

    def needs_redeliver(self, key: HostKey, session_id: str) -> bool:
        try:
            with secure_open_owner_verified_binary(self.redeliver_path(key, session_id)):
                pass
        except OSError as error:
            if error.errno == errno.ENOENT:
                return False
            raise
        return True

    def mark_redeliver(self, key: HostKey, session_id: str) -> None:
        atomic_write_owner_only_bytes(self.redeliver_path(key, session_id), b"")

    def clear_redeliver(self, key: HostKey, session_id: str) -> None:
        secure_unlink_owner_verified(self.redeliver_path(key, session_id))


def _encode(data: Mapping[str, object]) -> bytes:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
