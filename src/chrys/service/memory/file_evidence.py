# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Resolve and snapshot project files for verified saves and read-time basis checks."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from chrys.foundation.platform.files import secure_open_regular_binary
from chrys.foundation.platform.paths import is_absolute_path, resolve_workspace_path

from .storage.store import FileEvidence, MemoryInputRejected, MemoryOperationError

if TYPE_CHECKING:
    from .storage.store import MemoryScope

MAX_FILE_BYTES: Final = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ProjectFile:
    """A canonical project-relative path, with its physical root pinned by spelling."""

    root: Path
    path: Path
    relative: str


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """A stable size and optional hash, plus the actual bytes read for a budget."""

    sha256: str | None
    bytes: int
    read_bytes: int


class FileBudgetExceeded(OSError):
    """A stable file would exceed the caller's remaining byte budget."""


class SnapshotUnavailable(OSError):
    """A failed snapshot that still consumes the bytes reserved for its read."""

    def __init__(self, charged_bytes: int) -> None:
        super().__init__("The project file could not be read as a stable snapshot.")
        self.charged_bytes = charged_bytes


class InvalidProjectFile(ValueError):
    """The supplied path cannot name evidence within the bound project."""


def resolve_project_file(project_root: Path, path: str) -> ProjectFile:
    """Resolve an exact path and reject a real path outside the current root."""
    try:
        root = project_root.resolve(strict=True)
    except FileNotFoundError, NotADirectoryError:
        raise OSError("The current project root is unavailable.") from None
    if not root.is_dir():
        raise OSError("The current project root is unavailable.")
    if not path or (is_absolute_path(path) and not Path(path).is_absolute()):
        raise InvalidProjectFile("The source is not a current project file.")
    resolved = Path(resolve_workspace_path(path, base_cwd=root)).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        raise InvalidProjectFile("The source is outside the current project root.") from None
    if not relative.parts:
        raise InvalidProjectFile("The source is not a project file.")
    return ProjectFile(root, resolved, relative.as_posix())


def _signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def snapshot_file(source: ProjectFile, *, remaining_bytes: int = MAX_FILE_BYTES) -> FileSnapshot:
    """Read at most 2 MiB and reject file or path changes during the snapshot."""
    before = source.path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise InvalidProjectFile("The source is not a regular project file.")
    charged_bytes = 0
    try:
        with secure_open_regular_binary(source.path) as handle:
            opened = os.fstat(handle.fileno())
            if _signature(before) != _signature(opened):
                raise OSError("The project file changed while being opened.")
            if opened.st_size > MAX_FILE_BYTES:
                payload = None
            else:
                if opened.st_size > remaining_bytes:
                    raise FileBudgetExceeded("The file exceeds the remaining read budget.")
                charged_bytes = opened.st_size
                payload = handle.read(charged_bytes)
            after = os.fstat(handle.fileno())
            visible = source.path.stat()
            if (
                _signature(opened) != _signature(after)
                or _signature(after) != _signature(visible)
                or source.path.resolve() != source.path
                or source.root.resolve() != source.root
                or (payload is not None and len(payload) != after.st_size)
            ):
                raise OSError("The project file changed while being read.")
    except FileBudgetExceeded:
        raise
    except OSError, ValueError, RuntimeError:
        raise SnapshotUnavailable(charged_bytes) from None
    return FileSnapshot(
        hashlib.sha256(payload).hexdigest() if payload is not None else None,
        after.st_size,
        len(payload) if payload is not None else 0,
    )


def capture_file(project: MemoryScope | None, project_root: Path | None, path: str) -> FileEvidence:
    """Capture a file's current basis, without exposing absolute paths in errors."""
    if project is None or project.project_id is None or project_root is None:
        raise MemoryOperationError(
            "memory_evidence_not_found", "File evidence needs an available current project root."
        )
    try:
        source = resolve_project_file(project_root, path)
        snapshot = snapshot_file(source)
    except FileNotFoundError, NotADirectoryError, InvalidProjectFile:
        raise MemoryInputRejected(
            "memory_evidence_not_found", "The evidence file is unavailable or outside the current project root."
        ) from None
    except OSError, ValueError, RuntimeError:
        raise MemoryOperationError(
            "memory_evidence_not_found", "The evidence file is unavailable or outside the current project root."
        ) from None
    return FileEvidence(project.project_id, source.relative, snapshot.sha256, snapshot.bytes)
