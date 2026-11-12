# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Check whether the project files behind an engineering entry still match its saved basis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from .file_evidence import FileBudgetExceeded, SnapshotUnavailable, resolve_project_file, snapshot_file
from .storage.store import FileEvidence

if TYPE_CHECKING:
    from pathlib import Path

    from .storage.store import Evidence, MemoryScope

BasisState = Literal["unchanged", "changed", "unknown", "none"]
FileBasisState = Literal["unchanged", "changed", "unknown"]
MAX_BASIS_FILES: Final = 8
MAX_BASIS_BYTES: Final = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class FileBasis:
    """One file evidence item's read-time status, addressed by evidence index."""

    index: int
    state: FileBasisState
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class BasisResult:
    """The file statuses and their conservative overall result."""

    state: BasisState
    files: tuple[FileBasis, ...] = ()


def check_basis(evidence: Sequence[Evidence], project: MemoryScope | None, project_root: Path | None) -> BasisResult:
    """Check bounded file snapshots; callers run this blocking work in a thread."""
    results: list[FileBasis] = []
    checked = 0
    remaining = MAX_BASIS_BYTES
    for index, item in enumerate(evidence):
        if not isinstance(item, FileEvidence):
            continue
        if project is None or project.project_id is None:
            results.append(FileBasis(index, "unknown", "no current project"))
            continue
        if item.project != project.project_id:
            results.append(FileBasis(index, "unknown", "from another project"))
            continue
        if project_root is None:
            results.append(FileBasis(index, "unknown", "project root unavailable"))
            continue
        if item.sha256 is None:
            results.append(FileBasis(index, "unknown", "no saved hash"))
            continue
        if checked >= MAX_BASIS_FILES:
            results.append(FileBasis(index, "unknown", "file budget exceeded"))
            continue
        checked += 1
        try:
            source = resolve_project_file(project_root, item.path)
            snapshot = snapshot_file(source, remaining_bytes=remaining)
        except FileNotFoundError:
            results.append(FileBasis(index, "changed", "no longer exists"))
        except FileBudgetExceeded:
            results.append(FileBasis(index, "unknown", "byte budget exceeded"))
        except SnapshotUnavailable as error:
            remaining -= error.charged_bytes
            results.append(FileBasis(index, "unknown", "file could not be checked"))
        except OSError, ValueError, RuntimeError:
            results.append(FileBasis(index, "unknown", "file could not be checked"))
        else:
            remaining -= snapshot.read_bytes
            if snapshot.sha256 is None:
                results.append(FileBasis(index, "unknown", "file exceeds 2 MiB"))
            elif snapshot.sha256 == item.sha256:
                results.append(FileBasis(index, "unchanged"))
            else:
                results.append(FileBasis(index, "changed", "differs from the content when this was saved"))
    state: BasisState
    if any(result.state == "changed" for result in results):
        state = "changed"
    elif any(result.state == "unknown" for result in results):
        state = "unknown"
    elif results:
        state = "unchanged"
    else:
        state = "none"
    return BasisResult(state, tuple(results))
