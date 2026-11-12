# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Which project a working directory belongs to, for memory purposes.

Every worktree of one repository shares one project: the id hashes the
repository's common Git directory. Outside Git the directory itself is the
project. A failed Git query is reported, never answered with a different id.
"""

from __future__ import annotations

import hashlib
import os

from chrys.service.memory.storage.store import MemoryOperationError, MemoryScope
from chrys.service.mutations.git_state import canonical_path, locate_git_repository

_PROJECT_ID_SALT = b"memory-project-v1\0"


def project_id_for_anchor(anchor_path: str) -> str:
    """The 16-hex project id of a canonical anchor path (a common Git directory, or a plain directory)."""
    return hashlib.sha256(_PROJECT_ID_SALT + os.fsencode(anchor_path)).hexdigest()[:16]


def bind_project(cwd: str) -> MemoryScope:
    """The project scope of *cwd*. Blocks on a Git subprocess; async callers run it in a thread.

    Raises ``memory_store_unavailable`` when Git cannot answer, so entries are
    never written under, or looked up in, a project the directory is not in.
    """
    if not os.path.isdir(cwd):
        raise MemoryOperationError("memory_store_unavailable", "The working directory does not exist.")
    try:
        location = locate_git_repository(cwd)
    except OSError as exc:
        raise MemoryOperationError(
            "memory_store_unavailable", f"Could not tell which project this directory belongs to: {exc}."
        ) from exc
    if location is None:
        directory = canonical_path(cwd)
        return MemoryScope("project", project_id_for_anchor(directory), root=directory, anchor="dir")
    return MemoryScope("project", project_id_for_anchor(location.common_dir), root=location.root, anchor="git")
