# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The scope directories shared by memory entries and their change logs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .store import MemoryScope


def scope_directory(root: Path, scope: MemoryScope) -> Path:
    """Locate one validated scope without creating or inspecting its directory."""
    if scope.scope == "global":
        return root / "global"
    return root / "projects" / str(scope.project_id)
