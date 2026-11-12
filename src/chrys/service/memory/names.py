# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Memory tool identities shared by host schemas and source exclusion."""

from __future__ import annotations

from typing import Final

MEMORY_TOOL_NAMES: Final = frozenset(
    {
        "memory_read",
        "memory_search",
        "memory_catalog",
        "memory_save",
        "memory_forget",
        "memory_withdraw",
        "memory_merge",
        "memory_note_conflict",
    }
)


def is_memory_tool(name: str) -> bool:
    """Host namespaces do not turn a memory result into fresh evidence."""
    return name.rsplit("__", 1)[-1] in MEMORY_TOOL_NAMES
