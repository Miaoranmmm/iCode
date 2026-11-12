# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Host-independent metadata describing successful memory changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from .storage.store import MemoryScope, MemoryType, WrittenBy

ChangeOperation = Literal["create", "update", "withdraw", "restore", "merge", "delete"]


@dataclass(frozen=True, slots=True)
class MemoryChange:
    """One committed change's identity and versions, without memory contents."""

    time: str
    scope: MemoryScope
    ref: str
    type: MemoryType
    operation: ChangeOperation
    written_by: WrittenBy
    source: str
    before_version: str | None
    after_version: str | None


class ChangeLog(Protocol):
    """Append change metadata independently of the memory store's write result."""

    async def append(self, change: MemoryChange) -> None: ...
