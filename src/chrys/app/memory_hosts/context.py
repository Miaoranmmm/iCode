# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The context seam between the shared memory protocol and each host."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from chrys.service.memory.storage.store import MemoryScope
    from chrys.service.memory.verification import ConversationView

Mode = Literal["read-write", "read-only"]

SOURCE_UNAVAILABLE = (
    "Claude Code's memory source is unavailable. Submit a new prompt or start a new session before trying again."
)
READ_ONLY_REASON = (
    "Saving or deleting memory needs a host that can verify your words, such as Claude Code. "
    "This connection is read-only."
)
EMPTY_CATALOG = "No saved memories are available for this workspace."
CODEX_INSTRUCTIONS = (
    "This connection is read-only because it cannot verify the user's words. At the start of a conversation, "
    "call memory_catalog before relying on saved memories. Read relevant entries with memory_read. "
    "Saving and deleting memories are unavailable here; use Claude Code for changes."
)
CLAUDE_INSTRUCTIONS = (
    "The hook supplies the memory catalog. Call memory_catalog if you need a fresh listing, "
    "and memory_read before relying on an entry."
)
CLAUDE_READ_ONLY_INSTRUCTIONS = (
    CLAUDE_INSTRUCTIONS + " This Claude Code connection is configured read-only. Saving and deleting are "
    "unavailable until the write-verification checks pass."
)


@dataclass(frozen=True, slots=True)
class ReadContext:
    """The project visible to one read operation; global scope is always available."""

    project: MemoryScope | None


@dataclass(frozen=True, slots=True)
class WriteContext:
    """One frozen source view and the project belonging to that same source."""

    project: MemoryScope | None
    view: ConversationView


class MemoryContextProvider(Protocol):
    """Resolve one request, without taking model-supplied source locations."""

    async def read_context(self, metadata: Mapping[str, object] | None) -> ReadContext: ...

    async def write_context(self, metadata: Mapping[str, object] | None) -> WriteContext: ...
