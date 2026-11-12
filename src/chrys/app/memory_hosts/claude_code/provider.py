# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Freeze each Claude request's registration before inspecting its source."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING

from chrys.app.memory_hosts.claude_code.conversation import build_conversation, validate_anchor
from chrys.app.memory_hosts.claude_code.registry import (
    SOURCE_UNAVAILABLE,
    WORKSPACE_UNAVAILABLE,
    HostKey,
    ProcessTree,
    Registration,
    Registry,
    SystemProcessTree,
    find_host,
    project_source_supported,
    verify_host,
)
from chrys.app.memory_hosts.context import ReadContext, WriteContext
from chrys.service.memory.project import bind_project
from chrys.service.memory.storage.store import MemoryOperationError

if TYPE_CHECKING:
    from chrys.service.memory.storage.store import MemoryScope


class ClaudeCodeProvider:
    """Use one startup HostKey and a per-request immutable locator snapshot.

    Construction probes the process tree and belongs in a thread for async callers.
    """

    def __init__(self, registry: Registry, tree: ProcessTree | None = None) -> None:
        self.registry = registry
        self.tree = SystemProcessTree() if tree is None else tree
        self.host: HostKey | None = find_host(self.tree)

    def _snapshot(
        self, metadata: Mapping[str, object] | None, *, writing: bool
    ) -> tuple[Registration | None, str | None]:
        # This small owner-verified read must happen before the request's first await.
        registration = None if self.host is None else self.registry.read_registration(self.host)
        tool_id: str | None = None
        if metadata is not None:
            value = metadata.get("claudecode/toolUseId")
            if not isinstance(value, str) or not value:
                raise MemoryOperationError("memory_store_unavailable", SOURCE_UNAVAILABLE)
            tool_id = value
        if (writing or metadata is not None) and (registration is None or tool_id is None):
            raise MemoryOperationError("memory_store_unavailable", SOURCE_UNAVAILABLE)
        return registration, tool_id

    def _project(self, registration: Registration) -> MemoryScope | None:
        host = self.host
        if host is None or not verify_host(self.tree, host):
            raise MemoryOperationError("memory_store_unavailable", SOURCE_UNAVAILABLE)
        directory = registration.project_dir
        if (
            directory is None
            or not os.path.isabs(directory)
            or not os.path.isdir(directory)
            or not project_source_supported(self.tree, host)
        ):
            raise MemoryOperationError("memory_store_unavailable", WORKSPACE_UNAVAILABLE)
        try:
            return bind_project(directory)
        except MemoryOperationError, OSError:
            sys.stderr.write("Memory: project binding is unavailable; only global memory is available.\n")
            return None

    async def read_context(self, metadata: Mapping[str, object] | None) -> ReadContext:
        """Metadata-bearing reads must prove the same anchor as writes."""
        registration, tool_id = self._snapshot(metadata, writing=False)
        if registration is None:
            return ReadContext(None)
        return await asyncio.to_thread(self._read, registration, tool_id)

    def _read(self, registration: Registration, tool_id: str | None) -> ReadContext:
        project = self._project(registration)
        if tool_id is not None:
            validate_anchor(registration, tool_id)
        return ReadContext(project)

    async def write_context(self, metadata: Mapping[str, object] | None) -> WriteContext:
        """The view and project come from the exact locator captured at admission."""
        registration, tool_id = self._snapshot(metadata, writing=True)
        if registration is None or tool_id is None:
            raise MemoryOperationError("memory_store_unavailable", SOURCE_UNAVAILABLE)
        return await asyncio.to_thread(self._write, registration, tool_id)

    def _write(self, registration: Registration, tool_id: str) -> WriteContext:
        project = self._project(registration)
        try:
            view = build_conversation(registration, tool_id)
        except MemoryOperationError as error:
            if error.kind == "memory_invalid_input":
                raise MemoryOperationError("memory_store_unavailable", SOURCE_UNAVAILABLE) from error
            raise
        return WriteContext(project, view)
