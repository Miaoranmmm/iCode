# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Read-only memory context bound to the Codex server's startup directory."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

from chrys.app.memory_hosts.context import READ_ONLY_REASON, ReadContext, WriteContext
from chrys.service.memory.project import bind_project
from chrys.service.memory.storage.store import MemoryOperationError, MemoryScope


class CodexProvider:
    """Capture startup cwd once; never inspect Claude processes or registrations.

    Construction includes a Git query and belongs in a thread for async callers.
    """

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = str(Path(os.getcwd() if cwd is None else cwd).resolve())
        directory = Path(self.cwd)
        self.project: MemoryScope | None = None
        if directory == Path(directory.anchor) or directory == Path.home().resolve():
            sys.stderr.write("Memory: startup directory has no project binding; only global memory is available.\n")
            return
        try:
            self.project = bind_project(self.cwd)
        except MemoryOperationError, OSError:
            sys.stderr.write("Memory: project binding is unavailable; only global memory is available.\n")

    async def read_context(self, metadata: Mapping[str, object] | None) -> ReadContext:
        """Metadata cannot change this connection's startup project."""
        return ReadContext(self.project)

    async def write_context(self, metadata: Mapping[str, object] | None) -> WriteContext:
        """Keep a direct forged write from acquiring any conversation source."""
        raise MemoryOperationError("memory_store_unavailable", READ_ONLY_REASON)
