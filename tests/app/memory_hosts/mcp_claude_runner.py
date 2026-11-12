# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Run the production MCP main with a typed process tree for offline SDK tests."""

from __future__ import annotations

import os

from chrys.app.memory_hosts import mcp_server
from chrys.app.memory_hosts.claude_code import provider
from chrys.app.memory_hosts.claude_code.registry import ProcessInfo


class _ProcessTree:
    def inspect(self, pid: int) -> ProcessInfo | None:
        if pid == os.getpid():
            return ProcessInfo(pid, 20, "python", 1, ("python",))
        if pid == 20:
            return ProcessInfo(20, 0, "/fixture/claude", 1_700_000, ("claude",))
        return None


if __name__ == "__main__":
    provider.SystemProcessTree = _ProcessTree
    raise SystemExit(mcp_server.main())
