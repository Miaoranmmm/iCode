# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Test-only real-process writer, gated on stdin so all writers overlap."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from chrys.service.memory.change_log import MemoryChange
from chrys.service.memory.storage.file_change_log import FileChangeLog
from chrys.service.memory.storage.store import GLOBAL_SCOPE


async def write_records(root: Path, prefix: str, count: int, max_bytes: int) -> None:
    log = FileChangeLog(root, max_bytes=max_bytes)
    for index in range(count):
        await log.append(
            MemoryChange(
                time="2026-11-09T01:02:03Z",
                scope=GLOBAL_SCOPE,
                ref=f"global:{index:012x}",
                type="intent",
                operation="update",
                written_by="model",
                source=f"{prefix}-full-conversation-{index}",
                before_version=f"before-{index}",
                after_version=f"after-{index}",
            )
        )


def main() -> None:
    root, prefix, count, max_bytes = sys.argv[1:]
    sys.stdout.write("ready\n")
    sys.stdout.flush()
    if sys.stdin.readline() != "go\n":
        raise ValueError("The test writer was not released.")
    asyncio.run(write_records(Path(root), prefix, int(count), int(max_bytes)))


if __name__ == "__main__":
    main()
