# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Test-only crash window around an actual atomic extraction-state publication."""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

from chrys.service.memory import extraction as core
from chrys.service.memory.extraction import ProjectRunLock, SessionKey, SourceRevision, SourceSession, WatermarkStore
from chrys.service.memory.storage.store import MemoryScope
from chrys.service.memory.verification import SourceMessage
from tests.support.waiting import ENGINE_TURN_TIMEOUT

PROJECT = MemoryScope("project", "0123456789abcdef")
SOURCE = SourceSession(
    SessionKey("claude-code", "00000000-0000-4000-8000-000000000001"),
    PROJECT,
    SourceRevision(1, "a" * 64),
    (SourceMessage("00000000-0000-4000-8000-000000000001", 1, ("Synthetic lasting rule.",), item="synthetic-user"),),
)


async def commit_until_killed(root: Path, marker: Path, phase: str, point: str) -> None:
    store = WatermarkStore(root)
    lock = ProjectRunLock(root, PROJECT)
    if not await lock.try_acquire():
        raise RuntimeError("The test worker could not acquire its project lock.")
    try:
        ticket = None
        if phase != "attempt":
            prepared = await store.prepare(PROJECT, (SOURCE,))
            ticket = prepared.ticket
            if prepared.outcome != "committed" or ticket is None:
                raise RuntimeError("The test worker could not prepare its attempt.")
            if phase == "limited":
                await store.limited(ticket, "turn_limited")
                prepared = await store.prepare(PROJECT, (SOURCE,))
                ticket = prepared.ticket
                if prepared.outcome != "committed" or ticket is None:
                    raise RuntimeError("The test worker could not prepare its second attempt.")
        publisher = core.atomic_write_owner_only_bytes

        def blocked_publish(path: Path, payload: bytes, *, create_parents: bool = True) -> None:
            if point == "after":
                publisher(path, payload, create_parents=create_parents)
            publisher(marker, b"ready\n")
            if not threading.Event().wait(ENGINE_TURN_TIMEOUT):
                raise TimeoutError("The crash test did not kill its worker.")
            if point == "before":
                publisher(path, payload, create_parents=create_parents)

        core.atomic_write_owner_only_bytes = blocked_publish
        if phase == "attempt":
            await store.prepare(PROJECT, (SOURCE,))
        elif ticket is not None:
            if phase == "success":
                await store.succeed(ticket)
            else:
                await store.limited(ticket, "output_limited")
    finally:
        await lock.release()


def main() -> None:
    root, marker, phase, point = sys.argv[1:]
    asyncio.run(commit_until_killed(Path(root), Path(marker), phase, point))


if __name__ == "__main__":
    main()
