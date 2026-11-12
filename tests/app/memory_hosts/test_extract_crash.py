# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Hard-killed commit workers leave the actually published state, never a guessed rollback."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import psutil
import pytest

from chrys.foundation.platform.process import managed_subprocess
from chrys.foundation.util.lock import FileLock
from chrys.service.memory.extraction import ProjectRunLock, SessionKey, SourceRevision, WatermarkRecord, WatermarkStore
from chrys.service.memory.storage.store import MemoryScope
from tests.support.secure_files import plant_owner_only_bytes
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for

REPO = Path(__file__).resolve().parents[3]
PROJECT = MemoryScope("project", "0123456789abcdef")
KEY = SessionKey("claude-code", "00000000-0000-4000-8000-000000000001")
REVISION = SourceRevision(1, "a" * 64)


@pytest.mark.parametrize("phase", ["attempt", "success", "limited"])
@pytest.mark.parametrize("point", ["before", "after"])
async def test_kill_during_real_atomic_commit_reports_only_persistent_state(
    tmp_path: Path, phase: str, point: str
) -> None:
    root, marker, home = tmp_path / "memory", tmp_path / "ready", tmp_path / "home"
    await asyncio.to_thread(root.mkdir)
    await asyncio.to_thread(home.mkdir)
    await asyncio.to_thread(plant_owner_only_bytes, root / "extraction.json", b'{"version":1,"projects":{}}')
    env = dict(os.environ)
    env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "APPDATA": str(home),
            "XDG_CONFIG_HOME": str(home / "config"),
            "PYTHONPATH": os.pathsep.join((str(REPO), str(REPO / "src"))),
        }
    )
    captured: psutil.Process
    async with managed_subprocess(
        sys.executable,
        "-m",
        "tests.app.memory_hosts.extraction_commit_worker",
        str(root),
        str(marker),
        phase,
        point,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=tmp_path,
        env=env,
    ) as process:
        captured = psutil.Process(process.pid)
        await wait_for(lambda: marker.exists() or process.returncode is not None, timeout=ENGINE_TURN_TIMEOUT)
        if process.returncode is not None:
            stdout, stderr = await process.communicate()
            pytest.fail(f"Crash worker exited before its gate: {stdout!r} {stderr!r}")
        assert marker.exists() and captured.is_running()
        process.kill()
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=ENGINE_TURN_TIMEOUT)
        assert process.returncode != 0
        assert stdout == b"" and stderr == b""
    assert not captured.is_running()
    record = (await WatermarkStore(root).load()).record(str(PROJECT.project_id), KEY)
    if phase == "attempt":
        expected = WatermarkRecord() if point == "before" else WatermarkRecord(None, REVISION, 1)
    elif point == "before":
        expected = WatermarkRecord(None, REVISION, 1)
    elif phase == "success":
        expected = WatermarkRecord(REVISION, REVISION)
    else:
        expected = WatermarkRecord(None, REVISION, 0, 0, 1)
    assert record == expected
    # Process death releases both the whole-run lock and the transaction lock.
    lock = ProjectRunLock(root, PROJECT)
    assert await lock.try_acquire() is True
    await lock.release()

    def probe_transaction_lock() -> None:
        with FileLock(root / "extraction.lock", timeout=0):
            pass

    await asyncio.to_thread(probe_transaction_lock)
