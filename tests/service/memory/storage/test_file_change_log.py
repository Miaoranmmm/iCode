# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Change logs remain complete under rollover, concurrent writers and cancellation."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import psutil
import pytest

from chrys.foundation.platform.files import secure_open_owner_verified_binary
from chrys.foundation.platform.process import managed_subprocess
from chrys.service.memory.change_log import MemoryChange
from chrys.service.memory.storage import file_change_log
from chrys.service.memory.storage.file_change_log import DEFAULT_MAX_BYTES, FileChangeLog
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.layout import scope_directory
from chrys.service.memory.storage.store import GLOBAL_SCOPE, MemoryEntryDraft, MemoryScope, Quote
from tests.support.secure_files import plant_owner_only_bytes
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for

if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import Sequence

PROJECT = MemoryScope("project", "0123456789abcdef", root="/private/workspace-not-in-log", anchor="git")
OTHER_PROJECT = MemoryScope("project", "fedcba9876543210")
REPO = Path(__file__).resolve().parents[4]


def change(index: int = 0, scope: MemoryScope = GLOBAL_SCOPE) -> MemoryChange:
    return MemoryChange(
        time="2026-11-09T01:02:03Z",
        scope=scope,
        ref=f"{scope.scope}:{index:012x}",
        type="intent",
        operation="update",
        written_by="model",
        source=f"full-conversation-id-{index}",
        before_version=f"before-{index}",
        after_version=f"after-{index}",
    )


def archives(directory: Path) -> list[Path]:
    return sorted(directory.glob("changes.*.jsonl"), key=lambda path: int(path.name.split(".")[1]))


def read_rows(paths: Sequence[Path]) -> list[dict[str, object]]:
    """Evaluation readers skip malformed rows without discarding subsequent records."""
    rows: list[dict[str, object]] = []
    for path in paths:
        with secure_open_owner_verified_binary(path) as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except ValueError, UnicodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def all_rows(root: Path, scope: MemoryScope = GLOBAL_SCOPE) -> list[dict[str, object]]:
    directory = scope_directory(root, scope)
    return read_rows([*archives(directory), directory / "changes.jsonl"])


@pytest.mark.parametrize("scope", [GLOBAL_SCOPE, PROJECT])
async def test_exact_metadata_fields_and_shared_scope_layout(tmp_path: Path, scope: MemoryScope) -> None:
    root = tmp_path / "memory"
    event = change(1, scope)
    await FileChangeLog(root).append(event)
    directory = scope_directory(root, scope)
    assert await FileMemoryStore(root).location(scope) == str(directory)
    assert await asyncio.to_thread(all_rows, root, scope) == [
        {
            "time": "2026-11-09T01:02:03Z",
            "scope": {"scope": "global"}
            if scope == GLOBAL_SCOPE
            else {
                "scope": "project",
                "project_id": "0123456789abcdef",
            },
            "ref": f"{scope.scope}:000000000001",
            "type": "intent",
            "operation": "update",
            "written_by": "model",
            "source": "full-conversation-id-1",
            "before_version": "before-1",
            "after_version": "after-1",
        }
    ]
    payload = await asyncio.to_thread((directory / "changes.jsonl").read_text, encoding="utf-8")
    assert PROJECT.root not in payload
    assert "anchor" not in payload and "quotes" not in payload and "body" not in payload
    assert payload.endswith("\n") and len(payload.splitlines()) == 1
    files = await asyncio.to_thread(lambda: sorted(path.relative_to(root).as_posix() for path in root.rglob("*.*")))
    prefix = "global" if scope == GLOBAL_SCOPE else "projects/0123456789abcdef"
    assert files == [f"{prefix}/.lock", f"{prefix}/changes.jsonl"]


async def test_null_versions_are_preserved(tmp_path: Path) -> None:
    event = MemoryChange(
        "2026-11-09T01:02:03Z",
        GLOBAL_SCOPE,
        "global:000000000001",
        "intent",
        "create",
        "extraction",
        "run-full-id",
        None,
        "v1",
    )
    await FileChangeLog(tmp_path).append(event)
    [row] = await asyncio.to_thread(all_rows, tmp_path)
    assert row["before_version"] is None and row["after_version"] == "v1"


def test_default_rollover_threshold_is_one_mib() -> None:
    assert DEFAULT_MAX_BYTES == 1024 * 1024


async def test_rollover_happens_before_append_and_retains_every_archive(tmp_path: Path) -> None:
    log = FileChangeLog(tmp_path, max_bytes=1)
    for index in range(6):
        await log.append(change(index))
    directory = scope_directory(tmp_path, GLOBAL_SCOPE)
    assert [path.name for path in await asyncio.to_thread(archives, directory)] == [
        "changes.1.jsonl",
        "changes.2.jsonl",
        "changes.3.jsonl",
        "changes.4.jsonl",
        "changes.5.jsonl",
    ]
    rows = await asyncio.to_thread(all_rows, tmp_path)
    assert [row["source"] for row in rows] == [f"full-conversation-id-{index}" for index in range(6)]
    assert await asyncio.to_thread(read_rows, [directory / "changes.jsonl"]) == [rows[-1]]


async def test_rollover_boundary_fits_exactly_and_never_splits_one_record(tmp_path: Path) -> None:
    await FileChangeLog(tmp_path).append(change())
    path = scope_directory(tmp_path, GLOBAL_SCOPE) / "changes.jsonl"
    size = (await asyncio.to_thread(path.stat)).st_size
    log = FileChangeLog(tmp_path, max_bytes=size * 2)
    await log.append(change())
    assert await asyncio.to_thread(archives, path.parent) == []
    await log.append(change())
    assert (await asyncio.to_thread(path.parent.joinpath("changes.1.jsonl").stat)).st_size == size * 2
    assert (await asyncio.to_thread(path.stat)).st_size == size
    assert len(await asyncio.to_thread(all_rows, tmp_path)) == 3


async def test_numbering_continues_after_existing_archives(tmp_path: Path) -> None:
    directory = scope_directory(tmp_path, GLOBAL_SCOPE)
    await asyncio.to_thread(directory.mkdir, parents=True)
    await asyncio.to_thread(plant_owner_only_bytes, directory / "changes.9.jsonl", b"invalid archive row\n")
    log = FileChangeLog(tmp_path, max_bytes=1)
    await log.append(change(1))
    await log.append(change(2))
    assert [path.name for path in await asyncio.to_thread(archives, directory)] == [
        "changes.9.jsonl",
        "changes.10.jsonl",
    ]
    assert [row["source"] for row in await asyncio.to_thread(all_rows, tmp_path)] == [
        "full-conversation-id-1",
        "full-conversation-id-2",
    ]


async def test_concurrent_threads_append_and_roll_without_lost_or_duplicate_rows(tmp_path: Path) -> None:
    await asyncio.gather(*(FileChangeLog(tmp_path, max_bytes=800).append(change(index)) for index in range(30)))
    rows = await asyncio.to_thread(all_rows, tmp_path)
    assert len(rows) == 30
    assert {row["source"] for row in rows} == {f"full-conversation-id-{index}" for index in range(30)}


async def test_incomplete_prior_tail_is_separated_and_reader_skips_bad_lines(tmp_path: Path) -> None:
    directory = scope_directory(tmp_path, GLOBAL_SCOPE)
    await asyncio.to_thread(directory.mkdir, parents=True)
    path = directory / "changes.jsonl"
    await asyncio.to_thread(plant_owner_only_bytes, path, b'not-json\n{"unfinished":')
    await FileChangeLog(tmp_path).append(change(8))
    assert [row["source"] for row in await asyncio.to_thread(all_rows, tmp_path)] == ["full-conversation-id-8"]
    assert (await asyncio.to_thread(path.read_bytes)).startswith(b'not-json\n{"unfinished":\n{"time":')


async def test_partial_os_writes_are_completed_under_the_scope_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shadow = ModuleType("os")
    shadow.__dict__.update(vars(os))
    real_write = os.write

    def short_write(fd: int, data: bytes) -> int:
        return real_write(fd, data[:7])

    shadow.write = short_write
    monkeypatch.setattr(file_change_log, "os", shadow)
    await FileChangeLog(tmp_path).append(change(7))
    assert [row["source"] for row in await asyncio.to_thread(all_rows, tmp_path)] == ["full-conversation-id-7"]


async def test_failed_partial_append_rolls_back_before_later_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await FileChangeLog(tmp_path).append(change(1))
    shadow = ModuleType("os")
    shadow.__dict__.update(vars(os))
    real_write = os.write
    writes = 0

    def failed_write(fd: int, data: bytes) -> int:
        nonlocal writes
        writes += 1
        if writes == 1:
            return real_write(fd, data[:9])
        raise OSError("test write failure")

    shadow.write = failed_write
    with monkeypatch.context() as patch:
        patch.setattr(file_change_log, "os", shadow)
        with pytest.raises(OSError, match="test write failure"):
            await FileChangeLog(tmp_path).append(change(2))
    await FileChangeLog(tmp_path).append(change(3))
    assert [row["source"] for row in await asyncio.to_thread(all_rows, tmp_path)] == [
        "full-conversation-id-1",
        "full-conversation-id-3",
    ]


async def test_cancellation_propagates_while_thread_keeps_transaction_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    real_write_all = file_change_log._write_all

    def held_write(fd: int, payload: bytes) -> None:
        entered.set()
        if not release.wait(timeout=ENGINE_TURN_TIMEOUT):
            raise TimeoutError("The test writer was not released.")
        try:
            real_write_all(fd, payload)
        finally:
            finished.set()

    monkeypatch.setattr(file_change_log, "_write_all", held_write)
    task = asyncio.create_task(FileChangeLog(tmp_path).append(change(1)))
    try:
        await wait_for(entered.is_set, timeout=ENGINE_TURN_TIMEOUT)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await wait_for(finished.is_set, timeout=ENGINE_TURN_TIMEOUT)
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    await FileChangeLog(tmp_path).append(change(2))
    assert [row["source"] for row in await asyncio.to_thread(all_rows, tmp_path)] == [
        "full-conversation-id-1",
        "full-conversation-id-2",
    ]


async def test_active_and_archived_logs_are_invisible_to_store_reads(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path)
    draft = MemoryEntryDraft(
        "intent",
        "Use SQLite",
        "Use SQLite for local state.",
        "model",
        quotes=(Quote("Use SQLite", "message", "session-full"),),
    )
    await store.create(PROJECT, draft)
    before = (
        await store.list_entries(PROJECT),
        await store.catalog(PROJECT),
        await store.search(PROJECT, ""),
        await store.unreadable(PROJECT),
        await store.list_projects(),
    )
    log = FileChangeLog(tmp_path, max_bytes=1)
    for index in range(4):
        await log.append(change(index, PROJECT))
    after = (
        await store.list_entries(PROJECT),
        await store.catalog(PROJECT),
        await store.search(PROJECT, ""),
        await store.unreadable(PROJECT),
        await store.list_projects(),
    )
    assert after == before
    assert await store.search(PROJECT, "full-conversation-id") == ([], None)
    assert await store.catalog(GLOBAL_SCOPE) == [] and await store.unreadable(GLOBAL_SCOPE) == []
    assert not scope_directory(tmp_path, OTHER_PROJECT).exists()
    paths = [
        *await asyncio.to_thread(archives, scope_directory(tmp_path, PROJECT)),
        scope_directory(tmp_path, PROJECT) / "changes.jsonl",
    ]
    assert len(paths) == 4
    assert len(await asyncio.to_thread(read_rows, paths)) == 4


@pytest.mark.parametrize("max_bytes", [0, -1])
def test_rollover_limit_must_be_positive(tmp_path: Path, max_bytes: int) -> None:
    with pytest.raises(ValueError):
        FileChangeLog(tmp_path, max_bytes=max_bytes)


async def test_real_processes_share_rotation_lock_without_loss_or_duplicates(tmp_path: Path) -> None:
    import sys

    root = tmp_path / "memory"
    home = tmp_path / "home"
    await asyncio.to_thread(home.mkdir)
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
    processes: list[Process] = []
    captured: list[psutil.Process] = []
    async with AsyncExitStack() as stack:
        for index in range(3):
            process = await stack.enter_async_context(
                managed_subprocess(
                    sys.executable,
                    "-m",
                    "tests.service.memory.storage.change_log_worker",
                    str(root),
                    f"writer-{index}",
                    "20",
                    "800",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=tmp_path,
                    env=env,
                )
            )
            processes.append(process)
            captured.append(psutil.Process(process.pid))
        for process in processes:
            assert process.stdout is not None
            assert await asyncio.wait_for(process.stdout.readline(), timeout=ENGINE_TURN_TIMEOUT) == b"ready\n"
        for process in processes:
            assert process.stdin is not None
            process.stdin.write(b"go\n")
            await process.stdin.drain()
        results = await asyncio.wait_for(
            asyncio.gather(*(process.communicate() for process in processes)), timeout=ENGINE_TURN_TIMEOUT
        )
        for process, (stdout, stderr) in zip(processes, results, strict=True):
            assert process.returncode == 0, stderr.decode("utf-8")
            assert stdout == b"" and stderr == b""
    assert all(not process.is_running() for process in captured)
    rows = await asyncio.to_thread(all_rows, root)
    assert len(rows) == 60
    assert {row["source"] for row in rows} == {
        f"writer-{writer}-full-conversation-{index}" for writer in range(3) for index in range(20)
    }
    rotated = await asyncio.to_thread(archives, scope_directory(root, GLOBAL_SCOPE))
    assert len(rotated) > 1
    assert [path.name for path in rotated] == [f"changes.{index}.jsonl" for index in range(1, len(rotated) + 1)]
