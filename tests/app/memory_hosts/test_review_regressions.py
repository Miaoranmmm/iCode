# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Offline regression contracts for review cancellation, checkpoints and byte preservation."""

from __future__ import annotations

import asyncio
import errno
import json
import sys
from dataclasses import replace

import pytest

from chrys.app.memory_hosts import executor_icode, review, review_state
from chrys.app.memory_hosts.review_state import ReviewState, load_state
from chrys.foundation.platform.files import SecureFileError, atomic_write_owner_only_bytes
from chrys.foundation.util.lock import FileLock
from chrys.service.memory.extraction import ExtractionError
from chrys.service.memory.storage import file_store
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.layout import scope_directory
from chrys.service.memory.storage.store import MemoryOperationError, MergeResult
from tests.app.memory_hosts.executor_fakes import ClientFactory, ScriptedWire, call, response
from tests.app.memory_hosts.test_extract import PROFILE
from tests.app.memory_hosts.test_review import Offline, run
from tests.app.memory_hosts.test_review import library as library
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for


def test_existing_states_directory_missing_file_is_first_run(tmp_path):
    (tmp_path / "review" / "states").mkdir(parents=True, mode=0o700)
    assert load_state(tmp_path, "second") == ReviewState()


@pytest.mark.parametrize("code", [errno.EACCES, errno.ELOOP])
def test_insecure_state_is_not_first_run(tmp_path, monkeypatch, code):
    def denied(path):
        raise SecureFileError(code, "state rejected")

    monkeypatch.setattr(review_state, "secure_open_owner_only_binary", denied)
    with pytest.raises(OSError) as error:
        load_state(tmp_path, "project")
    assert error.value.errno == code


async def test_two_projects_share_one_memory_root(library):
    first = await run(library, Offline())
    second_project = library.project.parent / "second-project"
    second_project.mkdir()
    second = await review.run_review(second_project, library.root, PROFILE, executor=Offline())
    assert first.status == second.status == "nothing_to_review"
    assert first.project_id != second.project_id
    assert load_state(library.root, first.project_id).success_at > 0
    assert load_state(library.root, second.project_id).success_at > 0


async def test_cancel_native_response_never_admits_merge(library, monkeypatch):
    await library.seed()
    await library.seed()
    a, b = await library.store.list_entries(library.scope)
    entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def stalled(messages):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return response(
            call("memory_read", {"ref": a.entry.ref}, "a"),
            call("memory_read", {"ref": b.entry.ref}, "b"),
            call(
                "memory_merge",
                {
                    "keep_ref": a.entry.ref,
                    "keep_version": a.version,
                    "drop_ref": b.entry.ref,
                    "drop_version": b.version,
                },
                "merge",
            ),
        )

    wire = ScriptedWire([stalled, response(text="Done")])
    monkeypatch.setattr(executor_icode, "create_client", ClientFactory(wire))
    task = asyncio.create_task(review.run_review(library.project, library.root, PROFILE))
    try:
        await wait_for(entered.is_set, timeout=ENGINE_TURN_TIMEOUT)
        task.cancel()
        await wait_for(cancelled.is_set, timeout=ENGINE_TURN_TIMEOUT)
        release.set()
        await wait_for(task.done, timeout=ENGINE_TURN_TIMEOUT)
        report = await task
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert report.status == "failed" and report.counts["cancel_requested"]
    assert not report.merges
    assert len(wire.requests) == 1 and wire.closed == 1
    assert len(await library.store.list_entries(library.scope)) == 2
    assert load_state(library.root, report.project_id).snapshot == {}


async def test_waiting_for_run_lock_is_cancellable_without_late_acquisition(library, monkeypatch):
    root = library.root / "auto-extract"
    root.mkdir(parents=True, mode=0o700)
    observed = asyncio.Event()
    real_lock = review.private_file_lock
    loop = asyncio.get_running_loop()

    def instrumented(path, timeout):
        lock = real_lock(path, timeout)
        real_acquire = lock.acquire

        def acquire():
            loop.call_soon_threadsafe(observed.set)
            return real_acquire()

        lock.acquire = acquire
        return lock

    monkeypatch.setattr(review, "private_file_lock", instrumented)
    with FileLock(root / "run.lock"):
        task = asyncio.create_task(run(library, Offline()))
        try:
            await wait_for(observed.is_set, timeout=ENGINE_TURN_TIMEOUT)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await wait_for(task.done, timeout=ENGINE_TURN_TIMEOUT)
                await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    with FileLock(root / "run.lock", timeout=0):
        pass
    reports = list((library.root / "review" / "reports").rglob("*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text(encoding="utf-8"))["status"] == "failed"


async def test_partial_writes_do_not_reset_three_failure_pause(library, monkeypatch):
    await library.seed()
    await library.seed()
    attempts = 0

    async def partial(store, scope, keep_ref, keep_version, drop_ref, drop_version, draft):
        nonlocal attempts
        attempts += 1
        version = await store.update(scope, keep_ref, keep_version, draft)
        assert version != keep_version
        return MergeResult(version, False)

    monkeypatch.setattr(file_store, "_now", lambda: f"2099-01-01T00:00:0{attempts}Z")
    monkeypatch.setattr(FileMemoryStore, "merge", partial)
    for expected in (1, 2, 3):
        report = await run(library, Offline("partial"), automatic=True)
        state = load_state(library.root, report.project_id)
        assert report.status == "failed" and state.failures == expected
        assert report.merges[0].status == "partial"
        assert state.snapshot == {}
    assert state.paused
    assert (await run(library, Offline("partial"), automatic=True)).status == "paused"
    assert attempts == 3
    await library.seed()
    report = await run(library, Offline("failure"), automatic=True)
    assert report.status == "failed"
    state = load_state(library.root, report.project_id)
    assert not state.paused and state.failures == 1


async def test_drop_changed_after_read_is_rejected_before_report(library):
    await library.seed()
    await library.seed()

    class ConcurrentDrop(Offline):
        async def run(self, prompt, reader, writer, policy):
            a, b = (await reader.search("", "all")).entries
            await writer.read(reader, a.entry.ref)
            await writer.read(reader, b.entry.ref)
            await library.store.update(
                library.scope, b.entry.ref, b.version, replace(b.entry.draft(), title="Changed drop")
            )
            with pytest.raises(MemoryOperationError) as error:
                await writer.merge(a.entry.ref, a.version, b.entry.ref, b.version)
            assert error.value.kind == "memory_version_conflict"
            assert writer.merges == []
            assert (await library.store.get(library.scope, a.entry.ref)).version == a.version
            return await Offline().run(prompt, reader, writer, policy)

    report = await run(library, ConcurrentDrop())
    assert report.status == "completed" and report.merges == []
    assert len(await library.store.list_entries(library.scope)) == 2


async def test_catalog_keeps_chinese_metadata_and_excludes_body(library):
    ref = await library.seed(body="This body must stay out of the catalog.")
    item = await library.store.get(library.scope, ref)
    version = await library.store.update(
        library.scope, ref, item.version, replace(item.entry.draft(), title="先查找再读取", applies_when="搜索代码时")
    )
    text = review.catalog_text(tuple(await library.store.list_entries(library.scope)), (ref,))
    assert "先查找再读取" in text and "搜索代码时" in text
    assert "\\u5148" not in text and item.entry.body not in text
    rows = text.split("<review_catalog>\n")[1].split("\n</review_catalog>")[0].splitlines()
    row = json.loads(rows[0])
    assert set(row) == {"ref", "type", "scope", "title", "applies_when", "updated", "quotes", "evidence", "context"}
    assert row["ref"] == ref and version
    assert f"<changed_refs>\n{ref}\n</changed_refs>" in text


@pytest.mark.parametrize("unsafe", ["oversized", "symlink", "version"])
async def test_store_original_bytes_enforces_limits_and_identity(library, unsafe):
    ref = await library.seed()
    stored = await library.store.get(library.scope, ref)
    path = scope_directory(library.root, library.scope) / f"{stored.entry.id}.md"
    if unsafe == "oversized":
        atomic_write_owner_only_bytes(path, b"x" * (16 * 1024 + 1))
    elif unsafe == "symlink":
        if sys.platform == "win32":
            pytest.skip("Symlink creation requires Windows privileges")
        target = path.with_suffix(".original")
        path.rename(target)
        path.symlink_to(target)
    else:
        with pytest.raises(MemoryOperationError, match="version"):
            await library.store.read_version_bytes(library.scope, ref, "0" * 16)
        return
    with pytest.raises(MemoryOperationError):
        await library.store.read_version_bytes(library.scope, ref, stored.version)


def test_main_reports_preflight_kind_without_traceback(tmp_path, monkeypatch, capsys):
    def fail(**kwargs):
        raise ExtractionError("unsupported_extraction_protocol", "Not supported")

    monkeypatch.setattr(review, "bootstrap_runtime", fail)
    assert review.main(["--project", str(tmp_path)]) == 1
    assert capsys.readouterr().err == "unsupported_extraction_protocol: Memory review could not start.\n"


def test_main_preserves_os_error_reason(tmp_path, monkeypatch, capsys):
    def fail(**kwargs):
        raise PermissionError(errno.EACCES, "denied fixture")

    monkeypatch.setattr(review, "bootstrap_runtime", fail)
    assert review.main(["--project", str(tmp_path)]) == 1
    assert "denied fixture" in capsys.readouterr().err
