# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Review persistence order, cancellation drains and failure backoff with offline executors."""

from __future__ import annotations

import asyncio
import json
import stat
import sys
import threading
from dataclasses import replace
from types import ModuleType

import pytest

from chrys.app.memory_hosts import review
from chrys.app.memory_hosts.review_state import load_state
from chrys.foundation.util.lock import FileLock
from chrys.service.memory.extraction import ExtractionError, ProjectRunLock
from chrys.service.memory.storage.store import MemoryInputRejected, Quote
from tests.app.memory_hosts.test_extract import result
from tests.app.memory_hosts.test_review import Offline, report_file, run
from tests.app.memory_hosts.test_review import library as library
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX directory modes")
async def test_review_directories_are_private_even_when_existing(library):
    directories = [
        library.root / "review",
        library.root / "review" / "states",
        library.root / "review" / "reports",
        library.root / "review" / "reports" / str(library.scope.project_id),
    ]
    for path in directories:
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o755)
    report = await run(library, Offline())
    assert report.status == "nothing_to_review"
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o700 for path in directories)


@pytest.mark.parametrize("failure", ["catalog", "preflight"])
async def test_pre_model_failures_are_reported_and_pause_after_three(library, monkeypatch, failure):
    await library.seed()
    await library.seed()
    runner = Offline()
    policy = replace(review.REVIEW_V1, catalog_tokens=1) if failure == "catalog" else review.REVIEW_V1
    reason = "catalog_too_large" if failure == "catalog" else "unsupported_extraction_protocol"

    def rejected(profile, max_output):
        raise ExtractionError(reason, "Offline unsupported profile")

    if failure == "preflight":
        monkeypatch.setattr(review, "prepare_profile", rejected)
    for expected in (1, 2, 3):
        if failure == "preflight":
            with pytest.raises(ExtractionError):
                await run(library, runner, automatic=True, policy=policy)
        else:
            assert (await run(library, runner, automatic=True, policy=policy)).status == reason
        state = load_state(library.root, str(library.scope.project_id))
        assert state.failures == expected
        assert state.snapshot == {} and state.success_at == 0
    reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in (library.root / "review" / "reports").rglob("*.json")
    ]
    assert len(reports) == 3
    assert all(report["status"] == reason and report["counts"]["failure_count"] == 1 for report in reports)
    assert (await run(library, runner, automatic=True, policy=policy)).status == "paused"
    assert runner.calls == 0 and state.paused


async def test_run_lock_timeout_preserves_the_holders_lock(library, monkeypatch):
    root = library.root / "auto-extract"
    root.mkdir(parents=True)
    monkeypatch.setattr(review, "RUN_LOCK_TIMEOUT_SECONDS", 0)
    runner = Offline()
    with FileLock(root / "run.lock"):
        report = await run(library, runner)
        assert report.status == "run_lock_timeout" and runner.calls == 0
        with pytest.raises(TimeoutError), FileLock(root / "run.lock", timeout=0):
            pytest.fail("Review released the holder's lock")
    with FileLock(root / "run.lock", timeout=0):
        pass
    assert json.loads(report_file(library, report).read_text(encoding="utf-8"))["status"] == "run_lock_timeout"


async def test_withdrawn_entry_is_excluded_and_cannot_be_merged(library):
    await library.seed()
    await library.seed()
    ref = await library.seed()
    withdrawn = await library.store.get(library.scope, ref)
    version = await library.store.update(
        library.scope,
        ref,
        withdrawn.version,
        replace(
            withdrawn.entry.draft(),
            status="withdrawn",
            status_basis=Quote("Withdraw this rule", "message", "user", at="2026-11-13T00:00:00Z"),
        ),
    )

    class Checked(Offline):
        async def run(self, prompt, reader, writer, policy):
            catalog = prompt.split("</review_catalog>")[0]
            assert ref not in catalog
            active = (await reader.search("", "all")).entries
            assert len(active) == 2 and all(item.entry.ref != ref for item in active)
            keep = active[0]
            await writer.read(reader, keep.entry.ref)
            await writer.read(reader, ref)
            with pytest.raises(MemoryInputRejected):
                await writer.merge(keep.entry.ref, keep.version, ref, version)
            return result()

    report = await run(library, Checked())
    assert report.entry_count == 2 and report.status == "completed"
    assert (await library.store.get(library.scope, ref)).version == version
    assert len(await library.store.list_entries(library.scope)) == 3
    assert ref not in load_state(library.root, report.project_id).snapshot


async def test_report_and_state_writes_share_fifo_order(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    actual: list[str] = []
    calls = 0

    async def controlled_to_thread(write):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
        write()

    shadow = ModuleType("asyncio")
    shadow.__dict__.update(vars(asyncio))
    shadow.to_thread = controlled_to_thread
    monkeypatch.setattr(review, "asyncio", shadow)
    writes = review.ReviewWrites()
    first = asyncio.create_task(writes.submit(lambda: actual.append("report-old")))
    second = None
    try:
        await wait_for(entered.is_set, timeout=ENGINE_TURN_TIMEOUT)
        second = asyncio.create_task(writes.submit(lambda: actual.append("state-new")))
        await wait_for(lambda: len(writes.pending) == 2, timeout=ENGINE_TURN_TIMEOUT)
        await asyncio.sleep(0)
        assert calls == 1 and actual == []
        release.set()
        await asyncio.gather(first, second)
        await writes.drain()
        assert actual == ["report-old", "state-new"]
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second is not None else []), return_exceptions=True)


@pytest.mark.parametrize("blocked_write", ["report", "state"])
async def test_cancelled_write_drains_before_unlock_and_cannot_overwrite_next_run(library, monkeypatch, blocked_write):
    await library.seed()
    await library.seed()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    real_report, real_state = review.atomic_write_owner_only_bytes, review.save_state
    blocked = False
    queues: list[review.ReviewWrites] = []

    class TrackedWrites(review.ReviewWrites):
        def __init__(self):
            super().__init__()
            queues.append(self)

    monkeypatch.setattr(review, "ReviewWrites", TrackedWrites)

    def wait_once() -> bool:
        nonlocal blocked
        if blocked:
            return False
        blocked = True
        entered.set()
        if not release.wait(timeout=ENGINE_TURN_TIMEOUT):
            raise TimeoutError("Test did not release the admitted write")
        return True

    def report_write(path, payload, *, create_parents=True):
        first = blocked_write == "report" and wait_once()
        try:
            real_report(path, payload, create_parents=create_parents)
        finally:
            if first:
                finished.set()

    def state_write(root, project_id, state):
        first = blocked_write == "state" and wait_once()
        try:
            real_state(root, project_id, state)
        finally:
            if first:
                finished.set()

    monkeypatch.setattr(review, "atomic_write_owner_only_bytes", report_write)
    monkeypatch.setattr(review, "save_state", state_write)
    task = asyncio.create_task(run(library, Offline()))
    try:
        await wait_for(entered.is_set, timeout=ENGINE_TURN_TIMEOUT)
        admitted = next(iter(queues[0].pending))
        task.cancel()
        # The cancellation reaches the waiter, but cannot finish the admitted write.
        await asyncio.sleep(0)
        assert not admitted.done() and not task.done()
        with pytest.raises(TimeoutError), FileLock(library.root / "auto-extract" / "run.lock", timeout=0):
            pytest.fail("Global lock was released before the write drained")
        contender = ProjectRunLock(library.root, library.scope)
        acquired = await contender.try_acquire()
        try:
            assert not acquired
        finally:
            if acquired:
                await contender.release()
        release.set()
        await wait_for(task.done, timeout=ENGINE_TURN_TIMEOUT)
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if entered.is_set():
            await wait_for(finished.is_set, timeout=ENGINE_TURN_TIMEOUT)
    state = load_state(library.root, str(library.scope.project_id))
    assert state.snapshot == {} and state.failures == 1
    report_path = next((library.root / "review" / "reports").rglob("*.json"))
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "failed"
    next_run = await run(library, Offline("failure"))
    state = load_state(library.root, next_run.project_id)
    assert state.snapshot == {} and state.failures == 2
