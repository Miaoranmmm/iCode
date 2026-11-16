# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Whole-library review contracts using private real stores and offline executors."""

from __future__ import annotations

import asyncio
import hashlib
import json
import stat
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from chrys.app.memory_hosts import executor_icode
from chrys.app.memory_hosts.extract import ExtractionOptions, run_extraction
from chrys.app.memory_hosts.review import REVIEW_V1, ReviewWriter, load_state, run_review
from chrys.foundation.platform.files import atomic_write_owner_only_bytes
from chrys.foundation.util.lock import FileLock
from chrys.service.memory.extraction import ExtractionResult, ProjectRunLock, SessionKey, SourceRevision, SourceSession
from chrys.service.memory.operations import MemoryReader, SaveRequest
from chrys.service.memory.project import bind_project
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.layout import scope_directory
from chrys.service.memory.storage.store import (
    GLOBAL_SCOPE,
    MemoryEntryDraft,
    MemoryInputRejected,
    MemoryScope,
    MergeResult,
    Quote,
)
from chrys.service.memory.verification import ConversationView, QuoteClaim, SourceMessage
from tests.app.memory_hosts.executor_fakes import ClientFactory, ScriptedWire, call, response
from tests.app.memory_hosts.test_extract import PROFILE, Executor, Reader, result
from tests.support.secure_files import plant_owner_only_bytes
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for


@dataclass
class Library:
    project: Path
    root: Path
    scope: MemoryScope
    store: FileMemoryStore
    refs: list[str] = field(default_factory=list)

    async def seed(
        self, *, scope: MemoryScope | None = None, quotes: int = 1, body: str = "Keep rules precise."
    ) -> str:
        index = len(self.refs)
        saved = await self.store.create(
            scope or self.scope,
            MemoryEntryDraft(
                "intent",
                f"Rule {index}",
                body,
                "extraction",
                quotes=tuple(Quote(f"Words {index} {q}", "message", f"session-{index}-{q}") for q in range(quotes)),
            ),
        )
        self.refs.append(saved.ref)
        return saved.ref


@pytest.fixture
async def library(tmp_path: Path) -> Library:
    project = tmp_path / "project"
    await asyncio.to_thread(project.mkdir)
    return Library(
        project,
        tmp_path / "memory",
        await asyncio.to_thread(bind_project, str(project)),
        FileMemoryStore(tmp_path / "memory"),
    )


@dataclass
class Offline:
    action: str = "none"
    calls: int = 0

    async def run(self, prompt: str, reader: MemoryReader, writer: ReviewWriter, policy) -> ExtractionResult:
        self.calls += 1
        entries = (await reader.search("", "all")).entries
        a, b = entries[:2]
        if self.action == "failure":
            return result("failed")
        if self.action == "limited":
            return result("turn_limited")
        if self.action in {"merge", "concurrent", "concurrent_drop", "partial", "overflow"}:
            await writer.read(reader, a.entry.ref)
            await writer.read(reader, b.entry.ref)
            if self.action in {"concurrent", "concurrent_drop"}:
                item = a if self.action == "concurrent" else b
                new = replace(item.entry.draft(), title="Changed externally")
                version = await writer._store.update(writer._project, item.entry.ref, item.version, new)
                if self.action == "concurrent":
                    a = replace(a, version=version)
                else:
                    b = replace(b, version=version)
                await writer.read(reader, item.entry.ref)
            if self.action == "overflow":
                with pytest.raises(MemoryInputRejected):
                    await writer.merge(a.entry.ref, a.version, b.entry.ref, b.version)
            else:
                await writer.merge(a.entry.ref, a.version, b.entry.ref, b.version)
        elif self.action == "conflict":
            await writer.read(reader, a.entry.ref)
            await writer.read(reader, b.entry.ref)
            writer.note_conflict(
                a.entry.ref, a.version, b.entry.ref, b.version, "Different scopes; suspected duplicate"
            )
        return result()


async def run(library: Library, executor: Offline, **options):
    return await run_review(library.project, library.root, PROFILE, executor=executor, **options)


def report_file(library: Library, report):
    return library.root / "review" / "reports" / str(library.scope.project_id) / f"{report.run_id}.json"


async def test_empty_and_single_entry_never_start_model(library):
    runner = Offline()
    assert (await run(library, runner)).status == "nothing_to_review"
    await library.seed()
    assert (await run(library, runner)).status == "nothing_to_review"
    assert runner.calls == 0


async def test_zero_calls_success_snapshot_skip_and_policy_upgrade(library):
    await library.seed()
    await library.seed()
    runner = Offline()
    first = await run(library, runner)
    assert first.status == "completed" and runner.calls == 1
    assert (await run(library, runner)).status == "nothing_to_review" and runner.calls == 1
    assert (await run(library, runner, policy=replace(REVIEW_V1, version="review-v2"))).status == "completed"
    assert runner.calls == 2
    saved = load_state(library.root, str(library.scope.project_id))
    assert saved.policy == "review-v2" and saved.success_at > 0 and saved.failures == 0


async def test_successful_merge_report_preserves_exact_drop_bytes_and_log_source(library):
    await library.seed()
    await library.seed()
    original = {}
    for item in await library.store.list_entries(library.scope):
        path = scope_directory(library.root, library.scope) / f"{item.entry.id}.md"
        payload = path.read_bytes().replace(b"---\n", b"---\n# preserve this original formatting\n", 1)
        atomic_write_owner_only_bytes(path, payload)
        original[item.entry.ref] = payload.decode("utf-8")
    report = await run(library, Offline("merge"))
    assert report.status == "completed" and len(report.merges) == 1
    merge = report.merges[0]
    assert merge.drop_original == original[merge.drop]
    assert merge.drop_original.startswith("---\n") and "Keep rules precise." in merge.drop_original
    actual = {item.entry.ref: item.version for item in await library.store.list_entries(library.scope)}
    assert len(actual) == 1
    saved = load_state(library.root, str(library.scope.project_id))
    assert saved.snapshot == actual
    records = [
        json.loads(line)
        for line in (scope_directory(library.root, library.scope) / "changes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(r["operation"], r["source"]) for r in records] == [("merge", report.run_id), ("delete", report.run_id)]
    disk = json.loads(report_file(library, report).read_text(encoding="utf-8"))
    assert disk["merges"][0]["drop_original"] == original[merge.drop]
    if sys.platform != "win32":
        for path in (
            report_file(library, report),
            library.root / "review" / "states" / f"{library.scope.project_id}.json",
        ):
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert (await run(library, Offline())).status == "nothing_to_review"


async def test_file_store_version_is_original_file_sha256_prefix(library):
    await library.seed()
    item = (await library.store.list_entries(library.scope))[0]
    path = scope_directory(library.root, library.scope) / f"{item.entry.id}.md"
    original = path.read_bytes()
    assert item.version == hashlib.sha256(original).hexdigest()[:16]
    edited = original.replace(b"---\n", b"---\n# formatting changes also change the version\n", 1)
    atomic_write_owner_only_bytes(path, edited)
    reloaded = await library.store.get(library.scope, item.entry.ref)
    assert reloaded.entry == item.entry
    assert reloaded.version == hashlib.sha256(edited).hexdigest()[:16]
    assert reloaded.version != item.version


@pytest.mark.parametrize("mode", ["unread", "truncated", "stale"])
async def test_merge_requires_full_exact_version_reads(library, mode, monkeypatch):
    if mode == "truncated":
        from chrys.service.memory import render

        monkeypatch.setattr(render, "_READ_BUDGET", 100)
    await library.seed()
    await library.seed()

    class Checked(Offline):
        async def run(self, prompt, reader, writer, policy):
            a, b = (await reader.search("", "all")).entries
            if mode != "unread":
                await writer.read(reader, a.entry.ref)
                await writer.read(reader, b.entry.ref)
            if mode == "stale":
                a = replace(a, version="0" * 16)
            with pytest.raises(MemoryInputRejected):
                await writer.merge(a.entry.ref, a.version, b.entry.ref, b.version)
            return result()

    assert (await run(library, Checked())).status == "completed"
    assert len(await library.store.list_entries(library.scope)) == 2


async def test_review_writer_rejects_save_withdraw_forget(library):
    await library.seed()
    await library.seed()

    class Checked(Offline):
        async def run(self, prompt, reader, writer, policy):
            a = (await reader.search("", "all")).entries[0]
            claim = QuoteClaim("user_quote", "candidate", "Authorize new rule", session="s")
            view = ConversationView((SourceMessage("s", 1, (claim.text,), at="2026-11-13T12:00:00Z"),), None)
            with pytest.raises(MemoryInputRejected):
                await writer.save(SaveRequest("intent", "New rule", "New body", user_quotes=(claim,)), view)
            with pytest.raises(MemoryInputRejected):
                await writer.withdraw(a.entry.ref, a.version, claim, view)
            with pytest.raises(MemoryInputRejected):
                await writer.forget(a.entry.ref, a.version, claim, view)
            return result()

    report = await run(library, Checked())
    assert report.status == "completed" and len(await library.store.list_entries(library.scope)) == 2


async def test_merge_over_limit_retains_both(library):
    await library.seed(quotes=2)
    await library.seed(quotes=2)
    report = await run(library, Offline("overflow"))
    assert report.status == "completed" and report.merges[0].status == "rejected"
    assert len(await library.store.list_entries(library.scope)) == 2


@pytest.mark.parametrize("action", ["concurrent", "concurrent_drop"])
async def test_external_new_version_merge_not_absorbed_into_checkpoint(library, action):
    await library.seed()
    await library.seed()
    start = {item.entry.ref: item.version for item in await library.store.list_entries(library.scope)}
    report = await run(library, Offline(action))
    assert report.status == "completed" and report.merges[0].status == "completed"
    assert load_state(library.root, str(library.scope.project_id)).snapshot == start
    assert len(await library.store.list_entries(library.scope)) == 1


async def test_partial_merge_fails_without_advancing_snapshot(library, monkeypatch):
    await library.seed()
    await library.seed()
    real = FileMemoryStore.merge

    async def partial(store, scope, keep_ref, keep_version, drop_ref, drop_version, draft):
        merged = await real(store, scope, keep_ref, keep_version, drop_ref, drop_version, draft)
        return MergeResult(merged.version, False)

    monkeypatch.setattr(FileMemoryStore, "merge", partial)
    report = await run(library, Offline("partial"))
    saved = load_state(library.root, str(library.scope.project_id))
    assert report.status == "failed" and saved.snapshot == {} and saved.failures == 1
    assert report.merges[0].status == "partial" and report.merges[0].drop_original


@pytest.mark.parametrize("action", ["failure", "limited"])
async def test_three_failures_pause_until_snapshot_changes(library, action):
    await library.seed()
    await library.seed()
    runner = Offline(action)
    for _ in range(3):
        assert (await run(library, runner, automatic=True)).status == "failed"
    assert (await run(library, runner, automatic=True)).status == "paused" and runner.calls == 3
    await library.seed()
    assert (await run(library, runner, automatic=True)).status == "failed" and runner.calls == 4
    assert load_state(library.root, str(library.scope.project_id)).failures == 1


async def test_catalog_budget_prevents_model_and_unreadable_entries_are_reported(library):
    await library.seed()
    await library.seed()
    runner = Offline()
    assert (await run(library, runner, policy=replace(REVIEW_V1, catalog_tokens=1))).status == "catalog_too_large"
    assert runner.calls == 0
    path = scope_directory(library.root, library.scope) / "abcdef123456.md"
    plant_owner_only_bytes(path, b"broken entry")
    report = await run(library, runner)
    assert report.status == "completed_with_gaps" and len(report.unreadable) == 1
    assert "project:abcdef123456" not in load_state(library.root, str(library.scope.project_id)).snapshot


async def test_cross_scope_conflicts_are_report_only_and_require_read_versions(library):
    await library.seed()
    await library.seed(scope=GLOBAL_SCOPE)
    report = await run(library, Offline("conflict"))
    assert report.status == "completed" and len(report.conflicts) == len(report.cross_scope) == 1
    assert (
        len(await library.store.list_entries(library.scope)) == len(await library.store.list_entries(GLOBAL_SCOPE)) == 1
    )


async def test_review_holds_both_locks_and_refuses_running_extraction(library):
    await library.seed()
    await library.seed()
    project_lock = ProjectRunLock(library.root, library.scope)
    assert await project_lock.try_acquire()
    try:
        assert (await run(library, Offline())).status == "already_running"
    finally:
        await project_lock.release()
    entered, release = asyncio.Event(), asyncio.Event()

    class Gated(Offline):
        async def run(self, prompt, reader, writer, policy):
            entered.set()
            await release.wait()
            return result()

    task = asyncio.create_task(run(library, Gated()))
    try:
        await wait_for(lambda: entered.is_set() or task.done(), timeout=ENGINE_TURN_TIMEOUT)
        assert entered.is_set()
        contender = ProjectRunLock(library.root, library.scope)
        assert not await contender.try_acquire()
        source = SourceSession(
            SessionKey("claude-code", "offline-session"),
            library.scope,
            SourceRevision(1, "0" * 64),
            (SourceMessage("offline-session", 1, ("Keep rules precise.",)),),
        )
        extraction = Executor(result())
        blocked = await run_extraction(
            ExtractionOptions(library.project, library.root),
            PROFILE,
            readers={"claude-code": Reader((source,))},
            executor=extraction,
        )
        assert blocked.status == "already_running" and extraction.batches == []
        with pytest.raises(TimeoutError):
            await asyncio.to_thread(FileLock(library.root / "auto-extract" / "run.lock", timeout=0).acquire)
    finally:
        release.set()
        await task


async def test_native_runner_allows_zero_calls_and_owns_client(library, monkeypatch):
    await library.seed()
    await library.seed()
    wire = ScriptedWire([response(text="Done")])
    factory = ClientFactory(wire)
    monkeypatch.setattr(executor_icode, "create_client", factory)
    report = await run_review(library.project, library.root, PROFILE)
    assert report.status == "completed" and report.counts["call_count"] == 0
    assert wire.closed == 1


async def test_native_review_tools_read_merge_and_conflict(library, monkeypatch):
    await library.seed()
    await library.seed()
    a, b = await library.store.list_entries(library.scope)
    wire = ScriptedWire(
        [
            response(call("memory_read", {"ref": a.entry.ref})),
            response(call("memory_read", {"ref": b.entry.ref})),
            response(
                call(
                    "memory_note_conflict",
                    {
                        "ref_a": a.entry.ref,
                        "version_a": a.version,
                        "ref_b": b.entry.ref,
                        "version_b": b.version,
                        "reason": "Could be distinct",
                    },
                )
            ),
            response(
                call(
                    "memory_merge",
                    {
                        "keep_ref": a.entry.ref,
                        "keep_version": a.version,
                        "drop_ref": b.entry.ref,
                        "drop_version": b.version,
                    },
                )
            ),
            response(text="Done"),
        ]
    )
    monkeypatch.setattr(executor_icode, "create_client", ClientFactory(wire))
    report = await run_review(library.project, library.root, PROFILE)
    assert report.status == "completed" and report.counts["call_count"] == 4
    assert len(report.merges) == len(report.conflicts) == 1


async def test_own_merge_chain_is_absorbed_without_retriggering(library):
    for _ in range(3):
        await library.seed()

    class Chain(Offline):
        async def run(self, prompt, reader, writer, policy):
            a, b, c = (await reader.search("", "all")).entries
            for item in (a, b, c):
                await writer.read(reader, item.entry.ref)
            merged = await writer.merge(a.entry.ref, a.version, b.entry.ref, b.version)
            await writer.read(reader, a.entry.ref)
            await writer.merge(a.entry.ref, merged.version, c.entry.ref, c.version)
            return result()

    assert (await run(library, Chain())).status == "completed"
    actual = {item.entry.ref: item.version for item in await library.store.list_entries(library.scope)}
    assert load_state(library.root, str(library.scope.project_id)).snapshot == actual


async def test_basis_never_reads_external_files_and_conflicts_use_seen_versions(library, monkeypatch):
    await library.seed()
    await library.seed()
    from chrys.service.memory import operations

    def forbidden(evidence, project, project_root):
        raise AssertionError("Review may not inspect external basis files")

    monkeypatch.setattr(operations, "check_basis", forbidden)

    class Checked(Offline):
        async def run(self, prompt, reader, writer, policy):
            a, b = (await reader.search("", "all")).entries
            with pytest.raises(MemoryInputRejected):
                writer.note_conflict(a.entry.ref, a.version, b.entry.ref, b.version, "Unseen")
            for item in (a, b):
                await writer.read(reader, item.entry.ref)
            with pytest.raises(MemoryInputRejected):
                writer.note_conflict(a.entry.ref, "0" * 16, b.entry.ref, b.version, "Stale")
            writer.note_conflict(a.entry.ref, a.version, b.entry.ref, b.version, "Uncertain")
            return result()

    report = await run(library, Checked())
    assert report.status == "completed" and len(report.conflicts) == 1


@pytest.mark.parametrize("rejected", [False, True])
async def test_native_runner_handles_timeout_and_safe_rejections(library, monkeypatch, rejected):
    await library.seed(quotes=2)
    await library.seed(quotes=2)
    a, b = await library.store.list_entries(library.scope)
    if rejected:
        script = [
            response(call("memory_read", {"ref": a.entry.ref})),
            response(call("memory_read", {"ref": b.entry.ref})),
            response(
                call(
                    "memory_merge",
                    {
                        "keep_ref": a.entry.ref,
                        "keep_version": a.version,
                        "drop_ref": b.entry.ref,
                        "drop_version": b.version,
                    },
                )
            ),
            response(text="Done"),
        ]
    else:

        async def stalled(messages):
            await asyncio.Event().wait()
            return response(text="unreachable")

        script = [stalled]
    wire = ScriptedWire(script)
    monkeypatch.setattr(executor_icode, "create_client", ClientFactory(wire))
    policy = REVIEW_V1 if rejected else replace(REVIEW_V1, timeout_seconds=0.02)
    report = await run_review(library.project, library.root, PROFILE, policy=policy)
    assert report.status == ("completed" if rejected else "failed")
    assert len(await library.store.list_entries(library.scope)) == 2 and wire.closed == 1
    if rejected:
        assert report.counts["rejection_count"] == 1


async def test_native_review_treats_kernel_argument_errors_as_rejections(library, monkeypatch):
    await library.seed(quotes=2)
    await library.seed(quotes=2)
    wire = ScriptedWire([response(call("memory_read", {"ref": 7})), response(text="Done")])
    monkeypatch.setattr(executor_icode, "create_client", ClientFactory(wire))
    report = await run_review(library.project, library.root, PROFILE, policy=REVIEW_V1)
    assert report.status == "completed" and len(wire.requests) == 2
    assert (report.counts["failure_count"], report.counts["rejection_count"]) == (0, 1)
