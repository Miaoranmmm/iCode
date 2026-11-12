# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A failed metadata log must not change the outcome of a committed memory write."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import FrozenInstanceError, dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from chrys.service.memory import operations
from chrys.service.memory.change_log import ChangeOperation, MemoryChange
from chrys.service.memory.operations import MemoryWriter, SaveRequest
from chrys.service.memory.render import render_deleted, render_saved
from chrys.service.memory.storage.file_change_log import FileChangeLog
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import (
    GLOBAL_SCOPE,
    CreateResult,
    MemoryEntryDraft,
    MemoryOperationError,
    MemoryScope,
    MemoryType,
    Quote,
    WrittenBy,
    is_zoned_timestamp,
)
from chrys.service.memory.verification import ConversationView, QuoteClaim, SourceMessage

PROJECT = MemoryScope("project", "0123456789abcdef", root="/private/workspace", anchor="dir")
SOURCE = "session-with-full-unshortened-identity"
VIEW = ConversationView(
    (SourceMessage(SOURCE, 2, ("Use SQLite", "Use transactions", "Forget that entry"), item="user-uuid"),), SOURCE
)


@dataclass
class RecordingLog:
    changes: list[MemoryChange] = field(default_factory=list)

    async def append(self, change: MemoryChange) -> None:
        self.changes.append(change)


@dataclass
class FailingLog:
    error: BaseException
    calls: list[MemoryChange] = field(default_factory=list)

    async def append(self, change: MemoryChange) -> None:
        self.calls.append(change)
        raise self.error


def request(*, body: str = "Use SQLite for durable local state.", words: str = "Use SQLite") -> SaveRequest:
    return SaveRequest(
        "intent", "Use SQLite", body, user_quotes=(QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", words),)
    )


def forget_quote() -> QuoteClaim:
    return QuoteClaim("user_quote", "candidate", "Forget that entry")


@pytest.fixture
def store(tmp_path: Path) -> FileMemoryStore:
    return FileMemoryStore(tmp_path / "memory")


async def test_create_update_and_delete_append_only_nine_metadata_fields_after_commit(store: FileMemoryStore) -> None:
    log = RecordingLog()
    writer = MemoryWriter(store, PROJECT, "model", change_log=log, source=SOURCE)
    before = datetime.now(UTC)
    created = await writer.save(request(), VIEW)
    assert len(log.changes) == 1
    creation = log.changes[0]
    assert (creation.scope, creation.ref, creation.type, creation.operation, creation.written_by, creation.source) == (
        PROJECT,
        created.ref,
        "intent",
        "create",
        "model",
        SOURCE,
    )
    assert (creation.before_version, creation.after_version) == (None, created.version)
    assert creation.time.endswith("Z") and is_zoned_timestamp(creation.time)
    assert before <= datetime.fromisoformat(creation.time) <= datetime.now(UTC)
    assert (await store.get(PROJECT, created.ref)).version == creation.after_version

    update_request = replace(
        request(body="Use transactions for every write.", words="Use transactions"),
        ref=created.ref,
        version=created.version,
    )
    updated = await writer.save(update_request, VIEW)
    assert len(log.changes) == 2
    assert (log.changes[1].operation, log.changes[1].before_version, log.changes[1].after_version) == (
        "update",
        created.version,
        updated.version,
    )
    assert log.changes[1].written_by == "model"
    assert (await store.get(PROJECT, created.ref)).version == updated.version
    deleted = await writer.forget(created.ref, updated.version, forget_quote(), VIEW)
    assert (log.changes[2].operation, log.changes[2].before_version, log.changes[2].after_version) == (
        "delete",
        updated.version,
        None,
    )
    assert await store.list_entries(PROJECT) == []
    assert render_deleted(deleted) == f"Deleted {created.ref} (Use SQLite)."
    assert {description.name for description in fields(creation)} == {
        "time",
        "scope",
        "ref",
        "type",
        "operation",
        "written_by",
        "source",
        "before_version",
        "after_version",
    }
    with pytest.raises(FrozenInstanceError):
        creation.source = "changed"


async def test_exact_duplicate_creation_has_no_change_record(
    store: FileMemoryStore, caplog: pytest.LogCaptureFixture
) -> None:
    log = RecordingLog()
    writer = MemoryWriter(store, PROJECT, "model", change_log=log, source=SOURCE)
    created = await writer.save(request(), VIEW)
    duplicate = await writer.save(request(), VIEW)
    assert duplicate.status == "existing"
    assert (duplicate.ref, duplicate.version) == (created.ref, created.version)
    assert len(log.changes) == 1
    assert caplog.records == []


async def test_global_change_uses_global_scope_and_full_review_run_source(store: FileMemoryStore) -> None:
    log = RecordingLog()
    review_source = "review-run-0123456789-full-id"
    saved = await MemoryWriter(store, None, "extraction", change_log=log, source=review_source).save(
        replace(request(), scope="global"),
        VIEW,
    )
    assert (log.changes[0].scope, log.changes[0].source, log.changes[0].written_by) == (
        GLOBAL_SCOPE,
        review_source,
        "extraction",
    )
    assert log.changes[0].ref == saved.ref


async def test_title_only_update_preserves_entry_writer_while_the_log_records_the_actor(
    store: FileMemoryStore,
) -> None:
    created = await store.create(
        PROJECT, MemoryEntryDraft("intent", "Old title", "Keep body", "user", quotes=(Quote("Use SQLite", "cli"),))
    )
    log = RecordingLog()
    writer = MemoryWriter(store, PROJECT, "model", change_log=log, source=SOURCE)
    updated = await writer.save(
        SaveRequest("intent", "New title", "Keep body", ref=created.ref, version=created.version), VIEW
    )
    # The entry keeps its content author; the change log records who acted.
    assert log.changes[0].written_by == "model"
    assert (await store.get(PROJECT, created.ref)).entry.written_by == "user"
    await writer.forget(created.ref, updated.version, forget_quote(), VIEW)
    assert log.changes[1].written_by == "model"


@pytest.mark.parametrize("source", [None, ""])
async def test_configured_log_without_source_skips_record_with_fixed_warning(
    store: FileMemoryStore, caplog: pytest.LogCaptureFixture, source: str | None
) -> None:
    log = RecordingLog()
    with caplog.at_level(logging.WARNING, logger=operations.__name__):
        saved = await MemoryWriter(store, PROJECT, "model", change_log=log, source=source).save(request(), VIEW)
    assert (await store.get(PROJECT, saved.ref)).version == saved.version
    assert log.changes == []
    assert [record.message for record in caplog.records] == [
        "Memory change logging was skipped because no source was supplied."
    ]


async def test_legacy_writer_without_log_stays_silent(store: FileMemoryStore, caplog: pytest.LogCaptureFixture) -> None:
    saved = await MemoryWriter(store, PROJECT, "model").save(request(), VIEW)
    assert saved.status == "created" and caplog.records == []


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
@pytest.mark.parametrize(
    "error",
    [OSError("private /secret/log.jsonl"), MemoryOperationError("memory_store_unavailable", "private original words")],
    ids=["os-error", "business-error"],
)
async def test_append_failure_is_warning_only_after_successful_write(
    store: FileMemoryStore, caplog: pytest.LogCaptureFixture, operation: str, error: Exception
) -> None:
    log = FailingLog(error)
    baseline = None if operation == "create" else await MemoryWriter(store, PROJECT, "model").save(request(), VIEW)
    writer = MemoryWriter(store, PROJECT, "model", change_log=log, source=SOURCE)
    with caplog.at_level(logging.WARNING, logger=operations.__name__):
        if operation == "create":
            outcome = await writer.save(request(), VIEW)
            assert outcome.status == "created"
            assert render_saved(outcome).startswith(f"Saved {outcome.ref} (intent, version {outcome.version}).")
            assert (await store.get(PROJECT, outcome.ref)).version == outcome.version
        elif operation == "update":
            assert baseline is not None
            outcome = await writer.save(
                replace(
                    request(body="Use transactions", words="Use transactions"),
                    ref=baseline.ref,
                    version=baseline.version,
                ),
                VIEW,
            )
            assert outcome.status == "updated"
            assert (await store.get(PROJECT, outcome.ref)).entry.body == "Use transactions"
        else:
            assert baseline is not None
            deleted = await writer.forget(baseline.ref, baseline.version, forget_quote(), VIEW)
            assert render_deleted(deleted) == f"Deleted {baseline.ref} (Use SQLite)."
            assert await store.list_entries(PROJECT) == []
    assert len(log.calls) == 1
    assert [record.message for record in caplog.records] == ["Memory change logging failed after a successful write."]
    assert all(record.exc_info is None for record in caplog.records)


async def test_metadata_build_failure_keeps_committed_result(
    store: FileMemoryStore, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_build(
        *,
        time: str,
        scope: MemoryScope,
        ref: str,
        type: MemoryType,
        operation: ChangeOperation,
        written_by: WrittenBy,
        source: str,
        before_version: str | None,
        after_version: str | None,
    ) -> MemoryChange:
        raise RuntimeError("private metadata construction details")

    monkeypatch.setattr(operations, "MemoryChange", fail_build)
    log = RecordingLog()
    saved = await MemoryWriter(store, PROJECT, "model", change_log=log, source=SOURCE).save(request(), VIEW)
    assert saved.status == "created"
    assert (await store.get(PROJECT, saved.ref)).version == saved.version
    assert log.changes == []
    assert [record.message for record in caplog.records] == ["Memory change logging failed after a successful write."]


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
async def test_cancellation_during_log_propagates_after_store_commit(
    store: FileMemoryStore, caplog: pytest.LogCaptureFixture, operation: str
) -> None:
    baseline = None if operation == "create" else await MemoryWriter(store, PROJECT, "model").save(request(), VIEW)
    log = FailingLog(asyncio.CancelledError())
    writer = MemoryWriter(store, PROJECT, "model", change_log=log, source=SOURCE)
    with pytest.raises(asyncio.CancelledError):
        if operation == "create":
            await writer.save(request(), VIEW)
        elif operation == "update":
            assert baseline is not None
            await writer.save(
                replace(
                    request(body="Use transactions", words="Use transactions"),
                    ref=baseline.ref,
                    version=baseline.version,
                ),
                VIEW,
            )
        else:
            assert baseline is not None
            await writer.forget(baseline.ref, baseline.version, forget_quote(), VIEW)
    assert len(log.calls) == 1
    stored = await store.list_entries(PROJECT)
    if operation == "delete":
        assert stored == []
    else:
        assert len(stored) == 1
        assert stored[0].entry.body == (request().body if operation == "create" else "Use transactions")
        assert stored[0].version == log.calls[0].after_version
    assert caplog.records == []


class FailingStore(FileMemoryStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.failed_operation: str | None = None
        self.error: BaseException = MemoryOperationError("memory_store_unavailable", "store failed")

    async def create(self, scope: MemoryScope, draft: MemoryEntryDraft) -> CreateResult:
        if self.failed_operation == "create":
            raise self.error
        return await super().create(scope, draft)

    async def update(self, scope: MemoryScope, ref: str, version: str, draft: MemoryEntryDraft) -> str:
        if self.failed_operation == "update":
            raise self.error
        return await super().update(scope, ref, version, draft)

    async def delete(self, scope: MemoryScope, ref: str, version: str) -> None:
        if self.failed_operation == "delete":
            raise self.error
        await super().delete(scope, ref, version)


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
@pytest.mark.parametrize(
    "error",
    [
        OSError("store failed"),
        MemoryOperationError("memory_store_unavailable", "store failed"),
        asyncio.CancelledError(),
    ],
    ids=["os-error", "business-error", "cancelled"],
)
async def test_failed_store_write_never_appends(tmp_path: Path, operation: str, error: BaseException) -> None:
    store = FailingStore(tmp_path / "memory")
    baseline = None if operation == "create" else await MemoryWriter(store, PROJECT, "model").save(request(), VIEW)
    store.failed_operation = operation
    store.error = error
    log = RecordingLog()
    writer = MemoryWriter(store, PROJECT, "model", change_log=log, source=SOURCE)
    with pytest.raises(type(error)):
        if operation == "create":
            await writer.save(request(), VIEW)
        elif operation == "update":
            assert baseline is not None
            await writer.save(
                replace(
                    request(body="Use transactions", words="Use transactions"),
                    ref=baseline.ref,
                    version=baseline.version,
                ),
                VIEW,
            )
        else:
            assert baseline is not None
            await writer.forget(baseline.ref, baseline.version, forget_quote(), VIEW)
    assert log.changes == []


@pytest.mark.parametrize("failure", ["invalid-input", "quote-not-found", "version-conflict"])
async def test_precommit_validation_failures_never_append(store: FileMemoryStore, failure: str) -> None:
    baseline = await MemoryWriter(store, PROJECT, "model").save(request(), VIEW)
    log = RecordingLog()
    writer = MemoryWriter(store, PROJECT, "model", change_log=log, source=SOURCE)
    bad = request()
    if failure == "invalid-input":
        bad = replace(bad, title="")
    elif failure == "quote-not-found":
        bad = request(words="never said")
    else:
        bad = replace(bad, ref=baseline.ref, version="stale")
    with pytest.raises(MemoryOperationError):
        await writer.save(bad, VIEW)
    assert log.changes == []


@dataclass
class CommitCheckingLog:
    store: FileMemoryStore
    checked: list[MemoryChange] = field(default_factory=list)

    async def append(self, change: MemoryChange) -> None:
        if change.operation == "delete":
            assert all(entry.entry.ref != change.ref for entry in await self.store.list_entries(change.scope))
        else:
            assert (await self.store.get(change.scope, change.ref)).version == change.after_version
        self.checked.append(change)


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
async def test_log_append_observes_already_committed_store_state(store: FileMemoryStore, operation: str) -> None:
    baseline = None if operation == "create" else await MemoryWriter(store, PROJECT, "model").save(request(), VIEW)
    log = CommitCheckingLog(store)
    writer = MemoryWriter(store, PROJECT, "model", change_log=log, source=SOURCE)
    if operation == "create":
        await writer.save(request(), VIEW)
    elif operation == "update":
        assert baseline is not None
        await writer.save(
            replace(
                request(body="Use transactions", words="Use transactions"), ref=baseline.ref, version=baseline.version
            ),
            VIEW,
        )
    else:
        assert baseline is not None
        await writer.forget(baseline.ref, baseline.version, forget_quote(), VIEW)
    assert len(log.checked) == 1
    assert log.checked[0].operation == operation


async def test_writer_and_file_change_log_emit_exact_metadata_without_contents(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    store = FileMemoryStore(root)
    writer = MemoryWriter(store, PROJECT, "model", change_log=FileChangeLog(root), source=SOURCE)
    created = await writer.save(request(), VIEW)
    updated = await writer.save(
        replace(request(body="Use transactions", words="Use transactions"), ref=created.ref, version=created.version),
        VIEW,
    )
    await writer.forget(created.ref, updated.version, forget_quote(), VIEW)
    path = root / "projects" / "0123456789abcdef" / "changes.jsonl"
    raw = await asyncio.to_thread(path.read_text, encoding="utf-8")
    records = [json.loads(line) for line in raw.splitlines()]
    assert len(records) == 3
    for record, operation, before, after in zip(
        records,
        ("create", "update", "delete"),
        (None, created.version, updated.version),
        (created.version, updated.version, None),
        strict=True,
    ):
        assert record == {
            "time": record["time"],
            "scope": {"scope": "project", "project_id": "0123456789abcdef"},
            "ref": created.ref,
            "type": "intent",
            "operation": operation,
            "written_by": "model",
            "source": SOURCE,
            "before_version": before,
            "after_version": after,
        }
        assert record["time"].endswith("Z") and is_zoned_timestamp(record["time"])
    assert all(
        secret not in raw
        for secret in (
            "Use SQLite",
            "Use transactions",
            "Forget that entry",
            "applies_when",
            "/private/workspace",
            "user-uuid",
        )
    )
