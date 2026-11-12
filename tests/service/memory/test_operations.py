# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Memory operation behavior from the step-2 specification, against real files."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from chrys.service.memory.operations import MemoryReader, MemoryWriter, SaveRequest
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import (
    GLOBAL_SCOPE,
    MAX_APPLIES_WHEN,
    MAX_BODY,
    MAX_QUOTES,
    MAX_TITLE,
    CatalogItem,
    Context,
    CreateResult,
    MemoryEntryDraft,
    MemoryOperationError,
    MemoryScope,
    ProjectInfo,
    Quote,
    ReferenceEvidence,
    StoredMemory,
    UnreadableMemory,
)
from chrys.service.memory.verification import ConversationView, QuoteClaim, SourceMessage

PROJECT = MemoryScope("project", "0123456789abcdef")
VIEW = ConversationView((SourceMessage("current", 3, ("use SQLite", "new decision", "forget this")),), "current")
EMPTY_VIEW = ConversationView((), "current")


def quote(text: str = "use SQLite", session: str | None = None) -> QuoteClaim:
    return QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", text, session=session)


def request() -> SaveRequest:
    return SaveRequest("intent", "Use SQLite", "Use the local database.", "queue decisions", user_quotes=(quote(),))


@pytest.fixture
def store(tmp_path: Path) -> FileMemoryStore:
    return FileMemoryStore(tmp_path / "memory")


async def test_create_and_duplicate_return_structured_outcomes(store: FileMemoryStore) -> None:
    writer = MemoryWriter(store, PROJECT, "model")
    first = await writer.save(request(), VIEW)
    second = await writer.save(request(), VIEW)
    assert first.status == "created"
    assert second.status == "existing"
    assert (second.ref, second.version) == (first.ref, first.version)
    assert (first.type, first.title, first.applies_when) == ("intent", "Use SQLite", "queue decisions")
    assert first.quotes == (Quote("use SQLite", "message", "current"),)
    assert len(await store.list_entries(PROJECT)) == 1


@pytest.mark.parametrize(
    "bad",
    [
        replace(request(), type="engineering"),
        replace(request(), user_quotes=()),
        replace(request(), user_quotes=None),
        replace(request(), user_quotes=(quote(),) * (MAX_QUOTES + 1)),
        replace(request(), title="x" * (MAX_TITLE + 1)),
        replace(request(), body="x" * (MAX_BODY + 1)),
        replace(request(), applies_when="x" * (MAX_APPLIES_WHEN + 1)),
        replace(request(), title=""),
        replace(request(), body=""),
        replace(request(), version="version-without-ref"),
        replace(request(), ref="project:0123456789ab"),
        replace(request(), ref="bad", version="v"),
        replace(request(), ref="project:0123456789ab", version="v", scope="global"),
    ],
)
async def test_invalid_save_never_writes(store: FileMemoryStore, bad: SaveRequest) -> None:
    with pytest.raises(MemoryOperationError) as caught:
        await MemoryWriter(store, PROJECT, "model").save(bad, VIEW)
    assert caught.value.kind == "memory_invalid_input"
    assert await store.list_entries(PROJECT) == []
    assert await store.list_entries(GLOBAL_SCOPE) == []


async def test_input_errors_are_listed_together(store: FileMemoryStore) -> None:
    bad = replace(request(), title="x" * 81, body="x" * 1501, applies_when="x" * 201, user_quotes=())
    with pytest.raises(MemoryOperationError) as caught:
        await MemoryWriter(store, PROJECT, "model").save(bad, VIEW)
    for name in ("title", "body", "applies_when", "user_quotes"):
        assert name in caught.value.message


async def test_batch_missing_session_is_rejected_before_update_lookup(store: FileMemoryStore) -> None:
    bad = replace(request(), ref="project:0123456789ab", version="v")
    with pytest.raises(MemoryOperationError) as caught:
        await MemoryWriter(store, PROJECT, "extraction").save(bad, ConversationView((), None))
    assert caught.value.kind == "memory_invalid_input"


@pytest.mark.parametrize("session", [None, ""])
async def test_batch_create_requires_quote_session(store: FileMemoryStore, session: str | None) -> None:
    view = ConversationView((SourceMessage("source-full-id", 4, ("use SQLite",), item="source-item"),), None)
    submitted = replace(request(), user_quotes=(quote(session=session),))
    with pytest.raises(MemoryOperationError) as caught:
        await MemoryWriter(store, PROJECT, "extraction").save(submitted, view)
    assert caught.value.kind == "memory_invalid_input"
    assert caught.value.message == "Invalid memory input: user_quotes[0].session is required in batch mode."
    assert await store.list_entries(PROJECT) == []
    assert await store.list_entries(GLOBAL_SCOPE) == []


async def test_batch_update_requires_session_even_for_an_unchanged_stored_cli_quote(store: FileMemoryStore) -> None:
    original = MemoryEntryDraft(
        "intent", "Use SQLite", "Use the local database.", "user", quotes=(Quote("use SQLite", "cli"),)
    )
    saved = await store.create(PROJECT, original)
    before = await store.get(PROJECT, saved.ref)
    submitted = SaveRequest(
        "intent", "Use SQLite locally", original.body, user_quotes=(quote(),), ref=saved.ref, version=saved.version
    )
    with pytest.raises(MemoryOperationError) as caught:
        await MemoryWriter(store, PROJECT, "extraction").save(submitted, ConversationView((), None))
    assert caught.value.kind == "memory_invalid_input"
    assert caught.value.message == "Invalid memory input: user_quotes[0].session is required in batch mode."
    assert await store.get(PROJECT, saved.ref) == before


async def test_version_conflict_precedes_quote_verification(store: FileMemoryStore) -> None:
    writer = MemoryWriter(store, PROJECT, "model")
    saved = await writer.save(request(), VIEW)
    changed = replace(request(), ref=saved.ref, version="stale", body="Changed.", user_quotes=(quote("not said"),))
    with pytest.raises(MemoryOperationError) as caught:
        await writer.save(changed, EMPTY_VIEW)
    assert caught.value.kind == "memory_version_conflict"
    assert (await store.get(PROJECT, saved.ref)).version == saved.version


@pytest.mark.parametrize("operation", ["update", "forget"])
async def test_version_conflict_reports_current_version(store: FileMemoryStore, operation: str) -> None:
    writer = MemoryWriter(store, PROJECT, "model")
    saved = await writer.save(request(), VIEW)
    with pytest.raises(MemoryOperationError) as caught:
        if operation == "update":
            await writer.save(replace(request(), ref=saved.ref, version="stale"), VIEW)
        else:
            await writer.forget(saved.ref, "stale", QuoteClaim("user_quote", "candidate", "forget this"), VIEW)
    assert caught.value.kind == "memory_version_conflict"
    assert caught.value.message == (
        f"{saved.ref} changed since it was read (now version {saved.version}); read it again before changing it."
    )


@pytest.mark.parametrize("source", ["context", "evidence", "neither"])
async def test_title_only_update_preserves_all_sources_and_writer(store: FileMemoryStore, source: str) -> None:
    original = MemoryEntryDraft(
        "intent",
        "Old title",
        "same body",
        "user",
        "same\napplies",
        (Quote("old quote", "remember", "old-session", "2026-11-02T08:05:40Z", "old-item"),),
        Context("proposal", "old-session") if source == "context" else None,
        (ReferenceEvidence("document", "note"),) if source == "evidence" else (),
    )
    saved = await store.create(PROJECT, original)
    outcome = await MemoryWriter(store, PROJECT, "model").save(
        SaveRequest("intent", "New title", original.body, original.applies_when, ref=saved.ref, version=saved.version),
        EMPTY_VIEW,
    )
    updated = await store.get(PROJECT, saved.ref)
    assert outcome.status == "updated"
    assert outcome.version == updated.version
    assert updated.entry.draft() == replace(original, title="New title")


@pytest.mark.parametrize("source", ["context", "evidence"])
@pytest.mark.parametrize("change", ["body", "applies_when", "user_quotes"])
async def test_entries_with_existing_sources_can_be_updated(store: FileMemoryStore, source: str, change: str) -> None:
    original = MemoryEntryDraft(
        "intent",
        "Old title",
        "same body",
        "model",
        "same applies",
        (Quote("use SQLite", "message", "current"),),
        Context("proposal", "current") if source == "context" else None,
        (ReferenceEvidence("document"),) if source == "evidence" else (),
    )
    saved = await store.create(PROJECT, original)
    update = SaveRequest(
        "intent", "New title", original.body, original.applies_when, ref=saved.ref, version=saved.version
    )
    if change == "body":
        update = replace(update, body="changed body", user_quotes=(quote("new decision"),))
    elif change == "applies_when":
        update = replace(update, applies_when="changed applies", user_quotes=(quote("new decision"),))
    else:
        update = replace(update, user_quotes=(quote(),))
    outcome = await MemoryWriter(store, PROJECT, "model").save(update, VIEW)
    stored = await store.get(PROJECT, saved.ref)
    assert outcome.status == "updated"
    assert (stored.entry.title, stored.entry.body, stored.entry.applies_when) == (
        update.title,
        update.body,
        update.applies_when,
    )
    assert stored.version == outcome.version


async def test_substantive_update_uses_new_words_and_writer(store: FileMemoryStore) -> None:
    saved = await MemoryWriter(store, PROJECT, "user").save(request(), VIEW)
    update = replace(
        request(),
        ref=saved.ref,
        version=saved.version,
        body="Follow the new decision.",
        user_quotes=(quote("new decision"),),
    )
    result = await MemoryWriter(store, PROJECT, "extraction").save(update, VIEW)
    stored = await store.get(PROJECT, saved.ref)
    assert result.status == "updated"
    assert stored.entry.body == update.body
    assert stored.entry.quotes == (Quote("new decision", "message", "current"),)
    assert stored.entry.written_by == "extraction"


async def test_changed_body_requires_quotes(store: FileMemoryStore) -> None:
    writer = MemoryWriter(store, PROJECT, "model")
    saved = await writer.save(request(), VIEW)
    update = replace(request(), ref=saved.ref, version=saved.version, body="Changed.", user_quotes=None)
    with pytest.raises(MemoryOperationError) as caught:
        await writer.save(update, VIEW)
    assert caught.value.kind == "memory_invalid_input"
    assert (await store.get(PROJECT, saved.ref)).version == saved.version


async def test_existing_type_cannot_be_changed_to_intent(store: FileMemoryStore) -> None:
    saved = await store.create(
        PROJECT, MemoryEntryDraft("engineering", "Finding", "Observed.", "model", evidence=(ReferenceEvidence("doc"),))
    )
    with pytest.raises(MemoryOperationError) as caught:
        await MemoryWriter(store, PROJECT, "model").save(replace(request(), ref=saved.ref, version=saved.version), VIEW)
    assert caught.value.kind == "memory_invalid_input"
    assert (await store.get(PROJECT, saved.ref)).entry.type == "engineering"


async def test_global_ref_routes_update_and_forget_to_global_store(store: FileMemoryStore) -> None:
    writer = MemoryWriter(store, PROJECT, "model")
    saved = await writer.save(replace(request(), scope="global"), VIEW)
    updated = await writer.save(
        replace(request(), ref=saved.ref, version=saved.version, title="Global title", user_quotes=None), EMPTY_VIEW
    )
    assert (await store.get(GLOBAL_SCOPE, saved.ref)).entry.title == "Global title"
    assert await store.list_entries(PROJECT) == []
    outcome = await writer.forget(
        saved.ref, updated.version, QuoteClaim("user_quote", "candidate", "forget this"), VIEW
    )
    assert (outcome.ref, outcome.title) == (saved.ref, "Global title")
    assert await store.list_entries(GLOBAL_SCOPE) == []


@pytest.mark.parametrize(
    "version,text,kind",
    [("current", "not said", "memory_quote_not_found"), ("wrong", "not said", "memory_version_conflict")],
)
async def test_failed_forget_does_not_delete(store: FileMemoryStore, version: str, text: str, kind: str) -> None:
    writer = MemoryWriter(store, PROJECT, "model")
    saved = await writer.save(request(), VIEW)
    with pytest.raises(MemoryOperationError) as caught:
        await writer.forget(
            saved.ref,
            saved.version if version == "current" else version,
            QuoteClaim("user_quote", "candidate", text),
            VIEW,
        )
    assert caught.value.kind == kind
    assert (await store.get(PROJECT, saved.ref)).version == saved.version


@pytest.mark.parametrize("text", ["", "x" * 501])
async def test_forget_claim_entry_validation(store: FileMemoryStore, text: str) -> None:
    writer = MemoryWriter(store, PROJECT, "model")
    saved = await writer.save(request(), VIEW)
    with pytest.raises(MemoryOperationError) as caught:
        await writer.forget(saved.ref, saved.version, QuoteClaim("user_quote", "candidate", text), VIEW)
    assert caught.value.kind == "memory_invalid_input"
    assert (await store.get(PROJECT, saved.ref)).version == saved.version


async def test_reader_search_scopes_and_catalog(store: FileMemoryStore) -> None:
    writer = MemoryWriter(store, PROJECT, "model")
    project = await writer.save(request(), VIEW)
    global_ = await writer.save(replace(request(), scope="global", title="Global"), VIEW)
    reader = MemoryReader(store, PROJECT)
    assert (await reader.read(project.ref)).version == project.version
    assert (await reader.read(global_.ref)).version == global_.version
    assert {item.entry.ref for item in (await reader.search("", "project")).entries} == {project.ref}
    assert {item.entry.ref for item in (await reader.search("", "global")).entries} == {global_.ref}
    both = await reader.search("", "all")
    assert both.query == "" and both.next_cursor is None
    assert {item.entry.ref for item in both.entries} == {project.ref, global_.ref}
    catalog = await reader.catalog()
    assert tuple(item.ref for item in catalog.project) == (project.ref,)
    assert tuple(item.ref for item in catalog.global_) == (global_.ref,)


async def test_unbound_project_keeps_global_read_write_and_reports_project_unavailable(store: FileMemoryStore) -> None:
    writer = MemoryWriter(store, None, "model")
    reader = MemoryReader(store, None)
    saved = await writer.save(replace(request(), scope="global"), VIEW)
    assert (await reader.read(saved.ref)).version == saved.version
    assert tuple(item.entry.ref for item in (await reader.search("", "all")).entries) == (saved.ref,)
    assert tuple(item.entry.ref for item in (await reader.search("", "global")).entries) == (saved.ref,)
    snapshot = await reader.catalog()
    assert snapshot.project == ()
    assert tuple(item.ref for item in snapshot.global_) == (saved.ref,)
    for call in (
        reader.read("project:0123456789ab"),
        reader.search("", "project"),
        writer.save(request(), VIEW),
        writer.save(replace(request(), scope="project"), VIEW),
        writer.save(replace(request(), ref="project:0123456789ab", version="v"), VIEW),
        writer.forget("project:0123456789ab", "v", QuoteClaim("user_quote", "candidate", "forget this"), VIEW),
    ):
        with pytest.raises(MemoryOperationError) as caught:
            await call
        assert caught.value.kind == "memory_store_unavailable"
    deleted = await writer.forget(saved.ref, saved.version, QuoteClaim("user_quote", "candidate", "forget this"), VIEW)
    assert deleted.title == request().title


class CursorStore:
    """Only cursor passthrough uses a typed backend double, as section 7 allows."""

    def __init__(self) -> None:
        self.calls: list[tuple[MemoryScope, str, str | None, bool]] = []

    async def search(
        self, scope: MemoryScope, query: str, cursor: str | None = None, *, with_global: bool = False
    ) -> tuple[list[StoredMemory], str | None]:
        self.calls.append((scope, query, cursor, with_global))
        return [], 'next:"opaque"\nvalue'

    async def list_entries(self, scope: MemoryScope) -> list[StoredMemory]:
        raise NotImplementedError

    async def get(self, scope: MemoryScope, ref: str) -> StoredMemory:
        raise NotImplementedError

    async def catalog(self, scope: MemoryScope) -> list[CatalogItem]:
        raise NotImplementedError

    async def create(self, scope: MemoryScope, draft: MemoryEntryDraft) -> CreateResult:
        raise NotImplementedError

    async def update(self, scope: MemoryScope, ref: str, version: str, draft: MemoryEntryDraft) -> str:
        raise NotImplementedError

    async def delete(self, scope: MemoryScope, ref: str, version: str) -> None:
        raise NotImplementedError

    async def unreadable(self, scope: MemoryScope) -> list[UnreadableMemory]:
        raise NotImplementedError

    async def location(self, scope: MemoryScope) -> str | None:
        raise NotImplementedError

    async def list_projects(self) -> list[ProjectInfo]:
        raise NotImplementedError


async def test_opaque_cursor_and_next_cursor_pass_through_unchanged() -> None:
    backend = CursorStore()
    cursor = 'previous:"opaque"\nvalue'
    page = await MemoryReader(backend, PROJECT).search("query", "all", cursor)
    assert backend.calls == [(PROJECT, "query", cursor, True)]
    assert page.next_cursor == 'next:"opaque"\nvalue'
