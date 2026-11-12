# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Step-8 storage metadata, hidden withdrawn entries, and atomic merge effects."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import TracebackType

import pytest
import yaml

from chrys.foundation.platform.files import atomic_write_owner_only_bytes
from chrys.service.memory.storage import file_store
from chrys.service.memory.storage.file_store import FileMemoryStore, parse_entry, serialize_entry
from chrys.service.memory.storage.store import (
    MemoryEntry,
    MemoryEntryDraft,
    MemoryOperationError,
    MemoryScope,
    MergeResult,
    Quote,
    ReferenceEvidence,
    StoredMemory,
    payload_key,
)

PROJECT = MemoryScope("project", "0123456789abcdef")
BASIS = Quote(
    "That rule no longer applies.", "message", "synthetic-withdrawal", "2026-11-11T01:00:00Z", "withdraw-user"
)


def draft(title: str = "Keep changes small") -> MemoryEntryDraft:
    return MemoryEntryDraft(
        "intent",
        title,
        "Keep changes small.",
        "model",
        quotes=(Quote("Keep changes small.", "message", "synthetic-session"),),
    )


async def test_withdrawn_is_gettable_and_counted_but_absent_from_catalog_and_search(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path)
    created = await store.create(PROJECT, draft())
    current = await store.get(PROJECT, created.ref)
    withdrawn = replace(current.entry.draft(), status="withdrawn", status_basis=BASIS)
    version = await store.update(PROJECT, created.ref, created.version, withdrawn)
    stored = await store.get(PROJECT, created.ref)
    assert stored.version == version
    assert stored.entry.status == "withdrawn" and stored.entry.status_basis == BASIS
    assert [item.entry.ref for item in await store.list_entries(PROJECT)] == [created.ref]
    assert await store.catalog(PROJECT) == []
    assert await store.search(PROJECT, "") == ([], None)
    assert await store.search(PROJECT, "changes") == ([], None)


async def test_existing_identical_create_reports_withdrawn_without_changing_it(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path)
    created = await store.create(PROJECT, draft())
    current = await store.get(PROJECT, created.ref)
    version = await store.update(
        PROJECT,
        created.ref,
        created.version,
        replace(current.entry.draft(), status="withdrawn", status_basis=BASIS, merged_from=("project:abcdef012345",)),
    )
    existing = await store.create(PROJECT, replace(draft(), written_by="extraction"))
    assert (existing.ref, existing.version, existing.created, existing.status) == (
        created.ref,
        version,
        False,
        "withdrawn",
    )
    assert (await store.get(PROJECT, created.ref)).entry.status_basis == BASIS
    assert len(await store.list_entries(PROJECT)) == 1


def test_metadata_is_excluded_from_duplicate_payload_and_active_format_is_unchanged() -> None:
    original = draft()
    changed = replace(original, status="withdrawn", status_basis=BASIS, merged_from=("project:abcdef012345",))
    assert payload_key("project", original) == payload_key("project", changed)
    entry = MemoryEntry(
        "123456abcdef",
        "project",
        "2026-11-11T00:00:00Z",
        "2026-11-11T00:00:00Z",
        original.type,
        original.title,
        original.body,
        original.written_by,
        quotes=original.quotes,
    )
    serialized = serialize_entry(entry)
    front = yaml.safe_load(serialized.split("---\n")[1])
    assert set(front) == {
        "schema",
        "id",
        "type",
        "scope",
        "title",
        "applies_when",
        "created",
        "updated",
        "written_by",
        "quotes",
        "context",
        "evidence",
    }
    assert parse_entry(serialized.encode("utf-8"), scope="project", name=entry.id) == entry
    assert entry.draft() == original
    enriched = replace(entry, status="withdrawn", status_basis=BASIS, merged_from=("project:abcdef012345",))
    assert parse_entry(serialize_entry(enriched).encode("utf-8"), scope="project", name=enriched.id) == enriched
    assert enriched.draft() == changed


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "unknown", "status_basis": BASIS},
        {"status": "withdrawn"},
        {"status_basis": BASIS},
        {"status": "withdrawn", "status_basis": replace(BASIS, text="")},
        {"status": "withdrawn", "status_basis": replace(BASIS, session=None)},
        {"status": "withdrawn", "status_basis": replace(BASIS, at="not a time")},
        {"merged_from": ("project:short",)},
        {"merged_from": ("global:abcdef012345",)},
        {"merged_from": ("project:abcdef012345", "global:123456abcdef")},
        {"merged_from": (7,)},
    ],
)
async def test_invalid_metadata_is_rejected_before_a_write(tmp_path: Path, changes: dict[str, object]) -> None:
    store = FileMemoryStore(tmp_path)
    with pytest.raises(MemoryOperationError) as caught:
        await store.create(PROJECT, replace(draft(), **changes))
    assert caught.value.kind == "memory_invalid_input"
    assert await store.list_entries(PROJECT) == []
    assert not tmp_path.joinpath("projects").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "active"},
        {"status": "withdrawn"},
        {"status_basis": {"text": BASIS.text, "via": "message", "session": BASIS.session}},
        {
            "status": "withdrawn",
            "status_basis": {"text": BASIS.text, "via": "message", "session": BASIS.session, "position": 2},
        },
        {"merged_from": ["global:abcdef012345"]},
        {"merged_from": ["project:short"]},
        {"merged_from": "project:abcdef012345"},
    ],
)
async def test_bad_hand_edited_metadata_is_quarantined(tmp_path: Path, changes: dict[str, object]) -> None:
    store = FileMemoryStore(tmp_path)
    created = await store.create(PROJECT, draft())
    current = await store.get(PROJECT, created.ref)
    text = serialize_entry(current.entry)
    _, header, body = text.split("---\n", 2)
    front = yaml.safe_load(header)
    front.update(changes)
    payload = ("---\n" + yaml.safe_dump(front, sort_keys=False) + "---\n" + body).encode("utf-8")
    path = tmp_path / "projects" / str(PROJECT.project_id) / f"{current.entry.id}.md"
    await asyncio.to_thread(atomic_write_owner_only_bytes, path, payload)
    assert await store.catalog(PROJECT) == []
    assert await store.search(PROJECT, "") == ([], None)
    unreadable = await store.unreadable(PROJECT)
    assert len(unreadable) == 1 and unreadable[0].ref == created.ref


async def test_withdrawn_entries_still_fill_the_scope_capacity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_store, "MAX_ENTRIES_PER_SCOPE", 2)
    store = FileMemoryStore(tmp_path)
    for title in ("First", "Second"):
        await store.create(PROJECT, replace(draft(title), status="withdrawn", status_basis=BASIS))
    assert await store.catalog(PROJECT) == []
    assert await store.search(PROJECT, "") == ([], None)
    with pytest.raises(MemoryOperationError) as caught:
        await store.create(PROJECT, draft("Third"))
    assert caught.value.kind == "memory_full"
    assert len(await store.list_entries(PROJECT)) == 2


async def pair(store: FileMemoryStore) -> tuple[StoredMemory, StoredMemory, MemoryEntryDraft]:
    keep = await store.create(PROJECT, draft("Kept rule"))
    drop = await store.create(
        PROJECT,
        replace(draft("Duplicate rule"), quotes=(Quote("An additional source.", "message", "synthetic-second"),)),
    )
    kept, dropped = await store.get(PROJECT, keep.ref), await store.get(PROJECT, drop.ref)
    merged = replace(
        kept.entry.draft(),
        quotes=(*kept.entry.quotes, *dropped.entry.quotes),
        merged_from=(drop.ref,),
    )
    return kept, dropped, merged


async def test_merge_publishes_keep_then_deletes_drop_with_one_scope_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileMemoryStore(tmp_path)
    keep, drop, merged = await pair(store)
    original_lock = file_store.FileLock
    acquisitions: list[Path] = []

    class CountedLock:
        def __init__(self, path: Path, timeout: float | None = None) -> None:
            self.path = path
            self.inner = original_lock(path, timeout=timeout)

        def __enter__(self) -> CountedLock:
            acquisitions.append(self.path)
            self.inner.acquire()
            return self

        def __exit__(
            self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
        ) -> None:
            self.inner.release()

    monkeypatch.setattr(file_store, "FileLock", CountedLock)
    result = await store.merge(PROJECT, keep.entry.ref, keep.version, drop.entry.ref, drop.version, merged)
    assert result.deleted is True
    current = await store.get(PROJECT, keep.entry.ref)
    assert current.version == result.version and current.version != keep.version
    assert current.entry.draft() == merged
    assert current.entry.created == keep.entry.created
    assert current.entry.body == keep.entry.body and current.entry.title == keep.entry.title
    with pytest.raises(MemoryOperationError) as caught:
        await store.get(PROJECT, drop.entry.ref)
    assert caught.value.kind == "memory_not_found"
    assert acquisitions == [tmp_path / "projects" / str(PROJECT.project_id) / ".lock"]


@pytest.mark.parametrize("stale", ["keep", "drop"])
async def test_merge_checks_both_versions_before_publishing(tmp_path: Path, stale: str) -> None:
    store = FileMemoryStore(tmp_path)
    keep, drop, merged = await pair(store)
    with pytest.raises(MemoryOperationError) as caught:
        await store.merge(
            PROJECT,
            keep.entry.ref,
            "stale" if stale == "keep" else keep.version,
            drop.entry.ref,
            "stale" if stale == "drop" else drop.version,
            merged,
        )
    assert caught.value.kind == "memory_version_conflict"
    assert await store.get(PROJECT, keep.entry.ref) == keep
    assert await store.get(PROJECT, drop.entry.ref) == drop


@pytest.mark.parametrize("withdrawn", ["keep", "drop"])
async def test_merge_rejects_either_withdrawn_entry(tmp_path: Path, withdrawn: str) -> None:
    store = FileMemoryStore(tmp_path)
    keep, drop, merged = await pair(store)
    selected = keep if withdrawn == "keep" else drop
    await store.update(
        PROJECT,
        selected.entry.ref,
        selected.version,
        replace(selected.entry.draft(), status="withdrawn", status_basis=BASIS),
    )
    keep, drop = await store.get(PROJECT, keep.entry.ref), await store.get(PROJECT, drop.entry.ref)
    with pytest.raises(MemoryOperationError) as caught:
        await store.merge(PROJECT, keep.entry.ref, keep.version, drop.entry.ref, drop.version, merged)
    assert caught.value.kind == "memory_invalid_input"
    assert await store.get(PROJECT, keep.entry.ref) == keep
    assert await store.get(PROJECT, drop.entry.ref) == drop


async def test_self_merge_is_rejected_by_public_and_transaction_paths(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path)
    created = await store.create(PROJECT, draft())
    current = await store.get(PROJECT, created.ref)
    with pytest.raises(MemoryOperationError) as caught:
        await store.merge(PROJECT, created.ref, created.version, created.ref, created.version, current.entry.draft())
    assert caught.value.kind == "memory_invalid_input"
    with pytest.raises(MemoryOperationError) as caught:
        await asyncio.to_thread(
            store._merge,
            PROJECT,
            current.entry.id,
            created.ref,
            created.version,
            current.entry.id,
            created.ref,
            created.version,
            current.entry.draft(),
        )
    assert caught.value.kind == "memory_invalid_input"
    assert await store.get(PROJECT, created.ref) == current


async def test_merge_rejects_other_scope_and_other_type_without_writing(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path)
    keep, drop, merged = await pair(store)
    with pytest.raises(MemoryOperationError):
        await store.merge(PROJECT, keep.entry.ref, keep.version, "global:abcdef012345", drop.version, merged)
    engineering = await store.create(
        PROJECT,
        MemoryEntryDraft(
            "engineering",
            "Observed finding",
            "Observed once.",
            "model",
            evidence=(ReferenceEvidence("synthetic reference"),),
        ),
    )
    with pytest.raises(MemoryOperationError) as caught:
        await store.merge(PROJECT, keep.entry.ref, keep.version, engineering.ref, engineering.version, merged)
    assert caught.value.kind == "memory_invalid_input"
    assert await store.get(PROJECT, keep.entry.ref) == keep
    assert await store.get(PROJECT, drop.entry.ref) == drop


async def test_drop_delete_failure_returns_committed_keep_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileMemoryStore(tmp_path)
    keep, drop, merged = await pair(store)
    drop_path = tmp_path / "projects" / str(PROJECT.project_id) / f"{drop.entry.id}.md"
    original_unlink = Path.unlink
    keep_was_published = False

    def cannot_unlink(path: Path, missing_ok: bool = False) -> None:
        nonlocal keep_was_published
        if path == drop_path:
            keep_path = path.with_name(f"{keep.entry.id}.md")
            current = parse_entry(keep_path.read_bytes(), scope="project", name=keep.entry.id)
            keep_was_published = current.draft() == merged
            raise PermissionError("synthetic delete denial")
        original_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as local:
        local.setattr(Path, "unlink", cannot_unlink)
        result = await store.merge(PROJECT, keep.entry.ref, keep.version, drop.entry.ref, drop.version, merged)
    assert result.deleted is False and keep_was_published is True
    assert (await store.get(PROJECT, keep.entry.ref)).version == result.version
    assert await store.get(PROJECT, drop.entry.ref) == drop
    assert (await store.get(PROJECT, keep.entry.ref)).entry.draft() == merged


async def test_manual_drop_edit_after_keep_publication_is_not_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileMemoryStore(tmp_path)
    keep, drop, merged = await pair(store)
    directory = tmp_path / "projects" / str(PROJECT.project_id)
    keep_path, drop_path = directory / f"{keep.entry.id}.md", directory / f"{drop.entry.id}.md"
    real_publish = file_store.atomic_write_owner_only_bytes
    changed = replace(drop.entry, title="Manually changed duplicate")

    def edit_drop_after_keep(path: Path, payload: bytes, *, create_parents: bool = True) -> None:
        real_publish(path, payload, create_parents=create_parents)
        if path == keep_path:
            real_publish(drop_path, serialize_entry(changed).encode("utf-8"), create_parents=False)

    monkeypatch.setattr(file_store, "atomic_write_owner_only_bytes", edit_drop_after_keep)
    result = await store.merge(PROJECT, keep.entry.ref, keep.version, drop.entry.ref, drop.version, merged)
    assert result.deleted is False
    assert (await store.get(PROJECT, keep.entry.ref)).version == result.version
    assert (await store.get(PROJECT, drop.entry.ref)).entry == changed


async def test_keep_publish_failure_preserves_both_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = FileMemoryStore(tmp_path)
    keep, drop, merged = await pair(store)

    def fail_publication(path: Path, payload: bytes, *, create_parents: bool = True) -> None:
        raise OSError("synthetic publication denial")

    monkeypatch.setattr(file_store, "atomic_write_owner_only_bytes", fail_publication)
    with pytest.raises(MemoryOperationError) as caught:
        await store.merge(PROJECT, keep.entry.ref, keep.version, drop.entry.ref, drop.version, merged)
    assert caught.value.kind == "memory_store_unavailable"
    assert await store.get(PROJECT, keep.entry.ref) == keep
    assert await store.get(PROJECT, drop.entry.ref) == drop


async def test_concurrent_merges_do_not_overwrite_a_new_kept_version(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path)
    keep, drop, first_draft = await pair(store)
    third_result = await store.create(
        PROJECT, replace(draft("Third rule"), quotes=(Quote("Third source.", "message", "synthetic-third"),))
    )
    third = await store.get(PROJECT, third_result.ref)
    second_draft = replace(
        keep.entry.draft(), quotes=(*keep.entry.quotes, *third.entry.quotes), merged_from=(third.entry.ref,)
    )
    results = await asyncio.gather(
        store.merge(PROJECT, keep.entry.ref, keep.version, drop.entry.ref, drop.version, first_draft),
        store.merge(PROJECT, keep.entry.ref, keep.version, third.entry.ref, third.version, second_draft),
        return_exceptions=True,
    )
    successful = [result for result in results if isinstance(result, MergeResult)]
    failed = [result for result in results if isinstance(result, MemoryOperationError)]
    assert len(successful) == len(failed) == 1
    assert successful[0].deleted is True and failed[0].kind == "memory_version_conflict"
    current = await store.get(PROJECT, keep.entry.ref)
    assert current.version == successful[0].version
    assert current.entry.draft() in (first_draft, second_draft)
    assert len(await store.list_entries(PROJECT)) == 2
