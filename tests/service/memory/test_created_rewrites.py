# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A batch may revise only entries it actually created and still owns by version."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from chrys.service.memory.operations import MemoryWriter, SaveRequest
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import MemoryInputRejected, MemoryOperationError, MemoryScope
from chrys.service.memory.verification import (
    ConversationView,
    QuoteClaim,
    ReferenceClaim,
    SourceMessage,
    ToolResultClaim,
    UserReportClaim,
)

PROJECT = MemoryScope("project", "0123456789abcdef")
VIEW = ConversationView(
    (
        SourceMessage("s", 1, ("Keep the local store.",), at="2026-11-12T01:00:00Z", item="first"),
        SourceMessage("s", 2, ("The local store is durable.",), at="2026-11-12T01:01:00Z", item="report"),
        SourceMessage("s", 3, ("Stop using this.",), at="2026-11-12T02:00:00Z", item="withdraw"),
        SourceMessage("s", 4, ("Use the local store again.",), at="2026-11-12T03:00:00Z", item="restore"),
        SourceMessage(
            "s",
            5,
            ("The local store is durable.",),
            kind="tool_result",
            item="observed",
            attributes=(("tool", "shell"), ("args", "{}")),
        ),
        SourceMessage("s", 6, ("Use a local store.",), kind="assistant", item="proposal"),
    ),
    None,
)


def quote(text: str) -> QuoteClaim:
    return QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", text, session="s")


def request(kind: str) -> SaveRequest:
    if kind == "intent":
        return SaveRequest("intent", "Local store", "Use local storage.", user_quotes=(quote("Keep the local store."),))
    evidence = (
        ToolResultClaim("The local store is durable.", session="s")
        if kind == "tool_result"
        else UserReportClaim("The local store is durable.", session="s")
    )
    return SaveRequest("engineering", "Durable storage", "The store is durable.", evidence=(evidence,))


@pytest.mark.parametrize("kind", ["intent", "tool_result", "user_report"])
async def test_created_entry_can_be_reworded_repeatedly_with_its_original_basis(tmp_path: Path, kind: str) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    writer = MemoryWriter(store, PROJECT, "extraction", allow_created_rewrites=True)
    original = request(kind)
    saved = await writer.save(original, VIEW)
    assert saved.status == "created"
    for body in ("A clearer supported conclusion.", "The final supported conclusion."):
        saved = await writer.save(replace(original, ref=saved.ref, version=saved.version, body=body), VIEW)
        stored = await store.get(PROJECT, saved.ref)
        assert saved.status == "updated" and stored.entry.body == body
        assert stored.entry.quotes == saved.quotes and stored.entry.evidence == saved.evidence


@pytest.mark.parametrize("explicit", [False, True])
async def test_created_rewording_requires_an_explicit_opt_in(tmp_path: Path, explicit: bool) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    writer = (
        MemoryWriter(store, PROJECT, "extraction", allow_created_rewrites=False)
        if explicit
        else MemoryWriter(store, PROJECT, "extraction")
    )
    original = request("tool_result")
    saved = await writer.save(original, VIEW)
    with pytest.raises(MemoryInputRejected):
        await writer.save(replace(original, ref=saved.ref, version=saved.version, body="Reworded."), VIEW)
    assert (await store.get(PROJECT, saved.ref)).entry.body == original.body


async def test_existing_result_does_not_open_a_created_edit_window(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    original = request("tool_result")
    await MemoryWriter(store, PROJECT, "model").save(original, VIEW)
    writer = MemoryWriter(store, PROJECT, "extraction", allow_created_rewrites=True)
    saved = await writer.save(original, VIEW)
    assert saved.status == "existing"
    with pytest.raises(MemoryInputRejected):
        await writer.save(replace(original, ref=saved.ref, version=saved.version, body="Reworded."), VIEW)
    assert (await store.get(PROJECT, saved.ref)).version == saved.version


async def test_new_writer_cannot_reword_the_previous_batch_entry_without_new_basis(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    original = request("tool_result")
    saved = await MemoryWriter(store, PROJECT, "extraction", allow_created_rewrites=True).save(original, VIEW)
    next_batch = MemoryWriter(store, PROJECT, "extraction", allow_created_rewrites=True)
    with pytest.raises(MemoryInputRejected):
        await next_batch.save(replace(original, ref=saved.ref, version=saved.version, body="Reworded."), VIEW)


async def test_reading_the_latest_external_version_does_not_restore_created_rewrite_permission(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    writer = MemoryWriter(store, PROJECT, "extraction", allow_created_rewrites=True)
    original = request("tool_result")
    saved = await writer.save(original, VIEW)
    outside = await MemoryWriter(store, PROJECT, "model").save(
        replace(original, ref=saved.ref, version=saved.version, title="Externally revised title"), VIEW
    )
    latest = await store.get(PROJECT, saved.ref)
    assert latest.version == outside.version and latest.version != saved.version
    with pytest.raises(MemoryInputRejected):
        await writer.save(replace(original, ref=saved.ref, version=latest.version, body="Reworded."), VIEW)
    assert (await store.get(PROJECT, saved.ref)).entry.title == "Externally revised title"
    assert (await store.get(PROJECT, saved.ref)).entry.body == original.body


async def test_stale_version_remains_a_hard_conflict_for_a_created_entry(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    writer = MemoryWriter(store, PROJECT, "extraction", allow_created_rewrites=True)
    original = request("intent")
    saved = await writer.save(original, VIEW)
    changed = await writer.save(replace(original, ref=saved.ref, version=saved.version, body="New wording."), VIEW)
    with pytest.raises(MemoryOperationError) as caught:
        await writer.save(replace(original, ref=saved.ref, version=saved.version, body="Stale edit."), VIEW)
    assert caught.value.kind == "memory_version_conflict"
    assert not isinstance(caught.value, MemoryInputRejected)
    assert (await store.get(PROJECT, saved.ref)).version == changed.version


async def test_reference_and_context_cannot_replace_the_basis_of_a_created_entry(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    writer = MemoryWriter(store, PROJECT, "extraction", allow_created_rewrites=True)
    original = request("tool_result")
    saved = await writer.save(original, VIEW)
    with pytest.raises(MemoryInputRejected):
        await writer.save(
            replace(
                original,
                ref=saved.ref,
                version=saved.version,
                body="An unsupported new claim.",
                evidence=(ReferenceClaim("a reference"),),
                context_quote="Use a local store.",
                context_session="s",
            ),
            VIEW,
        )
    assert (await store.get(PROJECT, saved.ref)).version == saved.version


async def test_withdrawal_and_restoration_end_the_created_edit_window(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    writer = MemoryWriter(store, PROJECT, "extraction", allow_created_rewrites=True)
    original = request("intent")
    saved = await writer.save(original, VIEW)
    withdrawn = await writer.withdraw(saved.ref, saved.version, quote("Stop using this."), VIEW)
    with pytest.raises(MemoryInputRejected):
        await writer.save(replace(original, ref=saved.ref, version=withdrawn.version, body="Reworded."), VIEW)
    restored_request = replace(
        original,
        ref=saved.ref,
        version=withdrawn.version,
        user_quotes=(quote("Use the local store again."),),
    )
    restored = await writer.save(restored_request, VIEW)
    assert restored.restored
    with pytest.raises(MemoryInputRejected):
        await writer.save(replace(restored_request, version=restored.version, body="Later rewording."), VIEW)


async def test_merge_does_not_extend_the_created_edit_window(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    writer = MemoryWriter(store, PROJECT, "extraction", allow_created_rewrites=True)
    original = request("intent")
    first = await writer.save(original, VIEW)
    second = await writer.save(replace(original, title="Another title", body="Another wording."), VIEW)
    merged = await writer.merge(first.ref, first.version, second.ref, second.version)
    with pytest.raises(MemoryInputRejected):
        await writer.save(replace(original, ref=first.ref, version=merged.version, body="Later rewording."), VIEW)
