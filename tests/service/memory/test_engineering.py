# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Engineering saves, per-type new basis rules, files and context session retention."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from chrys.foundation.platform.files import atomic_write_owner_only_bytes
from chrys.service.memory.operations import MemoryWriter, SaveRequest
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import FileEvidence, MemoryOperationError
from chrys.service.memory.verification import (
    ConversationView,
    FileClaim,
    QuoteClaim,
    ReferenceClaim,
    SourceMessage,
    ToolResultClaim,
    UserReportClaim,
)
from tests.support.secure_files import plant_owner_only_bytes
from tests.support.symlinks import symlink_or_skip

PROJECT_ID = "0123456789abcdef"


def sources() -> ConversationView:
    return ConversationView(
        (
            SourceMessage(
                "s", 1, ("use the local store", "the new requirement", "old observed", "new observed"), item="human"
            ),
            SourceMessage("s", 2, ("old proposal", "new proposal"), kind="assistant", item="proposal"),
            SourceMessage(
                "s",
                3,
                ("old observed",),
                kind="tool_result",
                item="old-tool",
                attributes=(("tool", "shell"), ("args", "{}")),
            ),
            SourceMessage(
                "s",
                4,
                ("new observed",),
                kind="tool_result",
                item="new-tool",
                attributes=(("tool", "shell"), ("args", "{}")),
            ),
        ),
        "s",
    )


@pytest.fixture
async def setup(tmp_path: Path):
    from chrys.service.memory.storage.store import MemoryScope

    root = tmp_path / "project"
    await asyncio.to_thread(root.mkdir)
    project = MemoryScope("project", PROJECT_ID, root=str(root))
    store = FileMemoryStore(tmp_path / "memory")
    return store, project, root, MemoryWriter(store, project, "model", project_root=root)


def intent() -> SaveRequest:
    return SaveRequest(
        "intent",
        "Requirement",
        "Keep it local.",
        user_quotes=(QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", "use the local store"),),
    )


def engineering() -> SaveRequest:
    return SaveRequest("engineering", "Finding", "An observed fact.", evidence=(ToolResultClaim("old observed"),))


@pytest.mark.parametrize(
    "kind,basis,valid",
    [
        ("intent", "quotes", True),
        ("intent", "evidence", False),
        ("engineering", "evidence", True),
        ("engineering", "quotes", False),
    ],
)
async def test_entry_type_enforces_its_basis_kind(setup, kind: str, basis: str, valid: bool) -> None:
    store, project, _, writer = setup
    request = replace(
        intent(),
        type=kind,
        user_quotes=intent().user_quotes if basis == "quotes" else None,
        evidence=(UserReportClaim("old observed"),) if basis == "evidence" else None,
    )
    if valid:
        saved = await writer.save(request, sources())
        assert (await store.get(project, saved.ref)).entry.type == kind
    else:
        with pytest.raises(MemoryOperationError) as caught:
            await writer.save(request, sources())
        assert caught.value.kind == "memory_invalid_input"
        assert await store.list_entries(project) == []


@pytest.mark.parametrize("kind", ["intent", "engineering"])
async def test_substantive_change_with_only_retained_old_basis_is_refused(setup, kind: str) -> None:
    store, project, _, writer = setup
    request = intent() if kind == "intent" else engineering()
    original = await writer.save(request, sources())
    update = replace(request, ref=original.ref, version=original.version, body="Changed conclusion.")
    if kind == "intent":
        update = replace(update, user_quotes=(replace(request.user_quotes[0], session="s"),))
    else:
        update = replace(update, evidence=(ToolResultClaim("old observed", session="s"),))
    with pytest.raises(MemoryOperationError) as caught:
        await writer.save(update, sources())
    assert caught.value.kind == "memory_invalid_input"
    assert (await store.get(project, original.ref)).version == original.version


@pytest.mark.parametrize("kind", ["intent", "engineering"])
async def test_new_context_and_reference_are_not_substantive_basis(setup, kind: str) -> None:
    _, _, _, writer = setup
    request = intent() if kind == "intent" else engineering()
    saved = await writer.save(request, sources())
    update = replace(request, ref=saved.ref, version=saved.version, body="Changed.", context_quote="new proposal")
    if kind == "intent":
        update = replace(update, user_quotes=(replace(request.user_quotes[0], session="s"),))
    else:
        update = replace(update, evidence=(ReferenceClaim("new document"),))
    with pytest.raises(MemoryOperationError) as caught:
        await writer.save(update, sources())
    assert caught.value.kind == "memory_invalid_input"


@pytest.mark.parametrize("basis", ["tool_result", "user_report"])
async def test_new_verified_evidence_allows_body_change_and_old_context_retention(setup, basis: str) -> None:
    store, project, _, writer = setup
    original = await writer.save(replace(engineering(), context_quote="old proposal"), sources())
    claim = ToolResultClaim("new observed") if basis == "tool_result" else UserReportClaim("new observed")
    update = replace(
        engineering(),
        ref=original.ref,
        version=original.version,
        body="Updated finding.",
        evidence=(claim,),
        context_quote="old proposal",
        context_session="s",
    )
    # Context comes from an earlier view, so only the old value's explicit
    # session can retain it while the new evidence is checked in this view.
    view = ConversationView(tuple(source for source in sources().messages if source.kind != "assistant"), "s")
    saved = await writer.save(update, view)
    entry = (await store.get(project, saved.ref)).entry
    assert entry.body == update.body and entry.evidence[0].kind == basis
    assert entry.context == original.context


async def test_unchanged_file_hash_is_not_new_basis_but_changed_hash_is(setup) -> None:
    store, project, root, writer = setup
    path = root / "fact.txt"
    await asyncio.to_thread(plant_owner_only_bytes, path, b"old")
    request = replace(engineering(), evidence=(FileClaim("fact.txt"),))
    saved = await writer.save(request, sources())
    first = (await store.get(project, saved.ref)).entry.evidence[0]
    assert isinstance(first, FileEvidence) and first.sha256 == hashlib.sha256(b"old").hexdigest()
    update = replace(request, ref=saved.ref, version=saved.version, body="New conclusion.")
    with pytest.raises(MemoryOperationError) as caught:
        await writer.save(update, sources())
    assert caught.value.kind == "memory_invalid_input"
    await asyncio.to_thread(atomic_write_owner_only_bytes, path, b"new", create_parents=False)
    updated = await writer.save(update, sources())
    second = (await store.get(project, updated.ref)).entry.evidence[0]
    assert isinstance(second, FileEvidence) and second.sha256 == hashlib.sha256(b"new").hexdigest()


@pytest.mark.parametrize("change", ["new_file", "large_file"])
async def test_new_file_path_or_changed_hash_to_none_counts_as_new_basis(setup, change: str) -> None:
    store, project, root, writer = setup
    await asyncio.to_thread(plant_owner_only_bytes, root / "first.txt", b"old")
    request = replace(engineering(), evidence=(FileClaim("first.txt"),))
    saved = await writer.save(request, sources())
    target = root / ("second.txt" if change == "new_file" else "first.txt")
    await asyncio.to_thread(
        plant_owner_only_bytes, target, b"old" if change == "new_file" else b"x" * (2 * 1024 * 1024 + 1)
    )
    update = replace(
        request, ref=saved.ref, version=saved.version, body="New conclusion.", evidence=(FileClaim(target.name),)
    )
    result = await writer.save(update, sources())
    [evidence] = (await store.get(project, result.ref)).entry.evidence
    assert isinstance(evidence, FileEvidence)
    assert evidence.path == target.name
    assert (evidence.sha256 is None) is (change == "large_file")


@pytest.mark.parametrize("case", ["missing", "escape", "symlink", "no_root"])
async def test_file_evidence_requires_an_existing_file_inside_bound_root(setup, tmp_path: Path, case: str) -> None:
    store, project, root, writer = setup
    outside = tmp_path / "outside.txt"
    await asyncio.to_thread(plant_owner_only_bytes, outside, b"outside")
    if case == "symlink":
        await asyncio.to_thread(symlink_or_skip, root / "link", outside)
        path = "link"
    elif case == "escape":
        path = "../outside.txt"
    elif case == "no_root":
        writer = MemoryWriter(store, project, "model")
        path = str(outside)
    else:
        path = "missing.txt"
    with pytest.raises(MemoryOperationError) as caught:
        await writer.save(replace(engineering(), evidence=(FileClaim(path),)), sources())
    assert caught.value.kind == "memory_evidence_not_found"
    assert await store.list_entries(project) == []


async def test_file_evidence_normalizes_internal_absolute_path_to_posix(setup) -> None:
    store, project, root, writer = setup
    await asyncio.to_thread((root / "nested").mkdir)
    path = root / "nested" / "fact.txt"
    await asyncio.to_thread(plant_owner_only_bytes, path, b"fact")
    saved = await writer.save(replace(engineering(), evidence=(FileClaim(str(path)),)), sources())
    [file] = (await store.get(project, saved.ref)).entry.evidence
    assert isinstance(file, FileEvidence)
    assert (file.project, file.path, file.bytes) == (PROJECT_ID, "nested/fact.txt", 4)


async def test_writer_restrictions_hold_even_if_tool_schema_is_bypassed(setup) -> None:
    store, project, root, _ = setup
    restricted = MemoryWriter(
        store,
        project,
        "extraction",
        project_root=root,
        allowed_types=("intent",),
        allowed_evidence=("tool_result", "reference", "user_report"),
    )
    with pytest.raises(MemoryOperationError) as wrong_type:
        await restricted.save(engineering(), sources())
    assert wrong_type.value.kind == "memory_invalid_input"
    file_restricted = MemoryWriter(
        store, project, "extraction", project_root=root, allowed_evidence=("tool_result", "reference", "user_report")
    )
    with pytest.raises(MemoryOperationError) as file:
        await file_restricted.save(replace(engineering(), evidence=(FileClaim("anything"),)), sources())
    assert file.value.kind == "memory_invalid_input" and "file evidence" in file.value.message
    assert await store.list_entries(project) == []


async def test_submitted_file_is_rechecked_on_a_title_only_update(setup) -> None:
    store, project, root, writer = setup
    path = root / "fact.txt"
    await asyncio.to_thread(plant_owner_only_bytes, path, b"snapshot")
    request = replace(engineering(), evidence=(FileClaim("fact.txt"),))
    saved = await writer.save(request, sources())
    await asyncio.to_thread(path.unlink)
    with pytest.raises(MemoryOperationError) as caught:
        await writer.save(replace(request, title="New title", ref=saved.ref, version=saved.version), sources())
    assert caught.value.kind == "memory_evidence_not_found"
    assert (await store.get(project, saved.ref)).version == saved.version


async def test_same_file_digest_with_different_recorded_size_is_not_new_basis(setup) -> None:
    from chrys.service.memory.storage.store import MemoryEntryDraft

    store, project, root, writer = setup
    await asyncio.to_thread(plant_owner_only_bytes, root / "fact.txt", b"same")
    file = FileEvidence(PROJECT_ID, "fact.txt", hashlib.sha256(b"same").hexdigest(), 999)
    saved = await store.create(
        project, MemoryEntryDraft("engineering", "Finding", "Original.", "model", evidence=(file,))
    )
    with pytest.raises(MemoryOperationError) as caught:
        await writer.save(
            SaveRequest(
                "engineering",
                "Finding",
                "Changed.",
                ref=saved.ref,
                version=saved.version,
                evidence=(FileClaim("fact.txt"),),
            ),
            sources(),
        )
    assert caught.value.kind == "memory_invalid_input"


async def test_context_session_is_required_in_batch_even_for_retention(setup) -> None:
    store, project, _, writer = setup
    saved = await writer.save(replace(intent(), context_quote="old proposal"), sources())
    update = replace(
        intent(),
        title="New title",
        ref=saved.ref,
        version=saved.version,
        user_quotes=None,
        context_quote="old proposal",
    )
    with pytest.raises(MemoryOperationError) as caught:
        await writer.save(update, ConversationView((), None))
    assert caught.value.kind == "memory_invalid_input"
    assert caught.value.message == "Invalid memory input: context_session is required in batch mode."
    retained = await writer.save(replace(update, context_session="s"), ConversationView((), None))
    assert (await store.get(project, retained.ref)).entry.context == saved.context
