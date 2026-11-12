# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Final acceptance boundaries for source preservation and maintenance metadata."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from chrys.service.memory import operations
from chrys.service.memory.basis import BasisResult, check_basis
from chrys.service.memory.operations import MemoryWriter, SaveRequest
from chrys.service.memory.storage import file_store
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import (
    GLOBAL_SCOPE,
    Context,
    FileEvidence,
    MemoryInputRejected,
    MemoryOperationError,
    ReferenceEvidence,
)
from chrys.service.memory.verification import (
    ConversationView,
    ReferenceClaim,
    SourceMessage,
    ToolResultClaim,
    verify_evidence,
)
from tests.service.memory.storage.test_lifecycle_store import pair
from tests.service.memory.test_maintenance import PROJECT, RecordingLog, claim, message, original, update, withdrawn


async def test_reader_uses_the_module_level_basis_checker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = FileMemoryStore(tmp_path)
    saved = await store.create(PROJECT, original())
    seen: list[object] = []

    def checked(evidence: Sequence[object], project: object, project_root: Path | None) -> BasisResult:
        seen.append((tuple(evidence), project, project_root))
        return BasisResult("unknown")

    monkeypatch.setattr(operations, "check_basis", checked)
    result = await operations.MemoryReader(store, PROJECT).read(saved.ref)
    assert result.basis == BasisResult("unknown")
    assert seen == [(original().evidence, PROJECT, None)]


@pytest.mark.parametrize("project", [None, GLOBAL_SCOPE])
def test_no_current_project_has_a_distinct_unknown_basis_detail(project) -> None:
    item = FileEvidence(str(PROJECT.project_id), "file", "a" * 64, 1)
    result = check_basis((item,), project, None)
    assert result.state == "unknown"
    assert result.files[0].detail == "no current project"
    other = check_basis((replace(item, project="fedcba9876543210"),), PROJECT, None)
    assert other.files[0].detail == "from another project"


@pytest.mark.parametrize("stored", [(), (ReferenceEvidence("guide", "note"),)])
async def test_repeated_reference_claims_do_not_duplicate_evidence(stored) -> None:
    result = await verify_evidence(
        (ReferenceClaim("guide", "note"), ReferenceClaim("guide", "note")),
        ConversationView((), "current"),
        stored=stored,
    )
    assert result == (ReferenceEvidence("guide", "note"),)


@pytest.mark.parametrize("length", [200, 201])
async def test_verifier_keeps_exact_args_or_refuses_instead_of_truncating(length: int) -> None:
    args = "x" * length
    source = SourceMessage("s", 1, ("observed",), kind="tool_result", attributes=(("tool", "shell"), ("args", args)))
    view = ConversationView((source,), "s")
    if length == 200:
        evidence = await verify_evidence((ToolResultClaim("observed"),), view)
        assert evidence[0].args == args
    else:
        with pytest.raises(MemoryInputRejected) as caught:
            await verify_evidence((ToolResultClaim("observed"),), view)
        assert caught.value.kind == "memory_invalid_input"
        assert caught.value.message == "evidence[0].args is over 200 characters."


@pytest.mark.parametrize("with_context", [False, True])
@pytest.mark.parametrize("body_changed", [False, True])
async def test_restore_retains_omitted_context_like_other_omitted_sources(
    tmp_path: Path, with_context: bool, body_changed: bool
) -> None:
    store = FileMemoryStore(tmp_path)
    draft = replace(withdrawn(), context=original().context if with_context else None)
    saved = await store.create(PROJECT, draft)
    request = update(saved.ref, saved.version)
    if body_changed:
        request = replace(request, body="Use SQLite with its documented constraints.")
    outcome = await MemoryWriter(store, PROJECT, "user").save(
        request, ConversationView((message("use SQLite"),), "current")
    )
    current = (await store.get(PROJECT, saved.ref)).entry
    assert outcome.restored
    assert current.context == draft.context
    assert current.written_by == "user"


async def test_restore_can_explicitly_replace_the_stored_context(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path)
    saved = await store.create(PROJECT, withdrawn())
    request = replace(update(saved.ref, saved.version), context_quote="new proposal", context_session="current")
    source = SourceMessage("current", 20, ("new proposal",), kind="assistant", item="proposal")
    await MemoryWriter(store, PROJECT, "user").save(
        request, ConversationView((message("use SQLite"), source), "current")
    )
    assert (await store.get(PROJECT, saved.ref)).entry.context == Context("new proposal", "current", item="proposal")


async def test_withdraw_and_merge_preserve_content_writer_but_log_the_actor(tmp_path: Path) -> None:
    store, log = FileMemoryStore(tmp_path), RecordingLog()
    writer = MemoryWriter(store, PROJECT, "extraction", change_log=log, source="extract-synthetic")
    saved = await store.create(PROJECT, original())
    await writer.withdraw(saved.ref, saved.version, claim("stop"), ConversationView((message("stop"),), "current"))
    assert (await store.get(PROJECT, saved.ref)).entry.written_by == "model"
    keep = await store.create(PROJECT, replace(original(), title="Keep", body="Keep body"))
    drop = await store.create(PROJECT, replace(original(), title="Drop", body="Drop body", written_by="user"))
    await writer.merge(keep.ref, keep.version, drop.ref, drop.version)
    assert (await store.get(PROJECT, keep.ref)).entry.written_by == "model"
    assert [(row.operation, row.written_by) for row in log.changes] == [
        ("withdraw", "extraction"),
        ("merge", "extraction"),
        ("delete", "extraction"),
    ]


@pytest.mark.parametrize("stage", ["before_recheck", "at_unlink"])
async def test_merge_succeeds_if_drop_is_already_gone_after_keep_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    store = FileMemoryStore(tmp_path)
    keep, drop, merged = await pair(store)
    directory = tmp_path / "projects" / str(PROJECT.project_id)
    keep_path, drop_path = directory / f"{keep.entry.id}.md", directory / f"{drop.entry.id}.md"
    publish, unlink = file_store.atomic_write_owner_only_bytes, Path.unlink

    def write(path: Path, payload: bytes, *, create_parents: bool = True) -> None:
        publish(path, payload, create_parents=create_parents)
        if path == keep_path and stage == "before_recheck":
            unlink(drop_path)

    def remove(path: Path, missing_ok: bool = False) -> None:
        if path == drop_path and stage == "at_unlink":
            unlink(path)
            raise FileNotFoundError("Synthetic concurrent removal")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(file_store, "atomic_write_owner_only_bytes", write)
    monkeypatch.setattr(Path, "unlink", remove)
    outcome = await store.merge(PROJECT, keep.entry.ref, keep.version, drop.entry.ref, drop.version, merged)
    assert outcome.deleted is True
    assert (await store.get(PROJECT, keep.entry.ref)).entry.draft() == merged
    assert not drop_path.exists()


async def test_writer_refuses_a_seventh_evidence_item_before_writing(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path)
    request = SaveRequest(
        "engineering", "Finding", "Observed", evidence=tuple(ReferenceClaim(str(i)) for i in range(7))
    )
    with pytest.raises(MemoryOperationError) as caught:
        await MemoryWriter(store, PROJECT, "model").save(request, ConversationView((), "s"))
    assert caught.value.kind == "memory_invalid_input"
    assert caught.value.message == "Invalid memory input: evidence must be omitted or contain 1-6 items."
    assert await store.list_entries(PROJECT) == []


async def test_store_merge_preserves_the_content_author_even_if_draft_names_the_actor(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path)
    keep, drop, merged = await pair(store)
    result = await store.merge(
        PROJECT, keep.entry.ref, keep.version, drop.entry.ref, drop.version, replace(merged, written_by="extraction")
    )
    assert result.deleted
    assert (await store.get(PROJECT, keep.entry.ref)).entry.written_by == keep.entry.written_by
