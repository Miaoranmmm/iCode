# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Controlled input refusals are distinguished from uncertain execution failures."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from chrys.service.memory import file_evidence
from chrys.service.memory.operations import MemoryReader, MemoryWriter, SaveRequest
from chrys.service.memory.storage import file_store
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import (
    GLOBAL_SCOPE,
    MemoryEntryDraft,
    MemoryInputRejected,
    MemoryOperationError,
    MemoryScope,
    Quote,
)
from chrys.service.memory.verification import (
    ConversationView,
    QuoteClaim,
    SourceFailure,
    SourceMessage,
    SourceVerificationError,
    ToolResultClaim,
)

PROJECT = MemoryScope("project", "0123456789abcdef")
VIEW = ConversationView(
    (SourceMessage("s", 1, ("Keep local storage.",), at="2026-11-12T01:00:00Z", item="first"),), None
)


def claim(text: str) -> QuoteClaim:
    return QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", text, session="s")


def request() -> SaveRequest:
    return SaveRequest("intent", "Local storage", "Use local storage.", user_quotes=(claim("Keep local storage."),))


@pytest.mark.parametrize(
    "case",
    [
        "ref_format",
        "ref_path",
        "missing_entry",
        "search_cursor",
        "search_cursor_digits",
        "request_validation",
        "quote_not_found",
        "evidence_not_found",
        "update_basis",
        "update_type",
        "update_scope",
        "withdraw_source",
        "already_withdrawn",
        "merge_self",
        "merge_scope",
        "merge_source_limit",
        "restore_basis",
    ],
)
async def test_controlled_refusals_have_a_typed_no_write_guarantee(tmp_path: Path, case: str) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    writer, reader = MemoryWriter(store, PROJECT, "extraction"), MemoryReader(store, PROJECT)
    original = request()
    saved = await writer.save(original, VIEW)
    if case in {"already_withdrawn", "restore_basis"}:
        withdrawn = await store.create(
            PROJECT,
            MemoryEntryDraft(
                "intent",
                "Withdrawn rule",
                original.body,
                "model",
                quotes=(Quote("Keep local storage.", "message", "s", "2026-11-12T01:00:00Z", "first"),),
                status="withdrawn",
                status_basis=Quote("Stop using it.", "message", "s", "2026-11-12T02:00:00Z", "later"),
            ),
        )
        saved_ref, saved_version = withdrawn.ref, withdrawn.version
    else:
        saved_ref, saved_version = saved.ref, saved.version
    if case == "merge_source_limit":
        source_refs = []
        for index in range(2):
            source_refs.append(
                await store.create(
                    PROJECT,
                    MemoryEntryDraft(
                        "intent",
                        f"Rule {index}",
                        f"Conclusion {index}.",
                        "model",
                        quotes=tuple(Quote(f"source {index}-{offset}", "message", "old") for offset in range(2)),
                    ),
                )
            )
    before = await store.list_entries(PROJECT)
    with pytest.raises(MemoryInputRejected) as caught:
        if case == "ref_format":
            await reader.read("0123456789ab")
        elif case == "ref_path":
            await reader.read("project:../outside")
        elif case == "missing_entry":
            await reader.read("project:ffffffffffff")
        elif case == "search_cursor":
            # A non-ASCII digit passes str.isdigit() but not int().
            await reader.search("", "all", "\u00b2")
        elif case == "search_cursor_digits":
            await reader.search("", "all", "9" * 5000)
        elif case == "request_validation":
            await writer.save(replace(original, title="bad\ntitle"), VIEW)
        elif case == "quote_not_found":
            await writer.save(replace(original, user_quotes=(claim("Invented quote."),)), VIEW)
        elif case == "evidence_not_found":
            await writer.save(
                SaveRequest(
                    "engineering", "Finding", "Conclusion.", evidence=(ToolResultClaim("Invented.", session="s"),)
                ),
                VIEW,
            )
        elif case == "update_basis":
            await writer.save(replace(original, ref=saved_ref, version=saved_version, body="Reworded."), VIEW)
        elif case == "update_type":
            await writer.save(
                SaveRequest(
                    "engineering",
                    original.title,
                    original.body,
                    ref=saved_ref,
                    version=saved_version,
                    evidence=(ToolResultClaim("Any source.", session="s"),),
                ),
                VIEW,
            )
        elif case == "update_scope":
            await writer.save(replace(original, ref=saved_ref, version=saved_version, scope="global"), VIEW)
        elif case in {"withdraw_source", "already_withdrawn"}:
            await writer.withdraw(saved_ref, saved_version, claim("Not a withdrawal."), VIEW)
        elif case == "merge_self":
            await writer.merge(saved_ref, saved_version, saved_ref, saved_version)
        elif case == "merge_scope":
            await writer.merge(saved_ref, saved_version, "global:0123456789ab", "unused")
        elif case == "merge_source_limit":
            await writer.merge(source_refs[0].ref, source_refs[0].version, source_refs[1].ref, source_refs[1].version)
        else:
            await writer.save(replace(original, ref=saved_ref, version=saved_version), VIEW)
    assert caught.value.message
    if case == "merge_source_limit":
        assert caught.value.message.endswith("nothing was merged. Keep both entries unchanged.")
    if case.startswith("search_cursor"):
        # Fixed text: the model's cursor is never echoed back.
        assert caught.value.message == "This is not a search cursor returned by memory_search."
    assert await store.list_entries(PROJECT) == before
    assert await store.list_entries(GLOBAL_SCOPE) == []


@pytest.mark.parametrize("case", ["corrupt_entry", "read_denied", "version_conflict"])
async def test_storage_integrity_and_concurrency_failures_are_not_input_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    root = tmp_path / "memory"
    store = FileMemoryStore(root)
    writer = MemoryWriter(store, PROJECT, "extraction")
    saved = await writer.save(request(), VIEW)
    if case == "corrupt_entry":
        path = root / "projects" / str(PROJECT.project_id) / f"{saved.ref.split(':')[1]}.md"
        await asyncio.to_thread(path.write_text, "corrupt entry", encoding="utf-8")
    elif case == "read_denied":

        def denied(path: Path) -> bytes:
            raise PermissionError("Synthetic read failure.")

        monkeypatch.setattr(Path, "read_bytes", denied)
    with pytest.raises(MemoryOperationError) as caught:
        if case == "version_conflict":
            await writer.save(replace(request(), ref=saved.ref, version="outdated"), VIEW)
        else:
            await MemoryReader(store, PROJECT).read(saved.ref)
    assert not isinstance(caught.value, MemoryInputRejected)
    assert (
        caught.value.kind
        == {
            "corrupt_entry": "memory_entry_invalid",
            "read_denied": "memory_store_unavailable",
            "version_conflict": "memory_version_conflict",
        }[case]
    )


def test_missing_file_is_rejected_but_unreadable_file_remains_a_hard_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(MemoryInputRejected):
        file_evidence.capture_file(PROJECT, tmp_path, "missing")
    (tmp_path / "existing").write_bytes(b"original")

    def denied(path: Path):
        raise PermissionError("Synthetic file read failure.")

    monkeypatch.setattr(file_evidence, "secure_open_regular_binary", denied)
    with pytest.raises(MemoryOperationError) as caught:
        file_evidence.capture_file(PROJECT, tmp_path, "existing")
    assert not isinstance(caught.value, MemoryInputRejected)


async def test_full_store_is_a_hard_failure_and_keeps_existing_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    writer = MemoryWriter(store, PROJECT, "extraction")
    saved = await writer.save(request(), VIEW)
    before = await store.list_entries(PROJECT)
    monkeypatch.setattr(file_store, "MAX_ENTRIES_PER_SCOPE", 1)
    with pytest.raises(MemoryOperationError) as caught:
        await writer.save(replace(request(), title="Another rule", body="Another conclusion."), VIEW)
    assert caught.value.kind == "memory_full"
    assert not isinstance(caught.value, MemoryInputRejected)
    assert await store.list_entries(PROJECT) == before
    assert (await store.get(PROJECT, saved.ref)).version == saved.version
    repeated = await writer.save(request(), VIEW)
    assert repeated.status == "existing" and repeated.ref == saved.ref


@pytest.mark.parametrize("operation", ["read", "update", "delete"])
@pytest.mark.parametrize("error", [FileNotFoundError, NotADirectoryError])
async def test_only_missing_memory_paths_are_input_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, error: type[OSError]
) -> None:
    root = tmp_path / "memory"
    store = FileMemoryStore(root)
    saved = await MemoryWriter(store, PROJECT, "extraction").save(request(), VIEW)
    scope = root / "projects" / str(PROJECT.project_id)
    entry = scope / f"{saved.ref.split(':')[1]}.md"
    original_stat = Path.stat
    original_read = Path.read_bytes

    def stat_directory(path: Path, *, follow_symlinks: bool = True):
        if path == scope:
            raise error("Synthetic directory lookup failure.")
        return original_stat(path, follow_symlinks=follow_symlinks)

    def read_entry(path: Path) -> bytes:
        if path == entry:
            raise error("Synthetic entry read failure.")
        return original_read(path)

    monkeypatch.setattr(Path, "stat", stat_directory)
    monkeypatch.setattr(Path, "read_bytes", read_entry)
    draft = MemoryEntryDraft("intent", "Rule", "Keep it local.", "model", quotes=(Quote("Keep it local.", "cli"),))
    with pytest.raises(MemoryOperationError) as caught:
        if operation == "read":
            await store.get(PROJECT, saved.ref)
        elif operation == "update":
            await store.update(PROJECT, saved.ref, saved.version, draft)
        else:
            await store.delete(PROJECT, saved.ref, saved.version)
    assert isinstance(caught.value, MemoryInputRejected) is (error is FileNotFoundError)
    assert caught.value.kind == ("memory_not_found" if error is FileNotFoundError else "memory_store_unavailable")


@pytest.mark.parametrize("broken", ["missing", "root_file", "ancestor_file"])
def test_unavailable_bound_project_root_is_a_hard_failure(tmp_path: Path, broken: str) -> None:
    root = tmp_path / "project"
    if broken == "root_file":
        root.write_bytes(b"not a project directory")
    elif broken == "ancestor_file":
        root.write_bytes(b"not a project ancestor directory")
        root = root / "nested"
    with pytest.raises(MemoryOperationError) as caught:
        file_evidence.capture_file(PROJECT, root, "fact.txt")
    assert not isinstance(caught.value, MemoryInputRejected)


def test_an_evidence_path_with_a_non_directory_component_is_still_an_input_refusal(tmp_path: Path) -> None:
    (tmp_path / "file").write_bytes(b"a file in a valid project root")
    with pytest.raises(MemoryInputRejected):
        file_evidence.capture_file(PROJECT, tmp_path, "file/nested")


@pytest.mark.parametrize("tool", [None, ""])
async def test_tool_evidence_without_a_tool_name_is_an_unverified_source_without_writes(
    tmp_path: Path, tool: str | None
) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    attributes = (("args", "{}"), ("tool", tool)) if tool is not None else (("args", "{}"),)
    view = ConversationView(
        (
            SourceMessage(
                "s", 1, ("Sensitive raw tool output.",), kind="tool_result", item="tool", attributes=attributes
            ),
        ),
        None,
    )
    with pytest.raises(SourceVerificationError) as caught:
        await MemoryWriter(store, PROJECT, "extraction").save(
            SaveRequest(
                "engineering",
                "A finding",
                "A supported conclusion.",
                evidence=(ToolResultClaim("Sensitive raw tool output.", session="s"),),
            ),
            view,
        )
    assert caught.value.kind == "memory_evidence_not_found"
    assert caught.value.message == "Some sources could not be verified."
    assert caught.value.failures == (
        SourceFailure(
            "evidence[0]",
            "evidence[0].candidate",
            "Sensitive raw tool output.",
            "not_found",
            source_kind="tool_result",
            evidence=True,
        ),
    )
    assert "Sensitive raw tool output." not in caught.value.message
    assert await store.list_entries(PROJECT) == []
