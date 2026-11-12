# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Review tools follow scope, protocol admission, and durable-write ownership rules."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from chrys.app.memory_hosts import executor_icode as executor_module
from chrys.app.memory_hosts.executor_icode import ICodeExecutor, extraction_tool_schemas
from chrys.app.memory_hosts.mcp_server import MemoryTools
from chrys.app.memory_hosts.schemas import decode_request, tool_definitions
from chrys.service.memory import extraction as core
from chrys.service.memory.names import MEMORY_TOOL_NAMES
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import MemoryEntryDraft, Quote
from chrys.service.memory.verification import ConversationView, SourceMessage
from tests.app.memory_hosts.executor_fakes import (
    PROJECT,
    RUN_ID,
    SESSION,
    ClientFactory,
    RecordingLog,
    ScriptedWire,
    batch,
    call,
    prepared,
    response,
)
from tests.app.memory_hosts.test_mcp import Context, text

REVIEW_NAMES = {"memory_withdraw", "memory_merge", "memory_note_conflict"}
REF_A = "project:aaaaaaaaaaaa"
REF_B = "project:bbbbbbbbbbbb"


def review_arguments(name: str) -> dict[str, object]:
    if name == "memory_withdraw":
        return {"ref": REF_A, "version": "v", "user_quote": "No longer applicable.", "session": SESSION}
    if name == "memory_merge":
        return {"keep_ref": REF_A, "keep_version": "v", "drop_ref": REF_B, "drop_version": "v"}
    return {"ref_a": REF_A, "ref_b": REF_B}


def test_tool_schemas_keep_interactive_and_extraction_registration_separate() -> None:
    interactive = {tool.name for tool in tool_definitions(writable=True)}
    assert "memory_withdraw" in interactive
    assert "memory_merge" not in interactive and "memory_note_conflict" not in interactive
    assert {tool.name for tool in tool_definitions(writable=False)} == {
        "memory_read",
        "memory_search",
        "memory_catalog",
    }
    assert {tool["name"] for tool in extraction_tool_schemas(core.FULL_V3)} == {
        "memory_search",
        "memory_read",
        "memory_save",
        *REVIEW_NAMES,
    }
    assert {tool["name"] for tool in extraction_tool_schemas(core.USER_V2)} == {
        "memory_search",
        "memory_read",
        "memory_save",
    }
    withdraw = next(tool for tool in extraction_tool_schemas() if tool["name"] == "memory_withdraw")
    assert "session" in withdraw["parameters"]["required"]
    assert (
        "no longer applies"
        in next(tool for tool in tool_definitions(writable=True) if tool.name == "memory_withdraw").description.lower()
    )


@pytest.mark.parametrize("name", sorted(REVIEW_NAMES))
def test_each_review_tool_is_registered_in_names_decoder_and_native_allowlist(name: str) -> None:
    assert name in MEMORY_TOOL_NAMES
    assert name in executor_module._TOOL_NAMES
    assert decode_request(name, review_arguments(name)) is not None


@pytest.mark.parametrize("name", sorted(REVIEW_NAMES))
async def test_read_only_gate_refuses_every_review_tool_before_decoding_or_context(tmp_path: Path, name: str) -> None:
    context = Context()
    tools = MemoryTools(FileMemoryStore(tmp_path / "memory"), "read-only", context)
    result = await tools.call(name, None, None)
    assert result.isError is True and "read-only" in text(result)
    assert context.reads == [] and context.writes == []


@pytest.mark.parametrize("name", ["memory_merge", "memory_note_conflict"])
async def test_interactive_connection_never_dispatches_extraction_only_tools(tmp_path: Path, name: str) -> None:
    context = Context()
    result = await MemoryTools(FileMemoryStore(tmp_path / "memory"), "read-write", context).call(
        name, review_arguments(name), None
    )
    assert result.isError is True
    assert context.reads == [] and context.writes == []
    assert not await asyncio.to_thread((tmp_path / "memory").exists)


async def test_interactive_withdraw_hides_entry_and_retains_verified_basis(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    saved = await store.create(
        PROJECT,
        MemoryEntryDraft(
            "intent", "Use SQLite", "Use SQLite.", "model", quotes=(Quote("Use SQLite.", "message", SESSION),)
        ),
    )
    view = ConversationView(
        (SourceMessage(SESSION, 2, ("No longer applicable.",), item="withdrawal", at="2026-11-11T12:00:00Z"),), SESSION
    )
    context = Context(view=view)
    result = await MemoryTools(store, "read-write", context).call(
        "memory_withdraw",
        {"ref": saved.ref, "version": saved.version, "user_quote": "No longer applicable."},
        {"claudecode/toolUseId": "synthetic-withdraw"},
    )
    assert result.isError is False
    current = await store.get(PROJECT, saved.ref)
    assert current.entry.status == "withdrawn" and current.entry.status_basis.item == "withdrawal"
    assert await store.catalog(PROJECT) == []
    assert "Status: withdrawn" in text(
        await MemoryTools(store, "read-write", context).call("memory_read", {"ref": saved.ref}, None)
    )


@pytest.mark.parametrize("protocol", ["responses", "messages"])
async def test_note_conflict_counts_only_and_never_touches_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str
) -> None:
    from chrys.app.memory_hosts.executor_icode import prepare_profile
    from tests.app.memory_hosts.executor_fakes import messages_call, messages_profile, messages_response

    wire = ScriptedWire(
        [response(call("memory_note_conflict", {"ref_a": REF_A, "ref_b": REF_B})), response(text="Done")]
        if protocol == "responses"
        else [
            messages_response(
                messages_call("memory_note_conflict", {"ref_a": REF_A, "ref_b": REF_B}), reason="tool_use"
            ),
            messages_response(text="Done"),
        ]
    )
    monkeypatch.setattr(executor_module, "create_client", ClientFactory(wire))
    store, log = FileMemoryStore(tmp_path / "memory"), RecordingLog()
    result = await ICodeExecutor(
        prepared() if protocol == "responses" else prepare_profile(messages_profile()), store, log, RUN_ID
    ).run(batch())
    assert (result.termination, result.call_count, result.failure_count, result.conflict_count) == (
        "completed",
        1,
        0,
        1,
    )
    assert result.saved == () and log.changes == []
    assert not await asyncio.to_thread((tmp_path / "memory").exists)


@pytest.mark.parametrize("name", sorted(REVIEW_NAMES))
async def test_guard_rejects_review_calls_when_the_global_registration_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    wire = ScriptedWire([response(call(name, review_arguments(name)))])
    monkeypatch.setattr(executor_module, "create_client", ClientFactory(wire))
    monkeypatch.setattr(executor_module, "_TOOL_NAMES", executor_module._TOOL_NAMES - {name})
    store, log = FileMemoryStore(tmp_path / "memory"), RecordingLog()
    result = await ICodeExecutor(prepared(), store, log, RUN_ID).run(batch())
    assert (result.termination, result.call_count, result.stop_reason) == ("failed", 0, "unexpected_tool_call")
    assert result.saved == () and log.changes == []


@pytest.mark.parametrize("name", sorted(REVIEW_NAMES))
async def test_user_v2_guard_refuses_review_tools_without_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    wire = ScriptedWire([response(call(name, review_arguments(name)))])
    monkeypatch.setattr(executor_module, "create_client", ClientFactory(wire))
    result = await ICodeExecutor(
        prepared(), FileMemoryStore(tmp_path / "memory"), RecordingLog(), RUN_ID, material_profile=core.USER_V2
    ).run(batch())
    assert (result.termination, result.call_count, result.stop_reason) == ("failed", 0, "unexpected_tool_call")


@pytest.mark.parametrize("name", sorted(REVIEW_NAMES))
async def test_admitted_review_operation_survives_cancellation_with_typed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    from chrys.app.memory_hosts.schemas import NoteConflictRequest
    from chrys.service.memory.operations import MemoryWriter, MergeOutcome, WithdrawOutcome
    from chrys.service.memory.verification import QuoteClaim
    from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for

    store = FileMemoryStore(tmp_path / "memory")
    first = await store.create(
        PROJECT, MemoryEntryDraft("intent", "First", "First.", "model", quotes=(Quote("First.", "message", SESSION),))
    )
    second = await store.create(
        PROJECT,
        MemoryEntryDraft(
            "intent", "Second", "Second.", "model", quotes=(Quote("Second.", "message", "outside-batch"),)
        ),
    )
    entered, release = asyncio.Event(), asyncio.Event()
    if name == "memory_withdraw":
        original_withdraw = MemoryWriter.withdraw

        async def gated_withdraw(
            writer: MemoryWriter, ref: str, version: str, user_quote: QuoteClaim, view: ConversationView
        ) -> WithdrawOutcome:
            entered.set()
            await release.wait()
            return await original_withdraw(writer, ref, version, user_quote, view)

        monkeypatch.setattr(MemoryWriter, "withdraw", gated_withdraw)
        arguments = {
            "ref": first.ref,
            "version": first.version,
            "user_quote": "No longer applicable.",
            "session": SESSION,
        }
    elif name == "memory_merge":
        original_merge = MemoryWriter.merge

        async def gated_merge(
            writer: MemoryWriter, keep_ref: str, keep_version: str, drop_ref: str, drop_version: str
        ) -> MergeOutcome:
            entered.set()
            await release.wait()
            return await original_merge(writer, keep_ref, keep_version, drop_ref, drop_version)

        monkeypatch.setattr(MemoryWriter, "merge", gated_merge)
        arguments = {
            "keep_ref": first.ref,
            "keep_version": first.version,
            "drop_ref": second.ref,
            "drop_version": second.version,
        }
    else:
        original_note = ICodeExecutor._note_conflict

        async def gated_note(executor: ICodeExecutor, request: NoteConflictRequest) -> str:
            entered.set()
            await release.wait()
            return await original_note(executor, request)

        monkeypatch.setattr(ICodeExecutor, "_note_conflict", gated_note)
        arguments = {"ref_a": first.ref, "ref_b": second.ref}
    wire = ScriptedWire([response(call(name, arguments)), response(text="Done")])
    monkeypatch.setattr(executor_module, "create_client", ClientFactory(wire))
    log = RecordingLog()
    source = batch(
        messages=(SourceMessage(SESSION, 5, ("No longer applicable.",), item="withdrawal", at="2026-11-11T00:00:00Z"),)
    )
    task = asyncio.create_task(ICodeExecutor(prepared(), store, log, RUN_ID).run(source))
    try:
        await wait_for(lambda: entered.is_set() or task.done(), timeout=ENGINE_TURN_TIMEOUT)
        if task.done():
            await task
        assert entered.is_set() and not task.done()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        result = await task
        assert (result.termination, result.call_count, result.failure_count) == ("cancelled", 1, 0)
        assert wire.closed == 1
        if name == "memory_withdraw":
            assert result.withdrawn_count == 1 and result.saved[0].outcome == "withdrawn"
            assert (await store.get(PROJECT, first.ref)).entry.status == "withdrawn"
            assert [change.operation for change in log.changes] == ["withdraw"]
        elif name == "memory_merge":
            assert result.merged_count == 1 and result.saved[0].outcome == "merged"
            assert len(await store.list_entries(PROJECT)) == 1
            assert [change.operation for change in log.changes] == ["merge", "delete"]
        else:
            assert result.conflict_count == 1 and result.saved == () and log.changes == []
            assert len(await store.list_entries(PROJECT)) == 2
    finally:
        release.set()
        await task


async def test_executor_renders_actual_review_lists_and_preserves_related_ref_receipt_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chrys.app.memory_hosts.extraction_review import load_review_snapshot
    from chrys.service.memory.operations import MemoryWriter, SaveRequest
    from chrys.service.memory.related_refs import result_refs
    from chrys.service.memory.render import render_saved
    from chrys.service.memory.verification import QuoteClaim

    store = FileMemoryStore(tmp_path / "memory")
    writer = MemoryWriter(store, PROJECT, "model")
    view = ConversationView(
        (SourceMessage(SESSION, 1, ("Retain a supported finding.",), item="human", at="2026-11-11T00:00:00Z"),), SESSION
    )
    saved = await writer.save(
        SaveRequest(
            "intent",
            "Related title",
            "PRIVATE_BODY",
            user_quotes=(QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", "Retain a supported finding."),),
        ),
        view,
    )
    assert saved.ref in result_refs((render_saved(saved),))
    withdrawn = await store.create(
        PROJECT,
        MemoryEntryDraft(
            "intent",
            "Withdrawn title",
            "PRIVATE_WITHDRAWN_BODY",
            "model",
            quotes=(Quote("old", "message", SESSION),),
            status="withdrawn",
            status_basis=Quote(
                "PRIVATE_WITHDRAWAL", "message", SESSION, at="2026-11-11T00:00:00Z", item="old-withdraw"
            ),
        ),
    )
    original = batch()
    source = replace(original.sources[0], related_refs=(saved.ref,))
    selected = replace(original, sources=(source,))
    lists = (await load_review_snapshot(store, PROJECT)).lists(selected.sources)
    wire = ScriptedWire([response(call("memory_search", {"query": ""})), response(text="Done")])
    monkeypatch.setattr(executor_module, "create_client", ClientFactory(wire))
    result = await ICodeExecutor(prepared(), store, RecordingLog(), RUN_ID).run(selected)
    assert result.termination == "completed"
    prompt = "\n".join(content.text or "" for message in wire.requests[0][0] for content in message.contents)
    assert f"<related_entries>\n{lists.related_entries}\n</related_entries>" in prompt
    assert f"<withdrawn_entries>\n{lists.withdrawn_entries}\n</withdrawn_entries>" in prompt
    assert saved.ref in prompt and withdrawn.ref in prompt
    assert (
        "PRIVATE_BODY" not in prompt and "PRIVATE_WITHDRAWAL" not in prompt and "PRIVATE_WITHDRAWN_BODY" not in prompt
    )


@pytest.mark.parametrize("name", sorted(REVIEW_NAMES))
async def test_legacy_review_tool_failure_never_advances_success_watermark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    from chrys.app.memory_hosts import extract
    from chrys.app.memory_hosts.extract import ExtractionOptions, run_extraction
    from chrys.service.memory.extraction import WatermarkStore
    from tests.app.memory_hosts.executor_fakes import profile
    from tests.app.memory_hosts.test_extract import Reader

    monkeypatch.setattr(extract, "bind_project", lambda cwd: PROJECT)
    arguments = review_arguments(name)
    if name == "memory_note_conflict":
        arguments = {"ref_a": REF_A, "ref_b": REF_A}
    wire = ScriptedWire([response(call(name, arguments)), response(text="Done")])
    monkeypatch.setattr(executor_module, "create_client", ClientFactory(wire))
    options = ExtractionOptions(tmp_path, tmp_path / "memory")
    selected = batch(
        messages=(SourceMessage(SESSION, 1, ("No longer applicable.",), item="user", at="2026-11-11T00:00:00Z"),)
    )
    report = await run_extraction(
        options, profile(), readers={"claude-code": Reader(selected.sources)}, material_profile=core.FULL_V3
    )
    assert (report.status, report.exit_code) == ("failed", 1)
    assert report.batches[0]["failure_count"] == 1
    assert report.batches[0]["rejection_count"] == 0 and len(wire.requests) == 2
    record = (await WatermarkStore(options.memory_dir).load()).record(PROJECT.project_id, selected.sources[0].key)
    assert record.attempts == 1 and record.updated is None


async def test_run_summary_counts_conflict_without_change_log_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chrys.app.memory_hosts import extract
    from chrys.app.memory_hosts.extract import ExtractionOptions, run_extraction
    from tests.app.memory_hosts.executor_fakes import profile
    from tests.app.memory_hosts.test_extract import Reader

    monkeypatch.setattr(extract, "bind_project", lambda cwd: PROJECT)
    wire = ScriptedWire(
        [response(call("memory_note_conflict", {"ref_a": REF_A, "ref_b": REF_B})), response(text="Done")]
    )
    monkeypatch.setattr(executor_module, "create_client", ClientFactory(wire))
    options = ExtractionOptions(tmp_path, tmp_path / "memory")
    report = await run_extraction(options, profile(), readers={"claude-code": Reader(batch().sources)})
    assert (report.status, report.exit_code, report.conflict_count) == ("completed", 0, 1)
    assert report.batches[0]["conflict_count"] == 1
    assert not await asyncio.to_thread(lambda: tuple(options.memory_dir.rglob("changes*.jsonl")))


async def test_fake_batch_withdraws_and_restores_same_words_with_new_occurrence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chrys.service.memory.verification import QuoteClaim, SourceVerificationError, verify_quotes

    store = FileMemoryStore(tmp_path / "memory")
    saved = await store.create(
        PROJECT,
        MemoryEntryDraft(
            "intent",
            "First",
            "First.",
            "model",
            quotes=(Quote("First.", "message", SESSION, at="2026-11-11T00:00:00Z", item="old"),),
        ),
    )
    messages = (
        SourceMessage(SESSION, 1, ("First.",), item="old", at="2026-11-11T00:00:00Z"),
        SourceMessage(SESSION, 2, ("No longer applicable.",), item="withdrawal", at="2026-11-11T01:00:00Z"),
        SourceMessage(SESSION, 3, ("First.",), item="new", at="2026-11-11T02:00:00Z"),
    )
    selected = batch(messages=messages)
    with pytest.raises(SourceVerificationError) as caught:
        verify_quotes(
            (QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", "First.", session=SESSION),),
            selected.source_index.view,
        )
    candidate = next(
        candidate.token for candidate in caught.value.failures[0].candidates if candidate.at == "2026-11-11T02:00:00Z"
    )

    async def restore(messages):
        current = await store.get(PROJECT, saved.ref)
        return response(
            call(
                "memory_save",
                {
                    "ref": saved.ref,
                    "version": current.version,
                    "type": "intent",
                    "title": "First again",
                    "body": "First.",
                    "user_quotes": [{"text": "First.", "session": SESSION, "candidate": candidate}],
                },
            )
        )

    wire = ScriptedWire(
        [
            response(
                call(
                    "memory_withdraw",
                    {
                        "ref": saved.ref,
                        "version": saved.version,
                        "user_quote": "No longer applicable.",
                        "session": SESSION,
                    },
                )
            ),
            restore,
            response(call("memory_read", {"ref": saved.ref})),
            response(text="Done"),
        ]
    )
    monkeypatch.setattr(executor_module, "create_client", ClientFactory(wire))
    log = RecordingLog()
    result = await ICodeExecutor(prepared(), store, log, RUN_ID).run(selected)
    assert (result.termination, result.failure_count, result.withdrawn_count, result.restored_count) == (
        "completed",
        0,
        1,
        1,
    )
    assert [change.operation for change in log.changes] == ["withdraw", "restore"]
    assert all(change.source == RUN_ID for change in log.changes)
    current = await store.get(PROJECT, saved.ref)
    assert current.entry.status is None and current.entry.status_basis is None
    assert current.entry.quotes[0].item == "new"


async def test_native_partial_merge_retains_committed_receipt_and_fails_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    saved = []
    for word in ("first", "second"):
        saved.append(
            await store.create(
                PROJECT,
                MemoryEntryDraft("intent", word, word, "model", quotes=(Quote(word, "message", "out-of-batch"),)),
            )
        )
    keep, drop = saved
    unlink = Path.unlink

    def fail_drop(path: Path, missing_ok: bool = False) -> None:
        if path.name == drop.ref.split(":")[1] + ".md":
            raise PermissionError("Synthetic deletion refusal")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_drop)
    wire = ScriptedWire(
        [
            response(
                call(
                    "memory_merge",
                    {
                        "keep_ref": keep.ref,
                        "keep_version": keep.version,
                        "drop_ref": drop.ref,
                        "drop_version": drop.version,
                    },
                )
            ),
            response(text="Done"),
        ]
    )
    monkeypatch.setattr(executor_module, "create_client", ClientFactory(wire))
    log = RecordingLog()
    result = await ICodeExecutor(prepared(), store, log, RUN_ID).run(batch())
    assert (result.termination, result.failure_count, result.merged_count) == ("failed", 1, 1)
    assert result.saved[0].ref == keep.ref and result.saved[0].version == (await store.get(PROJECT, keep.ref)).version
    assert len(await store.list_entries(PROJECT)) == 2
    assert [change.operation for change in log.changes] == ["merge"]
