# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Candidate/error semantics, missing usage, and owned-operation cancellation."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage

from chrys.app.memory_hosts import executor_icode as module
from chrys.app.memory_hosts.executor_icode import ICodeExecutor, RawUsage
from chrys.app.memory_hosts.extraction_tasks import OperationNotAdmitted, OperationOwner
from chrys.foundation.errors import ProviderResponseError
from chrys.foundation.tool_result_metadata import TOOL_FAILED_METADATA_KEY, TOOL_RESULT_METADATA_KEY
from chrys.kernel import ChatContext, ChatResponse, LoopRecorder, Message
from chrys.service.memory.extraction import FULL_V3, FULL_V4, USER_V2, MaterialProfile
from chrys.service.memory.storage import file_store
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import MemoryOperationError
from chrys.service.memory.verification import SourceMessage
from tests.app.memory_hosts.executor_fakes import (
    PROJECT,
    RUN_ID,
    SESSION,
    WORDS,
    ClientFactory,
    RecordingLog,
    ScriptedWire,
    batch,
    call,
    prepared,
    response,
    save_arguments,
)
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for


async def test_pure_ambiguity_is_candidate_not_failed_and_can_be_abandoned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = batch(
        messages=(SourceMessage(SESSION, 1, (WORDS,), item="one"), SourceMessage(SESSION, 2, (WORDS,), item="two"))
    )
    wire = ScriptedWire([response(call("memory_save", save_arguments())), response(text="Done")])
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    store, log = FileMemoryStore(tmp_path / "memory"), RecordingLog()
    result = await ICodeExecutor(prepared(), store, log, RUN_ID).run(source)
    assert (result.termination, result.candidate_count, result.failure_count, result.created_count) == (
        "completed",
        1,
        0,
        0,
    )
    assert await store.list_entries(PROJECT) == [] and log.changes == []
    answer = next(
        content for message in wire.requests[-1][0] for content in message.contents if content.type == "function_result"
    )
    text = str(answer.result)
    assert text.startswith("Not saved. The quote could not be verified.") and "candidate" in text
    assert answer.additional_properties[TOOL_RESULT_METADATA_KEY][TOOL_FAILED_METADATA_KEY] is False


async def test_candidate_retry_preserves_typed_outcome_and_then_saves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = batch(
        messages=(SourceMessage(SESSION, 1, (WORDS,), item="one"), SourceMessage(SESSION, 2, (WORDS,), item="two"))
    )
    from chrys.service.memory.verification import QuoteClaim, SourceVerificationError, verify_quotes

    with pytest.raises(SourceVerificationError) as caught:
        verify_quotes(
            (QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", WORDS, session=SESSION),),
            source.source_index.view,
        )
    token = caught.value.failures[0].candidates[0].token
    wire = ScriptedWire(
        [
            response(call("memory_save", save_arguments())),
            response(
                call(
                    "memory_save", save_arguments(user_quotes=[{"text": WORDS, "session": SESSION, "candidate": token}])
                )
            ),
            response(text="Done"),
        ]
    )
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    log, store = RecordingLog(), FileMemoryStore(tmp_path / "memory")
    result = await ICodeExecutor(prepared(), store, log, RUN_ID).run(source)
    assert (result.termination, result.candidate_count, result.created_count, result.failure_count) == (
        "completed",
        1,
        1,
        0,
    )
    assert len(log.changes) == 1 and log.changes[0].source == RUN_ID


@pytest.mark.parametrize(
    "quotes",
    [
        [{"text": "not present", "session": SESSION}],
        [{"text": WORDS, "session": SESSION}, {"text": "not present", "session": SESSION}],
        [{"text": WORDS, "session": SESSION, "candidate": "expired-token"}],
    ],
)
@pytest.mark.parametrize("material_profile", [USER_V2, FULL_V3], ids=["user-v2", "full-v3"])
async def test_legacy_verification_errors_stay_failed_after_later_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    quotes: list[dict[str, str]],
    material_profile: MaterialProfile,
) -> None:
    source = batch(
        messages=(SourceMessage(SESSION, 1, (WORDS,), item="one"), SourceMessage(SESSION, 2, (WORDS,), item="two"))
    )
    wire = ScriptedWire(
        [
            response(call("memory_save", save_arguments(user_quotes=quotes))),
            response(call("memory_search", {"query": ""})),
            response(text="Done"),
        ]
    )
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    log, store = RecordingLog(), FileMemoryStore(tmp_path / "memory")
    result = await ICodeExecutor(prepared(), store, log, RUN_ID, material_profile=material_profile).run(source)
    assert result.termination == "failed" and result.failure_count == 1
    assert result.rejection_count == 0 and len(wire.requests) == 3
    assert result.call_count == 2 and result.success_count == 1
    assert log.changes == []


async def test_a_mistyped_session_is_a_candidate_and_the_corrected_retry_saves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mistyped = SESSION[:-1] + ("0" if SESSION[-1] != "0" else "1")
    source = batch()
    from chrys.service.memory.verification import QuoteClaim, SourceVerificationError, verify_quotes

    with pytest.raises(SourceVerificationError) as caught:
        verify_quotes(
            (QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", WORDS, session=mistyped),),
            source.source_index.view,
        )
    [offered] = caught.value.failures[0].candidates
    assert offered.session == SESSION
    wire = ScriptedWire(
        [
            response(call("memory_save", save_arguments(user_quotes=[{"text": WORDS, "session": mistyped}]))),
            response(
                call(
                    "memory_save",
                    save_arguments(user_quotes=[{"text": WORDS, "session": SESSION, "candidate": offered.token}]),
                )
            ),
            response(text="Done"),
        ]
    )
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    log, store = RecordingLog(), FileMemoryStore(tmp_path / "memory")
    result = await ICodeExecutor(prepared(), store, log, RUN_ID).run(source)
    assert (result.termination, result.candidate_count, result.created_count, result.failure_count) == (
        "completed",
        1,
        1,
        0,
    )
    [entry] = await store.list_entries(PROJECT)
    assert [quote.session for quote in entry.entry.quotes] == [SESSION]
    first_answer = next(
        content for message in wire.requests[1][0] for content in message.contents if content.type == "function_result"
    )
    text = str(first_answer.result)
    assert text.startswith("Not saved.") and f"session {SESSION}" in text and offered.token in text


@pytest.mark.parametrize(
    "error",
    [
        MemoryOperationError("memory_store_unavailable", "Unavailable."),
        MemoryOperationError("memory_version_conflict", "Changed."),
        RuntimeError("secret body credential"),
    ],
)
@pytest.mark.parametrize("material_profile", [USER_V2, FULL_V3, FULL_V4], ids=["user-v2", "full-v3", "full-v4"])
async def test_save_errors_are_typed_safe_and_do_not_become_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    material_profile: MaterialProfile,
) -> None:
    store = FileMemoryStore(tmp_path / "memory")

    async def fail_create(scope, draft):
        raise error

    monkeypatch.setattr(store, "create", fail_create)
    wire = ScriptedWire([response(call("memory_save", save_arguments())), response(text="Done")])
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    recorder = LoopRecorder()
    monkeypatch.setattr(module, "LoopRecorder", lambda: recorder)
    result = await ICodeExecutor(prepared(), store, RecordingLog(), RUN_ID, material_profile=material_profile).run(
        batch()
    )
    assert result.termination == "failed" and result.failure_count == 1 and result.candidate_count == 0
    assert len(wire.requests) == (1 if material_profile is FULL_V4 else 2) and recorder.loop_messages is not None
    answered = next(c for message in recorder.loop_messages for c in message.contents if c.type == "function_result")
    assert answered.additional_properties[TOOL_RESULT_METADATA_KEY][TOOL_FAILED_METADATA_KEY] is True
    if isinstance(error, RuntimeError):
        assert "Memory operation failed." in str(answered.result)
        assert "secret body credential" not in str(answered.result) + caplog.text + repr(result)


async def test_exact_existing_create_emits_no_second_change(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wire = ScriptedWire(
        [
            response(call("memory_save", save_arguments())),
            response(call("memory_save", save_arguments())),
            response(text="Done"),
        ]
    )
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    log = RecordingLog()
    result = await ICodeExecutor(prepared(), FileMemoryStore(tmp_path / "memory"), log, RUN_ID).run(batch())
    assert (result.created_count, result.existing_count, result.failure_count) == (1, 1, 0)
    assert len(log.changes) == 1


async def test_usage_real_decoder_preserves_zero_reasoning_cache_and_no_double_total(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    usage = ResponseUsage.model_construct(
        input_tokens=100,
        output_tokens=40,
        total_tokens=140,
        input_tokens_details=InputTokensDetails.model_construct(cached_tokens=0),
        output_tokens_details=OutputTokensDetails.model_construct(reasoning_tokens=30),
    )
    wire = ScriptedWire([response(call("memory_search", {"query": ""}), usage=usage), response(text="Done")])
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    result = await ICodeExecutor(prepared(), FileMemoryStore(tmp_path / "memory"), RecordingLog(), RUN_ID).run(batch())
    fields = {field.name: (field.observed_sum, field.missing_calls) for field in result.usage.fields}
    assert fields["input_token_count"] == (100, 1)
    assert fields["output_token_count"] == (40, 1)
    assert fields["total_token_count"] == (140, 1)
    assert fields["reasoning_output_token_count"] == (30, 1)
    assert fields["cache_read_input_token_count"] == (0, 1)
    assert result.usage.calls == result.request_count == 2
    assert result.first_input_tokens == 100


@pytest.mark.parametrize("failure", [False, True])
async def test_raw_usage_collector_partial_missing_and_failed_response(failure: bool) -> None:
    usage = RawUsage()
    context = ChatContext(None, [], {}, result=None)
    details = {"input_token_count": 0, "output_token_count": None, "reasoning_output_token_count": 7}

    async def finish() -> None:
        if failure:
            raise ProviderResponseError("server_error", "private", retryable=False, usage_details=details)
        context.result = ChatResponse(messages=[Message("assistant", ["Done"])], usage_details=details)

    if failure:
        with pytest.raises(ProviderResponseError):
            await usage.process(context, finish)
    else:
        await usage.process(context, finish)
    fields = {field.name: (field.observed_sum, field.missing_calls) for field in usage.summary().fields}
    assert fields["input_token_count"] == (0, 0)
    assert fields["output_token_count"] == (0, 1)
    assert fields["total_token_count"] == (0, 1)
    assert fields["reasoning_output_token_count"] == (7, 0)
    assert usage.summary().calls == 1


async def test_owner_drains_repeated_cancel_and_returns_committed_result() -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    owner = OperationOwner()

    async def operation() -> str:
        entered.set()
        await release.wait()
        return "committed"

    task = asyncio.create_task(owner.run(operation()))
    try:
        await wait_for(lambda: entered.is_set() or task.done())
        if task.done():
            await task
        assert entered.is_set() and not task.done()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        assert not task.done() and owner.pending_count == 1
        release.set()
        assert await task == "committed"
        assert owner.cancel_requested is True and owner.pending_count == 0
        with pytest.raises(OperationNotAdmitted):
            await owner.run(operation())
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_real_store_save_cancel_does_not_close_client_until_worker_and_log_settle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entered, release = threading.Event(), threading.Event()
    real_write = file_store.atomic_write_owner_only_bytes

    def blocked_write(path: Path, data: bytes, *, create_parents: bool = True) -> None:
        if path.suffix == ".md":
            entered.set()
            if not release.wait(ENGINE_TURN_TIMEOUT):
                raise TimeoutError("test release missing")
        real_write(path, data, create_parents=create_parents)

    monkeypatch.setattr(file_store, "atomic_write_owner_only_bytes", blocked_write)
    wire = ScriptedWire([response(call("memory_save", save_arguments())), response(text="unreachable")])
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    log, store = RecordingLog(), FileMemoryStore(tmp_path / "memory")
    executor = ICodeExecutor(prepared(), store, log, RUN_ID)
    task = asyncio.create_task(executor.run(batch()))
    try:
        await wait_for(lambda: entered.is_set() or task.done(), timeout=ENGINE_TURN_TIMEOUT)
        if task.done():
            await task
        assert entered.is_set() and not task.done() and wire.closed == 0
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        assert not task.done() and wire.closed == 0
        release.set()
        result = await task
        assert result.termination == "cancelled" and result.stop_reason == "saved_then_cancelled"
        assert result.created_count == 1 and result.saved[0].outcome == "created"
        assert result.cancel_requested is True and result.unsettled_count == 0 and result.failure_count == 0
        assert len(await store.list_entries(PROJECT)) == 1 and len(log.changes) == 1
        assert log.changes[0].source == RUN_ID and wire.closed == 1 and len(wire.requests) == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_cancel_before_native_admission_writes_nothing() -> None:
    owner = OperationOwner()
    owner.cancel_requested = True
    ran = False

    async def operation() -> str:
        nonlocal ran
        ran = True
        return "unexpected"

    with pytest.raises(OperationNotAdmitted):
        await owner.run(operation())
    assert ran is False


async def test_change_log_failure_does_not_invert_saved_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from chrys.service.memory.change_log import MemoryChange

    class FailingLog:
        async def append(self, change: MemoryChange) -> None:
            raise OSError("private path credential")

    wire = ScriptedWire([response(call("memory_save", save_arguments())), response(text="Done")])
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    store = FileMemoryStore(tmp_path / "memory")
    result = await ICodeExecutor(prepared(), store, FailingLog(), RUN_ID).run(batch())
    assert result.termination == "completed" and result.created_count == 1 and result.failure_count == 0
    assert len(await store.list_entries(PROJECT)) == 1
    assert "Memory change logging failed after a successful write." in caplog.text
    assert "private path credential" not in caplog.text


async def test_timeout_during_real_save_drains_receipt_and_is_failure_not_user_cancel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entered, release = threading.Event(), threading.Event()
    owner = OperationOwner()

    def make_owner() -> OperationOwner:
        return owner

    monkeypatch.setattr(module, "OperationOwner", make_owner)
    real_write = file_store.atomic_write_owner_only_bytes

    def blocked_write(path: Path, data: bytes, *, create_parents: bool = True) -> None:
        if path.suffix == ".md":
            entered.set()
            if not release.wait(ENGINE_TURN_TIMEOUT):
                raise TimeoutError("test release missing")
        real_write(path, data, create_parents=create_parents)

    monkeypatch.setattr(file_store, "atomic_write_owner_only_bytes", blocked_write)
    wire = ScriptedWire([response(call("memory_save", save_arguments()))])
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    log, store = RecordingLog(), FileMemoryStore(tmp_path / "memory")
    task = asyncio.create_task(ICodeExecutor(prepared(), store, log, RUN_ID, timeout_seconds=0.1).run(batch()))
    try:
        await wait_for(lambda: entered.is_set() or task.done(), timeout=ENGINE_TURN_TIMEOUT)
        if task.done():
            await task
        assert entered.is_set()
        await wait_for(lambda: owner.cancel_requested or task.done(), timeout=ENGINE_TURN_TIMEOUT)
        assert owner.cancel_requested and not task.done() and wire.closed == 0
        release.set()
        result = await task
        assert result.termination == "failed" and result.stop_reason == "timeout"
        assert result.cancel_requested is False and result.failure_count > 0
        assert result.created_count == 1 and len(result.saved) == 1
        assert len(await store.list_entries(PROJECT)) == len(log.changes) == 1
        assert log.changes[0].source == RUN_ID and wire.closed == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_extraction_read_checks_file_basis_against_the_bound_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import hashlib
    from dataclasses import replace

    from chrys.service.memory.storage.store import FileEvidence, MemoryEntryDraft

    root = tmp_path / "project"
    await asyncio.to_thread(root.mkdir)
    payload = b"fixed_port = 8765\n"
    await asyncio.to_thread((root / "runner.py").write_bytes, payload)
    project = replace(PROJECT, root=str(root))
    store = FileMemoryStore(tmp_path / "memory")
    existing = await store.create(
        project,
        MemoryEntryDraft(
            "engineering",
            "The runner uses a fixed port",
            "Tests must run serially.",
            "model",
            evidence=(
                FileEvidence(str(PROJECT.project_id), "runner.py", hashlib.sha256(payload).hexdigest(), len(payload)),
            ),
        ),
    )
    wire = ScriptedWire([response(call("memory_read", {"ref": existing.ref})), response(text="Done")])
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    result = await ICodeExecutor(prepared(), store, RecordingLog(), RUN_ID).run(replace(batch(), project=project))
    assert result.termination == "completed" and result.created_count == 0
    tool_results = [
        str(content.result)
        for message in wire.requests[-1][0]
        for content in message.contents
        if content.type == "function_result"
    ]
    assert len(tool_results) == 1 and "Basis: unchanged" in tool_results[0]
