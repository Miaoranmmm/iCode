# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Protocol completeness, middleware order, and cleanup through real decoding."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_custom_tool_call import ChatCompletionMessageCustomToolCall, Custom
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall, Function
from openai.types.responses import ResponseFunctionToolCall

from chrys.app.memory_hosts import executor_icode as module
from chrys.app.memory_hosts.executor_icode import ExtractionGuard, ICodeExecutor, RawUsage, prepare_profile
from chrys.kernel import ChatContext, ChatResponse, Content, Message
from chrys.service.agent_middleware.response_validation import ResponseValidationMiddleware
from chrys.service.llm.chat_completions.client import DEEPSEEK
from chrys.service.llm.chat_completions.decode import decode_completion as decode_chat
from chrys.service.memory.extraction import FULL_V3, FULL_V4, USER_V2, ExtractionError, MaterialProfile
from chrys.service.memory.storage.file_store import FileMemoryStore
from tests.app.memory_hosts.executor_fakes import (
    RUN_ID,
    ClientFactory,
    RecordingLog,
    ScriptedWire,
    batch,
    call,
    prepared,
    profile,
    response,
)
from tests.support.waiting import wait_for


@pytest.mark.parametrize("truncated", [False, True])
def test_chat_decoder_loss_does_not_justify_allowing_that_profile(truncated: bool) -> None:
    tool = (
        ChatCompletionMessageFunctionToolCall.model_construct(
            type="function", id="c", function=Function.model_construct(name="", arguments="{")
        )
        if truncated
        else ChatCompletionMessageCustomToolCall.model_construct(
            type="custom", id="c", custom=Custom.model_construct(name="custom", input="data")
        )
    )
    raw = ChatCompletion.model_construct(
        id="completion",
        model="synthetic",
        created=0,
        object="chat.completion",
        usage=None,
        choices=[
            Choice.model_construct(
                index=0,
                finish_reason="length" if truncated else "stop",
                message=ChatCompletionMessage.model_construct(
                    role="assistant", content="Done", tool_calls=[tool], refusal=None
                ),
            )
        ],
    )
    decoded = decode_chat(raw, {}, variant=DEEPSEEK)
    assert all(content.type != "function_call" for message in decoded.messages for content in message.contents)
    with pytest.raises(ExtractionError) as caught:
        prepare_profile(profile(api_style="chat_completions"))
    assert caught.value.kind == "unsupported_extraction_protocol"


@pytest.mark.parametrize("flag", ["informational_only", "provider_hosted"])
async def test_decoded_non_native_function_call_is_rejected_before_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: str
) -> None:
    def mark(decoded: ChatResponse) -> None:
        content = decoded.messages[0].contents[0]
        if flag == "informational_only":
            content.informational_only = True
        else:
            content.provider_hosted = True

    wire = ScriptedWire([response(call("memory_search", {"query": ""}))], decoded_hook=mark)
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    result = await ICodeExecutor(prepared(), FileMemoryStore(tmp_path / "memory"), RecordingLog(), RUN_ID).run(batch())
    assert result.stop_reason == "unexpected_tool_call" and result.failure_count == 1
    assert result.call_count == 0


async def test_missing_required_raw_call_id_is_rejected_after_real_decode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = ResponseFunctionToolCall.model_construct(
        type="function_call", name="memory_save", arguments="{}", id="item", status="completed"
    )
    wire = ScriptedWire([response(call("memory_search", {"query": ""}), missing)])
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    result = await ICodeExecutor(prepared(), FileMemoryStore(tmp_path / "memory"), RecordingLog(), RUN_ID).run(batch())
    assert result.termination == "failed" and result.failure_count == 1 and result.call_count == 0
    assert result.stop_reason == "invalid_call_id"


async def test_same_sdk_object_echo_duplicate_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    item = call("memory_search", {"query": ""})
    wire = ScriptedWire([response(item, item)])
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    result = await ICodeExecutor(prepared(), FileMemoryStore(tmp_path / "memory"), RecordingLog(), RUN_ID).run(batch())
    assert result.stop_reason == "duplicate_call_id" and result.call_count == 0


async def test_call_accepted_by_guard_but_echo_skipped_is_unseen_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    held: list[Content] = []

    def echo(decoded: ChatResponse) -> None:
        calls = [c for message in decoded.messages for c in message.contents if c.type == "function_call"]
        if not calls:
            return
        if held:
            decoded.messages[0].contents = [held[0]]
        else:
            held.append(calls[0])

    wire = ScriptedWire(
        [response(call("memory_search", {"query": ""})), response(call("memory_search", {"query": ""}), text="Done")],
        decoded_hook=echo,
    )
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    result = await ICodeExecutor(prepared(), FileMemoryStore(tmp_path / "memory"), RecordingLog(), RUN_ID).run(batch())
    assert result.termination == "failed" and result.unsettled_count == 1 and result.failure_count >= 1
    assert result.stop_reason == "unseen_call"
    assert ("executor:unsettled", 1) in result.failure_kinds


async def test_validation_guard_usage_order_and_no_second_registration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: list[str] = []
    original_validation = ResponseValidationMiddleware.process
    original_guard = ExtractionGuard.process
    original_usage = RawUsage.process

    async def validation(
        self: ResponseValidationMiddleware, context: ChatContext, next_call: Callable[[], Awaitable[None]]
    ) -> None:
        observed.append("validation_in")
        await original_validation(self, context, next_call)
        observed.append("validation_out")

    async def guard(self: ExtractionGuard, context: ChatContext, next_call: Callable[[], Awaitable[None]]) -> None:
        observed.append("guard_in")
        await original_guard(self, context, next_call)
        observed.append("guard_out")

    async def usage(self: RawUsage, context: ChatContext, next_call: Callable[[], Awaitable[None]]) -> None:
        observed.append("usage_in")
        await original_usage(self, context, next_call)
        observed.append("usage_out")

    monkeypatch.setattr(ResponseValidationMiddleware, "process", validation)
    monkeypatch.setattr(ExtractionGuard, "process", guard)
    monkeypatch.setattr(RawUsage, "process", usage)
    wire = ScriptedWire([response(call("memory_search", {"query": ""})), response(text="Done")])
    factory = ClientFactory(wire)
    monkeypatch.setattr(module, "create_client", factory)
    result = await ICodeExecutor(prepared(), FileMemoryStore(tmp_path / "memory"), RecordingLog(), RUN_ID).run(batch())
    assert result.termination == "completed"
    assert observed == ["validation_in", "guard_in", "usage_in", "usage_out", "guard_out", "validation_out"] * 2
    assert factory.client is not None and factory.client.inner.chat_middleware == []


async def test_reasoning_only_output_limit_stops_before_visible_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reasoning = SimpleNamespace(
        type="reasoning",
        id="r",
        summary=[],
        content=[SimpleNamespace(type="reasoning_text", text="internal reasoning")],
        encrypted_content=None,
    )
    wire = ScriptedWire([response(reasoning, status="incomplete", reason="max_output_tokens")])
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    result = await ICodeExecutor(prepared(), FileMemoryStore(tmp_path / "memory"), RecordingLog(), RUN_ID).run(batch())
    assert result.termination == "output_limited" and result.failure_count == 0 and result.call_count == 0


@pytest.mark.parametrize(
    "material_profile,requests,stop_reason,tail_seen",
    [
        (USER_V2, 2, "tool_error_stop", True),
        (FULL_V3, 2, "tool_error_stop", True),
        (FULL_V4, 1, "tool_failure", False),
    ],
    ids=["user-v2", "full-v3", "full-v4"],
)
async def test_prior_failure_and_turn_tail_cannot_be_clean_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    material_profile: MaterialProfile,
    requests: int,
    stop_reason: str,
    tail_seen: bool,
) -> None:
    wire = ScriptedWire([response(call("memory_save", {})), response(call("memory_search", {"query": ""}))])
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    result = await ICodeExecutor(
        prepared(),
        FileMemoryStore(tmp_path / "memory"),
        RecordingLog(),
        RUN_ID,
        max_turns=2,
        material_profile=material_profile,
    ).run(batch())
    assert result.termination == "failed" and result.failure_count >= 1
    assert result.stop_reason == stop_reason
    assert len(wire.requests) == requests and result.tail_seen is tail_seen


@pytest.mark.parametrize(
    "material_profile,termination,failures,rejections,requests",
    [
        (USER_V2, "failed", 10, 0, 11),
        (FULL_V3, "failed", 10, 0, 11),
        (FULL_V4, "completed", 0, 10, 11),
    ],
    ids=["user-v2", "full-v3", "full-v4"],
)
async def test_failed_metadata_respects_profile_without_false_kernel_error_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    material_profile: MaterialProfile,
    termination: str,
    failures: int,
    rejections: int,
    requests: int,
) -> None:
    wire = ScriptedWire(
        [response(call("memory_read", {"ref": "project:000000000000"})) for _ in range(10)] + [response(text="Done")]
    )
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    result = await ICodeExecutor(
        prepared(),
        FileMemoryStore(tmp_path / "memory"),
        RecordingLog(),
        RUN_ID,
        material_profile=material_profile,
    ).run(batch())
    assert (result.termination, result.failure_count, result.rejection_count) == (termination, failures, rejections)
    assert result.tail_seen is False and result.stop_reason != "tool_error_stop" and len(wire.requests) == requests


async def test_setup_failure_still_closes_factory_stack(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wire = ScriptedWire([])
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))

    def fail_agent(
        client: object,
        *,
        instructions: str,
        tools: Sequence[object],
        context_providers: Sequence[object],
        middleware: Sequence[object],
    ) -> None:
        raise RuntimeError("secret setup")

    monkeypatch.setattr(module, "Agent", fail_agent)
    result = await ICodeExecutor(prepared(), FileMemoryStore(tmp_path / "memory"), RecordingLog(), RUN_ID).run(batch())
    assert result.termination == "failed" and result.stop_reason == "executor_setup_error"
    assert wire.closed == 1 and wire.requests == []


async def test_client_phase_cancel_closes_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(messages: Sequence[Message]):
        entered.set()
        await release.wait()
        return response(text="Done")

    wire = ScriptedWire([blocked])
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    task = asyncio.create_task(
        ICodeExecutor(prepared(), FileMemoryStore(tmp_path / "memory"), RecordingLog(), RUN_ID).run(batch())
    )
    try:
        await wait_for(lambda: entered.is_set() or task.done())
        if task.done():
            await task
        assert entered.is_set() and not task.done()
        task.cancel()
        result = await task
        assert result.termination == "cancelled" and result.cancel_requested is True
        assert wire.closed == 1 and result.request_count == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_model_timeout_is_failure_and_closes_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    entered = asyncio.Event()

    async def blocked(messages: Sequence[Message]):
        entered.set()
        await asyncio.Event().wait()
        return response(text="unreachable")

    wire = ScriptedWire([blocked])
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    result = await ICodeExecutor(
        prepared(), FileMemoryStore(tmp_path / "memory"), RecordingLog(), RUN_ID, timeout_seconds=0.1
    ).run(batch())
    assert entered.is_set()
    assert result.termination == "failed" and result.stop_reason == "timeout" and result.cancel_requested is False
    assert wire.closed == 1


@pytest.mark.parametrize("status,reason", [("failed", None), ("cancelled", None), ("incomplete", "content_filter")])
async def test_terminal_protocol_failure_is_not_clean_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str, reason: str | None
) -> None:
    wire = ScriptedWire([response(text="cannot finish", status=status, reason=reason)])
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    result = await ICodeExecutor(prepared(), FileMemoryStore(tmp_path / "memory"), RecordingLog(), RUN_ID).run(batch())
    assert result.termination == "failed" and result.failure_count == 1
    assert result.cancel_requested is False and wire.closed == 1


async def test_continuation_is_failed_even_with_completed_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def continuing(decoded: ChatResponse) -> None:
        decoded.continuation_token = {"response_id": "remote-token"}

    wire = ScriptedWire([response(text="Done")], decoded_hook=continuing)
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    result = await ICodeExecutor(prepared(), FileMemoryStore(tmp_path / "memory"), RecordingLog(), RUN_ID).run(batch())
    assert result.termination == "failed" and result.stop_reason == "continuation"


async def test_blank_completed_response_is_validation_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wire = ScriptedWire([response(text=" ")])
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    result = await ICodeExecutor(prepared(), FileMemoryStore(tmp_path / "memory"), RecordingLog(), RUN_ID).run(batch())
    assert result.termination == "failed" and result.stop_reason == "client_or_validation_error"
    assert result.failure_count == 1 and wire.closed == 1
