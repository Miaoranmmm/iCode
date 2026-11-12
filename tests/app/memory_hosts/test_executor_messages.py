# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Two extraction wires share admission and settlement, using real wire decoders."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from anthropic.types.beta import (
    BetaContainerUploadBlock,
    BetaMCPToolResultBlock,
    BetaMCPToolUseBlock,
    BetaRedactedThinkingBlock,
    BetaServerToolUseBlock,
    BetaThinkingBlock,
    BetaUsage,
    BetaWebSearchToolResultBlock,
)

from chrys.app.memory_hosts import executor_icode as module
from chrys.app.memory_hosts.executor_icode import ICodeExecutor, prepare_profile
from chrys.app.memory_hosts.extract import ExtractionOptions, run_extraction
from chrys.kernel import ChatResponse, Content, FunctionTool, Message
from chrys.service.llm.anthropic_messages.request import build_request as messages_request
from chrys.service.llm.openai_responses.client import DEEPSEEK_RESPONSES
from chrys.service.llm.openai_responses.request import build_request as responses_request
from chrys.service.memory.extraction import ExtractionError
from chrys.service.memory.project import bind_project
from chrys.service.memory.storage.file_store import FileMemoryStore
from tests.app.memory_hosts.executor_fakes import (
    PROJECT,
    RUN_ID,
    ClientFactory,
    RecordingLog,
    ScriptedWire,
    batch,
    call,
    messages_call,
    messages_profile,
    messages_response,
    profile,
    response,
    save_arguments,
)
from tests.app.memory_hosts.test_extract import Reader


@pytest.mark.parametrize("wire", ["responses", "messages"])
def test_prepared_profile_admits_real_shape_and_preserves_original(wire: str) -> None:
    original = messages_profile() if wire == "messages" else profile()
    ready = prepare_profile(original)
    assert ready.wire == wire
    assert ready.profile is not original
    assert ready.profile.max_output_tokens == 16384
    options = json.loads(ready.profile.chat_options)
    assert options["tool_choice"] == "auto"
    assert ("store" in options) is (wire == "responses")
    assert "tool_choice" not in json.loads(original.chat_options)


@pytest.mark.parametrize("wire", ["responses", "messages"])
@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize(
    "key",
    [
        "thread",
        "stream",
        "messages",
        "input",
        "context_management",
        "mcp_servers",
        "tool_choice",
        "future_option",
        "continuation_token",
        "conversation",
        "conversation_id",
        "previous_response_id",
    ],
)
def test_options_allowlist_refuses_unknown_keys(wire: str, extra: bool, key: str) -> None:
    payload = {"extra_body": {key: "sentinel"}} if extra else {key: "sentinel"}
    original = messages_profile() if wire == "messages" else profile()
    with pytest.raises(ExtractionError) as error:
        prepare_profile(replace(original, chat_options=json.dumps(payload)))
    assert error.value.kind == "memory_invalid_input"
    assert (f"extra_body.{key}" if extra else key) in error.value.message


@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize("budget,valid", [(1023, False), (1024, True), (4095, True), (4096, False), (True, False)])
def test_thinking_budget_bounds(extra: bool, budget: int, valid: bool) -> None:
    payload: dict[str, Any] = {"thinking": {"type": "enabled", "budget_tokens": budget}}
    if extra:
        payload = {"extra_body": payload}
    original = replace(messages_profile(), chat_options=json.dumps(payload), max_output_tokens=4096)
    if valid:
        assert prepare_profile(original).profile.max_output_tokens == 4096
    else:
        with pytest.raises(ExtractionError) as error:
            prepare_profile(original)
        assert error.value.kind == "memory_invalid_input"


@pytest.mark.parametrize("extra", [False, True])
def test_output_config_only_allows_effort(extra: bool) -> None:
    payload = {"output_config": {"effort": "high", "format": {"type": "json"}}}
    with pytest.raises(ExtractionError, match=r"output_config\.format"):
        prepare_profile(
            replace(messages_profile(), chat_options=json.dumps({"extra_body": payload} if extra else payload))
        )


def test_real_request_builders_get_safe_caps_cache_and_auto_choice() -> None:
    tool = FunctionTool(
        name="memory_search", description="search", func=lambda: "", input_model={"type": "object", "properties": {}}
    )
    for original in (profile(), messages_profile()):
        ready = prepare_profile(original)
        options = {**json.loads(ready.profile.chat_options), "tools": [tool]}
        history = [Message(role="user", contents=[Content.from_text("synthetic input")])]
        if ready.wire == "messages":
            request = messages_request(
                history, options, {}, model="synthetic", base_url="https://relay.invalid", default_headers={}
            ).request
            assert "store" not in request and "store" not in request.get("extra_body", {})
            assert request["extra_body"]["cache_control"] == {"type": "ephemeral"}
            assert request["tool_choice"] == {"type": "auto"}
            assert request["max_tokens"] == 16384
        else:
            request = responses_request(history, options, model="synthetic", variant=DEEPSEEK_RESPONSES)
            assert request["store"] is False and request["tool_choice"] == "auto"
            assert request["max_output_tokens"] == 16384


async def execute_messages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, script: list[Any]):
    wire = ScriptedWire(script)
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    store = FileMemoryStore(tmp_path / "memory")
    result = await ICodeExecutor(prepare_profile(messages_profile()), store, RecordingLog(), RUN_ID).run(batch())
    return result, store, wire


@pytest.mark.parametrize(
    "reason,with_call,expected,kind",
    [
        ("end_turn", False, "completed", None),
        ("end_turn", True, "failed", "contradictory_response_status"),
        ("tool_use", True, "completed", None),
        ("tool_use", False, "failed", "contradictory_response_status"),
        ("max_tokens", False, "output_limited", "max_output_tokens"),
        ("refusal", False, "failed", "refusal"),
        ("refusal", True, "failed", "client_or_validation_error"),
        ("model_context_window_exceeded", False, "failed", "context_window_exceeded"),
        ("pause_turn", False, "failed", "contradictory_response_status"),
        ("stop_sequence", False, "failed", "contradictory_response_status"),
        ("compaction", False, "failed", "contradictory_response_status"),
        (None, False, "failed", "invalid_response_status"),
        ("future_reason", False, "failed", "invalid_response_status"),
    ],
)
async def test_messages_raw_stop_table(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reason: str | None,
    with_call: bool,
    expected: str,
    kind: str | None,
) -> None:
    script = [messages_response(messages_call("memory_search", {"query": ""}), reason="tool_use")]
    script.append(
        messages_response(
            *([messages_call("memory_search", {"query": ""})] if with_call else []), reason=reason, text="Done"
        )
    )
    if with_call and expected == "completed":
        script.append(messages_response(text="Done"))
    result, _, _ = await execute_messages(monkeypatch, tmp_path, script)
    assert (result.termination, result.stop_reason) == (expected, kind)


@pytest.mark.parametrize("bad_id", ["", None, 7, "call-one"])
async def test_messages_raw_ids_reject_entire_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_id: Any
) -> None:
    result, store, _ = await execute_messages(
        monkeypatch,
        tmp_path,
        [
            messages_response(
                messages_call("memory_search", {"query": ""}),
                messages_call("memory_save", save_arguments(), bad_id),
                reason="tool_use",
            )
        ],
    )
    assert result.stop_reason == ("duplicate_call_id" if bad_id == "call-one" else "invalid_call_id")
    assert result.call_count == 0 and await store.list_entries(PROJECT) == []


async def test_messages_decoder_dropped_block_after_success_still_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, _, _ = await execute_messages(
        monkeypatch,
        tmp_path,
        [
            messages_response(messages_call("memory_search", {"query": ""}), reason="tool_use"),
            messages_response(BetaContainerUploadBlock(type="container_upload", file_id="synthetic-file"), text="Done"),
        ],
    )
    assert result.termination == "failed" and result.success_count == 1
    assert result.stop_reason == "unsupported_raw_output"
    assert result.unsupported_raw_output == (("container_upload", 1),)


@pytest.mark.parametrize("wire", ["responses", "messages"])
@pytest.mark.parametrize("prior_failure", [False, True])
async def test_limited_save_never_dispatches_and_prior_failure_is_sticky(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, wire: str, prior_failure: bool
) -> None:
    if wire == "messages":
        script = [messages_response(messages_call("memory_save", {}), reason="tool_use")] if prior_failure else []
        script.append(messages_response(messages_call("memory_save", save_arguments()), reason="max_tokens"))
        original = messages_profile()
    else:
        partial = call("memory_save", save_arguments())
        partial.status = "incomplete"
        script = [response(call("memory_save", {}))] if prior_failure else []
        script.append(response(partial, status="incomplete", reason="max_output_tokens"))
        original = profile()
    wire_client = ScriptedWire(script)
    monkeypatch.setattr(module, "create_client", ClientFactory(wire_client))
    store = FileMemoryStore(tmp_path / "memory")
    result = await ICodeExecutor(prepare_profile(original), store, RecordingLog(), RUN_ID).run(batch())
    assert result.termination == ("failed" if prior_failure else "output_limited")
    assert result.call_count == int(prior_failure) and result.unsettled_count == 0
    assert await store.list_entries(PROJECT) == []


async def test_messages_search_save_end_turn_and_usage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    usage = BetaUsage.model_construct(
        input_tokens=10, output_tokens=7, cache_read_input_tokens=20, cache_creation_input_tokens=5
    )
    result, store, wire = await execute_messages(
        monkeypatch,
        tmp_path,
        [
            messages_response(messages_call("memory_search", {"query": ""}), reason="tool_use", usage=usage),
            messages_response(messages_call("memory_save", save_arguments()), reason="tool_use", usage=usage),
            messages_response(text="Done", usage=usage),
        ],
    )
    assert (result.termination, result.call_count, result.created_count) == ("completed", 2, 1)
    assert len(await store.list_entries(PROJECT)) == 1 and wire.closed == 1
    fields = {field.name: field for field in result.usage.fields}
    assert fields["input_token_count"].observed_sum == 105
    assert fields["output_token_count"].observed_sum == 21
    assert fields["cache_read_input_token_count"].observed_sum == 60
    assert fields["cache_creation_input_token_count"].observed_sum == 15
    assert fields["total_token_count"].missing_calls == 3


@pytest.mark.parametrize("wire", ["responses", "messages"])
async def test_full_entry_prepares_once_and_runs_real_executor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, wire: str
) -> None:
    home = tmp_path / "home"
    await asyncio.to_thread(home.mkdir)
    for key in ("HOME", "USERPROFILE", "APPDATA"):
        monkeypatch.setenv(key, str(home))
    project_dir = tmp_path / "project"
    await asyncio.to_thread(project_dir.mkdir)
    project = await asyncio.to_thread(bind_project, project_dir)
    original = messages_profile() if wire == "messages" else profile()
    source = replace(batch().sources[0], project=project)
    script = (
        [
            messages_response(messages_call("memory_search", {"query": ""}), reason="tool_use"),
            messages_response(text="Done"),
        ]
        if wire == "messages"
        else [response(call("memory_search", {"query": ""})), response(text="Done")]
    )
    factory = ClientFactory(ScriptedWire(script))
    monkeypatch.setattr(module, "create_client", factory)
    report = await run_extraction(
        ExtractionOptions(project_dir, tmp_path / "memory"), original, readers={"claude-code": Reader((source,))}
    )
    assert (report.status, report.exit_code) == ("completed", 0)
    assert report.profile["wire"] == wire
    assert len(factory.profiles) == 1


@pytest.mark.parametrize(
    "block",
    [
        BetaServerToolUseBlock.model_construct(type="server_tool_use", id="hosted", name="synthetic", input={}),
        BetaMCPToolUseBlock.model_construct(
            type="mcp_tool_use", id="hosted", name="synthetic", input={}, server_name="server"
        ),
        BetaMCPToolResultBlock.model_construct(
            type="mcp_tool_result", tool_use_id="hosted", content=[], is_error=False
        ),
        BetaWebSearchToolResultBlock.model_construct(type="web_search_tool_result", tool_use_id="hosted", content=[]),
    ],
)
async def test_messages_hosted_blocks_never_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, block: Any
) -> None:
    result, _, _ = await execute_messages(monkeypatch, tmp_path, [messages_response(block, text="Done")])
    assert result.termination == "failed" and result.stop_reason == "unexpected_tool_call"
    assert result.call_count == 0


async def test_messages_thinking_blocks_are_allowed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result, _, _ = await execute_messages(
        monkeypatch,
        tmp_path,
        [
            messages_response(messages_call("memory_search", {"query": ""}), reason="tool_use"),
            messages_response(
                BetaThinkingBlock(type="thinking", thinking="synthetic reasoning", signature="synthetic-signature"),
                BetaRedactedThinkingBlock(type="redacted_thinking", data="synthetic-data"),
                text="Done",
            ),
        ],
    )
    assert result.termination == "completed" and result.failure_count == 0


@pytest.mark.parametrize(
    "field,value,reason", [("type", "other", "invalid_response_status"), ("content", (), "invalid_raw_output")]
)
async def test_messages_raw_envelope_required_after_real_decode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, value: Any, reason: str
) -> None:
    def corrupt(decoded: ChatResponse) -> None:
        raw = decoded.raw_representation
        if field == "type":
            raw.type = value
        else:
            raw.content = value

    wire = ScriptedWire([messages_response(text="Done")], decoded_hook=corrupt)
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    result = await ICodeExecutor(
        prepare_profile(messages_profile()), FileMemoryStore(tmp_path / "memory"), RecordingLog(), RUN_ID
    ).run(batch())
    assert result.termination == "failed" and result.stop_reason == reason


@pytest.mark.parametrize("wire", ["responses", "messages"])
@pytest.mark.parametrize("extra", [False, True])
def test_known_option_allowlists_accept_every_declared_key(wire: str, extra: bool) -> None:
    payload: dict[str, Any] = {
        "temperature": 0.5,
        "top_p": 0.9,
        "metadata": {},
        "service_tier": "auto",
        "allow_multiple_tool_calls": True,
        "max_tokens": 90000,
        "max_output_tokens": 90001,
        "max_completion_tokens": 90002,
    }
    if wire == "responses":
        payload.update(
            reasoning={"effort": "high"}, store=True, parallel_tool_calls=True, prompt_cache_key="synthetic-cache"
        )
        original = profile()
    else:
        payload.update(
            top_k=40,
            thinking={"type": "enabled", "budget_tokens": 1024},
            betas=[],
            additional_beta_flags=[],
            thinking_block_binding="auto",
            auto_interleaved_thinking=True,
            output_config={"effort": "high"},
        )
        if extra:
            payload["cache_control"] = {"type": "ephemeral"}
        original = messages_profile()
    ready = prepare_profile(replace(original, chat_options=json.dumps({"extra_body": payload} if extra else payload)))
    options = json.loads(ready.profile.chat_options)
    assert options["max_tokens"] == 16384
    assert options["tool_choice"] == "auto"
    for location in (options, options.get("extra_body", {})):
        assert "max_output_tokens" not in location and "max_completion_tokens" not in location
    assert not any(
        alias in options.get("extra_body", {}) for alias in ("max_tokens", "max_output_tokens", "max_completion_tokens")
    )


@pytest.mark.parametrize(
    "wire,key",
    [("messages", "store"), ("messages", "reasoning"), ("responses", "thinking"), ("responses", "cache_control")],
)
@pytest.mark.parametrize("extra", [False, True])
def test_wire_specific_options_do_not_cross_allowlists(wire: str, key: str, extra: bool) -> None:
    original = messages_profile() if wire == "messages" else profile()
    payload = {key: {}}
    with pytest.raises(ExtractionError, match=key):
        prepare_profile(replace(original, chat_options=json.dumps({"extra_body": payload} if extra else payload)))


@pytest.mark.parametrize("wire", ["responses", "messages"])
async def test_native_rejection_before_limited_response_remains_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, wire: str
) -> None:
    if wire == "messages":
        script = [
            messages_response(messages_call("memory_read", {"ref": "project:000000000000"}), reason="tool_use"),
            messages_response(messages_call("memory_save", save_arguments()), reason="max_tokens"),
        ]
        original = messages_profile()
    else:
        script = [
            response(call("memory_read", {"ref": "project:000000000000"})),
            response(call("memory_save", save_arguments()), status="incomplete", reason="max_output_tokens"),
        ]
        original = profile()
    wire_client = ScriptedWire(script)
    monkeypatch.setattr(module, "create_client", ClientFactory(wire_client))
    store = FileMemoryStore(tmp_path / "memory")
    result = await ICodeExecutor(prepare_profile(original), store, RecordingLog(), RUN_ID).run(batch())
    assert result.termination == "failed" and result.failure_count == 0 and result.call_count == 1
    assert result.rejection_count == 1 and len(wire_client.requests) == 2
    assert await store.list_entries(PROJECT) == []


@pytest.mark.parametrize("wire", ["responses", "messages"])
async def test_limited_response_still_rejects_non_native_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, wire: str
) -> None:
    original = messages_profile() if wire == "messages" else profile()
    raw = (
        messages_response(messages_call("memory_forget", {}), reason="max_tokens")
        if wire == "messages"
        else response(call("memory_forget", {}), status="incomplete", reason="max_output_tokens")
    )
    wire_client = ScriptedWire([raw])
    monkeypatch.setattr(module, "create_client", ClientFactory(wire_client))
    result = await ICodeExecutor(
        prepare_profile(original), FileMemoryStore(tmp_path / "memory"), RecordingLog(), RUN_ID
    ).run(batch())
    assert result.termination == "failed" and result.stop_reason == "unexpected_tool_call"
    assert result.call_count == 0
