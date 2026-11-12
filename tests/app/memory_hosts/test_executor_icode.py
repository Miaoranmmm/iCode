# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Extraction success, strict tools, raw guard, and limits from the step-5 spec."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chrys.app.memory_hosts import executor_icode as module
from chrys.app.memory_hosts.executor_icode import ICodeExecutor, extraction_tool_schemas, prepare_profile
from chrys.foundation.tool_result_metadata import TOOL_FAILED_METADATA_KEY, TOOL_RESULT_METADATA_KEY
from chrys.kernel import Message
from chrys.service.memory.extraction import FULL_V3, FULL_V4, USER_V2, ExtractionError, MaterialProfile
from chrys.service.memory.storage.file_store import FileMemoryStore
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
    profile,
    response,
    save_arguments,
)


async def execute(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, script: list[Any], **limits: Any):
    wire = ScriptedWire(script)
    factory = ClientFactory(wire)
    monkeypatch.setattr(module, "create_client", factory)
    log = RecordingLog()
    store = FileMemoryStore(tmp_path / "memory")
    executor = ICodeExecutor(prepared(), store, log, RUN_ID, **limits)
    result = await executor.run(batch())
    return result, wire, factory, store, log


async def test_search_save_read_done_uses_real_decoder_core_and_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wire = ScriptedWire([])
    factory = ClientFactory(wire)
    monkeypatch.setattr(module, "create_client", factory)
    store, log = FileMemoryStore(tmp_path / "memory"), RecordingLog()

    async def read_saved(messages: list[Message]):
        stored = (await store.list_entries(PROJECT))[0]
        return response(call("memory_read", {"ref": stored.entry.ref}), text=None)

    wire.script = [
        response(call("memory_search", {"query": "", "scope": "all"})),
        response(call("memory_save", save_arguments())),
        read_saved,
        response(text="Done"),
    ]
    result = await ICodeExecutor(prepared(), store, log, RUN_ID).run(batch())
    assert (result.termination, result.call_count, result.failure_count, result.created_count) == ("completed", 3, 0, 1)
    assert result.unsettled_count == 0
    assert wire.closed == 1
    assert factory.client is not None and factory.client.max_iterations == 24
    assert len(log.changes) == 1
    assert (log.changes[0].written_by, log.changes[0].source) == ("extraction", RUN_ID)
    stored = (await store.list_entries(PROJECT))[0]
    assert (stored.entry.quotes[0].session, stored.entry.quotes[0].item, stored.entry.quotes[0].at) == (
        SESSION,
        "user-one",
        "2026-01-01T00:00:00Z",
    )
    names = {tool.name for tool in wire.requests[0][1]["tools"]}
    assert names == {
        "memory_search",
        "memory_read",
        "memory_save",
        "memory_withdraw",
        "memory_merge",
        "memory_note_conflict",
    }
    assert wire.requests[0][1]["max_tokens"] == 16_384
    assert wire.requests[0][1]["store"] is False
    assert wire.requests[0][1]["reasoning"] == {"effort": "high"}
    tool_results = [
        content for message in wire.requests[-1][0] for content in message.contents if content.type == "function_result"
    ]
    assert len(tool_results) == 3
    assert all(
        content.additional_properties[TOOL_RESULT_METADATA_KEY][TOOL_FAILED_METADATA_KEY] is False
        for content in tool_results
    )


@pytest.mark.parametrize("protocol", [("openai", "chat_completions"), ("deepseek-openai", "chat_completions")])
def test_other_protocols_rejected_without_client(protocol: tuple[str, str]) -> None:
    with pytest.raises(ExtractionError) as caught:
        prepare_profile(profile(provider=protocol[0], api_style=protocol[1]))
    assert caught.value.kind == "unsupported_extraction_protocol"
    assert caught.value.message == "Extraction requires Responses or Anthropic Messages."


@pytest.mark.parametrize("wire", ["responses", "messages"])
@pytest.mark.parametrize("location", ["top", "extra"])
@pytest.mark.parametrize("key", ["tools", "system", "instructions"])
def test_profile_cannot_inject_tools_or_instructions(location: str, key: str, wire: str) -> None:
    payload = {key: "private"}
    if location == "extra":
        payload = {"extra_body": payload}
    with pytest.raises(ExtractionError, match=key):
        prepare_profile(
            profile(provider="anthropic" if wire == "messages" else "deepseek-openai", chat_options=json.dumps(payload))
        )


def test_profile_cap_copy_and_aliases_are_normalized() -> None:
    original = profile(
        max_output_tokens=8000,
        chat_options=json.dumps(
            {
                "max_completion_tokens": 90000,
                "max_output_tokens": 90001,
                "store": True,
                "extra_body": {"max_tokens": 90002, "store": True},
                "reasoning": {"effort": "high"},
            }
        ),
    )
    normalized = prepare_profile(original).profile
    options = json.loads(normalized.chat_options)
    assert normalized is not original and original.max_output_tokens == 8000
    assert normalized.max_output_tokens == options["max_tokens"] == 8000
    assert options["store"] is False and options["extra_body"]["store"] is False
    assert "max_completion_tokens" not in options and "max_output_tokens" not in options
    assert "max_tokens" not in options["extra_body"] and "continuation_token" not in options
    assert options["reasoning"] == {"effort": "high"}


def test_extraction_schema_reuses_review_tools_and_requires_quote_sessions() -> None:
    schemas = extraction_tool_schemas()
    assert {schema["name"] for schema in schemas} == {
        "memory_search",
        "memory_read",
        "memory_save",
        "memory_withdraw",
        "memory_merge",
        "memory_note_conflict",
    }
    save = next(schema for schema in schemas if schema["name"] == "memory_save")
    assert save["parameters"]["properties"]["user_quotes"]["items"]["required"] == ["text", "session"]
    assert save["parameters"]["additionalProperties"] is False


@pytest.mark.parametrize(
    "status,reason,expected",
    [
        ("completed", "max_output_tokens", "completed"),
        ("incomplete", "unknown", "failed"),
        ("incomplete", None, "failed"),
        ("queued", None, "failed"),
        ("in_progress", None, "failed"),
        (None, None, "failed"),
    ],
)
async def test_raw_status_is_authoritative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: Any, reason: str | None, expected: str
) -> None:
    result, wire, _, _, _ = await execute(
        monkeypatch,
        tmp_path,
        [
            response(call("memory_search", {"query": ""})),
            response(text="Done", status=status, reason=reason),
        ],
    )
    assert result.termination == expected
    assert (result.failure_count == 0) is (expected == "completed")
    assert wire.closed == 1


@pytest.mark.parametrize("bad_id", ["", None, 7])
async def test_invalid_raw_call_id_blocks_entire_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_id: Any
) -> None:
    result, wire, _, store, log = await execute(
        monkeypatch,
        tmp_path,
        [
            response(
                call("memory_search", {"query": ""}, "valid"),
                call("memory_save", save_arguments(), bad_id),
            )
        ],
    )
    assert (result.termination, result.failure_count, result.call_count) == ("failed", 1, 0)
    assert result.stop_reason == "invalid_call_id"
    assert await store.list_entries(PROJECT) == [] and log.changes == []
    assert len(wire.requests) == 1


@pytest.mark.parametrize(
    "second_name,second_args,limited",
    [
        ("memory_save", {}, False),
        ("unknown_tool", {}, False),
        ("memory_search", {"query": "other"}, False),
        ("memory_search", {"query": ""}, False),
        ("memory_search", {"query": ""}, True),
    ],
)
async def test_any_same_response_duplicate_id_blocks_every_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, second_name: str, second_args: dict[str, object], limited: bool
) -> None:
    result, _, _, store, _ = await execute(
        monkeypatch,
        tmp_path,
        [
            response(
                call("memory_search", {"query": ""}),
                call(second_name, second_args),
                status="incomplete" if limited else "completed",
                reason="max_output_tokens" if limited else None,
            )
        ],
    )
    assert result.stop_reason == "duplicate_call_id"
    assert result.termination == "failed" and result.call_count == 0 and result.failure_count == 1
    assert result.failure_kinds == (("guard:duplicate_call_id", 1),)
    assert await store.list_entries(PROJECT) == []


async def test_cross_response_call_id_reuse_is_legal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result, _, _, _, _ = await execute(
        monkeypatch,
        tmp_path,
        [
            response(call("memory_search", {"query": ""})),
            response(call("memory_search", {"query": "small"})),
            response(text="Done"),
        ],
    )
    assert (result.termination, result.call_count, result.failure_count) == ("completed", 2, 0)


@pytest.mark.parametrize(
    "item,kind",
    [
        (call("memory_forget", {}), "unexpected_tool_call"),
        (SimpleNamespace(type="future_annotation"), "unsupported_raw_output"),
        (
            SimpleNamespace(type="tool_search_output", id="item", call_id="c", tools=[], status="completed"),
            "unexpected_tool_call",
        ),
        (
            SimpleNamespace(type="shell_call_output", id="item", call_id="c", output=[], status="completed"),
            "unexpected_tool_call",
        ),
        (
            SimpleNamespace(type="future_server_task", id="item", call_id="c", execution="server", status="completed"),
            "unexpected_tool_call",
        ),
        (
            SimpleNamespace(
                type="custom_tool_call", id="item", call_id="c", name="custom", input="data", status="completed"
            ),
            "unexpected_tool_call",
        ),
        (
            SimpleNamespace(type="apply_patch_call", id="item", call_id="c", operation={}, status="completed"),
            "unexpected_tool_call",
        ),
    ],
)
async def test_tool_boundary_and_unknown_raw_types(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, item: Any, kind: str
) -> None:
    result, _, _, _, _ = await execute(monkeypatch, tmp_path, [response(item, text="Done")])
    assert result.termination == "failed" and result.failure_count > 0
    assert result.stop_reason == kind
    if kind == "unsupported_raw_output":
        assert result.unsupported_raw_output == (("future_annotation", 1),)


@pytest.mark.parametrize("arguments", [None, [], "string", {}, {"query": 42}, {"query": "", "extra": True}])
@pytest.mark.parametrize(
    "material_profile,calls,requests",
    [(USER_V2, 3, 4), (FULL_V3, 3, 4), (FULL_V4, 2, 2)],
    ids=["user-v2", "full-v3", "full-v4"],
)
async def test_kernel_prevalidation_failure_is_sticky(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arguments: Any,
    material_profile: MaterialProfile,
    calls: int,
    requests: int,
) -> None:
    result, wire, _, _, _ = await execute(
        monkeypatch,
        tmp_path,
        [
            response(call("memory_search", {"query": ""})),
            response(call("memory_search", arguments)),
            response(call("memory_search", {"query": "fixed"})),
            response(text="Done"),
        ],
        material_profile=material_profile,
    )
    assert result.termination == "failed" and result.failure_count == 1
    assert result.failure_kinds == (("memory_search:argument_parsing", 1),)
    assert result.call_count == calls and len(wire.requests) == requests


async def test_turn_limit_prevents_extra_tail_wire_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result, wire, factory, _, _ = await execute(
        monkeypatch,
        tmp_path,
        [
            response(call("memory_search", {"query": "one"})),
            response(call("memory_search", {"query": "two"})),
        ],
        max_turns=2,
    )
    assert factory.client is not None and factory.client.max_iterations == 2
    assert (result.termination, result.call_count, result.failure_count, result.unsettled_count) == (
        "turn_limited",
        2,
        0,
        0,
    )
    assert result.tail_seen is True and len(wire.requests) == 2


@pytest.mark.parametrize("with_call", [False, True])
async def test_clean_output_limit_settles_calls_and_stops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, with_call: bool
) -> None:
    items = [call("memory_search", {"query": ""})] if with_call else []
    result, wire, _, _, _ = await execute(
        monkeypatch, tmp_path, [response(*items, status="incomplete", reason="max_output_tokens", text="partial")]
    )
    assert result.termination == "output_limited" and result.failure_count == 0
    assert result.call_count == 0 and result.unsettled_count == 0 and len(wire.requests) == 1


async def test_output_limit_does_not_dispatch_invalid_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, wire, _, _, _ = await execute(
        monkeypatch, tmp_path, [response(call("memory_save", {}), status="incomplete", reason="max_output_tokens")]
    )
    assert result.termination == "output_limited" and result.failure_count == 0
    assert result.call_count == 0 and result.unsettled_count == 0
    assert len(wire.requests) == 1


async def test_zero_calls_done_is_failure_even_with_empty_recorder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, _, _, _, _ = await execute(monkeypatch, tmp_path, [response(text="Done")])
    assert (result.termination, result.stop_reason, result.call_count, result.failure_count) == (
        "failed",
        "zero_calls",
        0,
        1,
    )
    assert result.failure_kinds == (("guard:zero_calls", 1),)


@pytest.mark.parametrize(
    "material_profile,failures,stop_reason,tail_seen",
    [
        (USER_V2, 10, "tool_error_stop", True),
        (FULL_V3, 10, "tool_error_stop", True),
        (FULL_V4, 1, "tool_failure", False),
    ],
    ids=["user-v2", "full-v3", "full-v4"],
)
async def test_kernel_exception_stop_policy_respects_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    material_profile: MaterialProfile,
    failures: int,
    stop_reason: str,
    tail_seen: bool,
) -> None:
    result, wire, _, _, _ = await execute(
        monkeypatch,
        tmp_path,
        [response(call("memory_save", {})) for _ in range(10)],
        material_profile=material_profile,
    )
    assert result.termination == "failed" and result.stop_reason == stop_reason
    assert result.failure_count == result.call_count == len(wire.requests) == failures
    assert result.tail_seen is tail_seen and len(wire.script) == 10 - failures


async def test_failure_after_save_keeps_durable_entry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result, _, _, store, log = await execute(
        monkeypatch, tmp_path, [response(call("memory_save", save_arguments())), RuntimeError("private token")]
    )
    assert result.termination == "failed" and result.created_count == 1
    assert len(await store.list_entries(PROJECT)) == len(log.changes) == 1
    assert "private" not in repr(result)
