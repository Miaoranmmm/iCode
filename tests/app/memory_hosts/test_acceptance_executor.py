# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Raw protocol priorities and run accounting from final acceptance review."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from anthropic.types.beta import (
    BetaContainerUploadBlock,
    BetaMCPToolResultBlock,
    BetaMCPToolUseBlock,
    BetaServerToolUseBlock,
    BetaWebSearchToolResultBlock,
)

from chrys.app.memory_hosts import executor_icode as module
from chrys.app.memory_hosts.executor_icode import ICodeExecutor, prepare_profile
from chrys.app.memory_hosts.extract import run_extraction
from chrys.kernel import ChatResponse
from chrys.service.memory.extraction import ExtractionError
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import MemoryOperationError
from tests.app.memory_hosts.executor_fakes import (
    RUN_ID,
    ClientFactory,
    RecordingLog,
    ScriptedWire,
    batch,
    messages_profile,
    messages_response,
    profile,
    response,
)
from tests.app.memory_hosts.test_extract import PROFILE, Reader
from tests.app.memory_hosts.test_extract import workspace as workspace


async def test_report_explains_requests_invisible_to_guard_and_usage(workspace) -> None:
    options, sources = workspace
    report = await run_extraction(replace(options, dry_run=True), PROFILE, readers={"claude-code": Reader(sources)})
    assert report.request_accounting_note == (
        "Adapter-internal thinking-binding resends and SDK retries (default 2) are not counted by the guard or usage totals."
    )


@pytest.mark.parametrize(
    "wire,hosted_kind",
    [
        ("messages", "server_tool_use"),
        ("messages", "mcp_tool_use"),
        ("messages", "mcp_tool_result"),
        ("messages", "web_search_tool_result"),
        ("responses", "web_search_call"),
    ],
)
@pytest.mark.parametrize("with_unknown", [False, True])
async def test_hosted_output_precedes_consistency_and_retains_unknown_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, wire: str, hosted_kind: str, with_unknown: bool
) -> None:
    if wire == "messages":
        if hosted_kind == "server_tool_use":
            hosted = BetaServerToolUseBlock.model_construct(type=hosted_kind, id="hosted", name="synthetic", input={})
        elif hosted_kind == "mcp_tool_use":
            hosted = BetaMCPToolUseBlock.model_construct(
                type=hosted_kind, id="hosted", name="synthetic", input={}, server_name="synthetic"
            )
        elif hosted_kind == "mcp_tool_result":
            hosted = BetaMCPToolResultBlock.model_construct(
                type=hosted_kind, tool_use_id="hosted", content=[], is_error=False
            )
        else:
            hosted = BetaWebSearchToolResultBlock.model_construct(type=hosted_kind, tool_use_id="hosted", content=[])
        unknown = BetaContainerUploadBlock(type="container_upload", file_id="synthetic")
        raw = messages_response(*([hosted, unknown, unknown] if with_unknown else [hosted]), reason="tool_use")
        original, unknown_kind = messages_profile(), "container_upload"
    else:
        hosted = SimpleNamespace(type="web_search_call", id="hosted", status="completed", action={})
        unknown = SimpleNamespace(type="future_block")
        raw = response(*([hosted, unknown, unknown] if with_unknown else [hosted]))
        original, unknown_kind = profile(), "future_block"
    turns: list[module.RawTurn] = []

    def observe(decoded: ChatResponse) -> None:
        turns.append(module._read_messages(decoded) if wire == "messages" else module._read_responses(decoded))

    wire_client = ScriptedWire([raw], decoded_hook=observe)
    monkeypatch.setattr(module, "create_client", ClientFactory(wire_client))
    result = await ICodeExecutor(prepare_profile(original), FileMemoryStore(tmp_path), RecordingLog(), RUN_ID).run(
        batch()
    )
    assert result.stop_reason == "unexpected_tool_call"
    assert result.call_count == 0
    assert result.unsupported_raw_output == (((unknown_kind, 2),) if with_unknown else ())
    assert turns[0].hosted == ((hosted_kind, 1),)
    assert turns[0].unsupported == result.unsupported_raw_output
    decoded = wire_client.script  # The real decoder and middleware consumed the raw payload.
    assert decoded == []


@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize(
    "mode,budget,valid", [("enabled", 2048, True), ("disabled", None, True), ("enabled", None, False)]
)
def test_thinking_templates_are_resolved_before_budget_validation(
    monkeypatch: pytest.MonkeyPatch, extra: bool, mode: str, budget: int | None, valid: bool
) -> None:
    monkeypatch.setenv("SYNTHETIC_THINKING_MODE", mode)
    thinking: dict[str, object] = {"type": "{{SYNTHETIC_THINKING_MODE}}"}
    if budget is not None:
        thinking["budget_tokens"] = budget
    options: dict[str, object] = {"thinking": thinking}
    if extra:
        options = {"extra_body": options}
    original = replace(messages_profile(), chat_options=json.dumps(options))
    if valid:
        prepared = prepare_profile(original)
        resolved = json.loads(prepared.profile.chat_options)
        assert (resolved["extra_body"] if extra else resolved)["thinking"]["type"] == mode
    else:
        with pytest.raises(ExtractionError) as caught:
            prepare_profile(original)
        assert caught.value.kind == "memory_invalid_input"
        assert "budget_tokens" in caught.value.message


@pytest.mark.parametrize("extra", [False, True])
def test_enabled_thinking_requires_a_budget_even_without_templates(extra: bool) -> None:
    options = {"thinking": {"type": "enabled"}}
    with pytest.raises(ExtractionError) as caught:
        prepare_profile(
            replace(messages_profile(), chat_options=json.dumps({"extra_body": options} if extra else options))
        )
    assert caught.value.kind == "memory_invalid_input"


def test_executor_rejects_unprepared_profiles_with_a_typed_input_error(tmp_path: Path) -> None:
    with pytest.raises(MemoryOperationError) as caught:
        ICodeExecutor(profile(), FileMemoryStore(tmp_path), RecordingLog(), RUN_ID)  # type: ignore[arg-type]
    assert caught.value.kind == "memory_invalid_input"
