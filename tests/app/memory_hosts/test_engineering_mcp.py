# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Step-7 interactive memory contracts, independent of the MCP implementation."""

from __future__ import annotations

from pathlib import Path

import pytest

from chrys.app.memory_hosts.mcp_server import MemoryTools
from chrys.app.memory_hosts.schemas import decode_request, tool_definitions
from chrys.service.memory.operations import SaveRequest
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import MemoryOperationError
from tests.app.memory_hosts.test_mcp import Context, text


def engineering(**fields: object) -> dict[str, object]:
    return {
        "type": "engineering",
        "title": "Use a transaction",
        "body": "The update needs a transaction.",
        "evidence": [{"kind": "reference", "ref": "synthetic design"}],
        **fields,
    }


def test_read_write_schema_offers_both_types_four_evidence_kinds_and_context() -> None:
    tools = {tool.name: tool for tool in tool_definitions(writable=True)}
    assert set(tools) == {
        "memory_save",
        "memory_read",
        "memory_search",
        "memory_forget",
        "memory_catalog",
        "memory_withdraw",
    }
    schema = tools["memory_save"].inputSchema
    properties = schema["properties"]
    assert properties["type"]["enum"] == ["intent", "engineering"]
    assert {"evidence", "context_quote", "context_candidate", "context_session"} <= properties.keys()
    variants = properties["evidence"]["items"]["oneOf"]
    assert {item["properties"]["kind"]["const"] for item in variants} == {
        "tool_result",
        "file",
        "reference",
        "user_report",
    }
    assert all(item["additionalProperties"] is False for item in variants)
    assert properties["evidence"]["minItems"] == 1 and properties["evidence"]["maxItems"] == 6
    assert properties["context_quote"]["maxLength"] == 600
    assert "memory_withdraw" in tools["memory_forget"].description
    assert "whether the files it was based on have changed since it was saved" in tools["memory_read"].description
    assert tools["memory_save"].description == (
        "Save what later conversations will need, so the user does not repeat it and you do not rediscover it. "
        'Two cases: the user stated a lasting preference, requirement or decision (type "intent", quote their words verbatim); '
        'you established a reusable engineering finding or failure cause from tool results (type "engineering", cite the evidence). '
        "Save when the user asks you to remember something. Do not save task progress, to-dos, one-off instructions, "
        "what the loaded reference files already say, or facts cheap to read from the code. Never turn a user requirement "
        "into an engineering entry or your own analysis into an intent entry. To change an entry, pass its ref and version."
    )


def test_read_only_tools_do_not_register_any_writer() -> None:
    tools = {tool.name: tool for tool in tool_definitions(writable=False)}
    assert set(tools) == {"memory_read", "memory_search", "memory_catalog"}
    assert "whether the files it was based on have changed since it was saved" in tools["memory_read"].description


@pytest.mark.parametrize(
    "evidence",
    [
        {"kind": "tool_result", "quote": "observed", "session": "full", "candidate": "token"},
        {"kind": "user_report", "quote": "observed", "session": "full", "candidate": "token"},
        {"kind": "file", "path": "src/example.py"},
        {"kind": "reference", "ref": "synthetic design", "note": "section 1"},
    ],
)
def test_decoder_accepts_only_model_supplied_evidence_fields(evidence: dict[str, object]) -> None:
    request = decode_request(
        "memory_save",
        engineering(
            evidence=[evidence], context_quote="proposal", context_candidate="choice", context_session="old-session"
        ),
    )
    assert isinstance(request, SaveRequest)
    assert request.type == "engineering" and request.evidence is not None
    assert request.evidence[0].kind == evidence["kind"]
    assert (request.context_quote, request.context_candidate, request.context_session) == (
        "proposal",
        "choice",
        "old-session",
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"kind": "tool_result", "quote": "result", "tool": "shell"},
        {"kind": "tool_result", "quote": "result", "args": "command"},
        {"kind": "tool_result", "quote": "result", "at": "now"},
        {"kind": "file", "path": "a", "sha256": "forged"},
        {"kind": "file", "path": "a", "project": "forged"},
        {"kind": "reference", "ref": "a", "session": "forged"},
        {"kind": "user_report", "quote": "x", "item": "forged"},
        {"kind": "unknown", "quote": "x"},
    ],
)
def test_decoder_rejects_forged_evidence_provenance(extra: dict[str, object]) -> None:
    with pytest.raises(MemoryOperationError) as error:
        decode_request("memory_save", engineering(evidence=[extra]))
    assert error.value.kind == "memory_invalid_input"


@pytest.mark.parametrize(
    ("evidence", "problem"),
    [
        (
            [{"kind": "reference", "ref": "a"}, {"kind": "tool_result", "quote": "x" * 601}],
            "evidence[1].quote is 601 characters; the limit is 600",
        ),
        ([{"kind": "user_report", "quote": "x" * 501}], "evidence[0].quote is 501 characters; the limit is 500"),
        ([{"kind": "file"}], "evidence[0] is missing path"),
        ([{"kind": "unknown", "quote": "x"}], "evidence[0].kind is not an allowed evidence kind"),
        ([], "evidence must be a list of 1 to 6 items"),
    ],
)
def test_decoder_names_the_failing_evidence_field(evidence: list[dict[str, object]], problem: str) -> None:
    with pytest.raises(MemoryOperationError) as error:
        decode_request("memory_save", engineering(evidence=evidence))
    assert error.value.message == f"Invalid memory tool arguments. {problem}. Follow the tool's schema."


@pytest.mark.parametrize(
    "changes",
    [
        {"context_quote": "x" * 601},
        {"context_quote": ""},
        {"context_quote": "proposal", "context_session": ""},
        {"context_session": None},
        {"context_candidate": None},
        {"context_session": "orphan"},
        {"context_candidate": "orphan"},
        {"user_quotes": [{"text": "user words"}]},
        {"evidence": []},
        {"evidence": None},
        {"evidence": [{"kind": "tool_result", "quote": "x" * 601}]},
    ],
)
def test_strict_engineering_and_context_arguments(changes: dict[str, object]) -> None:
    with pytest.raises(MemoryOperationError):
        decode_request("memory_save", engineering(**changes))


async def test_reference_engineering_save_is_visible_to_read_only_clients(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    context = Context()
    saved = await MemoryTools(store, "read-write", context).call("memory_save", engineering(), None)
    assert saved.isError is False
    entry = (await store.list_entries(context.project))[0]
    assert "Evidence: reference synthetic design" in text(saved)
    reader = MemoryTools(store, "read-only", context)
    read = await reader.call("memory_read", {"ref": entry.entry.ref}, None)
    assert read.isError is False
    assert "reference synthetic design" in text(read) and "Basis: none" in text(read)
    forged = await reader.call("memory_save", engineering(), None)
    assert forged.isError is True


async def test_engineering_tool_file_user_report_context_and_basis_round_trip(tmp_path: Path) -> None:
    import asyncio
    from dataclasses import replace

    from chrys.service.memory.verification import ConversationView, SourceMessage

    project_root = tmp_path / "project"
    await asyncio.to_thread(project_root.mkdir)
    source_file = project_root / "runner.py"
    await asyncio.to_thread(source_file.write_text, "fixed_port = 8765\n", encoding="utf-8")
    session = "synthetic-engineering-session"
    view = ConversationView(
        (
            SourceMessage(session, 1, ("The runner binds port 8765.",), item="user"),
            SourceMessage(session, 2, ("Run these tests serially.",), kind="assistant", item="proposal"),
            SourceMessage(
                session,
                3,
                ("OSError: Address already in use",),
                kind="tool_result",
                item="result",
                attributes=(("tool", "shell"), ("args", '{"command":"pytest"}')),
            ),
        ),
        session,
    )
    context = Context(view=view, project=replace(Context().project, root=str(project_root)))
    store = FileMemoryStore(tmp_path / "memory")
    tools = MemoryTools(store, "read-write", context)
    result = await tools.call(
        "memory_save",
        engineering(
            evidence=[
                {"kind": "tool_result", "quote": "Address already in use"},
                {"kind": "file", "path": "runner.py"},
                {"kind": "user_report", "quote": "The runner binds port 8765."},
            ],
            context_quote="Run these tests serially.",
            context_session=session,
        ),
        {},
    )
    assert result.isError is False
    entry = (await store.list_entries(context.project))[0]
    assert entry.entry.context is not None and entry.entry.context.item == "proposal"
    assert "Evidence: tool_result shell" in text(result) and "file runner.py (hashed)" in text(result)
    reader = MemoryTools(store, "read-only", context)
    before = text(await reader.call("memory_read", {"ref": entry.entry.ref}, None))
    assert "Basis: unchanged" in before and "not re-run" in before
    await asyncio.to_thread(source_file.write_text, "port = allocate_port()\n", encoding="utf-8")
    after = text(await reader.call("memory_read", {"ref": entry.entry.ref}, None))
    assert "Basis: changed" in after
    assert (await store.get(context.project, entry.entry.ref)).version == entry.version


async def test_context_session_retains_an_old_context_while_new_user_words_support_the_update(tmp_path: Path) -> None:
    from dataclasses import replace

    from chrys.service.memory.verification import ConversationView, SourceMessage

    original_view = ConversationView(
        (
            SourceMessage("old-session", 1, ("Use SQLite.",), item="u"),
            SourceMessage("old-session", 2, ("Use a local database.",), kind="assistant", item="a"),
        ),
        "old-session",
    )
    store = FileMemoryStore(tmp_path / "memory")
    context = Context(view=original_view)
    tools = MemoryTools(store, "read-write", context)
    first = await tools.call(
        "memory_save",
        {
            "type": "intent",
            "title": "Use SQLite",
            "body": "Use SQLite.",
            "user_quotes": [{"text": "Use SQLite."}],
            "context_quote": "Use a local database.",
        },
        None,
    )
    assert first.isError is False
    stored = (await store.list_entries(context.project))[0]
    new_view = ConversationView(
        (SourceMessage("new-session", 1, ("Enable WAL for SQLite.",), item="new"),), "new-session"
    )
    updated = await MemoryTools(store, "read-write", replace(context, view=new_view)).call(
        "memory_save",
        {
            "type": "intent",
            "title": "Use SQLite with WAL",
            "body": "Enable WAL.",
            "ref": stored.entry.ref,
            "version": stored.version,
            "user_quotes": [{"text": "Enable WAL for SQLite."}],
            "context_quote": "Use a local database.",
            "context_session": "old-session",
            "context_candidate": "outside-view-candidate",
        },
        None,
    )
    assert updated.isError is False
    assert (await store.get(context.project, stored.entry.ref)).entry.context == stored.entry.context


def test_user_v2_extraction_keeps_the_existing_intent_only_schema() -> None:
    from chrys.app.memory_hosts.executor_icode import extraction_tool_schemas
    from chrys.service.memory.extraction import USER_V2

    save = next(tool for tool in extraction_tool_schemas(USER_V2) if tool["name"] == "memory_save")
    assert save["parameters"]["properties"]["type"]["enum"] == ["intent"]
    assert set(save["parameters"]["properties"]) == {
        "type",
        "title",
        "body",
        "applies_when",
        "scope",
        "user_quotes",
        "ref",
        "version",
    }
    assert save["parameters"]["properties"]["user_quotes"]["items"]["required"] == ["text", "session"]
