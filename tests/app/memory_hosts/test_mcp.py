# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The shared tool boundary follows the public contract, independent of a host."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from mcp.types import CallToolResult, TextContent

from chrys.app.memory_hosts.context import (
    CLAUDE_INSTRUCTIONS,
    CLAUDE_READ_ONLY_INSTRUCTIONS,
    EMPTY_CATALOG,
    READ_ONLY_REASON,
    ReadContext,
    WriteContext,
)
from chrys.app.memory_hosts.mcp_server import MemoryTools, create_server
from chrys.app.memory_hosts.schemas import decode_request, tool_definitions
from chrys.service.memory.operations import SaveRequest
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import (
    GLOBAL_SCOPE,
    MemoryEntryDraft,
    MemoryOperationError,
    MemoryScope,
    Quote,
    StoredMemory,
)
from chrys.service.memory.verification import ConversationView, SourceMessage

PROJECT = MemoryScope("project", "0123456789abcdef")
VIEW = ConversationView(
    (SourceMessage("full-session-id", 3, ('Use SQLite.\nKeep "quotes".', "forget it"), item="u1"),), "full-session-id"
)


@dataclass
class Context:
    view: ConversationView = VIEW
    project: MemoryScope | None = PROJECT
    error: Exception | None = None
    reads: list[Mapping[str, object] | None] = field(default_factory=list)
    writes: list[Mapping[str, object] | None] = field(default_factory=list)

    async def read_context(self, metadata: Mapping[str, object] | None) -> ReadContext:
        self.reads.append(metadata)
        if self.error is not None:
            raise self.error
        return ReadContext(self.project)

    async def write_context(self, metadata: Mapping[str, object] | None) -> WriteContext:
        self.writes.append(metadata)
        if self.error is not None:
            raise self.error
        return WriteContext(self.project, self.view)


def text(result: CallToolResult) -> str:
    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def arguments() -> dict[str, object]:
    return {
        "type": "intent",
        "title": "Use SQLite",
        "body": "Use SQLite locally.",
        "user_quotes": [{"text": 'Use SQLite.\nKeep "quotes".'}],
    }


@pytest.fixture
def store(tmp_path: Path) -> FileMemoryStore:
    return FileMemoryStore(tmp_path / "memory")


async def test_save_read_search_catalog_forget_roundtrip(store: FileMemoryStore) -> None:
    context = Context()
    tools = MemoryTools(store, "read-write", context)
    meta = {"claudecode/toolUseId": "call-1"}
    assert text(await tools.call("memory_catalog", {}, None)) == EMPTY_CATALOG
    saved = await tools.call("memory_save", arguments(), meta)
    assert saved.isError is False
    entries = await store.list_entries(PROJECT)
    assert len(entries) == 1
    stored = entries[0]
    assert stored.entry.quotes == (Quote('Use SQLite.\nKeep "quotes".', "message", "full-session-id", item="u1"),)
    read = await tools.call("memory_read", {"ref": stored.entry.ref}, meta)
    assert read.isError is False
    assert r'"Use SQLite.\nKeep \"quotes\"."' in text(read)
    catalog = await tools.call("memory_catalog", {}, {})
    assert stored.entry.ref in text(catalog)
    search = await tools.call("memory_search", {"query": "SQLite"}, None)
    assert stored.entry.ref in text(search)
    deleted = await tools.call(
        "memory_forget", {"ref": stored.entry.ref, "version": stored.version, "user_quote": "forget it"}, meta
    )
    assert deleted.isError is False
    assert text(deleted).startswith("Deleted ")
    assert await store.list_entries(PROJECT) == []
    assert context.reads == [None, meta, {}, None]
    assert context.writes == [meta, meta]


@pytest.mark.parametrize("tool", ["memory_save", "memory_forget"])
async def test_read_only_forgery_never_resolves_source_or_writes(store: FileMemoryStore, tool: str) -> None:
    context = Context(error=RuntimeError("private /secret/path"))
    result = await MemoryTools(store, "read-only", context).call(tool, None, {})
    assert result.isError is True
    assert text(result) == "Error: " + READ_ONLY_REASON
    assert context.reads == context.writes == []
    assert not store._root.exists()


@pytest.mark.parametrize(
    ("name", "bad", "problem"),
    [
        ("memory_catalog", {"cwd": "/tmp"}, "arguments has fields the schema does not allow"),
        ("memory_catalog", None, "arguments must be an object"),
        ("memory_read", {"ref": None}, "ref must be a string"),
        (
            "memory_read",
            {"ref": "global:123456789abc", "item": "fake"},
            "arguments has fields the schema does not allow",
        ),
        ("memory_search", {"query": "", "scope": None}, "scope must be a string"),
        ("memory_search", {"query": "", "cursor": None}, "cursor must be a string"),
        ("memory_search", {"query": 1}, "query must be a string"),
        (
            "memory_forget",
            {"ref": "r", "version": "v", "user_quote": "x" * 501},
            "user_quote is 501 characters; the limit is 500",
        ),
        ("memory_save", {**arguments(), "user_quotes": []}, "user_quotes must be a list of 1 to 3 items"),
        ("memory_save", {**arguments(), "user_quotes": None}, "user_quotes must be a list of 1 to 3 items"),
        (
            "memory_save",
            {**arguments(), "user_quotes": [{"text": "x", "at": "now"}]},
            "user_quotes[0] has fields the schema does not allow",
        ),
        (
            "memory_save",
            {**arguments(), "user_quotes": [{"text": "x", "session": None}]},
            "user_quotes[0].session must be a string",
        ),
        (
            "memory_save",
            {**arguments(), "user_quotes": [{"text": "x" * 1501}]},
            "user_quotes[0].text is 1501 characters; the limit is 1500",
        ),
        ("memory_save", {**arguments(), "title": "two\nlines"}, "title must be one line"),
        ("memory_save", {**arguments(), "title": "x" * 81}, "title is 81 characters; the limit is 80"),
        ("memory_save", {**arguments(), "body": "x" * 1501}, "body is 1501 characters; the limit is 1500"),
        ("memory_save", {**arguments(), "applies_when": "x" * 201}, "applies_when is 201 characters; the limit is 200"),
        ("memory_save", {**arguments(), "scope": "all"}, "scope must be project or global"),
        ("memory_save", {**arguments(), "type": "engineering"}, "an engineering entry takes evidence, not user_quotes"),
        ("memory_save", {**arguments(), "ref": "global:123456789abc"}, "ref and version must be given together"),
        ("memory_save", {**arguments(), "version": "v"}, "ref and version must be given together"),
        # An unknown field is named generically: argument names come from the model.
        ("memory_save", {**arguments(), "written_by": "user"}, "arguments has fields the schema does not allow"),
    ],
)
async def test_strict_decode_rejects_before_source(
    store: FileMemoryStore, name: str, bad: object, problem: str
) -> None:
    context = Context()
    result = await MemoryTools(store, "read-write", context).call(name, bad, None)
    assert result.isError is True
    assert text(result) == f"Error: Invalid memory tool arguments. {problem}. Follow the tool's schema."
    assert context.reads == context.writes == []


def test_omitted_quotes_are_distinct_from_empty_and_claim_paths_are_exact() -> None:
    decoded = decode_request("memory_save", {"type": "intent", "title": "x", "body": "y"})
    assert isinstance(decoded, SaveRequest)
    assert decoded.user_quotes is None
    decoded = decode_request(
        "memory_save", {**arguments(), "user_quotes": [{"text": "x", "candidate": "opaque", "session": "full"}]}
    )
    assert isinstance(decoded, SaveRequest)
    assert decoded.user_quotes is not None
    claim = decoded.user_quotes[0]
    assert (claim.field, claim.candidate_field, claim.text, claim.candidate, claim.session) == (
        "user_quotes[0]",
        "user_quotes[0].candidate",
        "x",
        "opaque",
        "full",
    )


async def test_new_missing_quotes_and_over_500_are_writer_input_errors(store: FileMemoryStore) -> None:
    context = Context()
    tools = MemoryTools(store, "read-write", context)
    missing = await tools.call("memory_save", {"type": "intent", "title": "x", "body": "y"}, None)
    assert text(missing) == "Error: A new or substantively changed intent entry needs user_quotes."
    long = await tools.call("memory_save", {**arguments(), "user_quotes": [{"text": "x" * 501}]}, None)
    assert "over 500" in text(long)
    assert "source is unavailable" not in text(long)
    assert len(context.writes) == 2


async def test_stored_cli_quote_1500_is_retained_as_decoded_value(store: FileMemoryStore) -> None:
    quote = Quote('line 1\n"' + "x" * 1492, "cli", "original-full-id")
    assert len(quote.text) == 1500
    created = await store.create(PROJECT, MemoryEntryDraft("intent", "Original", "Body", "user", quotes=(quote,)))
    args = {
        "type": "intent",
        "title": "Renamed",
        "body": "Body",
        "ref": created.ref,
        "version": created.version,
        "user_quotes": [{"text": quote.text, "session": quote.session}],
    }
    result = await MemoryTools(store, "read-write", Context()).call("memory_save", args, None)
    assert result.isError is False
    assert (await store.get(PROJECT, created.ref)).entry.quotes == (quote,)


async def test_source_conversion_is_narrow_and_private_unexpected_errors_are_fixed(store: FileMemoryStore) -> None:
    invalid = Context(error=MemoryOperationError("memory_invalid_input", "bad view /private/path"))
    result = await MemoryTools(store, "read-write", invalid).call("memory_save", arguments(), None)
    assert result.isError is True
    assert text(result) == (
        "Error: Claude Code's memory source is unavailable. "
        "Submit a new prompt or start a new session before trying again."
    )
    assert "/private/path" not in text(result)
    for name, args in (("memory_read", {"ref": "x"}), ("memory_save", arguments())):
        result = await MemoryTools(store, "read-write", Context(error=RuntimeError("private path"))).call(
            name, args, {}
        )
        assert text(result) == "Error: Memory operation failed. Try again later."
        assert result.isError is True
    unknown = await MemoryTools(store, "read-write", Context()).call("unknown", {}, None)
    assert text(unknown) == "Error: Unknown memory tool."
    assert unknown.isError is True


async def test_quote_failure_ambiguity_expiry_and_version_conflict_keep_own_text(store: FileMemoryStore) -> None:
    tools = MemoryTools(store, "read-write", Context())
    missing = await tools.call("memory_save", {**arguments(), "user_quotes": [{"text": "not said"}]}, {})
    assert "not found in any user message" in text(missing)
    ambiguous_view = ConversationView(
        (SourceMessage("s", 1, ("same",), item="a"), SourceMessage("s", 2, ("same",), item="b")), "s"
    )
    ambiguous = await MemoryTools(store, "read-write", Context(view=ambiguous_view)).call(
        "memory_save", {**arguments(), "user_quotes": [{"text": "same"}]}, {}
    )
    assert ambiguous.isError is True
    assert "matches 2 user messages" in text(ambiguous)
    assert "user_quotes[0].candidate" in text(ambiguous)
    expired = await tools.call(
        "memory_save", {**arguments(), "user_quotes": [{"text": "not said", "candidate": "old"}]}, {}
    )
    assert "candidate no longer points" in text(expired)
    await tools.call("memory_save", arguments(), {})
    entry = (await store.list_entries(PROJECT))[0]
    conflict = await tools.call("memory_save", {**arguments(), "ref": entry.entry.ref, "version": "stale"}, {})
    assert "changed since it was read" in text(conflict)
    assert (await store.get(PROJECT, entry.entry.ref)).version == entry.version


class CursorStore(FileMemoryStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.received: tuple[MemoryScope, str, str | None, bool] | None = None

    async def search(
        self, scope: MemoryScope, query: str, cursor: str | None = None, *, with_global: bool = False
    ) -> tuple[list[StoredMemory], str | None]:
        self.received = (scope, query, cursor, with_global)
        return await self.list_entries(scope), 'next\n"opaque"'


async def test_cursor_is_opaque_and_empty_query_is_allowed(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "memory")
    await store.create(
        GLOBAL_SCOPE, MemoryEntryDraft("intent", "Title", "Body", "user", quotes=(Quote("words", "cli"),))
    )
    result = await MemoryTools(store, "read-only", Context(project=None)).call(
        "memory_search", {"query": "", "cursor": 'in\n"opaque"'}, None
    )
    assert result.isError is False
    assert store.received == (GLOBAL_SCOPE, "", 'in\n"opaque"', True)
    assert r"next\n\"opaque\"" in text(result)


def test_schemas_are_closed_and_instructions_match_claude_modes(store: FileMemoryStore) -> None:
    writable = tool_definitions(writable=True)
    assert [tool.name for tool in writable] == [
        "memory_save",
        "memory_read",
        "memory_search",
        "memory_forget",
        "memory_catalog",
        "memory_withdraw",
    ]
    assert [tool.name for tool in tool_definitions(writable=False)] == [
        "memory_read",
        "memory_search",
        "memory_catalog",
    ]
    save = writable[0].inputSchema
    assert save["additionalProperties"] is False
    assert save["properties"]["user_quotes"]["items"]["additionalProperties"] is False
    assert save["properties"]["user_quotes"]["items"]["properties"]["text"]["maxLength"] == 1500
    assert save["properties"]["user_quotes"]["items"]["properties"]["session"]["description"] == (
        "Copy the full session from memory_read to retain a stored quote on update."
    )
    assert save["dependentRequired"] == {
        "ref": ["version"],
        "version": ["ref"],
        "context_candidate": ["context_quote"],
        "context_session": ["context_quote"],
    }
    assert save["required"] == ["type", "title", "body"]
    assert create_server(store, "read-write", Context()).instructions == CLAUDE_INSTRUCTIONS
    assert create_server(store, "read-only", Context()).instructions == CLAUDE_READ_ONLY_INSTRUCTIONS


@pytest.mark.parametrize(
    ("tool_name", "description"),
    [
        (
            "memory_save",
            'Save what later conversations will need, so the user does not repeat it and you do not rediscover it. Two cases: the user stated a lasting preference, requirement or decision (type "intent", quote their words verbatim); you established a reusable engineering finding or failure cause from tool results (type "engineering", cite the evidence). Save when the user asks you to remember something. Do not save task progress, to-dos, one-off instructions, what the loaded reference files already say, or facts cheap to read from the code. Never turn a user requirement into an engineering entry or your own analysis into an intent entry. To change an entry, pass its ref and version.',
        ),
        (
            "memory_read",
            "Read one memory entry in full: its text, the user's words or the evidence behind it, and whether the files it was based on have changed since it was saved. Read an entry before relying on it.",
        ),
        (
            "memory_search",
            "Find entries by words in their title, applies_when, text, quotes or evidence, including entries absent from the catalog. An empty query lists entries newest first.",
        ),
        (
            "memory_forget",
            "Delete a memory entry only when the user asks for it to be permanently removed. Quote the user's request verbatim in user_quote. If the user says an entry no longer applies, use memory_withdraw. If evidence suggests an engineering entry is outdated, update it with memory_save instead of deleting it.",
        ),
        (
            "memory_catalog",
            "List saved memories for this project and global scope. Read an entry before relying on it.",
        ),
    ],
)
def test_exact_tool_descriptions_from_spec(tool_name: str, description: str) -> None:
    tool = next(tool for tool in tool_definitions(writable=True) if tool.name == tool_name)
    assert tool.description == description


def test_schema_rejects_newlines_null_sources_and_missing_version() -> None:
    from jsonschema import Draft202012Validator

    schema = tool_definitions(writable=True)[0].inputSchema
    validator = Draft202012Validator(schema)
    assert validator.is_valid(arguments())
    for changed in (
        {"title": "title\n"},
        {"title": "title\r"},
        {"applies_when": None},
        {"ref": "global:123456789abc"},
        {"version": "v"},
        {"user_quotes": [{"text": "x", "session": None}]},
        {"user_quotes": [{"text": "x", "item": "fake"}]},
        {"cwd": "/tmp"},
    ):
        assert not validator.is_valid({**arguments(), **changed})


class FailedReadStore(FileMemoryStore):
    def __init__(self, root: Path, error: MemoryOperationError) -> None:
        super().__init__(root)
        self.error = error

    async def get(self, scope: MemoryScope, ref: str) -> StoredMemory:
        raise self.error


@pytest.mark.parametrize(
    "kind",
    ["memory_not_found", "memory_version_conflict", "memory_full", "memory_entry_invalid", "memory_store_unavailable"],
)
async def test_store_business_errors_retain_their_rendering(tmp_path: Path, kind: str) -> None:
    from chrys.service.memory.storage.store import MemoryErrorKind

    kinds: dict[str, MemoryErrorKind] = {
        name: name
        for name in (
            "memory_not_found",
            "memory_version_conflict",
            "memory_full",
            "memory_entry_invalid",
            "memory_store_unavailable",
        )
    }
    store = FailedReadStore(tmp_path / "memory", MemoryOperationError(kinds[kind], "Fixed store diagnostic."))
    result = await MemoryTools(store, "read-only", Context()).call("memory_read", {"ref": "global:123456789abc"}, {})
    assert result.isError is True
    assert text(result) == "Error: Fixed store diagnostic."


def test_entrypoint_bootstrap_failure_is_a_fixed_stderr_diagnostic(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from chrys.app.memory_hosts import mcp_server
    from chrys.orchestration.startup import RuntimeBootstrap

    def failed_bootstrap(
        *, configure_stdio: bool, apply_patches: bool, setup_telemetry: bool, project_root: Path | None
    ) -> RuntimeBootstrap:
        assert (configure_stdio, apply_patches, setup_telemetry, project_root) == (True, False, False, None)
        raise RuntimeError("private /absolute/config/path")

    monkeypatch.setattr(mcp_server, "bootstrap_runtime", failed_bootstrap)
    assert mcp_server.main(["--host", "codex", "--mode", "read-only"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Memory MCP server could not start.\n"
