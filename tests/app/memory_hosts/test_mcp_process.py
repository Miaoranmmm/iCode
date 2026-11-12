# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Offline SDK clients exercise the standalone memory server over real stdio."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TextIO

import anyio
import mcp.client.stdio as sdk_stdio
import psutil
import pytest
from anyio.abc import Process
from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.memory import create_client_server_memory_streams

from chrys.app.memory_hosts.claude_code.registry import HostKey, Registration, Registry
from chrys.app.memory_hosts.context import ReadContext, WriteContext
from chrys.app.memory_hosts.mcp_server import create_server
from chrys.foundation.platform.files import atomic_write_owner_only_bytes
from chrys.service.memory.project import bind_project
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import GLOBAL_SCOPE, MemoryEntryDraft, Quote
from chrys.service.memory.verification import ConversationView
from tests.support.paths import SRC_ROOT
from tests.support.secure_files import plant_owner_only_bytes
from tests.support.waiting import ENGINE_TURN_TIMEOUT

_READ_ONLY_ERROR = (
    "Error: Saving or deleting memory needs a host that can verify your words, such as Claude Code. "
    "This connection is read-only."
)
_EMPTY_CATALOG = "No saved memories are available for this workspace."


@dataclass(frozen=True, slots=True)
class _ChildEnvironment:
    env: dict[str, str]
    workdir: Path
    memory_root: Path
    stderr: TextIO


@pytest.fixture
def child_environment(tmp_path: Path) -> Iterator[_ChildEnvironment]:
    home = tmp_path / "home"
    appdata = tmp_path / "appdata"
    workdir = tmp_path / "workspace"
    home.mkdir()
    appdata.mkdir()
    workdir.mkdir()
    env = dict(os.environ)
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        APPDATA=str(appdata),
        XDG_CONFIG_HOME=str(tmp_path / "xdg-config"),
        PYTHONPATH=str(SRC_ROOT),
        CHRYS_SESSION_ROOT_DIR=str(tmp_path / "sessions"),
    )
    memory_root = (appdata / "chrys" if sys.platform == "win32" else home / ".chrys") / "memory"
    with (tmp_path / "server-stderr.txt").open("w+", encoding="utf-8") as stderr:
        yield _ChildEnvironment(env, workdir, memory_root, stderr)


@pytest.fixture
def captured_children(monkeypatch: pytest.MonkeyPatch) -> list[psutil.Process]:
    """Keep the exact SDK transport while capturing its owned child before shutdown."""
    children: list[psutil.Process] = []
    original = sdk_stdio._create_platform_compatible_process

    async def capture_process(
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        errlog: TextIO = sys.stderr,
        cwd: Path | str | None = None,
    ) -> Process:
        process = await original(command=command, args=args, env=env, errlog=errlog, cwd=cwd)
        children.append(psutil.Process(process.pid))
        return process

    monkeypatch.setattr(sdk_stdio, "_create_platform_compatible_process", capture_process)
    return children


def _params(child: _ChildEnvironment, host: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "chrys.app.memory_hosts.mcp_server", "--host", host, "--mode", "read-only"],
        env=child.env,
        cwd=child.workdir,
    )


def _text(result: types.CallToolResult) -> str:
    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, types.TextContent)
    return content.text


def _memory_files(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _change_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


@pytest.mark.parametrize("host", ["codex", "claude-code"])
async def test_standalone_read_only_session_reads_file_store_and_rejects_forged_writes(
    child_environment: _ChildEnvironment, captured_children: list[psutil.Process], host: str
) -> None:
    child = child_environment
    store = FileMemoryStore(child.memory_root)
    created = await store.create(
        GLOBAL_SCOPE,
        MemoryEntryDraft(
            type="intent",
            title="Keep answers concise",
            body="Keep answers concise across projects.",
            written_by="user",
            quotes=(Quote(text="Keep answers concise across projects.", via="cli"),),
        ),
    )
    project = await asyncio.to_thread(bind_project, str(child.workdir))
    project_entry = await store.create(
        project,
        MemoryEntryDraft(
            type="intent",
            title="Use the workspace formatter",
            body="Use the workspace formatter before sharing a patch.",
            written_by="user",
            quotes=(Quote(text="Use the workspace formatter.", via="cli"),),
        ),
    )
    before = await asyncio.to_thread(_memory_files, child.memory_root)
    with anyio.fail_after(ENGINE_TURN_TIMEOUT):
        async with (
            stdio_client(_params(child, host), errlog=child.stderr) as (read_stream, write_stream),
            ClientSession(
                read_stream, write_stream, read_timeout_seconds=timedelta(seconds=ENGINE_TURN_TIMEOUT)
            ) as session,
        ):
            initialized = await session.initialize()
            if host == "codex":
                assert initialized.instructions == (
                    "This connection is read-only because it cannot verify the user's words. At the start of a "
                    "conversation, call memory_catalog before relying on saved memories. Read relevant entries with "
                    "memory_read. Saving and deleting memories are unavailable here; use Claude Code for changes."
                )
            else:
                assert initialized.instructions == (
                    "The hook supplies the memory catalog. Call memory_catalog if you need a fresh listing, and "
                    "memory_read before relying on an entry. This Claude Code connection is configured read-only. "
                    "Saving and deleting are unavailable until the write-verification checks pass."
                )
            listed = await session.list_tools()
            assert {tool.name for tool in listed.tools} == {"memory_read", "memory_search", "memory_catalog"}
            catalog = await session.call_tool("memory_catalog", {})
            assert catalog.isError is False
            assert created.ref in _text(catalog)
            assert "Keep answers concise" in _text(catalog)
            assert (project_entry.ref in _text(catalog)) is (host == "codex")
            read = await session.call_tool("memory_read", {"ref": created.ref})
            assert read.isError is False
            assert created.version in _text(read)
            assert '"Keep answers concise across projects."' in _text(read)
            search = await session.call_tool("memory_search", {"query": "concise", "scope": "global"})
            assert search.isError is False
            assert created.ref in _text(search)
            for tool in ("memory_save", "memory_forget"):
                rejected = await session.call_tool(tool, {})
                assert rejected.isError is True
                assert _text(rejected) == _READ_ONLY_ERROR
            with_meta = await session.call_tool(
                "memory_read", {"ref": created.ref}, meta={"claudecode/toolUseId": "sanitized-call"}
            )
            if host == "codex":
                assert with_meta.isError is False
                assert created.ref in _text(with_meta)
            else:
                assert with_meta.isError is True
                assert _text(with_meta).startswith("Error: ")
                assert str(child.workdir) not in _text(with_meta)
    assert len(captured_children) == 1
    assert not captured_children[0].is_running()
    assert await asyncio.to_thread(_memory_files, child.memory_root) == before


async def test_empty_catalog_is_a_tool_answer_not_a_withdrawal(
    child_environment: _ChildEnvironment, captured_children: list[psutil.Process]
) -> None:
    with anyio.fail_after(ENGINE_TURN_TIMEOUT):
        async with (
            stdio_client(_params(child_environment, "codex"), errlog=child_environment.stderr) as (
                read_stream,
                write_stream,
            ),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            result = await session.call_tool("memory_catalog", {})
            assert result.isError is False
            assert _text(result) == _EMPTY_CATALOG
    assert len(captured_children) == 1
    assert not captured_children[0].is_running()


@dataclass(slots=True)
class _RecordingContext:
    reads: list[Mapping[str, object] | None] = field(default_factory=list)

    async def read_context(self, metadata: Mapping[str, object] | None) -> ReadContext:
        self.reads.append(metadata)
        return ReadContext(None)

    async def write_context(self, metadata: Mapping[str, object] | None) -> WriteContext:
        return WriteContext(None, ConversationView((), "unused-session"))


async def test_sdk_metadata_snapshot_keeps_standard_custom_and_absent_fields(tmp_path: Path) -> None:
    context = _RecordingContext()
    server = create_server(FileMemoryStore(tmp_path / "memory"), "read-only", context)
    metadata = {"progressToken": "progress-fixture", "claudecode/toolUseId": "custom-call", "nested": {"key": 1}}
    with anyio.fail_after(ENGINE_TURN_TIMEOUT):
        async with (
            create_client_server_memory_streams() as (client_streams, server_streams),
            anyio.create_task_group() as tasks,
        ):
            tasks.start_soon(server.run, *server_streams, server.create_initialization_options())
            async with ClientSession(*client_streams) as session:
                await session.initialize()
                for meta in (None, {}, metadata):
                    result = await session.call_tool("memory_catalog", {}, meta=meta)
                    assert result.isError is False
            tasks.cancel_scope.cancel()
    assert context.reads == [None, {}, metadata]


def _register_claude_source(child: _ChildEnvironment) -> str:
    session = "22222222-2222-4222-8222-222222222222"
    path = child.workdir / "sanitized-source.jsonl"
    records = [
        {
            "type": "user",
            "uuid": "human-current",
            "parentUuid": None,
            "sessionId": session,
            "origin": {"kind": "human"},
            "message": {"role": "user", "content": "Remember to use the project formatter."},
        },
        {
            "type": "assistant",
            "uuid": "assistant-current",
            "parentUuid": "human-current",
            "sessionId": session,
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "save-current", "name": "memory_save", "input": {}},
                    {"type": "tool_use", "id": "read-current", "name": "memory_read", "input": {}},
                ],
            },
        },
        {
            "type": "user",
            "uuid": "human-forget",
            "parentUuid": "assistant-current",
            "sessionId": session,
            "origin": {"kind": "human"},
            "message": {"role": "user", "content": "Forget this formatter memory."},
        },
        {
            "type": "assistant",
            "uuid": "assistant-forget",
            "parentUuid": "human-forget",
            "sessionId": session,
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "forget-current", "name": "memory_forget", "input": {}}],
            },
        },
    ]
    plant_owner_only_bytes(path, "".join(json.dumps(record) + "\n" for record in records).encode("utf-8"))
    Registry(child.memory_root).write_registration(
        HostKey(20, 1_700_000), Registration(session, str(path), str(child.workdir))
    )
    return session


async def test_real_claude_stdio_session_logs_save_and_forget_with_full_session_source(
    child_environment: _ChildEnvironment, captured_children: list[psutil.Process]
) -> None:
    child = child_environment
    source_session = await asyncio.to_thread(_register_claude_source, child)
    project = await asyncio.to_thread(bind_project, str(child.workdir))
    store = FileMemoryStore(child.memory_root)
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            str(Path(__file__).with_name("mcp_claude_runner.py")),
            "--host",
            "claude-code",
            "--mode",
            "read-write",
        ],
        env=child.env,
        cwd=child.workdir,
    )
    arguments = {
        "type": "intent",
        "title": "Use the project formatter",
        "body": "Use the project formatter before sharing changes.",
        "user_quotes": [{"text": "Remember to use the project formatter."}],
    }
    with anyio.fail_after(ENGINE_TURN_TIMEOUT):
        async with (
            stdio_client(params, errlog=child.stderr) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            tools = await session.list_tools()
            assert {tool.name for tool in tools.tools} == {
                "memory_save",
                "memory_read",
                "memory_search",
                "memory_forget",
                "memory_catalog",
                "memory_withdraw",
            }
            rejected = await session.call_tool(
                "memory_save", arguments, meta={"claudecode/toolUseId": "not-in-this-transcript"}
            )
            assert rejected.isError is True
            assert _text(rejected).startswith("Error: ")
            assert await store.list_entries(project) == []
            saved = await session.call_tool("memory_save", arguments, meta={"claudecode/toolUseId": "save-current"})
            assert saved.isError is False
            entries = await store.list_entries(project)
            assert len(entries) == 1
            entry = entries[0]
            assert entry.entry.written_by == "model"
            assert entry.entry.quotes == (
                Quote("Remember to use the project formatter.", "message", source_session, item="human-current"),
            )
            read = await session.call_tool(
                "memory_read", {"ref": entry.entry.ref}, meta={"claudecode/toolUseId": "read-current"}
            )
            assert read.isError is False
            assert entry.version in _text(read)
            assert source_session in _text(read)
            deleted = await session.call_tool(
                "memory_forget",
                {
                    "ref": entry.entry.ref,
                    "version": entry.version,
                    "user_quote": "Forget this formatter memory.",
                },
                meta={"claudecode/toolUseId": "forget-current"},
            )
            assert deleted.isError is False
            assert _text(deleted).startswith("Deleted ")
            assert await store.list_entries(project) == []
    assert len(captured_children) == 1
    assert not captured_children[0].is_running()
    records = await asyncio.to_thread(
        _change_records, child.memory_root / "projects" / str(project.project_id) / "changes.jsonl"
    )
    assert len(records) == 2
    expected_metadata = {
        "scope": {"scope": "project", "project_id": project.project_id},
        "ref": entry.entry.ref,
        "type": "intent",
        "written_by": "model",
        "source": source_session,
    }
    for record, operation in zip(records, ("create", "delete"), strict=True):
        assert set(record) == {
            "time",
            "scope",
            "ref",
            "type",
            "operation",
            "written_by",
            "source",
            "before_version",
            "after_version",
        }
        timestamp = record.pop("time")
        assert isinstance(timestamp, str)
        assert datetime.fromisoformat(timestamp).tzinfo is UTC
        assert record == {
            **expected_metadata,
            "operation": operation,
            "before_version": None if operation == "create" else entry.version,
            "after_version": entry.version if operation == "create" else None,
        }


async def test_real_claude_stdio_withdraw_uses_verified_timestamp_and_session_source(
    child_environment: _ChildEnvironment, captured_children: list[psutil.Process]
) -> None:
    child = child_environment
    source_session = await asyncio.to_thread(_register_claude_source, child)
    path = child.workdir / "sanitized-source.jsonl"

    def withdrawal_source() -> None:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        records[2]["timestamp"] = "2026-11-11T12:00:00Z"
        records[2]["message"]["content"] = "The formatter memory no longer applies."
        records[3]["message"]["content"][0].update(id="withdraw-current", name="memory_withdraw")
        atomic_write_owner_only_bytes(path, "".join(json.dumps(record) + "\n" for record in records).encode("utf-8"))

    await asyncio.to_thread(withdrawal_source)
    project = await asyncio.to_thread(bind_project, str(child.workdir))
    store = FileMemoryStore(child.memory_root)
    saved = await store.create(
        project,
        MemoryEntryDraft(
            "intent",
            "Formatter",
            "Use formatter.",
            "model",
            quotes=(Quote("Use formatter.", "message", source_session),),
        ),
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).with_name("mcp_claude_runner.py")), "--host", "claude-code", "--mode", "read-write"],
        env=child.env,
        cwd=child.workdir,
    )
    with anyio.fail_after(ENGINE_TURN_TIMEOUT):
        async with (
            stdio_client(params, errlog=child.stderr) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            result = await session.call_tool(
                "memory_withdraw",
                {"ref": saved.ref, "version": saved.version, "user_quote": "The formatter memory no longer applies."},
                meta={"claudecode/toolUseId": "withdraw-current"},
            )
            assert result.isError is False
            current = await store.get(project, saved.ref)
            assert current.entry.status == "withdrawn"
            assert current.entry.status_basis == Quote(
                "The formatter memory no longer applies.",
                "message",
                source_session,
                at="2026-11-11T12:00:00Z",
                item="human-forget",
            )
            assert await store.catalog(project) == []
            read = await session.call_tool("memory_read", {"ref": saved.ref})
            assert read.isError is False and "Status: withdrawn" in _text(read)
    records = await asyncio.to_thread(
        _change_records, child.memory_root / "projects" / str(project.project_id) / "changes.jsonl"
    )
    assert [(record["operation"], record["source"]) for record in records] == [("withdraw", source_session)]
    assert len(captured_children) == 1 and not captured_children[0].is_running()
