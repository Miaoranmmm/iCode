# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Host context contracts use real locator files and invented transcript records."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from chrys.app.memory_hosts import codex
from chrys.app.memory_hosts.claude_code import provider as claude_provider
from chrys.app.memory_hosts.claude_code.provider import ClaudeCodeProvider
from chrys.app.memory_hosts.claude_code.registry import (
    SOURCE_UNAVAILABLE,
    WORKSPACE_UNAVAILABLE,
    HostKey,
    ProcessInfo,
    Registration,
    Registry,
    SystemProcessTree,
)
from chrys.app.memory_hosts.codex import CodexProvider
from chrys.app.memory_hosts.context import READ_ONLY_REASON, ReadContext, WriteContext
from chrys.service.memory.project import bind_project
from chrys.service.memory.storage.store import MemoryOperationError, MemoryScope
from tests.support.secure_files import plant_owner_only_bytes

_SESSION = "22222222-2222-4222-8222-222222222222"
_META: Mapping[str, object] = {"claudecode/toolUseId": "current-call"}


@dataclass(slots=True)
class _ProcessTree:
    records: dict[int, ProcessInfo | None]
    inspected: list[int] = field(default_factory=list)

    def inspect(self, pid: int) -> ProcessInfo | None:
        self.inspected.append(pid)
        return self.records.get(pid)


@dataclass(frozen=True, slots=True)
class _Source:
    registry: Registry
    tree: _ProcessTree
    host: HostKey
    registration: Registration
    workspace: Path


def _write_transcript(path: Path, *, session: str = _SESSION, call: str = "current-call", copied: bool = False) -> None:
    assistant: dict[str, object] = {
        "type": "assistant",
        "uuid": "assistant-current",
        "parentUuid": "user-current",
        "sessionId": session,
        "cwd": "/invented/drifting/shell/directory",
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": call, "name": "memory_save", "input": {}}],
        },
    }
    if copied:
        assistant["forkedFrom"] = {"sessionId": "parent-session", "messageUuid": "original-assistant"}
    records = [
        {
            "type": "user",
            "uuid": "user-current",
            "parentUuid": None,
            "sessionId": session,
            "origin": {"kind": "human"},
            "timestamp": "2026-11-06T18:30:00+08:00",
            "message": {"role": "user", "content": "Always use the project formatter."},
        },
        assistant,
    ]
    plant_owner_only_bytes(path, "".join(json.dumps(record) + "\n" for record in records).encode("utf-8"))


@pytest.fixture
def source(tmp_path: Path) -> _Source:
    workspace = tmp_path / "session-workspace"
    workspace.mkdir()
    path = tmp_path / "sanitized-transcript.jsonl"
    _write_transcript(path)
    registration = Registration(_SESSION, str(path), str(workspace))
    registry = Registry(tmp_path / "memory")
    host = HostKey(20, 1_700_000)
    registry.write_registration(host, registration)
    tree = _ProcessTree(
        {
            os.getpid(): ProcessInfo(os.getpid(), 10, "python", 1, ("python",)),
            10: ProcessInfo(10, 20, "uv", 2, ("uv", "run")),
            20: ProcessInfo(20, 30, "/installed/claude", host.started_us, ("claude",)),
            30: ProcessInfo(30, 0, "/outer/claude", 3, ("claude",)),
        }
    )
    return _Source(registry, tree, host, registration, workspace)


def _assert_source_error(error: pytest.ExceptionInfo[MemoryOperationError], expected: str = SOURCE_UNAVAILABLE) -> None:
    assert error.value.kind == "memory_store_unavailable"
    assert error.value.message == expected


async def test_claude_project_and_view_use_registered_workspace_despite_shell_and_environment_drift(
    source: _Source, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other-workspace"
    other.mkdir()
    monkeypatch.chdir(other)
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(other))
    provider = ClaudeCodeProvider(source.registry, source.tree)
    expected = await asyncio.to_thread(bind_project, str(source.workspace))
    reading = await provider.read_context(_META)
    writing = await provider.write_context(_META)
    assert reading.project == writing.project == expected
    assert writing.view.current_session == _SESSION
    assert len(writing.view.messages) == 1
    message = writing.view.messages[0]
    assert message.session == _SESSION
    assert message.item == "user-current"
    assert message.position == 1
    assert message.at == "2026-11-06T10:30:00Z"
    assert message.fragments == ("Always use the project formatter.",)


async def test_claude_absent_metadata_allows_internal_read_but_empty_metadata_is_invalid(source: _Source) -> None:
    provider = ClaudeCodeProvider(source.registry, source.tree)
    assert (await provider.read_context(None)).project is not None
    with pytest.raises(MemoryOperationError) as failure:
        await provider.read_context({})
    _assert_source_error(failure)


@pytest.mark.parametrize("metadata", [None, {}, {"claudecode/toolUseId": ""}, {"claudecode/toolUseId": 1}])
async def test_claude_write_requires_this_request_tool_id(
    source: _Source, metadata: Mapping[str, object] | None
) -> None:
    provider = ClaudeCodeProvider(source.registry, source.tree)
    with pytest.raises(MemoryOperationError) as failure:
        await provider.write_context(metadata)
    _assert_source_error(failure)


async def test_nearest_claude_without_registration_does_not_use_outer_registration(source: _Source) -> None:
    source.registry.write_registration(HostKey(30, 3), source.registration)
    source.registry.registration_path(source.host).unlink()
    provider = ClaudeCodeProvider(source.registry, source.tree)
    assert provider.host == source.host
    assert (await provider.read_context(None)).project is None
    for writing in (False, True):
        with pytest.raises(MemoryOperationError) as failure:
            if writing:
                await provider.write_context(_META)
            else:
                await provider.read_context(_META)
        _assert_source_error(failure)


class _OverwritingProvider(ClaudeCodeProvider):
    """Replace the on-disk locator only after execution has moved to the worker."""

    def __init__(self, source: _Source, replacement: Registration) -> None:
        super().__init__(source.registry, source.tree)
        self.key = source.host
        self.replacement = replacement

    def _read(self, registration: Registration, tool_id: str | None) -> ReadContext:
        self.registry.write_registration(self.key, self.replacement)
        return super()._read(registration, tool_id)

    def _write(self, registration: Registration, tool_id: str) -> WriteContext:
        self.registry.write_registration(self.key, self.replacement)
        return super()._write(registration, tool_id)


@pytest.mark.parametrize("writing", [False, True])
async def test_registration_is_frozen_before_first_thread_suspension(
    source: _Source, tmp_path: Path, writing: bool
) -> None:
    other = tmp_path / "replacement-workspace"
    other.mkdir()
    replacement_path = tmp_path / "replacement-transcript.jsonl"
    _write_transcript(replacement_path, session="replacement-full-session", call="replacement-call")
    replacement = Registration("replacement-full-session", str(replacement_path), str(other))
    provider = _OverwritingProvider(source, replacement)
    expected = await asyncio.to_thread(bind_project, str(source.workspace))
    if writing:
        result = await provider.write_context(_META)
        assert result.view.current_session == _SESSION
        assert result.project == expected
    else:
        assert (await provider.read_context(_META)).project == expected
    assert source.registry.read_registration(source.host) == replacement
    with pytest.raises(MemoryOperationError) as failure:
        await provider.read_context(_META)
    _assert_source_error(failure)


@pytest.mark.parametrize("invalid", ["missing_id", "parent_session", "forked_copy"])
@pytest.mark.parametrize("writing", [False, True])
async def test_bad_call_anchor_never_falls_back_to_startup_workspace(
    source: _Source, tmp_path: Path, invalid: str, writing: bool
) -> None:
    path = tmp_path / "bad-call-source.jsonl"
    _write_transcript(
        path,
        session="parent-session" if invalid == "parent_session" else _SESSION,
        call="another-call" if invalid == "missing_id" else "current-call",
        copied=invalid == "forked_copy",
    )
    source.registry.write_registration(source.host, replace(source.registration, transcript_path=str(path)))
    provider = ClaudeCodeProvider(source.registry, source.tree)
    with pytest.raises(MemoryOperationError) as failure:
        if writing:
            await provider.write_context(_META)
        else:
            await provider.read_context(_META)
    _assert_source_error(failure)


@pytest.mark.parametrize("workspace", ["none", "relative", "deleted"])
@pytest.mark.parametrize("writing", [False, True])
async def test_missing_or_invalid_session_workspace_has_the_fixed_workspace_diagnostic(
    source: _Source, tmp_path: Path, workspace: str, writing: bool
) -> None:
    project_dir = None if workspace == "none" else "relative" if workspace == "relative" else str(tmp_path / "deleted")
    source.registry.write_registration(source.host, replace(source.registration, project_dir=project_dir))
    provider = ClaudeCodeProvider(source.registry, source.tree)
    with pytest.raises(MemoryOperationError) as failure:
        if writing:
            await provider.write_context(_META)
        else:
            await provider.read_context(None)
    _assert_source_error(failure, WORKSPACE_UNAVAILABLE)


@pytest.mark.parametrize(
    "argv", [None, ("claude", "--project-config-root", "private"), ("claude", "--project-config-root=private")]
)
async def test_unsupported_project_override_or_unknown_argv_cannot_bind_a_project(
    source: _Source, argv: tuple[str, ...] | None
) -> None:
    process = source.tree.records[source.host.pid]
    assert process is not None
    source.tree.records[source.host.pid] = replace(process, argv=argv)
    provider = ClaudeCodeProvider(source.registry, source.tree)
    for writing in (False, True):
        with pytest.raises(MemoryOperationError) as failure:
            if writing:
                await provider.write_context(_META)
            else:
                await provider.read_context(_META)
        _assert_source_error(failure, WORKSPACE_UNAVAILABLE)
        assert "private" not in str(failure.value)


async def test_claude_keeps_startup_host_after_parent_chain_changes(source: _Source) -> None:
    source.registry.write_registration(HostKey(30, 3), replace(source.registration, project_dir=None))
    provider = ClaudeCodeProvider(source.registry, source.tree)
    source.tree.inspected.clear()
    source.tree.records[10] = ProcessInfo(10, 30, "uv", 2, ("uv",))
    assert (await provider.read_context(_META)).project is not None
    assert provider.host == source.host
    assert os.getpid() not in source.tree.inspected
    assert 10 not in source.tree.inspected
    assert 30 not in source.tree.inspected


@pytest.mark.parametrize("replacement", [None, ProcessInfo(20, 30, "claude", 9, ("claude",))])
async def test_dead_or_reused_startup_host_cannot_borrow_outer_registration(
    source: _Source, replacement: ProcessInfo | None
) -> None:
    provider = ClaudeCodeProvider(source.registry, source.tree)
    source.registry.write_registration(HostKey(30, 3), source.registration)
    source.tree.records[source.host.pid] = replacement
    with pytest.raises(MemoryOperationError) as failure:
        await provider.read_context(_META)
    _assert_source_error(failure)


@pytest.mark.parametrize("failure_kind", ["git_error", "missing_git"])
async def test_claude_git_binding_failure_retains_global_and_verifiable_view(
    source: _Source, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure_kind: str
) -> None:
    calls: list[str] = []

    def unavailable(directory: str) -> MemoryScope:
        calls.append(directory)
        if failure_kind == "missing_git":
            raise FileNotFoundError("private path")
        raise MemoryOperationError("memory_store_unavailable", "private path")

    monkeypatch.setattr(claude_provider, "bind_project", unavailable)
    provider = ClaudeCodeProvider(source.registry, source.tree)
    assert (await provider.read_context(_META)).project is None
    writing = await provider.write_context(_META)
    assert writing.project is None
    assert writing.view.current_session == _SESSION
    assert calls == [str(source.workspace), str(source.workspace)]
    assert capsys.readouterr().err == ("Memory: project binding is unavailable; only global memory is available.\n" * 2)


async def test_only_view_construction_invalid_input_is_mapped_to_source_failure(
    source: _Source, monkeypatch: pytest.MonkeyPatch
) -> None:
    def bad_view(registration: Registration, tool_use_id: str) -> None:
        raise MemoryOperationError("memory_invalid_input", "private source detail")

    monkeypatch.setattr(claude_provider, "build_conversation", bad_view)
    provider = ClaudeCodeProvider(source.registry, source.tree)
    with pytest.raises(MemoryOperationError) as failure:
        await provider.write_context(_META)
    _assert_source_error(failure)


async def test_codex_uses_only_startup_cwd_and_ignores_claude_metadata_and_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    startup = tmp_path / "startup"
    later = tmp_path / "later"
    startup.mkdir()
    later.mkdir()
    monkeypatch.chdir(startup)

    def forbidden_process(self: SystemProcessTree, pid: int) -> ProcessInfo | None:
        raise AssertionError("Codex must not inspect Claude ancestors")

    def forbidden_registration(self: Registry, key: HostKey) -> Registration | None:
        raise AssertionError("Codex must not read Claude registrations")

    monkeypatch.setattr(SystemProcessTree, "inspect", forbidden_process)
    monkeypatch.setattr(Registry, "read_registration", forbidden_registration)
    provider = await asyncio.to_thread(CodexProvider)
    expected = await asyncio.to_thread(bind_project, str(startup))
    monkeypatch.chdir(later)
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(later))
    for metadata in (None, {}, _META, {"cwd": str(later)}):
        assert (await provider.read_context(metadata)).project == expected
    assert provider.cwd == str(startup.resolve())
    with pytest.raises(MemoryOperationError) as failure:
        await provider.write_context(_META)
    _assert_source_error(failure, READ_ONLY_REASON)


@pytest.mark.parametrize("startup", ["root", "home"])
async def test_codex_root_and_home_are_unbound_without_attempting_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], startup: str
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    def forbidden_bind(directory: str) -> MemoryScope:
        raise AssertionError("Root and home must not become projects")

    monkeypatch.setattr(codex, "bind_project", forbidden_bind)
    provider = await asyncio.to_thread(CodexProvider, tmp_path.anchor if startup == "root" else str(home))
    assert (await provider.read_context(_META)).project is None
    assert capsys.readouterr().err == (
        "Memory: startup directory has no project binding; only global memory is available.\n"
    )


async def test_codex_git_failure_is_a_global_only_context_with_fixed_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing_git(directory: str) -> MemoryScope:
        raise FileNotFoundError("private path")

    monkeypatch.setattr(codex, "bind_project", missing_git)
    provider = await asyncio.to_thread(CodexProvider, str(tmp_path))
    assert (await provider.read_context(None)).project is None
    assert capsys.readouterr().err == ("Memory: project binding is unavailable; only global memory is available.\n")
