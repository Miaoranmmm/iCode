# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Interactive adapters use one startup tuning snapshot without changing v0 text."""

from __future__ import annotations

import io
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from mcp.types import TextContent

from chrys.app.memory_hosts import mcp_server
from chrys.app.memory_hosts.claude_code import hook
from chrys.app.memory_hosts.claude_code.registry import ProcessInfo, Registry
from chrys.app.memory_hosts.context import CLAUDE_INSTRUCTIONS, ReadContext, WriteContext
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import GLOBAL_SCOPE, MemoryEntryDraft, Quote
from chrys.service.memory.tuning import V0, MemoryTuning


@dataclass
class ProjectContext:
    async def read_context(self, metadata: Mapping[str, object] | None) -> ReadContext:
        return ReadContext(None)

    async def write_context(self, metadata: Mapping[str, object] | None) -> WriteContext:
        raise RuntimeError("No writes are admitted by this fixture.")


@dataclass
class Platform:
    config_dir: Path


class Tree:
    def inspect(self, pid: int) -> ProcessInfo | None:
        return None


@dataclass
class StartedServer:
    runs: int = 0

    def create_initialization_options(self) -> None:
        return None

    async def run(self, reader: object, writer: object, options: object) -> None:
        self.runs += 1


@dataclass
class Startup:
    expected_config: Path
    candidate: MemoryTuning
    calls: list[Path] = field(default_factory=list)
    seen: list[MemoryTuning] = field(default_factory=list)
    server: StartedServer = field(default_factory=StartedServer)

    def load(self, config_dir: Path) -> MemoryTuning:
        self.calls.append(config_dir)
        assert config_dir == self.expected_config
        return self.candidate

    def create(
        self,
        store: FileMemoryStore,
        mode: str,
        context: ProjectContext,
        *,
        change_log: object,
        tuning: MemoryTuning,
    ) -> StartedServer:
        self.seen.append(tuning)
        return self.server


@asynccontextmanager
async def stdio() -> AsyncIterator[tuple[object, object]]:
    yield object(), object()


def bootstrap(*, configure_stdio: bool, apply_patches: bool, setup_telemetry: bool, project_root: Path | None) -> None:
    pass


def test_hook_main_loads_exactly_one_startup_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    startup = Startup(tmp_path, replace(V0, version="v1", guidance_strength="describe"))
    monkeypatch.setattr(hook, "load_active", startup.load)
    monkeypatch.setattr(hook, "get_platform", lambda: Platform(tmp_path))
    monkeypatch.setattr(hook, "bootstrap_runtime", bootstrap)
    monkeypatch.setattr(hook.sys, "stdin", io.TextIOWrapper(io.BytesIO(b'{"hook_event_name":"unsupported"}')))
    assert hook.main(process_tree=Tree()) == 0
    assert startup.calls == [tmp_path]


async def test_mcp_run_loads_once_and_passes_the_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    startup = Startup(tmp_path, replace(V0, version="v1", catalog_budget=1000))
    monkeypatch.setattr(mcp_server, "load_active", startup.load)
    monkeypatch.setattr(mcp_server, "get_platform", lambda: Platform(tmp_path))
    monkeypatch.setattr(mcp_server, "CodexProvider", ProjectContext)
    monkeypatch.setattr(mcp_server, "create_server", startup.create)
    monkeypatch.setattr(mcp_server, "stdio_server", stdio)
    await mcp_server._run("codex", "read-only")
    assert startup.calls == [tmp_path]
    assert startup.seen == [startup.candidate]
    assert startup.server.runs == 1


async def test_mcp_catalog_and_instructions_use_the_selected_tuning(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    await store.create(
        GLOBAL_SCOPE,
        MemoryEntryDraft(
            "intent", "Existing", "A saved rule.", "model", quotes=(Quote("Use existing", "message", "s"),)
        ),
    )
    context = ProjectContext()
    candidate = replace(V0, guidance_strength="describe")
    default = mcp_server.create_server(store, "read-write", context)
    assert default.instructions == CLAUDE_INSTRUCTIONS
    selected = mcp_server.create_server(store, "read-write", context, tuning=candidate)
    assert (
        selected.instructions
        == "The hook supplies the memory catalog. Call memory_catalog if you need a fresh listing."
    )
    result = await mcp_server.MemoryTools(store, "read-write", context, tuning=candidate).call(
        "memory_catalog", {}, None
    )
    block = result.content[0]
    assert isinstance(block, TextContent)
    assert "Existing" in block.text
    assert "Read an entry with memory_read before relying on it." not in block.text
    assert await store.unreadable(GLOBAL_SCOPE) == []


class HostTree:
    def inspect(self, pid: int) -> ProcessInfo | None:
        if pid == 10:
            return ProcessInfo(10, 20, "python", 18, ())
        if pid == 20:
            return ProcessInfo(20, 0, "claude", 17, ("claude",))
        return None


async def test_hook_catalog_uses_the_supplied_startup_snapshot(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    await store.create(
        GLOBAL_SCOPE,
        MemoryEntryDraft(
            "intent", "Existing", "A saved rule.", "model", quotes=(Quote("Use existing", "message", "s"),)
        ),
    )
    output, diagnostic = io.StringIO(), io.StringIO()
    await hook.handle_hook(
        {
            "hook_event_name": "SessionStart",
            "source": "startup",
            "session_id": "synthetic-session",
            "transcript_path": str(tmp_path / "transcript.jsonl"),
        },
        tree=HostTree(),
        registry=Registry(tmp_path / "memory"),
        store=store,
        environ={"CLAUDE_PROJECT_DIR": str(tmp_path)},
        output=output,
        diagnostic=diagnostic,
        pid=10,
        tuning=replace(V0, guidance_strength="describe"),
    )
    assert "Existing" in output.getvalue()
    assert "Read an entry with memory_read before relying on it." not in output.getvalue()
