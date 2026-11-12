# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Corrupt policy files fail closed for extraction and degrade safely for hosts."""

from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path
from typing import TextIO

import pytest

from chrys.app.memory_hosts import executor_icode, mcp_server
from chrys.app.memory_hosts.claude_code import hook
from chrys.app.memory_hosts.claude_code.registry import ProcessTree, Registry
from chrys.app.memory_hosts.extract import run_extraction
from chrys.foundation.platform import get_platform
from chrys.service.memory.storage.store import MemoryStore
from chrys.service.memory.tuning import V0, MemoryTuning
from tests.app.memory_hosts.executor_fakes import ClientFactory, ScriptedWire
from tests.app.memory_hosts.test_extract import PROFILE, Reader
from tests.app.memory_hosts.test_extract import workspace as workspace
from tests.app.memory_hosts.test_tuning_adapters import Platform, ProjectContext, Startup, Tree, bootstrap, stdio
from tests.support.secure_files import plant_owner_only_bytes


def corrupt(config: Path) -> None:
    versions = config / "memory-tuning" / "versions"
    versions.mkdir(parents=True)
    plant_owner_only_bytes(versions / "1.json", b"{invalid")
    plant_owner_only_bytes(versions.parent / "active", b"1\n")


def test_corrupt_tuning_hook_warns_once_and_continues_with_v0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corrupt(tmp_path)
    seen: list[MemoryTuning] = []

    async def handle(
        payload: Mapping[str, object],
        *,
        tree: ProcessTree,
        registry: Registry,
        store: MemoryStore,
        environ: Mapping[str, str],
        output: TextIO,
        diagnostic: TextIO,
        pid: int | None = None,
        tuning: MemoryTuning = V0,
    ) -> None:
        seen.append(tuning)

    monkeypatch.setattr(hook, "bootstrap_runtime", bootstrap)
    monkeypatch.setattr(hook, "get_platform", lambda: Platform(tmp_path))
    monkeypatch.setattr(hook, "handle_hook", handle)
    monkeypatch.setattr(hook.sys, "stdin", io.TextIOWrapper(io.BytesIO(b'{"hook_event_name":"unsupported"}')))
    assert hook.main(process_tree=Tree()) == 0
    assert seen == [V0]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Memory tuning unavailable; using v0.\n"


async def test_corrupt_tuning_mcp_warns_once_on_stderr_and_serves_with_v0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corrupt(tmp_path)
    startup = Startup(tmp_path, V0)
    monkeypatch.setattr(mcp_server, "get_platform", lambda: Platform(tmp_path))
    monkeypatch.setattr(mcp_server, "CodexProvider", ProjectContext)
    monkeypatch.setattr(mcp_server, "create_server", startup.create)
    monkeypatch.setattr(mcp_server, "stdio_server", stdio)
    await mcp_server._run("codex", "read-only")
    assert startup.seen == [V0] and startup.server.runs == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Memory tuning unavailable; using v0.\n"


async def test_corrupt_tuning_extraction_fails_before_building_a_client(
    workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, sources = workspace
    corrupt(get_platform().config_dir)
    factory = ClientFactory(ScriptedWire([]))
    monkeypatch.setattr(executor_icode, "create_client", factory)
    report = await run_extraction(options, PROFILE, readers={"claude-code": Reader(sources)})
    assert (report.status, report.exit_code) == ("invalid_tuning", 1)
    assert report.diagnostics == [
        {"kind": "invalid_tuning", "message": "The requested memory tuning is unavailable or invalid."}
    ]
    assert factory.profiles == []
    assert not options.memory_dir.exists()
