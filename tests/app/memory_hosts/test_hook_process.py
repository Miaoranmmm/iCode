# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Offline hook entrypoint stdin/stdout contracts with deterministically reaped children."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import psutil
import pytest

from chrys.app.memory_hosts.claude_code.registry import HostKey, Registry
from chrys.foundation.platform import get_platform
from chrys.foundation.platform.process import windows_hidden_subprocess_kwargs
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import GLOBAL_SCOPE, MemoryEntryDraft, Quote
from tests.support.secure_files import plant_owner_only_bytes
from tests.support.waiting import ENGINE_TURN_TIMEOUT

REPO = Path(__file__).resolve().parents[3]
KEY = HostKey(777, 17)


@dataclass
class ChildResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass
class Environment:
    env: dict[str, str]
    workspace: Path
    root: Path

    def run(
        self, payload: bytes, *, fixture_host: bool, no_host: bool = False, bootstrap_report: Path | None = None
    ) -> ChildResult:
        module = "tests.app.memory_hosts.hook_runner" if fixture_host else "chrys.app.memory_hosts.claude_code.hook"
        command = [sys.executable, "-m", module]
        if no_host:
            command.append("--no-host")
        if bootstrap_report is not None:
            command.extend(("--bootstrap-failure", str(bootstrap_report)))
        with subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
            cwd=self.workspace,
            **windows_hidden_subprocess_kwargs(),
        ) as child:
            process = psutil.Process(child.pid)
            try:
                stdout, stderr = child.communicate(payload, timeout=ENGINE_TURN_TIMEOUT)
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=ENGINE_TURN_TIMEOUT)
        assert not process.is_running()
        return ChildResult(child.returncode, stdout.decode("utf-8"), stderr.decode("utf-8"))

    def payload(self, event: str, *, source: str = "startup") -> bytes:
        return json.dumps(
            {
                "hook_event_name": event,
                "session_id": "complete-session-id",
                "transcript_path": str(self.workspace / "transcript.jsonl"),
                "cwd": str(self.workspace / "shell-moved"),
                "source": source,
                "prompt": "Use SQLite",
                "prompt_id": "prompt-id",
                "last_assistant_message": "Done",
                "stop_hook_active": False,
            }
        ).encode("utf-8")


@pytest.fixture
def environment(tmp_path: Path) -> Environment:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir()
    workspace.mkdir()
    env = dict(os.environ)
    env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "APPDATA": str(home),
            "XDG_CONFIG_HOME": str(home / "config"),
            "CLAUDE_PROJECT_DIR": str(workspace),
            "PYTHONPATH": os.pathsep.join((str(REPO), str(REPO / "src"))),
        }
    )
    root = home / ("chrys" if get_platform().is_windows else ".chrys") / "memory"
    return Environment(env, workspace, root)


@pytest.mark.parametrize("event", ["SessionStart", "UserPromptSubmit", "Stop"])
def test_actual_hook_module_accepts_real_json(environment: Environment, event: str) -> None:
    result = environment.run(environment.payload(event), fixture_host=False)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == "" or "memory source is unavailable" in result.stderr


@pytest.mark.parametrize("event", ["SessionStart", "UserPromptSubmit", "Stop"])
def test_real_main_without_claude_ancestor_does_not_register(environment: Environment, event: str) -> None:
    result = environment.run(environment.payload(event), fixture_host=True, no_host=True)
    assert result.returncode == 0 and result.stdout == ""
    assert "memory source is unavailable" in result.stderr
    assert not (environment.root / "hosts").exists()


@pytest.mark.parametrize("payload", [b"not-json", b"[]", b'{"hook_event_name":"UserPromptSubmit"}'])
def test_actual_hook_bad_input_is_nonblocking(environment: Environment, payload: bytes) -> None:
    result = environment.run(payload, fixture_host=False)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr
    assert not (environment.root / "hosts").exists()


def test_actual_hook_oversized_stdin_drains_to_eof(environment: Environment) -> None:
    result = environment.run(b"{" + b" " * (4 * 1024 * 1024), fixture_host=False)
    assert result == ChildResult(0, "", "Memory hook input exceeded the 2 MiB limit and was discarded.\n")
    assert not (environment.root / "hosts").exists()


def test_real_main_session_start_ups_stop_and_redelivery(environment: Environment) -> None:
    store = FileMemoryStore(environment.root)
    asyncio.run(
        store.create(
            GLOBAL_SCOPE,
            MemoryEntryDraft(
                "intent",
                "Use SQLite",
                "Use SQLite for durable local state.",
                "model",
                quotes=(Quote("Use SQLite", "message", "earlier"),),
            ),
        )
    )
    registry = Registry(environment.root)
    start = environment.run(environment.payload("SessionStart"), fixture_host=True)
    assert start.returncode == 0 and start.stderr == ""
    assert start.stdout.startswith("<memory_catalog>\n") and "Use SQLite" in start.stdout
    assert registry.read_registration(KEY) is None
    first = environment.run(environment.payload("UserPromptSubmit"), fixture_host=True)
    assert first == ChildResult(0, "", "")
    registration = registry.read_registration(KEY)
    assert registration is not None
    assert registration.session_id == "complete-session-id"
    assert registration.project_dir == str(environment.workspace)
    before_registration = registry.registration_path(KEY).read_bytes()
    before_delivery = registry.delivery_path(KEY, registration.session_id).read_bytes()
    rows = [
        {
            "uuid": "user",
            "parentUuid": None,
            "type": "user",
            "promptId": "prompt-id",
            "origin": {"kind": "human"},
            "message": {"role": "user", "content": "Use SQLite"},
        },
        {
            "uuid": "assistant",
            "parentUuid": "user",
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Done"}]},
        },
        {"uuid": "summary", "parentUuid": "assistant", "type": "system", "subtype": "stop_hook_summary"},
        {"type": "last-prompt"},
    ]
    plant_owner_only_bytes(
        environment.workspace / "transcript.jsonl", ("\n".join(json.dumps(row) for row in rows) + "\n").encode("utf-8")
    )
    stop = environment.run(environment.payload("Stop"), fixture_host=True)
    assert stop == ChildResult(0, "", "")
    assert registry.needs_redeliver(KEY, registration.session_id) is True
    assert registry.registration_path(KEY).read_bytes() == before_registration
    assert registry.delivery_path(KEY, registration.session_id).read_bytes() == before_delivery
    after = environment.run(environment.payload("UserPromptSubmit"), fixture_host=True)
    assert after == ChildResult(0, start.stdout, "")
    assert registry.needs_redeliver(KEY, registration.session_id) is False


@pytest.mark.parametrize("oversized", [False, True], ids=["valid-1MiB-json", "oversized-4MiB"])
def test_real_main_drains_stdin_before_bootstrap_failure(environment: Environment, oversized: bool) -> None:
    payload = (
        b"{" + b" " * (4 * 1024 * 1024)
        if oversized
        else json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": "x" * (1024 * 1024)}).encode("utf-8")
    )
    report = environment.workspace / "stdin-report.json"
    result = environment.run(payload, fixture_host=True, bootstrap_report=report)
    assert result == ChildResult(0, "", "Memory hook is unavailable; continuing without an update.\n")
    assert json.loads(report.read_text(encoding="utf-8")) == {"bytes_read": len(payload), "eof": True}
    assert not (environment.root / "hosts").exists()
