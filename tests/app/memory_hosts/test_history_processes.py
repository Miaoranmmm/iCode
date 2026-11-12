# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Typed process facts enforce closed sources without examining real host processes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from chrys.app.memory_hosts.claude_code.holder import NativeClaudeHolder, proc_start_seconds
from chrys.app.memory_hosts.codex_history import CodexProcessCheck, ProcessExecutable
from tests.app.memory_hosts.history_fixtures import SESSION
from tests.support.secure_files import plant_owner_only_bytes

START = int(datetime(2026, 11, 1, 2, 3, 4, tzinfo=UTC).timestamp())


@dataclass
class Processes:
    seconds: int | None = START
    fails: bool = False
    calls: int = 0

    def started_seconds(self, pid: int) -> int | None:
        self.calls += 1
        if self.fails:
            raise OSError("Synthetic access refusal")
        return self.seconds


@dataclass
class Table:
    items: tuple[ProcessExecutable, ...] = ()
    user: str | int | None = 501
    fails: bool = False

    def current_user(self) -> str | int | None:
        return self.user

    def snapshot(self) -> tuple[ProcessExecutable, ...]:
        if self.fails:
            raise OSError("Synthetic enumeration refusal")
        return self.items


def registration(tmp_path: Path, **fields: object) -> None:
    plant_owner_only_bytes(
        tmp_path / "123.json",
        json.dumps(
            {
                "sessionId": SESSION,
                "pid": 123,
                "procStart": "Sun Nov  1 02:03:04 2026",
                "pidDomain": "darwin",
                "status": "idle",
                "startedAt": START * 1000 + 17_000,
                **fields,
            }
        ).encode("utf-8"),
    )


def test_native_idle_holder_is_active_and_registration_time_does_not_replace_procstart(tmp_path: Path) -> None:
    registration(tmp_path)
    assert NativeClaudeHolder(tmp_path, processes=Processes(), pid_domain="darwin").check(SESSION) == "active"
    assert proc_start_seconds("Sun Nov  1 02:03:04 2026") == START


@pytest.mark.parametrize("seconds", [None, START + 1, START - 1])
def test_dead_or_reused_pid_does_not_hold_session(tmp_path: Path, seconds: int | None) -> None:
    registration(tmp_path)
    assert NativeClaudeHolder(tmp_path, processes=Processes(seconds), pid_domain="darwin").check(SESSION) == "inactive"


@pytest.mark.parametrize(
    "fields",
    [
        {"procStart": None},
        {"procStart": "invalid"},
        {"pidDomain": None},
        {"pidDomain": "linux"},
        {"pid": "123"},
        {"pid": True},
    ],
)
def test_unprovable_domain_or_start_identity_is_unknown_before_pid_lookup(
    tmp_path: Path, fields: dict[str, object]
) -> None:
    registration(tmp_path, **fields)
    processes = Processes(None)
    assert NativeClaudeHolder(tmp_path, processes=processes, pid_domain="darwin").check(SESSION) == "unknown"
    assert processes.calls == 0


def test_holder_access_error_is_unknown_and_never_deletes_registration(tmp_path: Path) -> None:
    registration(tmp_path)
    assert (
        NativeClaudeHolder(tmp_path, processes=Processes(fails=True), pid_domain="darwin").check(SESSION) == "unknown"
    )
    assert (tmp_path / "123.json").is_file()


def test_native_missing_session_identity_and_malformed_registration_fail_closed(tmp_path: Path) -> None:
    plant_owner_only_bytes(tmp_path / "broken.json", b'{"pid":123}\n')
    assert NativeClaudeHolder(tmp_path, processes=Processes(None), pid_domain="darwin").check(SESSION) == "unknown"


def test_process_start_token_uses_utc_even_under_non_utc_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TZ", "Pacific/Honolulu")
    monkeypatch.setenv("LC_ALL", "zh_CN.UTF-8")
    assert proc_start_seconds("Sun Nov  1 02:03:04 2026") == START


@pytest.mark.parametrize(
    "exe",
    [
        "/usr/local/bin/codex",
        "/Applications/Codex.app/Contents/MacOS/Codex",
        "/Applications/Codex.app/Contents/Frameworks/Helper",
        r"C:\Tools\codex.exe",
    ],
)
def test_codex_current_user_real_executables_include_helpers_and_cli(exe: str) -> None:
    assert CodexProcessCheck(Table((ProcessExecutable(501, exe),))).check() == "active"


def test_other_user_hosts_and_unrelated_executables_do_not_block() -> None:
    assert (
        CodexProcessCheck(Table((ProcessExecutable(502, "/bin/codex"), ProcessExecutable(501, "/bin/python")))).check()
        == "inactive"
    )


@pytest.mark.parametrize(
    "table",
    [
        Table(user=None),
        Table(fails=True),
        Table((ProcessExecutable(None, None),)),
        Table((ProcessExecutable(501, None),)),
    ],
)
def test_unknown_process_identity_or_enumeration_never_passes(table: Table) -> None:
    assert CodexProcessCheck(table).check() == "unknown"
