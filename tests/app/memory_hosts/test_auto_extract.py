# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Automatic extraction admission, isolation and failure contracts without a model."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import BinaryIO

import pytest

from chrys.app.memory_hosts import extract
from chrys.app.memory_hosts.claude_code import auto_extract, hook
from chrys.foundation.util.lock import FileLock
from chrys.service.memory.tuning import V0
from tests.app.memory_hosts.test_extract import PROFILE, Reader
from tests.app.memory_hosts.test_extract import workspace as workspace
from tests.app.memory_hosts.test_hook import KEY, Harness
from tests.app.memory_hosts.test_hook import harness as harness
from tests.support.secure_files import plant_owner_only_bytes
from tests.support.waiting import ENGINE_TURN_TIMEOUT


@dataclass
class Clock:
    now: float = 100_000.0

    def time(self) -> float:
        return self.now


@dataclass
class Launcher:
    calls: list[list[str]] = field(default_factory=list)
    logs: list[BinaryIO] = field(default_factory=list)
    fail: bool = False

    def __call__(
        self,
        args: list[str],
        *,
        stdin: int,
        stdout: BinaryIO,
        stderr: BinaryIO,
        env: dict[str, str],
        start_new_session: bool | None = None,
        creationflags: int = 0,
    ) -> None:
        self.calls.append(args)
        self.logs.append(stdout)
        assert stdin == subprocess.DEVNULL
        assert stdout is stderr
        assert env["PYTHONUTF8"] == "1" and env["PYTHONIOENCODING"] == "utf-8"
        if sys.platform == "win32":
            assert creationflags == subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            assert start_new_session is None
        else:
            assert start_new_session is True and creationflags == 0
        if self.fail:
            raise OSError("Private launch details")


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch) -> Launcher:
    launcher = Launcher()
    shadow = ModuleType("subprocess")
    shadow.__dict__.update(vars(subprocess))
    shadow.Popen = launcher
    monkeypatch.setattr(auto_extract, "subprocess", shadow)
    return launcher


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    shadow = ModuleType("time")
    shadow.time = clock.time
    monkeypatch.setattr(auto_extract, "time", shadow)
    return clock


def invoke(harness: Harness, event: str = "UserPromptSubmit", profile: str | None = "X") -> str:
    harness.output.seek(0)
    harness.output.truncate()
    asyncio.run(
        hook.handle_hook(
            harness.event(event),
            tree=harness.tree,
            registry=harness.registry,
            store=harness.store,
            environ=harness.environ,
            output=harness.output,
            diagnostic=harness.diagnostic,
            pid=10,
            auto_extract_profile=profile,
        )
    )
    return harness.output.getvalue()


def state(harness: Harness) -> dict[str, float]:
    return json.loads((harness.registry.root / "auto-extract" / "state.json").read_text(encoding="utf-8"))


def test_opt_in_preserves_catalog_and_claims_only_after_delivery(
    harness: Harness,
    launcher: Launcher,
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness.seed()
    baseline = invoke(harness, profile=None)
    assert baseline.startswith("<memory_catalog>")
    assert launcher.calls == []
    assert not (harness.registry.root / "auto-extract").exists()
    harness.registry.mark_redeliver(KEY, "full-session-id")
    original = launcher.__call__

    def launch(
        args: list[str],
        *,
        stdin: int,
        stdout: BinaryIO,
        stderr: BinaryIO,
        env: dict[str, str],
        start_new_session: bool | None = None,
        creationflags: int = 0,
    ) -> None:
        assert harness.registry.read_registration(KEY) is not None
        assert harness.registry.read_delivery(KEY, "full-session-id").last_text == baseline.rstrip("\n")
        assert not harness.registry.needs_redeliver(KEY, "full-session-id")
        original(
            args,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            env=env,
            start_new_session=start_new_session,
            creationflags=creationflags,
        )

    monkeypatch.setattr(auto_extract.subprocess, "Popen", launch)
    assert invoke(harness) == baseline
    assert len(launcher.calls) == 1
    assert launcher.calls[0] == [
        sys.executable,
        "-m",
        "chrys.app.memory_hosts.extract",
        "-C",
        harness.environ["CLAUDE_PROJECT_DIR"],
        "--claude-source",
        str(harness.transcript.parent),
        "--model-profile",
        "X",
        "--batch-cap",
        "8",
        "--auto-extract-lock",
    ]
    assert "--codex-source" not in launcher.calls[0]
    assert state(harness) == {os.path.normcase(str(harness.transcript.parent.resolve())): clock.now}
    assert all(log.closed for log in launcher.logs)
    assert harness.diagnostic.getvalue() == ""
    if sys.platform != "win32":
        root = harness.registry.root / "auto-extract"
        assert root.stat().st_mode & 0o777 == 0o700
        assert (root / "logs").stat().st_mode & 0o777 == 0o700
        assert all(path.stat().st_mode & 0o077 == 0 for path in root.rglob("*") if path.is_file())


def test_cooldown_is_four_hours_per_transcript_directory(harness: Harness, launcher: Launcher, clock: Clock) -> None:
    invoke(harness)
    clock.now += 4 * 60 * 60 - 1
    invoke(harness)
    assert len(launcher.calls) == 1
    clock.now += 1
    invoke(harness)
    assert len(launcher.calls) == 2
    harness.transcript = harness.transcript.parent / "other-worktree" / "session.jsonl"
    invoke(harness)
    assert len(launcher.calls) == 3 and len(state(harness)) == 2
    assert harness.diagnostic.getvalue() == ""


@pytest.mark.parametrize("event", ["SessionStart", "Stop"])
def test_other_events_never_start(harness: Harness, launcher: Launcher, event: str) -> None:
    harness.seed()
    invoke(harness, event)
    assert launcher.calls == []
    assert not (harness.registry.root / "auto-extract").exists()


def test_missing_or_failed_project_binding_never_starts(
    harness: Harness,
    launcher: Launcher,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness.environ.clear()
    invoke(harness)
    assert launcher.calls == []
    harness.environ["CLAUDE_PROJECT_DIR"] = str(harness.transcript.parent)

    def fail(directory: str) -> None:
        raise OSError("Private binding details")

    monkeypatch.setattr(hook, "bind_project", fail)
    invoke(harness)
    assert launcher.calls == []
    assert not (harness.registry.root / "auto-extract").exists()


def test_launch_failure_preserves_output_and_retries_next_prompt(
    harness: Harness,
    launcher: Launcher,
    clock: Clock,
) -> None:
    harness.seed()
    baseline = invoke(harness, profile=None)
    harness.registry.mark_redeliver(KEY, "full-session-id")
    launcher.fail = True
    assert invoke(harness) == baseline
    assert not (harness.registry.root / "auto-extract" / "state.json").exists()
    assert harness.diagnostic.getvalue() == "Memory auto-extraction could not start.\n"
    assert harness.registry.read_registration(KEY) is not None
    assert not harness.registry.needs_redeliver(KEY, "full-session-id")
    launcher.fail = False
    invoke(harness)
    assert len(launcher.calls) == 2 and len(state(harness)) == 1
    assert all(log.closed for log in launcher.logs)


@pytest.mark.parametrize("payload", [b"{broken", b"[]", b'{"x":true}', b'{"x":NaN}'])
def test_corrupt_state_is_replaced(harness: Harness, launcher: Launcher, clock: Clock, payload: bytes) -> None:
    root = harness.registry.root / "auto-extract"
    root.mkdir(parents=True)
    plant_owner_only_bytes(root / "state.json", payload)
    invoke(harness)
    assert len(launcher.calls) == 1
    assert state(harness) == {os.path.normcase(str(harness.transcript.parent.resolve())): clock.now}


def test_future_time_is_due_and_log_keeps_only_latest_launch(
    harness: Harness,
    launcher: Launcher,
    clock: Clock,
) -> None:
    invoke(harness)
    root = harness.registry.root / "auto-extract"
    log = next((root / "logs").glob("*.log"))
    plant_owner_only_bytes(log, b"previous log")
    clock.now -= 1
    invoke(harness)
    assert len(launcher.calls) == 2
    assert log.read_bytes() == b""
    assert list((root / "logs").glob("*.log")) == [log]


def test_two_concurrent_claims_start_once(tmp_path: Path, launcher: Launcher, clock: Clock) -> None:
    barrier = threading.Barrier(2)

    def claim() -> None:
        barrier.wait(timeout=ENGINE_TURN_TIMEOUT)
        auto_extract.maybe_start(tmp_path / "memory", tmp_path / "source", tmp_path, "X")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(claim), pool.submit(claim)]
        for future in futures:
            future.result(timeout=ENGINE_TURN_TIMEOUT)
    assert len(launcher.calls) == 1


def test_busy_state_lock_skips_silently(harness: Harness, launcher: Launcher, monkeypatch: pytest.MonkeyPatch) -> None:
    root = harness.registry.root / "auto-extract"
    root.mkdir(parents=True)
    plant_owner_only_bytes(root / "state.lock", b"")
    monkeypatch.setattr(auto_extract, "STATE_LOCK_TIMEOUT_SECONDS", 0)
    with FileLock(root / "state.lock"):
        invoke(harness)
    assert launcher.calls == [] and harness.diagnostic.getvalue() == ""


def test_global_lock_serializes_extractors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_started, second_waiting, release = threading.Event(), threading.Event(), threading.Event()
    second_started = threading.Event()

    class ObservedLock(FileLock):
        def acquire(self) -> None:
            if first_started.is_set():
                second_waiting.set()
            super().acquire()

    monkeypatch.setattr(auto_extract, "FileLock", ObservedLock)

    def first() -> int:
        first_started.set()
        assert release.wait(ENGINE_TURN_TIMEOUT)
        return 7

    def second() -> int:
        second_started.set()
        return 9

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(auto_extract.run_locked, tmp_path, first)
        b = None
        try:
            assert first_started.wait(ENGINE_TURN_TIMEOUT)
            b = pool.submit(auto_extract.run_locked, tmp_path, second)
            assert second_waiting.wait(ENGINE_TURN_TIMEOUT)
            assert not b.done() and not second_started.is_set()
            release.set()
            assert a.result(timeout=ENGINE_TURN_TIMEOUT) == 7
            assert b.result(timeout=ENGINE_TURN_TIMEOUT) == 9
        finally:
            release.set()
            a.result(timeout=ENGINE_TURN_TIMEOUT)
            if b is not None:
                b.result(timeout=ENGINE_TURN_TIMEOUT)


def test_global_lock_timeout_fails_without_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "auto-extract"
    root.mkdir()
    plant_owner_only_bytes(root / "run.lock", b"")
    monkeypatch.setattr(auto_extract, "RUN_LOCK_TIMEOUT_SECONDS", 0)

    def forbidden() -> int:
        pytest.fail("Extraction began before obtaining the global lock")

    with FileLock(root / "run.lock"):
        assert auto_extract.run_locked(tmp_path, forbidden) != 0
    assert capsys.readouterr().err == "Memory auto-extraction timed out waiting for the run lock.\n"


async def test_batch_cap_overrides_tuning_without_changing_version(workspace) -> None:
    options, sources = workspace
    sources = (sources[0], replace(sources[1], key=replace(sources[1].key, host="codex")))
    tuning = replace(V0, version="v7", batch_cap=2)
    readers = {"claude-code": Reader(sources)}
    baseline = await extract.run_extraction(replace(options, dry_run=True), PROFILE, readers=readers, tuning=tuning)
    report = await extract.run_extraction(
        replace(options, dry_run=True, batch_cap=1),
        PROFILE,
        readers=readers,
        tuning=tuning,
    )
    assert baseline.exit_code == report.exit_code == 0
    assert len(baseline.batches) == 2 and len(report.batches) == 1
    assert report.profile["batch_cap"] == 1 and baseline.profile["batch_cap"] == 2
    assert report.tuning_version == "v7" and tuning.batch_cap == 2


@pytest.mark.parametrize(
    "args",
    [
        ["--batch-cap", "0"],
        ["--batch-cap", "-1"],
        ["--batch-cap", "1", "--replay", "frozen.json", "--memory-dir", "isolated"],
    ],
)
def test_batch_cap_cli_rejects_invalid_or_replay(tmp_path: Path, args: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        extract.main(["-C", str(tmp_path), *args])
    assert error.value.code == 2


@pytest.mark.parametrize("automatic", [False, True])
def test_cli_passes_batch_cap_and_bootstraps_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    automatic: bool,
) -> None:
    bootstraps: list[Path] = []
    seen: list[extract.ExtractionOptions] = []

    def bootstrap(*, project_root: Path, apply_patches: bool, setup_telemetry: bool) -> SimpleNamespace:
        bootstraps.append(project_root)
        return SimpleNamespace(
            loaded=SimpleNamespace(
                settings=SimpleNamespace(
                    raw_http_capture=False,
                    model_profile="X",
                )
            )
        )

    async def run(options: extract.ExtractionOptions, profile) -> extract.RunReport:
        seen.append(options)
        assert profile is PROFILE
        if automatic:
            with pytest.raises(TimeoutError), FileLock(tmp_path / "memory" / "auto-extract" / "run.lock", timeout=0):
                pytest.fail("The automatic extraction did not hold its global lock")
        return extract.RunReport(profile={"batch_cap": options.batch_cap})

    monkeypatch.setattr(extract, "bootstrap_runtime", bootstrap)
    monkeypatch.setattr(extract, "get_platform", lambda: SimpleNamespace(config_dir=tmp_path))
    monkeypatch.setattr(extract.ModelProfileRegistry, "load_all", lambda self: None)
    monkeypatch.setattr(extract, "resolve_selectable_profile", lambda registry, selector: PROFILE)
    monkeypatch.setattr(extract, "run_extraction", run)
    args = ["-C", str(tmp_path), "--claude-source", str(tmp_path / "source"), "--batch-cap", "3"]
    if automatic:
        args.append("--auto-extract-lock")
    assert extract.main(args) == 0
    assert bootstraps == [tmp_path]
    assert len(seen) == 1 and seen[0].batch_cap == 3
    assert seen[0].claude_source == tmp_path / "source" and seen[0].codex_source is None
    assert (tmp_path / "memory" / "auto-extract").exists() is automatic
