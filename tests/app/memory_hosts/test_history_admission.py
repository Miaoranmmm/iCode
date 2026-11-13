# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Real source activity gates admission, while unchanged bytes permit the final commit."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType

import pytest

from chrys.app.memory_hosts import codex_history, history
from chrys.app.memory_hosts.claude_code.history import ClaudeHistoryReader
from chrys.app.memory_hosts.codex_history import CodexHistoryReader
from chrys.app.memory_hosts.extract import ExtractionOptions, run_extraction
from chrys.service.memory.extraction import ExtractionBatch, ExtractionResult, ScanResult, WatermarkStore
from chrys.service.memory.project import bind_project
from chrys.service.memory.storage.store import MemoryScope
from tests.app.memory_hosts.history_fixtures import PARENT, SESSION, FixedCodex, FixedHolder, claude_user, write_jsonl
from tests.app.memory_hosts.test_codex_history import message, meta
from tests.app.memory_hosts.test_extract import PROFILE, Executor, result
from tests.support.secure_files import plant_owner_only_bytes


@pytest.mark.parametrize(
    ("host", "change"),
    [("claude-code", "active"), ("codex", "active"), ("codex", "reverted"), ("codex", "coordination_timeout")],
)
@pytest.mark.parametrize("stage", ["before-model", "after-model"])
def test_real_source_changes_gate_admission_and_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: str, change: str, stage: str
):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project = bind_project(str(project_dir))
    source_dir = tmp_path / "home" / "sessions"
    source_dir.mkdir(parents=True)
    options = ExtractionOptions(project_dir, tmp_path / "memory")
    holder = FixedHolder()
    sources = []
    now = [0.0]
    if change == "coordination_timeout":
        shadow = ModuleType("time")
        shadow.monotonic = lambda: now[0]

        def sleep(delay: float) -> None:
            now[0] += delay

        shadow.sleep = sleep
        monkeypatch.setattr(codex_history, "time", shadow)

    with ExitStack() as stack:
        if host == "codex":
            flock = pytest.importorskip("fcntl")
            locks = source_dir.parent / "thread-writer-locks"
            locks.mkdir()
            coordination = locks / ".coordination.lock"
            plant_owner_only_bytes(coordination, b"")
            coordinator = stack.enter_context(coordination.open("rb"))
            lock = locks / f"{SESSION}.lock"
            plant_owner_only_bytes(lock, b"")
            writer = stack.enter_context(lock.open("rb"))
            dated_source = source_dir / "2026" / "11" / "01"
            dated_source.mkdir(parents=True)
            path = dated_source / f"rollout-2026-11-01T10-00-00-{SESSION}.jsonl"
            reverted_dir = source_dir / "2026" / "11" / "02"
            reverted_dir.mkdir()
            write_jsonl(path, [meta(project_dir), message("Keep errors actionable.")])
            original_bytes = path.read_bytes()
            reader = CodexHistoryReader(source_dir, processes=FixedCodex("active"))

            def activate() -> None:
                if change == "coordination_timeout":
                    flock.flock(coordinator, flock.LOCK_EX | flock.LOCK_NB)
                    return
                flock.flock(coordinator, flock.LOCK_SH | flock.LOCK_NB)
                try:
                    flock.flock(writer, flock.LOCK_EX | flock.LOCK_NB)
                finally:
                    flock.flock(coordinator, flock.LOCK_UN)
                if change == "reverted":
                    write_jsonl(
                        reverted_dir / f"rollout-2026-11-02T10-00-00-{SESSION}_{PARENT}.jsonl",
                        [meta(project_dir), message("Retained after rollback")],
                    )
                    if stage == "before-model":
                        flock.flock(writer, flock.LOCK_UN)

        else:
            write_jsonl(
                source_dir / f"{SESSION}.jsonl", [claude_user("user", "Keep errors actionable.", cwd=str(project_dir))]
            )
            reader = ClaudeHistoryReader(source_dir, holder=holder)

            def activate() -> None:
                holder.activity = "active"

        original_scan = reader.scan

        async def scan(scope: MemoryScope) -> ScanResult:
            scanned = await original_scan(scope)
            sources.extend(scanned.sessions)
            if stage == "before-model":
                await asyncio.to_thread(activate)
            return scanned

        class ReopeningExecutor(Executor):
            async def run(self, batch: ExtractionBatch) -> ExtractionResult:
                outcome = await super().run(batch)
                if stage == "after-model":
                    await asyncio.to_thread(activate)
                return outcome

        monkeypatch.setattr(reader, "scan", scan)
        runner = ReopeningExecutor(result())
        report = asyncio.run(run_extraction(options, PROFILE, readers={host: reader}, executor=runner))
    assert len(sources) == 1
    source = sources[0]
    stored = asyncio.run(WatermarkStore(options.memory_dir).load()).record(str(project.project_id), source.key)
    if change == "reverted":
        assert path.read_bytes() == original_bytes
    if change == "coordination_timeout":
        assert now[0] == pytest.approx(1.0 if stage == "before-model" else 0.0)
    if stage == "before-model":
        assert report.exit_code == 0
        assert runner.batches == []
        assert (stored.updated, stored.attempted_updated, stored.attempts) == (None, None, 0)
        if change != "reverted":
            assert {"kind": "source_reopened", "host": host, "session": SESSION} in report.diagnostics
    elif change == "reverted":
        assert (report.status, report.exit_code) == ("failed", 1)
        assert len(runner.batches) == 1
        assert stored.updated is None and stored.attempted_updated == source.revision and stored.attempts == 1
    else:
        assert (report.status, report.exit_code) == ("completed", 0)
        assert len(runner.batches) == 1
        assert stored.updated == source.revision and stored.attempts == 0
    if change == "reverted":
        assert {"kind": "rollout_reverted", "host": host, "session": SESSION} in report.diagnostics


@pytest.mark.parametrize("host", ["claude-code", "codex"])
@pytest.mark.parametrize("require_idle", [False, True])
def test_revalidation_hashes_bytes_without_parsing_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: str, require_idle: bool
):
    project = bind_project(str(tmp_path))
    if host == "codex":
        path = tmp_path / f"rollout-2026-11-01T10-00-00-{SESSION}.jsonl"
        write_jsonl(path, [meta(tmp_path), message("Keep errors actionable.")])
        reader = CodexHistoryReader(tmp_path, processes=FixedCodex())
    else:
        path = tmp_path / f"{SESSION}.jsonl"
        write_jsonl(path, [claude_user("user", "Keep errors actionable.", cwd=str(tmp_path))])
        reader = ClaudeHistoryReader(tmp_path, holder=FixedHolder())
    source = asyncio.run(reader.scan(project)).sessions[0]

    def reject_parse(content: history.HistoryContent) -> history.HistorySnapshot:
        raise AssertionError("Revalidation must not decode JSON")

    monkeypatch.setattr(history, "parse_snapshot", reject_parse)
    assert asyncio.run(reader.revalidate(source, require_idle=require_idle)) is None
    with path.open("ab") as stream:
        stream.write(b"changed and invalid JSON")
    assert asyncio.run(reader.revalidate(source, require_idle=require_idle)) == "source_changed"
