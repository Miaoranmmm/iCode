# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Regression coverage for batch ownership, cancellation, and version-scoped caps."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from chrys.app.memory_hosts.extract import ExtractionOptions, run_extraction
from chrys.app.memory_hosts.extraction_tasks import OperationNotAdmitted
from chrys.foundation.util.lock import FileLock
from chrys.service.memory import extraction as core
from chrys.service.memory.extraction import (
    AttemptTicket,
    CommitResult,
    ExtractionBatch,
    ExtractionResult,
    ScanResult,
    SessionKey,
    SourceRevision,
    SourceSession,
    UsageSummary,
    WatermarkStore,
)
from chrys.service.memory.operations import CatalogSnapshot, MemoryReader
from chrys.service.memory.project import bind_project
from chrys.service.memory.storage.store import MemoryOperationError, MemoryScope
from chrys.service.memory.verification import SourceMessage
from chrys.service.profiles.models.schema import ModelProfile
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for

PROFILE = ModelProfile(
    "offline-response",
    "Offline Response",
    provider="deepseek-openai",
    api_style="responses",
    model_id="synthetic",
    max_context_tokens=100_000,
)


@dataclass
class Reader:
    sources: tuple[SourceSession, ...]

    async def scan(self, project: MemoryScope) -> ScanResult:
        assert all(source.project == project for source in self.sources)
        return ScanResult(self.sources)

    async def read_files(self, project: MemoryScope, paths: tuple[Path, ...]) -> ScanResult:
        return await self.scan(project)

    async def revalidate(self, source: SourceSession) -> str | None:
        return None


@dataclass
class Executor:
    limited: bool = False
    batches: list[ExtractionBatch] = field(default_factory=list)

    async def run(self, batch: ExtractionBatch) -> ExtractionResult:
        self.batches.append(batch)
        return ExtractionResult("turn_limited" if self.limited else "completed", 1, 0, 0, 0, None, UsageSummary())


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ExtractionOptions, tuple[SourceSession, ...]]:
    home = tmp_path / "home"
    home.mkdir()
    for name in ("HOME", "USERPROFILE", "APPDATA"):
        monkeypatch.setenv(name, str(home))
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project = bind_project(str(project_dir))
    sources = tuple(
        SourceSession(
            SessionKey("claude-code", f"synthetic-session-{index}"),
            project,
            SourceRevision(index + 1, f"{index:064x}"),
            (SourceMessage(f"synthetic-session-{index}", 1, ("Keep names concise.",), item="user-one"),),
        )
        for index in range(11)
    )
    return ExtractionOptions(project_dir, tmp_path / "memory"), sources


@pytest.mark.parametrize(
    ("action", "phase"), [("catalog", "catalog"), ("revalidate", "revalidation"), ("prepare", "attempt")]
)
async def test_second_batch_pre_admission_cancellation_never_rewrites_the_first_commit(
    workspace: tuple[ExtractionOptions, tuple[SourceSession, ...]],
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    phase: str,
) -> None:
    options, sources = workspace
    first_committed = False
    original_succeed = WatermarkStore.succeed
    original_catalog = MemoryReader.catalog
    original_prepare = WatermarkStore.prepare
    original_revalidate = Reader.revalidate

    async def succeed(store: WatermarkStore, ticket: AttemptTicket) -> CommitResult:
        nonlocal first_committed
        outcome = await original_succeed(store, ticket)
        first_committed = outcome.outcome == "committed"
        return outcome

    def catalog(reader: MemoryReader) -> Coroutine[Any, Any, CatalogSnapshot]:
        if first_committed and action == "catalog":
            raise OperationNotAdmitted()
        return original_catalog(reader)

    def prepare(
        store: WatermarkStore, project: MemoryScope, members: Sequence[SourceSession]
    ) -> Coroutine[Any, Any, CommitResult]:
        if first_committed and action == "prepare":
            raise OperationNotAdmitted()
        return original_prepare(store, project, members)

    def revalidate(reader: Reader, source: SourceSession) -> Coroutine[Any, Any, str | None]:
        if first_committed and action == "revalidate":
            raise OperationNotAdmitted()
        return original_revalidate(reader, source)

    monkeypatch.setattr(WatermarkStore, "succeed", succeed)
    monkeypatch.setattr(MemoryReader, "catalog", catalog)
    monkeypatch.setattr(WatermarkStore, "prepare", prepare)
    monkeypatch.setattr(Reader, "revalidate", revalidate)
    runner = Executor()
    report = await run_extraction(options, PROFILE, readers={"claude-code": Reader(sources)}, executor=runner)
    snapshot = await WatermarkStore(options.memory_dir).load()
    assert len(runner.batches) == 1
    for source in runner.batches[0].sources:
        record = snapshot.record(str(source.project.project_id), source.key)
        assert (record.updated, record.attempts) == (source.revision, 0)
    untouched = snapshot.record(str(sources[0].project.project_id), sources[0].key)
    assert untouched.updated is None and untouched.attempted_updated is None and untouched.attempts == 0
    assert (report.status, report.exit_code, report.cancel_requested) == ("cancelled_before_" + phase, 130, True)
    assert report.not_started_batches == 1
    assert len(report.batches) == 2
    first, second = report.batches
    assert (first["status"], first["commit_outcome"], first["cancel_requested"]) == ("completed", "committed", False)
    assert second["sessions"] == [{"host": "claude-code", "session": sources[0].key.session}]
    assert second["status"] == report.status
    assert second["cancel_requested"] is True
    assert second.get("commit_outcome") is None


@pytest.mark.parametrize(
    ("limited", "status", "code"),
    [
        (False, "committed_then_cancelled", 0),
        (True, "incomplete_limits_committed_then_cancelled", 2),
    ],
)
async def test_final_watermark_worker_cancellation_is_recorded_on_its_batch(
    workspace: tuple[ExtractionOptions, tuple[SourceSession, ...]],
    monkeypatch: pytest.MonkeyPatch,
    limited: bool,
    status: str,
    code: int,
) -> None:
    options, all_sources = workspace
    sources = all_sources[:2]
    entered, release = threading.Event(), threading.Event()
    original = core.atomic_write_owner_only_bytes
    writes = 0

    def blocked(path: Path, payload: bytes, *, create_parents: bool = True) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            entered.set()
            if not release.wait(ENGINE_TURN_TIMEOUT):
                raise TimeoutError("test worker gate was not released")
        original(path, payload, create_parents=create_parents)

    monkeypatch.setattr(core, "atomic_write_owner_only_bytes", blocked)
    task = asyncio.create_task(
        run_extraction(options, PROFILE, readers={"claude-code": Reader(sources)}, executor=Executor(limited))
    )
    try:
        await wait_for(lambda: entered.is_set() or task.done(), timeout=ENGINE_TURN_TIMEOUT)
        if task.done():
            await task
        assert entered.is_set() and not task.done()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        report = await task
        record = (await WatermarkStore(options.memory_dir).load()).record(
            str(sources[0].project.project_id), sources[0].key
        )
        assert record.attempts == 0
        assert record.updated == (None if limited else sources[0].revision)
        assert (report.status, report.exit_code, report.cancel_requested) == (status, code, True)
        assert report.batches[0]["status"] == status
        assert report.batches[0]["commit_outcome"] == "committed"
        assert report.batches[0]["cancel_requested"] is True
    finally:
        release.set()
        await task


@pytest.mark.parametrize(
    "mode",
    [
        "dry_run",
        "no_candidates",
        "already_running",
        "acquired",
        "empty_batch",
        "catalog_extraction_error",
        "catalog_memory_error",
        "catalog_unexpected_error",
    ],
)
async def test_absorbed_cancellation_cannot_escape_through_a_normal_return(
    workspace: tuple[ExtractionOptions, tuple[SourceSession, ...]],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    options, all_sources = workspace
    sources = () if mode == "no_candidates" else all_sources[:1]
    entered, release = asyncio.Event(), asyncio.Event()
    phase = (
        "catalog"
        if mode in {"dry_run", "no_candidates"} or mode.startswith("catalog_")
        else "revalidation"
        if mode == "empty_batch"
        else "run_lock"
    )
    catalog = MemoryReader.catalog
    acquire = core.ProjectRunLock.try_acquire

    async def gated_catalog(reader: MemoryReader) -> CatalogSnapshot:
        entered.set()
        await release.wait()
        if mode == "catalog_extraction_error":
            raise core.ExtractionError("offline_failure", "Safe failure.")
        if mode == "catalog_memory_error":
            raise MemoryOperationError("memory_store_unavailable", "Safe failure.")
        if mode == "catalog_unexpected_error":
            raise RuntimeError("PRIVATE_PREFLIGHT_FAILURE")
        return await catalog(reader)

    async def gated_acquire(lock: core.ProjectRunLock) -> bool:
        entered.set()
        await release.wait()
        return False if mode == "already_running" else await acquire(lock)

    class GatedReader(Reader):
        async def revalidate(self, source: SourceSession) -> str | None:
            entered.set()
            await release.wait()
            return "active"

    if phase == "catalog":
        monkeypatch.setattr(MemoryReader, "catalog", gated_catalog)
    elif phase == "run_lock":
        monkeypatch.setattr(core.ProjectRunLock, "try_acquire", gated_acquire)
    runner = Executor()
    task = asyncio.create_task(
        run_extraction(
            replace(options, dry_run=mode == "dry_run"),
            PROFILE,
            readers={"claude-code": GatedReader(sources) if mode == "empty_batch" else Reader(sources)},
            executor=runner,
        )
    )
    try:
        await wait_for(lambda: entered.is_set() or task.done(), timeout=ENGINE_TURN_TIMEOUT)
        if task.done():
            await task
        assert entered.is_set() and not task.done()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        report = await task
        assert (report.status, report.exit_code, report.cancel_requested) == ("cancelled_before_" + phase, 130, True)
        assert runner.batches == []
        assert "PRIVATE_PREFLIGHT_FAILURE" not in repr(report)
        assert not await asyncio.to_thread((options.memory_dir / "extraction.json").exists)
        if mode == "empty_batch":
            assert report.batches[0]["status"] == report.status
            assert report.batches[0]["cancel_requested"] is True
    finally:
        release.set()
        await task


@pytest.mark.parametrize("phase", ["attempt", "success", "incomplete_limits"])
async def test_watermark_lock_timeout_settles_the_batch_for_the_failed_commit_phase(
    workspace: tuple[ExtractionOptions, tuple[SourceSession, ...]],
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    options, all_sources = workspace
    sources = all_sources[:1]
    acquisitions = 0

    class WatermarkLock(FileLock):
        def acquire(self) -> None:
            nonlocal acquisitions
            if self._path == options.memory_dir / "extraction.lock":
                acquisitions += 1
                if acquisitions == (1 if phase == "attempt" else 2):
                    raise TimeoutError("PRIVATE_LOCK_TIMEOUT")
            super().acquire()

    monkeypatch.setattr(core, "FileLock", WatermarkLock)
    runner = Executor(phase == "incomplete_limits")
    report = await run_extraction(options, PROFILE, readers={"claude-code": Reader(sources)}, executor=runner)
    assert (report.status, report.exit_code) == ("extraction_failed", 1)
    summary = report.batches[0]
    assert summary["status"] == report.status
    assert summary["failed_phase"] == phase
    assert (summary["commit_phase"], summary["commit_outcome"]) == (phase, "not_committed")
    assert "PRIVATE_LOCK_TIMEOUT" not in repr(report)
    assert len(runner.batches) == (0 if phase == "attempt" else 1)
    record = (await WatermarkStore(options.memory_dir).load()).record(
        str(sources[0].project.project_id), sources[0].key
    )
    assert record.updated is None
    assert record.attempts == (0 if phase == "attempt" else 1)


@pytest.mark.parametrize("failure", ["extraction", "memory", "unexpected"])
async def test_every_non_cancel_exception_settles_the_active_batch(
    workspace: tuple[ExtractionOptions, tuple[SourceSession, ...]],
    failure: str,
) -> None:
    options, all_sources = workspace
    sources = all_sources[:1]

    class FailingExecutor:
        async def run(self, batch: ExtractionBatch) -> ExtractionResult:
            if failure == "extraction":
                raise core.ExtractionError("offline_failure", "Safe failure.")
            if failure == "memory":
                raise MemoryOperationError("memory_store_unavailable", "Safe failure.")
            raise RuntimeError("PRIVATE_EXECUTOR_FAILURE")

    report = await run_extraction(
        options, PROFILE, readers={"claude-code": Reader(sources)}, executor=FailingExecutor()
    )
    assert report.exit_code == 1
    summary = report.batches[0]
    assert summary["status"] == report.status
    assert summary["failed_phase"] == "model"
    assert (summary["commit_phase"], summary["commit_outcome"]) == ("attempt", "committed")
    assert "PRIVATE_EXECUTOR_FAILURE" not in repr(report)


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("version", ["unchanged", "touched", "changed"])
async def test_app_grouping_uses_caps_only_for_the_attempted_content_version(
    workspace: tuple[ExtractionOptions, tuple[SourceSession, ...]],
    dry_run: bool,
    version: str,
) -> None:
    options, all_sources = workspace
    sources = all_sources[:2]
    for _ in range(2):
        limited = await run_extraction(
            options, PROFILE, readers={"claude-code": Reader(sources)}, executor=Executor(True)
        )
        assert limited.exit_code == 2
    snapshot = await WatermarkStore(options.memory_dir).load()
    assert all(snapshot.record(str(source.project.project_id), source.key).batch_session_cap == 1 for source in sources)
    if version != "unchanged":
        sources = tuple(
            replace(
                source,
                revision=SourceRevision(
                    source.revision.updated_ns + 100,
                    "b" * 63 + str(index) if version == "changed" else source.revision.sha256,
                ),
            )
            for index, source in enumerate(sources)
        )
    runner = Executor()
    report = await run_extraction(
        replace(options, dry_run=dry_run), PROFILE, readers={"claude-code": Reader(sources)}, executor=runner
    )
    assert report.exit_code == 0
    sizes = (
        [len(batch["sessions"]) for batch in report.batches]
        if dry_run
        else [len(batch.sources) for batch in runner.batches]
    )
    assert sizes == ([2] if version == "changed" else [1, 1])


@pytest.mark.parametrize(
    ("point", "phase", "model_calls", "not_started"),
    [
        ("before_second_model", "model", 1, 1),
        ("before_second_success", "success", 2, 0),
    ],
)
async def test_replay_sticky_cancellation_does_not_admit_another_model_or_report_success(
    workspace: tuple[ExtractionOptions, tuple[SourceSession, ...]],
    point: str,
    phase: str,
    model_calls: int,
    not_started: int,
) -> None:
    options, originals = workspace
    frozen_dir = options.project_dir.parent / "frozen"
    await asyncio.to_thread(frozen_dir.mkdir)
    frozen: list[dict[str, object]] = []
    captured: list[SourceSession] = []
    for index, source in enumerate(originals):
        path = frozen_dir / f"{source.key.session}.jsonl"
        payload = json.dumps({"synthetic_snapshot": index}) + "\n"
        await asyncio.to_thread(path.write_text, payload, encoding="utf-8")
        sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        captured.append(replace(source, revision=SourceRevision(source.revision.updated_ns, sha)))
        frozen.append(
            {
                "host": source.key.host,
                "session": source.key.session,
                "path": path.name,
                "sha256": sha,
                "updated_ns": source.revision.updated_ns,
            }
        )
    sources = tuple(captured)
    batches = core.build_batches(sources[0].project, sources, 40_000)
    manifest = frozen_dir / "replay.json"
    await asyncio.to_thread(
        manifest.write_text,
        json.dumps(
            {
                "version": 1,
                "prompt_version": core.PROMPT_VERSION,
                "render_version": core.RENDER_VERSION,
                "project_id": sources[0].project.project_id,
                "material_budget": 40_000,
                "max_output_tokens": 16_384,
                "sources": frozen,
                "references": [],
                "baseline": [],
                "initial_store": [],
                "batches": [
                    {
                        "sessions": [
                            {"host": source.key.host, "session": source.key.session} for source in batch.sources
                        ],
                        "material_sha256": hashlib.sha256(batch.material.encode("utf-8")).hexdigest(),
                    }
                    for batch in batches
                ],
            }
        ),
        encoding="utf-8",
    )
    entered, release = asyncio.Event(), asyncio.Event()
    runner = Executor()
    calls_at_gate = 1 if point == "before_second_model" else 2

    class GatedReader(Reader):
        async def revalidate(self, source: SourceSession) -> str | None:
            if source.key == sources[0].key and len(runner.batches) == calls_at_gate:
                entered.set()
                await release.wait()
            return None

    task = asyncio.create_task(
        run_extraction(
            replace(options, replay=manifest), PROFILE, readers={"claude-code": GatedReader(sources)}, executor=runner
        )
    )
    try:
        await wait_for(lambda: entered.is_set() or task.done(), timeout=ENGINE_TURN_TIMEOUT)
        if task.done():
            await task
        assert entered.is_set() and not task.done()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        report = await task
        assert len(runner.batches) == model_calls
        assert (report.status, report.exit_code, report.cancel_requested) == ("cancelled_before_" + phase, 130, True)
        assert report.not_started_batches == not_started
        assert len(report.batches) == 2
        first, second = report.batches
        assert (first["status"], first["cancel_requested"]) == ("replay_completed", False)
        assert first.get("commit_outcome") is None
        assert (second["status"], second["cancel_requested"]) == (report.status, True)
        assert second.get("commit_outcome") is None
        assert not await asyncio.to_thread(options.memory_dir.exists)
    finally:
        release.set()
        await task
