# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Command orchestration, source rechecks, and real watermark commit points."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import pytest

from chrys.app.memory_hosts import extract
from chrys.app.memory_hosts.extract import ExtractionOptions, run_extraction
from chrys.service.memory import extraction as core
from chrys.service.memory.extraction import (
    ExtractionBatch,
    ExtractionResult,
    ScanDiagnostic,
    ScanResult,
    SessionKey,
    SourceAnnotation,
    SourceRevision,
    SourceSession,
    UsageSummary,
    WatermarkStore,
)
from chrys.service.memory.project import bind_project
from chrys.service.memory.storage.store import MemoryScope
from chrys.service.memory.verification import SourceMessage
from chrys.service.profiles.models.schema import ModelProfile
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for

PROFILE = ModelProfile(
    "fake-response",
    "Offline Responses",
    provider="deepseek-openai",
    api_style="responses",
    model_id="offline-only",
    max_context_tokens=100_000,
)


def result(termination: core.Termination = "completed", *, failures: int = 0, calls: int = 1) -> ExtractionResult:
    return ExtractionResult(termination, calls, failures, 0, 0, None, UsageSummary())


@dataclass
class Reader:
    sources: tuple[SourceSession, ...]
    checks: dict[SessionKey, list[str | None]] = field(default_factory=dict)

    async def scan(self, project: MemoryScope) -> ScanResult:
        assert all(source.project == project for source in self.sources)
        return ScanResult(self.sources)

    async def read_files(self, project: MemoryScope, paths: tuple[Path, ...]) -> ScanResult:
        return await self.scan(project)

    async def revalidate(self, source: SourceSession) -> str | None:
        values = self.checks.get(source.key, [])
        return values.pop(0) if values else None


@dataclass
class Executor:
    returned: ExtractionResult
    batches: list[ExtractionBatch] = field(default_factory=list)

    async def run(self, batch: ExtractionBatch) -> ExtractionResult:
        self.batches.append(batch)
        return self.returned


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[ExtractionOptions, tuple[SourceSession, ...]]:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project = bind_project(str(project_dir))
    sources = tuple(
        SourceSession(
            SessionKey("claude-code", f"synthetic-session-{index}"),
            project,
            SourceRevision(index + 1, str(index).zfill(64)),
            (SourceMessage(f"synthetic-session-{index}", 1, ("Keep public names concise.",), item="user-1"),),
        )
        for index in range(2)
    )
    return ExtractionOptions(project_dir, tmp_path / "memory"), sources


async def test_dry_run_and_preflight_do_not_create_client_lock_or_progress(
    workspace: tuple[ExtractionOptions, tuple[SourceSession, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options, sources = workspace
    runner = Executor(result())
    report = await run_extraction(
        replace(options, dry_run=True), PROFILE, readers={"claude-code": Reader(sources)}, executor=runner
    )
    assert report.exit_code == 0
    assert report.status == "dry_run"
    assert report.batches[0]["displayed_messages"] == 2
    assert runner.batches == []
    assert not await asyncio.to_thread(options.memory_dir.exists)
    assert report.profile["max_turns"] == 24
    assert report.profile["max_output_tokens"] == 16_384
    assert report.profile["batch_timeout_seconds"] == 600


@pytest.mark.parametrize("provider", ["openai", "deepseek-openai"])
async def test_other_protocols_fail_before_scan_attempt_or_executor(
    workspace: tuple[ExtractionOptions, tuple[SourceSession, ...]],
    provider: str,
) -> None:
    options, sources = workspace
    runner = Executor(result())
    report = await run_extraction(
        options,
        replace(PROFILE, provider=provider, api_style="chat_completions"),
        readers={"claude-code": Reader(sources)},
        executor=runner,
    )
    assert (report.status, report.exit_code) == ("unsupported_extraction_protocol", 1)
    assert runner.batches == []
    assert not await asyncio.to_thread(options.memory_dir.exists)


async def test_unmodified_success_not_retried_and_deleted_entries_do_not_reappear(
    workspace: tuple[ExtractionOptions, tuple[SourceSession, ...]],
) -> None:
    options, sources = workspace
    runner = Executor(result())
    readers = {"claude-code": Reader(sources)}
    first = await run_extraction(options, PROFILE, readers=readers, executor=runner)
    second = await run_extraction(options, PROFILE, readers=readers, executor=runner)
    assert (first.status, first.exit_code, second.status, second.exit_code) == ("completed", 0, "no_candidates", 0)
    assert len(runner.batches) == 1
    snapshot = await WatermarkStore(options.memory_dir).load()
    for source in sources:
        record = snapshot.record(str(source.project.project_id), source.key)
        assert record.updated == source.revision
        assert record.attempts == 0


@pytest.mark.parametrize(
    ("returned", "status", "code", "attempts"),
    [
        (result(calls=0), "failed", 1, 1),
        (result(failures=1), "failed", 1, 1),
        (result("turn_limited", failures=1), "failed", 1, 1),
        (result("output_limited", calls=0), "incomplete_limits", 2, 0),
        (result("cancelled"), "cancelled_before_commit", 130, 1),
    ],
)
async def test_result_classification_controls_real_final_progress(
    workspace: tuple[ExtractionOptions, tuple[SourceSession, ...]],
    returned: ExtractionResult,
    status: str,
    code: int,
    attempts: int,
) -> None:
    options, sources = workspace
    report = await run_extraction(
        options, PROFILE, readers={"claude-code": Reader(sources)}, executor=Executor(returned)
    )
    assert (report.status, report.exit_code) == (status, code)
    snapshot = await WatermarkStore(options.memory_dir).load()
    for source in sources:
        record = snapshot.record(str(source.project.project_id), source.key)
        assert record.updated is None
        assert record.attempts == attempts


async def test_two_clean_limits_split_next_run_without_counting_as_failure(
    workspace: tuple[ExtractionOptions, tuple[SourceSession, ...]],
) -> None:
    options, sources = workspace
    runner = Executor(result("turn_limited"))
    for _ in range(2):
        report = await run_extraction(options, PROFILE, readers={"claude-code": Reader(sources)}, executor=runner)
        assert report.exit_code == 2
    assert report.batches[0]["batch_split"]
    snapshot = await WatermarkStore(options.memory_dir).load()
    assert all(snapshot.record(str(source.project.project_id), source.key).batch_session_cap == 1 for source in sources)
    runner = Executor(result())
    completed = await run_extraction(options, PROFILE, readers={"claude-code": Reader(sources)}, executor=runner)
    assert completed.exit_code == 0
    assert [len(batch.sources) for batch in runner.batches] == [1, 1]


async def test_revalidation_removes_only_invalid_members_without_charging_them(
    workspace: tuple[ExtractionOptions, tuple[SourceSession, ...]],
) -> None:
    options, sources = workspace
    sources = (
        replace(sources[0], annotations=(SourceAnnotation("user_interrupt", 1),)),
        replace(sources[1], annotations=(SourceAnnotation("future_signal", 1), SourceAnnotation("future_signal", 2))),
    )
    runner = Executor(result())
    report = await run_extraction(
        options, PROFILE, readers={"claude-code": Reader(sources, {sources[0].key: ["active"]})}, executor=runner
    )
    assert report.exit_code == 0
    assert tuple(source.key for source in runner.batches[0].sources) == (sources[1].key,)
    assert report.batches[0]["annotated_sessions"] == 1
    assert report.batches[0]["annotations"] == {"user_interrupt": 0, "future_signal": 2}
    snapshot = await WatermarkStore(options.memory_dir).load()
    assert snapshot.record(str(sources[0].project.project_id), sources[0].key).attempted_updated is None


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("filtered_status", ["processed", "paused"])
async def test_run_and_batch_signals_count_only_selected_candidates(
    workspace: tuple[ExtractionOptions, tuple[SourceSession, ...]], dry_run: bool, filtered_status: str
) -> None:
    options, originals = workspace
    annotated = replace(
        originals[0],
        annotations=(
            SourceAnnotation("user_interrupt", 1),
            SourceAnnotation("user_interrupt", 2),
            SourceAnnotation("future_signal", 3),
            SourceAnnotation("future_signal", 4),
        ),
    )
    plain = originals[1]
    filtered = replace(annotated, key=SessionKey("claude-code", "filtered-session"))
    codex = replace(
        plain,
        key=SessionKey("codex", "codex-candidate"),
        messages=(SourceMessage("codex-candidate", 1, ("Keep public names concise.",)),),
    )
    watermarks = WatermarkStore(options.memory_dir)
    if filtered_status == "processed":
        prepared = await watermarks.prepare(filtered.project, (filtered,))
        assert prepared.ticket is not None
        assert (await watermarks.succeed(prepared.ticket)).outcome == "committed"
    else:
        for _ in range(3):
            assert (await watermarks.prepare(filtered.project, (filtered,))).outcome == "committed"

    @dataclass
    class DiagnosticReader(Reader):
        diagnostics: tuple[ScanDiagnostic, ...] = ()

        async def scan(self, project: MemoryScope) -> ScanResult:
            scanned = await super().scan(project)
            return ScanResult(scanned.sessions, self.diagnostics)

    runner = Executor(result())
    report = await run_extraction(
        replace(options, dry_run=dry_run),
        PROFILE,
        readers={
            "claude-code": DiagnosticReader(
                (annotated, plain, filtered),
                diagnostics=(
                    ScanDiagnostic("signals_unavailable", filtered.key),
                    ScanDiagnostic("signals_unavailable", SessionKey("codex", plain.key.session)),
                    ScanDiagnostic("signals_unavailable", SessionKey("claude-code", "unmapped-session")),
                ),
            ),
            "codex": DiagnosticReader(
                (codex,),
                diagnostics=(
                    ScanDiagnostic("signals_unavailable", codex.key),
                    ScanDiagnostic("signals_unavailable", codex.key, count=7),
                ),
            ),
        },
        executor=runner,
    )
    assert (report.status, report.exit_code) == ("dry_run" if dry_run else "completed", 0)
    assert asdict(report)["signals"] == {
        "candidate_sessions": 3,
        "annotated_sessions": 1,
        "annotations": {"user_interrupt": 2, "future_signal": 2},
        "signals_unavailable_sessions": 1,
    }
    assert len(report.batches) == 2
    for summary in report.batches:
        is_claude = summary["sessions"][0]["host"] == "claude-code"
        assert summary["annotated_sessions"] == (1 if is_claude else 0)
        assert summary["annotations"] == (
            {"user_interrupt": 2, "future_signal": 2} if is_claude else {"user_interrupt": 0}
        )
    assert len(runner.batches) == (0 if dry_run else 2)


async def test_source_reopened_after_model_preserves_attempt_and_blocks_success(
    workspace: tuple[ExtractionOptions, tuple[SourceSession, ...]],
) -> None:
    options, sources = workspace
    reader = Reader(sources, {sources[0].key: [None, "active"]})
    report = await run_extraction(options, PROFILE, readers={"claude-code": reader}, executor=Executor(result()))
    assert (report.status, report.exit_code) == ("failed", 1)
    record = (await WatermarkStore(options.memory_dir).load()).record(
        str(sources[0].project.project_id), sources[0].key
    )
    assert record.updated is None
    assert record.attempts == 1


@pytest.mark.parametrize(
    ("phase", "status", "code", "attempts", "successful"),
    [
        (1, "attempt_recorded_then_cancelled", 130, 1, False),
        (2, "committed_then_cancelled", 0, 0, True),
        (3, "incomplete_limits_committed_then_cancelled", 2, 0, False),
    ],
)
async def test_cancellation_inside_real_atomic_watermark_worker_reports_persistent_result(
    workspace: tuple[ExtractionOptions, tuple[SourceSession, ...]],
    monkeypatch: pytest.MonkeyPatch,
    phase: int,
    status: str,
    code: int,
    attempts: int,
    successful: bool,
) -> None:
    options, sources = workspace
    entered, release = threading.Event(), threading.Event()
    original = core.atomic_write_owner_only_bytes
    writes = 0

    def blocked(path: Path, payload: bytes, *, create_parents: bool = True) -> None:
        nonlocal writes
        writes += 1
        if writes == (1 if phase == 1 else 2):
            entered.set()
            if not release.wait(ENGINE_TURN_TIMEOUT):
                raise TimeoutError("test gate was not released")
        original(path, payload, create_parents=create_parents)

    monkeypatch.setattr(core, "atomic_write_owner_only_bytes", blocked)
    runner = Executor(result("turn_limited" if phase == 3 else "completed"))
    task = asyncio.create_task(
        run_extraction(options, PROFILE, readers={"claude-code": Reader(sources)}, executor=runner)
    )
    try:
        await wait_for(lambda: entered.is_set() or task.done(), timeout=ENGINE_TURN_TIMEOUT)
        if task.done():
            await task
        assert entered.is_set()
        assert not task.done()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        assert not task.done()
        second = await run_extraction(
            options, PROFILE, readers={"claude-code": Reader(sources)}, executor=Executor(result())
        )
        assert second.status == "already_running"
        release.set()
        report = await task
        assert (report.status, report.exit_code, report.cancel_requested) == (status, code, True)
        assert len(runner.batches) == (0 if phase == 1 else 1)
        snapshot = await WatermarkStore(options.memory_dir).load()
        for source in sources:
            record = snapshot.record(str(source.project.project_id), source.key)
            assert record.attempts == attempts
            assert (record.updated == source.revision) is successful
    finally:
        release.set()
        await task


async def test_reference_over_window_fails_before_attempts(
    workspace: tuple[ExtractionOptions, tuple[SourceSession, ...]],
) -> None:
    options, sources = workspace
    await asyncio.to_thread(
        (options.project_dir / "AGENTS.md").write_text, "Reference-only text. " * 12_000, encoding="utf-8"
    )
    runner = Executor(result())
    report = await run_extraction(
        options, replace(PROFILE, max_context_tokens=25_000), readers={"claude-code": Reader(sources)}, executor=runner
    )
    assert (report.status, report.exit_code) == ("input_budget_exceeded", 1)
    assert not await asyncio.to_thread(options.memory_dir.exists)
    assert runner.batches == []
    assert "AGENTS.md" in json.dumps(report.diagnostics)


@pytest.mark.parametrize(
    "args", [["--max-turns", "0"], ["--batch-timeout-seconds", "nan"], ["--replay", "frozen.json"]]
)
def test_cli_rejects_invalid_limits_and_unisolated_replay_before_bootstrap(args: list[str], tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        extract.main(["-C", str(tmp_path), *args])
    assert error.value.code == 2
