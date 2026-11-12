# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Real decoder/kernel outcomes drive durable progress and commit exit codes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from chrys.app.memory_hosts import executor_icode
from chrys.app.memory_hosts.extract import ExtractionOptions, run_extraction
from chrys.kernel import Message
from chrys.service.memory import extraction as core
from chrys.service.memory.extraction import ExtractionBatch, ExtractionResult, ScanResult, SourceSession, UsageSummary
from chrys.service.memory.project import bind_project
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import MemoryScope
from tests.app.memory_hosts.executor_fakes import (
    SESSION,
    WORDS,
    ClientFactory,
    ScriptedWire,
    batch,
    call,
    profile,
    response,
    save_arguments,
)
from tests.support.secure_files import plant_owner_only_bytes


@dataclass
class Reader:
    source: SourceSession

    async def scan(self, project: MemoryScope) -> ScanResult:
        assert project == self.source.project
        return ScanResult((self.source,))

    async def read_files(self, project: MemoryScope, paths: tuple[Path, ...]) -> ScanResult:
        return await self.scan(project)

    async def revalidate(self, source: SourceSession) -> str | None:
        assert source == self.source
        return None


@dataclass
class Completed:
    limited: bool = False
    calls: int = 0

    async def run(self, selected: ExtractionBatch) -> ExtractionResult:
        self.calls += 1
        return ExtractionResult("turn_limited" if self.limited else "completed", 1, 0, 0, 0, None, UsageSummary())


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[ExtractionOptions, SourceSession]:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project = bind_project(str(project_dir))
    selected = replace(batch().sources[0], project=project)
    return ExtractionOptions(project_dir, tmp_path / "memory"), selected


async def test_real_wire_search_save_read_commits_watermark_and_run_source(
    workspace: tuple[ExtractionOptions, SourceSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    options, selected = workspace
    wire = ScriptedWire([])
    factory = ClientFactory(wire)
    monkeypatch.setattr(executor_icode, "create_client", factory)
    store = FileMemoryStore(options.memory_dir)

    async def read_saved(messages: Sequence[Message]):
        stored = (await store.list_entries(selected.project))[0]
        return response(call("memory_read", {"ref": stored.entry.ref}))

    wire.script = [
        response(call("memory_search", {"query": "", "scope": "all"})),
        response(call("memory_save", save_arguments())),
        read_saved,
        response(text="Done"),
    ]
    report = await run_extraction(options, profile(), readers={"claude-code": Reader(selected)})
    assert (report.status, report.exit_code) == ("completed", 0)
    assert wire.closed == 1
    assert factory.client is not None and factory.client.max_iterations == 24
    record = (await core.WatermarkStore(options.memory_dir).load()).record(
        str(selected.project.project_id), selected.key
    )
    assert (record.updated, record.attempts) == (selected.revision, 0)
    saved = (await store.list_entries(selected.project))[0]
    assert (saved.entry.written_by, saved.entry.quotes[0].text, saved.entry.quotes[0].session) == (
        "extraction",
        WORDS,
        SESSION,
    )
    log_path = options.memory_dir / "projects" / str(selected.project.project_id) / "changes.jsonl"
    log_text = await asyncio.to_thread(log_path.read_text, encoding="utf-8")
    changes = [json.loads(line) for line in log_text.splitlines()]
    assert len(changes) == 1
    assert changes[0]["source"] == report.run_id
    assert changes[0]["written_by"] == "extraction"
    assert report.batches[0]["run_id"] == report.run_id
    assert report.batches[0]["commit_outcome"] == "committed"
    assert report.batches[0]["created_count"] == 1
    assert WORDS not in json.dumps(report.batches)


@pytest.mark.parametrize(
    "case,expected_code,expected_attempts",
    [
        ("completed", 0, 0),
        ("stale_incomplete_details", 0, 0),
        ("unknown_incomplete", 1, 1),
        ("missing_incomplete_reason", 1, 1),
        ("missing_status", 1, 1),
        ("queued", 1, 1),
        ("in_progress", 1, 1),
        ("duplicate_bad_save", 1, 1),
        ("duplicate_identical", 1, 1),
        ("duplicate_length", 1, 1),
        ("empty_call_id", 1, 1),
        ("null_call_id", 1, 1),
        ("numeric_call_id", 1, 1),
        ("unknown_tool", 1, 1),
        ("future_raw_item", 1, 1),
        ("missing_save_fields", 1, 1),
        ("non_object_args", 1, 1),
        ("wrong_type", 1, 1),
        ("client_error", 1, 1),
        ("zero_calls", 1, 1),
        ("zero_calls_length", 2, 0),
        ("search_length", 2, 0),
        ("invalid_args_length", 2, 0),
        ("prior_failure_length", 1, 1),
        ("turn_limit", 2, 0),
        ("cross_response_reuse", 0, 0),
    ],
)
async def test_actual_executor_outcome_controls_final_watermark(
    workspace: tuple[ExtractionOptions, SourceSession],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected_code: int,
    expected_attempts: int,
) -> None:
    options, selected = workspace
    search = response(call("memory_search", {"query": ""}))
    done = response(text="Done")
    scripts = {
        "completed": [search, done],
        "stale_incomplete_details": [search, response(text="Done", reason="max_output_tokens")],
        "unknown_incomplete": [search, response(text="Done", status="incomplete", reason="future_reason")],
        "missing_incomplete_reason": [search, response(text="Done", status="incomplete")],
        "missing_status": [search, response(text="Done", status=None)],
        "queued": [search, response(text="Done", status="queued")],
        "in_progress": [search, response(text="Done", status="in_progress")],
        "duplicate_bad_save": [response(call("memory_search", {"query": ""}), call("memory_save", {}))],
        "duplicate_identical": [response(call("memory_search", {"query": ""}), call("memory_search", {"query": ""}))],
        "duplicate_length": [
            response(
                call("memory_search", {"query": ""}),
                call("memory_search", {"query": ""}),
                status="incomplete",
                reason="max_output_tokens",
            )
        ],
        "empty_call_id": [response(call("memory_search", {"query": ""}), call("memory_save", save_arguments(), ""))],
        "null_call_id": [response(call("memory_search", {"query": ""}), call("memory_save", save_arguments(), None))],
        "numeric_call_id": [response(call("memory_search", {"query": ""}), call("memory_save", save_arguments(), 3))],
        "unknown_tool": [search, response(call("memory_forget", {}))],
        "future_raw_item": [search, response(SimpleNamespace(type="future_benign_annotation"), text="Done")],
        "missing_save_fields": [search, response(call("memory_save", {})), done],
        "non_object_args": [search, response(call("memory_save", [])), done],
        "wrong_type": [search, response(call("memory_search", {"query": 3})), done],
        "client_error": [search, RuntimeError("PRIVATE_CLIENT_ERROR")],
        "zero_calls": [done],
        "zero_calls_length": [response(text="Partial", status="incomplete", reason="max_output_tokens")],
        "search_length": [
            response(call("memory_search", {"query": ""}), status="incomplete", reason="max_output_tokens")
        ],
        "invalid_args_length": [response(call("memory_save", {}), status="incomplete", reason="max_output_tokens")],
        "prior_failure_length": [
            response(call("memory_save", {})),
            response(call("memory_save", save_arguments()), status="incomplete", reason="max_output_tokens"),
        ],
        "turn_limit": [search, search, done],
        "cross_response_reuse": [search, search, done],
    }
    wire = ScriptedWire(scripts[case])
    monkeypatch.setattr(executor_icode, "create_client", ClientFactory(wire))
    if case == "turn_limit":
        options = replace(options, max_turns=2)
    report = await run_extraction(options, profile(), readers={"claude-code": Reader(selected)})
    assert report.exit_code == expected_code
    assert wire.closed == 1
    record = (await core.WatermarkStore(options.memory_dir).load()).record(
        str(selected.project.project_id), selected.key
    )
    assert record.attempts == expected_attempts
    assert (record.updated == selected.revision) is (expected_code == 0)
    assert report.batches[0]["commit_outcome"] == "committed"
    assert "PRIVATE_CLIENT_ERROR" not in repr(report)
    if case == "turn_limit":
        assert len(wire.requests) == 2
        assert report.batches[0]["tail_seen"] is True
    if case.startswith("duplicate_") or case.endswith("call_id"):
        assert report.batches[0]["call_count"] == 0
    if case in {"search_length", "invalid_args_length"}:
        assert report.batches[0]["call_count"] == 0
        assert report.batches[0]["failure_count"] == 0
    if case == "prior_failure_length":
        assert report.batches[0]["call_count"] == 1
        assert report.batches[0]["failure_count"] == 1
    assert await FileMemoryStore(options.memory_dir).list_entries(selected.project) == []


@pytest.mark.parametrize("phase", ["attempt", "success", "incomplete_limits"])
@pytest.mark.parametrize("failure", ["before_publish", "after_publish", "confirmation_failed"])
async def test_atomic_commit_confirmation_drives_report_exit_codes(
    workspace: tuple[ExtractionOptions, SourceSession],
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    failure: str,
) -> None:
    options, selected = workspace
    options.memory_dir.mkdir()
    plant_owner_only_bytes(options.memory_dir / "extraction.json", b'{"version":1,"projects":{}}')
    original = core.atomic_write_owner_only_bytes
    writes = 0

    def publication(path: Path, payload: bytes, *, create_parents: bool = True) -> None:
        nonlocal writes
        writes += 1
        if writes == (1 if phase == "attempt" else 2):
            if failure == "after_publish":
                original(path, payload, create_parents=create_parents)
            elif failure == "confirmation_failed":
                original(path, b"unreadable state", create_parents=create_parents)
            raise OSError("PRIVATE_PUBLICATION_ERROR")
        original(path, payload, create_parents=create_parents)

    monkeypatch.setattr(core, "atomic_write_owner_only_bytes", publication)
    runner = Completed(phase == "incomplete_limits")
    report = await run_extraction(options, profile(), readers={"claude-code": Reader(selected)}, executor=runner)
    expected = (
        3
        if failure == "confirmation_failed"
        else 1
        if failure == "before_publish"
        else 2
        if phase == "incomplete_limits"
        else 0
    )
    assert report.exit_code == expected
    assert report.batches[0]["status"] == report.status
    assert "PRIVATE_PUBLICATION_ERROR" not in repr(report)
    if failure == "confirmation_failed":
        assert report.status == "commit_unknown"
    else:
        record = (await core.WatermarkStore(options.memory_dir).load()).record(
            str(selected.project.project_id), selected.key
        )
        if failure == "before_publish":
            assert record.updated is None
            assert record.attempts == (0 if phase == "attempt" else 1)
        else:
            assert record.updated == (selected.revision if phase != "incomplete_limits" else None)
            assert record.attempts == 0


async def test_cancellation_before_attempt_admission_creates_no_attempt(
    workspace: tuple[ExtractionOptions, SourceSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    options, selected = workspace
    original = core.WatermarkStore.prepare

    def cancel_before_admission(store: core.WatermarkStore, project: MemoryScope, sources: Sequence[SourceSession]):
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        return original(store, project, sources)

    monkeypatch.setattr(core.WatermarkStore, "prepare", cancel_before_admission)
    runner = Completed()
    report = await run_extraction(options, profile(), readers={"claude-code": Reader(selected)}, executor=runner)
    assert (report.status, report.exit_code) == ("cancelled_before_attempt", 130)
    assert runner.calls == 0
    assert (await core.WatermarkStore(options.memory_dir).load()).records == ()


async def test_cleanup_warning_does_not_reverse_committed_batch(
    workspace: tuple[ExtractionOptions, SourceSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    options, selected = workspace
    original = core.ProjectRunLock.release

    async def release_then_warn(lock: core.ProjectRunLock) -> None:
        await original(lock)
        raise OSError("PRIVATE_CLEANUP_ERROR")

    monkeypatch.setattr(core.ProjectRunLock, "release", release_then_warn)
    report = await run_extraction(options, profile(), readers={"claude-code": Reader(selected)}, executor=Completed())
    assert (report.status, report.exit_code) == ("completed", 0)
    assert any(item["kind"] == "cleanup_warning" for item in report.diagnostics)
    record = (await core.WatermarkStore(options.memory_dir).load()).record(
        str(selected.project.project_id), selected.key
    )
    assert record.updated == selected.revision
    assert "PRIVATE_CLEANUP_ERROR" not in repr(report)


async def test_saved_entry_survives_client_failure_and_retry_converges_without_new_change(
    workspace: tuple[ExtractionOptions, SourceSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    options, selected = workspace
    first_wire = ScriptedWire(
        [
            response(call("memory_search", {"query": ""})),
            response(call("memory_save", save_arguments())),
            RuntimeError("PRIVATE_NEXT_REQUEST_ERROR"),
        ]
    )
    monkeypatch.setattr(executor_icode, "create_client", ClientFactory(first_wire))
    first = await run_extraction(options, profile(), readers={"claude-code": Reader(selected)})
    assert (first.status, first.exit_code) == ("failed", 1)
    store = FileMemoryStore(options.memory_dir)
    saved = await store.list_entries(selected.project)
    assert len(saved) == 1
    record = (await core.WatermarkStore(options.memory_dir).load()).record(
        str(selected.project.project_id), selected.key
    )
    assert record.updated is None and record.attempts == 1
    log_path = options.memory_dir / "projects" / str(selected.project.project_id) / "changes.jsonl"
    first_log = await asyncio.to_thread(log_path.read_text, encoding="utf-8")
    assert json.loads(first_log)["source"] == first.run_id
    second_wire = ScriptedWire(
        [
            response(call("memory_search", {"query": ""})),
            response(call("memory_save", save_arguments())),
            response(text="Done"),
        ]
    )
    monkeypatch.setattr(executor_icode, "create_client", ClientFactory(second_wire))
    second = await run_extraction(options, profile(), readers={"claude-code": Reader(selected)})
    assert second.exit_code == 0
    assert second.batches[0]["existing_count"] == 1
    assert first.run_id != second.run_id
    assert await store.list_entries(selected.project) == saved
    assert await asyncio.to_thread(log_path.read_text, encoding="utf-8") == first_log
    assert first_wire.closed == second_wire.closed == 1
    assert "PRIVATE_NEXT_REQUEST_ERROR" not in repr(first)


async def test_save_thread_cancel_settles_log_and_client_before_releasing_run_lock(
    workspace: tuple[ExtractionOptions, SourceSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from chrys.service.memory.storage import file_store
    from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for

    options, selected = workspace
    entered, release = threading.Event(), threading.Event()
    original = file_store.atomic_write_owner_only_bytes
    wire = ScriptedWire(
        [
            response(call("memory_search", {"query": ""})),
            response(call("memory_save", save_arguments())),
            response(text="Done"),
        ]
    )
    monkeypatch.setattr(executor_icode, "create_client", ClientFactory(wire))

    def blocked_save(path: Path, payload: bytes, *, create_parents: bool = True) -> None:
        if path.suffix == ".md":
            entered.set()
            if not release.wait(ENGINE_TURN_TIMEOUT):
                raise TimeoutError("test save gate was not released")
        original(path, payload, create_parents=create_parents)

    monkeypatch.setattr(file_store, "atomic_write_owner_only_bytes", blocked_save)
    task = asyncio.create_task(run_extraction(options, profile(), readers={"claude-code": Reader(selected)}))
    try:
        await wait_for(lambda: entered.is_set() or task.done(), timeout=ENGINE_TURN_TIMEOUT)
        if task.done():
            await task
        assert entered.is_set() and not task.done()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        assert not task.done() and wire.closed == 0
        competing = await run_extraction(
            options, profile(), readers={"claude-code": Reader(selected)}, executor=Completed()
        )
        assert competing.status == "already_running"
        release.set()
        report = await task
        assert (report.status, report.exit_code, report.cancel_requested) == ("saved_then_cancelled", 130, True)
        assert wire.closed == 1
        assert report.batches[0]["created_count"] == 1
        assert report.batches[0]["unsettled_count"] == 0
        saved = await FileMemoryStore(options.memory_dir).list_entries(selected.project)
        assert len(saved) == 1
        log_path = options.memory_dir / "projects" / str(selected.project.project_id) / "changes.jsonl"
        log_text = await asyncio.to_thread(log_path.read_text, encoding="utf-8")
        assert json.loads(log_text)["source"] == report.run_id
        record = (await core.WatermarkStore(options.memory_dir).load()).record(
            str(selected.project.project_id), selected.key
        )
        assert record.updated is None and record.attempts == 1
    finally:
        release.set()
        await task


@pytest.mark.parametrize(
    "case,expected_code,candidates,saves",
    [
        ("candidate_abandoned", 0, 1, 0),
        ("candidate_retry", 0, 1, 1),
        ("not_found", 1, 0, 0),
        ("mixed_ambiguity", 1, 0, 0),
        ("candidate_expired", 1, 0, 0),
        ("store_unavailable", 1, 0, 0),
        ("unexpected_exception", 1, 0, 0),
        ("invalid_type", 1, 0, 0),
    ],
)
async def test_candidate_and_business_failures_have_exact_final_progress(
    workspace: tuple[ExtractionOptions, SourceSession],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected_code: int,
    candidates: int,
    saves: int,
) -> None:
    from chrys.service.memory.storage.store import CreateResult, MemoryEntryDraft, MemoryOperationError
    from chrys.service.memory.verification import (
        ConversationView,
        QuoteClaim,
        SourceMessage,
        SourceVerificationError,
        verify_quotes,
    )

    options, original_source = workspace
    messages = (SourceMessage(SESSION, 1, (WORDS,), item="one"), SourceMessage(SESSION, 2, (WORDS,), item="two"))
    selected = replace(original_source, messages=messages)
    args = save_arguments()
    if case in {"store_unavailable", "unexpected_exception"}:
        selected = replace(selected, messages=messages[:1])

        async def fail_create(store: FileMemoryStore, scope: MemoryScope, draft: MemoryEntryDraft) -> CreateResult:
            if case == "store_unavailable":
                raise MemoryOperationError("memory_store_unavailable", "Unavailable.")
            raise RuntimeError("PRIVATE_STORE_FAILURE")

        monkeypatch.setattr(FileMemoryStore, "create", fail_create)
    elif case == "not_found":
        args["user_quotes"] = [{"text": "absent quote", "session": SESSION}]
    elif case == "mixed_ambiguity":
        args["user_quotes"] = [{"text": WORDS, "session": SESSION}, {"text": "absent quote", "session": SESSION}]
    elif case == "candidate_expired":
        args["user_quotes"] = [{"text": WORDS, "session": SESSION, "candidate": "q-expired"}]
    elif case == "invalid_type":
        args["type"] = "engineering"
    script = [response(call("memory_search", {"query": ""})), response(call("memory_save", args))]
    if case == "candidate_retry":
        with pytest.raises(SourceVerificationError) as caught:
            verify_quotes(
                (QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", WORDS, session=SESSION),),
                ConversationView(messages, None),
            )
        token = caught.value.failures[0].candidates[0].token
        script.append(
            response(
                call(
                    "memory_save", save_arguments(user_quotes=[{"text": WORDS, "session": SESSION, "candidate": token}])
                )
            )
        )
    script.append(response(text="Done"))
    wire = ScriptedWire(script)
    monkeypatch.setattr(executor_icode, "create_client", ClientFactory(wire))
    report = await run_extraction(
        options, profile(), readers={"claude-code": Reader(selected)}, material_profile=core.USER_V2
    )
    assert report.exit_code == expected_code
    assert report.batches[0]["candidate_count"] == candidates
    assert report.batches[0]["created_count"] == saves
    assert (report.batches[0]["failure_count"] == 0) is (expected_code == 0)
    record = (await core.WatermarkStore(options.memory_dir).load()).record(
        str(selected.project.project_id), selected.key
    )
    assert (record.updated == selected.revision) is (expected_code == 0)
    assert record.attempts == (0 if expected_code == 0 else 1)
    assert len(await FileMemoryStore(options.memory_dir).list_entries(selected.project)) == saves
    assert "PRIVATE_STORE_FAILURE" not in repr(report)
