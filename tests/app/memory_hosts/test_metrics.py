# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Deterministic, read-only memory evaluation from invented store and host records."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from chrys.foundation.platform.files import atomic_write_owner_only_bytes
from chrys.foundation.platform.process import windows_hidden_subprocess_kwargs
from chrys.service.memory.change_log import MemoryChange
from chrys.service.memory.project import bind_project
from chrys.service.memory.storage.file_change_log import FileChangeLog
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.layout import scope_directory
from chrys.service.memory.storage.store import GLOBAL_SCOPE, MemoryEntryDraft, MemoryScope, Quote, ReferenceEvidence
from tests.app.memory_hosts.history_fixtures import PARENT, SESSION, claude_assistant, claude_user, write_jsonl
from tests.support.waiting import ENGINE_TURN_TIMEOUT

AT = "2026-11-11T10:00:00Z"
REF_A = "project:111111111111"
REF_B = "global:222222222222"
REF_C = "project:333333333333"


def catalog(*refs: str) -> str:
    return (
        "<memory_catalog>\nSaved memories\n"
        + "".join(f"- {ref} intent — synthetic\n" for ref in refs)
        + "</memory_catalog>"
    )


def hook(refs: tuple[str, ...], *, session: str = SESSION, item: str = "catalog") -> dict[str, object]:
    return {
        "type": "attachment",
        "uuid": item,
        "parentUuid": "human",
        "sessionId": session,
        "attachment": {"type": "hook_additional_context", "hookName": "UserPromptSubmit", "content": [catalog(*refs)]},
    }


def call(item: str, parent: str, tool: str, arguments: object, *, call_id: str = "read") -> dict[str, object]:
    return claude_assistant(
        item,
        parent,
        message={
            "id": "api-" + item,
            "content": [{"type": "tool_use", "id": call_id, "name": tool, "input": arguments}],
        },
    )


def output(item: str, parent: str, text: str, *, call_id: str = "read") -> dict[str, object]:
    return {
        "type": "user",
        "uuid": item,
        "parentUuid": parent,
        "sessionId": SESSION,
        "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": text}]},
    }


def codex_meta(project: Path) -> dict[str, object]:
    return {
        "type": "session_meta",
        "payload": {"id": SESSION, "cwd": str(project), "source": "vscode", "originator": "Codex Desktop"},
    }


def codex_call(name: str, arguments: object, call_id: str) -> dict[str, object]:
    return {
        "type": "response_item",
        "payload": {"type": "function_call", "name": name, "arguments": json.dumps(arguments), "call_id": call_id},
    }


def codex_output(text: object, call_id: str) -> dict[str, object]:
    return {"type": "response_item", "payload": {"type": "function_call_output", "output": text, "call_id": call_id}}


async def test_store_scan_reads_every_numbered_archive_and_skips_invalid_metadata(tmp_path: Path) -> None:
    from chrys.app.memory_hosts import evaluation

    root = tmp_path / "memory"
    project = bind_project(str(tmp_path))
    log = FileChangeLog(root, max_bytes=1)
    for index, operation in enumerate(("create", "update", "withdraw", "restore", "merge", "delete")):
        await log.append(MemoryChange(AT, project, REF_A, "intent", operation, "extraction", "run", None, str(index)))
    await log.append(MemoryChange(AT, GLOBAL_SCOPE, REF_B, "engineering", "create", "model", "session", None, "one"))
    directory = scope_directory(root, project)
    await asyncio.to_thread(
        atomic_write_owner_only_bytes, directory / "changes.9.jsonl", b'not-json\n{"body":"never counted"}\n'
    )
    await asyncio.to_thread(
        atomic_write_owner_only_bytes, directory / "changes.fake.jsonl", b'{"operation":"create"}\n'
    )
    snapshot = await evaluation.scan_store(root)
    assert [change.operation for change in snapshot.changes].count("create") == 2
    assert len(snapshot.changes) == 7
    assert len([note for note in snapshot.diagnostics if note.kind == "invalid_change_record"]) == 2
    assert {note.file for note in snapshot.diagnostics} == {"changes.9.jsonl"}


async def test_metrics_groups_changes_and_current_entries_without_creating_files(tmp_path: Path) -> None:
    from chrys.app.memory_hosts import metrics

    project_dir = tmp_path / "project"
    await asyncio.to_thread(project_dir.mkdir)
    project = bind_project(str(project_dir))
    root = tmp_path / "memory"
    store = FileMemoryStore(root)
    await store.create(
        project, MemoryEntryDraft("intent", "Synthetic", "body", "model", quotes=(Quote("words", "message", "old"),))
    )
    await store.create(
        GLOBAL_SCOPE,
        MemoryEntryDraft(
            "engineering",
            "Withdrawn",
            "body",
            "extraction",
            evidence=(ReferenceEvidence("synthetic"),),
            status="withdrawn",
            status_basis=Quote("not applicable", "message", "old", AT),
        ),
    )
    log = FileChangeLog(root)
    await log.append(MemoryChange(AT, project, REF_A, "intent", "restore", "extraction", "run", "before", "after"))
    before = await asyncio.to_thread(
        lambda: {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    )
    report = await metrics.run_metrics(metrics.MetricsOptions(project_dir, root))
    after = await asyncio.to_thread(
        lambda: {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    )
    assert before == after
    assert report.operations["restore"]["extraction"] == 1
    assert sum(sum(writers.values()) for writers in report.operations.values()) == 1
    assert [asdict(item) for item in report.entries] == [
        {"type": "engineering", "status": "withdrawn", "written_by": "extraction", "count": 1},
        {"type": "intent", "status": "active", "written_by": "model", "count": 1},
    ]
    assert report.unavailable == {
        "used_after_read": "unavailable",
        "corrections": "unavailable",
        "repeated_instructions": "unavailable",
    }
    assert report.rejection_count == "unavailable" and report.rejection_kinds == "unavailable"


async def test_metrics_reads_rejections_from_all_captured_batches_without_writing(tmp_path: Path) -> None:
    from chrys.app.memory_hosts import metrics

    root = tmp_path / "memory"
    captured = {
        "batches": [
            {
                "status": "completed",
                "rejection_count": 2,
                "rejection_kinds": [["memory_read:memory_invalid_input", 2]],
            },
            {
                "status": "failed",
                "failure_count": 1,
                "rejection_count": 1,
                "rejection_kinds": [["memory_read:memory_invalid_input", 1]],
            },
            {
                "status": "completed",
                "rejection_count": 1,
                "rejection_kinds": [["memory_save:memory_evidence_not_found", 1]],
            },
        ]
    }
    await asyncio.to_thread(atomic_write_owner_only_bytes, root / "report.json", json.dumps(captured).encode("utf-8"))
    before = await asyncio.to_thread((root / "report.json").read_bytes)
    report = await metrics.run_metrics(metrics.MetricsOptions(tmp_path, root))
    assert report.rejection_count == 4
    assert report.rejection_kinds == {
        "memory_read:memory_invalid_input": 3,
        "memory_save:memory_evidence_not_found": 1,
    }
    assert await asyncio.to_thread((root / "report.json").read_bytes) == before
    assert await asyncio.to_thread(lambda: sorted(path.name for path in root.iterdir())) == ["report.json"]


@pytest.mark.parametrize(
    "captured", [{}, {"batches": []}, {"batches": [{}]}, {"batches": [{"rejection_count": 2}, {}]}]
)
async def test_metrics_old_or_partial_reports_leave_rejections_unavailable(tmp_path: Path, captured: object) -> None:
    from chrys.app.memory_hosts import metrics

    root = tmp_path / "memory"
    await asyncio.to_thread(atomic_write_owner_only_bytes, root / "report.json", json.dumps(captured).encode("utf-8"))
    report = await metrics.run_metrics(metrics.MetricsOptions(tmp_path, root))
    assert report.rejection_count == "unavailable" and report.rejection_kinds == "unavailable"


async def test_metrics_observed_zero_rejections_is_not_unavailable(tmp_path: Path) -> None:
    from chrys.app.memory_hosts import metrics

    root = tmp_path / "memory"
    await asyncio.to_thread(
        atomic_write_owner_only_bytes,
        root / "report.json",
        b'{"batches":[{"rejection_count":0,"rejection_kinds":[]}]}',
    )
    report = await metrics.run_metrics(metrics.MetricsOptions(tmp_path, root))
    assert report.rejection_count == 0 and report.rejection_kinds == {}


async def test_metrics_invalid_captured_report_keeps_store_counts_and_reports_diagnostic(tmp_path: Path) -> None:
    from chrys.app.memory_hosts import metrics

    root = tmp_path / "memory"
    project = bind_project(str(tmp_path))
    await FileMemoryStore(root).create(
        project, MemoryEntryDraft("intent", "Synthetic", "body", "model", quotes=(Quote("words", "message", "s"),))
    )
    await asyncio.to_thread(atomic_write_owner_only_bytes, root / "report.json", b"invalid private report contents")
    report = await metrics.run_metrics(metrics.MetricsOptions(tmp_path, root))
    assert sum(item.count for item in report.entries) == 1
    assert report.rejection_count == "unavailable" and report.rejection_kinds == "unavailable"
    assert any(note.kind == "invalid_extraction_report" and note.file == "report.json" for note in report.diagnostics)
    assert "private report contents" not in json.dumps(asdict(report))


async def test_shared_scan_preserves_scope_identity_and_can_limit_to_current_project(tmp_path: Path) -> None:
    from chrys.app.memory_hosts import evaluation

    root = tmp_path / "memory"
    first = MemoryScope("project", "1111111111111111")
    other = MemoryScope("project", "2222222222222222")
    store = FileMemoryStore(root)
    draft = MemoryEntryDraft("intent", "Synthetic", "body", "model", quotes=(Quote("words", "message", "old"),))
    await store.create(first, draft)
    await store.create(other, draft)
    await store.create(GLOBAL_SCOPE, draft)
    assert {entry.scope for entry in (await evaluation.scan_store(root)).entries} == {first, other, GLOBAL_SCOPE}
    assert {entry.scope for entry in (await evaluation.scan_store(root, first)).entries} == {first, GLOBAL_SCOPE}
    absent = tmp_path / "absent"
    assert (await evaluation.scan_store(absent)).entries == ()
    assert not absent.exists()


async def test_claude_catalog_and_read_search_events_count_unique_observed_refs_per_session(tmp_path: Path) -> None:
    from chrys.app.memory_hosts import metrics

    project = tmp_path / "project"
    source = tmp_path / "claude"
    await asyncio.to_thread(project.mkdir)
    records = [
        claude_user("human", "Synthetic request", cwd=str(project)),
        hook((REF_A, REF_B)),
        hook((REF_A,), item="catalog-again"),
        call("read-call", "human", "mcp__memory__memory_read", {"ref": REF_A}),
        output("read-result", "read-call", "Read " + REF_A),
        call("search-call", "read-result", "memory_search", {"query": REF_C}, call_id="search"),
        output("search-result", "search-call", "Found " + REF_B, call_id="search"),
        call("non-memory", "human", "Read", {"path": REF_C}, call_id="file"),
        output("file-result", "non-memory", REF_C, call_id="file"),
    ]
    await asyncio.to_thread(write_jsonl, source / f"{SESSION}.jsonl", records)
    report = await metrics.run_metrics(metrics.MetricsOptions(project, tmp_path / "memory", claude_source=source))
    assert [asdict(item) for item in report.sessions] == [
        {
            "host": "claude-code",
            "session": SESSION,
            "catalog_shown_refs": (REF_B, REF_A),
            "read_refs": (REF_A,),
            "search_refs": (REF_B, REF_C),
            "shown": 2,
            "read": 1,
            "hits": 1,
            "hit_rate": 0.5,
        }
    ]
    assert asdict(report.totals) == {"shown": 2, "read": 1, "hits": 1, "hit_rate": 0.5}


async def test_claude_sidechains_inherited_records_and_unpaired_results_do_not_count(tmp_path: Path) -> None:
    from chrys.app.memory_hosts import metrics

    source = tmp_path / "claude"
    records = [
        claude_user("human", "Synthetic", cwd=str(tmp_path)),
        hook((REF_A,)),
        {**hook((REF_B,), item="side"), "isSidechain": True},
        hook((REF_C,), item="inherited", session=PARENT),
        output("unpaired", "human", REF_A),
        {**call("side-call", "human", "memory_read", {"ref": REF_B}), "agentId": "subagent"},
        output("side-result", "side-call", REF_B),
    ]
    await asyncio.to_thread(write_jsonl, source / f"{SESSION}.jsonl", records)
    report = await metrics.run_metrics(metrics.MetricsOptions(tmp_path, tmp_path / "memory", claude_source=source))
    assert asdict(report.totals) == {"shown": 1, "read": 0, "hits": 0, "hit_rate": 0.0}


async def test_codex_memory_catalog_and_same_turn_read_search_results(tmp_path: Path) -> None:
    from chrys.app.memory_hosts import metrics

    source = tmp_path / "codex"
    records = [
        codex_meta(tmp_path),
        {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "first"}},
        codex_call("memory_catalog", {}, "catalog"),
        codex_output(catalog(REF_A, REF_B), "catalog"),
        codex_call("memory_read", {"ref": REF_A}, "read"),
        codex_output("Read " + REF_A, "read"),
        codex_call("memory_search", {"query": "synthetic"}, "search"),
        codex_output([{"type": "text", "text": REF_C}], "search"),
        codex_call("memory_read", {"ref": REF_C}, "orphan"),
        {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "second"}},
        codex_output("Old result " + REF_B, "orphan"),
    ]
    await asyncio.to_thread(write_jsonl, source / f"rollout-{SESSION}.jsonl", records)
    report = await metrics.run_metrics(metrics.MetricsOptions(tmp_path, tmp_path / "memory", codex_source=source))
    assert asdict(report.totals) == {"shown": 2, "read": 2, "hits": 1, "hit_rate": 0.5}


async def test_metrics_totals_sum_session_intersections_instead_of_global_deduplication(tmp_path: Path) -> None:
    from chrys.app.memory_hosts import metrics

    source = tmp_path / "claude"
    for session, refs, read_ref in ((SESSION, (REF_A, REF_B), REF_A), (PARENT, (REF_A,), REF_C)):
        records = [claude_user("human", "Synthetic", sessionId=session, cwd=str(tmp_path)), hook(refs, session=session)]
        call_record = call("read-call", "human", "memory_read", {"ref": read_ref})
        call_record["sessionId"] = session
        records.append(call_record)
        await asyncio.to_thread(write_jsonl, source / f"{session}.jsonl", records)
    report = await metrics.run_metrics(metrics.MetricsOptions(tmp_path, tmp_path / "memory", claude_source=source))
    assert asdict(report.totals) == {"shown": 3, "read": 2, "hits": 1, "hit_rate": 1 / 3}


async def test_metrics_unknown_or_other_project_sources_are_diagnostic_and_content_is_absent(tmp_path: Path) -> None:
    from chrys.app.memory_hosts import metrics

    source = tmp_path / "claude"
    await asyncio.to_thread(
        write_jsonl,
        source / f"{SESSION}.jsonl",
        [claude_user("human", "PRIVATE SENTINEL", cwd=str(tmp_path / "deleted")), hook((REF_A,))],
    )
    report = await metrics.run_metrics(metrics.MetricsOptions(tmp_path, tmp_path / "memory", claude_source=source))
    assert report.sessions == ()
    assert any(note.kind == "source_mapping_unconfirmed" for note in report.diagnostics)
    assert "PRIVATE SENTINEL" not in json.dumps(asdict(report))
    assert report.totals.hit_rate is None


async def test_search_alone_never_inflates_actual_read_hits(tmp_path: Path) -> None:
    from chrys.app.memory_hosts import metrics

    source = tmp_path / "claude"
    records = [
        claude_user("human", "Synthetic", cwd=str(tmp_path)),
        hook((REF_A,)),
        call("search", "human", "memory_search", {"query": REF_A}),
        output("search-result", "search", REF_A),
    ]
    await asyncio.to_thread(write_jsonl, source / f"{SESSION}.jsonl", records)
    report = await metrics.run_metrics(metrics.MetricsOptions(tmp_path, tmp_path / "memory", claude_source=source))
    assert report.sessions[0].catalog_shown_refs == (REF_A,)
    assert report.sessions[0].read_refs == ()
    assert report.sessions[0].search_refs == (REF_A,)
    assert asdict(report.totals) == {"shown": 1, "read": 0, "hits": 0, "hit_rate": 0.0}


async def test_duplicate_rollout_files_merge_observations_by_session_key(tmp_path: Path) -> None:
    from chrys.app.memory_hosts import metrics

    source = tmp_path / "codex"
    for name, refs in (("first", (REF_A,)), ("second", (REF_A, REF_B))):
        records = [
            codex_meta(tmp_path),
            codex_call("memory_catalog", {}, "catalog"),
            codex_output(catalog(*refs), "catalog"),
        ]
        await asyncio.to_thread(write_jsonl, source / f"{name}-{SESSION}.jsonl", records)
    report = await metrics.run_metrics(metrics.MetricsOptions(tmp_path, tmp_path / "memory", codex_source=source))
    assert len(report.sessions) == 1
    assert report.sessions[0].catalog_shown_refs == (REF_B, REF_A)
    assert report.totals.shown == 2


async def test_codex_subagents_and_mismatched_output_kind_are_excluded(tmp_path: Path) -> None:
    from chrys.app.memory_hosts import metrics

    source = tmp_path / "codex"
    records = [
        codex_meta(tmp_path),
        codex_call("memory_catalog", {}, "catalog"),
        {
            "type": "response_item",
            "payload": {"type": "custom_tool_call_output", "call_id": "catalog", "output": catalog(REF_A)},
        },
    ]
    await asyncio.to_thread(write_jsonl, source / f"main-{SESSION}.jsonl", records)
    subagent_meta = codex_meta(tmp_path)
    subagent_meta["payload"]["parent_thread_id"] = SESSION
    await asyncio.to_thread(
        write_jsonl,
        source / f"subagent-{SESSION}.jsonl",
        [subagent_meta, codex_call("memory_catalog", {}, "catalog"), codex_output(catalog(REF_B), "catalog")],
    )
    report = await metrics.run_metrics(metrics.MetricsOptions(tmp_path, tmp_path / "memory", codex_source=source))
    assert report.totals.shown == 0
    assert any(note.kind == "not_root" for note in report.diagnostics)


def test_metrics_real_module_is_model_free_and_uses_only_explicit_temp_roots(tmp_path: Path) -> None:
    from chrys.app.memory_hosts import metrics

    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    source = tmp_path / "claude"
    write_jsonl(source / f"{SESSION}.jsonl", [claude_user("human", "Synthetic", cwd=str(project)), hook((REF_A,))])
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    memory = tmp_path / "memory"
    repo = Path(metrics.__file__).resolve().parents[4]
    env = {
        **os.environ,
        "HOME": str(home),
        "USERPROFILE": str(home),
        "APPDATA": str(home / "appdata"),
        "XDG_CONFIG_HOME": str(home / "config"),
        "PYTHONPATH": str(repo / "src"),
        "CHRYS_MODEL_PROFILE": "not-a-model-profile",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "chrys.app.memory_hosts.metrics",
            "-C",
            str(project),
            "--memory-dir",
            str(memory),
            "--claude-source",
            str(source),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        cwd=project,
        env=env,
        timeout=ENGINE_TURN_TIMEOUT,
        **windows_hidden_subprocess_kwargs(),
    )
    assert result.returncode == 0, result.stderr.decode("utf-8")
    report = json.loads(result.stdout)
    assert report["sessions"][0]["catalog_shown_refs"] == [REF_A]
    assert report["totals"] == {"shown": 1, "read": 0, "hits": 0, "hit_rate": 0.0}
    assert report["unavailable"]["used_after_read"] == "unavailable"
    assert not memory.exists()
    assert not (home / ".claude").exists()
    assert not (home / ".codex").exists()
    after = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    assert before == after


async def test_invalid_longer_ref_is_not_truncated_into_a_catalog_hit(tmp_path: Path) -> None:
    from chrys.app.memory_hosts import metrics

    source = tmp_path / "claude"
    records = [
        claude_user("human", "Synthetic", cwd=str(tmp_path)),
        hook((REF_A,)),
        call("bad-read", "human", "memory_read", {"ref": REF_A + "a"}),
    ]
    await asyncio.to_thread(write_jsonl, source / f"{SESSION}.jsonl", records)
    report = await metrics.run_metrics(metrics.MetricsOptions(tmp_path, tmp_path / "memory", claude_source=source))
    assert report.sessions[0].read_refs == ()
    assert report.totals.hits == 0
