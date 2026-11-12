# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Versioned attempts and durable commit outcomes from the step-5 state tables."""

from __future__ import annotations

import asyncio
import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from chrys.foundation.platform.files import atomic_write_owner_only_bytes, secure_open_owner_only_binary
from chrys.service.memory import extraction
from chrys.service.memory.extraction import (
    AttemptTicket,
    ExtractionError,
    ProjectRunLock,
    SessionKey,
    SourceSession,
    WatermarkRecord,
    WatermarkSnapshot,
    WatermarkStore,
    candidate_status,
    revision_for_bytes,
)
from chrys.service.memory.storage.store import MemoryScope
from chrys.service.memory.verification import SourceMessage
from tests.support.secure_files import plant_owner_only_bytes

PROJECT = MemoryScope("project", "0123456789abcdef")
OTHER = MemoryScope("project", "fedcba9876543210")


def source(
    index: int = 1,
    *,
    project: MemoryScope = PROJECT,
    host: str = "claude-code",
    updated: int | None = None,
    content: bytes | None = None,
) -> SourceSession:
    session = f"00000000-0000-4000-8000-{index:012d}"
    revision = revision_for_bytes(content or f"synthetic-{index}".encode(), index if updated is None else updated)
    return SourceSession(
        SessionKey(host, session), project, revision, (SourceMessage(session, 1, ("A lasting rule.",), item="u"),)
    )


def record_json(record: WatermarkRecord) -> dict[str, object]:
    return {
        "updated": None
        if record.updated is None
        else {"updated_ns": record.updated.updated_ns, "sha256": record.updated.sha256},
        "attempted_updated": None
        if record.attempted_updated is None
        else {"updated_ns": record.attempted_updated.updated_ns, "sha256": record.attempted_updated.sha256},
        "attempts": record.attempts,
        "clean_limit_streak": record.clean_limit_streak,
        "batch_session_cap": record.batch_session_cap,
    }


def ticket(result: extraction.CommitResult) -> AttemptTicket:
    assert result.outcome == "committed"
    assert result.ticket is not None
    return result.ticket


async def test_missing_state_read_is_read_only_and_revision_includes_tail(tmp_path: Path) -> None:
    root = tmp_path / "not-created"
    assert await WatermarkStore(root).load() == WatermarkSnapshot()
    assert not root.exists()
    assert revision_for_bytes(b"user\nmetadata\n", 10).sha256 != revision_for_bytes(b"user\n", 10).sha256


async def test_attempt_success_and_processed_content_touch(tmp_path: Path) -> None:
    store = WatermarkStore(tmp_path / "memory")
    selected = source()
    before = await store.load()
    assert candidate_status(selected, before.record(PROJECT.project_id, selected.key)) == "candidate"
    receipt = ticket(await store.prepare(PROJECT, (selected,)))
    prepared = (await store.load()).record(PROJECT.project_id, selected.key)
    assert prepared == WatermarkRecord(None, selected.revision, 1, 0, 10)
    assert await store.succeed(receipt) == extraction.CommitResult("committed", "success", receipt)
    completed = (await store.load()).record(PROJECT.project_id, selected.key)
    assert completed == WatermarkRecord(selected.revision, selected.revision, 0, 0, 10)
    assert (
        candidate_status(replace(selected, revision=replace(selected.revision, updated_ns=9000)), completed)
        == "already_processed"
    )
    with secure_open_owner_only_binary(store.path) as stream:
        document = json.loads(stream.read())
    assert document == {
        "version": 1,
        "projects": {PROJECT.project_id: {selected.key.host: {selected.key.session: record_json(completed)}}},
    }
    if os.name != "nt":
        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


async def test_three_failures_pause_same_content_new_version_resets(tmp_path: Path) -> None:
    store = WatermarkStore(tmp_path / "memory")
    selected = source()
    for expected in (1, 2, 3):
        ticket(await store.prepare(PROJECT, (selected,)))
        assert (await store.load()).record(PROJECT.project_id, selected.key).attempts == expected
    record = (await store.load()).record(PROJECT.project_id, selected.key)
    touched = replace(selected, revision=replace(selected.revision, updated_ns=100))
    assert candidate_status(touched, record) == "paused"
    assert (await store.prepare(PROJECT, (touched,))).reason == "version_changed"
    changed = source(content=b"new content", updated=200)
    assert candidate_status(changed, record) == "candidate"
    ticket(await store.prepare(PROJECT, (changed,)))
    assert (await store.load()).record(PROJECT.project_id, changed.key) == WatermarkRecord(None, changed.revision, 1)
    assert candidate_status(source(content=b"rewritten", updated=0), record) == "source_rewritten"


async def test_content_rewrite_never_replaces_a_more_recent_success(tmp_path: Path) -> None:
    store = WatermarkStore(tmp_path / "memory")
    selected = source(updated=200)
    ticket_value = ticket(await store.prepare(PROJECT, (selected,)))
    assert (await store.succeed(ticket_value)).outcome == "committed"
    old = source(content=b"old different content", updated=100)
    assert (await store.prepare(PROJECT, (old,))).outcome == "not_committed"
    assert (await store.load()).record(PROJECT.project_id, selected.key).updated == selected.revision


async def test_clean_limits_revoke_only_own_increment_and_reduce_next_batch(tmp_path: Path) -> None:
    store = WatermarkStore(tmp_path / "memory")
    selected = tuple(source(index) for index in range(1, 7))
    # One prior real failure must survive both subsequent clean limits.
    ticket(await store.prepare(PROJECT, selected))
    first = ticket(await store.prepare(PROJECT, selected))
    result = await store.limited(first, "turn_limited")
    assert result.outcome == "committed" and result.splits == ()
    assert (await store.load()).record(PROJECT.project_id, selected[0].key) == WatermarkRecord(
        None, selected[0].revision, 1, 1, 10
    )
    second = ticket(await store.prepare(PROJECT, selected))
    assert (await store.load()).record(PROJECT.project_id, selected[0].key).clean_limit_streak == 0
    result = await store.limited(second, "output_limited")
    assert len(result.splits) == 6
    assert all(
        split.old_cap == 10 and split.new_cap == 3 and split.reason == "output_limited" for split in result.splits
    )
    assert (await store.load()).record(PROJECT.project_id, selected[0].key) == WatermarkRecord(
        None, selected[0].revision, 1, 0, 3
    )
    caps = {selected[0].key: 3}
    groups = extraction.select_members(selected, caps)
    assert all(len(group) <= 3 for group in groups if selected[0] in group)


async def test_failure_or_kill_after_prepare_breaks_streak_without_clearing_cap(tmp_path: Path) -> None:
    store = WatermarkStore(tmp_path / "memory")
    selected = tuple(source(index) for index in range(1, 5))
    first = ticket(await store.prepare(PROJECT, selected))
    await store.limited(first, "turn_limited")
    ticket(await store.prepare(PROJECT, selected))  # A failure or process kill has no finalization.
    third = ticket(await store.prepare(PROJECT, selected))
    limited = await store.limited(third, "turn_limited")
    assert limited.splits == ()
    assert (await store.load()).record(PROJECT.project_id, selected[0].key).clean_limit_streak == 1
    fourth = ticket(await store.prepare(PROJECT, selected))
    await store.limited(fourth, "turn_limited")
    ticket(await store.prepare(PROJECT, selected))
    assert (await store.load()).record(PROJECT.project_id, selected[0].key).batch_session_cap == 2


async def test_single_session_limits_report_no_split_and_success_resets(tmp_path: Path) -> None:
    store = WatermarkStore(tmp_path / "memory")
    selected = source()
    for _ in (1, 2):
        result = await store.limited(ticket(await store.prepare(PROJECT, (selected,))), "turn_limited")
    assert result.splits[0].reason == "split_not_possible_single_session"
    assert result.splits[0].new_cap == 1
    assert (await store.load()).record(PROJECT.project_id, selected.key) == WatermarkRecord(
        None, selected.revision, 0, 0, 1
    )
    receipt = ticket(await store.prepare(PROJECT, (selected,)))
    await store.succeed(receipt)
    assert (await store.load()).record(PROJECT.project_id, selected.key) == WatermarkRecord(
        selected.revision, selected.revision
    )


async def test_later_count_or_version_is_not_overwritten_by_stale_finalize(tmp_path: Path) -> None:
    store = WatermarkStore(tmp_path / "memory")
    first = ticket(await store.prepare(PROJECT, (source(),)))
    second = ticket(await store.prepare(PROJECT, (source(),)))
    assert (await store.succeed(first)).reason == "version_changed"
    assert (await store.limited(first, "turn_limited")).reason == "version_changed"
    assert (await store.load()).record(PROJECT.project_id, source().key).attempts == 2
    third = ticket(await store.prepare(PROJECT, (source(content=b"new", updated=9),)))
    assert (await store.succeed(second)).reason == "version_changed"
    assert (await store.succeed(third)).outcome == "committed"


async def test_project_host_isolation_and_concurrent_locked_merges(tmp_path: Path) -> None:
    store = WatermarkStore(tmp_path / "memory")
    sources = (source(), source(host="codex"), source(project=OTHER))
    prepared = await asyncio.gather(*(store.prepare(selected.project, (selected,)) for selected in sources))
    await asyncio.gather(*(store.succeed(ticket(result)) for result in prepared))
    snapshot = await store.load()
    assert len(snapshot.records) == 3
    assert all(
        snapshot.record(selected.project.project_id, selected.key).updated == selected.revision for selected in sources
    )


@pytest.mark.parametrize(
    "malformed",
    [
        b"not json",
        b"{}",
        b'{"version":2,"projects":{}}',
        b'{"version":1,"projects":[]}',
        b'{"version":1,"projects":{"bad":{}}}',
    ],
)
async def test_bad_watermark_is_fatal_never_treated_as_empty(tmp_path: Path, malformed: bytes) -> None:
    root = tmp_path / "memory"
    root.mkdir()
    plant_owner_only_bytes(root / "extraction.json", malformed)
    with pytest.raises(ExtractionError) as caught:
        await WatermarkStore(root).load()
    assert caught.value.kind == "watermark_invalid"


async def test_old_record_defaults_new_limit_fields_without_writing(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    root.mkdir()
    values = record_json(WatermarkRecord(attempted_updated=source().revision, attempts=1))
    del values["clean_limit_streak"], values["batch_session_cap"]
    payload = json.dumps(
        {"version": 1, "projects": {PROJECT.project_id: {"claude-code": {source().key.session: values}}}}
    ).encode()
    plant_owner_only_bytes(root / "extraction.json", payload)
    record = (await WatermarkStore(root).load()).record(PROJECT.project_id, source().key)
    assert record.clean_limit_streak == 0 and record.batch_session_cap == 10
    assert (root / "extraction.json").read_bytes() == payload
    assert not (root / "extraction.lock").exists()


@pytest.mark.parametrize("phase", ["attempt", "success", "incomplete_limits"])
@pytest.mark.parametrize("failure", ["before_publish", "after_publish", "confirmation_failed"])
async def test_each_commit_phase_confirms_persistent_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, failure: str
) -> None:
    root = tmp_path / "memory"
    store = WatermarkStore(root)
    selected = source()
    admitted: AttemptTicket | None = None
    if phase != "attempt":
        admitted = ticket(await store.prepare(PROJECT, (selected,)))
    before = await store.load()
    if not store.path.exists():
        root.mkdir(parents=True, exist_ok=True)
        plant_owner_only_bytes(store.path, b'{"version":1,"projects":{}}')

    def faulty_publish(path: Path, payload: bytes, *, create_parents: bool = True) -> None:
        if failure == "after_publish":
            atomic_write_owner_only_bytes(path, payload, create_parents=create_parents)
        elif failure == "confirmation_failed":
            atomic_write_owner_only_bytes(path, b"unreadable partial state", create_parents=create_parents)
        raise OSError("PRIVATE_EXCEPTION_SENTINEL")

    monkeypatch.setattr(extraction, "atomic_write_owner_only_bytes", faulty_publish)
    if phase == "attempt":
        result = await store.prepare(PROJECT, (selected,))
    else:
        assert admitted is not None
        result = await store.succeed(admitted) if phase == "success" else await store.limited(admitted, "turn_limited")
    expected = {
        "before_publish": "not_committed",
        "after_publish": "committed",
        "confirmation_failed": "commit_unknown",
    }[failure]
    assert result.outcome == expected
    assert result.phase == phase
    assert "PRIVATE_EXCEPTION_SENTINEL" not in repr(result)
    if failure == "before_publish":
        assert await store.load() == before
    elif failure == "after_publish":
        record = (await store.load()).record(PROJECT.project_id, selected.key)
        assert record.attempts == (1 if phase == "attempt" else 0)
        assert record.updated == (selected.revision if phase == "success" else None)


async def test_one_project_run_lock_covers_both_hosts_and_others_can_run(tmp_path: Path) -> None:
    first = ProjectRunLock(tmp_path, PROJECT)
    competing = ProjectRunLock(tmp_path, PROJECT)
    other = ProjectRunLock(tmp_path, OTHER)
    assert await first.try_acquire() is True
    try:
        assert await competing.try_acquire() is False
        assert await other.try_acquire() is True
        await other.release()
    finally:
        await first.release()
    assert await competing.try_acquire() is True
    await competing.release()


async def test_missing_confirmation_after_failed_first_publish_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed_publish(path: Path, payload: bytes, *, create_parents: bool = True) -> None:
        raise OSError("write failed")

    monkeypatch.setattr(extraction, "atomic_write_owner_only_bytes", failed_publish)
    result = await WatermarkStore(tmp_path / "memory").prepare(PROJECT, (source(),))
    assert result.outcome == "commit_unknown"
    assert result.reason == "confirmation_failed"
