# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Native writer protocol and compression lifecycle, using only temporary sources."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import BinaryIO

import pytest

from chrys.app.memory_hosts import codex_history, evaluation, extraction_freeze, history
from chrys.app.memory_hosts.codex_history import CodexHistoryReader
from chrys.app.memory_hosts.extract import ExtractionOptions, run_extraction
from chrys.app.memory_hosts.extraction_replay import load_manifest
from chrys.service.memory.extraction import ScanDiagnostic, SessionKey, WatermarkStore
from chrys.service.memory.project import bind_project
from tests.app.memory_hosts.history_fixtures import PARENT, SESSION, FixedCodex, write_jsonl
from tests.app.memory_hosts.test_codex_history import message, meta
from tests.app.memory_hosts.test_extract import PROFILE, Executor, result
from tests.support.secure_files import plant_owner_only_bytes
from tests.support.symlinks import symlink_or_skip
from tests.support.waiting import DEFAULT_WAIT_TIMEOUT, ENGINE_TURN_TIMEOUT


@dataclass
class Workspace:
    project: Path
    source: Path
    locks: Path
    path: Path

    def reader(self, activity="active") -> CodexHistoryReader:
        return CodexHistoryReader(self.source, processes=FixedCodex(activity))

    def scan(self, reader: CodexHistoryReader):
        return asyncio.run(reader.scan(bind_project(str(self.project))))


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    source = tmp_path / "home" / "sessions"
    source.mkdir(parents=True)
    locks = source.parent / "thread-writer-locks"
    locks.mkdir()
    plant_owner_only_bytes(locks / ".coordination.lock", b"")
    path = source / f"rollout-2026-11-01T10-00-00-{SESSION}.jsonl"
    project = tmp_path / "project"
    project.mkdir()
    write_jsonl(path, [meta(project), message("Keep errors actionable.")])
    return Workspace(project, source, locks, path)


@pytest.fixture
def flock():
    return pytest.importorskip("fcntl")


def compress(path: Path) -> bytes:
    zstd = pytest.importorskip("compression.zstd")
    payload = path.read_bytes()
    plant_owner_only_bytes(Path(str(path) + ".zst"), zstd.compress(payload))
    path.unlink()
    return payload


@pytest.mark.parametrize("fallback", ["no-coordination", "wrong-name", "win32"])
@pytest.mark.parametrize("activity", ["active", "unknown"])
def test_fallback_retains_whole_host_diagnostic(workspace, monkeypatch, fallback, activity):
    if fallback == "no-coordination":
        (workspace.locks / ".coordination.lock").unlink()
    elif fallback == "wrong-name":
        workspace.source = workspace.source.rename(workspace.source.with_name("other"))
    else:
        shadow = ModuleType("sys")
        shadow.platform = "win32"
        monkeypatch.setattr(codex_history, "sys", shadow)
    reader = workspace.reader(activity)
    scan = workspace.scan(reader)
    assert scan.fatal is True and scan.sessions == ()
    assert scan.diagnostics == (ScanDiagnostic("codex_hosts_" + activity),)
    assert reader.processes.calls == 1


@pytest.mark.parametrize("directory", ["sessions", "archived_sessions"])
@pytest.mark.parametrize("stale", [False, True])
def test_idle_locks_read_without_host_check_and_leave_files_untouched(workspace, flock, directory, stale):
    workspace.source = workspace.source.rename(workspace.source.with_name(directory))
    if stale:
        plant_owner_only_bytes(workspace.locks / f"{SESSION}.lock", b"stale")
    before = {path.name: path.read_bytes() for path in workspace.locks.iterdir()}
    reader = workspace.reader()
    scan = workspace.scan(reader)
    assert scan.fatal is False and len(scan.sessions) == 1
    assert reader.processes.calls == 0
    assert {path.name: path.read_bytes() for path in workspace.locks.iterdir()} == before


def test_active_skips_only_one_session_without_reading_it(workspace, flock, monkeypatch):
    other = workspace.source / f"rollout-2026-11-01T10-00-00-{PARENT}.jsonl"
    write_jsonl(other, [meta(workspace.project, id=PARENT), message("Other session")])
    lock = workspace.locks / f"{SESSION}.lock"
    plant_owner_only_bytes(lock, b"")
    original = codex_history.read_rollout_snapshot

    def read(path):
        assert path != workspace.path, "An active rollout must never be read"
        return original(path)

    monkeypatch.setattr(codex_history, "read_rollout_snapshot", read)
    before = sorted(workspace.locks.iterdir())
    with lock.open("rb") as writer:
        flock.flock(writer, flock.LOCK_EX | flock.LOCK_NB)
        scan = workspace.scan(workspace.reader())
    assert scan.fatal is False and [s.key.session for s in scan.sessions] == [PARENT]
    assert ScanDiagnostic("active", SessionKey("codex", SESSION)) in scan.diagnostics
    assert sorted(workspace.locks.iterdir()) == before


def test_coordination_timeout_waits_once_for_three_sessions_and_next_scan_recovers(workspace, flock, monkeypatch):
    identities = (SESSION, PARENT, "33333333-3333-4333-8333-333333333333")
    for identity in identities[1:]:
        path = workspace.source / f"rollout-2026-11-01T10-00-00-{identity}.jsonl"
        write_jsonl(path, [meta(workspace.project, id=identity), message("Other session")])
    shadow = ModuleType("time")
    now = [0.0]
    waits = []
    probes = []
    original = codex_history._thread_activity

    def probe(directory, thread):
        probes.append(thread)
        return original(directory, thread)

    def sleep(delay):
        waits.append((probes[-1], delay))
        now[0] += delay

    shadow.monotonic = lambda: now[0]
    shadow.sleep = sleep
    monkeypatch.setattr(codex_history, "time", shadow)
    monkeypatch.setattr(codex_history, "_thread_activity", probe)
    before = {path.name: path.read_bytes() for path in workspace.locks.iterdir()}
    reader = workspace.reader()
    with (workspace.locks / ".coordination.lock").open("rb") as coordination:
        flock.flock(coordination, flock.LOCK_EX | flock.LOCK_NB)
        scan = workspace.scan(reader)
    assert scan.fatal is False and scan.sessions == ()
    assert scan.diagnostics == (
        ScanDiagnostic("source_files", count=3),
        *(ScanDiagnostic("unknown", SessionKey("codex", identity)) for identity in identities),
    )
    assert now[0] == pytest.approx(1.0)
    assert probes == [SESSION]
    assert waits and all(thread == SESSION and 0 < delay <= 0.05 for thread, delay in waits)
    assert {path.name: path.read_bytes() for path in workspace.locks.iterdir()} == before

    # The same reader must retry normally on the next scan after coordination is released.
    recovered = workspace.scan(reader)
    assert recovered.fatal is False
    assert tuple(source.key.session for source in recovered.sessions) == identities
    assert not any(note.kind == "unknown" for note in recovered.diagnostics)
    assert probes == [SESSION, *identities]
    assert now[0] == pytest.approx(1.0)
    assert {path.name: path.read_bytes() for path in workspace.locks.iterdir()} == before


def test_rollback_filename_is_skipped_with_tail_session_key(workspace, flock, monkeypatch):
    workspace.path.rename(workspace.path.with_name(f"rollout-2026-11-01T10-00-00-{SESSION}_{PARENT}.jsonl"))

    def reject_open_or_read(path):
        raise AssertionError("A thread with only a rollback version must not be opened")

    monkeypatch.setattr(codex_history, "secure_open_regular_binary", reject_open_or_read)
    monkeypatch.setattr(codex_history, "read_rollout_snapshot", reject_open_or_read)
    scan = workspace.scan(workspace.reader())
    assert scan.fatal is False and scan.sessions == ()
    assert scan.diagnostics == (
        ScanDiagnostic("source_files", count=1),
        ScanDiagnostic("rollout_reverted", SessionKey("codex", PARENT)),
    )


def test_probe_holds_exclusive_coordination_only_while_probing_thread(workspace, flock, monkeypatch):
    lock = workspace.locks / f"{SESSION}.lock"
    plant_owner_only_bytes(lock, b"")
    coordination_path = workspace.locks / ".coordination.lock"
    original_open = codex_history.secure_open_regular_binary
    original_read = codex_history.read_rollout_snapshot
    observed = []

    def probe_open(path: Path) -> BinaryIO:
        if path == lock:
            with original_open(coordination_path) as contender, pytest.raises(BlockingIOError):
                flock.flock(contender, flock.LOCK_SH | flock.LOCK_NB)
            observed.append("exclusive")
        return original_open(path)

    def read(path):
        with original_open(coordination_path) as contender:
            flock.flock(contender, flock.LOCK_EX | flock.LOCK_NB)
        observed.append("released")
        return original_read(path)

    monkeypatch.setattr(codex_history, "secure_open_regular_binary", probe_open)
    monkeypatch.setattr(codex_history, "read_rollout_snapshot", read)
    assert len(workspace.scan(workspace.reader()).sessions) == 1
    assert observed == ["exclusive", "released"]


@pytest.mark.parametrize("direction", ["compress", "decompress"])
def test_revalidation_resolves_representation_and_ignores_activity_and_mtime(workspace, flock, direction):
    payload = compress(workspace.path) if direction == "decompress" else workspace.path.read_bytes()
    reader = workspace.reader()
    source = workspace.scan(reader).sessions[0]
    assert reader.paths[source.key] == workspace.path
    if direction == "compress":
        compress(workspace.path)
    else:
        plant_owner_only_bytes(workspace.path, payload)
        Path(str(workspace.path) + ".zst").unlink()
    physical = workspace.path if direction == "decompress" else Path(str(workspace.path) + ".zst")
    os.utime(physical, ns=(1, 2))
    lock = workspace.locks / f"{SESSION}.lock"
    plant_owner_only_bytes(lock, b"")
    with lock.open("rb") as writer:
        flock.flock(writer, flock.LOCK_EX | flock.LOCK_NB)
        assert asyncio.run(reader.revalidate(source, require_idle=True)) == "source_reopened"
        assert asyncio.run(reader.revalidate(source, require_idle=False)) is None
    write_jsonl(workspace.path, [meta(workspace.project), message("Changed")])
    assert asyncio.run(reader.revalidate(source, require_idle=False)) == "source_changed"
    workspace.path.unlink()
    if physical.exists():
        physical.unlink()
    assert asyncio.run(reader.revalidate(source, require_idle=False)) == "source_unavailable"


@pytest.mark.parametrize("nested", [False, True])
def test_only_zstd_is_recognized_and_plain_wins_when_both_exist(workspace, nested):
    if nested:
        directory = workspace.source / "2026" / "11" / "01"
        directory.mkdir(parents=True)
        workspace.path = workspace.path.rename(directory / workspace.path.name)
    before = history.read_snapshot(workspace.path)
    payload = compress(workspace.path)
    reader = workspace.reader("inactive")
    scan = workspace.scan(reader)
    assert scan.fatal is False and len(scan.sessions) == 1
    assert scan.sessions[0].revision.sha256 == hashlib.sha256(payload).hexdigest() == before.revision.sha256
    assert history.read_rollout_snapshot(workspace.path).records == before.records
    compressed = Path(str(workspace.path) + ".zst")
    assert scan.sessions[0].revision.updated_ns == compressed.stat().st_mtime_ns
    assert history.read_content(compressed).payload == payload
    write_jsonl(workspace.path, [meta(workspace.project), message("Plain preferred")])
    scan = workspace.scan(reader)
    assert scan.sessions[0].messages[0].fragments == ("Plain preferred",)
    assert [note for note in scan.diagnostics if note.kind == "source_files"] == [
        ScanDiagnostic("source_files", count=1)
    ]


@pytest.mark.parametrize(
    "kind", ["zstd_decode_failed", "zstd_unavailable", "partial_line", "invalid_json", "source_line_too_large"]
)
def test_compressed_failures_are_per_session(workspace, monkeypatch, kind):
    zstd = pytest.importorskip("compression.zstd")
    payload = {"partial_line": b'{"half":', "invalid_json": b"broken\n"}.get(kind, b'{"long":"xxxxxxxxxxxxxxxx"}\n')
    compress(workspace.path)
    compressed = Path(str(workspace.path) + ".zst")
    compressed.unlink()
    plant_owner_only_bytes(compressed, b"not zstd" if kind == "zstd_decode_failed" else zstd.compress(payload))
    if kind == "zstd_unavailable":
        monkeypatch.setitem(sys.modules, "compression.zstd", None)
        import compression

        monkeypatch.delattr(compression, "zstd")
    if kind == "source_line_too_large":
        monkeypatch.setattr(history, "MAX_HISTORY_LINE_BYTES", 8)
    scan = workspace.scan(workspace.reader("inactive"))
    assert scan.fatal is False and scan.sessions == ()
    assert (
        ScanDiagnostic(
            kind,
            SessionKey("codex", SESSION),
            details=(("position", "1"),) if kind in {"invalid_json", "source_line_too_large"} else (),
        )
        in scan.diagnostics
    )


def test_compressed_size_limit_stops_at_limit_plus_one(workspace, monkeypatch):
    zstd = pytest.importorskip("compression.zstd")
    compress(workspace.path)
    compressed = Path(str(workspace.path) + ".zst")
    compressed.unlink()
    plant_owner_only_bytes(compressed, zstd.compress(b"x" * 10_000_000))
    monkeypatch.setattr(history, "MAX_HISTORY_BYTES", 1024)
    original = zstd.ZstdFile.read
    reads = []

    def bounded_read(stream, size=-1):
        assert 0 <= size <= 1025
        payload = original(stream, size)
        reads.append(len(payload))
        return payload

    monkeypatch.setattr(zstd.ZstdFile, "read", bounded_read)
    scan = workspace.scan(workspace.reader("inactive"))
    assert scan.fatal is False and scan.sessions == ()
    assert ScanDiagnostic("source_too_large", SessionKey("codex", SESSION)) in scan.diagnostics
    assert reads == [1025]


def test_successful_extraction_watermark_survives_compression(workspace):
    options = ExtractionOptions(workspace.project, workspace.project / "memory", codex_source=workspace.source)
    runner = Executor(result())
    reader = workspace.reader("inactive")
    first = asyncio.run(run_extraction(options, PROFILE, readers={"codex": reader}, executor=runner))
    assert (first.status, first.exit_code) == ("completed", 0)
    source = runner.batches[0].sources[0]
    stored = asyncio.run(WatermarkStore(options.memory_dir).load()).record(str(source.project.project_id), source.key)
    assert stored.updated == source.revision
    compress(workspace.path)
    second = asyncio.run(
        run_extraction(options, PROFILE, readers={"codex": workspace.reader("inactive")}, executor=runner)
    )
    assert (second.status, second.exit_code) == ("no_candidates", 0)
    assert len(runner.batches) == 1


@pytest.mark.parametrize("when", ["before-scan", "after-scan"])
def test_freeze_and_usage_read_compressed_logical_content(workspace, monkeypatch, when):
    # Include one memory tool observation to prove usage reads the actual records.
    write_jsonl(
        workspace.path,
        [
            meta(workspace.project),
            message("Keep errors actionable."),
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": "use-1",
                    "name": "memory_read",
                    "arguments": '{"ref":"project:123456abcdef"}',
                },
            },
        ],
    )
    payload = workspace.path.read_bytes()
    if when == "before-scan":
        compress(workspace.path)
    else:
        original = extraction_freeze._read_source

        def copy(source):
            compress(workspace.path)
            return original(source)

        monkeypatch.setattr(extraction_freeze, "_read_source", copy)
    options = ExtractionOptions(
        workspace.project,
        workspace.project / "memory",
        codex_source=workspace.source,
        dry_run=True,
        freeze_to=workspace.project.parent / "frozen",
    )
    report = asyncio.run(run_extraction(options, PROFILE, readers={"codex": workspace.reader("inactive")}))
    assert (report.status, report.exit_code) == ("dry_run", 0)
    manifest = load_manifest(options.freeze_to / "manifest.json")
    assert len(manifest.sources) == 1
    frozen = manifest.sources[0]
    assert frozen.path.suffix == ".jsonl"
    assert frozen.path.read_bytes() == payload
    assert frozen.sha256 == hashlib.sha256(payload).hexdigest()
    usage = asyncio.run(evaluation.scan_usage(bind_project(str(workspace.project)), codex_source=workspace.source))
    assert len(usage.sessions) == 1
    assert usage.sessions[0].key == SessionKey("codex", SESSION)
    assert usage.sessions[0].read == frozenset({"project:123456abcdef"})


@pytest.mark.parametrize("failure", ["thread_open", "coordination_open", "flock", "filename"])
def test_non_timeout_unknown_does_not_prevent_next_session_probe(workspace, flock, monkeypatch, failure):
    other = workspace.source / f"rollout-2026-11-01T10-00-00-{PARENT}.jsonl"
    write_jsonl(other, [meta(workspace.project, id=PARENT), message("Other session")])
    lock = workspace.locks / f"{SESSION}.lock"
    plant_owner_only_bytes(lock, b"")
    coordination_path = workspace.locks / ".coordination.lock"
    original_open = codex_history.secure_open_regular_binary
    original_flock = flock.flock
    coordination_opens = []
    failed = False

    if failure == "filename":
        workspace.path.rename(workspace.path.with_name(f"rollout-0000-invalid-{SESSION}.jsonl"))

    def open_file(path: Path) -> BinaryIO:
        nonlocal failed
        if path == coordination_path:
            coordination_opens.append(path)
            if failure == "coordination_open" and not failed:
                failed = True
                raise PermissionError("Uninspectable coordination lock")
        if path == lock and failure == "thread_open":
            raise PermissionError("Uninspectable writer lock")
        return original_open(path)

    def lock_file(fd, operation):
        nonlocal failed
        if failure == "flock" and not failed:
            failed = True
            raise OSError("Unsupported locking")
        return original_flock(fd, operation)

    monkeypatch.setattr(codex_history, "secure_open_regular_binary", open_file)
    monkeypatch.setattr(flock, "flock", lock_file)
    before = sorted(workspace.locks.iterdir())
    scan = workspace.scan(workspace.reader())
    assert scan.fatal is False
    assert tuple(source.key.session for source in scan.sessions) == (PARENT,)
    expected = "unknown_format" if failure == "filename" else "unknown"
    assert tuple(note for note in scan.diagnostics if note.kind in {"unknown", "unknown_format"}) == (
        ScanDiagnostic(expected, SessionKey("codex", SESSION)),
    )
    assert len(coordination_opens) == (1 if failure == "filename" else 2)
    assert sorted(workspace.locks.iterdir()) == before


@pytest.mark.parametrize("change", ["append", "replace"])
def test_compressed_read_checks_open_handle_and_final_path_identity(workspace, monkeypatch, change):
    payload = compress(workspace.path)
    compressed = Path(str(workspace.path) + ".zst")
    original = history._read_zstd

    def read(handle, limit):
        data = original(handle, limit)
        if change == "append":
            with compressed.open("ab") as writer:
                writer.write(b"changed")
        else:
            zstd = pytest.importorskip("compression.zstd")
            compressed.unlink()
            plant_owner_only_bytes(compressed, zstd.compress(payload))
        return data

    monkeypatch.setattr(history, "_read_zstd", read)
    scan = workspace.scan(workspace.reader("inactive"))
    assert scan.fatal is False and scan.sessions == ()
    assert ScanDiagnostic("source_changed", SessionKey("codex", SESSION)) in scan.diagnostics


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_reverted_thread_skips_all_versions_without_opening_locks_or_reading(
    workspace, flock, monkeypatch, compressed, selected
):
    if compressed:
        compress(workspace.path)
    reverted = workspace.path.with_name(f"rollout-2026-11-02T10-00-00-{SESSION}_{PARENT}.jsonl")
    write_jsonl(reverted, [meta(workspace.project), message("Reverted history")])
    unrelated = "33333333-3333-4333-8333-333333333333"
    other = workspace.path.with_name(f"rollout-2026-11-01T10-00-00-{unrelated}.jsonl")
    original_open = codex_history.secure_open_regular_binary
    original_read = codex_history.read_rollout_snapshot
    lock_opens = []
    reads = []
    before = sorted(workspace.locks.iterdir())

    def open_file(path: Path) -> BinaryIO:
        if path.parent == workspace.locks:
            assert other.exists(), "Reverted threads must not open any writer lock"
            assert path.name in {".coordination.lock", f"{unrelated}.lock"}
            lock_opens.append(path.name)
        return original_open(path)

    def read(path):
        assert path == other, "No version of a reverted thread may be read"
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(codex_history, "secure_open_regular_binary", open_file)
    monkeypatch.setattr(codex_history, "read_rollout_snapshot", read)
    notes = (
        ScanDiagnostic("rollout_reverted", SessionKey("codex", SESSION)),
        ScanDiagnostic("rollout_reverted", SessionKey("codex", PARENT)),
    )
    if selected:
        notes = notes[:1]
    reader = CodexHistoryReader(
        workspace.source, processes=FixedCodex("active"), sessions=frozenset({SESSION}) if selected else frozenset()
    )
    scan = workspace.scan(reader)
    assert scan.fatal is False and scan.sessions == ()
    assert scan.diagnostics == (ScanDiagnostic("source_files", count=2), *notes)
    assert reads == [] and lock_opens == []

    write_jsonl(other, [meta(workspace.project, id=unrelated), message("Independent thread")])
    scan = workspace.scan(reader)
    assert scan.fatal is False
    assert tuple(source.key.session for source in scan.sessions) == (() if selected else (unrelated,))
    assert tuple(note for note in scan.diagnostics if note.kind == "rollout_reverted") == notes
    assert reads == ([] if selected else [other])
    assert lock_opens == ([] if selected else [".coordination.lock", f"{unrelated}.lock"])
    assert sorted(workspace.locks.iterdir()) == before


def test_empty_native_sessions_directory_is_fatal_no_session_files(workspace, flock):
    workspace.path.unlink()
    reader = workspace.reader()
    scan = workspace.scan(reader)
    assert scan.fatal is True and scan.sessions == ()
    assert scan.diagnostics == (ScanDiagnostic("no_session_files"),)
    assert reader.processes.calls == 0


@pytest.mark.skipif(sys.platform == "win32", reason="Writer locks and FIFOs use POSIX APIs")
@pytest.mark.parametrize("lock_name", ["thread", "coordination"])
def test_writer_lock_fifo_is_unknown_without_blocking(workspace, flock, lock_name):
    path = workspace.locks / (f"{SESSION}.lock" if lock_name == "thread" else ".coordination.lock")
    if path.exists():
        path.unlink()
    os.mkfifo(path)
    before = sorted(workspace.locks.iterdir())
    with ThreadPoolExecutor(max_workers=1) as worker:
        probe = worker.submit(codex_history._thread_activity, workspace.locks, SESSION)
        try:
            assert probe.result(timeout=DEFAULT_WAIT_TIMEOUT) == "unknown"
        finally:
            if not probe.done():
                # Release a regressed blocking reader without writing, then reap the worker.
                peer = os.open(path, os.O_RDWR | os.O_NONBLOCK)
                try:
                    probe.result(timeout=ENGINE_TURN_TIMEOUT)
                finally:
                    os.close(peer)
    assert sorted(workspace.locks.iterdir()) == before


@pytest.mark.parametrize("lock_name", ["thread", "coordination"])
def test_writer_lock_symlink_to_regular_file_is_unknown(workspace, flock, lock_name):
    target = workspace.project / "lock-target"
    plant_owner_only_bytes(target, b"unchanged target")
    path = workspace.locks / (f"{SESSION}.lock" if lock_name == "thread" else ".coordination.lock")
    if path.exists():
        path.unlink()
    symlink_or_skip(path, target)
    before = sorted(workspace.locks.iterdir())
    assert codex_history._thread_activity(workspace.locks, SESSION) == "unknown"
    assert path.is_symlink() and target.read_bytes() == b"unchanged target"
    assert sorted(workspace.locks.iterdir()) == before


def test_missing_coordination_lock_is_unknown(workspace, flock):
    (workspace.locks / ".coordination.lock").unlink()
    assert codex_history._thread_activity(workspace.locks, SESSION) == "unknown"
    assert list(workspace.locks.iterdir()) == []


@pytest.mark.parametrize("require_idle", [False, True])
def test_revalidation_rechecks_cross_date_rollbacks_and_replay_bypasses_them(workspace, flock, require_idle):
    reader = workspace.reader()
    source = workspace.scan(reader).sessions[0]
    replay = CodexHistoryReader(workspace.source, processes=FixedCodex("active"), replay=True)
    frozen = asyncio.run(replay.read_files(source.project, (workspace.path,))).sessions[0]
    directory = workspace.source / "2026" / "11" / "02"
    directory.mkdir(parents=True)
    write_jsonl(directory / f"rollout-2026-11-02T10-00-00-{SESSION}_{PARENT}.jsonl", [meta(workspace.project)])
    lock = workspace.locks / f"{SESSION}.lock"
    plant_owner_only_bytes(lock, b"")
    with lock.open("rb") as writer:
        flock.flock(writer, flock.LOCK_EX | flock.LOCK_NB)
        assert reader._revalidate(source, require_idle=require_idle) == "rollout_reverted"
        assert replay._revalidate(frozen, require_idle=require_idle) is None
    assert reader.processes.calls == 0 and replay.processes.calls == 0


@pytest.mark.parametrize("require_idle", [False, True])
def test_revalidation_directory_failure_precedes_activity_and_content(workspace, flock, monkeypatch, require_idle):
    reader = workspace.reader()
    source = workspace.scan(reader).sessions[0]

    def unavailable(directory):
        raise OSError("Rollout discovery failed")

    def reject_open_or_read(path):
        raise AssertionError("A failed rollback check must stop before activity and content checks")

    monkeypatch.setattr(codex_history, "discover_rollouts", unavailable)
    monkeypatch.setattr(codex_history, "secure_open_regular_binary", reject_open_or_read)
    monkeypatch.setattr(codex_history, "read_rollout_content", reject_open_or_read)
    assert reader._revalidate(source, require_idle=require_idle) == "source_unavailable"


def test_revalidation_non_directory_source_has_no_rollbacks(workspace, flock):
    reader = workspace.reader()
    source = workspace.scan(reader).sessions[0]
    reader.source = workspace.project / "missing"
    assert reader._revalidate(source, require_idle=False) is None


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0, reason="Directory permissions do not deny listing here"
)
def test_unreadable_date_directory_fails_listing_instead_of_hiding_a_rollback(workspace, flock):
    reader = workspace.reader()
    source = workspace.scan(reader).sessions[0]
    hidden = workspace.source / "2026" / "11" / "02"
    hidden.mkdir(parents=True)
    write_jsonl(hidden / f"rollout-2026-11-02T10-00-00-{SESSION}_{PARENT}.jsonl", [meta(workspace.project)])
    hidden.chmod(0)
    try:
        with pytest.raises(PermissionError):
            history.discover_rollouts(workspace.source)
        assert reader._revalidate(source, require_idle=False) == "source_unavailable"
        scan = workspace.scan(workspace.reader())
        assert scan.fatal is True and scan.diagnostics == (ScanDiagnostic("source_unavailable"),)
        usage = asyncio.run(evaluation.scan_usage(bind_project(str(workspace.project)), codex_source=workspace.source))
        assert usage.sessions == ()
        assert usage.diagnostics == (evaluation.EvaluationDiagnostic("source_unavailable", "sessions"),)
    finally:
        hidden.chmod(0o700)
    assert reader._revalidate(source, require_idle=False) == "rollout_reverted"


@pytest.mark.parametrize("reverted_in", ["sessions", "archived_sessions"])
def test_revert_in_the_other_home_directory_excludes_the_thread(workspace, flock, reverted_in):
    archived = workspace.source.with_name("archived_sessions")
    archived.mkdir()
    if reverted_in == "sessions":
        workspace.path = workspace.path.rename(archived / workspace.path.name)
        target, other = archived, workspace.source
    else:
        target, other = workspace.source, archived
    reader = CodexHistoryReader(target, processes=FixedCodex("active"))
    source = workspace.scan(reader).sessions[0]
    revert = other / f"rollout-2026-11-02T10-00-00-{SESSION}_{PARENT}.jsonl"
    write_jsonl(revert, [meta(workspace.project)])
    assert reader._revalidate(source, require_idle=False) == "rollout_reverted"
    scan = workspace.scan(CodexHistoryReader(target, processes=FixedCodex("active")))
    assert scan.fatal is False and scan.sessions == ()
    assert ScanDiagnostic("rollout_reverted", SessionKey("codex", SESSION)) in scan.diagnostics

    real = history.discover_rollouts

    def sibling_fails(directory):
        if directory == other:
            raise OSError("Sibling listing failed")
        return real(directory)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(codex_history, "discover_rollouts", sibling_fails)
        assert reader._revalidate(source, require_idle=False) == "source_unavailable"
        scan = workspace.scan(CodexHistoryReader(target, processes=FixedCodex("active")))
        assert scan.fatal is True and scan.diagnostics == (ScanDiagnostic("source_unavailable"),)


@pytest.mark.parametrize("with_valid", [False, True])
def test_unrecognized_codex_file_names_fail_the_scan_only_when_none_match(workspace, flock, with_valid):
    changed = workspace.source / f"rollout-2026-11-01T10-00-00.250-{PARENT}.jsonl"
    write_jsonl(changed, [meta(workspace.project, id=PARENT), message("New grammar")])
    if not with_valid:
        workspace.path.unlink()
    scan = workspace.scan(workspace.reader())
    assert scan.fatal is (not with_valid)
    assert scan.diagnostics.count(ScanDiagnostic("codex_format_unrecognized")) == 1
    assert ScanDiagnostic("unknown_format", SessionKey("codex", PARENT)) in scan.diagnostics
    assert ScanDiagnostic("unknown", SessionKey("codex", PARENT)) not in scan.diagnostics
    assert tuple(source.key.session for source in scan.sessions) == ((SESSION,) if with_valid else ())


@pytest.mark.parametrize(
    ("filename", "session", "thread", "valid", "reverted"),
    [
        (f"rollout-2026-11-01T10-00-00-{SESSION}.jsonl", SESSION, SESSION, True, False),
        (f"rollout-2026-11-01T10-00-00-{SESSION}_{PARENT}.jsonl.zst", PARENT, SESSION, True, True),
        (f"rollout-2026-11-01T10-00-00.250-{SESSION}.jsonl", SESSION, None, False, False),
        (f"{SESSION}.jsonl", SESSION, None, False, False),
        ("unidentified.jsonl", None, None, False, False),
    ],
)
def test_rollout_identity_preserves_session_and_distinguishes_thread_grammar(
    filename, session, thread, valid, reverted
):
    identity = codex_history.rollout_identity(Path(filename))
    if session is None:
        assert identity is None
    else:
        assert identity is not None
        assert (identity.session, identity.thread, identity.valid_name, identity.reverted) == (
            session,
            thread,
            valid,
            reverted,
        )
