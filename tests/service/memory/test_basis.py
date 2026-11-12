# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""File evidence and read-time basis states from the step-7 specification."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from chrys.service.memory import file_evidence
from chrys.service.memory.basis import check_basis
from chrys.service.memory.file_evidence import capture_file
from chrys.service.memory.operations import MemoryReader
from chrys.service.memory.render import render_entry
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import (
    FileEvidence,
    MemoryEntryDraft,
    MemoryOperationError,
    MemoryScope,
    ReferenceEvidence,
    ToolResultEvidence,
    UserReportEvidence,
)
from tests.support.symlinks import symlink_or_skip

PROJECT = MemoryScope("project", "0123456789abcdef")
OTHER_PROJECT = "fedcba9876543210"
MIB = 1024 * 1024


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def evidence(path: str, payload: bytes = b"original") -> FileEvidence:
    return FileEvidence(str(PROJECT.project_id), path, digest(payload), len(payload))


def test_file_capture_uses_project_relative_posix_path_and_bytes(tmp_path: Path) -> None:
    directory = tmp_path / "sub"
    directory.mkdir()
    path = directory / "file.txt"
    payload = "source\n原文".encode()
    path.write_bytes(payload)
    relative = capture_file(PROJECT, tmp_path, "sub/file.txt")
    absolute = capture_file(PROJECT, tmp_path, str(path))
    assert relative == absolute == FileEvidence(str(PROJECT.project_id), "sub/file.txt", digest(payload), len(payload))
    path.write_bytes(b"new")
    assert capture_file(PROJECT, tmp_path, "sub/file.txt").sha256 == digest(b"new")


@pytest.mark.parametrize("kind", ["missing", "outside_relative", "outside_absolute", "missing_root", "missing_project"])
def test_file_capture_refuses_unavailable_or_outside_project(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    path = "missing.txt"
    selected_root: Path | None = root
    project: MemoryScope | None = PROJECT
    if kind == "outside_relative":
        path = "../outside.txt"
    elif kind == "outside_absolute":
        path = str(outside)
    elif kind == "missing_root":
        selected_root = None
    elif kind == "missing_project":
        project = None
    with pytest.raises(MemoryOperationError) as caught:
        capture_file(project, selected_root, path)
    assert caught.value.kind == "memory_evidence_not_found"
    assert str(tmp_path) not in caught.value.message


def test_file_capture_refuses_symlink_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    symlink_or_skip(root / "linked", outside)
    with pytest.raises(MemoryOperationError) as caught:
        capture_file(PROJECT, root, "linked")
    assert caught.value.kind == "memory_evidence_not_found"


def test_file_capture_large_file_keeps_size_without_hash(tmp_path: Path) -> None:
    payload = b"x" * (2 * MIB + 1)
    (tmp_path / "large").write_bytes(payload)
    result = capture_file(PROJECT, tmp_path, "large")
    assert result == FileEvidence(str(PROJECT.project_id), "large", None, len(payload))


@pytest.mark.parametrize("kind", ["unchanged", "changed", "deleted", "no_hash", "other_project", "no_root", "escape"])
def test_basis_file_states_match_specification(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"original")
    item = evidence("file")
    project_root: Path | None = tmp_path
    if kind == "changed":
        path.write_bytes(b"different")
    elif kind == "deleted":
        path.unlink()
    elif kind == "no_hash":
        item = replace(item, sha256=None)
    elif kind == "other_project":
        item = replace(item, project=OTHER_PROJECT)
    elif kind == "no_root":
        project_root = None
    elif kind == "escape":
        item = replace(item, path="../outside")
    result = check_basis((item,), PROJECT, project_root)
    expected = "changed" if kind in {"changed", "deleted"} else "unchanged" if kind == "unchanged" else "unknown"
    assert result.state == expected
    assert result.files[0].state == expected
    assert result.files[0].index == 0
    if kind == "other_project":
        assert result.files[0].detail == "from another project"


def test_basis_symlink_escape_is_unknown(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"original")
    symlink_or_skip(root / "file", outside)
    result = check_basis((evidence("file"),), PROJECT, root)
    assert result.state == "unknown"


def test_basis_permission_error_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "file").write_bytes(b"original")

    def denied(path: Path):
        raise PermissionError("synthetic permission failure")

    monkeypatch.setattr(file_evidence, "secure_open_regular_binary", denied)
    result = check_basis((evidence("file"),), PROJECT, tmp_path)
    assert result.state == "unknown"


def test_basis_changed_during_read_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"original")
    original = file_evidence.secure_open_regular_binary

    def opened_then_changed(target: Path):
        handle = original(target)
        target.write_bytes(b"rewritten after open")
        return handle

    monkeypatch.setattr(file_evidence, "secure_open_regular_binary", opened_then_changed)
    result = check_basis((evidence("file"),), PROJECT, tmp_path)
    assert result.state == "unknown"


@pytest.mark.parametrize("case", ["none", "unchanged", "unknown", "changed"])
def test_basis_aggregation_priorities(tmp_path: Path, case: str) -> None:
    (tmp_path / "file").write_bytes(b"original")
    items = [
        ReferenceEvidence("not-fetched"),
        ToolResultEvidence("shell", "{}", "not re-run"),
        UserReportEvidence("reported"),
    ]
    if case != "none":
        items.append(evidence("file"))
    if case in {"unknown", "changed"}:
        items.append(replace(evidence("file"), project=OTHER_PROJECT))
    if case == "changed":
        items.append(evidence("missing"))
    assert check_basis(items, PROJECT, tmp_path).state == case


def test_basis_limits_files_per_read(tmp_path: Path) -> None:
    (tmp_path / "file").write_bytes(b"original")
    result = check_basis(tuple(evidence("file") for _ in range(9)), PROJECT, tmp_path)
    assert [item.state for item in result.files] == ["unchanged"] * 8 + ["unknown"]
    assert result.files[-1].detail == "file budget exceeded"


def test_basis_limits_each_file_and_total_bytes(tmp_path: Path) -> None:
    two_mib = b"x" * (2 * MIB)
    (tmp_path / "two").write_bytes(two_mib)
    (tmp_path / "larger").write_bytes(two_mib + b"x")
    items = tuple(evidence("two", two_mib) for _ in range(5))
    result = check_basis(items, PROJECT, tmp_path)
    assert [item.state for item in result.files] == ["unchanged"] * 4 + ["unknown"]
    assert result.files[-1].detail == "byte budget exceeded"
    result = check_basis((evidence("larger", two_mib),), PROJECT, tmp_path)
    assert result.state == "unknown"


async def test_memory_read_checks_basis_again_against_current_worktree(tmp_path: Path) -> None:
    root = tmp_path / "project"
    await asyncio.to_thread(root.mkdir)
    path = root / "file"
    await asyncio.to_thread(path.write_bytes, b"original")
    store = FileMemoryStore(tmp_path / "memory")
    saved = await store.create(
        PROJECT, MemoryEntryDraft("engineering", "Finding", "Observed result", "model", evidence=(evidence("file"),))
    )
    reader = MemoryReader(store, PROJECT, project_root=root)
    first = await reader.read(saved.ref)
    assert first.basis is not None and first.basis.state == "unchanged"
    await asyncio.to_thread(path.write_bytes, b"different")
    second = await reader.read(saved.ref)
    assert second.basis is not None and second.basis.state == "changed"
    assert second.version == first.version == saved.version
    assert "Basis: changed." in render_entry(second)


def test_file_capture_inside_symlink_uses_canonical_relative_path(tmp_path: Path) -> None:
    directory = tmp_path / "nested"
    directory.mkdir()
    target = directory / "file"
    target.write_bytes(b"original")
    symlink_or_skip(tmp_path / "alias", target)
    assert capture_file(PROJECT, tmp_path, "alias") == evidence("nested/file")


def test_basis_missing_project_root_is_unknown(tmp_path: Path) -> None:
    result = check_basis((evidence("file"),), PROJECT, tmp_path / "missing-root")
    assert result.state == "unknown"


def test_basis_file_budget_prevents_ninth_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from chrys.service.memory import basis
    from chrys.service.memory.file_evidence import FileSnapshot, ProjectFile

    (tmp_path / "file").write_bytes(b"original")
    original = basis.snapshot_file
    remaining: list[int] = []

    def observed(source: ProjectFile, *, remaining_bytes: int) -> FileSnapshot:
        remaining.append(remaining_bytes)
        return original(source, remaining_bytes=remaining_bytes)

    monkeypatch.setattr(basis, "snapshot_file", observed)
    result = check_basis(tuple(evidence("file") for _ in range(9)), PROJECT, tmp_path)
    assert len(remaining) == 8
    assert result.files[-1].state == "unknown"


def test_basis_byte_budget_limits_actual_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import TracebackType
    from typing import BinaryIO

    payload = b"x" * (2 * MIB)
    (tmp_path / "two").write_bytes(payload)
    original = file_evidence.secure_open_regular_binary
    read_bytes: list[int] = []

    class CountingReader:
        def __init__(self, handle: BinaryIO) -> None:
            self.handle = handle

        def fileno(self) -> int:
            return self.handle.fileno()

        def read(self, size: int = -1) -> bytes:
            result = self.handle.read(size)
            read_bytes.append(len(result))
            return result

        def __enter__(self) -> CountingReader:
            return self

        def __exit__(
            self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
        ) -> None:
            self.handle.close()

    def observed(path: Path) -> CountingReader:
        return CountingReader(original(path))

    monkeypatch.setattr(file_evidence, "secure_open_regular_binary", observed)
    result = check_basis(tuple(evidence("two", payload) for _ in range(5)), PROJECT, tmp_path)
    assert read_bytes == [2 * MIB] * 4
    assert sum(read_bytes) == 8 * MIB
    assert result.files[-1].state == "unknown"


def test_basis_file_disappearing_after_open_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "file").write_bytes(b"original")
    original = file_evidence.secure_open_regular_binary

    def opened_then_deleted(target: Path):
        handle = original(target)
        target.unlink()
        return handle

    monkeypatch.setattr(file_evidence, "secure_open_regular_binary", opened_then_deleted)
    assert check_basis((evidence("file"),), PROJECT, tmp_path).state == "unknown"


def test_basis_charges_a_snapshot_that_fails_after_reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import TracebackType
    from typing import BinaryIO

    payload = b"x" * (2 * MIB)
    target = tmp_path / "file"
    target.write_bytes(payload)
    original = file_evidence.secure_open_regular_binary
    read_bytes: list[int] = []

    class ChangingReader:
        def __init__(self, handle: BinaryIO) -> None:
            self.handle = handle

        def fileno(self) -> int:
            return self.handle.fileno()

        def read(self, size: int = -1) -> bytes:
            result = self.handle.read(size)
            read_bytes.append(len(result))
            if len(read_bytes) == 1:
                target.write_bytes(b"y" * len(payload))
            return result

        def __enter__(self) -> ChangingReader:
            return self

        def __exit__(
            self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
        ) -> None:
            self.handle.close()

    def opened(path: Path) -> ChangingReader:
        return ChangingReader(original(path))

    monkeypatch.setattr(file_evidence, "secure_open_regular_binary", opened)
    result = check_basis(tuple(evidence("file", payload) for _ in range(5)), PROJECT, tmp_path)
    assert result.files[0].state == "unknown"
    assert read_bytes == [2 * MIB] * 4
    assert result.files[-1].detail == "byte budget exceeded"


def test_basis_growing_file_never_reads_beyond_the_single_file_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import TracebackType
    from typing import BinaryIO

    payload = b"x" * (2 * MIB)
    target = tmp_path / "file"
    target.write_bytes(payload)
    original = file_evidence.secure_open_regular_binary
    read_bytes: list[int] = []

    class GrowingReader:
        def __init__(self, handle: BinaryIO) -> None:
            self.handle = handle

        def fileno(self) -> int:
            return self.handle.fileno()

        def read(self, size: int = -1) -> bytes:
            target.write_bytes(payload + b"y")
            result = self.handle.read(size)
            read_bytes.append(len(result))
            return result

        def __enter__(self) -> GrowingReader:
            return self

        def __exit__(
            self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
        ) -> None:
            self.handle.close()

    def opened(path: Path) -> GrowingReader:
        return GrowingReader(original(path))

    monkeypatch.setattr(file_evidence, "secure_open_regular_binary", opened)
    result = check_basis((evidence("file", payload),), PROJECT, tmp_path)
    assert result.state == "unknown"
    assert read_bytes == [2 * MIB]
