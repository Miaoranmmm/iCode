# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bounded, read-only host transcript snapshots with explicit parse failures."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

from chrys.foundation.platform.files import secure_open_regular_binary
from chrys.service.memory.extraction import SourceRevision

MAX_HISTORY_BYTES = 128 * 1024 * 1024
MAX_HISTORY_LINE_BYTES = 16 * 1024 * 1024
JsonObject = dict[str, object]


class HistoryError(ValueError):
    """A snapshot cannot prove complete, stable source material."""

    def __init__(self, kind: str, *, details: tuple[tuple[str, str], ...] = ()) -> None:
        self.kind = kind
        self.details = details
        super().__init__(kind)


@dataclass(frozen=True, slots=True)
class HistorySnapshot:
    """Complete original records; the revision hashes the logical (decompressed) bytes and keeps the physical mtime."""

    revision: SourceRevision
    records: tuple[tuple[int, JsonObject], ...]


def object_value(value: object) -> JsonObject | None:
    """Narrow external JSON mappings at the decoding boundary."""
    return cast(JsonObject, value) if isinstance(value, dict) else None


@dataclass(frozen=True, slots=True)
class HistoryContent:
    """Bounded logical bytes and the physical file's modification time."""

    payload: bytes
    revision: SourceRevision


def canonical_rollout_path(path: Path) -> Path:
    """Keep session identity independent of Codex's compression lifecycle."""
    return path.with_suffix("") if path.name.endswith(".jsonl.zst") else path


def _raise_walk_error(error: OSError) -> None:
    raise error


def discover_rollouts(source: Path) -> tuple[Path, ...]:
    """Deduplicate by logical path; an unreadable directory fails the listing instead of hiding a rollout."""
    return tuple(
        sorted(
            {
                canonical_rollout_path(directory / name)
                for directory, _, files in source.walk(on_error=_raise_walk_error)
                for name in files
                if name.endswith((".jsonl", ".jsonl.zst"))
            }
        )
    )


def _read_zstd(handle: BinaryIO, limit: int) -> bytes:
    try:
        from compression import zstd
    except ImportError:
        raise HistoryError("zstd_unavailable") from None
    try:
        with zstd.open(handle, "rb") as stream:
            return stream.read(limit)
    except zstd.ZstdError, EOFError:
        raise HistoryError("zstd_decode_failed") from None


def read_content(path: Path) -> HistoryContent:
    """Bound allocation before parsing, and prove the physical file stayed stable."""
    compressed = path.name.endswith(".jsonl.zst")
    try:
        with secure_open_regular_binary(path) as handle:
            before = os.fstat(handle.fileno())
            if not compressed and before.st_size > MAX_HISTORY_BYTES:
                raise HistoryError("source_too_large")
            payload = _read_zstd(handle, MAX_HISTORY_BYTES + 1) if compressed else handle.read(MAX_HISTORY_BYTES + 1)
            after = os.fstat(handle.fileno())
        final = path.stat()
    except HistoryError:
        raise
    except OSError, ValueError:
        raise HistoryError("source_unavailable") from None

    def identity(stat: os.stat_result) -> tuple[int, int, int, int]:
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    if (
        identity(before) != identity(after)
        or identity(after) != identity(final)
        or (not compressed and len(payload) != before.st_size)
    ):
        raise HistoryError("source_changed")
    if len(payload) > MAX_HISTORY_BYTES:
        raise HistoryError("source_too_large")
    return HistoryContent(payload, SourceRevision(before.st_mtime_ns, hashlib.sha256(payload).hexdigest()))


def read_rollout_content(path: Path) -> HistoryContent:
    """Resolve the current physical representation on every read, preferring plain."""
    plain = canonical_rollout_path(path)
    try:
        plain.stat()
    except FileNotFoundError:
        return read_content(plain.with_suffix(".jsonl.zst"))
    except OSError, ValueError:
        raise HistoryError("source_unavailable") from None
    return read_content(plain)


def read_snapshot(path: Path) -> HistorySnapshot:
    """Do not accept malformed or partial EOF as a completed conversation."""
    return parse_snapshot(read_content(path))


def read_rollout_snapshot(path: Path) -> HistorySnapshot:
    """Apply the same snapshot rules to plain and compressed Codex rollouts."""
    return parse_snapshot(read_rollout_content(path))


def parse_snapshot(content: HistoryContent) -> HistorySnapshot:
    """Parse complete logical bytes with the shared line and record limits."""
    payload = content.payload
    if payload and not payload.endswith(b"\n"):
        raise HistoryError("partial_line")
    records: list[tuple[int, JsonObject]] = []
    for position, line in enumerate(payload.splitlines(), 1):
        if len(line) > MAX_HISTORY_LINE_BYTES:
            raise HistoryError("source_line_too_large", details=(("position", str(position)),))
        try:
            record = object_value(json.loads(line))
        except ValueError, UnicodeError, RecursionError:
            record = None
        if record is None:
            raise HistoryError("invalid_json", details=(("position", str(position)),))
        records.append((position, record))
    return HistorySnapshot(content.revision, tuple(records))
