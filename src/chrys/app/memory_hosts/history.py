# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bounded, read-only host transcript snapshots with explicit parse failures."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

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
    """Complete original records and the revision of their exact file bytes."""

    revision: SourceRevision
    records: tuple[tuple[int, JsonObject], ...]


def object_value(value: object) -> JsonObject | None:
    """Narrow external JSON mappings at the decoding boundary."""
    return cast(JsonObject, value) if isinstance(value, dict) else None


def read_snapshot(path: Path) -> HistorySnapshot:
    """Do not accept malformed or partial EOF as a completed conversation."""
    try:
        with secure_open_regular_binary(path) as handle:
            before = os.fstat(handle.fileno())
            if before.st_size > MAX_HISTORY_BYTES:
                raise HistoryError("source_too_large")
            payload = handle.read(MAX_HISTORY_BYTES + 1)
            after = os.fstat(handle.fileno())
        final = path.stat()
    except HistoryError:
        raise
    except OSError, ValueError:
        raise HistoryError("source_unavailable") from None

    def identity(stat: os.stat_result) -> tuple[int, int, int, int]:
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    if identity(before) != identity(after) or identity(after) != identity(final) or len(payload) != before.st_size:
        raise HistoryError("source_changed")
    if len(payload) > MAX_HISTORY_BYTES:
        raise HistoryError("source_too_large")
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
    return HistorySnapshot(SourceRevision(before.st_mtime_ns, hashlib.sha256(payload).hexdigest()), tuple(records))
