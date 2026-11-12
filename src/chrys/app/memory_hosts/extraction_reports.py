# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Read optional captured extraction reports and summarize observed rejections."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from chrys.foundation.platform.files import secure_open_regular_binary
from chrys.service.memory.extraction import ExtractionError

Unavailable = Literal["unavailable"]


@dataclass(frozen=True, slots=True)
class RejectionSummary:
    """A captured run's counts; missing observations must not become zeroes."""

    count: int | Unavailable = "unavailable"
    kinds: dict[str, int] | Unavailable = "unavailable"


def read_captured_report(root: Path, *, error_kind: str = "invalid_extraction_report") -> dict[str, object] | None:
    """Read report.json without creating paths; reject malformed captured reports."""
    try:
        with secure_open_regular_binary(root / "report.json") as handle:
            value = json.loads(handle.read())
    except FileNotFoundError:
        return None
    except OSError, ValueError, UnicodeError:
        raise ExtractionError(error_kind, "A captured extraction report could not be read.") from None
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ExtractionError(error_kind, "A captured extraction report must be a JSON object.")
    return value


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _kinds(value: object) -> dict[str, int] | None:
    if not isinstance(value, list | tuple):
        return None
    kinds: dict[str, int] = {}
    for row in value:
        if not isinstance(row, list | tuple) or len(row) != 2:
            return None
        label, count = row
        if not isinstance(label, str) or not label or label in kinds or (parsed := _integer(count)) is None:
            return None
        kinds[label] = parsed
    return kinds


def summarize_rejections(report: Mapping[str, object] | None) -> RejectionSummary:
    """Sum every captured batch, leaving absent or partial measurements unavailable."""
    batches = report.get("batches") if report is not None else None
    if not isinstance(batches, list) or not batches:
        return RejectionSummary()
    count = 0
    kinds: Counter[str] = Counter()
    complete_kinds = True
    for batch in batches:
        if not isinstance(batch, dict) or (observed := _integer(batch.get("rejection_count"))) is None:
            return RejectionSummary()
        count += observed
        observed_kinds = _kinds(batch.get("rejection_kinds"))
        if observed_kinds is None or sum(observed_kinds.values()) != observed:
            complete_kinds = False
        else:
            kinds.update(observed_kinds)
    return RejectionSummary(count, dict(sorted(kinds.items())) if complete_kinds else "unavailable")
