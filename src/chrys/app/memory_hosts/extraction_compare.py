# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Compare isolated replay memory roots and optional captured report.json files.

Each positional directory is the isolated --memory-dir used by a replay. Its
optional report.json is that run's captured JSON stdout. Missing measurements
remain unavailable. Baseline Markdown is displayed only for human comparison;
this command does not build prompts or model clients.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from chrys.app.memory_hosts.evaluation import scan_store
from chrys.app.memory_hosts.extraction_reports import read_captured_report, summarize_rejections
from chrys.foundation.platform.files import secure_open_regular_binary
from chrys.service.memory.extraction import ExtractionError
from chrys.service.memory.storage.store import ToolResultEvidence, UserReportEvidence

if TYPE_CHECKING:
    from chrys.app.memory_hosts.evaluation import ScopedEntry


def _object(value: object) -> dict[str, object] | None:
    return dict(value) if isinstance(value, dict) and all(isinstance(key, str) for key in value) else None


def _number(value: object) -> int | float | None:
    return (
        value
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
        else None
    )


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _report(root: Path) -> dict[str, object] | None:
    return read_captured_report(root, error_kind="invalid_compare_report")


def _usage(batches: Sequence[dict[str, object]]) -> dict[str, object] | str:
    if not batches:
        return "unavailable"
    observations: list[tuple[int, dict[str, tuple[int | float, int]]]] = []
    for batch in batches:
        usage = _object(batch.get("usage"))
        if usage is None or (calls := _integer(usage.get("calls"))) is None:
            return "unavailable"
        rows = usage.get("fields")
        if not isinstance(rows, list):
            return "unavailable"
        fields: dict[str, tuple[int | float, int]] = {}
        for row in rows:
            field = _object(row)
            if field is None:
                return "unavailable"
            name = field.get("name")
            # Field names follow the extraction report: asdict(UsageField).
            value, missing = _number(field.get("observed_sum")), _integer(field.get("missing_calls"))
            if not isinstance(name, str) or value is None or missing is None or name in fields:
                return "unavailable"
            fields[name] = value, missing
        observations.append((calls, fields))
    names = sorted({name for _, fields in observations for name in fields})
    totals: list[dict[str, object]] = []
    for name in names:
        values = [fields.get(name, (0, calls)) for calls, fields in observations]
        totals.append(
            {
                "name": name,
                "observed_sum": sum(value for value, _ in values),
                "missing_calls": sum(missing for _, missing in values),
            }
        )
    return {"calls": sum(calls for calls, _ in observations), "fields": totals}


def _measurements(report: dict[str, object] | None) -> dict[str, object]:
    report = report or {}
    rejections = summarize_rejections(report)
    raw_batches = report.get("batches")
    batches = (
        [parsed for item in raw_batches if (parsed := _object(item)) is not None]
        if isinstance(raw_batches, list)
        else []
    )
    profile = _object(report.get("profile")) or {}
    tuning = profile.get("tuning_version", report.get("tuning_version"))
    elapsed, conflicts = _number(report.get("elapsed_seconds")), _integer(report.get("conflict_count"))
    known = [value for batch in batches if (value := _batch_outcome(batch)) is not None]
    limits: dict[str, int] | str = (
        {
            "turn_limited_batches": known.count("turn_limited"),
            "output_limited_batches": known.count("output_limited"),
            # A failed or cancelled batch left its sessions partly unread; without
            # these counts a cut-short run reads as a clean one.
            "failed_batches": known.count("failed"),
            "cancelled_batches": known.count("cancelled"),
            # Batches whose report shows neither a termination nor a final status.
            "unavailable_batches": len(batches) - len(known),
        }
        if known
        else "unavailable"
    )
    return {
        "usage": _usage(batches),
        "elapsed_seconds": elapsed if elapsed is not None else "unavailable",
        "limits": limits,
        "conflict_count": conflicts if conflicts is not None else "unavailable",
        "rejection_count": rejections.count,
        "rejection_kinds": rejections.kinds,
        "tuning_version": tuning if isinstance(tuning, str) and tuning else "unavailable",
    }


# Final batch statuses that mean the batch's work stood; cancellation after a commit keeps it.
_KEPT_STATUSES = frozenset(
    {
        "completed",
        "replay_completed",
        "incomplete_limits",
        "replay_incomplete_limits",
        "committed_then_cancelled",
        "incomplete_limits_committed_then_cancelled",
    }
)


def _batch_outcome(batch: Mapping[str, object]) -> str | None:
    """The batch's final result: a run can finish its model turn and still fail afterwards.

    Any final status outside the kept set is a cut-short batch, so a new failure status
    can never be counted as clean. Reports without a status fall back to the termination.
    """
    status, termination = batch.get("status"), batch.get("termination")
    if termination == "failed":
        return "failed"
    if isinstance(status, str) and status not in _KEPT_STATUSES:
        return "cancelled" if "cancelled" in status else "failed"
    if isinstance(termination, str) and termination in {"completed", "turn_limited", "output_limited", "cancelled"}:
        return termination
    return None


def _source_keys(item: ScopedEntry) -> frozenset[tuple[str, str | None]]:
    entry = item.stored.entry
    return frozenset(
        [
            *((quote.text, quote.session) for quote in entry.quotes),
            *(
                (evidence.excerpt, evidence.session)
                for evidence in entry.evidence
                if isinstance(evidence, ToolResultEvidence | UserReportEvidence)
            ),
        ]
    )


def _entry(item: ScopedEntry) -> dict[str, object]:
    return {
        "ref": item.stored.entry.ref,
        "project_id": item.scope.project_id,
        "version": item.stored.version,
        "entry": asdict(item.stored.entry),
    }


def _aligned(a: Sequence[ScopedEntry], b: Sequence[ScopedEntry]) -> dict[str, object]:
    entries = (*a, *b)
    parents = list(range(len(entries)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    by_source: dict[tuple[str, str | None], tuple[list[int], list[int]]] = {}
    for index, item in enumerate(entries):
        for key in _source_keys(item):
            group = by_source.setdefault(key, ([], []))
            group[0 if index < len(a) else 1].append(index)
    for left, right in by_source.values():
        # A shared source must cross the run boundary; overlap within A alone
        # cannot make another A entry appear in B.
        if left and right:
            for index in (*left, *right):
                parents[root(index)] = root(left[0])
    groups: dict[int, tuple[list[dict[str, object]], list[dict[str, object]]]] = {}
    for index, item in enumerate(entries):
        group = groups.setdefault(root(index), ([], []))
        group[0 if index < len(a) else 1].append(_entry(item))
    common: list[dict[str, object]] = []
    a_only: list[dict[str, object]] = []
    b_only: list[dict[str, object]] = []
    common_a, common_b = 0, 0
    for left, right in groups.values():
        if left and right:
            common.append({"a": left, "b": right})
            common_a += len(left)
            common_b += len(right)
        else:
            a_only.extend(left)
            b_only.extend(right)
    return {
        "common": common,
        "a_only": a_only,
        "b_only": b_only,
        "counts": {
            "common_groups": len(common),
            "common_a_entries": common_a,
            "common_b_entries": common_b,
            "a_only": len(a_only),
            "b_only": len(b_only),
        },
    }


def _baseline(directory: Path | None) -> list[dict[str, str]]:
    if directory is None:
        return []
    if not directory.is_dir():
        raise ExtractionError("invalid_compare_baseline", "The manual baseline must be a directory.")
    try:
        result = []
        for path in sorted(directory.glob("*.md")):
            with secure_open_regular_binary(path) as handle:
                result.append({"name": path.name, "text": handle.read().decode("utf-8")})
        return result
    except OSError, UnicodeError:
        raise ExtractionError("invalid_compare_baseline", "The manual baseline could not be read.") from None


async def compare_runs(a: Path, b: Path, *, baseline: Path | None = None) -> dict[str, object]:
    """Align connected source matches, retaining one-to-many differences for a human."""
    if not await asyncio.to_thread(a.is_dir) or not await asyncio.to_thread(b.is_dir):
        raise ExtractionError("invalid_compare", "Compare needs two isolated memory directories.")
    left, right = await asyncio.gather(scan_store(a), scan_store(b))
    left_report, right_report, manual = await asyncio.gather(
        asyncio.to_thread(_report, a),
        asyncio.to_thread(_report, b),
        asyncio.to_thread(_baseline, baseline),
    )
    return {
        **_aligned(left.entries, right.entries),
        "runs": {"a": _measurements(left_report), "b": _measurements(right_report)},
        "diagnostics": {
            "a": [asdict(item) for item in left.diagnostics],
            "b": [asdict(item) for item in right.diagnostics],
        },
        "baseline": manual,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare isolated replay memory roots; optional report.json is captured extract stdout."
    )
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--baseline", type=Path, help="Display top-level Markdown only for manual comparison.")
    args = parser.parse_args(argv)
    try:
        from chrys.orchestration.startup import bootstrap_runtime

        bootstrap_runtime()
        result = asyncio.run(compare_runs(args.a, args.b, baseline=args.baseline))
    except ExtractionError as error:
        sys.stderr.write(f"Memory replay comparison failed: {error.kind}.\n")
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception:
        sys.stderr.write("Memory replay comparison failed: input_unavailable.\n")
        return 1
    sys.stdout.write(json.dumps(result, ensure_ascii=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
