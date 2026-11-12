# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Report observed memory delivery and maintenance counts without model judgment."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from chrys.app.memory_hosts.evaluation import EvaluationDiagnostic, scan_store, scan_usage
from chrys.app.memory_hosts.extraction_reports import RejectionSummary, read_captured_report, summarize_rejections
from chrys.foundation.platform import get_platform
from chrys.orchestration.startup import bootstrap_runtime
from chrys.service.memory.extraction import ExtractionError
from chrys.service.memory.project import bind_project
from chrys.service.memory.storage.store import WRITERS, MemoryOperationError, MemoryType, WrittenBy


@dataclass(frozen=True, slots=True)
class MetricsOptions:
    project_dir: Path
    memory_dir: Path
    claude_source: Path | None = None
    codex_source: Path | None = None


@dataclass(frozen=True, slots=True)
class EntryCount:
    type: MemoryType
    status: Literal["active", "withdrawn"]
    written_by: WrittenBy
    count: int


@dataclass(frozen=True, slots=True)
class SessionMetrics:
    host: str
    session: str
    catalog_shown_refs: tuple[str, ...]
    read_refs: tuple[str, ...]
    search_refs: tuple[str, ...]
    shown: int
    read: int
    hits: int
    hit_rate: float | None


@dataclass(frozen=True, slots=True)
class UsageTotals:
    shown: int = 0
    read: int = 0
    hits: int = 0
    hit_rate: float | None = None


@dataclass(frozen=True, slots=True)
class MetricsReport:
    project_id: str
    operations: dict[str, dict[str, int]]
    entries: tuple[EntryCount, ...]
    sessions: tuple[SessionMetrics, ...]
    totals: UsageTotals
    unavailable: dict[str, str] = field(
        default_factory=lambda: {
            "used_after_read": "unavailable",
            "corrections": "unavailable",
            "repeated_instructions": "unavailable",
        }
    )
    diagnostics: tuple[EvaluationDiagnostic, ...] = ()
    rejection_count: int | Literal["unavailable"] = "unavailable"
    rejection_kinds: dict[str, int] | Literal["unavailable"] = "unavailable"


async def run_metrics(options: MetricsOptions) -> MetricsReport:
    """Count observed use and, when present, rejections in a captured report.json."""
    project = await asyncio.to_thread(bind_project, str(options.project_dir))
    store = await scan_store(options.memory_dir, project)
    usage = await scan_usage(project, options.claude_source, options.codex_source)
    report_diagnostics: tuple[EvaluationDiagnostic, ...] = ()
    try:
        captured = await asyncio.to_thread(read_captured_report, options.memory_dir)
    except ExtractionError:
        rejections = RejectionSummary()
        report_diagnostics = (EvaluationDiagnostic("invalid_extraction_report", "report.json"),)
    else:
        rejections = summarize_rejections(captured)
    operations: dict[str, dict[str, int]] = {
        operation: dict.fromkeys(WRITERS, 0)
        for operation in ("create", "update", "withdraw", "restore", "merge", "delete")
    }
    for change in store.changes:
        operations[change.operation][change.written_by] += 1
    counts: Counter[tuple[MemoryType, Literal["active", "withdrawn"], WrittenBy]] = Counter(
        (item.stored.entry.type, item.stored.entry.status or "active", item.stored.entry.written_by)
        for item in store.entries
    )
    entries = tuple(
        EntryCount(type_, status, writer, count) for (type_, status, writer), count in sorted(counts.items())
    )
    sessions = tuple(
        SessionMetrics(
            item.key.host,
            item.key.session,
            tuple(sorted(item.shown)),
            tuple(sorted(item.read)),
            tuple(sorted(item.search)),
            len(item.shown),
            len(item.read),
            len(item.shown & item.read),
            len(item.shown & item.read) / len(item.shown) if item.shown else None,
        )
        for item in usage.sessions
    )
    shown = sum(item.shown for item in sessions)
    read = sum(item.read for item in sessions)
    hits = sum(item.hits for item in sessions)
    totals = UsageTotals(shown, read, hits, hits / shown if shown else None)
    return MetricsReport(
        project.project_id or "",
        operations,
        entries,
        sessions,
        totals,
        diagnostics=(*store.diagnostics, *usage.diagnostics, *report_diagnostics),
        rejection_count=rejections.count,
        rejection_kinds=rejections.kinds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Bootstrap the standalone read-only evaluator and print one JSON report."""
    parser = argparse.ArgumentParser(description="Report memory maintenance and observed use metrics")
    parser.add_argument("-C", type=Path, required=True)
    parser.add_argument("--claude-source", type=Path)
    parser.add_argument("--codex-source", type=Path)
    parser.add_argument("--memory-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        bootstrap_runtime(project_root=args.C, apply_patches=False, setup_telemetry=False)
        options = MetricsOptions(
            args.C, args.memory_dir or get_platform().config_dir / "memory", args.claude_source, args.codex_source
        )
        report = asyncio.run(run_metrics(options))
        sys.stdout.write(json.dumps(asdict(report), ensure_ascii=True) + "\n")
        return 0
    except MemoryOperationError as error:
        sys.stderr.write(f"{error.kind}: Memory evaluation failed.\n")
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception:
        sys.stderr.write("Memory evaluation could not start.\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
