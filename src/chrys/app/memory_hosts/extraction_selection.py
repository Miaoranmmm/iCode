# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Compare frozen extraction selection policies offline, reporting counts only."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from chrys.app.memory_hosts.extraction_replay import load_frozen_sources, load_manifest
from chrys.foundation.platform import get_platform
from chrys.service.memory.extraction import (
    DEFAULT_MATERIAL_PROFILE,
    ExtractionError,
    OversizeSelection,
    build_batches,
    tuned_material_profile,
)
from chrys.service.memory.storage.store import MemoryScope
from chrys.service.memory.tuning import MemoryTuning, load_active, load_candidate

if TYPE_CHECKING:
    from chrys.service.memory.extraction import ExtractionBatch


@dataclass(frozen=True, slots=True)
class PolicyCounts:
    """Coverage counters contain no source identity, source text or model data."""

    batch_count: int
    omitted_tool_results: int
    omitted_error_results: int
    omitted_signal_neighbors: int
    omitted_assistant_messages: int

    def as_dict(self) -> dict[str, int]:
        return {
            "batch_count": self.batch_count,
            "omitted_tool_results": self.omitted_tool_results,
            "omitted_error_results": self.omitted_error_results,
            "omitted_signal_neighbors": self.omitted_signal_neighbors,
            "omitted_assistant_messages": self.omitted_assistant_messages,
        }


def _counts(batches: tuple[ExtractionBatch, ...]) -> PolicyCounts:
    return PolicyCounts(
        len(batches),
        sum(count.omitted for batch in batches for count in batch.material_counts if count.kind == "tool_result"),
        sum(batch.omitted_error_results for batch in batches),
        sum(batch.omitted_signal_neighbors for batch in batches),
        sum(batch.omitted_assistant_messages for batch in batches),
    )


async def compare_selection(
    path: Path, *, rules: tuple[OversizeSelection, ...] | None = None, tuning: MemoryTuning | None = None
) -> dict[str, object]:
    """Compare all frozen sessions using the shared planner and neighborhood counts."""
    manifest = await asyncio.to_thread(load_manifest, path)
    project = MemoryScope("project", manifest.project_id)
    scanned = await load_frozen_sources(manifest, project)
    if tuning is None:
        try:
            tuning = await asyncio.to_thread(load_active, get_platform().config_dir)
        except (OSError, ValueError) as error:
            raise ExtractionError("invalid_tuning", "The active memory tuning is unavailable or invalid.") from error
    profile = tuned_material_profile(DEFAULT_MATERIAL_PROFILE, tuning)
    budget = min(manifest.material_budget, tuning.material_budget)
    rules = rules or tuple(dict.fromkeys((tuning.oversize_selection, "chronological", "head_tail", "signal_first")))
    policies: dict[str, dict[str, int]] = {}
    for policy in rules:
        batches = build_batches(
            project,
            scanned.sessions,
            budget,
            profile=profile,
            oversize_selection=policy,
            max_batches=len(scanned.sessions),
        )
        policies[policy] = _counts(batches).as_dict()
    return {
        "material_profile": profile.name,
        "material_budget": budget,
        "source_sessions": len(scanned.sessions),
        "policies": policies,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare frozen memory extraction selection policies without a model.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--rule", action="append", choices=("chronological", "head_tail", "signal_first"))
    parser.add_argument("--tuning", help="Tuning number or JSON file, without activating it")
    args = parser.parse_args(argv)
    try:
        from chrys.orchestration.startup import bootstrap_runtime

        bootstrap_runtime()
        try:
            tuning = load_candidate(get_platform().config_dir, args.tuning) if args.tuning is not None else None
        except (OSError, ValueError) as error:
            raise ExtractionError("invalid_tuning", "The requested memory tuning is unavailable or invalid.") from error
        report = asyncio.run(
            compare_selection(args.manifest, rules=tuple(args.rule) if args.rule else None, tuning=tuning)
        )
    except ExtractionError as error:
        sys.stderr.write(f"Memory selection comparison failed: {error.kind}.\n")
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("Memory selection comparison cancelled.\n")
        return 130
    except Exception:
        sys.stderr.write("Memory selection comparison failed: input_unavailable.\n")
        return 1
    sys.stdout.write(json.dumps(report, ensure_ascii=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
