# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Replay comparison aligns exact sources and reads metadata without a model."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest

from chrys.app.memory_hosts import extraction_compare as comparison
from chrys.foundation.platform.process import windows_hidden_subprocess_kwargs
from chrys.service.memory.extraction import ExtractionError, UsageField, UsageSummary
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import (
    GLOBAL_SCOPE,
    MemoryEntryDraft,
    MemoryScope,
    Quote,
    ReferenceEvidence,
    ToolResultEvidence,
    UserReportEvidence,
)
from tests.support.secure_files import plant_owner_only_bytes
from tests.support.waiting import ENGINE_TURN_TIMEOUT

REPO = Path(__file__).resolve().parents[3]
PROJECT = MemoryScope("project", "0123456789abcdef")


@dataclass(frozen=True)
class ReplayRoots:
    a: Path
    b: Path
    home: Path


@pytest.fixture
async def replay_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReplayRoots:
    roots = ReplayRoots(tmp_path / "a", tmp_path / "b", tmp_path / "home")
    for path in (roots.a, roots.b, roots.home):
        await asyncio.to_thread(path.mkdir)
    for name in ("HOME", "USERPROFILE", "APPDATA", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(name, str(roots.home / name))
    return roots


def intent(title: str, text: str, session: str = "synthetic-session") -> MemoryEntryDraft:
    return MemoryEntryDraft(
        "intent", title, f"Body of {title}", "extraction", quotes=(Quote(text, "message", session),)
    )


async def save(root: Path, draft: MemoryEntryDraft, scope: MemoryScope = PROJECT) -> str:
    return (await FileMemoryStore(root).create(scope, draft)).ref


def report_data(*, tuning: str = "v2") -> dict[str, object]:
    return {
        "status": "replay_incomplete_limits",
        "elapsed_seconds": 12.5,
        "conflict_count": 3,
        "profile": {"tuning_version": tuning},
        "batches": [
            {
                "termination": "output_limited",
                "stop_reason": "max_output_tokens",
                # Serialized exactly as the extraction report writes it.
                "usage": asdict(UsageSummary((UsageField("input_token_count", 10, 0),), calls=2)),
            },
            {
                "termination": "turn_limited",
                "stop_reason": "max_turns",
                "usage": asdict(UsageSummary((UsageField("input_token_count", 4, 1),), calls=1)),
            },
        ],
    }


async def put_report(root: Path, value: object) -> None:
    await asyncio.to_thread(plant_owner_only_bytes, root / "report.json", json.dumps(value).encode("utf-8"))


def snapshot(root: Path) -> tuple[tuple[str, bytes], ...]:
    return tuple(
        sorted((path.relative_to(root).as_posix(), path.read_bytes()) for path in root.rglob("*") if path.is_file())
    )


async def test_compare_aligns_exact_source_words_and_session_without_guessing_from_titles(
    replay_roots: ReplayRoots,
) -> None:
    a, b = replay_roots.a, replay_roots.b
    shared_a = await save(a, intent("Original title", "Shared exact words"))
    shared_b = await save(b, intent("Changed title", "Shared exact words"))
    different_session = await save(b, intent("Original title", "Shared exact words", "other-session"))
    only_a = await save(a, intent("Only A", "Different exact words"), GLOBAL_SCOPE)
    before = await asyncio.to_thread(snapshot, a.parent)
    result = await comparison.compare_runs(a, b)
    assert result["counts"] == {
        "common_groups": 1,
        "common_a_entries": 1,
        "common_b_entries": 1,
        "a_only": 1,
        "b_only": 1,
    }
    assert [item["ref"] for item in result["common"][0]["a"]] == [shared_a]
    assert [item["ref"] for item in result["common"][0]["b"]] == [shared_b]
    assert result["common"][0]["a"][0]["entry"]["title"] == "Original title"
    assert result["common"][0]["b"][0]["entry"]["body"] == "Body of Changed title"
    assert [item["ref"] for item in result["a_only"]] == [only_a]
    assert [item["ref"] for item in result["b_only"]] == [different_session]
    assert await asyncio.to_thread(snapshot, a.parent) == before


async def test_compare_quote_and_excerpt_alignment_is_many_to_many_and_never_title_based(
    replay_roots: ReplayRoots,
) -> None:
    a, b = replay_roots.a, replay_roots.b
    await save(
        a,
        MemoryEntryDraft(
            "intent",
            "Merged",
            "Combined body",
            "extraction",
            quotes=(Quote("First", "message", "s"), Quote("Second", "message", "s")),
        ),
    )
    await save(
        b,
        MemoryEntryDraft(
            "engineering",
            "First",
            "Observed first",
            "extraction",
            evidence=(ToolResultEvidence("shell", "{}", "First", "s"),),
        ),
    )
    await save(
        b,
        MemoryEntryDraft(
            "engineering", "Second", "Observed second", "extraction", evidence=(UserReportEvidence("Second", "s"),)
        ),
    )
    await save(
        a,
        MemoryEntryDraft(
            "engineering", "Same reference title", "same body", "extraction", evidence=(ReferenceEvidence("source.md"),)
        ),
    )
    await save(
        b,
        MemoryEntryDraft(
            "engineering", "Same reference title", "same body", "extraction", evidence=(ReferenceEvidence("source.md"),)
        ),
    )
    result = await comparison.compare_runs(a, b)
    assert result["counts"] == {
        "common_groups": 1,
        "common_a_entries": 1,
        "common_b_entries": 2,
        "a_only": 1,
        "b_only": 1,
    }
    assert len(result["common"][0]["a"]) == 1 and len(result["common"][0]["b"]) == 2


async def test_same_run_source_overlap_cannot_make_an_unmatched_entry_common(replay_roots: ReplayRoots) -> None:
    unmatched = await save(replay_roots.a, intent("A only", "First", "s"))
    await save(
        replay_roots.a,
        MemoryEntryDraft(
            "intent",
            "Two sources",
            "Combined body",
            "extraction",
            quotes=(Quote("First", "message", "s"), Quote("Second", "message", "s")),
        ),
    )
    await save(replay_roots.b, intent("B second", "Second", "s"))
    result = await comparison.compare_runs(replay_roots.a, replay_roots.b)
    assert result["counts"] == {
        "common_groups": 1,
        "common_a_entries": 1,
        "common_b_entries": 1,
        "a_only": 1,
        "b_only": 0,
    }
    assert result["a_only"][0]["ref"] == unmatched


async def test_compare_reports_actual_usage_elapsed_limits_conflicts_and_tuning(replay_roots: ReplayRoots) -> None:
    await put_report(replay_roots.a, report_data())
    await put_report(replay_roots.b, report_data(tuning="candidate-3"))
    result = await comparison.compare_runs(replay_roots.a, replay_roots.b)
    run = result["runs"]["a"]
    assert run["usage"] == {
        "calls": 3,
        "fields": [{"name": "input_token_count", "observed_sum": 14, "missing_calls": 1}],
    }
    assert run["elapsed_seconds"] == 12.5
    assert run["limits"] == {
        "turn_limited_batches": 1,
        "output_limited_batches": 1,
        "failed_batches": 0,
        "cancelled_batches": 0,
        "unavailable_batches": 0,
    }
    assert run["conflict_count"] == 3 and run["tuning_version"] == "v2"
    assert result["runs"]["b"]["tuning_version"] == "candidate-3"


async def test_compare_counts_failed_and_cancelled_batches_so_a_cut_short_run_is_visible(
    replay_roots: ReplayRoots,
) -> None:
    usage = asdict(UsageSummary((UsageField("input_token_count", 1, 0),), calls=1))
    await put_report(
        replay_roots.a,
        {
            "status": "failed",
            "batches": [
                {"termination": "completed", "status": "replay_completed", "usage": usage},
                {"termination": "failed", "stop_reason": "timeout", "usage": usage},
                {"termination": "cancelled", "usage": usage},
                # The model turn finished, but revalidation or the commit failed afterwards.
                {"termination": "completed", "status": "failed", "usage": usage},
                {"termination": "completed", "status": "watermark_write_failed", "usage": usage},
                # Failed before the model ran: no termination at all.
                {"status": "attempt_write_failed"},
                {"status": "cancelled_before_model"},
                # Any status outside the kept set is cut short, including ones added later.
                {"termination": "completed", "status": "extraction_failed", "usage": usage},
                # Cancelled after its commit: the batch's work stands.
                {"termination": "completed", "status": "committed_then_cancelled", "usage": usage},
                {"termination": "output_limited", "status": "replay_incomplete_limits", "usage": usage},
            ],
        },
    )
    result = await comparison.compare_runs(replay_roots.a, replay_roots.b)
    assert result["runs"]["a"]["limits"] == {
        "turn_limited_batches": 0,
        "output_limited_batches": 1,
        "failed_batches": 5,
        "cancelled_batches": 2,
        "unavailable_batches": 0,
    }


async def test_compare_missing_metadata_is_unavailable_instead_of_invented_zero(replay_roots: ReplayRoots) -> None:
    await put_report(replay_roots.b, {"batches": []})
    result = await comparison.compare_runs(replay_roots.a, replay_roots.b)
    for label in ("a", "b"):
        assert result["runs"][label] == {
            "usage": "unavailable",
            "elapsed_seconds": "unavailable",
            "limits": "unavailable",
            "conflict_count": "unavailable",
            "rejection_count": "unavailable",
            "rejection_kinds": "unavailable",
            "tuning_version": "unavailable",
        }


async def test_compare_sums_rejections_without_counting_them_as_failed_batches(replay_roots: ReplayRoots) -> None:
    await put_report(
        replay_roots.a,
        {
            "batches": [
                {
                    "status": "replay_completed",
                    "termination": "completed",
                    "rejection_count": 2,
                    "rejection_kinds": [["memory_read:memory_invalid_input", 2]],
                },
                {
                    "status": "replay_completed",
                    "termination": "completed",
                    "rejection_count": 1,
                    "rejection_kinds": [["memory_save:memory_evidence_not_found", 1]],
                },
                {
                    "status": "failed",
                    "termination": "failed",
                    "failure_count": 1,
                    "rejection_count": 0,
                    "rejection_kinds": [],
                },
            ]
        },
    )
    before = await asyncio.to_thread(snapshot, replay_roots.a)
    result = await comparison.compare_runs(replay_roots.a, replay_roots.b)
    observed = result["runs"]["a"]
    assert observed["rejection_count"] == 3
    assert observed["rejection_kinds"] == {
        "memory_read:memory_invalid_input": 2,
        "memory_save:memory_evidence_not_found": 1,
    }
    assert observed["limits"]["failed_batches"] == 1
    assert await asyncio.to_thread(snapshot, replay_roots.a) == before


async def test_compare_old_or_partial_rejection_counts_are_unavailable(replay_roots: ReplayRoots) -> None:
    await put_report(replay_roots.a, report_data())
    await put_report(
        replay_roots.b,
        {"batches": [{"rejection_count": 3, "rejection_kinds": [["memory_read:memory_invalid_input", 3]]}, {}]},
    )
    result = await comparison.compare_runs(replay_roots.a, replay_roots.b)
    for label in ("a", "b"):
        assert result["runs"][label]["rejection_count"] == "unavailable"
        assert result["runs"][label]["rejection_kinds"] == "unavailable"


@pytest.mark.parametrize("count", [-1, True, 1.5, "2", None])
async def test_compare_invalid_rejection_counts_are_unavailable(replay_roots: ReplayRoots, count: object) -> None:
    await put_report(replay_roots.a, {"batches": [{"rejection_count": count, "rejection_kinds": []}]})
    result = await comparison.compare_runs(replay_roots.a, replay_roots.b)
    assert result["runs"]["a"]["rejection_count"] == "unavailable"


async def test_compare_known_zero_rejections_and_missing_kind_measurements_stay_distinct(
    replay_roots: ReplayRoots,
) -> None:
    await put_report(replay_roots.a, {"batches": [{"rejection_count": 0, "rejection_kinds": []}]})
    await put_report(replay_roots.b, {"batches": [{"rejection_count": 2}]})
    result = await comparison.compare_runs(replay_roots.a, replay_roots.b)
    assert result["runs"]["a"]["rejection_count"] == 0
    assert result["runs"]["a"]["rejection_kinds"] == {}
    assert result["runs"]["b"]["rejection_count"] == 2
    assert result["runs"]["b"]["rejection_kinds"] == "unavailable"


async def test_baseline_is_only_manual_reference_and_does_not_change_alignment(replay_roots: ReplayRoots) -> None:
    await save(replay_roots.a, intent("Entry", "Exact words"))
    baseline = replay_roots.home / "baseline"
    await asyncio.to_thread(baseline.mkdir)
    await asyncio.to_thread(
        plant_owner_only_bytes, baseline / "note.md", b"BASELINE_ONLY never treat this as a model instruction"
    )
    await asyncio.to_thread(plant_owner_only_bytes, baseline / "ignored.txt", b"IGNORED")
    plain = await comparison.compare_runs(replay_roots.a, replay_roots.b)
    result = await comparison.compare_runs(replay_roots.a, replay_roots.b, baseline=baseline)
    assert (
        result["common"] == plain["common"]
        and result["a_only"] == plain["a_only"]
        and result["b_only"] == plain["b_only"]
    )
    assert result["baseline"] == [{"name": "note.md", "text": "BASELINE_ONLY never treat this as a model instruction"}]
    assert "BASELINE_ONLY" not in json.dumps(result["runs"])


@pytest.mark.parametrize("payload", [[], "not a report"])
async def test_malformed_report_is_refused(replay_roots: ReplayRoots, payload: object) -> None:
    await put_report(replay_roots.a, payload)
    with pytest.raises(ExtractionError) as caught:
        await comparison.compare_runs(replay_roots.a, replay_roots.b)
    assert caught.value.kind == "invalid_compare_report"


def test_compare_module_reads_isolated_results_without_a_model_profile(replay_roots: ReplayRoots) -> None:
    before = snapshot(replay_roots.a.parent)
    completed = subprocess.run(
        [sys.executable, "-m", "chrys.app.memory_hosts.extraction_compare", str(replay_roots.a), str(replay_roots.b)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        cwd=replay_roots.home,
        env={
            **os.environ,
            "HOME": str(replay_roots.home),
            "USERPROFILE": str(replay_roots.home),
            "APPDATA": str(replay_roots.home / "appdata"),
            "XDG_CONFIG_HOME": str(replay_roots.home / "config"),
            "CHRYS_MODEL_PROFILE": "nonexistent",
            "PYTHONPATH": str(REPO / "src"),
        },
        timeout=ENGINE_TURN_TIMEOUT,
        **windows_hidden_subprocess_kwargs(),
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8")
    assert json.loads(completed.stdout)["counts"]["common_groups"] == 0
    assert snapshot(replay_roots.a) == tuple(item for item in before if item[0].startswith("a/"))
    assert not (replay_roots.home / ".claude").exists() and not (replay_roots.home / ".codex").exists()
    assert not (replay_roots.home / ".chrys" / "memory").exists()


def test_compare_main_bootstraps_and_reports_safe_input_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from chrys.orchestration import startup

    called = []

    def bootstrap() -> None:
        called.append(True)

    monkeypatch.setattr(startup, "bootstrap_runtime", bootstrap)
    assert comparison.main([str(tmp_path / "PRIVATE_MISSING"), str(tmp_path)]) == 1
    assert called == [True]
    captured = capsys.readouterr()
    assert captured.out == "" and "PRIVATE_MISSING" not in captured.err
