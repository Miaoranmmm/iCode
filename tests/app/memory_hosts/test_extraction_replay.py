# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Frozen replay uses immutable synthetic inputs and never touches extraction state."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest

from chrys.app.memory_hosts.claude_code.history import ClaudeHistoryReader
from chrys.app.memory_hosts.extract import ExtractionOptions, run_extraction
from chrys.app.memory_hosts.extraction_replay import ReplayManifest, load_manifest
from chrys.foundation.platform import get_platform
from chrys.service.memory.extraction import (
    PROMPT_VERSION,
    RENDER_VERSION,
    ExtractionBatch,
    ExtractionError,
    ExtractionResult,
    UsageSummary,
    build_batches,
)
from chrys.service.memory.operations import CatalogSnapshot, MemoryReader
from chrys.service.memory.project import bind_project
from chrys.service.profiles.models.schema import ModelProfile
from tests.support.secure_files import plant_owner_only_bytes
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for

SESSION = "10000000-0000-4000-8000-000000000001"
PROFILE = ModelProfile(
    "offline-profile",
    "Replay Responses",
    provider="deepseek-openai",
    api_style="responses",
    model_id="offline",
    max_context_tokens=100_000,
)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass
class NoSaveExecutor:
    calls: int = 0

    async def run(self, batch: ExtractionBatch) -> ExtractionResult:
        self.calls += 1
        assert batch.source_index.view.current_session is None
        assert batch.source_index.view.messages[0].fragments == ("Use clear public names.",)
        return ExtractionResult("completed", 1, 0, 0, 0, None, UsageSummary())


@pytest.fixture
async def frozen(tmp_path: Path) -> tuple[ExtractionOptions, dict[str, object]]:
    project_dir = tmp_path / "project"
    sources = tmp_path / "snapshots"
    await asyncio.to_thread(project_dir.mkdir)
    await asyncio.to_thread(sources.mkdir)
    source = sources / f"{SESSION}.jsonl"
    record = {
        "type": "user",
        "uuid": "synthetic-user",
        "parentUuid": None,
        "sessionId": SESSION,
        "entrypoint": "claude-desktop",
        "cwd": str(tmp_path / "removed-historical-workspace"),
        "origin": {"kind": "human"},
        "timestamp": "2026-01-01T00:00:00Z",
        "message": {"role": "user", "content": "Use clear public names."},
    }
    payload = (json.dumps(record) + "\n").encode()
    await asyncio.to_thread(plant_owner_only_bytes, source, payload)
    reference = tmp_path / "reference.txt"
    baseline = tmp_path / "baseline.txt"
    await asyncio.to_thread(plant_owner_only_bytes, reference, b"Reference-only rule.")
    await asyncio.to_thread(plant_owner_only_bytes, baseline, b"PRIVATE EVALUATION ONLY.")
    project = await asyncio.to_thread(bind_project, str(project_dir))
    scanned = await ClaudeHistoryReader(sources, replay=True, sessions=frozenset({SESSION})).read_files(
        project, (source,)
    )
    assert len(scanned.sessions) == 1
    batch = build_batches(project, scanned.sessions, 40_000)[0]
    manifest: dict[str, object] = {
        "version": 1,
        "prompt_version": PROMPT_VERSION,
        "render_version": RENDER_VERSION,
        "project_id": project.project_id,
        "material_budget": 40_000,
        "max_output_tokens": 16_384,
        "sources": [
            {
                "host": "claude-code",
                "session": SESSION,
                "path": source.relative_to(tmp_path).as_posix(),
                "sha256": sha(payload),
                "updated_ns": scanned.sessions[0].revision.updated_ns,
            }
        ],
        "references": [{"name": "AGENTS.md", "path": "reference.txt", "sha256": sha(b"Reference-only rule.")}],
        "baseline": [{"name": "auto-memory", "path": "baseline.txt", "sha256": sha(b"PRIVATE EVALUATION ONLY.")}],
        "initial_store": [],
        "batches": [
            {"sessions": [{"host": "claude-code", "session": SESSION}], "material_sha256": sha(batch.material.encode())}
        ],
    }
    path = tmp_path / "replay.json"
    await asyncio.to_thread(plant_owner_only_bytes, path, json.dumps(manifest).encode())
    return ExtractionOptions(project_dir, tmp_path / "isolated-memory", replay=path), manifest


async def test_replay_ignores_corrupt_watermarks_and_creates_no_state_or_lock(
    frozen: tuple[ExtractionOptions, dict[str, object]],
) -> None:
    options, _ = frozen
    await asyncio.to_thread(options.memory_dir.mkdir)
    state = options.memory_dir / "extraction.json"
    await asyncio.to_thread(plant_owner_only_bytes, state, b"deliberately invalid ignored state")
    runner = NoSaveExecutor()
    reports = [await run_extraction(options, PROFILE, executor=runner) for _ in range(2)]
    assert all((report.status, report.exit_code) == ("replay_completed", 0) for report in reports)
    assert all(
        asdict(report)["signals"]
        == {
            "candidate_sessions": 1,
            "annotated_sessions": 0,
            "annotations": {"user_interrupt": 0},
            "signals_unavailable_sessions": 0,
        }
        for report in reports
    )
    assert reports[0].run_id != reports[1].run_id
    assert runner.calls == 2
    assert await asyncio.to_thread(state.read_bytes) == b"deliberately invalid ignored state"
    assert await asyncio.to_thread(lambda: sorted(path.name for path in options.memory_dir.rglob("*"))) == [
        "extraction.json"
    ]
    assert "PRIVATE EVALUATION ONLY" not in json.dumps([asdict(report) for report in reports])


async def test_replay_dry_run_has_no_client_store_or_watermark_side_effect(
    frozen: tuple[ExtractionOptions, dict[str, object]],
) -> None:
    options, _ = frozen
    runner = NoSaveExecutor()
    report = await run_extraction(replace(options, dry_run=True), PROFILE, executor=runner)
    assert (report.status, report.exit_code, runner.calls) == ("replay_dry_run", 0, 0)
    assert asdict(report)["signals"] == {
        "candidate_sessions": 1,
        "annotated_sessions": 0,
        "annotations": {"user_interrupt": 0},
        "signals_unavailable_sessions": 0,
    }
    assert report.batches[0]["annotated_sessions"] == 0
    assert report.batches[0]["annotations"] == {"user_interrupt": 0}
    assert not await asyncio.to_thread(options.memory_dir.exists)
    assert report.batches[0]["sessions"] == [{"host": "claude-code", "session": SESSION}]


async def test_render_version_one_is_rejected_before_input_or_material_verification(
    frozen: tuple[ExtractionOptions, dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    options, manifest = frozen
    manifest["render_version"] = "1"
    await asyncio.to_thread(options.replay.write_text, json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ExtractionError) as error:
        await asyncio.to_thread(load_manifest, options.replay)
    assert error.value.kind == "invalid_replay"

    def verify_inputs(manifest: ReplayManifest, memory_dir: Path) -> None:
        pytest.fail("An obsolete render version must be rejected before verifying input hashes.")

    def verify_batches(manifest: ReplayManifest, batches: tuple[ExtractionBatch, ...]) -> None:
        pytest.fail("An obsolete render version must be rejected before verifying material hashes.")

    monkeypatch.setattr(ReplayManifest, "verify_inputs", verify_inputs)
    monkeypatch.setattr(ReplayManifest, "verify_batches", verify_batches)
    runner = NoSaveExecutor()
    report = await run_extraction(options, PROFILE, executor=runner)
    assert (report.status, report.exit_code, runner.calls) == ("invalid_replay", 1, 0)
    assert not await asyncio.to_thread(options.memory_dir.exists)


async def test_replay_dry_run_reports_cancellation_absorbed_by_initial_catalog(
    frozen: tuple[ExtractionOptions, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options, _ = frozen
    entered, release = asyncio.Event(), asyncio.Event()
    catalog = MemoryReader.catalog

    async def gated_catalog(reader: MemoryReader) -> CatalogSnapshot:
        entered.set()
        await release.wait()
        return await catalog(reader)

    monkeypatch.setattr(MemoryReader, "catalog", gated_catalog)
    runner = NoSaveExecutor()
    task = asyncio.create_task(run_extraction(replace(options, dry_run=True), PROFILE, executor=runner))
    try:
        await wait_for(lambda: entered.is_set() or task.done(), timeout=ENGINE_TURN_TIMEOUT)
        if task.done():
            await task
        assert entered.is_set() and not task.done()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        report = await task
        assert (report.status, report.exit_code, report.cancel_requested) == ("cancelled_before_catalog", 130, True)
        assert runner.calls == 0
        assert not await asyncio.to_thread(options.memory_dir.exists)
    finally:
        release.set()
        await task


@pytest.mark.parametrize(
    "field",
    [
        "prompt_version",
        "render_version",
        "source_hash",
        "reference_hash",
        "baseline_hash",
        "material_hash",
        "initial_store",
        "window",
        "output",
    ],
)
async def test_replay_rejects_any_frozen_input_or_render_mismatch_before_model(
    frozen: tuple[ExtractionOptions, dict[str, object]],
    field: str,
) -> None:
    options, manifest = frozen
    if field.endswith("_version"):
        manifest[field] = "future"
    elif field == "source_hash":
        manifest["sources"][0]["sha256"] = "0" * 64
    elif field == "reference_hash":
        manifest["references"][0]["sha256"] = "0" * 64
    elif field == "baseline_hash":
        manifest["baseline"][0]["sha256"] = "0" * 64
    elif field == "material_hash":
        manifest["batches"][0]["material_sha256"] = "0" * 64
    elif field == "initial_store":
        manifest["initial_store"] = [{"path": "global/expected.md", "sha256": "0" * 64}]
    elif field == "output":
        manifest["max_output_tokens"] = 2048
    await asyncio.to_thread(options.replay.write_text, json.dumps(manifest), encoding="utf-8")
    runner = NoSaveExecutor()
    profile = replace(PROFILE, max_context_tokens=50_000) if field == "window" else PROFILE
    report = await run_extraction(options, profile, executor=runner)
    assert report.exit_code == 1
    assert report.status in {"invalid_replay", "input_budget_exceeded"}
    assert runner.calls == 0
    assert not await asyncio.to_thread(options.memory_dir.exists)


async def test_replay_refuses_formal_memory_root_even_with_an_explicit_flag(
    frozen: tuple[ExtractionOptions, dict[str, object]],
) -> None:
    options, _ = frozen
    runner = NoSaveExecutor()
    report = await run_extraction(
        replace(options, memory_dir=get_platform().config_dir / "memory"), PROFILE, executor=runner
    )
    assert (report.status, report.exit_code, runner.calls) == ("invalid_replay", 1, 0)


async def test_reference_changed_after_initial_verification_cannot_enter_the_prompt(
    frozen: tuple[ExtractionOptions, dict[str, object]],
) -> None:
    options, _ = frozen
    manifest = await asyncio.to_thread(load_manifest, options.replay)
    await asyncio.to_thread(manifest.verify_inputs, options.memory_dir)
    await asyncio.to_thread(manifest.references[0].path.write_text, "Changed reference.", encoding="utf-8")
    with pytest.raises(ExtractionError, match="manifest or input does not match"):
        await asyncio.to_thread(manifest.read_references)


async def test_replay_cannot_change_the_frozen_batch_membership(
    frozen: tuple[ExtractionOptions, dict[str, object]],
) -> None:
    options, _ = frozen
    runner = NoSaveExecutor()
    report = await run_extraction(replace(options, sessions=frozenset({"not-in-manifest"})), PROFILE, executor=runner)
    assert (report.status, report.exit_code, runner.calls) == ("invalid_replay", 1, 0)


@pytest.mark.parametrize("bad_path", ["../escape.md", "/absolute.md", "extraction.json", "global/.lock"])
async def test_manifest_rejects_initial_store_paths_that_escape_or_name_progress(
    frozen: tuple[ExtractionOptions, dict[str, object]],
    bad_path: str,
) -> None:
    options, manifest = frozen
    manifest["initial_store"] = [{"path": bad_path, "sha256": "0" * 64}]
    await asyncio.to_thread(options.replay.write_text, json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ExtractionError, match="manifest or input does not match"):
        await asyncio.to_thread(load_manifest, options.replay)
