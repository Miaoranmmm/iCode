# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Freeze complete, private, replayable regression sets from the existing dry-run plan."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest

from chrys.app.memory_hosts import extract
from chrys.app.memory_hosts import extraction_freeze as freezing
from chrys.app.memory_hosts.claude_code.history import ClaudeHistoryReader
from chrys.app.memory_hosts.codex_history import CodexHistoryReader
from chrys.app.memory_hosts.extract import ExtractionOptions, HistoryReader, run_extraction
from chrys.app.memory_hosts.extraction_replay import FrozenSource, ReplayManifest, load_manifest
from chrys.foundation.platform import get_platform
from chrys.service.memory.extraction import DEFAULT_MATERIAL_PROFILE, ExtractionBatch, MaterialProfile, ScanResult
from chrys.service.memory.operations import CatalogSnapshot, MemoryReader
from chrys.service.memory.storage.store import MemoryScope
from tests.app.memory_hosts.history_fixtures import PARENT, SESSION, FixedCodex, FixedHolder, claude_user, write_jsonl
from tests.app.memory_hosts.test_codex_history import message, meta
from tests.app.memory_hosts.test_extraction_replay import PROFILE
from tests.support.secure_files import plant_owner_only_bytes
from tests.support.symlinks import symlink_or_skip
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for


@dataclass(frozen=True)
class Workspace:
    root: Path
    project: Path
    memory: Path
    claude: Path
    codex: Path
    target: Path
    baseline: Path

    def options(self, *, baseline_dir: Path | None = None) -> ExtractionOptions:
        return ExtractionOptions(
            self.project,
            self.memory,
            self.claude,
            self.codex,
            dry_run=True,
            freeze_to=self.target,
            baseline_dir=baseline_dir,
        )

    def readers(self) -> Mapping[str, HistoryReader]:
        return {
            "claude-code": ClaudeHistoryReader(self.claude, holder=FixedHolder()),
            "codex": CodexHistoryReader(self.codex, processes=FixedCodex()),
        }


@pytest.fixture
async def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    home = tmp_path / "home"
    await asyncio.to_thread(home.mkdir)
    for key in ("HOME", "USERPROFILE", "APPDATA"):
        monkeypatch.setenv(key, str(home))
    project, claude, codex, baseline = (tmp_path / name for name in ("project", "claude", "codex", "baseline"))
    for directory in (project, claude, codex, baseline):
        await asyncio.to_thread(directory.mkdir)
    await asyncio.to_thread(
        write_jsonl,
        claude / f"{SESSION}.jsonl",
        [claude_user("human", "Use clear public names.", cwd=str(project))],
    )
    await asyncio.to_thread(
        write_jsonl,
        codex / f"rollout-2026-11-01T10-00-00-{PARENT}.jsonl",
        [meta(project, id=PARENT), message("Keep errors actionable.")],
    )
    await asyncio.to_thread((project / "AGENTS.md").write_text, "Reference text used by this run.\n", encoding="utf-8")
    return Workspace(tmp_path, project, tmp_path / "memory", claude, codex, tmp_path / "regression", baseline)


@pytest.mark.parametrize("wire", ["responses", "messages"])
async def test_freeze_round_trip_preserves_the_report_and_replays_the_same_batches(
    workspace: Workspace, wire: str
) -> None:
    original_profile = replace(
        PROFILE,
        provider="anthropic" if wire == "messages" else "deepseek-openai",
        chat_options='{"extra_body":{"cache_control":{"type":"ephemeral"}}}'
        if wire == "messages"
        else '{"reasoning":{"effort":"high"}}',
    )
    await asyncio.to_thread(
        write_jsonl,
        workspace.claude / f"{SESSION}.jsonl",
        [claude_user("long", "Keep the full original. " + "x" * 2200, cwd=str(workspace.project))],
    )
    options = workspace.options()
    report = await run_extraction(options, original_profile, readers=workspace.readers())
    assert (report.status, report.exit_code) == ("dry_run", 0)
    manifest = await asyncio.to_thread(load_manifest, workspace.target / "manifest.json")
    await asyncio.to_thread(manifest.verify_inputs, workspace.memory)
    saved = json.loads(await asyncio.to_thread((workspace.target / "report.json").read_text, encoding="utf-8"))
    assert saved == json.loads(json.dumps(asdict(report)))
    assert (
        await asyncio.to_thread((workspace.target / "report.json").read_text, encoding="utf-8")
        == json.dumps(asdict(report), ensure_ascii=True) + "\n"
    )
    assert saved["profile"]["wire"] == wire
    assert saved["profile"]["max_turns"] == 24
    assert saved["profile"]["batch_timeout_seconds"] == 600
    assert saved["diagnostics"] and saved["signals"] == asdict(report.signals)
    assert saved["batches"] == report.batches
    assert sum(batch["omitted_characters"] for batch in saved["batches"]) > 0
    for source in manifest.sources:
        payload = await asyncio.to_thread(source.path.read_bytes)
        assert hashlib.sha256(payload).hexdigest() == source.sha256
        assert source.path == workspace.target / "sources" / source.key.host / f"{source.key.session}.jsonl"
        original = next(
            path for path in (workspace.claude if source.key.host == "claude-code" else workspace.codex).glob("*.jsonl")
        )
        assert source.updated_ns == (await asyncio.to_thread(original.stat)).st_mtime_ns
    assert len(manifest.sources) == 2 and len(manifest.batches) == 2
    assert (
        await asyncio.to_thread((workspace.target / "references" / "AGENTS.md").read_text, encoding="utf-8")
        == "Reference text used by this run.\n"
    )
    assert manifest.baseline == () and manifest.initial_store == ()
    assert (
        manifest.material_profile.name,
        manifest.render_version,
        manifest.prompt_version,
        manifest.tuning_version,
    ) == ("full-v4", "3", "3", "v0")
    assert saved["profile"]["material_profile"] == "full-v4"
    replay = await run_extraction(
        ExtractionOptions(workspace.project, workspace.memory, replay=workspace.target / "manifest.json", dry_run=True),
        original_profile,
    )
    assert (replay.status, replay.exit_code) == ("replay_dry_run", 0)
    assert [batch["sessions"] for batch in replay.batches] == [batch["sessions"] for batch in report.batches]
    assert not await asyncio.to_thread(workspace.memory.exists)
    assert not await asyncio.to_thread(lambda: list(workspace.root.glob(".regression*")))


async def test_source_revision_change_between_scan_and_copy_aborts_the_entire_freeze(workspace: Workspace) -> None:
    class ChangingReader(ClaudeHistoryReader):
        async def scan(self, project: MemoryScope) -> ScanResult:
            scanned = await super().scan(project)
            path = workspace.claude / f"{SESSION}.jsonl"
            payload = await asyncio.to_thread(path.read_bytes)
            await asyncio.to_thread(path.write_bytes, payload + b'{"type":"metadata"}\n')
            return scanned

    readers = dict(workspace.readers())
    readers["claude-code"] = ChangingReader(workspace.claude, holder=FixedHolder())
    report = await run_extraction(workspace.options(), PROFILE, readers=readers)
    assert (report.status, report.exit_code) == ("source_changed", 1)
    assert not await asyncio.to_thread(workspace.target.exists)
    assert not await asyncio.to_thread(lambda: list(workspace.root.glob(".regression*")))


@pytest.mark.parametrize("filename", ["global/123456789abc.md", "global/changes.jsonl", "notes.txt"])
async def test_nonempty_initial_store_is_refused_before_freezing(workspace: Workspace, filename: str) -> None:
    path = workspace.memory / filename
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(plant_owner_only_bytes, path, b"Synthetic existing store content.")
    report = await run_extraction(workspace.options(), PROFILE, readers=workspace.readers())
    assert report.exit_code == 1
    assert "empty" in json.dumps(report.diagnostics).lower()
    assert not await asyncio.to_thread(workspace.target.exists)
    assert await asyncio.to_thread(path.read_bytes) == b"Synthetic existing store content."


async def test_existing_state_files_are_ignored_and_never_rewritten(workspace: Workspace) -> None:
    state = workspace.memory / "extraction.json"
    await asyncio.to_thread(state.parent.mkdir, parents=True)
    await asyncio.to_thread(plant_owner_only_bytes, state, b'{"version":1,"projects":{}}')
    lock = workspace.memory / "projects" / "synthetic" / ".extract.lock"
    await asyncio.to_thread(lock.parent.mkdir, parents=True)
    await asyncio.to_thread(plant_owner_only_bytes, lock, b"Synthetic lock marker.")
    before = {state: await asyncio.to_thread(state.read_bytes), lock: await asyncio.to_thread(lock.read_bytes)}
    report = await run_extraction(workspace.options(), PROFILE, readers=workspace.readers())
    assert report.exit_code == 0
    manifest = await asyncio.to_thread(load_manifest, workspace.target / "manifest.json")
    assert manifest.initial_store == ()
    assert {path: await asyncio.to_thread(path.read_bytes) for path in before} == before


@pytest.mark.parametrize("root", ["formal", "memory", "baseline", "claude", "codex", "project"])
async def test_target_inside_every_forbidden_root_is_refused_before_writing(workspace: Workspace, root: str) -> None:
    roots = {
        "formal": get_platform().config_dir / "memory",
        "memory": workspace.memory,
        "baseline": workspace.baseline,
        "claude": workspace.claude,
        "codex": workspace.codex,
        "project": workspace.project,
    }
    target = roots[root] / "frozen"
    options = replace(workspace.options(baseline_dir=workspace.baseline), freeze_to=target)
    report = await run_extraction(options, PROFILE, readers=workspace.readers())
    assert report.exit_code == 1
    assert not await asyncio.to_thread(target.exists)
    assert not await asyncio.to_thread(lambda: list(roots[root].glob(".frozen*")))


async def test_target_containing_a_not_yet_created_memory_root_is_also_refused(workspace: Workspace) -> None:
    options = replace(workspace.options(), memory_dir=workspace.target / "memory")
    report = await run_extraction(options, PROFILE, readers=workspace.readers())
    assert report.exit_code == 1
    assert not await asyncio.to_thread(workspace.target.exists)


async def test_existing_target_is_never_replaced(workspace: Workspace) -> None:
    await asyncio.to_thread(workspace.target.mkdir)
    marker = workspace.target / "keep.txt"
    await asyncio.to_thread(marker.write_text, "Existing target.", encoding="utf-8")
    report = await run_extraction(workspace.options(), PROFILE, readers=workspace.readers())
    assert report.exit_code == 1
    assert await asyncio.to_thread(marker.read_text, encoding="utf-8") == "Existing target."
    assert await asyncio.to_thread(lambda: sorted(path.name for path in workspace.target.iterdir())) == ["keep.txt"]


async def test_symlinked_target_parent_is_refused_without_writing_through_it(workspace: Workspace) -> None:
    real = workspace.root / "real-parent"
    alias = workspace.root / "linked-parent"
    await asyncio.to_thread(real.mkdir)
    await asyncio.to_thread(symlink_or_skip, alias, real, target_is_directory=True)
    options = replace(workspace.options(), freeze_to=alias / "frozen")
    report = await run_extraction(options, PROFILE, readers=workspace.readers())
    assert report.exit_code == 1
    assert not await asyncio.to_thread(lambda: list(real.iterdir()))


async def test_baseline_types_are_copied_without_entering_material_or_reports(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    for kind in ("feedback", "user", "project", "reference"):
        await asyncio.to_thread(
            (workspace.baseline / f"BASELINE_FILENAME_{kind}.md").write_text,
            f"---\nmetadata:\n  type: {kind}\n---\nBASELINE_BODY_{kind}\n",
            encoding="utf-8",
        )
    await asyncio.to_thread((workspace.baseline / "MEMORY.md").write_text, "BASELINE_NO_FRONTMATTER", encoding="utf-8")
    nested = workspace.baseline / "nested"
    await asyncio.to_thread(nested.mkdir)
    await asyncio.to_thread(
        (nested / "ignored.md").write_text, "---\nmetadata:\n  type: user\n---\nNESTED_BASELINE", encoding="utf-8"
    )
    prompts: list[str] = []
    original = extract.render_prompt

    def render(
        batch: ExtractionBatch,
        catalog: str,
        references: tuple[tuple[str, str], ...],
        *,
        profile: MaterialProfile = DEFAULT_MATERIAL_PROFILE,
        related_entries: str = "",
        withdrawn_entries: str = "",
    ) -> str:
        prompt = original(
            batch,
            catalog,
            references,
            profile=profile,
            related_entries=related_entries,
            withdrawn_entries=withdrawn_entries,
        )
        prompts.append(prompt)
        return prompt

    monkeypatch.setattr(extract, "render_prompt", render)
    report = await run_extraction(
        workspace.options(baseline_dir=workspace.baseline), PROFILE, readers=workspace.readers()
    )
    assert report.exit_code == 0
    manifest = await asyncio.to_thread(load_manifest, workspace.target / "manifest.json")
    assert {item.name for item in manifest.baseline} == {"BASELINE_FILENAME_feedback.md", "BASELINE_FILENAME_user.md"}
    assert report.baseline_counts == {"feedback": 1, "user": 1}
    assert prompts and all("BASELINE_" not in prompt for prompt in prompts)
    assert "BASELINE_" not in json.dumps(asdict(report))
    assert "BASELINE_" not in await asyncio.to_thread((workspace.target / "report.json").read_text, encoding="utf-8")
    for item in manifest.baseline:
        payload = await asyncio.to_thread(item.path.read_bytes)
        assert hashlib.sha256(payload).hexdigest() == item.sha256


async def test_references_come_from_the_read_used_for_planning(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = MemoryReader.catalog

    async def catalog(reader: MemoryReader) -> CatalogSnapshot:
        await asyncio.to_thread((workspace.project / "AGENTS.md").write_text, "Later replacement.\n", encoding="utf-8")
        return await original(reader)

    monkeypatch.setattr(MemoryReader, "catalog", catalog)
    report = await run_extraction(workspace.options(), PROFILE, readers=workspace.readers())
    assert report.exit_code == 0
    assert (
        await asyncio.to_thread((workspace.target / "references" / "AGENTS.md").read_text, encoding="utf-8")
        == "Reference text used by this run.\n"
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
async def test_every_frozen_file_and_directory_is_owner_only(workspace: Workspace) -> None:
    report = await run_extraction(workspace.options(), PROFILE, readers=workspace.readers())
    assert report.exit_code == 0
    for path in (workspace.target, *await asyncio.to_thread(lambda: list(workspace.target.rglob("*")))):
        info = await asyncio.to_thread(path.stat)
        assert stat.S_IMODE(info.st_mode) == (0o700 if stat.S_ISDIR(info.st_mode) else 0o600)


async def test_failed_self_check_removes_the_entire_staged_result(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject(manifest: ReplayManifest, batches: tuple[ExtractionBatch, ...]) -> None:
        raise extract.ExtractionError("invalid_replay", "Synthetic self-check rejection.")

    monkeypatch.setattr(ReplayManifest, "verify_batches", reject)
    report = await run_extraction(workspace.options(), PROFILE, readers=workspace.readers())
    assert report.exit_code == 1
    assert not await asyncio.to_thread(workspace.target.exists)
    assert not await asyncio.to_thread(lambda: list(workspace.root.glob(".regression*")))


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--freeze-to", "frozen"], "--freeze-to requires --dry-run"),
        (
            ["--freeze-to", "frozen", "--dry-run", "--replay", "old.json", "--memory-dir", "isolated"],
            "--freeze-to cannot be used with --replay",
        ),
        (["--baseline-dir", "baseline", "--dry-run"], "--baseline-dir requires --freeze-to"),
    ],
)
def test_freeze_cli_constraints_are_argparse_errors_before_bootstrap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], flags: list[str], message: str
) -> None:
    with pytest.raises(SystemExit) as error:
        extract.main(["-C", str(tmp_path), *flags])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


async def test_only_sources_in_the_actual_planned_batches_are_copied(workspace: Workspace) -> None:
    for index in range(1, 31):
        session = f"50000000-0000-4000-8000-{index:012d}"
        await asyncio.to_thread(
            write_jsonl,
            workspace.claude / f"{session}.jsonl",
            [claude_user("human", "Keep names clear.", sessionId=session, cwd=str(workspace.project))],
        )
    options = replace(workspace.options(), codex_source=None)
    report = await run_extraction(
        options,
        PROFILE,
        readers={"claude-code": ClaudeHistoryReader(workspace.claude, holder=FixedHolder())},
    )
    assert report.exit_code == 0
    assert report.signals.candidate_sessions == 31
    manifest = await asyncio.to_thread(load_manifest, workspace.target / "manifest.json")
    planned = {item["session"] for batch in report.batches for item in batch["sessions"]}
    assert len(manifest.sources) == len(planned) == 30
    assert {source.key.session for source in manifest.sources} == planned
    copied = await asyncio.to_thread(lambda: list((workspace.target / "sources" / "claude-code").glob("*.jsonl")))
    assert {path.stem for path in copied} == planned


async def test_a_racing_empty_target_is_not_overwritten(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    original = freezing._publish_directory
    identity: list[tuple[int, int]] = []

    def publish(stage: Path, target: Path) -> None:
        target.mkdir(mode=0o700)
        info = target.stat()
        identity.append((info.st_dev, info.st_ino))
        original(stage, target)

    monkeypatch.setattr(freezing, "_publish_directory", publish)
    report = await run_extraction(workspace.options(), PROFILE, readers=workspace.readers())
    assert (report.status, report.exit_code) == ("freeze_target_exists", 1)
    info = await asyncio.to_thread(workspace.target.stat)
    assert identity == [(info.st_dev, info.st_ino)]
    assert not await asyncio.to_thread(lambda: list(workspace.target.iterdir()))
    assert not await asyncio.to_thread(lambda: list(workspace.root.glob(".regression*")))


@pytest.mark.parametrize("phase", ["source", "report"])
async def test_a_write_failure_never_publishes_a_partial_regression_set(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    original = freezing.atomic_write_owner_only_bytes
    writes = 0

    def write(path: Path, payload: bytes, *, create_parents: bool = True) -> None:
        nonlocal writes
        writes += 1
        if (phase == "source" and writes == 2) or (phase == "report" and path.name == "report.json"):
            raise OSError("Synthetic freeze write failure.")
        original(path, payload, create_parents=create_parents)

    monkeypatch.setattr(freezing, "atomic_write_owner_only_bytes", write)
    report = await run_extraction(workspace.options(), PROFILE, readers=workspace.readers())
    assert (report.status, report.exit_code) == ("freeze_write_failed", 1)
    assert not await asyncio.to_thread(workspace.target.exists)
    assert not await asyncio.to_thread(lambda: list(workspace.root.glob(".regression*")))


@pytest.mark.parametrize("phase", ["copy", "publish"])
async def test_cancelled_freeze_drains_its_worker_and_removes_the_result(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    entered, release = threading.Event(), threading.Event()
    read = freezing._read_source
    publish = freezing._publish

    def gate() -> None:
        entered.set()
        if not release.wait(ENGINE_TURN_TIMEOUT):
            raise TimeoutError("The test did not release the freeze worker.")

    def blocked_read(source: FrozenSource) -> bytes:
        gate()
        return read(source)

    def blocked_publish(
        stage: Path, inputs: freezing.FreezeInputs, report: Mapping[str, object]
    ) -> freezing.FreezePublication:
        gate()
        return publish(stage, inputs, report)

    if phase == "copy":
        monkeypatch.setattr(freezing, "_read_source", blocked_read)
    else:
        monkeypatch.setattr(freezing, "_publish", blocked_publish)
    task = asyncio.create_task(run_extraction(workspace.options(), PROFILE, readers=workspace.readers()))
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
        assert (report.status, report.exit_code, report.cancel_requested) == ("cancelled_before_freeze", 130, True)
        assert not await asyncio.to_thread(workspace.target.exists)
        assert not await asyncio.to_thread(lambda: list(workspace.root.glob(".regression*")))
    finally:
        release.set()
        await task


@pytest.mark.parametrize("root", ["memory", "baseline", "claude", "codex"])
async def test_forbidden_root_aliases_are_compared_by_real_path(workspace: Workspace, root: str) -> None:
    real = {
        "memory": workspace.memory,
        "baseline": workspace.baseline,
        "claude": workspace.claude,
        "codex": workspace.codex,
    }[root]
    await asyncio.to_thread(real.mkdir, exist_ok=True)
    alias = workspace.root / f"{root}-alias"
    await asyncio.to_thread(symlink_or_skip, alias, real, target_is_directory=True)
    options = workspace.options(baseline_dir=workspace.baseline)
    if root == "memory":
        options = replace(options, memory_dir=alias)
    elif root == "baseline":
        options = replace(options, baseline_dir=alias)
    elif root == "claude":
        options = replace(options, claude_source=alias)
    else:
        options = replace(options, codex_source=alias)
    target = real / "frozen"
    report = await run_extraction(replace(options, freeze_to=target), PROFILE, readers=workspace.readers())
    assert (report.status, report.exit_code) == ("invalid_freeze_target", 1)
    assert not await asyncio.to_thread(target.exists)


async def test_malformed_baseline_frontmatter_aborts_without_reporting_its_name_or_text(workspace: Workspace) -> None:
    await asyncio.to_thread(
        (workspace.baseline / "BASELINE_PRIVATE_NAME.md").write_text,
        "---\nmetadata: [ BASELINE_PRIVATE_TEXT\n---\n",
        encoding="utf-8",
    )
    report = await run_extraction(
        workspace.options(baseline_dir=workspace.baseline), PROFILE, readers=workspace.readers()
    )
    assert (report.status, report.exit_code) == ("baseline_invalid", 1)
    assert "BASELINE_PRIVATE" not in json.dumps(asdict(report))
    assert not await asyncio.to_thread(workspace.target.exists)
    assert not await asyncio.to_thread(lambda: list(workspace.root.glob(".regression*")))


async def test_an_empty_plan_is_not_published_as_an_unreplayable_manifest(workspace: Workspace) -> None:
    await asyncio.to_thread(
        write_jsonl, workspace.claude / f"{SESSION}.jsonl", [{"type": "last-prompt", "leafUuid": None}]
    )
    options = replace(workspace.options(), codex_source=None)
    report = await run_extraction(
        options, PROFILE, readers={"claude-code": ClaudeHistoryReader(workspace.claude, holder=FixedHolder())}
    )
    assert (report.status, report.exit_code) == ("freeze_empty", 1)
    assert not await asyncio.to_thread(workspace.target.exists)
