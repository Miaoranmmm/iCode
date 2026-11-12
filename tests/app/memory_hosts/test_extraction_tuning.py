# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Versioned tuning reaches the planner, requests and frozen replay contract."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from chrys.app.memory_hosts import extract
from chrys.app.memory_hosts.extract import run_extraction
from chrys.app.memory_hosts.extraction_replay import text_digest
from chrys.foundation.platform import get_platform
from chrys.service.memory.extraction import FULL_V3, USER_V2, ExtractionError, tuned_material_profile
from chrys.service.memory.tuning import V0, propose
from tests.app.memory_hosts.test_extract import PROFILE, Executor, Reader, result
from tests.app.memory_hosts.test_extract import workspace as workspace
from tests.app.memory_hosts.test_extraction_freeze import workspace as _freeze_workspace
from tests.app.memory_hosts.test_extraction_replay import frozen as frozen

freeze_workspace = _freeze_workspace


def test_tuning_changes_only_replaceable_prompt_segments_and_types() -> None:
    tuned = replace(
        V0, extraction_types=("intent",), task_description="Synthetic task.", do_not_save="Synthetic exclusions."
    )
    profile = tuned_material_profile(FULL_V3, tuned)
    assert profile.allowed_types == ("intent",)
    assert profile.prompt.startswith("Synthetic task.")
    assert "Synthetic exclusions." in profile.prompt
    # The verifier contract remains verbatim even with both editable segments replaced.
    verifier = next(line for line in FULL_V3.prompt.splitlines() if line.startswith("7. Copy"))
    assert verifier in profile.prompt
    assert tuned_material_profile(FULL_V3, V0) == FULL_V3
    assert tuned_material_profile(USER_V2, V0) == USER_V2


@pytest.mark.parametrize("base", [USER_V2, FULL_V3], ids=["user-v2", "full-v3"])
def test_prompt_slots_replace_each_profiles_own_sentence_once(base) -> None:
    tuned = replace(V0, do_not_save="Slot one.", pasted_text="Slot two.", body_limits="Slot three.")
    profile = tuned_material_profile(base, tuned)
    for old, new in (
        (V0.do_not_save, "Slot one."),
        (base.pasted_text_sentence, "Slot two."),
        (base.body_limits_sentence, "Slot three."),
    ):
        assert old not in profile.prompt
        assert profile.prompt.count(new) == 1
    # Everything outside the three sentences is byte-identical.
    restored = (
        profile.prompt.replace("Slot one.", V0.do_not_save)
        .replace("Slot two.", base.pasted_text_sentence)
        .replace("Slot three.", base.body_limits_sentence)
    )
    assert restored == base.prompt
    # Only the slot that is set changes.
    only_body = tuned_material_profile(base, replace(V0, body_limits="Slot three."))
    assert base.pasted_text_sentence in only_body.prompt and V0.do_not_save in only_body.prompt


@pytest.mark.parametrize("base", [USER_V2, FULL_V3], ids=["user-v2", "full-v3"])
def test_one_slots_new_text_is_never_taken_for_another_slots_target(base) -> None:
    tuned = replace(V0, do_not_save=base.body_limits_sentence, body_limits="Slot three.")
    profile = tuned_material_profile(base, tuned)
    assert profile.prompt.count(base.body_limits_sentence) == 1
    assert profile.prompt.count("Slot three.") == 1 and V0.do_not_save not in profile.prompt


@pytest.mark.parametrize(
    "tuned",
    [replace(V0, pasted_text="x"), replace(V0, body_limits="x"), replace(V0, do_not_save="x")],
    ids=["pasted_text", "body_limits", "do_not_save"],
)
@pytest.mark.parametrize("target", ["missing", "repeated"])
def test_a_set_slot_whose_sentence_is_not_exactly_once_is_refused(tuned, target: str) -> None:
    if target == "missing":
        base = replace(FULL_V3, prompt="Task.\n\nNo tunable sentences here.")
    else:
        repeated = "\n".join((FULL_V3.pasted_text_sentence, FULL_V3.body_limits_sentence, V0.do_not_save))
        base = replace(FULL_V3, prompt=FULL_V3.prompt + "\n" + repeated)
    with pytest.raises(ExtractionError) as caught:
        tuned_material_profile(base, tuned)
    assert caught.value.kind == "invalid_tuning"


async def test_active_tuning_limits_plan_and_is_read_once(workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    options, sources = workspace
    sources = (sources[0], replace(sources[1], key=replace(sources[1].key, host="codex")))
    tuned = replace(V0, version="v1", batch_cap=1, max_turns=7, max_output_tokens=2048, batch_timeout_seconds=30)
    calls: list[Path] = []

    def active(config_dir: Path):
        calls.append(config_dir)
        return tuned

    monkeypatch.setattr(extract, "load_active", active)
    report = await run_extraction(replace(options, dry_run=True), PROFILE, readers={"claude-code": Reader(sources)})
    assert report.exit_code == 0
    assert calls == [get_platform().config_dir]
    assert report.tuning_version == "v1"
    assert report.profile["max_turns"] == 7
    assert report.profile["max_output_tokens"] == 2048
    assert report.profile["batch_timeout_seconds"] == 30
    assert report.batches[0]["material_sha256"]
    assert len(report.batches) == 1


async def test_strict_replay_uses_manifest_version_instead_of_active(frozen, monkeypatch: pytest.MonkeyPatch) -> None:
    options, manifest = frozen
    active = replace(V0, version="v1", material_budget=1, max_output_tokens=256)
    monkeypatch.setattr(extract, "load_active", lambda config_dir: active)
    report = await run_extraction(replace(options, dry_run=True), PROFILE)
    assert (report.exit_code, report.status, report.tuning_version) == (0, "replay_dry_run", "v0")
    assert report.profile["max_output_tokens"] == 16384
    assert report.batches[0]["material_sha256"] == manifest["batches"][0]["material_sha256"]
    assert not report.material_changed
    assert not options.memory_dir.exists()


async def test_candidate_replay_replans_and_marks_changed_without_watermarks(frozen, tmp_path: Path) -> None:
    options, manifest = frozen
    candidate = replace(V0, material_budget=600)
    version = await asyncio.to_thread(propose, get_platform().config_dir, candidate)
    # The source is unchanged, but a different material budget is allowed in candidate mode.
    report = await run_extraction(replace(options, dry_run=True, tuning=str(version.version)), PROFILE)
    assert (report.exit_code, report.status) == (0, "replay_dry_run")
    assert report.material_changed is True
    assert report.tuning_version == version.version
    assert report.batches[0]["material_sha256"] == manifest["batches"][0]["material_sha256"]
    assert report.batches[0]["estimated_material_tokens"] <= 600
    assert not options.memory_dir.exists()


async def test_candidate_replay_keeps_source_hash_validation(frozen) -> None:
    options, manifest = frozen
    source = options.replay.parent / manifest["sources"][0]["path"]
    await asyncio.to_thread(source.write_text, "changed", encoding="utf-8")
    report = await run_extraction(replace(options, dry_run=True, tuning="0"), PROFILE)
    assert (report.status, report.exit_code) == ("invalid_replay", 1)
    assert not options.memory_dir.exists()


async def test_candidate_reports_actual_index_and_batch_digest(workspace) -> None:
    options, sources = workspace
    tuned = replace(V0, version="v1", material_budget=300)
    runner = Executor(result())
    report = await run_extraction(
        options, PROFILE, readers={"claude-code": Reader(sources)}, executor=runner, tuning=tuned
    )
    assert report.exit_code == 0
    assert report.batches[0]["material_sha256"] == text_digest(runner.batches[0].material)
    assert report.batches[0]["displayed_messages"] == len(runner.batches[0].source_index.view.messages)
    assert "Synthetic" not in json.dumps(report.profile)


async def test_nonzero_tuning_freeze_and_replay_ignore_configuration_files(freeze_workspace) -> None:
    workspace = freeze_workspace
    tuned = await asyncio.to_thread(propose, get_platform().config_dir, replace(V0, material_budget=1500))
    report = await run_extraction(workspace.options(), PROFILE, readers=workspace.readers(), tuning=tuned)
    assert report.exit_code == 0
    manifest = json.loads((workspace.target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tuning_version"] == tuned.version
    assert manifest["material_budget"] == 1500
    assert manifest["initial_store"] == []
    replay_options = replace(workspace.options(), freeze_to=None, replay=workspace.target / "manifest.json")
    replay = await run_extraction(replay_options, PROFILE)
    assert replay.exit_code == 0 and replay.tuning_version == tuned.version
    assert [batch["material_sha256"] for batch in replay.batches] == [
        batch["material_sha256"] for batch in report.batches
    ]
    assert not workspace.memory.exists()


async def test_candidate_file_changes_material_and_rebuilds_displayed_sources(freeze_workspace, tmp_path: Path) -> None:
    from tests.app.memory_hosts.history_fixtures import SESSION, claude_user, write_jsonl

    workspace = freeze_workspace
    await asyncio.to_thread(
        write_jsonl,
        workspace.claude / f"{SESSION}.jsonl",
        [
            claude_user("first", "甲" * 600, cwd=str(workspace.project)),
            claude_user("last", "乙" * 600, parentUuid="first", cwd=str(workspace.project)),
        ],
    )
    frozen_report = await run_extraction(workspace.options(), PROFILE, readers=workspace.readers())
    assert frozen_report.exit_code == 0
    candidate = tmp_path / "candidate.json"
    candidate.write_text('{"material_budget":600}', encoding="utf-8")
    options = replace(
        workspace.options(), freeze_to=None, replay=workspace.target / "manifest.json", tuning=str(candidate)
    )
    report = await run_extraction(options, PROFILE)
    assert report.exit_code == 0 and report.material_changed
    assert report.tuning_version.startswith("file:")
    actual = next(batch for batch in report.batches if batch["sessions"][0]["host"] == "claude-code")
    original = next(batch for batch in frozen_report.batches if batch["sessions"][0]["host"] == "claude-code")
    assert actual["material_sha256"] != original["material_sha256"]
    assert actual["displayed_messages"] == 1
    assert actual["omitted_messages"] == 1
    assert not workspace.memory.exists()
