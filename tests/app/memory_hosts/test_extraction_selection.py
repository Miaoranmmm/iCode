# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Offline selection comparison reads only frozen inputs and reports counts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from chrys.app.memory_hosts import extraction_selection as selection
from chrys.foundation.platform import get_platform
from chrys.foundation.platform.process import windows_hidden_subprocess_kwargs
from chrys.service.memory.extraction import ExtractionError
from tests.app.memory_hosts.history_fixtures import SESSION, claude_assistant, claude_user, write_jsonl
from tests.support.secure_files import plant_owner_only_bytes
from tests.support.waiting import ENGINE_TURN_TIMEOUT

REPO = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class FrozenInputs:
    manifest: Path
    source: Path
    home: Path
    before: tuple[tuple[str, bytes], ...]


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
async def frozen_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FrozenInputs:
    home = tmp_path / "home"
    source_dir = tmp_path / "frozen" / "sources" / "claude-code"
    await asyncio.to_thread(source_dir.mkdir, parents=True)
    await asyncio.to_thread(home.mkdir)
    for name in ("HOME", "USERPROFILE", "APPDATA", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(name, str(home / name))
    source = source_dir / f"{SESSION}.jsonl"
    records = [claude_user("u", "PRIVATE_USER keep names clear.", cwd=str(tmp_path / "no-live-project"))]
    parent = "u"
    for index in range(3):
        call_id = f"call-{index}"
        assistant_id = f"assistant-{index}"
        content: list[dict[str, object]] = []
        if index == 0:
            content.append({"type": "text", "text": "PRIVATE_ASSISTANT " + "甲" * 2500})
        content.append({"type": "tool_use", "id": call_id, "name": "Shell", "input": {"command": "synthetic"}})
        records.append(claude_assistant(assistant_id, parent, message={"id": f"response-{index}", "content": content}))
        result_id = f"result-{index}"
        block: dict[str, object] = {
            "type": "tool_result",
            "tool_use_id": call_id,
            "content": "PRIVATE_TOOL " + "乙" * 2500,
        }
        if index < 2:
            block["is_error"] = True
        records.append(
            {
                "type": "user",
                "uuid": result_id,
                "parentUuid": assistant_id,
                "sessionId": SESSION,
                "entrypoint": "claude-desktop",
                "message": {"content": [block]},
            }
        )
        parent = result_id
    records.append(
        claude_assistant(
            "conclusion",
            parent,
            message={
                "id": "conclusion-response",
                "content": [{"type": "text", "text": "PRIVATE_CONCLUSION " + "丙" * 2500}],
            },
        )
    )
    await asyncio.to_thread(write_jsonl, source, records)
    reference = source_dir.parents[1] / "reference.txt"
    baseline = source_dir.parents[1] / "baseline.txt"
    await asyncio.to_thread(plant_owner_only_bytes, reference, b"PRIVATE_REFERENCE")
    await asyncio.to_thread(plant_owner_only_bytes, baseline, b"PRIVATE_BASELINE")
    payload = await asyncio.to_thread(source.read_bytes)
    revision = await asyncio.to_thread(source.stat)
    manifest = source_dir.parents[1] / "manifest.json"
    data = {
        "version": 1,
        "render_version": "2",
        "prompt_version": "1",
        "project_id": "0123456789abcdef",
        "material_budget": 600,
        "max_output_tokens": 16384,
        "sources": [
            {
                "host": "claude-code",
                "session": SESSION,
                "path": source.relative_to(manifest.parent).as_posix(),
                "sha256": digest(payload),
                "updated_ns": revision.st_mtime_ns,
            }
        ],
        "references": [{"name": "AGENTS.md", "path": "reference.txt", "sha256": digest(b"PRIVATE_REFERENCE")}],
        "baseline": [{"name": "synthetic-baseline", "path": "baseline.txt", "sha256": digest(b"PRIVATE_BASELINE")}],
        "initial_store": [],
        "batches": [{"sessions": [{"host": "claude-code", "session": SESSION}], "material_sha256": "0" * 64}],
    }
    await asyncio.to_thread(plant_owner_only_bytes, manifest, json.dumps(data).encode())
    before = await asyncio.to_thread(
        lambda: tuple(
            sorted(
                (path.relative_to(manifest.parent).as_posix(), path.read_bytes())
                for path in manifest.parent.rglob("*")
                if path.is_file()
            )
        )
    )
    return FrozenInputs(manifest, source, home, before)


async def test_comparison_reports_three_policies_with_source_omission_counts_only(
    frozen_selection: FrozenInputs,
) -> None:
    report = await selection.compare_selection(frozen_selection.manifest)
    assert report["material_profile"] == "full-v4"
    assert report["material_budget"] == 600
    assert report["source_sessions"] == 1
    assert set(report["policies"]) == {"chronological", "head_tail", "signal_first"}
    for counts in report["policies"].values():
        assert counts == {
            "batch_count": 1,
            "omitted_tool_results": 3,
            "omitted_error_results": 2,
            "omitted_signal_neighbors": 4,
            "omitted_assistant_messages": 2,
        }
    text = json.dumps(report)
    for forbidden in (
        SESSION,
        "assistant-",
        "result-",
        "PRIVATE_",
        "甲",
        "乙",
        "丙",
        "synthetic-baseline",
        str(frozen_selection.manifest.parent),
    ):
        assert forbidden not in text
    after = await asyncio.to_thread(
        lambda: tuple(
            sorted(
                (path.relative_to(frozen_selection.manifest.parent).as_posix(), path.read_bytes())
                for path in frozen_selection.manifest.parent.rglob("*")
                if path.is_file()
            )
        )
    )
    assert after == frozen_selection.before


async def test_changed_frozen_source_is_refused(frozen_selection: FrozenInputs) -> None:
    await asyncio.to_thread(frozen_selection.source.write_text, "changed", encoding="utf-8")
    with pytest.raises(ExtractionError) as error:
        await selection.compare_selection(frozen_selection.manifest)
    assert error.value.kind == "invalid_replay"


@pytest.mark.parametrize("field", ["render_version", "reference", "baseline"])
async def test_invalid_frozen_input_is_refused(frozen_selection: FrozenInputs, field: str) -> None:
    if field == "render_version":
        data = json.loads(await asyncio.to_thread(frozen_selection.manifest.read_text, encoding="utf-8"))
        data["render_version"] = "unknown"
        await asyncio.to_thread(frozen_selection.manifest.write_text, json.dumps(data), encoding="utf-8")
    else:
        await asyncio.to_thread(
            (frozen_selection.manifest.parent / f"{field}.txt").write_text, "changed", encoding="utf-8"
        )
    with pytest.raises(ExtractionError) as error:
        await selection.compare_selection(frozen_selection.manifest)
    assert error.value.kind == "invalid_replay"


def test_real_module_runs_without_any_model_profile_and_preserves_frozen_files(frozen_selection: FrozenInputs) -> None:
    env = {
        **os.environ,
        "HOME": str(frozen_selection.home),
        "USERPROFILE": str(frozen_selection.home),
        "APPDATA": str(frozen_selection.home / "appdata"),
        "XDG_CONFIG_HOME": str(frozen_selection.home / "config"),
        "CHRYS_MODEL_PROFILE": "nonexistent-profile",
        "PYTHONPATH": str(REPO / "src"),
    }
    completed = subprocess.run(
        [sys.executable, "-m", "chrys.app.memory_hosts.extraction_selection", str(frozen_selection.manifest)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        cwd=frozen_selection.home,
        env=env,
        timeout=ENGINE_TURN_TIMEOUT,
        **windows_hidden_subprocess_kwargs(),
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8")
    report = json.loads(completed.stdout)
    assert set(report["policies"]) == {"chronological", "head_tail", "signal_first"}
    assert b"PRIVATE_" not in completed.stdout + completed.stderr
    assert SESSION.encode() not in completed.stdout + completed.stderr
    assert not (frozen_selection.home / ".claude").exists()
    assert not (frozen_selection.home / ".codex").exists()
    assert not (frozen_selection.home / ".chrys" / "memory").exists()
    after = tuple(
        sorted(
            (path.relative_to(frozen_selection.manifest.parent).as_posix(), path.read_bytes())
            for path in frozen_selection.manifest.parent.rglob("*")
            if path.is_file()
        )
    )
    assert after == frozen_selection.before


def test_main_bootstraps_without_resolving_a_model_profile(
    frozen_selection: FrozenInputs, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from chrys.orchestration import startup

    calls: list[bool] = []

    def bootstrap() -> None:
        calls.append(True)

    monkeypatch.setattr(startup, "bootstrap_runtime", bootstrap)
    assert selection.main([str(frozen_selection.manifest)]) == 0
    assert calls == [True]
    assert json.loads(capsys.readouterr().out)["source_sessions"] == 1


def test_main_failure_is_safe_and_does_not_echo_input_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from chrys.orchestration import startup

    monkeypatch.setattr(startup, "bootstrap_runtime", lambda: None)
    path = tmp_path / "PRIVATE_missing_manifest.json"
    assert selection.main([str(path)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "invalid_replay" in output.err
    assert "PRIVATE_" not in output.err and str(tmp_path) not in output.err


def test_main_reports_a_corrupt_tuning_file_like_extract(
    frozen_selection: FrozenInputs, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from chrys.orchestration import startup

    monkeypatch.setattr(startup, "bootstrap_runtime", lambda: None)
    candidate = frozen_selection.manifest.parent / "PRIVATE_candidate.json"
    candidate.write_text("{not json", encoding="utf-8")
    assert selection.main([str(frozen_selection.manifest), "--tuning", str(candidate)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "Memory selection comparison failed: invalid_tuning.\n"


def test_main_reports_a_corrupt_active_tuning_like_extract(
    frozen_selection: FrozenInputs, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from chrys.orchestration import startup

    monkeypatch.setattr(startup, "bootstrap_runtime", lambda: None)
    tuning_dir = get_platform().config_dir / "memory-tuning"
    tuning_dir.mkdir(mode=0o700, parents=True)
    plant_owner_only_bytes(tuning_dir / "active", b"1\n2\n")
    assert selection.main([str(frozen_selection.manifest)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "Memory selection comparison failed: invalid_tuning.\n"


async def test_comparison_renders_every_frozen_source_beyond_the_paid_run_batch_cap(
    frozen_selection: FrozenInputs,
) -> None:
    data = json.loads(await asyncio.to_thread(frozen_selection.manifest.read_text, encoding="utf-8"))
    original = await asyncio.to_thread(frozen_selection.source.read_text, encoding="utf-8")
    for index in range(1, 4):
        session = f"40000000-0000-4000-8000-{index:012d}"
        payload = original.replace(SESSION, session).encode()
        source = frozen_selection.source.with_name(f"{session}.jsonl")
        await asyncio.to_thread(plant_owner_only_bytes, source, payload)
        revision = await asyncio.to_thread(source.stat)
        data["sources"].append(
            {
                "host": "claude-code",
                "session": session,
                "path": source.relative_to(frozen_selection.manifest.parent).as_posix(),
                "sha256": digest(payload),
                "updated_ns": revision.st_mtime_ns,
            }
        )
    await asyncio.to_thread(frozen_selection.manifest.write_text, json.dumps(data), encoding="utf-8")
    report = await selection.compare_selection(frozen_selection.manifest)
    assert report["source_sessions"] == 4
    for counts in report["policies"].values():
        assert counts == {
            "batch_count": 4,
            "omitted_tool_results": 12,
            "omitted_error_results": 8,
            "omitted_signal_neighbors": 16,
            "omitted_assistant_messages": 8,
        }


async def test_missing_frozen_source_is_reported_as_invalid_replay(frozen_selection: FrozenInputs) -> None:
    await asyncio.to_thread(frozen_selection.source.unlink)
    with pytest.raises(ExtractionError) as error:
        await selection.compare_selection(frozen_selection.manifest)
    assert error.value.kind == "invalid_replay"


async def test_comparison_uses_all_three_policies_on_the_shared_full_planner(
    frozen_selection: FrozenInputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections.abc import Sequence

    from chrys.service.memory.extraction import ExtractionBatch, MaterialProfile, OversizeSelection, SourceSession
    from chrys.service.memory.storage.store import MemoryScope

    original = selection.build_batches
    policies: list[str] = []

    def observed(
        project: MemoryScope,
        sources: Sequence[SourceSession],
        budget: int,
        *,
        profile: MaterialProfile,
        oversize_selection: OversizeSelection,
        max_batches: int,
    ) -> tuple[ExtractionBatch, ...]:
        assert profile.name == "full-v4"
        assert budget == 600
        assert max_batches == len(sources)
        policies.append(oversize_selection)
        return original(
            project, sources, budget, profile=profile, oversize_selection=oversize_selection, max_batches=max_batches
        )

    monkeypatch.setattr(selection, "build_batches", observed)
    await selection.compare_selection(frozen_selection.manifest)
    assert policies == ["chronological", "head_tail", "signal_first"]


async def test_explicit_rule_limits_comparison_to_that_policy(frozen_selection: FrozenInputs) -> None:
    report = await selection.compare_selection(frozen_selection.manifest, rules=("signal_first",))
    assert set(report["policies"]) == {"signal_first"}


async def test_tuning_rule_is_compared_first(frozen_selection: FrozenInputs) -> None:
    from dataclasses import replace

    from chrys.service.memory.tuning import V0

    report = await selection.compare_selection(
        frozen_selection.manifest, tuning=replace(V0, oversize_selection="signal_first")
    )
    assert list(report["policies"]) == ["signal_first", "chronological", "head_tail"]


@pytest.mark.parametrize("tuned_budget,actual_budget", [(500, 500), (900, 600)])
async def test_tuning_material_budget_matches_the_report_and_shared_planner(
    frozen_selection: FrozenInputs,
    monkeypatch: pytest.MonkeyPatch,
    tuned_budget: int,
    actual_budget: int,
) -> None:
    from collections.abc import Sequence
    from dataclasses import replace

    from chrys.service.memory.extraction import ExtractionBatch, MaterialProfile, OversizeSelection, SourceSession
    from chrys.service.memory.storage.store import MemoryScope
    from chrys.service.memory.tuning import V0

    original = selection.build_batches
    observed_budgets: list[int] = []

    def observed(
        project: MemoryScope,
        sources: Sequence[SourceSession],
        budget: int,
        *,
        profile: MaterialProfile,
        oversize_selection: OversizeSelection,
        max_batches: int,
    ) -> tuple[ExtractionBatch, ...]:
        observed_budgets.append(budget)
        batches = original(
            project, sources, budget, profile=profile, oversize_selection=oversize_selection, max_batches=max_batches
        )
        assert all(batch.estimated_tokens <= actual_budget for batch in batches)
        return batches

    monkeypatch.setattr(selection, "build_batches", observed)
    report = await selection.compare_selection(
        frozen_selection.manifest, tuning=replace(V0, material_budget=tuned_budget)
    )
    assert report["material_budget"] == actual_budget
    assert observed_budgets == [actual_budget, actual_budget, actual_budget]
