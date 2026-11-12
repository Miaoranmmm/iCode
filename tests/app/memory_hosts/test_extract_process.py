# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The standalone module bootstraps and dry-runs synthetic files without a model."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from chrys.app.memory_hosts import extract
from chrys.foundation.platform.process import windows_hidden_subprocess_kwargs
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.profiles.models.serializer import save_profile
from tests.support.waiting import ENGINE_TURN_TIMEOUT

REPO = Path(__file__).resolve().parents[3]
SESSION = "20000000-0000-4000-8000-000000000002"


@pytest.fixture(autouse=True)
def temporary_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "temporary-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("APPDATA", str(home / "appdata"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "config"))


@pytest.mark.parametrize("freeze", [False, True])
def test_real_module_reads_only_temp_home_and_source_and_does_not_connect(tmp_path: Path, freeze: bool) -> None:
    home = tmp_path / "home"
    config = home / ".chrys"
    profiles = config / "models"
    profiles.mkdir(parents=True)
    (profiles / "offline.yaml").write_text(
        "id: offline\nname: Offline Response\nprovider: deepseek-openai\napi_style: responses\n"
        "model_id: fake-no-network\nbase_url: http://127.0.0.1:1\napi_key: private-sentinel\n"
        "max_context_tokens: 100000\n",
        encoding="utf-8",
    )
    project = tmp_path / "project"
    source = tmp_path / "source"
    project.mkdir()
    source.mkdir()
    (source / f"{SESSION}.jsonl").write_text(
        json.dumps(
            {
                "type": "user",
                "uuid": "synthetic-human",
                "parentUuid": None,
                "sessionId": SESSION,
                "cwd": str(project),
                "entrypoint": "claude-desktop",
                "origin": {"kind": "human"},
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "Use clear names."},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    memory = tmp_path / "memory"
    frozen = tmp_path / "regression"
    env = {
        **os.environ,
        "HOME": str(home),
        "USERPROFILE": str(home),
        "APPDATA": str(home / "appdata"),
        "PYTHONPATH": str(REPO / "src"),
        "CHRYS_MODEL_PROFILE": "offline",
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chrys.app.memory_hosts.extract",
            "-C",
            str(project),
            "--claude-source",
            str(source),
            "--memory-dir",
            str(memory),
            "--dry-run",
            *(["--freeze-to", str(frozen)] if freeze else []),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        env=env,
        cwd=project,
        timeout=ENGINE_TURN_TIMEOUT,
        **windows_hidden_subprocess_kwargs(),
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8")
    report = json.loads(completed.stdout)
    assert report["status"] == "dry_run"
    assert report["profile"]["model_id"] == "fake-no-network"
    assert report["batches"][0]["displayed_messages"] == 1
    assert report["signals"] == {
        "candidate_sessions": 1,
        "annotated_sessions": 0,
        "annotations": {"user_interrupt": 0},
        "signals_unavailable_sessions": 0,
    }
    if freeze:
        assert (frozen / "report.json").read_bytes() == completed.stdout
        assert (frozen / "manifest.json").is_file()
    assert not memory.exists()
    assert not (home / ".codex").exists()
    assert not (home / ".claude").exists()
    assert b"private-sentinel" not in completed.stdout + completed.stderr


@pytest.mark.parametrize("selector", ["id", "name", "active"])
def test_main_selects_id_unique_name_or_effective_active_without_changing_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selector: str,
) -> None:
    profile = ModelProfile(
        "offline", "Offline Response", provider="deepseek-openai", api_style="responses", model_id="fake"
    )
    saved = save_profile(profile)
    before = saved.read_bytes()
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "offline")
    source = tmp_path / "source"
    source.mkdir()
    # A valid metadata-only file is filtered to zero candidates, not missing input.
    (source / f"{SESSION}.jsonl").write_text('{"type":"last-prompt","leafUuid":null}\n', encoding="utf-8")
    args = ["-C", str(tmp_path), "--claude-source", str(source), "--memory-dir", str(tmp_path / "memory"), "--dry-run"]
    if selector != "active":
        args.extend(("--model-profile", "offline" if selector == "id" else "offline response"))
    assert extract.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["profile"]["id"] == "offline"
    assert saved.read_bytes() == before


@pytest.mark.parametrize("failure", ["missing", "hollow", "ambiguous", "raw_http"])
def test_profile_or_capture_preflight_does_not_start_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "missing")
    if failure in {"hollow", "ambiguous"}:
        save_profile(ModelProfile("one", "Duplicate", model_id="" if failure == "hollow" else "fake"))
        if failure == "ambiguous":
            save_profile(ModelProfile("two", "Duplicate", model_id="fake"))
    if failure == "raw_http":
        monkeypatch.setenv("CHRYS_DEBUG_LLM_RAW_HTTP_LOG", "1")
    selector = "Duplicate" if failure == "ambiguous" else "one" if failure == "hollow" else "missing"
    memory = tmp_path / "memory"
    assert (
        extract.main(["-C", str(tmp_path), "--model-profile", selector, "--memory-dir", str(memory), "--dry-run"]) == 1
    )
    output = capsys.readouterr()
    assert "preflight failed" in output.err
    assert output.out == ""
    assert not memory.exists()
