# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The standalone review command reviews an empty temporary library without model access."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from chrys.app.memory_hosts import review
from chrys.foundation.platform.process import windows_hidden_subprocess_kwargs
from tests.support.waiting import ENGINE_TURN_TIMEOUT


def test_manual_module_empty_library_is_offline(tmp_path: Path):
    home = tmp_path / "home"
    profiles = home / ".chrys" / "models"
    profiles.mkdir(parents=True)
    (profiles / "offline.yaml").write_text(
        "id: offline\nname: Offline\nprovider: deepseek-openai\napi_style: responses\nmodel_id: fake\n"
        "base_url: http://127.0.0.1:1\napi_key: unused\nmax_context_tokens: 100000\n",
        encoding="utf-8",
    )
    project = tmp_path / "project"
    project.mkdir()
    env = {
        **os.environ,
        "HOME": str(home),
        "USERPROFILE": str(home),
        "APPDATA": str(home / "appdata"),
        "XDG_CONFIG_HOME": str(home / "config"),
        "PYTHONPATH": str(Path(review.__file__).parents[3]),
        "CHRYS_MODEL_PROFILE": "offline",
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chrys.app.memory_hosts.review",
            "--project",
            str(project),
            "--memory-dir",
            str(tmp_path / "memory"),
            "--model-profile",
            "offline",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=project,
        timeout=ENGINE_TURN_TIMEOUT,
        check=False,
        **windows_hidden_subprocess_kwargs(),
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "nothing_to_review"


def test_hook_review_state_gate_does_not_import_model_stack(tmp_path):
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "USERPROFILE": str(tmp_path),
        "APPDATA": str(tmp_path / "appdata"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "PYTHONPATH": str(Path(review.__file__).parents[3]),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from pathlib import Path; "
                "from chrys.app.memory_hosts.claude_code.auto_extract import _review_due; "
                "assert _review_due(Path(sys.argv[1]), 'project', None); "
                "assert 'chrys.app.memory_hosts.review' not in sys.modules; "
                "assert 'chrys.app.memory_hosts.executor_icode' not in sys.modules; "
                "assert 'chrys.kernel' not in sys.modules"
            ),
            str(tmp_path / "memory"),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=tmp_path,
        timeout=ENGINE_TURN_TIMEOUT,
        check=False,
        **windows_hidden_subprocess_kwargs(),
    )
    assert completed.returncode == 0, completed.stderr
