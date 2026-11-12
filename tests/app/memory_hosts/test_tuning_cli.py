# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tuning activation requires an explicit human TTY confirmation."""

from __future__ import annotations

import io
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from chrys.app.memory_hosts import tuning
from chrys.service.memory.tuning import V0, load_active, propose


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def no_runtime_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    def bootstrap(**kwargs: object) -> None:
        pass

    monkeypatch.setattr(tuning, "bootstrap_runtime", bootstrap)


def invoke(config: Path, args: list[str], input_: io.StringIO | None = None) -> tuple[int, str, str]:
    output, error = io.StringIO(), io.StringIO()
    code = tuning.main(
        args, config_dir=config, input_stream=input_ or io.StringIO(), output_stream=output, error_stream=error
    )
    return code, output.getvalue(), error.getvalue()


def test_show_list_and_propose_never_activate(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(asdict(replace(V0, oversize_selection="signal_first"))), encoding="utf-8")
    assert json.loads(invoke(tmp_path, ["show"])[1])["version"] == "v0"
    code, output, error = invoke(tmp_path, ["propose", str(candidate)])
    assert (code, output, error) == (0, "Proposed v1; it is not active.\n", "")
    assert load_active(tmp_path) == V0
    assert invoke(tmp_path, ["list"]) == (0, "v0 active\nv1\n", "")


def test_apply_without_tty_is_rejected_without_reading_or_changing_config(tmp_path: Path) -> None:
    candidate = propose(tmp_path, replace(V0, max_turns=12))
    input_ = io.StringIO(candidate.version + "\n")
    code, output, error = invoke(tmp_path, ["apply", candidate.version], input_)
    assert code == 1
    assert output == ""
    assert "stdin must be a TTY" in error
    assert input_.tell() == 0
    assert load_active(tmp_path) == V0


@pytest.mark.parametrize("name", ["1", "v1"])
def test_propose_numeric_filename_always_reads_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    from chrys.service.memory.tuning import load_version

    propose(tmp_path, replace(V0, oversize_selection="head_tail"))
    monkeypatch.chdir(tmp_path)
    Path(name).write_text('{"oversize_selection":"signal_first"}', encoding="utf-8")
    assert invoke(tmp_path, ["propose", name])[0] == 0
    assert load_version(tmp_path, "2").oversize_selection == "signal_first"


@pytest.mark.parametrize("confirmation,success", [("v1\n", True), ("1\n", True), ("v2\n", False), ("", False)])
def test_apply_requires_the_selected_version_then_can_reactivate_v0(
    tmp_path: Path, confirmation: str, success: bool
) -> None:
    candidate = propose(tmp_path, replace(V0, max_turns=12))
    code, output, error = invoke(tmp_path, ["apply", "v1"], Terminal(confirmation))
    assert "Type v1 to activate it:" in output
    assert code == (0 if success else 1)
    assert load_active(tmp_path) == (candidate if success else V0)
    if success:
        assert error == ""
        assert invoke(tmp_path, ["apply", "v0"], Terminal("v0\n"))[0] == 0
        assert load_active(tmp_path) == V0
    else:
        assert "confirmation did not match" in error
