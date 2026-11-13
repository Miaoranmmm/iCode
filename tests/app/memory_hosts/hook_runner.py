# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Real hook main with a typed process-tree fixture, without production switches."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import BinaryIO, NoReturn

from chrys.app.memory_hosts.claude_code import hook
from chrys.app.memory_hosts.claude_code.hook import main
from chrys.app.memory_hosts.claude_code.registry import ProcessInfo


class Tree:
    def inspect(self, pid: int) -> ProcessInfo | None:
        if pid == os.getpid():
            return ProcessInfo(pid, 777, "/fixtures/python", 19, ())
        if pid == 777:
            return ProcessInfo(777, 0, "/fixtures/claude", 17, ("claude",))
        return None


class MissingTree:
    def inspect(self, pid: int) -> ProcessInfo | None:
        return None


class ObservedInput:
    """Observe real reads, including the EOF read, without buffering extra input."""

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.bytes_read = 0
        self.eof = False

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        self.bytes_read += len(data)
        if not data:
            self.eof = True
        return data


class ObservedStdin:
    def __init__(self, stream: ObservedInput) -> None:
        self.buffer = stream


def bootstrap_failure_after_observed_input(report_path: Path) -> int:
    observed = ObservedInput(sys.stdin.buffer)
    sys.stdin = ObservedStdin(observed)

    def fail_bootstrap(
        *, configure_stdio: bool, apply_patches: bool, setup_telemetry: bool, project_root: Path | None
    ) -> NoReturn:
        raise RuntimeError("private bootstrap details must not be shown")

    hook.bootstrap_runtime = fail_bootstrap
    status = main([], process_tree=Tree())
    report_path.write_text(json.dumps({"bytes_read": observed.bytes_read, "eof": observed.eof}), encoding="utf-8")
    return status


def observe_auto_child(report_path: Path) -> None:
    """Give the test ownership of every detached child, indexed by its claim."""
    from chrys.app.memory_hosts.claude_code import auto_extract

    children: dict[str, int] = {}
    current_claim = ""
    original_launch = auto_extract._launch

    def observe_claim(root: Path, key: str, host: str, source: Path, project: Path, selector: str) -> None:
        nonlocal current_claim
        current_claim = key
        original_launch(root, key, host, source, project, selector)

    auto_extract._launch = observe_claim

    def launch(
        args: list[str],
        *,
        stdin: int,
        stdout: BinaryIO,
        stderr: BinaryIO,
        env: dict[str, str],
        start_new_session: bool = False,
        creationflags: int = 0,
    ) -> subprocess.Popen[bytes]:
        child = subprocess.Popen(
            args,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            env=env,
            start_new_session=start_new_session,
            creationflags=creationflags,
        )
        children[current_claim] = child.pid
        report_path.write_text(json.dumps(children), encoding="utf-8")
        return child

    shadow = ModuleType("subprocess")
    shadow.__dict__.update(vars(subprocess))
    shadow.Popen = launch
    auto_extract.subprocess = shadow


if __name__ == "__main__":
    if "--bootstrap-failure" in sys.argv:
        raise SystemExit(bootstrap_failure_after_observed_input(Path(sys.argv[-1])))
    args = [arg for arg in sys.argv[1:] if arg != "--no-host"]
    if "--child-report" in args:
        index = args.index("--child-report")
        observe_auto_child(Path(args[index + 1]))
        del args[index : index + 2]
    raise SystemExit(main(args, process_tree=MissingTree() if "--no-host" in sys.argv else Tree()))
