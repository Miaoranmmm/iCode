# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Real hook main with a typed process-tree fixture, without production switches."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
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
    status = main(process_tree=Tree())
    report_path.write_text(json.dumps({"bytes_read": observed.bytes_read, "eof": observed.eof}), encoding="utf-8")
    return status


if __name__ == "__main__":
    if "--bootstrap-failure" in sys.argv:
        raise SystemExit(bootstrap_failure_after_observed_input(Path(sys.argv[-1])))
    raise SystemExit(main(process_tree=MissingTree() if "--no-host" in sys.argv else Tree()))
