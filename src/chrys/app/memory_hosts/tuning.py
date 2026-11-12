# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Inspect, propose and explicitly activate immutable memory tuning versions."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from chrys.foundation.platform import get_platform
from chrys.orchestration.startup import bootstrap_runtime
from chrys.service.memory.tuning import (
    apply_version,
    list_versions,
    load_active,
    load_candidate_file,
    load_version,
    propose,
)


def main(
    argv: Sequence[str] | None = None,
    *,
    config_dir: Path | None = None,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    error_stream: TextIO | None = None,
) -> int:
    """Keep proposal and activation separate; activation needs a human TTY."""
    parser = argparse.ArgumentParser(description="Inspect and activate memory tuning")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show")
    commands.add_parser("list")
    proposed = commands.add_parser("propose")
    proposed.add_argument("file", type=Path)
    applied = commands.add_parser("apply")
    applied.add_argument("version")
    args = parser.parse_args(argv)
    input_ = sys.stdin if input_stream is None else input_stream
    output = sys.stdout if output_stream is None else output_stream
    error = sys.stderr if error_stream is None else error_stream
    try:
        bootstrap_runtime(configure_stdio=True, apply_patches=False, setup_telemetry=False, project_root=None)
        directory = get_platform().config_dir if config_dir is None else config_dir
        if args.command == "show":
            print(json.dumps(asdict(load_active(directory)), ensure_ascii=False, indent=2), file=output)
        elif args.command == "list":
            active = load_active(directory).version
            for tuning in list_versions(directory):
                print(tuning.version + (" active" if tuning.version == active else ""), file=output)
        elif args.command == "propose":
            tuning = propose(directory, load_candidate_file(args.file))
            print(f"Proposed {tuning.version}; it is not active.", file=output)
        else:
            if not input_.isatty():
                raise ValueError("stdin must be a TTY to apply memory tuning.")
            tuning = load_version(directory, args.version)
            print(f"Type {tuning.version} to activate it: ", end="", file=output, flush=True)
            confirmation = input_.readline().strip()
            if confirmation not in {tuning.version, tuning.version[1:]}:
                raise ValueError("The tuning version confirmation did not match; nothing changed.")
            apply_version(directory, tuning.version)
            print(f"Activated {tuning.version}.", file=output)
    except (OSError, ValueError) as failure:
        print(f"Memory tuning: {failure}", file=error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
