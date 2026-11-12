# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Nonblocking SessionStart, UserPromptSubmit and Stop hooks for Claude Code."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import BinaryIO, TextIO

from chrys.app.memory_hosts.claude_code.conversation import catalog_visibility
from chrys.app.memory_hosts.claude_code.registry import (
    SOURCE_UNAVAILABLE,
    WORKSPACE_UNAVAILABLE,
    DeliveryState,
    ProcessTree,
    Registration,
    Registry,
    SystemProcessTree,
    find_host,
    project_directory,
)
from chrys.foundation.platform import get_platform
from chrys.orchestration.startup import bootstrap_runtime
from chrys.service.memory.operations import MemoryReader
from chrys.service.memory.project import bind_project
from chrys.service.memory.render import WITHDRAWN_CATALOG, render_catalog
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import MemoryOperationError, MemoryStore
from chrys.service.memory.tuning import V0, MemoryTuning, load_active

STDIN_LIMIT = 2 * 1024 * 1024
_INPUT_CHUNK = 64 * 1024
_SESSION_SOURCES = frozenset({"startup", "resume", "clear", "compact", "fork"})


def read_input(stream: BinaryIO) -> Mapping[str, object] | None:
    """Drain oversized input to EOF without parsing a partial JSON object."""
    chunks: list[bytes] = []
    total = 0
    oversized = False
    while chunk := stream.read(_INPUT_CHUNK):
        total += len(chunk)
        if total > STDIN_LIMIT:
            oversized = True
            chunks.clear()
        if not oversized:
            chunks.append(chunk)
    if oversized:
        return None
    value = json.loads(b"".join(chunks))
    if not isinstance(value, dict):
        raise ValueError("Hook input must be an object.")
    return value


async def handle_hook(
    payload: Mapping[str, object],
    *,
    tree: ProcessTree,
    registry: Registry,
    store: MemoryStore,
    environ: Mapping[str, str],
    output: TextIO,
    diagnostic: TextIO,
    pid: int | None = None,
    tuning: MemoryTuning = V0,
    auto_extract_profile: str | None = None,
) -> None:
    """Handle one input; the CLI boundary converts failures to short diagnostics."""
    event = payload.get("hook_event_name")
    if event not in {"SessionStart", "UserPromptSubmit", "Stop"}:
        return
    if event == "SessionStart" and payload.get("source") not in _SESSION_SOURCES:
        return
    session_id = payload.get("session_id")
    transcript_path = payload.get("transcript_path")
    if not isinstance(session_id, str) or not session_id or not isinstance(transcript_path, str) or not transcript_path:
        if event != "Stop":
            print("Memory hook input is unavailable.", file=diagnostic)
        return
    key = await asyncio.to_thread(find_host, tree, pid)
    if key is None:
        print(SOURCE_UNAVAILABLE, file=diagnostic)
        return
    if event == "Stop":
        state = await asyncio.to_thread(registry.read_delivery, key, session_id)
        if state is None or not state.valid or not state.ever_nonempty:
            return
        prompt_id = payload.get("prompt_id")
        assistant = payload.get("last_assistant_message")
        if not isinstance(prompt_id, str) or not isinstance(assistant, str):
            return
        target = state.last_text if state.last_text is not None else WITHDRAWN_CATALOG
        visible = await asyncio.to_thread(catalog_visibility, transcript_path, prompt_id, assistant, target)
        if visible is False:
            await asyncio.to_thread(registry.mark_redeliver, key, session_id)
        return

    project_dir = await asyncio.to_thread(project_directory, tree, key, environ)
    if event == "UserPromptSubmit":
        await asyncio.to_thread(
            registry.write_registration, key, Registration(session_id, transcript_path, project_dir)
        )
    if project_dir is None:
        print(WORKSPACE_UNAVAILABLE, file=diagnostic)
        return
    try:
        project = await asyncio.to_thread(bind_project, project_dir)
    except MemoryOperationError, OSError:
        project = None
        print("Memory project binding is unavailable; only global memories are available.", file=diagnostic)
    text = render_catalog(await MemoryReader(store, project).catalog(), tuning=tuning)
    previous = await asyncio.to_thread(registry.read_delivery, key, session_id)
    ever_nonempty = text is not None or (previous is not None and previous.ever_nonempty)
    redeliver = await asyncio.to_thread(registry.needs_redeliver, key, session_id)
    changed = previous is None or not previous.valid or previous.last_text != text
    outgoing: str | None = None
    if event == "SessionStart":
        outgoing = text
        if text is None and payload.get("source") in {"resume", "fork", "compact"}:
            outgoing = WITHDRAWN_CATALOG
    elif changed or redeliver:
        outgoing = text if text is not None else WITHDRAWN_CATALOG if ever_nonempty else None
    if outgoing is not None:
        output.write(outgoing + "\n")
        output.flush()
    await asyncio.to_thread(registry.write_delivery, key, DeliveryState(session_id, text, ever_nonempty))
    await asyncio.to_thread(registry.clear_redeliver, key, session_id)
    if event == "UserPromptSubmit" and project is not None and auto_extract_profile is not None:
        try:
            from chrys.app.memory_hosts.claude_code.auto_extract import maybe_start

            await asyncio.to_thread(
                maybe_start, registry.root, Path(transcript_path).parent, Path(project_dir), auto_extract_profile
            )
        except Exception:
            diagnostic.write("Memory auto-extraction could not start.\n")


def auto_extract_profile(argv: Sequence[str] | None) -> str | None:
    """A bad hook argument only turns automatic extraction off.

    argparse would exit 2, and Claude Code treats exit 2 from UserPromptSubmit
    as blocking the user's prompt.
    """
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument("--auto-extract-profile")
    try:
        args, unknown = parser.parse_known_args(argv)
    except argparse.ArgumentError:
        args, unknown = None, ["invalid"]
    if args is None or unknown or args.auto_extract_profile == "":
        sys.stderr.write("Memory hook arguments are invalid; automatic extraction is off.\n")
        return None
    return args.auto_extract_profile


def main(argv: Sequence[str] | None = None, *, process_tree: ProcessTree | None = None) -> int:
    """Run a real hook process; captured failures always let the host continue."""
    try:
        profile = auto_extract_profile(argv)
        payload = read_input(sys.stdin.buffer)
        bootstrap_runtime(configure_stdio=True, apply_patches=False, setup_telemetry=False, project_root=None)
        if payload is None:
            sys.stderr.write("Memory hook input exceeded the 2 MiB limit and was discarded.\n")
            return 0
        config_dir = get_platform().config_dir
        try:
            tuning = load_active(config_dir)
        except ValueError:
            sys.stderr.write("Memory tuning unavailable; using v0.\n")
            tuning = V0
        root = config_dir / "memory"
        asyncio.run(
            handle_hook(
                payload,
                tree=SystemProcessTree() if process_tree is None else process_tree,
                registry=Registry(root),
                store=FileMemoryStore(root),
                environ=os.environ,
                output=sys.stdout,
                diagnostic=sys.stderr,
                tuning=tuning,
                auto_extract_profile=profile,
            )
        )
    except Exception:
        sys.stderr.write("Memory hook is unavailable; continuing without an update.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
