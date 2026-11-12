# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Export chrys memories as Claude Code auto-memory notes, for leaving the trial.

The export is lossy: quotes become plain text in the note, and versions, basis
checks and withdrawn entries are dropped. Project entries go to the directory
Claude Code keys by repository root (``<out>/<key>/memory/``); global entries
have no auto-memory home and go to ``<out>/_global/`` for the user to place.
Only files named ``chrys-*.md`` and index lines pointing at them are written or
removed, so existing auto-memory notes are never touched. Without ``--write``
the command only reports what it would do.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from chrys.foundation.platform import get_platform
from chrys.orchestration.startup import bootstrap_runtime
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import GLOBAL_SCOPE, MemoryScope
from chrys.service.mutations.git_state import locate_git_repository

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TextIO

    from chrys.service.memory.storage.store import Evidence, MemoryEntry

PREFIX = "chrys-"
INDEX = "MEMORY.md"
# Claude Code 2.1.317 loads at most this much of MEMORY.md.
INDEX_MAX_LINES = 200
INDEX_MAX_BYTES = 25_000
_INDEX_LINK = re.compile(r"\]\(" + re.escape(PREFIX) + r"[0-9a-f]+\.md\)")


@dataclass(frozen=True, slots=True)
class Note:
    filename: str
    text: str
    index_line: str


@dataclass(slots=True)
class Target:
    label: str
    directory: Path
    notes: list[Note]
    withdrawn: int = 0


def claude_project_key(root: str) -> str:
    """Claude Code keys auto-memory by the main worktree's path, with every other character replaced by ``-``."""
    location = locate_git_repository(root)
    base = root
    if location is not None:
        common = Path(location.common_dir)
        base = str(common.parent) if common.name == ".git" else location.root
    return re.sub(r"[^A-Za-z0-9_-]", "-", base)


def _yaml(value: str) -> str:
    # A JSON string is a valid double-quoted YAML scalar.
    return json.dumps(value, ensure_ascii=False)


def _quoted(text: str) -> str:
    return "\n".join("> " + line for line in text.splitlines() or [""])


def _evidence(item: Evidence) -> str:
    if item.kind == "tool_result":
        return f"tool result of `{item.tool}`:\n{_quoted(item.excerpt)}"
    if item.kind == "file":
        return f"file `{item.path}` ({item.bytes} bytes when saved)"
    if item.kind == "reference":
        return f"reference `{item.ref}`" + (f": {item.note}" if item.note else "")
    return f"user report:\n{_quoted(item.excerpt)}"


def render_note(entry: MemoryEntry) -> Note:
    """One auto-memory note; *Why* carries the saved quotes or evidence as plain text."""
    filename = f"{PREFIX}{entry.id}.md"
    title = " ".join(entry.title.split())
    lines = [
        "---",
        f"name: {PREFIX}{entry.id}",
        f"description: {_yaml(title)}",
        "metadata:",
        f"  type: {'feedback' if entry.type == 'intent' else 'project'}",
        f"  source: {_yaml('chrys-memory ' + entry.ref)}",
        f"  modified: {_yaml(entry.updated)}",
        "---",
        "",
        entry.body.strip(),
        "",
    ]
    if entry.type == "intent":
        lines.append("**Why:** the user's own words:")
        for quote in entry.quotes:
            lines += ["", _quoted(quote.text) + (f" ({quote.at[:10]})" if quote.at else "")]
        if entry.context is not None:
            lines += ["", "In answer to:", "", _quoted(entry.context.text)]
    else:
        lines.append("**Why:** found from:")
        for item in entry.evidence:
            lines += ["", "- " + _evidence(item)]
        lines += [
            "",
            "Exported without its basis check: verify against the current files before relying on it.",
        ]
    if entry.applies_when:
        lines += ["", f"**How to apply:** {entry.applies_when}"]
    return Note(filename, "\n".join(lines).rstrip() + "\n", f"- [{title}]({filename}) — saved in chrys-memory")


async def _collect(store: FileMemoryStore, out: Path) -> tuple[list[Target], list[str]]:
    targets: list[Target] = []
    skipped: list[str] = []
    scopes: list[tuple[str, Path, MemoryScope]] = [("global", out / "_global", GLOBAL_SCOPE)]
    for project in await store.list_projects():
        if project.root is None:
            skipped.append(f"project {project.project_id}: its root is unknown")
            continue
        try:
            key = claude_project_key(project.root)
        except OSError as error:
            skipped.append(f"project {project.project_id} ({project.root}): {error}")
            continue
        scope = MemoryScope("project", project.project_id, root=project.root)
        scopes.append((project.root, out / key / "memory", scope))
    for label, directory, scope in scopes:
        target = Target(label, directory, [])
        for stored in await store.list_entries(scope):
            if stored.entry.status == "withdrawn":
                target.withdrawn += 1
            else:
                target.notes.append(render_note(stored.entry))
        if target.notes or target.withdrawn:
            targets.append(target)
    return targets, skipped


def _index(existing: str, notes: Sequence[Note]) -> str:
    kept = [line for line in existing.splitlines() if not _INDEX_LINK.search(line)]
    while kept and not kept[-1].strip():
        kept.pop()
    lines = kept + ([""] if kept else []) + [note.index_line for note in notes]
    return "\n".join(lines) + "\n"


def _apply(target: Target) -> tuple[int, str]:
    target.directory.mkdir(parents=True, exist_ok=True)
    wanted = {note.filename for note in target.notes}
    removed = 0
    for stale in target.directory.glob(f"{PREFIX}*.md"):
        if stale.name not in wanted:
            stale.unlink()
            removed += 1
    for note in target.notes:
        (target.directory / note.filename).write_text(note.text, encoding="utf-8")
    index = target.directory / INDEX
    existing = index.read_text(encoding="utf-8") if index.exists() else ""
    text = _index(existing, target.notes)
    index.write_text(text, encoding="utf-8")
    return removed, text


def _limits(text: str) -> str | None:
    lines, size = text.count("\n"), len(text.encode("utf-8"))
    if lines > INDEX_MAX_LINES or size > INDEX_MAX_BYTES:
        return f"MEMORY.md would be {lines} lines / {size} bytes; Claude Code loads only {INDEX_MAX_LINES} / {INDEX_MAX_BYTES}"
    return None


def main(argv: Sequence[str] | None = None, *, output: TextIO | None = None, error: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export chrys memories as Claude Code auto-memory notes")
    parser.add_argument("--out", type=Path, required=True, help="the Claude Code projects directory, or a stand-in")
    parser.add_argument("--memory-root", type=Path, help="default: <config_dir>/memory")
    parser.add_argument("--write", action="store_true", help="write the notes; otherwise only report")
    args = parser.parse_args(argv)
    out = sys.stdout if output is None else output
    err = sys.stderr if error is None else error
    bootstrap_runtime(configure_stdio=True, apply_patches=False, setup_telemetry=False, project_root=None)
    root = args.memory_root or get_platform().config_dir / "memory"
    targets, skipped = asyncio.run(_collect(FileMemoryStore(root), args.out))
    for target in targets:
        index = target.directory / INDEX
        existing = index.read_text(encoding="utf-8") if index.exists() else ""
        removed, text = _apply(target) if args.write else (0, _index(existing, target.notes))
        verb = "wrote" if args.write else "would write"
        print(f"{target.label}: {verb} {len(target.notes)} notes to {target.directory}", end="", file=out)
        print(f"; skipped {target.withdrawn} withdrawn" if target.withdrawn else "", end="", file=out)
        print(f"; removed {removed} stale notes" if removed else "", file=out)
        warning = _limits(text)
        if warning:
            print(f"  warning: {warning}", file=out)
    if any(target.label == "global" for target in targets):
        print(
            "Global entries have no auto-memory home: copy what you want from _global/ into a project or CLAUDE.md.",
            file=out,
        )
    for line in skipped:
        print(f"skipped {line}", file=err)
    return 0


if __name__ == "__main__":
    sys.exit(main())
