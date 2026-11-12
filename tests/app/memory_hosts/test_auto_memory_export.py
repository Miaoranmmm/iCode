# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Exporting chrys entries as Claude Code auto-memory notes, using invented text only."""

from __future__ import annotations

import asyncio
import io
import subprocess
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from chrys.app.memory_hosts import auto_memory_export
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import (
    GLOBAL_SCOPE,
    MemoryEntryDraft,
    MemoryScope,
    Quote,
    ToolResultEvidence,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

PROJECT_ID = "5e1f0c9a7b2d4e61"
OWN_NOTE = "---\nname: own\ndescription: mine\nmetadata:\n  type: user\n---\n\nWritten by Claude.\n"


def intent(title: str, words: str, applies_when: str | None = None) -> MemoryEntryDraft:
    quote = Quote(words, "message", "session-one", "2026-11-12T07:25:03Z", "item-1")
    return MemoryEntryDraft("intent", title, f"Rule: {words}", "model", applies_when=applies_when, quotes=(quote,))


@pytest.fixture(autouse=True)
def bootstraps(monkeypatch: pytest.MonkeyPatch) -> list[tuple[bool, bool, bool, str | None]]:
    calls: list[tuple[bool, bool, bool, str | None]] = []

    def bootstrap(
        *, configure_stdio: bool, apply_patches: bool, setup_telemetry: bool, project_root: str | None
    ) -> None:
        calls.append((configure_stdio, apply_patches, setup_telemetry, project_root))

    monkeypatch.setattr(auto_memory_export, "bootstrap_runtime", bootstrap)
    return calls


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    directory = tmp_path / "work" / "shop app"
    directory.mkdir(parents=True)
    return directory


@pytest.fixture
def scope(project_root: Path) -> MemoryScope:
    return MemoryScope("project", PROJECT_ID, root=str(project_root), anchor="dir")


@pytest.fixture
def store(tmp_path: Path) -> FileMemoryStore:
    return FileMemoryStore(tmp_path / "memory")


def notes_dir(out: Path, project_root: Path) -> Path:
    return out / str(project_root).replace("/", "-").replace(" ", "-") / "memory"


async def run(tmp_path: Path, out: Path, *extra: str) -> None:
    argv = ["--memory-root", str(tmp_path / "memory"), "--out", str(out), *extra]
    output, error = io.StringIO(), io.StringIO()
    assert await asyncio.to_thread(auto_memory_export.main, argv, output=output, error=error) == 0
    assert error.getvalue() == ""


async def test_export_writes_notes_beside_existing_ones_and_skips_withdrawn(
    tmp_path: Path, store: FileMemoryStore, scope: MemoryScope, project_root: Path
) -> None:
    kept = await store.create(scope, intent("Answer in Chinese", "以后回答都用中文", applies_when="every reply"))
    finding = await store.create(
        scope,
        MemoryEntryDraft(
            "engineering",
            "Tests need uv run",
            "Run pytest through uv.",
            "model",
            evidence=(ToolResultEvidence("Bash", "pytest", "command not found: pytest\nexit 127"),),
        ),
    )
    gone = await store.create(scope, intent("Old rule", "以前的规矩"))
    stored = await store.get(scope, gone.ref)
    await store.update(
        scope,
        gone.ref,
        stored.version,
        replace(
            stored.entry.draft(),
            status="withdrawn",
            status_basis=Quote("不用了", "message", "session-two", "2026-11-12T08:00:00Z", "item-2"),
        ),
    )
    await store.create(GLOBAL_SCOPE, intent("Short commits", "提交信息写一行"))
    out = tmp_path / "projects"
    target = notes_dir(out, project_root)
    target.mkdir(parents=True)
    (target / "own.md").write_text(OWN_NOTE, encoding="utf-8")
    (target / "MEMORY.md").write_text("- [Own](own.md) — mine\n", encoding="utf-8")

    await run(tmp_path, out, "--write")

    kept_id, finding_id = kept.ref.split(":")[1], finding.ref.split(":")[1]
    assert sorted(path.name for path in target.iterdir()) == sorted(
        ["MEMORY.md", "own.md", f"chrys-{kept_id}.md", f"chrys-{finding_id}.md"]
    )
    assert (target / "own.md").read_text(encoding="utf-8") == OWN_NOTE
    note = (target / f"chrys-{kept_id}.md").read_text(encoding="utf-8")
    created = (await store.get(scope, kept.ref)).entry.updated
    assert note == (
        f'---\nname: chrys-{kept_id}\ndescription: "Answer in Chinese"\nmetadata:\n  type: feedback\n'
        f'  source: "chrys-memory {kept.ref}"\n  modified: "{created}"\n---\n\nRule: 以后回答都用中文\n\n'
        "**Why:** the user's own words:\n\n> 以后回答都用中文 (2026-11-12)\n\n**How to apply:** every reply\n"
    )
    engineering = (target / f"chrys-{finding_id}.md").read_text(encoding="utf-8")
    assert "  type: project\n" in engineering
    assert "- tool result of `Bash`:\n> command not found: pytest\n> exit 127\n" in engineering
    assert engineering.endswith("verify against the current files before relying on it.\n")
    index = (target / "MEMORY.md").read_text(encoding="utf-8").splitlines()
    assert index[:2] == ["- [Own](own.md) — mine", ""]
    assert sorted(index[2:]) == sorted(
        [
            f"- [Answer in Chinese](chrys-{kept_id}.md) — saved in chrys-memory",
            f"- [Tests need uv run](chrys-{finding_id}.md) — saved in chrys-memory",
        ]
    )
    assert [path.name for path in (out / "_global").glob("chrys-*.md")] != []


async def test_rerun_drops_deleted_entries_and_changes_nothing_else(
    tmp_path: Path, store: FileMemoryStore, scope: MemoryScope, project_root: Path
) -> None:
    first = await store.create(scope, intent("First", "第一条"))
    second = await store.create(scope, intent("Second", "第二条"))
    out = tmp_path / "projects"
    target = notes_dir(out, project_root)
    target.mkdir(parents=True)
    (target / "MEMORY.md").write_text("- [Own](own.md) — mine\n", encoding="utf-8")
    await run(tmp_path, out, "--write")
    snapshot = {path.name: path.read_text(encoding="utf-8") for path in target.iterdir()}
    await run(tmp_path, out, "--write")
    assert {path.name: path.read_text(encoding="utf-8") for path in target.iterdir()} == snapshot

    await store.delete(scope, second.ref, (await store.get(scope, second.ref)).version)
    await run(tmp_path, out, "--write")
    first_id = first.ref.split(":")[1]
    assert sorted(path.name for path in target.iterdir()) == ["MEMORY.md", f"chrys-{first_id}.md"]
    assert (target / "MEMORY.md").read_text(encoding="utf-8") == (
        f"- [Own](own.md) — mine\n\n- [First](chrys-{first_id}.md) — saved in chrys-memory\n"
    )


async def test_without_write_nothing_is_created(
    tmp_path: Path, store: FileMemoryStore, scope: MemoryScope, bootstraps: list[tuple[bool, bool, bool, str | None]]
) -> None:
    await store.create(scope, intent("First", "第一条"))
    out = tmp_path / "projects"
    await run(tmp_path, out)
    assert not out.exists()
    assert bootstraps == [(True, False, False, None)]


def test_a_worktree_shares_its_main_repository_key(tmp_path: Path, git_repo_factory: Callable[[Path], Path]) -> None:
    main = git_repo_factory(tmp_path / "main repo")
    worktree = tmp_path / "linked"
    subprocess.run(
        ["git", "-C", str(main), "worktree", "add", "-q", str(worktree)],
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
    )
    expected = str(main.resolve()).replace("/", "-").replace(" ", "-")
    assert auto_memory_export.claude_project_key(str(worktree)) == expected
    assert auto_memory_export.claude_project_key(str(main)) == expected
