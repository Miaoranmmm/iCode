# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A directory's project: worktrees of one repository share it, clones and plain directories have their own."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from chrys.service.memory.project import bind_project
from chrys.service.memory.storage.store import MemoryOperationError


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_worktrees_of_one_repository_share_a_project(tmp_path: Path, git_repo_factory: Callable[[Path], Path]) -> None:
    repo = git_repo_factory(tmp_path / "repo")
    _git(repo, "worktree", "add", "-q", str(tmp_path / "feature"), "-b", "feature")
    (repo / "src" / "deep").mkdir(parents=True)

    main = bind_project(str(repo))
    feature = bind_project(str(tmp_path / "feature"))
    nested = bind_project(str(repo / "src" / "deep"))

    assert main.project_id == feature.project_id == nested.project_id
    assert main.anchor == feature.anchor == "git"
    assert main.root == os.path.realpath(repo)
    assert feature.root == os.path.realpath(tmp_path / "feature")
    assert nested.root == main.root


def test_a_separate_clone_is_a_different_project(tmp_path: Path, git_repo_factory: Callable[[Path], Path]) -> None:
    first = git_repo_factory(tmp_path / "first")
    second = git_repo_factory(tmp_path / "second")

    assert bind_project(str(first)).project_id != bind_project(str(second)).project_id


def test_a_plain_directory_is_its_own_project_by_real_path(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    link = tmp_path / "link"
    link.symlink_to(plain, target_is_directory=True)

    scope = bind_project(str(plain))

    assert scope.anchor == "dir"
    assert scope.root == os.path.realpath(plain)
    assert bind_project(str(link)).project_id == scope.project_id
    other = tmp_path / "other"
    other.mkdir()
    assert bind_project(str(other)).project_id != scope.project_id


def test_without_git_the_project_is_not_guessed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))

    with pytest.raises(MemoryOperationError) as caught:
        bind_project(str(plain))
    assert caught.value.kind == "memory_store_unavailable"
    assert str(tmp_path) not in caught.value.message


def test_a_missing_directory_has_no_project(tmp_path: Path) -> None:
    with pytest.raises(MemoryOperationError) as caught:
        bind_project(str(tmp_path / "gone"))
    assert caught.value.kind == "memory_store_unavailable"


def test_a_broken_worktree_is_a_failure_not_a_plain_directory(
    tmp_path: Path, git_repo_factory: Callable[[Path], Path]
) -> None:
    repo = git_repo_factory(tmp_path / "repo")
    worktree = tmp_path / "feature"
    _git(repo, "worktree", "add", "-q", str(worktree), "-b", "feature")
    # The worktree's record in the main repository is gone; its .git file now points nowhere.
    for record in (repo / ".git" / "worktrees").iterdir():
        shutil.rmtree(record)

    with pytest.raises(MemoryOperationError) as caught:
        bind_project(str(worktree))
    assert caught.value.kind == "memory_store_unavailable"
