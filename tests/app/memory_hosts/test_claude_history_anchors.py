# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A final checkpoint can authorize only later nodes on its descendant chain."""

from __future__ import annotations

from pathlib import Path

import pytest

from chrys.app.memory_hosts.claude_code.history import decode_claude_history
from chrys.app.memory_hosts.history import read_snapshot
from chrys.service.memory.storage.store import MemoryScope
from tests.app.memory_hosts.history_fixtures import SESSION, claude_assistant, claude_user, write_jsonl

PROJECT = MemoryScope("project", "0123456789abcdef")


@pytest.mark.parametrize("node_type", ["attachment", "system", "custom_main_node"])
@pytest.mark.parametrize("parent", ["root", None, "missing"])
def test_checkpoint_does_not_hide_a_later_non_dialogue_main_chain_node(
    tmp_path: Path,
    node_type: str,
    parent: str | None,
) -> None:
    path = tmp_path / f"{SESSION}.jsonl"
    write_jsonl(
        path,
        [
            claude_user("root", "Common rule."),
            claude_user("old", "Older branch rule.", "root"),
            {"type": "last-prompt", "leafUuid": "old"},
            {"type": node_type, "uuid": "later", "parentUuid": parent},
        ],
    )
    source, diagnostics = decode_claude_history(read_snapshot(path), SESSION, PROJECT)
    assert source is None
    assert diagnostics[-1].kind == ("invalid_chain" if parent == "missing" else "ambiguous_final_branch")


@pytest.mark.parametrize("node_type", ["attachment", "system", "custom_main_node"])
def test_later_non_dialogue_descendant_preserves_the_checkpoint_history(tmp_path: Path, node_type: str) -> None:
    path = tmp_path / f"{SESSION}.jsonl"
    write_jsonl(
        path,
        [
            claude_user("root", "Common rule."),
            claude_user("old", "Selected rule.", "root"),
            {"type": "last-prompt", "leafUuid": "old"},
            {"type": node_type, "uuid": "later", "parentUuid": "old"},
        ],
    )
    source, _ = decode_claude_history(read_snapshot(path), SESSION, PROJECT)
    assert source is not None
    assert [message.item for message in source.messages if message.kind == "user"] == ["root", "old"]


@pytest.mark.parametrize("node_type", ["attachment", "system"])
def test_explicit_empty_checkpoint_cannot_hide_later_main_chain_history(tmp_path: Path, node_type: str) -> None:
    path = tmp_path / f"{SESSION}.jsonl"
    write_jsonl(
        path,
        [
            claude_user("root", "Still available rule."),
            {"type": "last-prompt", "leafUuid": None, "explicit": True},
            {"type": node_type, "uuid": "later", "parentUuid": "root"},
        ],
    )
    source, _ = decode_claude_history(read_snapshot(path), SESSION, PROJECT)
    assert source is not None
    assert [message.item for message in source.messages if message.kind == "user"] == ["root"]


def test_later_stop_metadata_under_anchor_ignores_abandoned_pre_checkpoint_branches(tmp_path: Path) -> None:
    path = tmp_path / f"{SESSION}.jsonl"
    write_jsonl(
        path,
        [
            claude_user("root", "Common rule."),
            claude_user("abandoned", "Discarded branch.", "root"),
            claude_user("anchor", "Selected rule.", "root"),
            {"type": "last-prompt", "leafUuid": "anchor"},
            {"type": "system", "subtype": "stop_hook_summary", "uuid": "summary", "parentUuid": "anchor"},
        ],
    )
    source, _ = decode_claude_history(read_snapshot(path), SESSION, PROJECT)
    assert source is not None
    assert [message.item for message in source.messages if message.kind == "user"] == ["root", "anchor"]


def test_later_dialogue_from_anchor_ancestor_is_a_new_ambiguous_branch(tmp_path: Path) -> None:
    path = tmp_path / f"{SESSION}.jsonl"
    write_jsonl(
        path,
        [
            claude_user("root", "Common rule."),
            claude_user("anchor", "Selected rule.", "root"),
            {"type": "last-prompt", "leafUuid": "anchor"},
            claude_user("rewound", "New branch.", "root"),
        ],
    )
    source, diagnostics = decode_claude_history(read_snapshot(path), SESSION, PROJECT)
    assert source is None
    assert diagnostics[-1].kind == "ambiguous_final_branch"


def test_later_parallel_tails_with_same_humans_select_latest_tail(tmp_path: Path) -> None:
    path = tmp_path / f"{SESSION}.jsonl"
    write_jsonl(
        path,
        [
            claude_user("root", "Common rule."),
            claude_user("abandoned", "Discarded branch.", "root"),
            claude_user("anchor", "Selected rule.", "root"),
            {"type": "last-prompt", "leafUuid": "anchor"},
            claude_assistant("earlier-tail", "anchor"),
            claude_user("latest-tail", "[Request interrupted by user]", "anchor", userType="external", origin=None),
        ],
    )
    source, _ = decode_claude_history(read_snapshot(path), SESSION, PROJECT)
    assert source is not None
    assert [message.item for message in source.messages if message.kind == "user"] == ["root", "anchor"]
    assert [(annotation.kind, annotation.item, annotation.position) for annotation in source.annotations] == [
        ("user_interrupt", "latest-tail", 6)
    ]


def test_later_parallel_tails_with_different_humans_remain_ambiguous(tmp_path: Path) -> None:
    path = tmp_path / f"{SESSION}.jsonl"
    write_jsonl(
        path,
        [
            claude_user("root", "Common rule."),
            claude_user("anchor", "Selected rule.", "root"),
            {"type": "last-prompt", "leafUuid": "anchor"},
            claude_user("first-tail", "First later rule.", "anchor"),
            claude_user("second-tail", "Different later rule.", "anchor"),
        ],
    )
    source, diagnostics = decode_claude_history(read_snapshot(path), SESSION, PROJECT)
    assert source is None
    assert diagnostics[-1].kind == "ambiguous_final_branch"


@pytest.mark.parametrize("later_tail", [False, True])
@pytest.mark.parametrize("invalid_parent", [42, "", ["root"]])
def test_invalid_parent_on_abandoned_branch_does_not_reject_selected_history(
    tmp_path: Path, later_tail: bool, invalid_parent: object
) -> None:
    path = tmp_path / f"{SESSION}.jsonl"
    records = [
        claude_user("root", "Common rule."),
        claude_assistant("abandoned", None, parentUuid=invalid_parent),
        claude_user("anchor", "Selected rule.", "root"),
        {"type": "last-prompt", "leafUuid": "anchor"},
    ]
    if later_tail:
        records.append(claude_assistant("tail", "anchor"))
    write_jsonl(path, records)
    source, _ = decode_claude_history(read_snapshot(path), SESSION, PROJECT)
    assert source is not None
    assert [message.item for message in source.messages if message.kind == "user"] == ["root", "anchor"]


@pytest.mark.parametrize("invalid_parent", [42, "", ["root"]])
def test_invalid_parent_on_the_selected_chain_still_rejects_the_session(tmp_path: Path, invalid_parent: object) -> None:
    path = tmp_path / f"{SESSION}.jsonl"
    write_jsonl(
        path,
        [
            claude_user("root", "Common rule."),
            claude_assistant("invalid", None, parentUuid=invalid_parent),
            claude_user("anchor", "Selected rule.", "invalid"),
            {"type": "last-prompt", "leafUuid": "anchor"},
        ],
    )
    source, diagnostics = decode_claude_history(read_snapshot(path), SESSION, PROJECT)
    assert source is None
    assert diagnostics[-1].kind == "invalid_parent"
    assert diagnostics[-1].details == (("position", "2"),)
