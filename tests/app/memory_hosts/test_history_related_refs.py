# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Review references come from proven memory calls, never quotable evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.app.memory_hosts.history_fixtures import claude_user
from tests.app.memory_hosts.test_claude_history import decode as decode_claude
from tests.app.memory_hosts.test_codex_history import decode as decode_codex
from tests.app.memory_hosts.test_codex_history import message, meta
from tests.app.memory_hosts.test_history_sources import ccall, cresult, task, tool, xcall, xresult

FIRST = "project:111111111111"
SECOND = "global:222222222222"
THIRD = "project:333333333333"


@pytest.mark.parametrize("name", ["memory_read", "memory_save", "memory_forget", "memory_withdraw"])
def test_claude_related_refs_include_full_call_arguments_before_summary_truncation(tmp_path: Path, name: str) -> None:
    source, _ = decode_claude(
        tmp_path,
        [
            claude_user("u", "Review the saved guidance."),
            ccall("call", "u", "M", tool("a", "mcp__memory__" + name, {"body": "x" * 500, "ref": FIRST})),
            cresult("result", "call", "a", f"Saved {SECOND} @opaque-version. {SECOND}"),
        ],
    )
    assert source is not None
    assert source.related_refs == (FIRST, SECOND)
    assert [item.kind for item in source.messages] == ["user"]


def test_claude_related_refs_reuse_parallel_pairing_and_reject_unproven_branches(tmp_path: Path) -> None:
    source, _ = decode_claude(
        tmp_path,
        [
            claude_user("u", "Review."),
            ccall("selected", "u", "M", tool("a", "memory_read", {"ref": FIRST})),
            ccall("parallel", "selected", "M", tool("b", "memory_search", {"ref": "global:aaaaaaaaaaaa"})),
            cresult("parallel-result", "parallel", "b", SECOND),
            ccall("old", "u", "old-M", tool("c", "memory_read", {"ref": "project:aaaaaaaaaaaa"})),
            cresult("old-result", "old", "c", "global:bbbbbbbbbbbb"),
            ccall(
                "child", "selected", "M", tool("d", "memory_read", {"ref": "project:cccccccccccc"}), isSidechain=True
            ),
            cresult("child-result", "child", "d", "global:dddddddddddd"),
            cresult("orphan", "selected", "missing", "project:eeeeeeeeeeee"),
            cresult("selected-result", "selected", "a", THIRD),
            cresult("answered", "selected-result", "a", "global:ffffffffffff"),
            {"type": "last-prompt", "leafUuid": "answered"},
        ],
    )
    assert source is not None
    assert source.related_refs == (FIRST, SECOND, THIRD)
    assert [item.kind for item in source.messages] == ["user"]


def test_claude_nonmemory_text_and_invalid_argument_refs_are_not_review_targets(tmp_path: Path) -> None:
    source, _ = decode_claude(
        tmp_path,
        [
            claude_user("u", f"Mentioning {FIRST} is not a memory call."),
            ccall("call", "u", "M", tool("a", "shell", {"ref": SECOND}), tool("b", "memory_read", {"ref": "bad"})),
            cresult("result", "call", "a", THIRD),
        ],
    )
    assert source is not None and source.related_refs == ()
    assert source.messages[-1].kind == "tool_result"


@pytest.mark.parametrize("name", ["memory_read", "memory_save", "memory_forget", "memory_withdraw"])
def test_codex_related_refs_include_calls_and_paired_memory_outputs(tmp_path: Path, name: str) -> None:
    source, _ = decode_codex(
        tmp_path,
        [
            meta(tmp_path),
            message("Review the saved guidance."),
            xcall("a", "mcp__memory__" + name, '{"ref":"' + FIRST + '"}'),
            xresult("a", [{"type": "text", "text": SECOND}, {"type": "text", "text": SECOND}]),
        ],
    )
    assert source is not None and source.related_refs == (FIRST, SECOND)
    assert [item.kind for item in source.messages] == ["user"]


def test_codex_related_refs_apply_turn_pairing_and_ignore_nonmemory_refs(tmp_path: Path) -> None:
    source, _ = decode_codex(
        tmp_path,
        [
            meta(tmp_path),
            task("first"),
            message("Review."),
            xcall("old", "memory_read", '{"ref":"' + FIRST + '"}'),
            task("second"),
            xresult("old", "global:aaaaaaaaaaaa"),
            xcall("current", "memory_catalog"),
            xresult("current", SECOND),
            xresult("current", "project:bbbbbbbbbbbb"),
            xcall("shell", "exec_command", '{"ref":"project:cccccccccccc"}'),
            xresult("shell", "global:dddddddddddd"),
        ],
    )
    assert source is not None and source.related_refs == (FIRST, SECOND)
