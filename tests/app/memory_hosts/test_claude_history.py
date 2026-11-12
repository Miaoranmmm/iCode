# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Historical selection contracts from the memory trial step-five specification."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from chrys.app.memory_hosts import history as history_io
from chrys.app.memory_hosts.claude_code import history as claude_history
from chrys.app.memory_hosts.claude_code.history import ClaudeHistoryReader, decode_claude_history
from chrys.app.memory_hosts.history import HistoryError, read_snapshot
from chrys.service.memory.extraction import SessionKey, SourceRevision
from chrys.service.memory.project import bind_project
from chrys.service.memory.storage.store import MemoryScope
from tests.app.memory_hosts.history_fixtures import (
    PARENT,
    SESSION,
    FixedHolder,
    claude_assistant,
    claude_user,
    compact,
    write_jsonl,
)

PROJECT = MemoryScope("project", "0123456789abcdef")


def decode(tmp_path: Path, records: list[dict[str, object]]):
    path = tmp_path / f"{SESSION}.jsonl"
    write_jsonl(path, records)
    return decode_claude_history(read_snapshot(path), SESSION, PROJECT)


def kinds(notes):
    return {note.kind for note in notes}


def test_desktop_human_filters_and_inherited_content_reuse_exact_fragments(tmp_path: Path) -> None:
    records = [
        claude_user("inherited", "Inherited rule", sessionId=PARENT),
        claude_user("summary", "Summary", "inherited", isCompactSummary=True, turnOrigin="human"),
        claude_user("meta", "Continue from where you left off.", "summary", isMeta=True),
        claude_user("bash", "<bash-input>!echo fixture</bash-input>", "meta"),
        claude_user(
            "paste",
            'Before<pasted_content id="p">Quoted paste</pasted_content id="p">After',
            "bash",
            promptSource="sdk",
        ),
        claude_user(
            "reply", "<!-- reply -->\n> Model proposal\nUse clear names", "paste", origin=None, turnOrigin="human"
        ),
        claude_assistant("done", "reply"),
    ]
    source, notes = decode(tmp_path, records)
    assert source is not None
    assert [
        (message.item, message.position, message.fragments) for message in source.messages if message.kind == "user"
    ] == [
        ("inherited", 1, ("Inherited rule",)),
        ("paste", 5, ("Before", "Quoted paste", "After")),
        ("reply", 6, ("Use clear names",)),
    ]
    assert all(
        message.session == SESSION and message.at == "2026-11-01T02:00:00Z"
        for message in source.messages
        if message.kind == "user"
    )
    assert {"non_copy", "raw_human_markers", "eligible_human", "display_messages"} <= kinds(notes)


@pytest.mark.parametrize("entry", [None, "sdk-cli", "sdk-ts", "typed", "queued", "unknown"])
def test_unknown_or_non_desktop_user_entrypoint_rejects_entire_session(tmp_path: Path, entry: str | None) -> None:
    source, notes = decode(tmp_path, [claude_user("a", "Keep rules", entrypoint=entry)])
    assert source is None
    assert "unsupported_entrypoint" in kinds(notes)


def test_mixed_entrypoints_reject_and_report_real_human_count(tmp_path: Path) -> None:
    source, notes = decode(
        tmp_path,
        [
            claude_user("a", "Desktop"),
            claude_user("b", "SDK", "a", entrypoint="sdk-ts"),
            claude_user("c", "Summary", "b", isCompactSummary=True, turnOrigin="human"),
        ],
    )
    assert source is None
    assert "mixed_entrypoint" in kinds(notes)
    assert next(note.count for note in notes if note.kind == "raw_human_markers") == 3
    assert next(note.count for note in notes if note.kind == "eligible_human") == 2


@pytest.mark.parametrize("fields", [{"sessionId": None}, {"sessionId": ""}, {"sessionId": 123}])
def test_missing_identity_is_unknown_format_not_copy(tmp_path: Path, fields: dict[str, object]) -> None:
    source, notes = decode(tmp_path, [claude_user("a", "Rule", **fields)])
    assert source is None and "unknown_format" in kinds(notes) and "copied_only" not in kinds(notes)


def test_only_valid_other_session_records_are_copied_only(tmp_path: Path) -> None:
    source, notes = decode(tmp_path, [claude_user("a", "Rule", sessionId=PARENT)])
    assert source is None and "copied_only" in kinds(notes)
    source, notes = decode(tmp_path, [{"type": "last-prompt", "leafUuid": None}])
    assert source is None and "empty_session" in kinds(notes) and "copied_only" not in kinds(notes)


def test_last_prompt_without_explicit_selects_final_branch(tmp_path: Path) -> None:
    source, _ = decode(
        tmp_path,
        [
            claude_user("root", "Common"),
            claude_user("old", "Abandoned", "root"),
            claude_user("new", "Chosen", "root"),
            {"type": "last-prompt", "leafUuid": "old"},
        ],
    )
    assert source is not None
    assert [message.item for message in source.messages if message.kind == "user"] == ["root", "old"]


def test_checkpoint_followed_by_descendant_dialogue_extends_chain(tmp_path: Path) -> None:
    source, _ = decode(
        tmp_path,
        [
            claude_user("root", "Common"),
            {"type": "last-prompt", "leafUuid": "root"},
            claude_user("new", "New rule", "root"),
            claude_assistant("done", "new"),
        ],
    )
    assert source is not None and [message.item for message in source.messages if message.kind == "user"] == [
        "root",
        "new",
    ]


def test_checkpoint_cannot_authorize_a_non_descendant_new_tail(tmp_path: Path) -> None:
    source, notes = decode(
        tmp_path,
        [
            claude_user("root", "Common"),
            claude_user("old", "Older", "root"),
            {"type": "last-prompt", "leafUuid": "old"},
            claude_user("new", "Other", "root"),
        ],
    )
    assert source is None and "ambiguous_final_branch" in kinds(notes)


@pytest.mark.parametrize("leaf", [None, "00000000-0000-0000-0000-000000000000"])
def test_ordinary_null_and_zero_anchors_cannot_resolve_human_branches(tmp_path: Path, leaf: str | None) -> None:
    source, notes = decode(
        tmp_path,
        [
            claude_user("root", "Common"),
            claude_user("old", "Older", "root"),
            claude_user("new", "Other", "root"),
            {"type": "last-prompt", "leafUuid": leaf},
        ],
    )
    assert source is None and "ambiguous_final_branch" in kinds(notes)


def test_explicit_null_is_an_empty_chain(tmp_path: Path) -> None:
    source, notes = decode(
        tmp_path, [claude_user("old", "Rule"), {"type": "last-prompt", "leafUuid": None, "explicit": True}]
    )
    assert source is None and "no_user_messages" in kinds(notes)


def test_parallel_tool_leaves_with_same_human_ancestry_are_allowed(tmp_path: Path) -> None:
    source, _ = decode(
        tmp_path,
        [
            claude_user("root", "Common"),
            claude_assistant("first", "root"),
            claude_assistant("second", "first"),
            claude_user("r1", [{"type": "tool_result", "content": "First"}], "first"),
            claude_user("r2", [{"type": "tool_result", "content": "Second"}], "second"),
        ],
    )
    assert source is not None and [message.item for message in source.messages if message.kind == "user"] == ["root"]


def test_head_boundary_points_forward_to_parent_chain_returning_to_boundary(tmp_path: Path) -> None:
    source, notes = decode(
        tmp_path,
        [
            {"type": "file-history-snapshot"},
            compact("boundary", "target"),
            claude_user("target", "Available original", "boundary"),
            claude_assistant("done", "target"),
        ],
    )
    assert source is not None and [message.item for message in source.messages if message.kind == "user"] == ["target"]
    assert {"head_boundary", "earlier_history_unavailable"} <= kinds(notes)
    diagnostic = next(note for note in notes if note.kind == "head_boundary")
    assert dict(diagnostic.details) == {"boundary_position": "2", "target_position": "3", "returns_to_boundary": "yes"}


def test_back_bridge_uses_original_occurrence_before_replayed_uuid(tmp_path: Path) -> None:
    source, notes = decode(
        tmp_path,
        [
            claude_user("original", "Old original"),
            claude_assistant("target", "original"),
            compact("boundary", "target"),
            claude_user("original", "Old original", "boundary"),
            claude_user("new", "New original", "original"),
            claude_assistant("done", "new"),
        ],
    )
    assert source is not None
    assert [(message.item, message.position) for message in source.messages if message.kind == "user"] == [
        ("original", 1),
        ("new", 5),
    ]
    assert "head_boundary" not in kinds(notes)


@pytest.mark.parametrize(
    "records, expected",
    [
        ([compact("boundary", "missing"), claude_user("u", "Rule", "boundary")], "missing_compact_bridge"),
        (
            [claude_user("prior", "Prior"), compact("boundary", "target"), claude_user("target", "Rule", "boundary")],
            "unknown_head_boundary",
        ),
        ([compact("boundary", "target"), claude_user("target", "Rule")], "no_error"),
        ([claude_user("a", "First", "b"), claude_assistant("b", "a")], "invalid_chain"),
        ([claude_user("a", "First", "missing")], "invalid_chain"),
        (
            [claude_user("a", "First"), compact("boundary", "a"), claude_user("a", "Changed", "boundary")],
            "conflicting_uuid",
        ),
    ],
)
def test_compact_missing_unknown_shapes_and_cycles_fail_closed(
    tmp_path: Path, records: list[dict[str, object]], expected: str
) -> None:
    source, notes = decode(tmp_path, records)
    if expected == "no_error":
        assert source is not None
    else:
        assert source is None and expected in kinds(notes)


@pytest.mark.parametrize("marker", ["[Request interrupted by user]", "[Request interrupted by user for tool use]"])
def test_structural_interrupt_annotations_keep_order_and_are_separate_from_words(tmp_path: Path, marker: str) -> None:
    source, _ = decode(
        tmp_path,
        [
            claude_user("root", "Common"),
            claude_user(
                "interruption",
                [{"type": "text", "text": marker}],
                "root",
                userType="external",
                origin=None,
                isMeta=False,
            ),
            claude_user("abandoned", "Omitted branch", "interruption"),
            claude_user("chosen", "Keep this", "interruption"),
            claude_user("later-interruption", marker, "chosen", userType="external", origin=None),
            claude_user("after", "After the interruption", "later-interruption"),
            {"type": "last-prompt", "leafUuid": "after"},
        ],
    )
    assert source is not None
    assert [(item.kind, item.item, item.position) for item in source.annotations] == [
        ("user_interrupt", "interruption", 2),
        ("user_interrupt", "later-interruption", 5),
    ]
    assert [message.item for message in source.messages if message.kind == "user"] == ["root", "chosen", "after"]


def test_parallel_tool_use_and_result_fork_produces_no_structural_annotation(tmp_path: Path) -> None:
    source, _ = decode(
        tmp_path,
        [
            claude_user("root", "Use clear names."),
            claude_assistant(
                "call-a", "root", message={"content": [{"type": "tool_use", "id": "a", "name": "Read", "input": {}}]}
            ),
            claude_assistant(
                "call-b", "root", message={"content": [{"type": "tool_use", "id": "b", "name": "Read", "input": {}}]}
            ),
            claude_user("result-a", [{"type": "tool_result", "tool_use_id": "a", "content": "Synthetic A"}], "call-a"),
            claude_user("result-b", [{"type": "tool_result", "tool_use_id": "b", "content": "Synthetic B"}], "call-b"),
            {"type": "last-prompt", "leafUuid": "result-b"},
        ],
    )
    assert source is not None
    assert [message.item for message in source.messages if message.kind == "user"] == ["root"]
    assert source.annotations == ()


def test_human_literal_interrupt_and_literal_signal_tag_do_not_generate_annotations(tmp_path: Path) -> None:
    source, _ = decode(
        tmp_path,
        [
            claude_user("typed", "[Request interrupted by user]"),
            claude_user(
                "paste",
                '<pasted_content id="p"><structure_signal kind="user_interrupt"/></pasted_content id="p">',
                "typed",
            ),
        ],
    )
    assert source is not None and source.annotations == ()
    assert source.messages[1].fragments == ('<structure_signal kind="user_interrupt"/>',)


@pytest.mark.parametrize("marker", ["[Request interrupted by user]", "[Request interrupted by user for tool use]"])
@pytest.mark.parametrize("human", [{"origin": {"kind": "human"}}, {"origin": None, "turnOrigin": "human"}])
def test_human_marked_external_interrupt_literal_is_user_text(
    tmp_path: Path,
    marker: str,
    human: dict[str, object],
) -> None:
    source, notes = decode(tmp_path, [claude_user("typed", marker, userType="external", **human)])
    assert source is not None
    assert source.annotations == ()
    assert source.messages[0].fragments == (marker,)
    assert "unsupported_interrupt_shape" not in kinds(notes)


def test_annotation_parent_resolution_stays_linear_for_a_long_selected_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 300
    records = [
        claude_user(f"user-{index}", f"Rule {index}", f"user-{index - 1}" if index else None) for index in range(count)
    ]
    original = claude_history._HistoryChain._plain_parent
    resolutions = 0

    def counted(
        self: claude_history._HistoryChain,
        node: claude_history._HistoryNode,
    ) -> claude_history._Occurrence | None:
        nonlocal resolutions
        resolutions += 1
        return original(self, node)

    monkeypatch.setattr(claude_history._HistoryChain, "_plain_parent", counted)
    source, _ = decode(tmp_path, records)
    assert source is not None and len(source.messages) == count
    assert source.annotations == ()
    assert resolutions <= 4 * count


@pytest.mark.parametrize("suffix,kind", [(b"broken\n", "invalid_json"), (b'{"half":', "partial_line")])
def test_finished_snapshot_never_silently_accepts_broken_eof(tmp_path: Path, suffix: bytes, kind: str) -> None:
    path = tmp_path / "fixture.jsonl"
    write_jsonl(path, [claude_user("u", "Words")], suffix)
    with pytest.raises(HistoryError) as error:
        read_snapshot(path)
    assert error.value.kind == kind


@pytest.mark.parametrize("activity", ["active", "unknown"])
async def test_reader_does_not_accept_holder_active_or_unknown(tmp_path: Path, activity) -> None:
    write_jsonl(tmp_path / f"{SESSION}.jsonl", [claude_user("u", "Words")])
    result = await ClaudeHistoryReader(tmp_path, holder=FixedHolder(activity)).scan(PROJECT)
    assert result.sessions == () and activity in kinds(result.diagnostics)


async def test_all_copy_scan_is_fatal_and_does_not_write_any_host_state(tmp_path: Path) -> None:
    path = tmp_path / f"{SESSION}.jsonl"
    write_jsonl(path, [claude_user("u", "Words", sessionId=PARENT)])
    before = path.read_bytes()
    result = await ClaudeHistoryReader(tmp_path, holder=FixedHolder()).scan(PROJECT)
    assert result.fatal is True and result.sessions == () and path.read_bytes() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]


async def test_revalidation_checks_original_snapshot_and_holder_and_replay_bypasses_only_holder(tmp_path: Path) -> None:
    path = tmp_path / f"{SESSION}.jsonl"
    project = await asyncio.to_thread(bind_project, str(tmp_path))
    write_jsonl(path, [claude_user("u", "Words", cwd=str(tmp_path))])
    holder = FixedHolder()
    reader = ClaudeHistoryReader(tmp_path, holder=holder)
    source = (await reader.scan(project)).sessions[0]
    assert source.key == SessionKey("claude-code", SESSION)
    assert await reader.revalidate(source) is None
    holder.activity = "active"
    assert await reader.revalidate(source) == "source_reopened"
    replay = ClaudeHistoryReader(tmp_path, holder=holder, replay=True)
    source = (await replay.read_files(project, (path,))).sessions[0]
    assert await replay.revalidate(replace(source, revision=SourceRevision(1, source.revision.sha256))) is None
    write_jsonl(path, [claude_user("u", "Changed words")])
    assert await replay.revalidate(source) == "source_changed"


@pytest.mark.parametrize(
    "setting, limit, expected",
    [("MAX_HISTORY_BYTES", 16, "source_too_large"), ("MAX_HISTORY_LINE_BYTES", 16, "source_line_too_large")],
)
def test_snapshot_allocation_limits_are_explicit_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting: str, limit: int, expected: str
) -> None:
    path = tmp_path / "bounded.jsonl"
    write_jsonl(path, [claude_user("u", "Synthetic long content")])
    monkeypatch.setattr(history_io, setting, limit)
    with pytest.raises(HistoryError) as error:
        read_snapshot(path)
    assert error.value.kind == expected


@pytest.mark.parametrize("directory_exists", [True, False])
async def test_missing_source_or_no_top_level_session_files_is_fatal(tmp_path: Path, directory_exists: bool) -> None:
    root = tmp_path if directory_exists else tmp_path / "missing"
    result = await ClaudeHistoryReader(root, holder=FixedHolder()).scan(PROJECT)
    assert result.fatal is True and "no_session_files" in kinds(result.diagnostics)


async def test_only_top_level_files_are_read_and_unknown_identity_with_copy_is_fatal(tmp_path: Path) -> None:
    write_jsonl(tmp_path / f"{SESSION}.jsonl", [claude_user("copy", "Inherited", sessionId=PARENT)])
    write_jsonl(tmp_path / f"{PARENT}.jsonl", [claude_user("unknown", "Unknown", sessionId=None)])
    nested = tmp_path / "subagents"
    nested.mkdir()
    write_jsonl(nested / "not-read.jsonl", [claude_user("own", "Nested")])
    result = await ClaudeHistoryReader(tmp_path, holder=FixedHolder()).scan(PROJECT)
    assert result.fatal is True and result.sessions == ()
    assert {"unknown_format", "copied_only"} <= kinds(result.diagnostics)
    assert sum(note.count for note in result.diagnostics if note.kind == "files") == 2


async def test_unconfirmed_project_group_requires_explicit_session_selection(tmp_path: Path) -> None:
    path = tmp_path / f"{SESSION}.jsonl"
    write_jsonl(path, [claude_user("u", "Words", cwd=str(tmp_path / "removed-workspace"))])
    default = await ClaudeHistoryReader(tmp_path, holder=FixedHolder()).scan(PROJECT)
    assert default.sessions == () and "source_mapping_unconfirmed" in kinds(default.diagnostics)
    note = next(note for note in default.diagnostics if note.kind == "source_mapping_unconfirmed")
    assert note.details == (("reason", "unbound"),)
    selected = await ClaudeHistoryReader(tmp_path, holder=FixedHolder(), sessions=frozenset({SESSION})).scan(PROJECT)
    assert len(selected.sessions) == 1 and "source_group" in kinds(selected.diagnostics)
    assert (
        next(note for note in selected.diagnostics if note.kind == "source_mapping_unconfirmed").details == note.details
    )


@pytest.mark.parametrize("shape", ["later_deleted", "later_other_repo", "inherited_first", "compact_head"])
async def test_only_the_first_self_produced_user_cwd_binds_the_project(
    tmp_path: Path, git_repo_factory: Callable[[Path], Path], shape: str
) -> None:
    project_dir = await asyncio.to_thread(git_repo_factory, tmp_path / "project")
    other = await asyncio.to_thread(git_repo_factory, tmp_path / "other")
    project = await asyncio.to_thread(bind_project, str(project_dir))
    deleted = project_dir / "deleted-subdirectory"
    await asyncio.to_thread(deleted.mkdir)
    await asyncio.to_thread(deleted.rmdir)
    path = tmp_path / f"{SESSION}.jsonl"
    if shape == "inherited_first":
        records = [
            claude_user("inherited", "Inherited rule.", sessionId=PARENT, cwd=str(other)),
            claude_user("start", "Own rule.", "inherited", cwd=str(project_dir)),
        ]
        all_cwds = {str(other), str(project_dir)}
    elif shape == "compact_head":
        records = [
            {**compact("boundary", "kept"), "sessionId": SESSION},
            claude_user("summary", "Summary.", "boundary", isCompactSummary=True, cwd=str(project_dir)),
            claude_user("kept", "Preserved original rule.", "summary", cwd=str(deleted)),
        ]
        all_cwds = {str(project_dir), str(deleted)}
    else:
        later = other if shape == "later_other_repo" else deleted
        records = [
            claude_user("start", "Starting rule.", cwd=str(project_dir)),
            claude_user("later", "Later rule.", "start", cwd=str(later)),
        ]
        all_cwds = {str(project_dir), str(later)}
    await asyncio.to_thread(write_jsonl, path, records)
    scanned = await ClaudeHistoryReader(tmp_path, holder=FixedHolder()).scan(project)
    assert len(scanned.sessions) == 1
    assert "source_mapping_unconfirmed" not in kinds(scanned.diagnostics)
    group = next(note for note in scanned.diagnostics if note.kind == "source_group")
    assert group.details == (*(("cwd", cwd) for cwd in sorted(all_cwds)), ("start_cwd", str(project_dir)))


async def test_starting_in_another_existing_repo_is_unconfirmed(
    tmp_path: Path, git_repo_factory: Callable[[Path], Path]
) -> None:
    project_dir = await asyncio.to_thread(git_repo_factory, tmp_path / "project")
    other = await asyncio.to_thread(git_repo_factory, tmp_path / "other")
    project = await asyncio.to_thread(bind_project, str(project_dir))
    await asyncio.to_thread(
        write_jsonl,
        tmp_path / f"{SESSION}.jsonl",
        [
            claude_user("start", "Starting rule.", cwd=str(other)),
            claude_user("later", "Later rule.", "start", cwd=str(project_dir)),
        ],
    )
    scanned = await ClaudeHistoryReader(tmp_path, holder=FixedHolder()).scan(project)
    assert scanned.sessions == ()
    note = next(note for note in scanned.diagnostics if note.kind == "source_mapping_unconfirmed")
    assert note.details == (("reason", "other_project"),)


@pytest.mark.parametrize("fields", [{}, {"cwd": None}, {"cwd": ""}, {"cwd": 42}, {"cwd": ["invalid"]}])
async def test_invalid_first_self_user_cwd_never_skips_forward_to_a_later_cwd(
    tmp_path: Path, fields: dict[str, object]
) -> None:
    project = await asyncio.to_thread(bind_project, str(tmp_path))
    await asyncio.to_thread(
        write_jsonl,
        tmp_path / f"{SESSION}.jsonl",
        [
            claude_user("start", "Starting rule.", **fields),
            claude_user("later", "Later rule.", "start", cwd=str(tmp_path)),
        ],
    )
    scanned = await ClaudeHistoryReader(tmp_path, holder=FixedHolder()).scan(project)
    assert scanned.sessions == ()
    note = next(note for note in scanned.diagnostics if note.kind == "source_mapping_unconfirmed")
    assert note.details == (("reason", "no_start"),)
    group = next(note for note in scanned.diagnostics if note.kind == "source_group")
    assert ("start_cwd", "<unrecorded>") in group.details


async def test_without_a_self_user_record_the_project_is_unconfirmed(tmp_path: Path) -> None:
    project = await asyncio.to_thread(bind_project, str(tmp_path))
    await asyncio.to_thread(
        write_jsonl,
        tmp_path / f"{SESSION}.jsonl",
        [
            claude_user("inherited", "Inherited rule.", sessionId=PARENT, cwd=str(tmp_path)),
            claude_assistant("own", "inherited"),
        ],
    )
    scanned = await ClaudeHistoryReader(tmp_path, holder=FixedHolder()).scan(project)
    assert scanned.sessions == ()
    assert next(note for note in scanned.diagnostics if note.kind == "source_mapping_unconfirmed").details == (
        ("reason", "no_start"),
    )


@pytest.mark.parametrize("shape", ["copied_only", "unsupported_entrypoint", "missing_compact_bridge"])
async def test_decoder_rejections_never_run_the_project_mapping_check(tmp_path: Path, shape: str) -> None:
    missing = str(tmp_path / "removed-workspace")
    if shape == "copied_only":
        records = [claude_user("copy", "Inherited rule.", sessionId=PARENT, cwd=missing)]
    elif shape == "unsupported_entrypoint":
        records = [claude_user("own", "Own rule.", entrypoint="sdk", cwd=missing)]
    else:
        records = [compact("boundary", "missing"), claude_user("own", "Own rule.", "boundary", cwd=missing)]
    await asyncio.to_thread(write_jsonl, tmp_path / f"{SESSION}.jsonl", records)
    scanned = await ClaudeHistoryReader(tmp_path, holder=FixedHolder()).scan(PROJECT)
    assert scanned.sessions == () and shape in kinds(scanned.diagnostics)
    assert "source_mapping_unconfirmed" not in kinds(scanned.diagnostics)


def test_unknown_interrupt_shape_is_reported_without_changing_human_source_rules(tmp_path: Path) -> None:
    source, notes = decode(
        tmp_path,
        [
            claude_user("human", "Rule"),
            claude_user("unsupported", "[Request interrupted by user]", "human", origin=None),
        ],
    )
    assert source is not None and [message.item for message in source.messages if message.kind == "user"] == ["human"]
    assert source.annotations == () and "unsupported_interrupt_shape" in kinds(notes)


def test_sidechain_and_agent_children_do_not_create_branch_signals(tmp_path: Path) -> None:
    source, _ = decode(
        tmp_path,
        [
            claude_user("root", "Rule"),
            claude_user("side", "Excluded", "root", isSidechain=True),
            claude_user("agent", "Excluded", "root", agentId="synthetic-child"),
            claude_user("main", "Selected", "root"),
        ],
    )
    assert source is not None and source.annotations == ()
