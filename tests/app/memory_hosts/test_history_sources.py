# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Engineering history sources use host structure and synthetic records only."""

from __future__ import annotations

from pathlib import Path

import pytest

from chrys.service.memory.extraction import SourceAnnotation
from tests.app.memory_hosts.history_fixtures import SESSION, claude_assistant, claude_user, compact
from tests.app.memory_hosts.test_claude_history import decode as decode_claude
from tests.app.memory_hosts.test_codex_history import decode as decode_codex
from tests.app.memory_hosts.test_codex_history import message, meta


def tool(call_id: str, name: str = "shell", arguments: object = None) -> dict[str, object]:
    return {"type": "tool_use", "id": call_id, "name": name, "input": {} if arguments is None else arguments}


def ccall(
    item: str, parent: str, message_id: str | None, *blocks: dict[str, object], **fields: object
) -> dict[str, object]:
    body: dict[str, object] = {"content": list(blocks)}
    if message_id is not None:
        body["id"] = message_id
    return claude_assistant(item, parent, message=body, **fields)


def cresult(item: str, parent: str, call_id: str, output: object, **fields: object) -> dict[str, object]:
    return claude_user(
        item, [{"type": "tool_result", "tool_use_id": call_id, "content": output, **fields}], parent, origin=None
    )


def txt(value: str) -> dict[str, object]:
    return {"type": "text", "text": value}


def test_claude_history_reuses_parallel_collector_and_only_selected_assistant_text(tmp_path: Path) -> None:
    source, _ = decode_claude(
        tmp_path,
        [
            claude_user("u", "Inspect both paths."),
            ccall("a", "u", "M", txt("Selected proposal."), tool("a", "Read", {"path": "a.py"})),
            ccall("b", "a", "M", txt("Sibling proposal is not selected."), tool("b", "Read", {"path": "b.py"})),
            cresult("ra", "a", "a", "First observation."),
            cresult("rb", "b", "b", [txt("Second observation."), {"type": "image", "source": {}}], is_error=True),
            {"type": "last-prompt", "leafUuid": "ra"},
        ],
    )
    assert source is not None
    assert [(item.kind, item.item, item.part, item.fragments) for item in source.messages] == [
        ("user", "u", 0, ("Inspect both paths.",)),
        ("assistant", "a", 0, ("Selected proposal.",)),
        ("tool_result", "ra", 0, ("First observation.",)),
        ("tool_result", "rb", 0, ("Second observation.",)),
    ]
    assert dict(source.messages[-1].attributes) == {"tool": "Read", "args": '{"path":"b.py"}', "is_error": "true"}


@pytest.mark.parametrize("shape", ["rewind", "retry", "missing_id", "sidechain"])
def test_claude_history_sibling_counter_shapes_never_admit_results(tmp_path: Path, shape: str) -> None:
    message_id = None if shape == "missing_id" else "M" if shape == "sidechain" else "other-response"
    parent = "u" if shape == "rewind" else "a"
    fields = {"isSidechain": True} if shape == "sidechain" else {}
    source, _ = decode_claude(
        tmp_path,
        [
            claude_user("u", "Inspect."),
            ccall("a", "u", "M", tool("a")),
            ccall("b", parent, message_id, tool("b"), **fields),
            cresult("rb", "b", "b", "Excluded."),
            cresult("ra", "a", "a", "Included."),
            {"type": "last-prompt", "leafUuid": "ra"},
        ],
    )
    assert source is not None
    assert [item.item for item in source.messages if item.kind == "tool_result"] == ["ra"]


def test_claude_compact_replay_retains_earliest_sources_for_every_kind(tmp_path: Path) -> None:
    source, _ = decode_claude(
        tmp_path,
        [
            claude_user("u", "Original request."),
            ccall("a", "u", "old-M", txt("Original proposal."), tool("a")),
            cresult("r", "a", "a", "Original observation."),
            compact("boundary", "r"),
            claude_user("u", "Original request.", "boundary"),
            ccall("a", "u", "new-M", txt("Original proposal."), tool("a")),
            cresult("r", "a", "a", "Original observation."),
            ccall("last", "r", "last-M", txt("Current conclusion.")),
        ],
    )
    assert source is not None
    assert [(item.kind, item.item, item.position) for item in source.messages] == [
        ("user", "u", 1),
        ("assistant", "a", 2),
        ("tool_result", "r", 3),
        ("assistant", "last", 8),
    ]


def test_claude_failure_signal_tracks_each_tool_and_exact_result_parts(tmp_path: Path) -> None:
    combined = claude_user(
        "results",
        [
            {"type": "image", "source": {}},
            {"type": "tool_result", "tool_use_id": "a", "content": "First", "is_error": True},
            {"type": "tool_result", "tool_use_id": "other", "content": "Other succeeded"},
            {"type": "tool_result", "tool_use_id": "b", "content": "Second", "is_error": True},
            {"type": "tool_result", "tool_use_id": "c", "content": "Recovered"},
        ],
        "calls",
        origin=None,
    )
    source, _ = decode_claude(
        tmp_path,
        [
            claude_user("u", "Inspect."),
            ccall("calls", "u", "M", tool("a"), tool("other", "Read"), tool("b"), tool("c")),
            combined,
            ccall("conclusion", "results", "last-M", txt("A conclusion.")),
        ],
    )
    assert source is not None
    assert source.annotations == (
        SourceAnnotation("tool_failures_then_success", 3, "results", start_position=3, start_part=1, part=4),
    )
    assert [dict(item.attributes)["is_error"] for item in source.messages if item.kind == "tool_result"] == [
        "true",
        "false",
        "true",
        "false",
    ]


@pytest.mark.parametrize("reset", [False, True])
def test_claude_error_text_does_not_count_and_real_user_resets_streak(tmp_path: Path, reset: bool) -> None:
    records = [
        claude_user("u", "Inspect."),
        ccall("a", "u", "M1", tool("a")),
        cresult("ra", "a", "a", "Error text is not the flag.", is_error=True),
        ccall("b", "ra", "M2", tool("b")),
        cresult("rb", "b", "b", "Second failure.", is_error=True),
    ]
    parent = "rb"
    if reset:
        records.append(claude_user("new-user", "A new human request.", parent))
        parent = "new-user"
    records.extend(
        [
            ccall("c", parent, "M3", tool("c")),
            cresult("rc", "c", "c", "Error: misleading prose on a successful result."),
        ]
    )
    source, _ = decode_claude(tmp_path, records)
    assert source is not None
    assert [item.kind for item in source.annotations] == ([] if reset else ["tool_failures_then_success"])


def test_claude_same_response_id_never_crosses_compact_segments(tmp_path: Path) -> None:
    source, _ = decode_claude(
        tmp_path,
        [
            claude_user("u", "Old request."),
            ccall("old", "u", "shared-M", tool("old")),
            cresult("old-result", "old", "old", "Old output."),
            compact("boundary", "u"),
            claude_user("new", "New request.", "boundary"),
            ccall("selected", "new", "shared-M", tool("new")),
            cresult("unpaired", "selected", "old", "Must not pair backward."),
            cresult("new-result", "selected", "new", "Current output."),
            {"type": "last-prompt", "leafUuid": "new-result"},
        ],
    )
    assert source is not None
    assert [item.item for item in source.messages if item.kind == "tool_result"] == ["new-result"]


def task(turn: str) -> dict[str, object]:
    return {"type": "event_msg", "payload": {"type": "task_started", "turn_id": turn}}


def xcall(call_id: str, name: str, arguments: str = "{}", *, custom: bool = False) -> dict[str, object]:
    return {
        "type": "response_item",
        "payload": {
            "type": "custom_tool_call" if custom else "function_call",
            "call_id": call_id,
            "name": name,
            "input" if custom else "arguments": arguments,
        },
    }


def xresult(call_id: str, output: object, *, custom: bool = False) -> dict[str, object]:
    return {
        "type": "response_item",
        "payload": {
            "type": "custom_tool_call_output" if custom else "function_call_output",
            "call_id": call_id,
            "output": output,
        },
    }


def xassistant(*blocks: dict[str, object]) -> dict[str, object]:
    return {
        "type": "response_item",
        "payload": {"type": "message", "role": "assistant", "id": "assistant-message", "content": list(blocks)},
    }


def test_codex_assistant_and_function_custom_outputs_are_independent_sources(tmp_path: Path) -> None:
    source, notes = decode_codex(
        tmp_path,
        [
            meta(tmp_path),
            task("turn-one"),
            message("Inspect."),
            xassistant(
                {"type": "output_text", "text": "Proposal."},
                {"type": "reasoning", "text": "Hidden."},
                {"type": "output_text", "text": "Second proposal."},
            ),
            xcall("a", "exec_command", '{"cmd":"pytest"}'),
            xresult("a", "Observed failure."),
            xcall("b", "custom-editor", "edit synthetic", custom=True),
            xresult("b", [txt("First output."), {"type": "image", "source": {}}, txt("Last output.")], custom=True),
        ],
    )
    assert source is not None
    assert [(item.kind, item.part, item.fragments) for item in source.messages] == [
        ("user", 0, ("Inspect.",)),
        ("assistant", 0, ("Proposal.",)),
        ("assistant", 2, ("Second proposal.",)),
        ("tool_result", 0, ("Observed failure.",)),
        ("tool_result", 0, ("First output.", "Last output.")),
    ]
    assert dict(source.messages[3].attributes) == {"tool": "exec_command", "args": '{"cmd":"pytest"}'}
    assert dict(source.messages[4].attributes) == {"tool": "custom-editor", "args": '"edit synthetic"'}
    assert source.annotations == () and any(note.kind == "signals_unavailable" for note in notes)


def test_codex_task_started_boundaries_block_old_unanswered_ids(tmp_path: Path) -> None:
    source, _ = decode_codex(
        tmp_path,
        [
            meta(tmp_path),
            task("first"),
            message("Inspect."),
            xcall("reused", "old-tool"),
            task("second"),
            xresult("reused", "Old turn is out of scope."),
            xcall("reused", "new-tool"),
            xresult("reused", "New turn result."),
            xresult("reused", "Already answered."),
        ],
    )
    assert source is not None
    assert [
        (dict(item.attributes)["tool"], item.fragments) for item in source.messages if item.kind == "tool_result"
    ] == [("new-tool", ("New turn result.",))]


def test_codex_without_task_markers_treats_whole_file_as_one_turn(tmp_path: Path) -> None:
    source, _ = decode_codex(
        tmp_path,
        [
            meta(tmp_path),
            message("First", item="first-user"),
            xcall("a", "exec_command"),
            message("Second", item="second-user"),
            xresult("a", "Same file turn."),
        ],
    )
    assert source is not None
    assert source.messages[-1].kind == "tool_result" and source.messages[-1].fragments == ("Same file turn.",)


@pytest.mark.parametrize("name", ["memory_read", "mcp__memory__memory_save", "memory_withdraw"])
def test_codex_memory_results_cannot_become_new_evidence(tmp_path: Path, name: str) -> None:
    source, _ = decode_codex(
        tmp_path,
        [meta(tmp_path), message("Inspect."), xcall("a", name), xresult("a", "Stored words must not be evidence.")],
    )
    assert source is not None and [item.kind for item in source.messages] == ["user"]


def test_codex_nearest_reused_call_and_unanswered_pairing_preserve_provenance(tmp_path: Path) -> None:
    source, _ = decode_codex(
        tmp_path,
        [
            meta(tmp_path),
            message("Inspect."),
            xresult("a", "Output before a call."),
            xcall("a", "first-tool"),
            xresult("a", "First."),
            xcall("a", "second-tool"),
            xresult("a", "Second."),
            xresult("a", "No third call."),
        ],
    )
    assert source is not None
    assert [
        (dict(item.attributes)["tool"], item.fragments) for item in source.messages if item.kind == "tool_result"
    ] == [("first-tool", ("First.",)), ("second-tool", ("Second.",))]


@pytest.mark.parametrize("host", ["claude", "codex"])
def test_new_source_kinds_do_not_rescue_a_session_without_eligible_user_words(tmp_path: Path, host: str) -> None:
    if host == "claude":
        source, notes = decode_claude(
            tmp_path,
            [
                claude_user("meta", "Host continuation.", isMeta=True),
                ccall("a", "meta", "M", txt("An assistant finding."), tool("a")),
                cresult("r", "a", "a", "An observation."),
            ],
        )
    else:
        source, notes = decode_codex(
            tmp_path,
            [
                meta(tmp_path),
                message("Environment, not user words.", provenance="environment"),
                xassistant({"type": "output_text", "text": "An assistant finding."}),
                xcall("a", "exec_command"),
                xresult("a", "An observation."),
            ],
        )
    assert source is None and any(note.kind == "no_user_messages" for note in notes)


def test_codex_failure_prose_never_becomes_a_structural_signal(tmp_path: Path) -> None:
    source, notes = decode_codex(
        tmp_path,
        [
            meta(tmp_path),
            message("Inspect."),
            xcall("a", "exec_command"),
            xresult("a", "Error: first failure"),
            xcall("b", "exec_command"),
            xresult("b", "Error: second failure"),
            xcall("c", "exec_command"),
            xresult("c", "Success"),
        ],
    )
    assert source is not None and source.annotations == ()
    assert any(note.kind == "signals_unavailable" for note in notes)


def test_claude_image_only_human_message_resets_tool_failure_streak(tmp_path: Path) -> None:
    source, _ = decode_claude(
        tmp_path,
        [
            claude_user("u", "Inspect."),
            ccall("a", "u", "M1", tool("a")),
            cresult("ra", "a", "a", "First failure.", is_error=True),
            ccall("b", "ra", "M2", tool("b")),
            cresult("rb", "b", "b", "Second failure.", is_error=True),
            claude_user("image", [{"type": "image", "source": {}}], "rb"),
            ccall("c", "image", "M3", tool("c")),
            cresult("rc", "c", "c", "Recovered."),
        ],
    )
    assert source is not None and source.annotations == ()


def test_claude_empty_error_outputs_still_supply_structural_failure_locations(tmp_path: Path) -> None:
    source, _ = decode_claude(
        tmp_path,
        [
            claude_user("u", "Inspect."),
            ccall("a", "u", "M1", tool("a")),
            cresult("ra", "a", "a", "", is_error=True),
            ccall("b", "ra", "M2", tool("b")),
            cresult("rb", "b", "b", [], is_error=True),
            ccall("c", "rb", "M3", tool("c")),
            cresult("rc", "c", "c", "Recovered."),
        ],
    )
    assert source is not None
    assert source.annotations == (SourceAnnotation("tool_failures_then_success", 7, "rc", start_position=3),)
    assert [item.item for item in source.messages if item.kind == "tool_result"] == ["rc"]


@pytest.mark.parametrize("marker", [{"isSidechain": True}, {"agentId": "synthetic-child"}])
@pytest.mark.parametrize(
    "bad", [{"entrypoint": "sdk-cli"}, {"entrypoint": None}, {"sessionId": None}, {"sessionId": 123}]
)
def test_claude_sidechain_identity_and_entrypoint_never_poison_main_admission(
    tmp_path: Path, marker: dict[str, object], bad: dict[str, object]
) -> None:
    source, notes = decode_claude(
        tmp_path,
        [
            claude_user("child", "Internal child words.", **marker, **bad),
            claude_user("main", "Main human words."),
            ccall("done", "main", "M", txt("Main answer.")),
        ],
    )
    assert source is not None
    assert [item.item for item in source.messages if item.kind == "user"] == ["main"]
    assert not {"unknown_format", "mixed_entrypoint", "unsupported_entrypoint"} & {note.kind for note in notes}


def test_claude_self_produced_sidechain_cannot_rescue_copied_only_main_history(tmp_path: Path) -> None:
    from tests.app.memory_hosts.history_fixtures import PARENT

    source, notes = decode_claude(
        tmp_path,
        [
            claude_user("inherited", "Copied original.", sessionId=PARENT),
            claude_user("child", "Self-produced child.", "inherited", isSidechain=True),
        ],
    )
    assert source is None and any(note.kind == "copied_only" for note in notes)


async def test_claude_starting_cwd_and_source_group_exclude_sidechain_users(tmp_path: Path) -> None:
    import asyncio

    from chrys.app.memory_hosts.claude_code.history import ClaudeHistoryReader
    from chrys.service.memory.project import bind_project
    from tests.app.memory_hosts.history_fixtures import FixedHolder, write_jsonl

    missing = tmp_path / "deleted-child-directory"
    path = tmp_path / f"{SESSION}.jsonl"
    await asyncio.to_thread(
        write_jsonl,
        path,
        [
            claude_user("child", "Child setup.", isSidechain=True, cwd=str(missing)),
            claude_user("main", "Main request.", cwd=str(tmp_path)),
        ],
    )
    project = await asyncio.to_thread(bind_project, tmp_path)
    scan = await ClaudeHistoryReader(tmp_path, holder=FixedHolder()).scan(project)
    assert len(scan.sessions) == 1
    assert not any(note.kind == "source_mapping_unconfirmed" for note in scan.diagnostics)
    group = next(note for note in scan.diagnostics if note.kind == "source_group")
    assert group.details == (("cwd", str(tmp_path)), ("start_cwd", str(tmp_path)))
