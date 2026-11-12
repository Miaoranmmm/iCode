# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Claude interactive source admission follows transcript structure, never inferred prose."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chrys.app.memory_hosts.claude_code.conversation import build_conversation
from chrys.service.memory.storage.store import MemoryOperationError
from tests.app.memory_hosts.test_conversation import SESSION, JsonObject, assistant, boundary, transcript, user


def calls(uuid: str, parent: str, message_id: str | None, *blocks: JsonObject, **fields: object) -> JsonObject:
    message: JsonObject = {"role": "assistant", "content": list(blocks)}
    if message_id is not None:
        message["id"] = message_id
    return {"type": "assistant", "uuid": uuid, "parentUuid": parent, "sessionId": SESSION, "message": message, **fields}


def tool(call_id: str, name: str = "shell", arguments: object = None) -> JsonObject:
    return {"type": "tool_use", "id": call_id, "name": name, "input": {} if arguments is None else arguments}


def text(value: str) -> JsonObject:
    return {"type": "text", "text": value}


def result(uuid: str, parent: str, call_id: str, content: object, **fields: object) -> JsonObject:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": SESSION,
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": call_id, "content": content, **fields}],
        },
    }


def sources(tmp_path: Path, records: list[JsonObject]):
    return build_conversation(transcript(tmp_path, records), "call-current").messages


def test_parallel_calls_admitted_by_response_id_but_sibling_text_is_not(tmp_path: Path) -> None:
    records = [
        user("u", "Inspect both files."),
        calls("a", "u", "M", text("Selected proposal."), tool("a", "Read", {"path": "a.py"})),
        calls("b", "a", "M", text("Off-chain proposal."), tool("b", "Read", {"path": "b.py"})),
        result("ra", "a", "a", "First observed output."),
        result("rb", "b", "b", "Second observed output."),
        assistant("save", "ra"),
    ]
    view = sources(tmp_path, records)
    assert [(item.kind, item.item, item.part, item.fragments) for item in view] == [
        ("user", "u", 0, ("Inspect both files.",)),
        ("assistant", "a", 0, ("Selected proposal.",)),
        ("tool_result", "ra", 0, ("First observed output.",)),
        ("tool_result", "rb", 0, ("Second observed output.",)),
    ]
    assert view[-1].attributes == (("tool", "Read"), ("args", '{"path":"b.py"}'))


@pytest.mark.parametrize("branch_kind", ["rewind", "retry", "missing_message_id", "sidechain", "agent_id"])
def test_only_same_response_main_calls_can_admit_sibling_results(tmp_path: Path, branch_kind: str) -> None:
    fields = (
        {"isSidechain": True}
        if branch_kind == "sidechain"
        else {"agentId": "synthetic-agent"}
        if branch_kind == "agent_id"
        else {}
    )
    message_id = None if branch_kind == "missing_message_id" else "M" if fields else "different-response"
    branch_parent = "u" if branch_kind == "rewind" else "a"
    branch = calls("branch", branch_parent, message_id, text("Unselected text."), tool("branch-call"), **fields)
    records = [
        user("u", "Inspect."),
        calls("a", "u", "M", tool("selected")),
        branch,
        result("branch-result", "branch", "branch-call", "Unselected result."),
        result("selected-result", "a", "selected", "Selected result."),
        assistant("save", "selected-result"),
    ]
    view = sources(tmp_path, records)
    assert [(item.kind, item.item) for item in view] == [("user", "u"), ("tool_result", "selected-result")]


@pytest.mark.parametrize("flag", ["isSidechain", "agentId"])
def test_sidechain_records_are_excluded_for_every_source_kind(tmp_path: Path, flag: str) -> None:
    marker: object = True if flag == "isSidechain" else "synthetic-agent"
    side_result = result("child-result", "call", "shell-call", "Internal child result.")
    side_result[flag] = marker
    records = [
        user("u", "Main request."),
        user("child-user", "Internal child words.", "u", **{flag: marker}),
        calls("child-text", "child-user", "M", text("Internal child proposal."), **{flag: marker}),
        calls("call", "child-text", "main", tool("shell-call", "Task")),
        side_result,
        result("task-result", "call", "shell-call", "Main Task wrapper result."),
        assistant("save", "task-result"),
    ]
    view = sources(tmp_path, records)
    assert [(item.kind, item.item) for item in view] == [("user", "u"), ("tool_result", "task-result")]
    assert view[-1].attributes[0] == ("tool", "Task")


def test_results_pair_only_to_unanswered_own_ancestors_and_id_reuse_is_local(tmp_path: Path) -> None:
    records = [
        user("u", "Check."),
        calls("old", "u", "M-old", tool("reused", "OldTool", {"round": 1})),
        result("old-result", "old", "reused", "Old result."),
        result("duplicate-result", "old-result", "reused", "Already answered, not new evidence."),
        calls("new", "old-result", "M-new", tool("reused", "NewTool", {"round": 2})),
        result("not-descendant", "u", "reused", "An unrelated global id match."),
        result("new-result", "new", "reused", "New result."),
        assistant("save", "new-result"),
    ]
    view = sources(tmp_path, records)
    evidence = [item for item in view if item.kind == "tool_result"]
    assert [(item.item, item.attributes[0]) for item in evidence] == [
        ("old-result", ("tool", "OldTool")),
        ("new-result", ("tool", "NewTool")),
    ]


def test_orphan_and_unprovable_off_chain_results_do_not_poison_selected_view(tmp_path: Path) -> None:
    invalid = result("bad-parent", "call", "shell-call", "Cannot prove parent.")
    invalid["parentUuid"] = 42
    records = [
        user("u", "Main words."),
        calls("call", "u", "M", tool("shell-call")),
        result("orphan", "missing", "shell-call", "Missing parent."),
        invalid,
        result("good", "call", "shell-call", "Good result."),
        assistant("save", "good"),
    ]
    assert [(item.kind, item.item) for item in sources(tmp_path, records)] == [("user", "u"), ("tool_result", "good")]


def test_text_blocks_keep_parts_and_tool_result_list_fragments_without_images(tmp_path: Path) -> None:
    records = [
        user("u", "Keep fragments separate."),
        calls(
            "call",
            "u",
            "M",
            text("Proposal one."),
            {"type": "thinking", "thinking": "private reasoning"},
            text("Proposal two."),
            tool("shell-call", "shell", {"command": "line one\nline two"}),
        ),
        result(
            "r",
            "call",
            "shell-call",
            [text("First output."), {"type": "image", "source": {}}, text("Second output.")],
            is_error=True,
        ),
        assistant("save", "r"),
    ]
    view = sources(tmp_path, records)
    assert [(item.kind, item.part, item.fragments) for item in view] == [
        ("user", 0, ("Keep fragments separate.",)),
        ("assistant", 0, ("Proposal one.",)),
        ("assistant", 2, ("Proposal two.",)),
        ("tool_result", 0, ("First output.", "Second output.")),
    ]
    assert "\n" not in view[-1].attributes[1][1]
    assert json.loads(view[-1].attributes[1][1]) == {"command": "line one\nline two"}


def test_each_result_block_has_its_original_part(tmp_path: Path) -> None:
    combined = result("results", "call", "a", "First result.")
    combined["message"] = {
        "content": [
            {"type": "image", "source": {}},
            {"type": "tool_result", "tool_use_id": "a", "content": "First result."},
            {"type": "tool_result", "tool_use_id": "b", "content": "Second result."},
        ]
    }
    view = sources(
        tmp_path,
        [user("u", "Check."), calls("call", "u", "M", tool("a"), tool("b")), combined, assistant("save", "results")],
    )
    assert [(item.part, item.fragments) for item in view if item.kind == "tool_result"] == [
        (1, ("First result.",)),
        (2, ("Second result.",)),
    ]


@pytest.mark.parametrize("length", [190, 300])
def test_args_summary_includes_ellipsis_inside_200_character_limit(tmp_path: Path, length: int) -> None:
    value = {"command": "x" * length}
    view = sources(
        tmp_path,
        [
            user("u", "Check."),
            calls("call", "u", "M", tool("a", arguments=value)),
            result("r", "call", "a", "Output."),
            assistant("save", "r"),
        ],
    )
    summary = dict(view[-1].attributes)["args"]
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    assert summary == (raw if len(raw) <= 200 else raw[:199] + "…")
    assert len(summary) <= 200


@pytest.mark.parametrize(
    "name",
    [
        "memory_read",
        "memory_search",
        "memory_catalog",
        "memory_save",
        "memory_forget",
        "memory_withdraw",
        "memory_merge",
        "memory_note_conflict",
        "mcp__memory__memory_read",
        "prefix__memory_save",
    ],
)
def test_memory_tool_results_never_become_evidence(tmp_path: Path, name: str) -> None:
    view = sources(
        tmp_path,
        [
            user("u", "Check."),
            calls("call", "u", "M", tool("memory-call", name)),
            result("r", "call", "memory-call", "Old memory quote."),
            assistant("save", "r"),
        ],
    )
    assert [(item.kind, item.item) for item in view] == [("user", "u")]


def test_non_memory_tool_with_similar_name_keeps_its_result(tmp_path: Path) -> None:
    view = sources(
        tmp_path,
        [
            user("u", "Check."),
            calls("call", "u", "M", tool("a", "memory_read_file")),
            result("r", "call", "a", "Observed output."),
            assistant("save", "r"),
        ],
    )
    assert view[-1].kind == "tool_result" and view[-1].fragments == ("Observed output.",)


def test_anchor_record_and_every_later_source_are_excluded(tmp_path: Path) -> None:
    anchor = assistant("save", "call")
    anchor["message"] = {
        "id": "M",
        "content": [text("Current proposal is after the cutoff."), tool("call-current", "memory_save")],
    }
    records = [
        user("u", "Before."),
        calls("call", "u", "M", tool("a")),
        anchor,
        result("late-result", "call", "a", "After cutoff."),
        user("later", "Later words.", "save"),
    ]
    assert [(item.kind, item.item) for item in sources(tmp_path, records)] == [("user", "u")]


def test_replayed_sources_use_earliest_occurrences_before_deduplication(tmp_path: Path) -> None:
    original_user = user("u", "Original words.", "boundary")
    original_call = calls("call", "u", "M", text("Original proposal."), tool("a", "shell"))
    original_result = result("r", "call", "a", "Original output.")
    records = [
        boundary(),
        original_user,
        original_call,
        original_result,
        original_user.copy(),
        original_call.copy(),
        original_result.copy(),
        assistant("save", "r"),
    ]
    view = sources(tmp_path, records)
    assert [(item.kind, item.item, item.position) for item in view] == [
        ("user", "u", 2),
        ("assistant", "call", 3),
        ("tool_result", "r", 4),
    ]


def test_replayed_call_is_paired_physically_before_source_deduplication(tmp_path: Path) -> None:
    original = calls("call", "u", "M", tool("a", "shell"))
    records = [
        user("u", "Check."),
        original,
        result("first", "call", "a", "First result."),
        original.copy(),
        result("second", "call", "a", "Second result."),
        assistant("save", "second"),
    ]
    view = sources(tmp_path, records)
    assert [item.fragments for item in view if item.kind == "tool_result"] == [("First result.",), ("Second result.",)]


def test_compact_boundary_prevents_same_message_id_call_admission_from_old_segment(tmp_path: Path) -> None:
    records = [
        user("old-u", "Old."),
        calls("old-call", "old-u", "M", tool("old")),
        boundary(),
        user("u", "Current.", "boundary"),
        calls("current", "u", "M", tool("new")),
        result("old-r", "current", "old", "Old id is not a current ancestor."),
        result("new-r", "current", "new", "Current output."),
        assistant("save", "new-r"),
    ]
    assert [(item.kind, item.item) for item in sources(tmp_path, records)] == [("user", "u"), ("tool_result", "new-r")]


def test_selected_chain_keeps_strict_parent_validation(tmp_path: Path) -> None:
    invalid = calls("call", "u", "M", tool("a"))
    invalid["parentUuid"] = 42
    with pytest.raises(MemoryOperationError):
        sources(tmp_path, [user("u", "Check."), invalid, result("r", "call", "a", "Output."), assistant("save", "r")])


def test_anchor_response_id_can_prove_an_earlier_sibling_call(tmp_path: Path) -> None:
    anchor = assistant("save", "u")
    anchor["message"] = {"id": "M", "content": [tool("call-current", "memory_save")]}
    view = sources(
        tmp_path,
        [
            user("u", "Inspect."),
            calls("sibling", "u", "M", tool("a", "Read")),
            result("r", "sibling", "a", "Observed before the save call."),
            anchor,
        ],
    )
    assert [(item.kind, item.item) for item in view] == [("user", "u"), ("tool_result", "r")]


def test_selected_call_without_message_id_still_has_direct_chain_proof(tmp_path: Path) -> None:
    view = sources(
        tmp_path,
        [
            user("u", "Inspect."),
            calls("selected", "u", None, tool("a", "Read")),
            result("r", "selected", "a", "Directly proven result."),
            assistant("save", "r"),
        ],
    )
    assert [(item.kind, item.item) for item in view] == [("user", "u"), ("tool_result", "r")]


def test_retry_call_with_reused_id_cannot_answer_older_selected_call(tmp_path: Path) -> None:
    records = [
        user("u", "Inspect."),
        calls("selected", "u", "M", tool("reused", "SelectedTool")),
        calls("retry", "selected", "other-M", tool("reused", "RetriedTool")),
        result("retry-result", "retry", "reused", "Unselected retry output."),
        result("selected-result", "selected", "reused", "Selected output."),
        assistant("save", "selected-result"),
    ]
    evidence = [item for item in sources(tmp_path, records) if item.kind == "tool_result"]
    assert [(item.item, dict(item.attributes)["tool"]) for item in evidence] == [("selected-result", "SelectedTool")]


def test_answered_new_call_blocks_fallback_to_an_older_unanswered_reused_id(tmp_path: Path) -> None:
    records = [
        user("u", "Inspect."),
        calls("old", "u", "old-M", tool("reused", "OldTool")),
        calls("new", "old", "new-M", tool("reused", "NewTool")),
        result("new-result", "new", "reused", "New output."),
        result("duplicate-result", "new-result", "reused", "Duplicate new output."),
        assistant("save", "duplicate-result"),
    ]
    evidence = [item for item in sources(tmp_path, records) if item.kind == "tool_result"]
    assert [(item.item, dict(item.attributes)["tool"]) for item in evidence] == [("new-result", "NewTool")]


@pytest.mark.parametrize("excluded", ["missing_message_id", "sidechain", "agent_id"])
def test_ineligible_nearest_reused_declaration_cannot_redirect_a_result(tmp_path: Path, excluded: str) -> None:
    flags = (
        {"isSidechain": True}
        if excluded == "sidechain"
        else {"agentId": "synthetic-child"}
        if excluded == "agent_id"
        else {}
    )
    message_id = None if excluded == "missing_message_id" else "M"
    records = [
        user("u", "Inspect."),
        calls("selected", "u", "M", tool("reused", "SelectedTool")),
        calls("excluded", "selected", message_id, tool("reused", "ExcludedTool"), **flags),
        result("excluded-result", "excluded", "reused", "Unproven output."),
        result("selected-result", "selected", "reused", "Selected output."),
        assistant("save", "selected-result"),
    ]
    evidence = [item for item in sources(tmp_path, records) if item.kind == "tool_result"]
    assert [(item.item, dict(item.attributes)["tool"]) for item in evidence] == [("selected-result", "SelectedTool")]
