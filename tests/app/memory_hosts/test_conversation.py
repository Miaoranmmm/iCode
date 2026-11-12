# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Transcript contracts from the memory trial specification, using invented text only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from chrys.app.memory_hosts.claude_code import conversation
from chrys.app.memory_hosts.claude_code.conversation import build_conversation, user_fragments, validate_anchor
from chrys.app.memory_hosts.claude_code.registry import Registration
from chrys.service.memory.storage.store import MemoryOperationError, Quote
from chrys.service.memory.verification import QuoteClaim, SourceVerificationError, verify_quotes
from tests.support.secure_files import plant_owner_only_bytes

if TYPE_CHECKING:
    from collections.abc import Sequence

JsonObject = dict[str, object]
SESSION = "22222222-2222-4222-8222-222222222222"
PARENT_SESSION = "11111111-1111-4111-8111-111111111111"


def user(uuid: str, text: object, parent: str | None = None, **fields: object) -> JsonObject:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": SESSION,
        "origin": {"kind": "human"},
        "message": {"role": "user", "content": text},
        **fields,
    }


def assistant(uuid: str, parent: str | None, call: str = "call-current", **fields: object) -> JsonObject:
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": SESSION,
        "message": {
            "role": "assistant",
            "id": "message-shared",
            "content": [
                {"type": "tool_use", "id": call, "name": "memory_save", "input": {"title": "Invented fixture"}},
            ],
        },
        **fields,
    }


def boundary(uuid: str = "boundary", preserved: object = None) -> JsonObject:
    return {
        "type": "system",
        "subtype": "compact_boundary",
        "uuid": uuid,
        "parentUuid": None,
        "compactMetadata": {
            "preservedMessages": {
                "uuids": ["short-list"],
                "allUuids": preserved if preserved is not None else ["old-call"],
            }
        },
    }


def transcript(tmp_path: Path, records: Sequence[JsonObject], suffix: bytes = b"") -> Registration:
    path = tmp_path / "transcript.jsonl"
    payload = b"".join((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8") for record in records)
    plant_owner_only_bytes(path, payload + suffix)
    return Registration(SESSION, str(path), str(tmp_path))


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("Keep spaces  \n和 Unicode。", ("Keep spaces  \n和 Unicode。",)),
        (
            [{"type": "text", "text": "first"}, {"type": "image", "source": {}}, {"type": "text", "text": "second"}],
            ("first", "second"),
        ),
        ("<!-- reply 9 -->\n> Earlier suggestion\n> second line\n\nMy response", ("\nMy response",)),
        ("<!-- reply -->\n> Earlier suggestion\nResponse", ("Response",)),
        ("> Plain quotation\nReply", ("Reply",)),
        (
            'before<pasted_content id="a">pasted\n  text</pasted_content id="a">after',
            ("before", "pasted\n  text", "after"),
        ),
        (
            '<pasted_content id="a">one</pasted_content id="a"><pasted_content id="a">two</pasted_content id="a">',
            ("one", "two"),
        ),
        (
            (
                '<pasted_content id="a"><bash-stdout>literal</bash-stdout>\n> literal reply\n'
                '<system-reminder>literal</system-reminder><unknown/></pasted_content id="a">'
            ),
            (
                (
                    "<bash-stdout>literal</bash-stdout>\n> literal reply\n"
                    "<system-reminder>literal</system-reminder><unknown/>"
                ),
            ),
        ),
        ('> <pasted_content id="a">model text</pasted_content id="a">\nHuman response', ("Human response",)),
        (
            (
                'left<artifact-view-context>{"other":"words"}<artifact-content-authored-by-others/>'
                "</artifact-view-context>right"
            ),
            ("left", "right"),
        ),
        ("<system-reminder>host</system-reminder>Real words", ("Real words",)),
        ("left<artifact-content-authored-by-others/>right", ("left", "right")),
        ("Words discussing <pid> and <module> placeholders", ("Words discussing <pid> and <module> placeholders",)),
        ('<pasted_content id="a">unclosed', ()),
        ('</pasted_content id="a">orphan', ()),
        ('<pasted_content id="a">mismatch</pasted_content id="b">', ()),
        ("left<system-reminder>unclosed", ()),
        ("left</system-reminder>orphan", ()),
        ("left<artifact-view-context>unclosed", ()),
        ("left<artifact-content-authored-by-others>untrusted</artifact-content-authored-by-others>right", ()),
        ("Words <unknown><artifact-content-authored-by-others/></unknown> after", ()),
        (" \n<unknown>possibly user XML</unknown>after", ()),
        (" \n<unknown/>after", ()),
        ("<system-reminder>host</system-reminder><unknown>after</unknown>", ()),
    ],
)
def test_fragments_follow_outer_source_and_preserve_boundaries(content: object, expected: tuple[str, ...]) -> None:
    assert user_fragments(user("u", content)) == expected


@pytest.mark.parametrize(
    "tag",
    [
        "bash-input",
        "bash-stdout",
        "bash-stderr",
        "command-name",
        "command-message",
        "command-args",
        "local-command-caveat",
        "local-command-stdout",
        "task-notification",
    ],
)
def test_known_command_and_bash_families_exclude_whole_human_message(tag: str) -> None:
    assert user_fragments(user("u", f"Human prefix<{tag}>injected</{tag}>Human suffix")) == ()


@pytest.mark.parametrize(
    "fields",
    [
        {"isMeta": True},
        {"isCompactSummary": True},
        {"isSidechain": True},
        {"agentId": "child"},
        {"origin": {"kind": "sdk"}},
        {"origin": None, "promptSource": "typed"},
        {"origin": None, "promptSource": "sdk"},
    ],
)
def test_human_filter_exclusions_precede_positive_provenance(fields: JsonObject) -> None:
    assert user_fragments(user("u", "Words", **fields)) == ()


def test_only_two_human_markers_are_positive_and_wrappers_still_lose() -> None:
    assert user_fragments(user("u", "Synthetic turn marker", origin=None, turnOrigin="human")) == (
        "Synthetic turn marker",
    )
    assert user_fragments(user("u", "Typed text", promptSource="typed", turnOrigin="human")) == ("Typed text",)
    assert (
        user_fragments(user("u", [{"type": "text", "text": "prefix"}, {"type": "tool_result", "content": "x"}])) == ()
    )
    assert user_fragments({"type": "attachment", "origin": {"kind": "human"}, "content": "not body"}) == ()


@pytest.mark.parametrize(
    ("text", "fields"),
    [
        ("[Request interrupted by user]", {"origin": None}),
        ("This session is being continued from a previous conversation.", {"isCompactSummary": True}),
        (
            "This session is being continued from a previous conversation.",
            {"isCompactSummary": True, "turnOrigin": "human"},
        ),
        ("Continue from where you left off.", {"isMeta": True}),
    ],
)
def test_automatic_continuations_have_no_user_fragments(text: str, fields: JsonObject) -> None:
    assert user_fragments(user("u", text, **fields)) == ()


def test_view_is_active_chain_not_physical_order_and_preserves_inherited_sources(tmp_path: Path) -> None:
    records = [
        user("old", "Use clear names", sessionId=PARENT_SESSION, timestamp="2026-11-06T12:00:00+08:00"),
        {"type": "system", "subtype": "stop_hook_summary", "uuid": "summary", "parentUuid": "old"},
        user("inactive", "Discard this branch", "summary"),
        user("new", "Use clear names", "summary", timestamp="invalid"),
        assistant("a1", "new", "parallel-other"),
        assistant("a2", "a1"),
        {
            "type": "user",
            "uuid": "r1",
            "parentUuid": "a1",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "parallel-other", "content": "Not human"},
                ]
            },
        },
    ]
    registration = transcript(tmp_path, records)
    view = build_conversation(registration, "call-current")
    assert view.current_session == SESSION
    assert [(message.item, message.position, message.at, message.session) for message in view.messages] == [
        ("old", 1, "2026-11-06T04:00:00Z", SESSION),
        ("new", 4, None, SESSION),
    ]
    claim = QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", "Use clear names")
    with pytest.raises(SourceVerificationError) as failure:
        verify_quotes([claim], view)
    candidates = failure.value.failures[0].candidates
    plant_owner_only_bytes(Path(registration.transcript_path), (json.dumps({"type": "progress"}) + "\n").encode())
    with pytest.raises(SourceVerificationError) as appended:
        verify_quotes([claim], build_conversation(registration, "call-current"))
    assert appended.value.failures[0].candidates == candidates


def test_parallel_result_interleaving_and_meta_nodes_are_walked(tmp_path: Path) -> None:
    registration = transcript(
        tmp_path,
        [
            user("u", "Keep this instruction"),
            assistant("a1", "u", "other"),
            {
                "type": "user",
                "uuid": "result",
                "parentUuid": "a1",
                "message": {
                    "content": [
                        {"type": "tool_result", "content": "Tool output"},
                    ]
                },
            },
            {"type": "attachment", "uuid": "attachment", "parentUuid": "result"},
            assistant("a2", "attachment"),
        ],
    )
    assert [message.item for message in build_conversation(registration, "call-current").messages] == ["u"]


@pytest.mark.parametrize("parent", [None, "missing"])
def test_null_root_is_valid_but_non_null_missing_parent_is_not(tmp_path: Path, parent: str | None) -> None:
    registration = transcript(tmp_path, [user("u", "Original", parent), assistant("a", "u")])
    if parent is None:
        assert len(build_conversation(registration, "call-current").messages) == 1
    else:
        with pytest.raises(MemoryOperationError, match="memory source is unavailable"):
            build_conversation(registration, "call-current")


def test_missing_parent_field_is_a_root(tmp_path: Path) -> None:
    original = user("u", "Original")
    del original["parentUuid"]
    assert build_conversation(transcript(tmp_path, [original, assistant("a", "u")]), "call-current").messages


@pytest.mark.parametrize(
    "records",
    [
        [user("u", "Original"), assistant("a", "u", sessionId=PARENT_SESSION)],
        [user("u", "Original"), assistant("a", "u", forkedFrom={"sessionId": PARENT_SESSION, "messageUuid": "a"})],
        [user("u", "Original"), assistant("a", "u", isSidechain=True)],
        [user("u", "Original"), assistant("a", "u", agentId="child")],
        [user("u", "Original"), assistant("a", "u"), assistant("other", "u")],
        [user("u", "Original"), assistant("a", "u"), assistant("a", "u", sessionId=PARENT_SESSION)],
        [user("u", "Original", "a"), assistant("a", "u")],
        [user("u", "Original"), assistant("a", "missing")],
    ],
)
def test_bad_sources_fail_closed(tmp_path: Path, records: list[JsonObject]) -> None:
    with pytest.raises(MemoryOperationError) as error:
        build_conversation(transcript(tmp_path, records), "call-current")
    assert error.value.kind == "memory_store_unavailable"
    assert str(tmp_path) not in str(error.value)


def test_equivalent_uuid_replay_is_one_logical_node(tmp_path: Path) -> None:
    current = assistant("a", "u")
    registration = transcript(tmp_path, [user("u", "Original"), current, current.copy()])
    assert len(build_conversation(registration, "call-current").messages) == 1
    validate_anchor(registration, "call-current")


@pytest.mark.parametrize("tool_id", ["", "not-landed", "old-call"])
def test_no_id_or_wrong_file_never_guesses_latest_call(tmp_path: Path, tool_id: str) -> None:
    registration = transcript(tmp_path, [user("u", "Original"), assistant("a", "u")])
    with pytest.raises(MemoryOperationError):
        validate_anchor(registration, tool_id)


@pytest.mark.parametrize("preserved", [[], [1], "old-call", None])
def test_unknown_compact_preservation_shape_rejects_calls(tmp_path: Path, preserved: object) -> None:
    compact = boundary(preserved=preserved)
    if preserved is None:
        compact["compactMetadata"] = {"preservedMessages": {"uuids": ["old-call"]}}
    registration = transcript(tmp_path, [compact, user("u", "New words", "boundary"), assistant("a", "u")])
    with pytest.raises(MemoryOperationError):
        build_conversation(registration, "call-current")


def test_compact_replay_anchor_rejected_even_without_original_row_or_fork_marker(tmp_path: Path) -> None:
    registration = transcript(
        tmp_path, [boundary(preserved=["a"]), user("u", "Replayed user", "boundary"), assistant("a", "u")]
    )
    with pytest.raises(MemoryOperationError):
        validate_anchor(registration, "call-current")


def test_compact_accepts_replayed_users_but_stops_before_old_text(tmp_path: Path) -> None:
    registration = transcript(
        tmp_path,
        [
            user("before", "Old words"),
            assistant("old-call", "before", "old"),
            boundary(),
            user("replayed", "Replayed words", "boundary", sessionId=PARENT_SESSION),
            user("summary", "Summary words", "replayed", isCompactSummary=True, turnOrigin="human"),
            assistant("new", "summary"),
        ],
    )
    view = build_conversation(registration, "call-current")
    assert [message.item for message in view.messages] == ["replayed"]
    with pytest.raises(SourceVerificationError) as error:
        verify_quotes([QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", "Old words")], view)
    assert error.value.kind == "memory_quote_not_found"
    quote = Quote("Old words", "message", session=SESSION, item="before")
    assert verify_quotes(
        [QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", "Old words", session=SESSION)], view, stored=[quote]
    ) == (quote,)


def test_cross_paste_and_text_block_quotes_cannot_be_joined(tmp_path: Path) -> None:
    registration = transcript(
        tmp_path,
        [
            user("u", 'before<pasted_content id="x">pasted</pasted_content id="x">after'),
            assistant("a", "u"),
        ],
    )
    view = build_conversation(registration, "call-current")
    assert len(verify_quotes([QuoteClaim("q", "c", "pasted")], view)) == 1
    for text in ("beforepasted", "pastedafter"):
        with pytest.raises(SourceVerificationError):
            verify_quotes([QuoteClaim("q", "c", text)], view)


def test_current_bad_line_rejects_but_old_bad_segment_and_final_half_line_do_not(tmp_path: Path) -> None:
    registration = transcript(
        tmp_path,
        [],
        b"broken\n"
        + (json.dumps(boundary()) + "\n").encode()
        + (json.dumps(user("u", "Original", "boundary")) + "\n").encode()
        + (json.dumps(assistant("a", "u")) + "\n").encode()
        + b'{"half":',
    )
    assert build_conversation(registration, "call-current").messages
    plant_owner_only_bytes(Path(registration.transcript_path), b"\nbroken\n")
    with pytest.raises(MemoryOperationError):
        validate_anchor(registration, "call-current")


def test_write_limit_is_current_segment_and_read_anchor_remains_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(conversation, "MAX_SEGMENT_BYTES", 700)
    registration = transcript(tmp_path, [user("u", "x" * 800), assistant("a", "u")])
    with pytest.raises(MemoryOperationError, match="Current transcript segment exceeds the 128 MiB"):
        build_conversation(registration, "call-current")
    validate_anchor(registration, "call-current")


def test_large_old_segment_is_discarded_on_compact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conversation, "MAX_SEGMENT_BYTES", 800)
    registration = transcript(
        tmp_path, [user("old", "x" * 900), boundary(), user("u", "New", "boundary"), assistant("a", "u")]
    )
    assert [message.item for message in build_conversation(registration, "call-current").messages] == ["u"]


def test_single_line_limit_is_enforced_only_in_current_segment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conversation, "MAX_LINE_BYTES", 500)
    registration = transcript(
        tmp_path, [user("old", "x" * 900), boundary(), user("u", "New", "boundary"), assistant("a", "u")]
    )
    assert build_conversation(registration, "call-current").messages


@dataclass
class FixedClock:
    values: tuple[float, ...] = (0.0,)
    position: int = 0

    def monotonic(self) -> float:
        value = self.values[min(self.position, len(self.values) - 1)]
        self.position += 1
        return value


def answer(uuid: str, parent: str, text: str = "Done") -> JsonObject:
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "message": {
            "content": [
                {"type": "text", "text": text},
            ]
        },
    }


def attachment(uuid: str, parent: str | None, content: object) -> JsonObject:
    return {
        "type": "attachment",
        "uuid": uuid,
        "parentUuid": parent,
        "attachment": {
            "type": "hook_additional_context",
            "hookName": "SessionStart",
            "hookEvent": "SessionStart",
            "content": content,
        },
    }


def visibility(
    tmp_path: Path, records: Sequence[JsonObject], *, target: str = "Catalog", suffix: bytes = b""
) -> bool | None:
    registration = transcript(tmp_path, records, suffix)
    return conversation.catalog_visibility(
        registration.transcript_path, "prompt-current", "Done", target, clock=FixedClock()
    )


def stopped_turn(parent: str | None = None) -> list[JsonObject]:
    return [user("u", "Please work", parent, promptId="prompt-current"), answer("a", "u")]


@pytest.mark.parametrize("extra_newline", ["", "\n"])
def test_stop_exact_hook_catalog_or_one_terminal_newline_is_visible(tmp_path: Path, extra_newline: str) -> None:
    assert (
        visibility(tmp_path, [attachment("catalog", None, ["Catalog" + extra_newline]), *stopped_turn("catalog")])
        is True
    )


@pytest.mark.parametrize("content", [["Cat", "alog"], ["Catalog\n\n"], [" Catalog"], ["Catalog "]])
def test_stop_never_joins_arrays_or_trims_catalog_text(tmp_path: Path, content: object) -> None:
    assert visibility(tmp_path, [attachment("catalog", None, content), *stopped_turn("catalog")]) is False


def test_stop_same_text_older_copy_is_visible(tmp_path: Path) -> None:
    records = [attachment("copy", None, ["Catalog"]), *stopped_turn("copy")]
    assert visibility(tmp_path, records) is True


def test_stop_missing_catalog_is_known_only_after_ready_turn(tmp_path: Path) -> None:
    assert visibility(tmp_path, stopped_turn()) is False


def test_stop_only_hook_attachment_counts_not_tool_result_catalog(tmp_path: Path) -> None:
    records = [
        *stopped_turn(),
        {
            "type": "user",
            "uuid": "result",
            "parentUuid": "a",
            "message": {
                "content": [
                    {"type": "tool_result", "content": "Catalog"},
                ]
            },
        },
    ]
    assert visibility(tmp_path, records) is False


def test_stop_uses_rewind_new_tail_and_skips_only_non_main_meta(tmp_path: Path) -> None:
    records = [
        attachment("catalog", None, ["Catalog"]),
        user("old", "Old", "catalog", promptId="old-prompt"),
        answer("old-a", "old"),
        *stopped_turn(),
        {"type": "system", "subtype": "stop_hook_summary", "uuid": "summary", "parentUuid": "a"},
        {"type": "system", "subtype": "turn_duration", "uuid": "duration", "parentUuid": "summary"},
        {"type": "progress", "uuid": "progress", "parentUuid": "old-a"},
        user("child", "Child text", "old-a", isSidechain=True),
        {"type": "system", "subtype": "stop_hook_summary"},
    ]
    assert visibility(tmp_path, records) is False


def test_stop_parallel_results_follow_last_uuid_chain(tmp_path: Path) -> None:
    records = [
        attachment("catalog", None, ["Catalog"]),
        user("u", "Please work", "catalog", promptId="prompt-current"),
        assistant("parallel-a", "u", "first"),
        assistant("parallel-b", "parallel-a", "second"),
        {
            "type": "user",
            "uuid": "r1",
            "parentUuid": "parallel-a",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "first", "content": "First"},
                ]
            },
        },
        {
            "type": "user",
            "uuid": "r2",
            "parentUuid": "parallel-b",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "second", "content": "Second"},
                ]
            },
        },
        answer("a", "r2"),
    ]
    assert visibility(tmp_path, records) is True


@pytest.mark.parametrize(
    "records",
    [
        [user("u", "Prompt", promptId="prompt-current")],
        [answer("a", "missing")],
        [user("u", "Prompt", promptId="other"), answer("a", "u")],
        [user("u", "Prompt"), answer("a", "u")],
        [user("u", "Prompt", promptId="prompt-current"), answer("a", "u", "Not Done")],
        [
            user("old", "Old", promptId="old"),
            answer("old-a", "old"),
            user("u", "New", "old-a", promptId="prompt-current"),
        ],
        [user("u", "Prompt", "a", promptId="prompt-current"), answer("a", "u")],
        [user("u", "Prompt", promptId="prompt-current"), assistant("a", "u")],
        [user("u", "Continue from where you left off.", promptId="prompt-current", isMeta=True), answer("a", "u")],
    ],
)
def test_stop_incomplete_wrong_turn_or_bad_chain_is_unknown(tmp_path: Path, records: list[JsonObject]) -> None:
    assert visibility(tmp_path, records) is None


@pytest.mark.parametrize("content", ["Catalog", ["Catalog", 3], None])
def test_stop_unknown_hook_attachment_shape_is_unknown(tmp_path: Path, content: object) -> None:
    assert visibility(tmp_path, [attachment("catalog", None, content), *stopped_turn("catalog")]) is None


def test_stop_assistant_text_blocks_join_with_newlines_and_trim(tmp_path: Path) -> None:
    last = answer("a", "u")
    last["message"] = {
        "content": [
            {"type": "text", "text": "  First"},
            {"type": "thinking", "thinking": "hidden"},
            {"type": "text", "text": "Second  "},
        ]
    }
    registration = transcript(tmp_path, [*stopped_turn()[:1], last])
    assert (
        conversation.catalog_visibility(
            registration.transcript_path, "prompt-current", "First\nSecond", "Catalog", clock=FixedClock()
        )
        is False
    )


def test_stop_compact_boundary_is_a_complete_root(tmp_path: Path) -> None:
    assert (
        visibility(tmp_path, [attachment("old-catalog", None, ["Catalog"]), boundary(), *stopped_turn("boundary")])
        is False
    )


def test_stop_bad_old_segment_is_discarded_at_boundary(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    payload = b"broken\n" + b"".join(
        (json.dumps(row) + "\n").encode() for row in [boundary(), *stopped_turn("boundary")]
    )
    plant_owner_only_bytes(path, payload)
    assert conversation.catalog_visibility(str(path), "prompt-current", "Done", "Catalog", clock=FixedClock()) is False


@pytest.mark.parametrize("suffix", [b"broken\n", b'{"half":'])
def test_stop_bad_line_and_tail_half_line_leave_visibility_unknown(tmp_path: Path, suffix: bytes) -> None:
    assert visibility(tmp_path, stopped_turn(), suffix=suffix) is None


@pytest.mark.parametrize(
    ("prompt", "last"), [(None, "Done"), ("prompt-current", None), ("", "Done"), ("prompt-current", "")]
)
def test_stop_missing_readiness_fields_are_unknown(tmp_path: Path, prompt: str | None, last: str | None) -> None:
    registration = transcript(tmp_path, stopped_turn())
    assert conversation.catalog_visibility(registration.transcript_path, prompt, last, "Catalog") is None


def test_stop_empty_transcript_path_is_unknown() -> None:
    assert conversation.catalog_visibility("", "prompt", "Done", "Catalog") is None


def test_stop_parsing_budget_is_injected_and_never_uses_sleep(tmp_path: Path) -> None:
    registration = transcript(tmp_path, stopped_turn())
    clock = FixedClock((0.0, 0.101))
    assert (
        conversation.catalog_visibility(registration.transcript_path, "prompt-current", "Done", "Catalog", clock=clock)
        is None
    )


def test_stop_record_budget_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conversation, "STOP_MAX_RECORDS", 2)
    assert visibility(tmp_path, [*stopped_turn(), {"type": "progress"}]) is None


def test_stop_tail_window_does_not_expand_to_find_missing_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [attachment("catalog", None, ["Catalog"]), *stopped_turn("catalog")]
    payload_size = len(b"".join((json.dumps(row) + "\n").encode() for row in records))
    monkeypatch.setattr(conversation, "STOP_WINDOW_BYTES", payload_size - 10)
    assert visibility(tmp_path, records) is None


def test_sanitized_human_fixture_keeps_only_submitted_fragments(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "human_fragments.jsonl"
    path = tmp_path / "fixture.jsonl"
    plant_owner_only_bytes(path, fixture.read_text(encoding="utf-8").encode("utf-8"))
    view = build_conversation(Registration("fixture-session", str(path), str(tmp_path)), "fixture-call")
    assert [(item.item, item.fragments) for item in view.messages] == [
        ("paste", ("Before", "Pasted requirement", "After")),
        ("reply", ("Use shorter names.",)),
        ("artifact", ("Agree before", "Agree after")),
        ("typed", ("Always explain the reason.",)),
    ]


def test_sanitized_inherited_parallel_fixture_uses_full_child_session(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "inherited_parallel.jsonl"
    path = tmp_path / "fixture.jsonl"
    plant_owner_only_bytes(path, fixture.read_text(encoding="utf-8").encode("utf-8"))
    registration = Registration("child-session", str(path), str(tmp_path))
    view = build_conversation(registration, "second-call")
    assert [(item.item, item.session, item.position) for item in view.messages] == [
        ("parent-words", "child-session", 1),
        ("child-words", "child-session", 4),
    ]
    with pytest.raises(MemoryOperationError):
        validate_anchor(registration, "copied-call")


@pytest.mark.parametrize("text", ["<!-- reply -->", "<!-- reply 2 -->\n> quoted only\n", "> quoted only\n"])
def test_orphan_reply_and_quote_markers_never_become_words(text: str) -> None:
    assert user_fragments(user("u", text)) == ()


def test_nested_host_blocks_are_discarded_as_one_outer_source() -> None:
    text = "Before<system-reminder>host<system-reminder>nested</system-reminder>more host</system-reminder>After"
    assert user_fragments(user("u", text)) == ("Before", "After")


def test_paste_inside_host_block_cannot_turn_host_words_into_quotes() -> None:
    text = '<system-reminder><pasted_content id="x">Foreign</pasted_content id="x"></system-reminder>Mine'
    assert user_fragments(user("u", text)) == ("Mine",)


def deep_json_record() -> bytes:
    # This is valid JSON well below either byte limit, but exceeds CPython's decoding depth.
    return b'{"type":"progress","uuid":"deep","data":' + b"[" * 100_000 + b"0" + b"]" * 100_000 + b"}\n"


@pytest.mark.parametrize("reading", [False, True])
def test_deep_current_json_is_source_unavailable(tmp_path: Path, reading: bool) -> None:
    registration = transcript(tmp_path, [user("u", "Original"), assistant("a", "u")], deep_json_record())
    operation = validate_anchor if reading else build_conversation
    with pytest.raises(MemoryOperationError) as error:
        operation(registration, "call-current")
    assert error.value.kind == "memory_store_unavailable"
    assert error.value.message == conversation.SOURCE_UNAVAILABLE


def test_deep_prior_json_is_discarded_at_compact_boundary(tmp_path: Path) -> None:
    records = [boundary(), user("u", "New words", "boundary"), assistant("a", "u")]
    payload = deep_json_record() + b"".join((json.dumps(row) + "\n").encode("utf-8") for row in records)
    registration = transcript(tmp_path, [], payload)
    assert [message.item for message in build_conversation(registration, "call-current").messages] == ["u"]


def test_deep_json_makes_stop_unknown_without_throwing(tmp_path: Path) -> None:
    assert visibility(tmp_path, stopped_turn(), suffix=deep_json_record()) is None


def test_deep_prior_json_does_not_poison_stop_current_segment(tmp_path: Path) -> None:
    payload = deep_json_record() + b"".join(
        (json.dumps(row) + "\n").encode("utf-8") for row in [boundary(), *stopped_turn("boundary")]
    )
    registration = transcript(tmp_path, [], payload)
    assert (
        conversation.catalog_visibility(
            registration.transcript_path, "prompt-current", "Done", "Catalog", clock=FixedClock()
        )
        is False
    )


def test_fingerprint_depth_failure_is_a_source_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registration = transcript(tmp_path, [user("u", "Original"), assistant("a", "u")])

    class FingerprintDepthFailure:
        @staticmethod
        def loads(data: bytes) -> object:
            return json.loads(data)

        @staticmethod
        def dumps(value: object, *, sort_keys: bool, ensure_ascii: bool, separators: tuple[str, str]) -> str:
            raise RecursionError

    monkeypatch.setattr(conversation, "json", FingerprintDepthFailure)
    with pytest.raises(MemoryOperationError) as error:
        build_conversation(registration, "call-current")
    assert error.value.kind == "memory_store_unavailable"
    assert (
        conversation.catalog_visibility(
            registration.transcript_path, "prompt-current", "Done", "Catalog", clock=FixedClock()
        )
        is None
    )


@pytest.mark.parametrize("on_readiness", [True, False])
def test_stop_budget_expiring_after_parsing_returns_unknown(tmp_path: Path, on_readiness: bool) -> None:
    registration = transcript(tmp_path, stopped_turn())
    # Start + two parse checks + post-parse check all finish within budget.
    # Expire either while traversing readiness or after both complete traversals.
    clock = FixedClock((0.0,) * (4 if on_readiness else 8) + (0.101,))
    assert (
        conversation.catalog_visibility(registration.transcript_path, "prompt-current", "Done", "Catalog", clock=clock)
        is None
    )
    assert clock.position == (5 if on_readiness else 9)


def test_stop_final_budget_check_also_vetoes_visible_catalog(tmp_path: Path) -> None:
    registration = transcript(tmp_path, [attachment("catalog", None, ["Catalog"]), *stopped_turn("catalog")])
    clock = FixedClock((0.0,) * 10 + (0.101,))
    assert (
        conversation.catalog_visibility(registration.transcript_path, "prompt-current", "Done", "Catalog", clock=clock)
        is None
    )
    assert clock.position == 11


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            '先看 <pasted_content id="a">日志</pasted_content id="a">\n<b>重要</b> 以后都这样',
            ("先看 ", "日志", "\n<b>重要</b> 以后都这样"),
        ),
        (
            "Before<system-reminder>host context</system-reminder>\n<b>Important</b> after",
            ("Before", "\n<b>Important</b> after"),
        ),
        (
            (
                "Before<artifact-view-context>foreign content<artifact-content-authored-by-others/>"
                "</artifact-view-context>\n<b>Important</b> after"
            ),
            ("Before", "\n<b>Important</b> after"),
        ),
        (
            "Before<artifact-content-authored-by-others/>\n<b>Important</b> after",
            ("Before", "\n<b>Important</b> after"),
        ),
        (
            "Before\n> Quoted proposal\n<b>Important</b> after",
            ("Before\n", "<b>Important</b> after"),
        ),
        (
            "Before<!-- reply 2 -->\n> Quoted proposal\n<b>Important</b> after",
            ("Before", "<b>Important</b> after"),
        ),
    ],
)
def test_unknown_leading_tag_is_checked_once_per_outside_text_block(text: str, expected: tuple[str, ...]) -> None:
    assert user_fragments(user("u", text)) == expected


@pytest.mark.parametrize(
    "text",
    [
        "<foo>bar</foo> 以后",
        '<pasted_content id="a">Typed in paste</pasted_content id="a">\n<b>Important</b> after',
        "<system-reminder>host context</system-reminder>\n<b>Important</b> after",
        (
            "<artifact-view-context>foreign<artifact-content-authored-by-others/>"
            "</artifact-view-context>\n<b>Important</b> after"
        ),
        "<artifact-content-authored-by-others/>\n<b>Important</b> after",
        "<!-- reply 2 -->\n> Quoted proposal\n<b>Important</b> after",
        "> Quoted proposal\n<b>Important</b> after",
    ],
)
def test_removed_known_structures_and_paste_body_do_not_supply_outside_prefix(text: str) -> None:
    assert user_fragments(user("u", text)) == ()


def test_outside_prefix_validation_does_not_join_quotes_across_structures(tmp_path: Path) -> None:
    text = 'Before<pasted_content id="a">Pasted</pasted_content id="a"><b>after</b>'
    registration = transcript(tmp_path, [user("u", text), assistant("a", "u")])
    view = build_conversation(registration, "call-current")
    assert view.messages[0].fragments == ("Before", "Pasted", "<b>after</b>")
    for claim in ("BeforePasted", "Pasted<b>after", "Before<b>after"):
        with pytest.raises(SourceVerificationError):
            verify_quotes([QuoteClaim("q", "c", claim)], view)


def test_outside_prefix_validation_does_not_cross_text_blocks() -> None:
    content = [{"type": "text", "text": "Earlier block"}, {"type": "text", "text": "<b>Important</b> later block"}]
    assert user_fragments(user("u", content)) == ()


def test_bash_policy_switch_keeps_only_actual_input_and_never_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conversation, "EXCLUDE_BASH_MESSAGES", False)
    assert user_fragments(
        user(
            "u",
            "<bash-input>!echo synthetic</bash-input><bash-stdout>printed text</bash-stdout><bash-stderr>diagnostic</bash-stderr>",
        )
    ) == ("!echo synthetic",)
    assert user_fragments(
        user("u", '<pasted_content id="p"><bash-stdout>literal user paste</bash-stdout></pasted_content id="p">')
    ) == ("<bash-stdout>literal user paste</bash-stdout>",)
