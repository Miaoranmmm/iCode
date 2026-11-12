# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Step-7 material profiles, source boundaries, and structural selection policies."""

from __future__ import annotations

from dataclasses import replace

import pytest

from chrys.service.memory.extraction import (
    DEFAULT_MATERIAL_PROFILE,
    EXTRACTION_PROMPT,
    FULL_V3,
    FULL_V4,
    INSTRUCTIONS,
    USER_V2,
    ExtractionBatch,
    ExtractionError,
    OversizeSelection,
    SessionKey,
    SourceAnnotation,
    SourceRevision,
    SourceSession,
    build_batches,
    material_profile_for_versions,
    ordered_sources,
    project_source,
    render_batch,
    render_prompt,
    select_members,
    signal_neighborhood,
    source_identity,
    tuned_material_profile,
)
from chrys.service.memory.storage.store import MemoryScope
from chrys.service.memory.tuning import V0
from chrys.service.memory.verification import (
    QuoteClaim,
    SourceMessage,
    SourceVerificationError,
    ToolResultClaim,
    verify_context,
    verify_evidence,
)

PROJECT = MemoryScope("project", "0123456789abcdef")
SESSION = "00000000-0000-4000-8000-000000000001"


def user(position: int, text: str = "A durable preference.") -> SourceMessage:
    return SourceMessage(SESSION, position, (text,), item=f"user-{position}", at="2026-11-11T00:00:00Z")


def assistant(position: int, text: str = "A supported engineering conclusion.") -> SourceMessage:
    return SourceMessage(SESSION, position, (text,), kind="assistant", item=f"assistant-{position}")


def tool(position: int, text: str = "A tool observation.", *, error: bool = False, part: int = 0) -> SourceMessage:
    return SourceMessage(
        SESSION,
        position,
        (text,),
        kind="tool_result",
        part=part,
        item=f"tool-{position}",
        attributes=(("tool", "shell"), ("args", '{"command":"synthetic"}'), ("is_error", "true" if error else "false")),
    )


def source(messages: tuple[SourceMessage, ...], annotations: tuple[SourceAnnotation, ...] = ()) -> SourceSession:
    return SourceSession(
        SessionKey("claude-code", SESSION), PROJECT, SourceRevision(1, "a" * 64), messages, annotations
    )


def shown(batch: ExtractionBatch) -> tuple[tuple[str, int, int], ...]:
    return tuple((message.kind, message.position, message.part) for message in batch.source_index.view.messages)


def test_profiles_bundle_projection_permissions_and_request_contract() -> None:
    assert DEFAULT_MATERIAL_PROFILE is FULL_V4
    assert material_profile_for_versions("2", "1") is USER_V2
    assert material_profile_for_versions("3", "2") is FULL_V3
    assert material_profile_for_versions("3", "3") is FULL_V4
    assert FULL_V4.allow_input_rejections and FULL_V4.allow_created_rewrites
    assert not USER_V2.allow_input_rejections and not USER_V2.allow_created_rewrites
    assert not FULL_V3.allow_input_rejections and not FULL_V3.allow_created_rewrites
    assert (USER_V2.source_kinds, USER_V2.signal_kinds, USER_V2.allowed_types) == (
        ("user",),
        ("user_interrupt",),
        ("intent",),
    )
    assert USER_V2.instructions == INSTRUCTIONS and USER_V2.prompt == EXTRACTION_PROMPT
    assert FULL_V3.allowed_types == ("intent", "engineering")
    assert "file" not in FULL_V3.allowed_evidence
    assert FULL_V3.allowed_evidence == ("tool_result", "reference", "user_report")
    assert (USER_V2.tool_schema, FULL_V3.tool_schema) == ("intent", "engineering")
    for versions in (("3", "1"), ("2", "2"), ("99", "99")):
        with pytest.raises(ExtractionError) as caught:
            material_profile_for_versions(*versions)
        assert caught.value.kind == "invalid_replay"


def test_full_v4_only_changes_error_recovery_readback_and_done_prompt_rules() -> None:
    old, current = FULL_V3.prompt.splitlines(), FULL_V4.prompt.splitlines()
    assert len(old) == len(current)
    changed = [after.split(".", 1)[0] for before, after in zip(old, current, strict=True) if before != after]
    assert changed == ["8", "9", "12"]
    recovery, readback, done = (
        next(line for line in current if line.startswith(f"{number}.")) for number in (8, 9, 12)
    )
    assert "A Not saved response with candidates means no save happened." in recovery
    assert "input rejection" in recovery and "without changing any memory" in recovery
    assert "Correct it and resubmit" in recovery and "abandon that item" in recovery
    assert "Stop on any other tool error and report it." in recovery
    assert "newly created in this batch" in readback
    assert "same sources" in readback and "latest version" in readback
    assert "corrected or abandoned input rejections do not prevent Done" in done
    assert FULL_V4.instructions == FULL_V3.instructions
    assert FULL_V4.render_version == FULL_V3.render_version == "3"
    original = source((user(1), tool(2), assistant(3)))
    assert render_batch(PROJECT, (original,), 40_000, profile=FULL_V4) == render_batch(
        PROJECT, (original,), 40_000, profile=FULL_V3
    )


def test_full_v4_accepts_v2_tuning_in_all_three_replaceable_sentences() -> None:
    tuning = replace(
        V0,
        version="v2",
        do_not_save="Leave temporary tasks out of memory.",
        pasted_text="User-endorsed pasted feedback is quotable.",
        body_limits="Keep only conditions shown by the sources.",
    )
    tuned = tuned_material_profile(FULL_V4, tuning)
    assert tuned.allow_input_rejections and tuned.allow_created_rewrites
    restored = tuned.prompt
    for target, replacement in (
        (V0.do_not_save, tuning.do_not_save),
        (FULL_V4.pasted_text_sentence, tuning.pasted_text),
        (FULL_V4.body_limits_sentence, tuning.body_limits),
    ):
        assert replacement is not None
        assert FULL_V4.prompt.count(target) == 1
        assert tuned.prompt.count(replacement) == 1 and target not in tuned.prompt
        restored = restored.replace(replacement, target)
    assert restored == FULL_V4.prompt


def test_projection_precedes_signal_order_and_tool_singleton_grouping() -> None:
    first = source((user(1), tool(2)), (SourceAnnotation("tool_failures_then_success", 2, "tool-2", start_position=2),))
    second = replace(
        source((user(1),)),
        key=SessionKey("claude-code", "source-two"),
        messages=(replace(user(1), session="source-two"),),
        revision=SourceRevision(2, "b" * 64),
    )
    third = replace(
        second,
        key=SessionKey("claude-code", "source-three"),
        messages=(replace(user(1), session="source-three"),),
        annotations=(SourceAnnotation("user_interrupt", 1),),
        revision=SourceRevision(3, "c" * 64),
    )
    assert [item.key.session for item in ordered_sources((first, second, third), profile=USER_V2)] == [
        "source-three",
        "source-two",
        SESSION,
    ]
    assert [item.key.session for item in ordered_sources((first, second, third), profile=FULL_V3)] == [
        "source-three",
        SESSION,
        "source-two",
    ]
    assert len(select_members((first, second), profile=USER_V2)) == 1
    full_groups = select_members((first, second), profile=FULL_V3)
    assert len(full_groups) == 2 and full_groups[0] == (first,)
    assert project_source(first, USER_V2).messages == (user(1),)
    assert project_source(first, USER_V2).annotations == ()


def test_user_budget_is_reserved_before_large_nonuser_material() -> None:
    original = source((user(1, "old " * 100), tool(2, "t" * 3000), assistant(3, "a" * 3000), user(4, "new " * 100)))
    users_only = replace(original, messages=(original.messages[0], original.messages[-1]))
    budget = len(render_batch(PROJECT, (users_only,), 40_000, count_tokens=len).batch.material)
    batch = render_batch(PROJECT, (original,), budget, count_tokens=len).batch
    assert shown(batch) == (("user", 1, 0), ("user", 4, 0))
    counts = {count.kind: count for count in batch.material_counts}
    assert (counts["user"].displayed, counts["user"].omitted) == (2, 0)
    assert (counts["tool_result"].displayed, counts["tool_result"].omitted) == (0, 1)
    assert (counts["assistant"].displayed, counts["assistant"].omitted) == (0, 1)


def test_oversized_users_keep_latest_suffix_and_include_following_conclusions() -> None:
    original = source(
        (
            user(1, "old " * 300),
            tool(2, "old tool"),
            user(3, "new " * 200),
            tool(4, "new tool"),
            assistant(5, "new conclusion"),
        )
    )
    retained = replace(original, messages=original.messages[2:])
    budget = len(render_batch(PROJECT, (retained,), 40_000, count_tokens=len).batch.material)
    batch = render_batch(PROJECT, (original,), budget, count_tokens=len).batch
    assert shown(batch) == (("user", 3, 0), ("tool_result", 4, 0), ("assistant", 5, 0))
    assert "old tool" not in batch.material
    assert batch.omitted_messages == 2


async def test_nonuser_head_tail_are_independent_fragments_with_one_identity() -> None:
    body = "a" * 1000 + "UNSEEN_MIDDLE" * 100 + "z" * 1000
    original = source((user(1), tool(2, body, part=3), assistant(3, body)))
    batch = render_batch(PROJECT, (original,), 40_000).batch
    tools = [message for message in batch.source_index.view.messages if message.kind == "tool_result"]
    assert len(tools) == 1 and tools[0].part == 3
    assert tools[0].fragments == ("a" * 1000, "z" * 1000)
    assert tools[0].attributes == original.messages[1].attributes
    assert "UNSEEN_MIDDLE" not in batch.material
    assert "…[truncated]" in batch.material
    assert batch.material.count("\n…[middle omitted]\n") == 2
    assert (
        batch.material.index("a" * 1000) < batch.material.index("…[middle omitted]") < batch.material.index("z" * 1000)
    )
    assert all(
        "…[middle omitted]" not in fragment
        for message in batch.source_index.view.messages
        for fragment in message.fragments
    )
    assert 'Tool: "shell"' in batch.material
    evidence = await verify_evidence((ToolResultClaim("z" * 100, session=SESSION),), batch.source_index.view)
    assert evidence[0].item == "tool-2"
    context = verify_context(
        QuoteClaim("context_quote", "context_candidate", "a" * 100, session=SESSION), batch.source_index.view
    )
    assert context.item == "assistant-3"
    for text in ("az", "UNSEEN_MIDDLE", "synthetic"):
        with pytest.raises(SourceVerificationError):
            await verify_evidence((ToolResultClaim(text, session=SESSION),), batch.source_index.view)


async def test_original_tool_text_blocks_are_not_joined_to_fill_head_or_tail() -> None:
    message = replace(tool(2), fragments=("first block", "second block", "third block"))
    batch = render_batch(PROJECT, (source((user(1), message)),), 40_000).batch
    retained = batch.source_index.view.messages[1]
    assert retained.fragments == message.fragments
    with pytest.raises(SourceVerificationError):
        await verify_evidence((ToolResultClaim("blocksecond", session=SESSION),), batch.source_index.view)


def test_signal_neighborhood_uses_exact_failure_interval_and_next_assistant() -> None:
    messages = (
        user(1),
        tool(2, part=0, error=True),
        tool(2, part=1, error=True),
        assistant(3),
        tool(4),
        tool(5),
        assistant(6),
        tool(7),
        assistant(8),
    )
    annotation = SourceAnnotation("tool_failures_then_success", 5, "tool-5", start_position=2, start_part=1)
    original = source(messages, (annotation,))
    assert signal_neighborhood(original) == frozenset(source_identity(message) for message in messages[2:7])
    interrupt = replace(original, annotations=(SourceAnnotation("user_interrupt", 7),))
    assert signal_neighborhood(interrupt) == frozenset(source_identity(message) for message in messages[4:7])


@pytest.mark.parametrize(
    "policy,positions,omitted_errors,omitted_neighbors",
    [
        ("chronological", (1, 2), 2, 3),
        ("head_tail", (1, 6), 2, 2),
        ("signal_first", (4, 5), 0, 1),
    ],
)
def test_three_policies_select_different_nonuser_material_and_count_same_neighborhood(
    policy: OversizeSelection,
    positions: tuple[int, int],
    omitted_errors: int,
    omitted_neighbors: int,
) -> None:
    messages = (user(0), *(tool(index, "x" * 200, error=index in (4, 5)) for index in range(1, 7)))
    original = source(messages, (SourceAnnotation("user_interrupt", 7),))
    two = replace(original, messages=(messages[0], messages[1], messages[2]))
    budget = len(render_batch(PROJECT, (two,), 40_000, count_tokens=len).batch.material)
    batch = render_batch(PROJECT, (original,), budget, count_tokens=len, oversize_selection=policy).batch
    assert (
        tuple(message.position for message in batch.source_index.view.messages if message.kind == "tool_result")
        == positions
    )
    assert batch.omitted_error_results == omitted_errors
    assert batch.omitted_signal_neighbors == omitted_neighbors
    assert batch.omitted_messages == 4


def test_full_prompt_updates_type_context_evidence_and_review_rules() -> None:
    batch = render_batch(PROJECT, (source((user(1), tool(2))),), 40_000).batch
    full = render_prompt(batch, "catalog", profile=FULL_V3)
    legacy = render_prompt(batch, "catalog", profile=USER_V2)
    assert FULL_V3.prompt in full and USER_V2.prompt in legacy
    assert "A substantive intent update needs at least one newly verified user_quote" in full
    assert "a substantive engineering update needs at least one newly verified tool_result or user_report" in full
    assert "include context_session" in full and "File evidence is unavailable" in full
    assert "memory_withdraw" in full and "memory_merge" in full and "memory_note_conflict" in full
    assert "Permanent deletion is unavailable in extraction" in full
    assert "Entries with context/evidence, non-intent entries" in legacy


def test_offline_planning_can_cover_more_than_three_batches_without_changing_default_limit() -> None:
    originals = tuple(
        replace(
            source((user(1), tool(2))),
            key=SessionKey("claude-code", f"source-{index}"),
            messages=(replace(user(1), session=f"source-{index}"), replace(tool(2), session=f"source-{index}")),
        )
        for index in range(6)
    )
    assert len(build_batches(PROJECT, originals, 40_000)) == 3
    assert len(build_batches(PROJECT, originals, 40_000, max_batches=len(originals))) == 6


def test_full_profile_keeps_unrecognized_structural_annotations_but_legacy_does_not() -> None:
    original = source((user(1),), (SourceAnnotation("future_signal", 1, "synthetic-marker"),))
    assert project_source(original, FULL_V3).annotations == original.annotations
    assert project_source(original, USER_V2).annotations == ()
    assert "future_signal" in render_batch(PROJECT, (original,), 40_000, profile=FULL_V3).batch.material
    assert "future_signal" not in render_batch(PROJECT, (original,), 40_000, profile=USER_V2).batch.material


def test_users_across_member_sessions_get_budget_before_any_assistant() -> None:
    first = source((user(1, "first user " * 20), assistant(2, "a" * 4000)))
    second_session = "00000000-0000-4000-8000-000000000002"
    second = replace(
        first,
        key=SessionKey("claude-code", second_session),
        revision=SourceRevision(2, "b" * 64),
        messages=(
            replace(user(1, "second user " * 20), session=second_session),
            replace(assistant(2, "b" * 4000), session=second_session),
        ),
    )
    only_users = tuple(
        replace(original, messages=tuple(message for message in original.messages if message.kind == "user"))
        for original in (first, second)
    )
    budget = len(build_batches(PROJECT, only_users, 40_000, count_tokens=len)[0].material)
    batches = build_batches(PROJECT, (first, second), budget, count_tokens=len)
    assert len(batches) == 1 and len(batches[0].sources) == 2
    assert {(message.session, message.kind) for message in batches[0].source_index.view.messages} == {
        (SESSION, "user"),
        (second_session, "user"),
    }
    assert batches[0].omitted_assistant_messages == 2


def test_tool_singleton_does_not_bypass_invalid_member_cap() -> None:
    original = source((user(1), tool(2)))
    with pytest.raises(ExtractionError) as caught:
        select_members((original,), {original.key: 0}, profile=FULL_V3)
    assert caught.value.kind == "watermark_invalid"
