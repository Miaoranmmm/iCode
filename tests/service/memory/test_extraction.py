# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Extraction boundaries from the step-5 batch, source, and outcome contract."""

from __future__ import annotations

from dataclasses import replace

import pytest

from chrys.service.memory.extraction import (
    CONTEXT_MARGIN,
    EXTRACTION_PROMPT,
    MAX_MATERIAL_TOKENS,
    PROMPT_VERSION,
    RENDER_VERSION,
    USER_V2,
    ExtractionError,
    ExtractionResult,
    SessionKey,
    SourceAnnotation,
    SourceRevision,
    SourceSession,
    UsageSummary,
    build_batches,
    classify_result,
    conflicting_sessions,
    material_budget,
    ordered_sources,
    render_batch,
    render_prompt,
    select_members,
)
from chrys.service.memory.storage.store import MemoryScope
from chrys.service.memory.verification import QuoteClaim, SourceMessage, SourceVerificationError, verify_quotes

PROJECT = MemoryScope("project", "0123456789abcdef")


def source(
    index: int,
    *,
    host: str = "claude-code",
    annotations: bool = False,
    messages: tuple[SourceMessage, ...] | None = None,
) -> SourceSession:
    session = f"00000000-0000-4000-8000-{index:012d}"
    selected = (
        messages
        if messages is not None
        else (SourceMessage(session, 2, (f"Keep rule {index}.",), item=f"item-{index}", at="2026-11-09T10:00:00Z"),)
    )
    return SourceSession(
        SessionKey(host, session),
        PROJECT,
        SourceRevision(index, f"{index:064x}"),
        selected,
        (SourceAnnotation("user_interrupt", 1, f"interrupt-{index}"),) if annotations else (),
    )


def claim(text: str, session: str) -> QuoteClaim:
    return QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", text, session=session)


def test_signal_order_is_stable_newest_first_and_without_signals_is_not_filtered() -> None:
    sources = (source(1), source(2, annotations=True), source(3), source(4, annotations=True))
    assert [item.key.session[-1] for item in ordered_sources(sources, profile=USER_V2)] == ["4", "2", "3", "1"]
    batches = build_batches(PROJECT, sources, 40_000, profile=USER_V2)
    assert batches[0].sources == ordered_sources(sources, profile=USER_V2)
    assert len(batches[0].source_index.view.messages) == 4


def test_mtime_ties_use_host_then_full_session_and_conflicts_are_explicit() -> None:
    first = source(2, host="codex")
    second = replace(source(1), revision=first.revision)
    assert ordered_sources((first, second), profile=USER_V2) == (second, first)
    assert conflicting_sessions((source(1), source(1, host="codex"), source(2))) == frozenset({source(1).key.session})


def test_membership_budget_and_host_rules_are_separate() -> None:
    sources = tuple(source(index) for index in range(1, 24)) + tuple(
        source(index, host="codex") for index in range(24, 38)
    )
    members = select_members(sources, profile=USER_V2)
    assert all(len(group) <= 10 and len({item.key.host for item in group}) == 1 for group in members)
    assert sum(map(len, members)) == 37
    assert len(build_batches(PROJECT, sources, 40_000, profile=USER_V2)) == 3
    assert all(
        batch.estimated_tokens <= MAX_MATERIAL_TOKENS
        for batch in build_batches(PROJECT, sources, 40_000, profile=USER_V2)
    )
    cap = {sources[0].key: 1, sources[-1].key: 2}
    assert all(
        len(group) <= min(cap.get(item.key, 10) for item in group)
        for group in select_members(sources, cap, profile=USER_V2)
    )


def test_budget_defers_whole_members_then_oversized_session_owns_batch() -> None:
    sources = (source(2), source(1))
    exact = len(render_batch(PROJECT, sources[:1], 20_000, count_tokens=len, profile=USER_V2).batch.material)
    rendered = render_batch(PROJECT, sources, exact, count_tokens=len, profile=USER_V2)
    assert rendered.batch.sources == sources[:1]
    assert rendered.remaining == sources[1:]
    assert len(build_batches(PROJECT, sources, exact, count_tokens=len, profile=USER_V2)) == 2


def test_long_message_view_contains_only_displayed_prefix_with_fragment_boundaries() -> None:
    session = source(1).key.session
    message = SourceMessage(
        session, 7, ("a" * 1900, "visible-tail" + "b" * 100 + "unseen-tail"), item="long", at="2026-11-09T00:00:00Z"
    )
    selected = source(1, messages=(message,))
    batch = render_batch(PROJECT, (selected,), 40_000, profile=USER_V2).batch
    view = batch.source_index.view
    assert view.current_session is None
    assert view.messages[0].fragments == ("a" * 1900, ("visible-tail" + "b" * 100)[:100])
    assert sum(map(len, view.messages[0].fragments)) == 2000
    assert "…[truncated]" in batch.material
    assert batch.omitted_characters == sum(map(len, message.fragments)) - 2000
    assert verify_quotes((claim("visible-tail", session),), view)[0].item == "long"
    for forbidden in ("unseen-tail", "aavisible-tail", "…[truncated]", "long", f"## {session}"):
        with pytest.raises(SourceVerificationError) as caught:
            verify_quotes((claim(forbidden, session),), view)
        assert caught.value.failures[0].reason == "not_found"


def test_oversized_session_uses_last_complete_messages_in_original_order() -> None:
    session = source(1).key.session
    messages = tuple(
        SourceMessage(session, index, (f"Original message {index} " + "x" * 200,), item=f"m-{index}")
        for index in range(1, 5)
    )
    selected = source(1, messages=messages)
    last_two = render_batch(
        PROJECT, (replace(selected, messages=messages[-2:]),), 10000, count_tokens=len, profile=USER_V2
    ).batch
    batch = render_batch(
        PROJECT, (selected, source(2)), len(last_two.material), count_tokens=len, profile=USER_V2
    ).batch
    assert batch.sources == (selected,)
    assert [message.position for message in batch.source_index.view.messages] == [3, 4]
    assert batch.omitted_messages == 2
    assert batch.omitted_characters == sum(sum(map(len, message.fragments)) for message in messages[:2])
    assert "Original message 1" not in batch.material


def test_structural_signals_use_successor_or_header_and_never_enter_view() -> None:
    session = source(1).key.session
    message = SourceMessage(session, 5, ("Keep this literal structure_signal wording.",), item="u")
    selected = replace(
        source(1, messages=(message,)),
        annotations=(
            SourceAnnotation("user_interrupt", 4, "interrupt"),
            SourceAnnotation("user_interrupt", 8, "tail-signal"),
        ),
    )
    batch = render_batch(PROJECT, (selected,), 40_000, profile=USER_V2).batch
    tail = '<structure_signal kind="user_interrupt" item="tail-signal" position="8"/>'
    preceding = '<structure_signal kind="user_interrupt" item="interrupt" position="4"/>'
    assert batch.material.index(tail) < batch.material.index(preceding) < batch.material.index("Message item=")
    assert batch.source_index.view.messages == (message,)
    for text in ("tail-signal", tail, preceding):
        with pytest.raises(SourceVerificationError):
            verify_quotes((claim(text, session),), batch.source_index.view)


def test_material_versions_keep_the_legacy_contract_explicit() -> None:
    assert (RENDER_VERSION, PROMPT_VERSION) == ("3", "3")
    assert (USER_V2.render_version, USER_V2.prompt_version) == ("2", "1")


def test_references_catalog_and_task_are_context_only_and_prompt_order_is_fixed() -> None:
    batch = render_batch(PROJECT, (source(1),), 40_000, profile=USER_V2).batch
    prompt = render_prompt(batch, "CATALOG_SENTINEL", (("AGENTS.md", "REFERENCE_SENTINEL"),), profile=USER_V2)
    assert (
        prompt.index("<extraction_task>")
        < prompt.index("CATALOG_SENTINEL")
        < prompt.index("<reference_files>")
        < prompt.index("<source_sessions>")
    )
    assert EXTRACTION_PROMPT in prompt
    assert "Read the whole displayed batch before deciding what to write." in prompt
    assert "Check existing entries first." in prompt
    assert "Add usable detail without adding claims." in prompt
    assert "Read back every actual save" in prompt
    assert "<reference_files>\nnone\n</reference_files>" in render_prompt(batch, "none", profile=USER_V2)
    for text in ("CATALOG_SENTINEL", "REFERENCE_SENTINEL", "Reply Done"):
        with pytest.raises(SourceVerificationError):
            verify_quotes((claim(text, source(1).key.session),), batch.source_index.view)


def test_new_quote_cannot_select_another_session_or_join_fragments() -> None:
    first = source(1, messages=(SourceMessage(source(1).key.session, 1, ("Only first", "separate")),))
    batch = render_batch(PROJECT, (first, source(2)), 40_000, profile=USER_V2).batch
    for text, session in (("Only first", source(2).key.session), ("Only firstseparate", first.key.session)):
        with pytest.raises(SourceVerificationError):
            verify_quotes((claim(text, session),), batch.source_index.view)


def test_window_accounting_caps_material_and_never_silently_crops_references() -> None:
    assert material_budget(350_000, 16_384, "references") == 40_000
    assert material_budget(30_000, 1000, "x" * 1000, count_tokens=len) == 8000
    for window, reference, margin in (
        (20_001, "xx", CONTEXT_MARGIN),
        (100_000, "x" * 80_000, CONTEXT_MARGIN),
        (100_000, "x", 19999),
    ):
        with pytest.raises(ExtractionError) as caught:
            material_budget(window, 1000, reference, context_margin=margin, count_tokens=len)
        assert caught.value.kind == "input_budget_exceeded"
    with pytest.raises(ExtractionError, match="one displayed user message"):
        render_batch(PROJECT, (source(1),), 1, count_tokens=len, profile=USER_V2)


@pytest.mark.parametrize(
    "termination,calls,failures,pending,cancel_requested,expected",
    [
        ("completed", 1, 0, 0, False, "success"),
        ("completed", 0, 0, 0, False, "failure"),
        ("completed", 3, 1, 0, False, "failure"),
        ("completed", 3, 0, 1, False, "failure"),
        ("completed", 3, 0, 0, True, "cancelled"),
        ("turn_limited", 0, 0, 0, False, "incomplete_limits"),
        ("output_limited", 0, 0, 0, False, "incomplete_limits"),
        ("turn_limited", 2, 1, 0, False, "failure"),
        ("output_limited", 1, 0, 1, False, "failure"),
        ("cancelled", 1, 0, 0, False, "cancelled"),
        ("failed", 1, 0, 0, False, "failure"),
    ],
)
def test_own_observations_classify_outcomes(
    termination: str, calls: int, failures: int, pending: int, cancel_requested: bool, expected: str
) -> None:
    result = ExtractionResult(
        termination, calls, failures, 4, pending, None, UsageSummary(), cancel_requested=cancel_requested
    )  # type: ignore[arg-type]
    assert classify_result(result) == expected


@pytest.mark.parametrize(
    "termination,expected",
    [("completed", "success"), ("turn_limited", "failure"), ("output_limited", "failure")],
)
def test_rejections_only_allow_success_after_normal_completion(termination: str, expected: str) -> None:
    result = ExtractionResult(termination, 1, 0, 0, 0, None, UsageSummary(), rejection_count=1)  # type: ignore[arg-type]
    assert classify_result(result) == expected
