# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Quote verification contracts for the memory trial's conversation view and verify_quotes."""

# ruff: noqa: RUF001  # Full-width source text tests exact matching.

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace

import pytest

from chrys.service.memory.storage.store import MemoryOperationError, Quote
from chrys.service.memory.verification import (
    ConversationView,
    QuoteClaim,
    SourceMessage,
    SourceVerificationError,
    verify_quotes,
)

AT = "2026-11-04T08:41:19Z"


def claim(text: str, *, session: str | None = None, candidate: str | None = None) -> QuoteClaim:
    return QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", text, candidate, session)


def message(text: str, position: int = 0, *, session: str = "session-full-id") -> SourceMessage:
    return SourceMessage(session, position, (text,), at=AT)


def token_for(value: SourceMessage) -> str:
    identity = ("item", value.item) if value.item is not None else ("position", value.position)
    payload = json.dumps(
        [value.session, value.kind, *identity, value.part, list(value.fragments)],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return "q-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def failure(text: str, view: ConversationView, *, candidate: str | None = None) -> SourceVerificationError:
    with pytest.raises(SourceVerificationError) as caught:
        verify_quotes((claim(text, candidate=candidate),), view)
    return caught.value


def test_exact_substring_carries_original_source_metadata() -> None:
    source = SourceMessage("session-full-id", 9, ("before: use SQLite WAL. after",), item="persisted-item", at=AT)
    result = verify_quotes((claim("use SQLite WAL"),), ConversationView((source,), source.session))
    assert result == (Quote("use SQLite WAL", "message", source.session, AT, "persisted-item"),)


@pytest.mark.parametrize("text", ["use sqlite", "use  SQLite", "use ＳＱＬｉｔｅ"])
def test_matching_does_not_fold_case_whitespace_or_width(text: str) -> None:
    view = ConversationView((message("use SQLite"),), "session-full-id")
    assert failure(text, view).kind == "memory_quote_not_found"


@pytest.mark.parametrize(
    "messages",
    [
        (message("以后"), message("都用中文", 1)),
        (SourceMessage("session-full-id", 0, ("以后", "都用中文")),),
    ],
)
def test_joining_messages_or_fragments_is_rejected(messages: tuple[SourceMessage, ...]) -> None:
    error = failure("以后都用中文", ConversationView(messages, "session-full-id"))
    assert error.failures[0].reason == "not_found"


def test_two_fragments_in_one_message_count_as_one_match() -> None:
    source = SourceMessage("s", 4, ("yes now", "yes later"))
    assert verify_quotes((claim("yes"),), ConversationView((source,), "s")) == (Quote("yes", "message", "s"),)


def test_ambiguity_lists_messages_in_view_order_with_stable_tokens() -> None:
    sources = (message("yes, first", 4), message("yes, second", 8))
    error = failure("yes", ConversationView(sources, "session-full-id"))
    assert error.kind == "memory_evidence_ambiguous"
    [failed] = error.failures
    assert failed.field == "user_quotes[0]"
    assert failed.candidate_field == "user_quotes[0].candidate"
    assert failed.text == "yes"
    assert failed.reason == "ambiguous"
    assert [candidate.token for candidate in failed.candidates] == [token_for(source) for source in sources]
    assert all(re.fullmatch(r"q-[0-9a-f]{12}", candidate.token) for candidate in failed.candidates)
    assert [(item.session, item.at) for item in failed.candidates] == [("session-full-id", AT)] * 2
    assert "first" in failed.candidates[0].snippet


def test_correct_candidate_selects_the_message() -> None:
    sources = (message("yes", 4), replace(message("yes", 8), item="second"))
    result = verify_quotes(
        (claim("yes", candidate=token_for(sources[1])),), ConversationView(sources, "session-full-id")
    )
    assert result == (Quote("yes", "message", "session-full-id", AT, "second"),)


@pytest.mark.parametrize("change", ["gone", "changed", "wrong_text", "only_elsewhere"])
def test_candidate_expiration_never_falls_back(change: str) -> None:
    original = message("yes, proceed", 4)
    selected = token_for(original)
    if change == "gone":
        messages = ()
        text = "yes"
    elif change == "changed":
        messages = (replace(original, fragments=("yes, proceed!",)),)
        text = "yes"
    elif change == "wrong_text":
        messages = (original,)
        text = "no"
    else:
        messages = (message("yes, proceed", 9),)
        text = "yes"
    error = failure(text, ConversationView(messages, "session-full-id"), candidate=selected)
    assert error.kind == "memory_evidence_candidate_expired"
    assert error.failures[0].reason == "candidate_expired"
    assert error.failures[0].candidates == ()


def test_not_found_keeps_claim_text_verbatim() -> None:
    text = 'line one\n"line two"'
    error = failure(text, ConversationView((), "s"))
    assert error.failures[0].text == text


def test_batch_requires_session_even_for_a_stored_cli_quote() -> None:
    quote = Quote("direct", "cli")
    with pytest.raises(MemoryOperationError) as caught:
        verify_quotes((claim("direct"),), ConversationView((), None), stored=(quote,))
    assert caught.value.kind == "memory_invalid_input"


def test_batch_saves_only_from_the_named_session() -> None:
    view = ConversationView((message("yes", session="a"), message("no", session="b")), None)
    assert verify_quotes((claim("yes", session="a"),), view)[0].session == "a"
    with pytest.raises(SourceVerificationError) as caught:
        verify_quotes((claim("yes", session="b"),), view)
    # Nothing is attributed to another session; the real one is only offered for a resubmission.
    assert caught.value.kind == "memory_source_wrong_session"
    assert [item.session for item in caught.value.failures[0].candidates] == ["a"]


@pytest.mark.parametrize("session,via", [("old-full-session", "message"), (None, "cli")])
def test_stored_quote_is_preserved_without_a_view_match(session: str | None, via: str) -> None:
    old = Quote("old words", via, session, AT, "old-item")  # type: ignore[arg-type]
    [result] = verify_quotes((claim(old.text, session=session),), ConversationView((), "new"), stored=(old,))
    assert result is old


def test_stored_quote_with_a_different_session_is_not_preserved() -> None:
    old = Quote("old words", "message", "old")
    with pytest.raises(SourceVerificationError):
        verify_quotes((claim(old.text, session="different"),), ConversationView((), "new"), stored=(old,))


def test_batch_quote_in_another_shown_session_offers_that_session_without_saving() -> None:
    real = SourceMessage("9d2a6307-real", 3, ("keep it short",), at=AT)
    other = SourceMessage("eba130bf-other", 1, ("unrelated",), at=AT)
    view = ConversationView((real, other), None)
    with pytest.raises(SourceVerificationError) as caught:
        verify_quotes((claim("keep it short", session="9d2a6308-real"),), view)
    assert caught.value.kind == "memory_source_wrong_session"
    [failed] = caught.value.failures
    assert failed.reason == "wrong_session"
    assert [(item.session, item.token) for item in failed.candidates] == [(real.session, token_for(real))]
    # The corrected session saves; a missing text stays not_found.
    assert verify_quotes((claim("keep it short", session=real.session),), view)[0].session == real.session
    with pytest.raises(SourceVerificationError) as missing:
        verify_quotes((claim("never said", session=real.session),), view)
    assert missing.value.failures[0].reason == "not_found"


def test_a_stored_quote_is_retained_when_only_another_session_repeats_its_text() -> None:
    # Continued sessions repeat earlier user messages; the stored source still stands.
    old = Quote("keep it short", "message", "old-session", AT, "old-item")
    view = ConversationView((SourceMessage("continued-session", 2, ("keep it short",), at=AT),), None)
    [kept] = verify_quotes((claim(old.text, session=old.session),), view, stored=(old,), view_first=True)
    assert kept is old


def test_conversation_mode_has_no_cross_session_offer() -> None:
    view = ConversationView((message("keep it short"),), "session-full-id")
    error = failure("not here", view)
    assert error.failures[0].reason == "not_found"


@pytest.mark.parametrize(
    "reasons,kind",
    [
        (("ambiguous", "candidate_expired", "not_found"), "memory_quote_not_found"),
        (("ambiguous", "candidate_expired"), "memory_evidence_candidate_expired"),
        (("ambiguous", "ambiguous"), "memory_evidence_ambiguous"),
    ],
)
def test_all_failures_are_reported_and_kind_has_specified_priority(reasons: tuple[str, ...], kind: str) -> None:
    view = ConversationView((message("yes", 1), message("yes", 2)), "session-full-id")
    claims = tuple(
        QuoteClaim(
            f"user_quotes[{index}]",
            f"user_quotes[{index}].candidate",
            "missing" if reason == "not_found" else "yes",
            "q-expired" if reason == "candidate_expired" else None,
        )
        for index, reason in enumerate(reasons)
    )
    with pytest.raises(SourceVerificationError) as caught:
        verify_quotes(claims, view)
    assert caught.value.kind == kind
    assert tuple(item.reason for item in caught.value.failures) == reasons
    assert tuple(item.field for item in caught.value.failures) == tuple(value.field for value in claims)


def test_appending_to_a_view_does_not_change_existing_tokens() -> None:
    sources = (message("yes", 1), message("yes", 2))
    before = failure("yes", ConversationView(sources, "session-full-id"))
    after = failure("yes", ConversationView((*sources, message("yes", 3)), "session-full-id"))
    assert after.failures[0].candidates[:2] == before.failures[0].candidates


def test_item_and_position_identities_do_not_collide() -> None:
    first = replace(message("yes", 7), item="7")
    second = message("yes", 7)
    error = failure("yes", ConversationView((first, second), "session-full-id"))
    tokens = tuple(item.token for item in error.failures[0].candidates)
    assert tokens == (token_for(first), token_for(second))
    assert tokens[0] != tokens[1]


def test_fragment_boundaries_are_encoded_as_an_array() -> None:
    split = SourceMessage("s", 1, ("a", "b"))
    joined = SourceMessage("s", 1, ("a\0b",))
    split_token = token_for(split)
    joined_token = token_for(joined)
    assert split_token != joined_token
    assert verify_quotes((claim("a", candidate=split_token),), ConversationView((split,), "s"))
    assert (
        failure("a", ConversationView((joined,), "s"), candidate=split_token).kind
        == "memory_evidence_candidate_expired"
    )
    assert verify_quotes((claim("a", candidate=joined_token),), ConversationView((joined,), "s"))


@pytest.mark.parametrize("text", ["", "x" * 501])
def test_new_claim_empty_or_over_500_is_invalid(text: str) -> None:
    with pytest.raises(MemoryOperationError) as caught:
        verify_quotes((claim(text),), ConversationView((message("x" * 501),), "session-full-id"))
    assert caught.value.kind == "memory_invalid_input"


def test_1200_character_direct_quote_can_be_preserved() -> None:
    old = Quote("x" * 1200, "remember", "old", AT)
    assert verify_quotes((claim(old.text, session="old"),), ConversationView((), "new"), stored=(old,)) == (old,)


@pytest.mark.parametrize(
    "case",
    [
        "empty_session",
        "other_session",
        "duplicate_position",
        "duplicate_item",
        "no_fragments",
        "empty_fragment",
        "naive_at",
    ],
)
def test_view_rejects_invalid_host_snapshots(case: str) -> None:
    with pytest.raises(MemoryOperationError) as caught:
        current = "s"
        if case == "empty_session":
            messages = (SourceMessage("", 0, ("words",)),)
            current = None
        elif case == "other_session":
            messages = (message("words", session="other"),)
        elif case == "duplicate_position":
            messages = (message("words", session="s"), message("different", session="s"))
        elif case == "duplicate_item":
            messages = (
                replace(message("a", 1, session="s"), item="same"),
                replace(message("b", 2, session="s"), item="same"),
            )
        elif case == "no_fragments":
            messages = (SourceMessage("s", 0, ()),)
        elif case == "empty_fragment":
            messages = (SourceMessage("s", 0, ("ok", "")),)
        else:
            messages = (SourceMessage("s", 0, ("ok",), at="2026-11-04T08:41:19"),)
        ConversationView(messages, current)
    assert caught.value.kind == "memory_invalid_input"


def test_conversation_ignores_claim_session_for_new_matches() -> None:
    source = message("new words", session="current")
    [result] = verify_quotes((claim("new words", session="unrelated"),), ConversationView((source,), "current"))
    assert result.session == "current"


def test_batch_preserves_stored_quote_with_explicit_session_outside_view() -> None:
    old = Quote("old words", "message", "old", AT)
    assert verify_quotes((claim(old.text, session="old"),), ConversationView((), None), stored=(old,)) == (old,)
