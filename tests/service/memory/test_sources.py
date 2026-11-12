# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Per-kind source verification, identity, retention and candidate contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from chrys.service.memory.storage.store import Context, MemoryOperationError, ToolResultEvidence
from chrys.service.memory.verification import (
    ConversationView,
    QuoteClaim,
    SourceMessage,
    SourceVerificationError,
    ToolResultClaim,
    UserReportClaim,
    verify_context,
    verify_evidence,
    verify_quotes,
)


async def verify(target: str, text: str, view: ConversationView, candidate: str | None = None):
    if target == "context":
        return verify_context(QuoteClaim("context_quote", "context_candidate", text, candidate), view)
    if target == "quote":
        return verify_quotes((QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", text, candidate),), view)
    claim = ToolResultClaim(text, candidate) if target == "tool_result" else UserReportClaim(text, candidate)
    return await verify_evidence((claim,), view)


@pytest.mark.parametrize(
    "target,kind",
    [("quote", "user"), ("user_report", "user"), ("context", "assistant"), ("tool_result", "tool_result")],
)
async def test_each_text_source_kind_has_match_not_found_ambiguity_and_expiration(target: str, kind: str) -> None:
    first = SourceMessage(
        "s", 1, ("before exact text after",), kind=kind, item="one", attributes=(("tool", "shell"), ("args", "{}"))
    )  # type: ignore[arg-type]
    view = ConversationView((first,), "s")
    assert await verify(target, "exact text", view)
    with pytest.raises(SourceVerificationError) as missing:
        await verify(target, "absent", view)
    assert missing.value.kind == ("memory_quote_not_found" if target == "quote" else "memory_evidence_not_found")
    second = replace(first, position=2, item="two")
    with pytest.raises(SourceVerificationError) as ambiguous:
        await verify(target, "exact text", ConversationView((first, second), "s"))
    assert ambiguous.value.kind == "memory_evidence_ambiguous"
    token = ambiguous.value.failures[0].candidates[0].token
    assert await verify(target, "exact text", view, token)
    with pytest.raises(SourceVerificationError) as expired:
        await verify(target, "exact text", ConversationView((second,), "s"), token)
    assert expired.value.kind == "memory_evidence_candidate_expired"


@pytest.mark.parametrize(
    "target,other_kind",
    [
        ("quote", "assistant"),
        ("quote", "tool_result"),
        ("user_report", "tool_result"),
        ("context", "user"),
        ("context", "tool_result"),
        ("tool_result", "assistant"),
        ("tool_result", "user"),
    ],
)
async def test_identical_text_from_another_kind_cannot_be_verified(target: str, other_kind: str) -> None:
    source = SourceMessage("s", 1, ("same exact text",), kind=other_kind, attributes=(("tool", "shell"),))  # type: ignore[arg-type]
    with pytest.raises(SourceVerificationError) as caught:
        await verify(target, "same exact text", ConversationView((source,), "s"))
    assert caught.value.failures[0].reason == "not_found"


@pytest.mark.parametrize(
    "target,kind",
    [("quote", "user"), ("user_report", "user"), ("context", "assistant"), ("tool_result", "tool_result")],
)
async def test_no_kind_can_join_fragments(target: str, kind: str) -> None:
    source = SourceMessage("s", 1, ("first", "second"), kind=kind, attributes=(("tool", "shell"),))  # type: ignore[arg-type]
    with pytest.raises(SourceVerificationError) as caught:
        await verify(target, "firstsecond", ConversationView((source,), "s"))
    assert caught.value.failures[0].reason == "not_found"


async def test_kind_and_part_are_in_message_identity_and_candidate_tokens() -> None:
    values = (
        SourceMessage("s", 1, ("same",), item="shared"),
        SourceMessage("s", 1, ("same",), item="shared", part=1),
        SourceMessage("s", 1, ("same",), kind="assistant", item="shared"),
        SourceMessage("s", 1, ("same",), kind="assistant", item="shared", part=1),
    )
    view = ConversationView(values, "s")
    tokens: list[str] = []
    for target in ("quote", "context"):
        with pytest.raises(SourceVerificationError) as caught:
            await verify(target, "same", view)
        tokens.extend(candidate.token for candidate in caught.value.failures[0].candidates)
    assert len(set(tokens)) == 4
    with pytest.raises(SourceVerificationError) as wrong_kind:
        await verify("context", "same", view, tokens[0])
    assert wrong_kind.value.kind == "memory_evidence_candidate_expired"


@pytest.mark.parametrize(
    "name",
    [
        "memory_read",
        "mcp__memory__memory_search",
        "mcp__memory__memory_save",
        "memory_catalog",
        "memory_forget",
        "memory_withdraw",
        "memory_merge",
        "memory_note_conflict",
    ],
)
async def test_memory_tool_results_are_not_evidence_for_either_host_name_shape(name: str) -> None:
    source = SourceMessage("s", 1, ("echoed old quote",), kind="tool_result", attributes=(("tool", name),))
    with pytest.raises(SourceVerificationError) as caught:
        await verify_evidence((ToolResultClaim("echoed old quote"),), ConversationView((source,), "s"))
    assert caught.value.kind == "memory_evidence_not_found"


async def test_tool_evidence_copies_attributes_and_keeps_summarized_args_unchanged() -> None:
    source = SourceMessage(
        "s",
        9,
        ("observed result",),
        kind="tool_result",
        item="result",
        at="2026-11-10T08:00:00Z",
        attributes=(("tool", "Task"), ("args", "x" * 199 + "…")),
    )
    [evidence] = await verify_evidence((ToolResultClaim("result"),), ConversationView((source,), "s"))
    assert isinstance(evidence, ToolResultEvidence)
    assert (evidence.tool, evidence.args) == ("Task", "x" * 199 + "…")
    assert len(evidence.args) == 200
    assert (evidence.session, evidence.item, evidence.at) == ("s", "result", source.at)


async def test_retention_only_reuses_same_evidence_kind_and_text_session() -> None:
    stored = ToolResultEvidence("shell", "{}", "same", "old", item="tool-item")
    view = ConversationView((), "current")
    [retained] = await verify_evidence((ToolResultClaim("same", session="old"),), view, stored=(stored,))
    assert retained is stored
    with pytest.raises(SourceVerificationError):
        await verify_evidence((UserReportClaim("same", session="old"),), view, stored=(stored,))


def test_context_session_retains_old_context_without_candidate_or_view_lookup() -> None:
    stored = Context("old proposal", "old", item="old-assistant")
    claim = QuoteClaim("context_quote", "context_candidate", "old proposal", "expired-token", "old")
    assert verify_context(claim, ConversationView((), "new"), stored=stored) is stored
    with pytest.raises(SourceVerificationError):
        verify_context(replace(claim, session="different", candidate=None), ConversationView((), "new"), stored=stored)


def test_batch_context_requires_session_even_when_retaining_context() -> None:
    with pytest.raises(MemoryOperationError) as caught:
        verify_context(
            QuoteClaim("context_quote", "context_candidate", "old"), ConversationView((), None), stored=Context("old")
        )
    assert caught.value.kind == "memory_invalid_input"
    assert "session is required in batch mode" in caught.value.message


@pytest.mark.parametrize(
    "field,value", [("kind", "other"), ("part", -1), ("part", True), ("attributes", (("tool", 42),))]
)
def test_new_source_fields_are_validated(field: str, value: object) -> None:
    with pytest.raises(MemoryOperationError):
        replace(SourceMessage("s", 1, ("text",)), **{field: value})


async def test_user_report_and_reference_retention_preserves_exact_stored_sources() -> None:
    from chrys.service.memory.storage.store import ReferenceEvidence, UserReportEvidence
    from chrys.service.memory.verification import ReferenceClaim

    report = UserReportEvidence("a fact", "old", "2026-11-10T08:00:00Z", "human-old")
    reference = ReferenceEvidence("https://reference.invalid/document", "the supplied note")
    results = await verify_evidence(
        (UserReportClaim("a fact", session="old"), ReferenceClaim(reference.ref, reference.note)),
        ConversationView((), "new"),
        stored=(report, reference),
    )
    assert results[0] is report and results[1] is reference
    [changed_note] = await verify_evidence(
        (ReferenceClaim(reference.ref, "other note"),), ConversationView((), "new"), stored=(reference,)
    )
    assert changed_note == ReferenceEvidence(reference.ref, "other note")


@pytest.mark.parametrize(
    "target,kind,limit",
    [
        ("quote", "user", 500),
        ("user_report", "user", 500),
        ("tool_result", "tool_result", 600),
        ("context", "assistant", 600),
    ],
)
async def test_each_text_claim_uses_its_kind_length_limit(target: str, kind: str, limit: int) -> None:
    source = SourceMessage("s", 1, ("x" * (limit + 1),), kind=kind, attributes=(("tool", "shell"),))  # type: ignore[arg-type]
    view = ConversationView((source,), "s")
    assert await verify(target, "x" * limit, view)
    with pytest.raises(MemoryOperationError) as caught:
        await verify(target, "x" * (limit + 1), view)
    assert caught.value.kind == "memory_invalid_input"
