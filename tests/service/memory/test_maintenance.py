# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Withdrawals, restoration and lossless merges from the step-8 specification."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from chrys.service.memory import operations, render
from chrys.service.memory.change_log import MemoryChange
from chrys.service.memory.operations import MemoryWriter, SaveRequest
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import (
    GLOBAL_SCOPE,
    Context,
    MemoryEntryDraft,
    MemoryOperationError,
    MemoryScope,
    MergeResult,
    Quote,
    ReferenceEvidence,
    ToolResultEvidence,
    UserReportEvidence,
)
from chrys.service.memory.verification import (
    ConversationView,
    QuoteClaim,
    SourceMessage,
    SourceVerificationError,
    ToolResultClaim,
    UserReportClaim,
    verify_evidence,
    verify_quotes,
)

PROJECT = MemoryScope("project", "0123456789abcdef")
AT = "2026-11-11T10:00:00Z"
LATER = "2026-11-11T11:00:00Z"
EARLIER = "2026-11-11T09:00:00Z"
RESTORE_ERROR = "This entry was withdrawn; updating it needs words the user said after withdrawing it."


@dataclass
class RecordingLog:
    changes: list[MemoryChange] = field(default_factory=list)

    async def append(self, change: MemoryChange) -> None:
        self.changes.append(change)


@pytest.fixture
def store(tmp_path: Path) -> FileMemoryStore:
    return FileMemoryStore(tmp_path / "memory")


def original(*, engineering: bool = False) -> MemoryEntryDraft:
    return MemoryEntryDraft(
        "engineering" if engineering else "intent",
        "Local storage",
        "Use the local store.",
        "model",
        quotes=() if engineering else (Quote("use SQLite", "message", "old", EARLIER, "old-item"),),
        context=Context("a proposal", "old", EARLIER, "proposal"),
        evidence=(UserReportEvidence("use SQLite", "old", EARLIER, "old-item"),) if engineering else (),
    )


def message(
    text: str,
    *,
    position: int = 10,
    session: str = "current",
    item: str | None = "new-item",
    at: str | None = LATER,
    kind: str = "user",
) -> SourceMessage:
    return SourceMessage(session, position, (text,), kind=kind, item=item, at=at)  # type: ignore[arg-type]


def claim(text: str, session: str | None = None, candidate: str | None = None) -> QuoteClaim:
    return QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", text, candidate, session)


def withdrawn(*, engineering: bool = False, basis: Quote | None = None) -> MemoryEntryDraft:
    return replace(
        original(engineering=engineering),
        status="withdrawn",
        status_basis=basis or Quote("stop using this", "message", "current", AT, "withdraw-item"),
        merged_from=("project:111111111111",),
    )


def update(ref: str, version: str, *, engineering: bool = False, words: str | None = "use SQLite") -> SaveRequest:
    return SaveRequest(
        "engineering" if engineering else "intent",
        "Local storage restored",
        original().body,
        ref=ref,
        version=version,
        user_quotes=(claim(words),) if words is not None and not engineering else None,
        evidence=(UserReportClaim(words),) if words is not None and engineering else None,
    )


async def test_withdraw_verifies_user_words_preserves_content_and_logs_the_committed_version(
    store: FileMemoryStore,
) -> None:
    created = await store.create(PROJECT, original())
    log = RecordingLog()
    writer = MemoryWriter(store, PROJECT, "extraction", change_log=log, source="extract-synthetic")
    view = ConversationView((message("stop using this", position=6, item="withdraw-item", at=AT),), "current")
    result = await writer.withdraw(created.ref, created.version, claim("stop using this"), view)
    saved = await store.get(PROJECT, created.ref)
    basis = Quote("stop using this", "message", "current", AT, "withdraw-item")
    assert (result.ref, result.version, result.title, result.basis) == (
        created.ref,
        saved.version,
        original().title,
        basis,
    )
    assert saved.entry.draft() == replace(original(), status="withdrawn", status_basis=basis)
    assert [(row.operation, row.before_version, row.after_version) for row in log.changes] == [
        ("withdraw", created.version, saved.version)
    ]
    assert log.changes[0].source == "extract-synthetic"


async def test_batch_withdraw_with_a_mistyped_session_changes_nothing(store: FileMemoryStore) -> None:
    # Only memory_save turns wrong_session into a retryable candidate; withdraw refuses outright.
    created = await store.create(PROJECT, original())
    log = RecordingLog()
    writer = MemoryWriter(store, PROJECT, "extraction", change_log=log, source="extract-synthetic")
    view = ConversationView((message("stop using this", position=6, item="withdraw-item", at=AT),), None)
    with pytest.raises(SourceVerificationError) as caught:
        await writer.withdraw(created.ref, created.version, claim("stop using this", "current-typo"), view)
    assert caught.value.kind == "memory_source_wrong_session"
    assert (await store.get(PROJECT, created.ref)).version == created.version and log.changes == []


@pytest.mark.parametrize("kind", ["assistant", "tool_result"])
async def test_withdraw_cannot_quote_another_source_kind(store: FileMemoryStore, kind: str) -> None:
    created = await store.create(PROJECT, original())
    with pytest.raises(MemoryOperationError) as caught:
        await MemoryWriter(store, PROJECT, "model").withdraw(
            created.ref,
            created.version,
            claim("stop using this"),
            ConversationView((message("stop using this", kind=kind),), "current"),
        )
    assert caught.value.kind == "memory_quote_not_found"
    assert (await store.get(PROJECT, created.ref)).version == created.version


@pytest.mark.parametrize("missing", ["time", "batch-session"])
async def test_withdraw_requires_time_and_explicit_batch_session(store: FileMemoryStore, missing: str) -> None:
    created = await store.create(PROJECT, original())
    view = ConversationView((message("stop using this", at=None if missing == "time" else AT),), None)
    with pytest.raises(MemoryOperationError) as caught:
        await MemoryWriter(store, PROJECT, "extraction").withdraw(
            created.ref, created.version, claim("stop using this", "current" if missing == "time" else None), view
        )
    assert caught.value.kind == "memory_invalid_input"
    if missing == "batch-session":
        assert caught.value.message == "Invalid quote claims: user_quotes[0].session is required in batch mode."
    assert (await store.get(PROJECT, created.ref)).version == created.version


async def test_already_withdrawn_and_stale_version_fail_without_logging(store: FileMemoryStore) -> None:
    created = await store.create(PROJECT, withdrawn())
    log = RecordingLog()
    writer = MemoryWriter(store, PROJECT, "model", change_log=log, source="current")
    view = ConversationView((message("stop using this"),), "current")
    for version, kind in (("stale", "memory_version_conflict"), (created.version, "memory_invalid_input")):
        with pytest.raises(MemoryOperationError) as caught:
            await writer.withdraw(created.ref, version, claim("stop using this"), view)
        assert caught.value.kind == kind
    assert log.changes == []


@pytest.mark.parametrize("engineering", [False, True], ids=["intent", "engineering"])
async def test_restoration_rechecks_repeated_words_before_retention_and_logs_restore(
    store: FileMemoryStore, engineering: bool
) -> None:
    draft = withdrawn(engineering=engineering)
    created = await store.create(PROJECT, draft)
    log = RecordingLog()
    writer = MemoryWriter(store, PROJECT, "extraction", change_log=log, source="extract-synthetic")
    view = ConversationView((message("use SQLite"),), "current")
    request = update(created.ref, created.version, engineering=engineering)
    if engineering:
        request = replace(request, evidence=(UserReportClaim("use SQLite", session="old"),))
    else:
        request = replace(request, user_quotes=(claim("use SQLite", "old"),))
    result = await writer.save(request, view)
    saved = await store.get(PROJECT, created.ref)
    assert result.restored is True
    assert result.entry_status is None
    assert (saved.entry.status, saved.entry.status_basis) == (None, None)
    assert saved.entry.merged_from == draft.merged_from
    if engineering:
        assert saved.entry.evidence == (UserReportEvidence("use SQLite", "current", LATER, "new-item"),)
    else:
        assert saved.entry.quotes == (Quote("use SQLite", "message", "current", LATER, "new-item"),)
    assert [(row.operation, row.before_version, row.after_version) for row in log.changes] == [
        ("restore", created.version, saved.version)
    ]


@pytest.mark.parametrize("shown", ["keep using it", "use SQLite; keep using it"], ids=["distinct", "repeated"])
@pytest.mark.parametrize("engineering", [False, True], ids=["intent", "engineering"])
async def test_restore_retains_out_of_view_old_sources_but_only_new_occurrences_authorize_it(
    store: FileMemoryStore, engineering: bool, shown: str
) -> None:
    created = await store.create(PROJECT, withdrawn(engineering=engineering))
    request = update(created.ref, created.version, engineering=engineering)
    if engineering:
        request = replace(
            request,
            evidence=(
                UserReportClaim("use SQLite", session="old"),
                UserReportClaim("keep using it", session="current"),
            ),
        )
    else:
        request = replace(request, user_quotes=(claim("use SQLite", "old"), claim("keep using it", "current")))
    # "repeated": a continued session in view repeats the old words, which must not void the old source.
    result = await MemoryWriter(store, PROJECT, "model").save(request, ConversationView((message(shown),), None))
    assert result.restored is True
    saved = (await store.get(PROJECT, created.ref)).entry
    if engineering:
        assert saved.evidence[0] == original(engineering=True).evidence[0]
    else:
        assert saved.quotes[0] == original().quotes[0]


@pytest.mark.parametrize("engineering", [False, True], ids=["intent", "engineering"])
@pytest.mark.parametrize("case", ["title-only", "retained-only", "before", "same-time", "no-time"])
async def test_restore_rejects_without_new_words_after_withdrawal(
    store: FileMemoryStore, engineering: bool, case: str
) -> None:
    created = await store.create(PROJECT, withdrawn(engineering=engineering))
    request = update(
        created.ref, created.version, engineering=engineering, words=None if case == "title-only" else "use SQLite"
    )
    if case == "retained-only":
        request = replace(
            request,
            user_quotes=None if engineering else (claim("use SQLite", "old"),),
            evidence=(UserReportClaim("use SQLite", session="old"),) if engineering else None,
        )
    texts = (
        ()
        if case in ("title-only", "retained-only")
        else (message("use SQLite", at={"before": EARLIER, "same-time": AT, "no-time": None}[case]),)
    )
    with pytest.raises(MemoryOperationError) as caught:
        await MemoryWriter(store, PROJECT, "model").save(request, ConversationView(texts, "current"))
    assert caught.value.kind == "memory_invalid_input"
    assert caught.value.message == RESTORE_ERROR
    assert (await store.get(PROJECT, created.ref)).version == created.version


@pytest.mark.parametrize("new_position,expected", [(4, False), (6, True)])
async def test_same_session_unique_withdraw_item_uses_position_even_when_clocks_disagree(
    store: FileMemoryStore, new_position: int, expected: bool
) -> None:
    created = await store.create(PROJECT, withdrawn())
    view = ConversationView(
        (
            message("stop using this", position=5, item="withdraw-item", at=AT),
            message("use SQLite", position=new_position, at=None if expected else LATER),
        ),
        "current",
    )
    writer = MemoryWriter(store, PROJECT, "model")
    if expected:
        assert (await writer.save(update(created.ref, created.version), view)).restored is True
    else:
        with pytest.raises(MemoryOperationError) as caught:
            await writer.save(update(created.ref, created.version), view)
        assert caught.value.message == RESTORE_ERROR


@pytest.mark.parametrize("item", [None, "withdraw-item"])
async def test_absent_or_unlocatable_withdraw_item_uses_parsed_datetime(
    store: FileMemoryStore, item: str | None
) -> None:
    basis = Quote("stop using this", "message", "current", AT, item)
    created = await store.create(PROJECT, withdrawn(basis=basis))
    view = ConversationView((message("use SQLite", position=1, at="2026-11-11T10:00:00.001Z"),), "current")
    result = await MemoryWriter(store, PROJECT, "model").save(update(created.ref, created.version), view)
    assert result.restored is True


@pytest.mark.parametrize("engineering", [False, True], ids=["intent", "engineering"])
@pytest.mark.parametrize("failure", ["ambiguous", "candidate_expired"])
async def test_restore_does_not_hide_ambiguity_or_expired_candidates_with_retention(
    store: FileMemoryStore, engineering: bool, failure: str
) -> None:
    created = await store.create(PROJECT, withdrawn(engineering=engineering))
    messages = (message("use SQLite", item="one"), message("use SQLite", item="two", position=11))
    request = update(created.ref, created.version, engineering=engineering)
    candidate = "q-no-longer-valid" if failure == "candidate_expired" else None
    request = replace(
        request,
        user_quotes=(claim("use SQLite", "old", candidate),) if not engineering else None,
        evidence=(UserReportClaim("use SQLite", candidate=candidate, session="old"),) if engineering else None,
    )
    with pytest.raises(SourceVerificationError) as caught:
        await MemoryWriter(store, PROJECT, "model").save(request, ConversationView(messages, "current"))
    assert caught.value.failures[0].reason == failure
    assert (await store.get(PROJECT, created.ref)).version == created.version


async def test_restore_candidate_selects_the_new_repeated_occurrence(store: FileMemoryStore) -> None:
    created = await store.create(PROJECT, withdrawn())
    view = ConversationView(
        (message("use SQLite", item="old", position=1, at=EARLIER), message("use SQLite", position=10)), "current"
    )
    with pytest.raises(SourceVerificationError) as ambiguous:
        verify_quotes((claim("use SQLite"),), view)
    token = ambiguous.value.failures[0].candidates[1].token
    result = await MemoryWriter(store, PROJECT, "model").save(
        replace(update(created.ref, created.version), user_quotes=(claim("use SQLite", "old", token),)), view
    )
    assert result.restored is True
    assert result.quotes[0].item == "new-item"


async def test_duplicate_withdrawn_create_reports_status_without_restoring_or_logging(store: FileMemoryStore) -> None:
    created = await store.create(PROJECT, withdrawn())
    log = RecordingLog()
    result = await MemoryWriter(store, PROJECT, "model", change_log=log, source="current").save(
        SaveRequest(
            "intent",
            original().title,
            original().body,
            user_quotes=(claim("use SQLite", "old"),),
            context_quote="a proposal",
            context_session="old",
        ),
        ConversationView(
            (
                message("use SQLite", session="old", item="old-item", at=EARLIER),
                message("a proposal", session="old", item="proposal", at=EARLIER, kind="assistant", position=11),
            ),
            "old",
        ),
    )
    assert (result.ref, result.status, result.entry_status, result.restored) == (
        created.ref,
        "existing",
        "withdrawn",
        False,
    )
    assert "Status: withdrawn" in render.render_saved(result)
    assert "new words" in render.render_saved(result)
    assert log.changes == []


async def test_read_displays_withdrawal_basis_without_changing_catalog_header(store: FileMemoryStore) -> None:
    created = await store.create(PROJECT, withdrawn())
    text = render.render_entry(await store.get(PROJECT, created.ref))
    assert "Status: withdrawn" in text
    assert 'Withdrawal basis: "stop using this"' in text
    assert "current" in text and AT in text


async def test_merge_is_lossless_full_payload_union_without_a_source_view(store: FileMemoryStore) -> None:
    first = original()
    second = replace(first, title="Duplicate", quotes=(replace(first.quotes[0], at=AT),))
    keep = await store.create(PROJECT, first)
    drop = await store.create(PROJECT, second)
    log = RecordingLog()
    result = await MemoryWriter(store, PROJECT, "extraction", change_log=log, source="extract-synthetic").merge(
        keep.ref, keep.version, drop.ref, drop.version
    )
    saved = await store.get(PROJECT, keep.ref)
    assert (result.ref, result.version, result.title, result.drop_ref) == (
        keep.ref,
        saved.version,
        first.title,
        drop.ref,
    )
    assert (saved.entry.title, saved.entry.body, saved.entry.applies_when) == (
        first.title,
        first.body,
        first.applies_when,
    )
    assert saved.entry.quotes == (*first.quotes, *second.quotes)
    assert saved.entry.context == first.context
    assert saved.entry.merged_from == (drop.ref,)
    assert [(row.operation, row.ref, row.before_version, row.after_version) for row in log.changes] == [
        ("merge", keep.ref, keep.version, saved.version),
        ("delete", drop.ref, drop.version, None),
    ]
    assert all(row.source == "extract-synthetic" for row in log.changes)
    assert len(await store.list_entries(PROJECT)) == 1
    assert render.render_merged(result) == f"Merged {drop.ref} into {keep.ref} (version {saved.version})."


@pytest.mark.parametrize(
    "case",
    [
        "same-ref",
        "different-scope",
        "different-type",
        "withdrawn-keep",
        "withdrawn-drop",
        "context",
        "quotes",
        "evidence",
    ],
)
async def test_merge_rejects_invalid_pairs_before_writing(store: FileMemoryStore, case: str) -> None:
    first = original(engineering=case == "evidence")
    second = replace(first, title="Duplicate")
    if case == "different-type":
        second = original(engineering=True)
    if case == "context":
        second = replace(second, context=Context("different proposal"))
    if case == "quotes":
        first = replace(first, quotes=tuple(Quote(str(i), "message", "old") for i in range(3)))
        second = replace(second, quotes=(Quote("another", "message", "old"),))
    if case == "evidence":
        first = replace(first, evidence=tuple(ReferenceEvidence(str(i)) for i in range(6)))
        second = replace(second, evidence=(ReferenceEvidence("another"),))
    if case == "withdrawn-keep":
        first = replace(first, status="withdrawn", status_basis=withdrawn().status_basis)
    if case == "withdrawn-drop":
        second = replace(second, status="withdrawn", status_basis=withdrawn().status_basis)
    keep = await store.create(PROJECT, first)
    drop_scope = GLOBAL_SCOPE if case == "different-scope" else PROJECT
    drop = await store.create(drop_scope, second)
    drop_ref, drop_version = (keep.ref, keep.version) if case == "same-ref" else (drop.ref, drop.version)
    with pytest.raises(MemoryOperationError) as caught:
        await MemoryWriter(store, PROJECT, "extraction").merge(keep.ref, keep.version, drop_ref, drop_version)
    assert caught.value.kind == "memory_invalid_input"
    assert (await store.get(PROJECT, keep.ref)).version == keep.version
    assert (await store.get(drop_scope, drop.ref)).version == drop.version


async def test_engineering_merge_keeps_distinct_tool_payloads_and_reference_notes(store: FileMemoryStore) -> None:
    tool = ToolResultEvidence("Shell", "{}", "observed", "old", AT, "result")
    first = replace(original(engineering=True), evidence=(tool, ReferenceEvidence("doc", "first")))
    second = replace(
        first, title="Duplicate", evidence=(replace(tool, args='{"mode":2}'), ReferenceEvidence("doc", "second"))
    )
    keep = await store.create(PROJECT, first)
    drop = await store.create(PROJECT, second)
    result = await MemoryWriter(store, PROJECT, "extraction").merge(keep.ref, keep.version, drop.ref, drop.version)
    saved = await store.get(PROJECT, result.ref)
    assert saved.entry.evidence == (*first.evidence, *second.evidence)


class PartialMergeStore(FileMemoryStore):
    async def merge(
        self,
        scope: MemoryScope,
        keep_ref: str,
        keep_version: str,
        drop_ref: str,
        drop_version: str,
        keep_draft: MemoryEntryDraft,
    ) -> MergeResult:
        version = await self.update(scope, keep_ref, keep_version, keep_draft)
        return MergeResult(version, deleted=False)


async def test_partial_merge_reports_the_actual_kept_version_and_logs_only_the_commit(tmp_path: Path) -> None:
    store = PartialMergeStore(tmp_path / "partial-memory")
    keep = await store.create(PROJECT, original())
    drop = await store.create(PROJECT, replace(original(), title="Duplicate"))
    log = RecordingLog()
    writer = MemoryWriter(store, PROJECT, "extraction", change_log=log, source="extract-synthetic")
    with pytest.raises(operations.MergePartialError) as caught:
        await writer.merge(keep.ref, keep.version, drop.ref, drop.version)
    saved = await store.get(PROJECT, keep.ref)
    assert (caught.value.kind, caught.value.ref, caught.value.version, caught.value.drop_ref) == (
        "merge_partial",
        keep.ref,
        saved.version,
        drop.ref,
    )
    assert saved.version != keep.version
    assert saved.version in caught.value.message
    assert (await store.get(PROJECT, drop.ref)).version == drop.version
    assert [(row.operation, row.ref, row.after_version) for row in log.changes] == [("merge", keep.ref, saved.version)]


async def test_active_update_preserves_merge_lineage(store: FileMemoryStore) -> None:
    draft = replace(original(), merged_from=("project:111111111111",))
    created = await store.create(PROJECT, draft)
    await MemoryWriter(store, PROJECT, "model").save(
        update(created.ref, created.version, words=None), ConversationView((), "current")
    )
    assert (await store.get(PROJECT, created.ref)).entry.merged_from == draft.merged_from


async def test_restore_across_sessions_uses_time_and_never_cross_compares_line_numbers(store: FileMemoryStore) -> None:
    created = await store.create(PROJECT, withdrawn())
    view = ConversationView((message("use SQLite", position=1, session="next", at=LATER),), "next")
    result = await MemoryWriter(store, PROJECT, "model").save(update(created.ref, created.version), view)
    assert result.restored is True


async def test_restore_fallback_does_not_count_a_retained_later_timestamp_as_a_new_occurrence(
    store: FileMemoryStore,
) -> None:
    draft = replace(withdrawn(), quotes=(Quote("use SQLite", "message", "elsewhere", LATER, "stored"),))
    created = await store.create(PROJECT, draft)
    request = replace(update(created.ref, created.version), user_quotes=(claim("use SQLite", "elsewhere"),))
    with pytest.raises(MemoryOperationError) as caught:
        await MemoryWriter(store, PROJECT, "model").save(request, ConversationView((), "current"))
    assert caught.value.message == RESTORE_ERROR


async def test_restore_with_no_item_still_allows_same_words_to_be_a_new_substantive_basis(
    store: FileMemoryStore,
) -> None:
    draft = replace(withdrawn(), quotes=(Quote("use SQLite", "message", "current", EARLIER),))
    created = await store.create(PROJECT, draft)
    request = replace(
        update(created.ref, created.version),
        body="Use SQLite everywhere.",
        user_quotes=(claim("use SQLite", "current"),),
    )
    result = await MemoryWriter(store, PROJECT, "model").save(
        request, ConversationView((message("use SQLite", item=None),), "current")
    )
    assert result.restored is True


async def test_merge_full_payload_deduplicates_both_sides_and_carries_single_context(store: FileMemoryStore) -> None:
    quote = original().quotes[0]
    keep_draft = replace(original(), quotes=(quote, quote), context=None, merged_from=("project:111111111111",))
    drop_draft = replace(original(), title="Duplicate", quotes=(quote,))
    keep = await store.create(PROJECT, keep_draft)
    drop = await store.create(PROJECT, drop_draft)
    await MemoryWriter(store, PROJECT, "extraction").merge(keep.ref, keep.version, drop.ref, drop.version)
    entry = (await store.get(PROJECT, keep.ref)).entry
    assert entry.quotes == (quote,)
    assert entry.context == drop_draft.context
    assert entry.merged_from == ("project:111111111111", drop.ref)


async def test_withdrawn_render_stays_within_read_budget_and_preserves_status(store: FileMemoryStore) -> None:
    basis = Quote('"' * 500, "message", "current", AT, "withdraw-item")
    draft = replace(withdrawn(basis=basis), quotes=(Quote("original", "message", "old"),))
    created = await store.create(PROJECT, draft)
    from chrys.foundation.text.tokenizer import TOKEN_UNITS_PER_TOKEN, MixedLanguageTokenizer

    stored = await store.get(PROJECT, created.ref)
    value = replace(stored, entry=replace(stored.entry, status_basis=replace(basis, session="s" * 20000)))
    text = render.render_entry(value)
    assert "Status: withdrawn" in text
    assert "Output truncated" in text
    assert MixedLanguageTokenizer().count_token_units(text) <= 4000 * TOKEN_UNITS_PER_TOKEN


@pytest.mark.parametrize("source_kind", ["quote", "tool_result", "user_report"])
@pytest.mark.parametrize("restoring", [False, True], ids=["explicit-update", "restore-with-new-words"])
async def test_merged_full_payload_sources_survive_later_explicit_retention(
    store: FileMemoryStore, source_kind: str, restoring: bool
) -> None:
    engineering = source_kind != "quote"
    if source_kind == "quote":
        first_source = Quote("same words", "remember", "old", EARLIER, "first")
        second_source = Quote("same words", "message", "old", AT, "second")
        first = replace(original(), quotes=(first_source,))
        second = replace(first, title="Duplicate", quotes=(second_source,))
    elif source_kind == "tool_result":
        first_source = ToolResultEvidence("Shell", "{}", "same words", "old", EARLIER, "first")
        second_source = ToolResultEvidence("Read", '{"path":"file"}', "same words", "old", AT, "second")
        first = replace(original(engineering=True), evidence=(first_source,))
        second = replace(first, title="Duplicate", evidence=(second_source,))
    else:
        first_source = UserReportEvidence("same words", "old", EARLIER, "first")
        second_source = UserReportEvidence("same words", "old", AT, "second")
        first = replace(original(engineering=True), evidence=(first_source,))
        second = replace(first, title="Duplicate", evidence=(second_source,))
    keep = await store.create(PROJECT, first)
    drop = await store.create(PROJECT, second)
    writer = MemoryWriter(store, PROJECT, "extraction")
    merged = await writer.merge(keep.ref, keep.version, drop.ref, drop.version)
    version = merged.version
    if restoring:
        withdrawn_result = await writer.withdraw(
            keep.ref,
            version,
            claim("stop using this"),
            ConversationView((message("stop using this", item="withdraw-item", at=AT),), "current"),
        )
        version = withdrawn_result.version
    if source_kind == "quote":
        quotes = (claim("same words", "old"), claim("same words", "old"))
        if restoring:
            quotes = (*quotes, claim("still valid", "current"))
        request = replace(update(keep.ref, version), user_quotes=quotes)
    else:
        evidence = (
            (ToolResultClaim("same words", session="old"),) * 2
            if source_kind == "tool_result"
            else (UserReportClaim("same words", session="old"),) * 2
        )
        if restoring:
            evidence = (*evidence, UserReportClaim("still valid", session="current"))
        request = replace(update(keep.ref, version, engineering=True), evidence=evidence)
    view = ConversationView((message("still valid"),) if restoring else (), "current")
    result = await writer.save(request, view)
    saved = (await store.get(PROJECT, keep.ref)).entry
    assert result.restored is restoring
    retained = saved.evidence if engineering else saved.quotes
    assert retained[:2] == (first_source, second_source)
    assert len(retained) == (3 if restoring else 2)


@pytest.mark.parametrize("source_kind", ["quote", "tool_result", "user_report"])
async def test_retention_consumes_distinct_stored_occurrences_before_reusing_the_first(source_kind: str) -> None:
    if source_kind == "quote":
        first = Quote("same words", "remember", "old", EARLIER, "first")
        second = Quote("same words", "message", "old", AT, "second")
        result = verify_quotes(
            (claim("same words", "old"),) * 3, ConversationView((), "current"), stored=(first, second)
        )
    elif source_kind == "tool_result":
        first = ToolResultEvidence("Shell", "{}", "same words", "old", EARLIER, "first")
        second = ToolResultEvidence("Read", "{}", "same words", "old", AT, "second")
        result = await verify_evidence(
            (ToolResultClaim("same words", session="old"),) * 3,
            ConversationView((), "current"),
            stored=(first, second),
        )
    else:
        first = UserReportEvidence("same words", "old", EARLIER, "first")
        second = UserReportEvidence("same words", "old", AT, "second")
        result = await verify_evidence(
            (UserReportClaim("same words", session="old"),) * 3,
            ConversationView((), "current"),
            stored=(first, second),
        )
    assert result == (first, second, first)
