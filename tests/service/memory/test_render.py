# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Catalog and tool-result text, budgets and round-trip guarantees from sections 5 through 7."""

# ruff: noqa: RUF001  # Full-width source text is intentional.

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from chrys.foundation.text.tokenizer import TOKEN_UNITS_PER_TOKEN, MixedLanguageTokenizer
from chrys.service.memory.operations import (
    CatalogSnapshot,
    DeleteOutcome,
    MemoryWriter,
    SaveOutcome,
    SaveRequest,
    SearchPage,
)
from chrys.service.memory.project import bind_project
from chrys.service.memory.render import (
    CATALOG_TOKEN_BUDGET,
    WITHDRAWN_CATALOG,
    render_catalog,
    render_deleted,
    render_entry,
    render_error,
    render_saved,
    render_search,
)
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import (
    CatalogItem,
    Context,
    FileEvidence,
    MemoryEntry,
    MemoryOperationError,
    MemoryScope,
    Quote,
    ReferenceEvidence,
    StoredMemory,
    ToolResultEvidence,
    UserReportEvidence,
)
from chrys.service.memory.verification import (
    ConversationView,
    QuoteClaim,
    SourceMessage,
    SourceVerificationError,
    verify_quotes,
)

REF = "project:c08d51f3a9e2"
VERSION = "7a1c03e9d4b25f60"
AT = "2026-11-02T09:19:48Z"
SESSION = "5e23e8ff-69c9-4430-9122-8f8e09edf317"
WORDS = "就用方案 B，别引入 Redis，运维不允许加新组件。"
TITLE = "本地队列用 SQLite WAL，不引入 Redis"
APPLIES = "任务队列、缓存或新增基础组件的选型"
BODY = "任务队列用 SQLite WAL。"
TRUNCATED = "Output truncated; the user can view the full entry in the stored memory file."


def stored() -> StoredMemory:
    return StoredMemory(
        MemoryEntry(
            "c08d51f3a9e2",
            "project",
            "2026-11-02T09:20:00Z",
            "2026-11-02T09:20:00Z",
            "intent",
            TITLE,
            BODY,
            "model",
            APPLIES,
            (Quote(WORDS, "message", SESSION, AT),),
        ),
        VERSION,
    )


def outcome(status: str = "created") -> SaveOutcome:
    return SaveOutcome(REF, VERSION, status, "intent", TITLE, APPLIES, stored().entry.quotes)  # type: ignore[arg-type]


def quote_json_from_read(text: str) -> str:
    line = next(line for line in text.splitlines() if line.startswith("1. "))
    value, _ = json.JSONDecoder().raw_decode(line[3:])
    assert isinstance(value, str)
    return value


@pytest.mark.parametrize(
    "status,first",
    [
        ("created", f"Saved {REF} (intent, version {VERSION})."),
        ("existing", f"Already saved as {REF} (intent, version {VERSION}); nothing was written."),
        ("updated", f"Updated {REF} (intent, version {VERSION})."),
    ],
)
def test_saved_text_matches_specification(status: str, first: str) -> None:
    expected = (
        f"{first}\nTitle: {TITLE}\nApplies when: {APPLIES}\nUser's words: {json.dumps(WORDS, ensure_ascii=False)}"
    )
    if status != "existing":
        expected += "\nIt appears in the memory catalog from the next turn; memory_read and memory_search see it now."
    assert render_saved(outcome(status)) == expected


def test_saved_omits_absent_applies_and_preserves_quote_json() -> None:
    text = 'say "yes"\nthen continue'
    result = replace(outcome(), applies_when=None, quotes=(Quote(text, "message", SESSION, AT),))
    rendered = render_saved(result)
    assert "Applies when:" not in rendered
    json_text = next(
        line.removeprefix("User's words: ") for line in rendered.splitlines() if line.startswith("User's words: ")
    )
    assert json.loads(json_text) == text


def test_read_text_matches_specification() -> None:
    expected = (
        f"{REF} · intent · version {VERSION}\n"
        f"Title: {TITLE}\nApplies when: {APPLIES}\nSaved by the model on 2026-11-02.\n\n"
        f"{BODY}\n\nUser's words:\n1. {json.dumps(WORDS, ensure_ascii=False)} (session {SESSION}, via message, {AT})"
    )
    assert render_entry(stored()) == expected


@pytest.mark.parametrize("writer,label", [("model", "the model"), ("user", "the user"), ("extraction", "extraction")])
def test_read_shows_all_three_writer_labels(writer: str, label: str) -> None:
    value = stored()
    text = render_entry(replace(value, entry=replace(value.entry, written_by=writer)))  # type: ignore[arg-type]
    assert f"Saved by {label} on 2026-11-02." in text
    assert "Basis:" not in text


def test_read_shows_every_quote_session_verbatim_and_no_session() -> None:
    value = stored()
    quotes = (
        Quote("first", "message", "full/session:a-b", AT),
        Quote("second", "remember", "another-session", AT),
        Quote("cli", "cli", at=AT),
    )
    text = render_entry(replace(value, entry=replace(value.entry, quotes=quotes)))
    assert f'1. "first" (session full/session:a-b, via message, {AT})' in text
    assert f'2. "second" (session another-session, via remember, {AT})' in text
    assert f'3. "cli" (no session, via cli, {AT})' in text


async def test_read_json_quotes_decode_and_round_trip_without_new_verification(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    scope = MemoryScope("project", "0123456789abcdef")
    words = 'say "yes"\nthen continue'
    applies = "first line\nsecond line"
    view = ConversationView((SourceMessage(SESSION, 3, (words,), at=AT),), SESSION)
    writer = MemoryWriter(store, scope, "model")
    saved = await writer.save(
        SaveRequest(
            "intent",
            "Original title",
            "body",
            applies,
            user_quotes=(QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", words),),
        ),
        view,
    )
    read = render_entry(await store.get(scope, saved.ref))
    decoded = quote_json_from_read(read)
    assert decoded == words
    assert f"Applies when: {applies}\n" in read
    updated = await writer.save(
        SaveRequest(
            "intent",
            "Changed title",
            "body",
            applies,
            user_quotes=(QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", decoded, session=SESSION),),
            ref=saved.ref,
            version=saved.version,
        ),
        ConversationView((), "new"),
    )
    assert updated.quotes == (Quote(words, "message", SESSION, AT),)
    title_only = await writer.save(
        SaveRequest("intent", "Title again", "body", applies, ref=saved.ref, version=updated.version),
        ConversationView((), "new"),
    )
    assert title_only.applies_when == applies


async def test_saved_and_updated_full_versions_can_be_reused_directly(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    scope = MemoryScope("project", "0123456789abcdef")
    view = ConversationView((SourceMessage("s", 0, ("keep it",)),), "s")
    writer = MemoryWriter(store, scope, "model")
    saved = await writer.save(
        SaveRequest(
            "intent",
            "Title",
            "body",
            user_quotes=(QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", "keep it"),),
        ),
        view,
    )
    first_version = re.search(r"version ([0-9a-f]{16})", render_saved(saved))
    assert first_version is not None
    updated = await writer.save(
        SaveRequest("intent", "New title", "body", ref=saved.ref, version=first_version.group(1)), view
    )
    next_version = re.search(r"version ([0-9a-f]{16})", render_saved(updated))
    assert next_version is not None
    assert next_version.group(1) == updated.version
    final = await writer.save(
        SaveRequest("intent", "Last title", "body", ref=saved.ref, version=next_version.group(1)), view
    )
    assert final.status == "updated"


def test_read_budget_preserves_title_applies_until_body_and_quotes_are_reduced() -> None:
    value = stored()
    applies = "applicable\n" + "场" * 189
    entry = replace(
        value.entry,
        title="title\nunchanged",
        applies_when=applies,
        body="文" * 18000,
        quotes=(Quote("话" * 1200, "remember", SESSION, AT),),
    )
    text = render_entry(replace(value, entry=entry))
    assert text.startswith(f"{REF} · intent · version {VERSION}\n")
    assert f"Title: {entry.title}\n" in text
    assert f"Applies when: {applies}\n" in text
    assert "…[truncated]" in text
    assert text.endswith(TRUNCATED)
    assert "management" not in text.lower()
    assert MixedLanguageTokenizer().count_token_units(text) <= 4000 * TOKEN_UNITS_PER_TOKEN


@pytest.mark.parametrize("field", ["reference", "file", "tool", "args", "context", "session", "title", "applies"])
def test_read_budget_covers_every_unbounded_display_field(field: str) -> None:
    value = stored()
    huge = "界" * 16000
    if field == "reference":
        entry = replace(value.entry, evidence=(ReferenceEvidence(huge, huge),))
    elif field == "file":
        entry = replace(value.entry, evidence=(FileEvidence("0123456789abcdef", huge, None, 1),))
    elif field in ("tool", "args"):
        evidence = ToolResultEvidence(
            huge if field == "tool" else "shell", huge if field == "args" else "args", "excerpt", SESSION, AT
        )
        entry = replace(value.entry, evidence=(evidence,))
    elif field == "context":
        entry = replace(value.entry, context=Context(huge, SESSION, AT))
    elif field == "session":
        entry = replace(value.entry, quotes=(Quote("words", "message", huge, AT),))
    elif field == "title":
        entry = replace(value.entry, title=huge)
    else:
        entry = replace(value.entry, applies_when=huge)
    text = render_entry(replace(value, entry=entry))
    assert text.startswith(f"{REF} · intent · version {VERSION}\n")
    assert MixedLanguageTokenizer().count_token_units(text) <= 4000 * TOKEN_UNITS_PER_TOKEN
    assert text.endswith(TRUNCATED)


def test_read_lists_context_and_all_evidence_kinds_without_basis() -> None:
    value = stored()
    entry = replace(
        value.entry,
        context=Context("agreed proposal", SESSION, AT),
        evidence=(
            ToolResultEvidence("shell", "uv sync --offline", "network disabled", SESSION, AT),
            FileEvidence("0123456789abcdef", "uv.lock", "a" * 64, 42),
            ReferenceEvidence("https://example.com/doc", "documentation"),
            UserReportEvidence("reported result", SESSION, AT),
        ),
    )
    text = render_entry(replace(value, entry=entry))
    for phrase in (
        "agreed proposal",
        "shell",
        "uv sync --offline",
        "network disabled",
        "uv.lock",
        "https://example.com/doc",
        "documentation",
        "reported result",
    ):
        assert phrase in text
    assert "Evidence:" in text
    assert "Basis:" not in text


def test_read_truncates_sources_from_the_end_before_earlier_fields() -> None:
    value = stored()
    entry = replace(
        value.entry,
        context=Context("keep context", SESSION, AT),
        evidence=(ReferenceEvidence("keep first"), ReferenceEvidence("界" * 16000)),
    )
    text = render_entry(replace(value, entry=entry))
    assert "keep first" in text
    assert "keep context" in text
    assert WORDS in text
    assert BODY in text
    assert f"Title: {TITLE}" in text
    assert f"Applies when: {APPLIES}" in text


def test_search_text_matches_trial_format_without_page_numbers() -> None:
    value = stored()
    expected = f'1 matches for "SQLite":\n- {REF} intent — {TITLE} · {TITLE}\nNo more results.'
    assert render_search(SearchPage("SQLite", (value,), None)) == expected
    assert "page " not in render_search(SearchPage("SQLite", (value,), "10")).lower()
    assert render_search(SearchPage("nothing", (), None)) == 'No matches for "nothing".'


def test_search_opaque_next_cursor_is_json_round_trippable() -> None:
    cursor = 'next:"quoted"\nline'
    text = render_search(SearchPage("SQLite", (stored(),), cursor))
    last = text.splitlines()[-1]
    assert last.startswith("More results: call memory_search with cursor ")
    assert json.loads(last.removeprefix("More results: call memory_search with cursor ").removesuffix(".")) == cursor


@pytest.mark.parametrize(
    "text,query,needle",
    [
        ("before ＳＱＬｉｔｅ after", "sqlite", "ＳＱＬｉｔｅ"),
        ("before SQLiTe after", "sqlite", "SQLiTe"),
        ("before Straße after", "STRASSE", "Straße"),
    ],
)
def test_search_snippet_maps_normalized_matches_back_to_original(text: str, query: str, needle: str) -> None:
    value = stored()
    entry = replace(
        value.entry, title="unrelated title", applies_when=None, body="prefix " * 20 + text + " suffix" * 20
    )
    rendered = render_search(SearchPage(query, (replace(value, entry=entry),), None))
    snippet = rendered.split(" · ", 1)[1].splitlines()[0]
    assert needle in snippet
    assert not snippet.startswith("prefix prefix")
    assert len(snippet) <= 100


def test_search_uses_only_first_eight_terms_and_falls_back_to_body() -> None:
    value = stored()
    entry = replace(value.entry, title="ninthmatch", applies_when=None, body="fallback body\nstart " + "x" * 80)
    rendered = render_search(
        SearchPage("one two three four five six seven eight ninthmatch", (replace(value, entry=entry),), None)
    )
    snippet = rendered.split(" · ", 1)[1].splitlines()[0]
    assert snippet.startswith("fallback body start")
    assert "ninthmatch" not in snippet
    assert "fallback body start" in render_search(SearchPage("", (replace(value, entry=entry),), None))


@pytest.mark.parametrize("field", ["title", "applies", "body", "quote", "context", "evidence"])
def test_search_snippet_uses_first_matching_field_in_store_order(field: str) -> None:
    value = stored()
    fields = ("title", "applies", "body", "quote", "context", "evidence")
    marked = {
        name: f"{name} marker" if fields.index(name) >= fields.index(field) else f"{name} unrelated" for name in fields
    }
    entry = replace(
        value.entry,
        title=marked["title"],
        applies_when=marked["applies"],
        body=marked["body"],
        quotes=(Quote(marked["quote"], "message", SESSION, AT),),
        context=Context(marked["context"], SESSION, AT),
        evidence=(ReferenceEvidence(marked["evidence"]),),
    )
    rendered = render_search(SearchPage("marker", (replace(value, entry=entry),), None))
    assert rendered.split(" · ", 1)[1].startswith(f"{field} marker")


def test_search_unmappable_normalization_falls_back_to_field_beginning() -> None:
    value = stored()
    source = "field beginning " + "x" * 100 + " e\u0301 ending"
    entry = replace(value.entry, title="unrelated", applies_when=None, body=source)
    text = render_search(SearchPage("é", (replace(value, entry=entry),), None))
    assert text.split(" · ", 1)[1].startswith("field beginning")


def test_search_folds_only_newlines_in_title_and_snippet() -> None:
    value = stored()
    entry = replace(
        value.entry, title="first\nsecond\u2028third", body="before\nneedle\r\nafter\u0085last", applies_when=None
    )
    text = render_search(SearchPage("needle", (replace(value, entry=entry),), None))
    assert "intent — first second third · before needle after last" in text


def test_deleted_text_matches_specification() -> None:
    assert (
        render_deleted(DeleteOutcome("global:9b7e04c1d2aa", "回答用中文、简洁、不复述问题"))
        == "Deleted global:9b7e04c1d2aa (回答用中文、简洁、不复述问题)."
    )


def test_plain_errors_preserve_safe_messages_without_paths() -> None:
    message = "The memory store could not be read."
    assert render_error(MemoryOperationError("memory_store_unavailable", message), "memory_save") == message
    assert str(Path.cwd()) not in render_error(MemoryOperationError("memory_store_unavailable", message), "memory_save")
    invalid_update = "Changing the entry needs new verified evidence."
    assert render_error(MemoryOperationError("memory_invalid_input", invalid_update), "memory_save") == invalid_update


def test_project_binding_failure_renders_fixed_message_without_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    with pytest.raises(MemoryOperationError) as caught:
        bind_project(str(tmp_path))
    rendered = render_error(caught.value, "memory_save")
    assert rendered == caught.value.message
    assert str(tmp_path) not in rendered


def test_read_exact_budget_then_one_character_over() -> None:
    value = stored()
    entry = replace(
        value.entry, title="Title", applies_when=None, body="", quotes=(Quote("words", "message", "s", AT),)
    )
    prefix = f"{REF} · intent · version {VERSION}\nTitle: Title\nSaved by the model on 2026-11-02.\n\n"
    suffix = f'\n\nUser\'s words:\n1. "words" (session s, via message, {AT})'
    tokenizer = MixedLanguageTokenizer()
    budget = 4000 * TOKEN_UNITS_PER_TOKEN
    padding = (budget - tokenizer.count_token_units(prefix + suffix)) // 25
    entry = replace(entry, body="x" * padding)
    expected = prefix + entry.body + suffix
    assert tokenizer.count_token_units(expected) == budget
    assert render_entry(replace(value, entry=entry)) == expected
    over = render_entry(replace(value, entry=replace(entry, body=entry.body + "x")))
    assert tokenizer.count_token_units(over) <= budget
    assert over.endswith(TRUNCATED)


def test_verification_error_quotes_json_and_includes_every_failure() -> None:
    text = 'missing\n"quote"'
    view = ConversationView((SourceMessage("s", 1, ("yes",), at=AT), SourceMessage("s", 2, ("yes again",), at=AT)), "s")
    claims = (
        QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", text),
        QuoteClaim("user_quotes[1]", "user_quotes[1].candidate", "yes"),
    )
    with pytest.raises(SourceVerificationError) as caught:
        verify_quotes(claims, view)
    rendered = render_error(caught.value, "memory_save")
    assert rendered.startswith(
        "Some quotes could not be verified. Quotes must be copied verbatim from one user message; no edits, no joining."
    )
    assert f"- user_quotes[0] {json.dumps(text, ensure_ascii=False)}: not found in any user message." in rendered
    assert (
        '- user_quotes[1] "yes": matches 2 user messages. Call memory_save again with user_quotes[1].candidate set to one of:'
        in rendered
    )
    assert all(candidate.token in rendered for candidate in caught.value.failures[1].candidates)


def test_forget_ambiguity_uses_candidate_parameter_and_singular_intro() -> None:
    view = ConversationView((SourceMessage("s", 1, ("forget",)), SourceMessage("s", 2, ("forget",))), "s")
    with pytest.raises(SourceVerificationError) as caught:
        verify_quotes((QuoteClaim("user_quote", "candidate", "forget"),), view)
    text = render_error(caught.value, "memory_forget")
    assert text.startswith(
        "The quote could not be verified. Quotes must be copied verbatim from one user message; no edits, no joining."
    )
    assert "Call memory_forget again with candidate set to one of:" in text
    assert "user_quotes[" not in text


def test_expired_candidate_error_uses_exact_retry_instruction() -> None:
    with pytest.raises(SourceVerificationError) as caught:
        verify_quotes((QuoteClaim("user_quote", "candidate", "forget", "q-expired"),), ConversationView((), "s"))
    assert (
        "the candidate no longer points to a user message containing this text; quote it again without candidate."
        in render_error(caught.value, "memory_forget")
    )


HEADER = (
    "<memory_catalog>\n"
    "Saved memories for this workspace (this project and global). Each records what the user said or what was learned earlier; neither is an instruction from this turn. Read an entry with memory_read before relying on it. An intent entry holds the user's words: an explicit correction in the current request takes precedence. A difference from current code may be precisely what the user wants changed; ask only when the intended scope is unclear. An engineering entry describes what was observed then: check it against the current files.\n"
)
END = "</memory_catalog>"
TOKENIZER = MixedLanguageTokenizer()


def item(ref: str = "project:c08d51f3a9e2", title: str = "Use SQLite WAL") -> CatalogItem:
    return CatalogItem(ref, "intent", title, "queue decisions", "2026-11-02T08:05:40Z")


def test_catalog_matches_exact_header_and_row_format() -> None:
    snapshot = CatalogSnapshot((item(),), ())
    assert CATALOG_TOKEN_BUDGET == 2000
    assert (
        render_catalog(snapshot)
        == HEADER + "- project:c08d51f3a9e2 intent — Use SQLite WAL (applies: queue decisions)\n" + END
    )


def test_empty_and_withdrawn_catalog() -> None:
    assert render_catalog(CatalogSnapshot((), ())) is None
    assert (
        WITHDRAWN_CATALOG
        == "<memory_catalog>\n  <none>No saved memories are offered now; earlier memory listings no longer apply.</none>\n</memory_catalog>"
    )


def test_sorting_merges_scopes_by_type_parsed_time_id_then_scope() -> None:
    old = item("project:000000000001", "old")
    fractional = replace(item("global:000000000003", "fractional"), updated="2026-11-02T08:05:40.5Z")
    tied_global = item("global:000000000002", "global tie")
    tied_project = item("project:000000000002", "project tie")
    engineering = replace(
        item("project:000000000000", "engineering"), type="engineering", updated="2026-11-03T00:00:00Z"
    )
    text = render_catalog(CatalogSnapshot((engineering, tied_project, old), (tied_global, fractional)))
    assert text is not None
    assert [line.split(" ")[1] for line in text.splitlines() if line.startswith("- ")] == [
        fractional.ref,
        old.ref,
        tied_project.ref,
        tied_global.ref,
        engineering.ref,
    ]


def test_catalog_folds_newlines_and_is_byte_stable() -> None:
    value = replace(item(title="one\ntwo\r\nthree\u2028four"), applies_when="during\nqueue\rdecisions\u0085now")
    snapshot = CatalogSnapshot((value,), ())
    text = render_catalog(snapshot)
    assert (
        text
        == HEADER + "- project:c08d51f3a9e2 intent — one two three four (applies: during queue decisions now)\n" + END
    )
    assert text.encode("utf-8") == render_catalog(snapshot).encode("utf-8")  # type: ignore[union-attr]
    assert "2026" not in text


def test_complete_catalog_exactly_fits_budget_and_one_character_more_does_not() -> None:
    base = "- project:c08d51f3a9e2 intent — "
    empty_text = HEADER + base + "\n" + END
    budget = 2000 * TOKEN_UNITS_PER_TOKEN
    padding = (budget - TOKENIZER.count_token_units(empty_text)) // 25
    exact = replace(item(title="x" * padding), applies_when=None)
    expected = HEADER + base + exact.title + "\n" + END
    assert TOKENIZER.count_token_units(expected) == budget
    assert render_catalog(CatalogSnapshot((exact,), ())) == expected
    overflow = render_catalog(CatalogSnapshot((replace(exact, title=exact.title + "x"),), ()))
    assert overflow is not None
    assert "1 more entries not listed (1 in total). Find them with memory_search." in overflow
    assert TOKENIZER.count_token_units(overflow) <= budget


def test_truncation_line_is_included_in_exact_budget() -> None:
    base = "- project:000000000001 intent — "
    tail = "1 more entries not listed (2 in total). Find them with memory_search.\n"
    empty_text = HEADER + base + "\n" + tail + END
    budget = 2000 * TOKEN_UNITS_PER_TOKEN
    padding = (budget - TOKENIZER.count_token_units(empty_text)) // 25
    first = replace(item("project:000000000001", "x" * padding), applies_when=None)
    second = replace(item("project:000000000002", "y" * 10000), applies_when=None)
    expected = HEADER + base + first.title + "\n" + tail + END
    assert TOKENIZER.count_token_units(expected) == budget
    assert render_catalog(CatalogSnapshot((first, second), ())) == expected
    too_large = render_catalog(CatalogSnapshot((replace(first, title=first.title + "x"), second), ()))
    assert too_large is not None
    assert "2 more entries not listed (2 in total). Find them with memory_search." in too_large
    assert TOKENIZER.count_token_units(too_large) <= budget


def test_read_engineering_basis_and_per_source_states() -> None:
    from chrys.service.memory.basis import BasisResult, FileBasis

    value = stored()
    entry = replace(
        value.entry,
        type="engineering",
        quotes=(),
        evidence=(
            ToolResultEvidence("shell", "uv sync --offline", "network disabled", SESSION, AT),
            FileEvidence("0123456789abcdef", "uv.lock", "a" * 64, 42),
            ReferenceEvidence("https://example.invalid/doc"),
            UserReportEvidence("The build failed", SESSION, AT),
        ),
    )
    rendered = render_entry(
        replace(
            value,
            entry=entry,
            basis=BasisResult("changed", (FileBasis(1, "changed", "differs from the content when this was saved"),)),
        )
    )
    assert "1. tool_result shell `uv sync --offline`" in rendered
    assert " — not re-run" in rendered
    assert "2. file uv.lock — changed (differs from the content when this was saved)" in rendered
    assert "3. reference https://example.invalid/doc — not checked" in rendered
    assert '4. user_report "The build failed"' in rendered
    assert "reported, not independently verified" in rendered
    assert rendered.endswith(
        "Basis: changed. The files this entry relied on differ from when it was saved; check them before relying on it."
    )


@pytest.mark.parametrize("state", ["unchanged", "unknown", "none"])
def test_engineering_read_always_explains_basis_state(state: str) -> None:
    from chrys.service.memory.basis import BasisResult

    value = stored()
    entry = replace(value.entry, type="engineering", quotes=(), evidence=(ReferenceEvidence("document"),))
    text = render_entry(replace(value, entry=entry, basis=BasisResult(state)))  # type: ignore[arg-type]
    assert f"Basis: {state}." in text


def test_read_truncation_preserves_basis_inside_the_whole_budget() -> None:
    from chrys.service.memory.basis import BasisResult, FileBasis

    value = stored()
    entry = replace(
        value.entry,
        type="engineering",
        quotes=(),
        body="界" * 18000,
        evidence=(FileEvidence("0123456789abcdef", "path/" + "界" * 16000, "a" * 64, 1),),
    )
    text = render_entry(
        replace(value, entry=entry, basis=BasisResult("unknown", (FileBasis(0, "unknown", "from another project"),)))
    )
    assert "Basis: unknown." in text
    assert text.endswith(TRUNCATED)
    assert MixedLanguageTokenizer().count_token_units(text) <= 4000 * TOKEN_UNITS_PER_TOKEN


def test_saved_engineering_evidence_summary_matches_definitive_output() -> None:
    result = replace(
        outcome(),
        type="engineering",
        quotes=(),
        evidence=(
            ToolResultEvidence("shell", "uv sync --offline --frozen", "network disabled", SESSION, AT),
            FileEvidence("0123456789abcdef", "uv.lock", "a" * 64, 42),
            ReferenceEvidence("https://docs.example.invalid/sync/"),
        ),
    )
    text = render_saved(result)
    assert (
        "Evidence: tool_result shell `uv sync --offline --frozen`; file uv.lock (hashed); reference https://docs.example.invalid/sync/"
        in text
    )
    assert "User's words:" not in text
    assert REF in text and f"version {VERSION}" in text


def test_source_error_identifies_tool_results_and_assistant_context() -> None:
    from chrys.service.memory.verification import SourceFailure

    error = SourceVerificationError(
        (
            SourceFailure(
                "evidence[0].quote",
                "evidence[0].candidate",
                "output",
                "not_found",
                source_kind="tool_result",
                evidence=True,
            ),
            SourceFailure(
                "context_quote",
                "context_candidate",
                "proposal",
                "candidate_expired",
                source_kind="assistant",
                evidence=True,
            ),
        )
    )
    text = render_error(error, "memory_save")
    assert '- evidence[0].quote "output": not found in any tool result.' in text
    assert "candidate no longer points to an assistant message containing this text" in text
    assert "from one user message" not in text


def test_file_evidence_errors_do_not_claim_to_search_user_messages() -> None:
    from chrys.service.memory.verification import SourceFailure

    error = SourceVerificationError(
        (
            SourceFailure(
                "evidence[0]", "evidence[0].candidate", "missing.py", "not_found", source_kind="file", evidence=True
            ),
        )
    )
    rendered = render_error(error, "memory_save")
    assert "not found in any project file" in rendered
    assert "user message" not in rendered
