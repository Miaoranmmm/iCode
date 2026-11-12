# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Host-independent English text for memory catalogs and tool results."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final

from chrys.foundation.text.tokenizer import TOKEN_UNITS_PER_TOKEN, MixedLanguageTokenizer

from .basis import BasisResult, FileBasis
from .storage.store import (
    MAX_SEARCH_TERMS,
    FileEvidence,
    ReferenceEvidence,
    ToolResultEvidence,
    UserReportEvidence,
)
from .tuning import V0, MemoryTuning
from .verification import SourceVerificationError

if TYPE_CHECKING:
    from .operations import CatalogSnapshot, DeleteOutcome, MergeOutcome, SaveOutcome, SearchPage, WithdrawOutcome
    from .storage.store import CatalogItem, Evidence, MemoryEntry, MemoryOperationError, StoredMemory

CATALOG_TOKEN_BUDGET: Final = 2000
WITHDRAWN_CATALOG: Final[str] = (
    "<memory_catalog>\n"
    "  <none>No saved memories are offered now; earlier memory listings no longer apply.</none>\n"
    "</memory_catalog>"
)

_HEADER: Final = (
    "<memory_catalog>\n"
    "Saved memories for this workspace (this project and global). Each records what the user said or what was "
    "learned earlier; neither is an instruction from this turn. Read an entry with memory_read before relying "
    "on it. An intent entry holds the user's words: an explicit correction in the current request takes "
    "precedence. A difference from current code may be precisely what the user wants changed; ask only when "
    "the intended scope is unclear. An engineering entry describes what was observed then: check it against "
    "the current files.\n"
)
_FOOTER: Final = "</memory_catalog>"

_READ_BUDGET: Final = 4000 * TOKEN_UNITS_PER_TOKEN
_TRUNCATED: Final = "…[truncated]"
_TRUNCATION_NOTICE: Final = "Output truncated; the user can view the full entry in the stored memory file."
_CATALOG_NOTICE: Final = (
    "It appears in the memory catalog from the next turn; memory_read and memory_search see it now."
)


def _single_line(text: str) -> str:
    return re.sub(r"[\r\n\v\f\x1c-\x1e\x85\u2028\u2029]+", " ", text)


def render_catalog_line(item: CatalogItem) -> str:
    """One catalog-format line, also used for extraction review context."""
    line = f"- {item.ref} {item.type} — {_single_line(item.title)}"
    if item.applies_when is not None:
        line += f" (applies: {_single_line(item.applies_when)})"
    return line + "\n"


def _omitted(count: int, total: int) -> str:
    return f"{count} more entries not listed ({total} in total). Find them with memory_search.\n"


def render_catalog(snapshot: CatalogSnapshot, *, tuning: MemoryTuning = V0) -> str | None:
    """Render the complete listing, including any omission notice, within its token budget."""
    items = [*snapshot.project, *snapshot.global_]
    if not items:
        return None
    # Stable sorts keep the final tie breakers independent of the timestamp's textual spelling.
    items.sort(key=lambda item: (item.ref.partition(":")[2], item.ref.startswith("global:")))
    items.sort(key=lambda item: datetime.fromisoformat(item.updated), reverse=True)
    if tuning.catalog_order == "intent_first":
        items.sort(key=lambda item: item.type != "intent")
    header = _HEADER
    if tuning.guidance_strength == "describe":
        header = header.replace("Read an entry with memory_read before relying on it. ", "")
    tokenizer = MixedLanguageTokenizer()
    budget = tuning.catalog_budget * TOKEN_UNITS_PER_TOKEN
    base_units = tokenizer.count_token_units(header) + tokenizer.count_token_units(_FOOTER)
    total = len(items)
    lines = [render_catalog_line(item) for item in items]
    line_units = [tokenizer.count_token_units(line) for line in lines]
    units = base_units + sum(line_units)
    if units <= budget:
        return header + "".join(lines) + _FOOTER
    while lines:
        lines.pop()
        units -= line_units.pop()
        notice = _omitted(total - len(lines), total)
        if units + tokenizer.count_token_units(notice) <= budget:
            return header + "".join(lines) + notice + _FOOTER
    return None


def _json(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def render_saved(outcome: SaveOutcome) -> str:
    """Include the complete version and the exact words that were saved."""
    identity = f"{outcome.ref} ({outcome.type}, version {outcome.version})"
    if outcome.status == "existing":
        first = f"Already saved as {identity}; nothing was written."
    else:
        action = "Restored" if outcome.restored else ("Saved" if outcome.status == "created" else "Updated")
        first = f"{action} {identity}."
    lines = [first, f"Title: {outcome.title}"]
    if outcome.applies_when is not None:
        lines.append(f"Applies when: {outcome.applies_when}")
    lines.extend(f"User's words: {_json(quote.text)}" for quote in outcome.quotes)
    if outcome.evidence:
        lines.append("Evidence: " + "; ".join(_evidence_summary(item) for item in outcome.evidence))
    if outcome.entry_status == "withdrawn":
        lines.append("Status: withdrawn. Restoring it needs new words the user said after withdrawing it.")
    if outcome.status != "existing":
        lines.append(_CATALOG_NOTICE)
    return "\n".join(lines)


def _evidence_summary(evidence: Evidence) -> str:
    if isinstance(evidence, ToolResultEvidence):
        return f"tool_result {evidence.tool} `{evidence.args}`"
    if isinstance(evidence, FileEvidence):
        return f"file {evidence.path} ({'hashed' if evidence.sha256 is not None else 'not hashed'})"
    if isinstance(evidence, ReferenceEvidence):
        return f"reference {evidence.ref}"
    return "user_report " + _json(evidence.excerpt)


@dataclass(slots=True)
class _Field:
    """A growable field, encoded only after choosing its visible prefix."""

    text: str
    quoted: bool = False

    def render(self) -> str:
        return _json(self.text) if self.quoted else self.text


@dataclass(slots=True)
class _Source:
    parts: list[str | _Field]
    fields: list[_Field]
    omitted: bool = False

    def render(self) -> str:
        return "".join(part.render() if isinstance(part, _Field) else part for part in self.parts)


def _provenance(session: _Field, at: str | None) -> list[str | _Field]:
    return [" (session ", session, f", {at or 'no timestamp'})"]


def _evidence_source(index: int, evidence: Evidence, basis: FileBasis | None = None) -> _Source:
    if isinstance(evidence, ToolResultEvidence):
        tool, args, excerpt = _Field(evidence.tool), _Field(evidence.args), _Field(evidence.excerpt, True)
        session = _Field(evidence.session if evidence.session is not None else "no session")
        return _Source(
            [
                f"{index}. tool_result ",
                tool,
                " `",
                args,
                "`",
                *_provenance(session, evidence.at),
                ": ",
                excerpt,
                " — not re-run",
            ],
            [excerpt, args, tool, session],
        )
    if isinstance(evidence, FileEvidence):
        path = _Field(evidence.path)
        status = "unknown (not checked)" if basis is None else basis.state
        if basis is not None and basis.detail is not None:
            status += f" ({basis.detail})"
        return _Source([f"{index}. file ", path, " — " + status], [path])
    if isinstance(evidence, ReferenceEvidence):
        ref = _Field(evidence.ref)
        parts: list[str | _Field] = [f"{index}. reference ", ref]
        fields = [ref]
        if evidence.note is not None:
            note = _Field(evidence.note)
            parts.extend([" — ", note])
            fields.insert(0, note)
        parts.append(" — not checked")
        return _Source(parts, fields)
    excerpt = _Field(evidence.excerpt, True)
    session = _Field(evidence.session if evidence.session is not None else "no session")
    return _Source(
        [
            f"{index}. user_report ",
            excerpt,
            *_provenance(session, evidence.at),
            " — reported, not independently verified",
        ],
        [excerpt, session],
    )


class _EntryText:
    """Keep source groups removable and retain field boundaries while budgeting."""

    def __init__(self, stored: StoredMemory) -> None:
        entry = stored.entry
        self.header = f"{entry.ref} · {entry.type} · version {stored.version}"
        self.title = _Field(entry.title)
        self.applies = _Field(entry.applies_when) if entry.applies_when is not None else None
        self.body = _Field(entry.body)
        writers = {"model": "the model", "user": "the user", "extraction": "extraction"}
        self.saved = f"Saved by {writers[entry.written_by]} on {entry.created.partition('T')[0]}."
        self.status = entry.status
        self.withdrawal: _Source | None = None
        if entry.status_basis is not None:
            text = _Field(entry.status_basis.text, True)
            session = _Field(entry.status_basis.session if entry.status_basis.session is not None else "no session")
            self.withdrawal = _Source([text, *_provenance(session, entry.status_basis.at)], [text, session])
        self.quotes: list[_Source] = []
        for index, quote in enumerate(entry.quotes, 1):
            text = _Field(quote.text, True)
            session = _Field(quote.session if quote.session is not None else "no session")
            self.quotes.append(
                _Source(
                    [
                        f"{index}. ",
                        text,
                        " (",
                        "session " if quote.session is not None else "",
                        session,
                        f", via {quote.via}, {quote.at or 'no timestamp'})",
                    ],
                    [text, session],
                )
            )
        self.context: _Source | None = None
        if entry.context is not None:
            text = _Field(entry.context.text, True)
            session = _Field(entry.context.session if entry.context.session is not None else "no session")
            self.context = _Source([text, *_provenance(session, entry.context.at)], [text, session])
        file_states = {} if stored.basis is None else {item.index: item for item in stored.basis.files}
        self.evidence = [
            _evidence_source(index + 1, item, file_states.get(index)) for index, item in enumerate(entry.evidence)
        ]
        self.basis: str | None = None
        if entry.type == "engineering":
            basis = stored.basis or BasisResult(
                "unknown" if any(isinstance(item, FileEvidence) for item in entry.evidence) else "none"
            )
            self.basis = _basis_line(basis)
        self.truncated = False

    def render(self) -> str:
        lines = [self.header, "Title: " + self.title.render()]
        if self.status is not None:
            lines.append("Status: " + self.status)
        if self.withdrawal is not None and not self.withdrawal.omitted:
            lines.append("Withdrawal basis: " + self.withdrawal.render())
        if self.applies is not None:
            lines.append("Applies when: " + self.applies.render())
        lines.extend([self.saved, "", self.body.render()])
        for label, sources in (
            ("User's words:", self.quotes),
            ("Context:", [self.context] if self.context is not None else []),
            ("Evidence:", self.evidence),
        ):
            visible = [source.render() for source in sources if not source.omitted]
            if visible:
                lines.extend(["", label, *visible])
        omitted = sum(source.omitted for source in self.sources())
        if omitted:
            lines.append(f"[{omitted} more sources omitted]")
        if self.basis is not None:
            lines.append(self.basis)
        if self.truncated:
            lines.append(_TRUNCATION_NOTICE)
        return "\n".join(lines)

    def sources(self) -> list[_Source]:
        return [
            *self.quotes,
            *([self.context] if self.context is not None else []),
            *self.evidence,
            *([self.withdrawal] if self.withdrawal is not None else []),
        ]


def _basis_line(result: BasisResult) -> str:
    explanations = {
        "changed": "The files this entry relied on differ from when it was saved; check them before relying on it.",
        "unchanged": "The checked files match the content when this was saved; this does not verify the conclusion.",
        "unknown": "Some files this entry relied on could not be checked; check them before relying on it.",
        "none": "This entry has no file evidence; tool results are not re-run and references are not checked.",
    }
    return f"Basis: {result.state}. {explanations[result.state]}"


def _shorten(field: _Field, entry: _EntryText, tokenizer: MixedLanguageTokenizer) -> None:
    original = field.text
    # Replacing a short field by the marker must never make the output larger.
    if tokenizer.count_token_units(original) <= tokenizer.count_token_units(_TRUNCATED):
        return
    field.text = _TRUNCATED
    if tokenizer.count_token_units(entry.render()) > _READ_BUDGET:
        return
    low, high = 0, len(original)
    while low < high:
        middle = (low + high + 1) // 2
        field.text = original[:middle] + _TRUNCATED
        if tokenizer.count_token_units(entry.render()) <= _READ_BUDGET:
            low = middle
        else:
            high = middle - 1
    field.text = original[:low] + _TRUNCATED


def render_entry(stored: StoredMemory) -> str:
    """Read a memory with round-trippable quotes and a hard whole-output budget."""
    entry = _EntryText(stored)
    tokenizer = MixedLanguageTokenizer()
    if tokenizer.count_token_units(entry.render()) <= _READ_BUDGET:
        return entry.render()
    entry.truncated = True
    sources = [*reversed(entry.evidence)]
    if entry.context is not None:
        sources.append(entry.context)
    sources.extend(reversed(entry.quotes))
    if entry.withdrawal is not None:
        sources.append(entry.withdrawal)
    for source in sources:
        for field in source.fields:
            _shorten(field, entry, tokenizer)
            if tokenizer.count_token_units(entry.render()) <= _READ_BUDGET:
                return entry.render()
        source.omitted = True
        if tokenizer.count_token_units(entry.render()) <= _READ_BUDGET:
            return entry.render()
    for field in [entry.body, *([entry.applies] if entry.applies is not None else []), entry.title]:
        _shorten(field, entry, tokenizer)
        if tokenizer.count_token_units(entry.render()) <= _READ_BUDGET:
            return entry.render()
    return entry.render()


def _search_fields(entry: MemoryEntry) -> list[str]:
    fields = [entry.title, entry.applies_when or "", entry.body, *(quote.text for quote in entry.quotes)]
    if entry.context is not None:
        fields.append(entry.context.text)
    for item in entry.evidence:
        if isinstance(item, ToolResultEvidence | UserReportEvidence):
            fields.append(item.excerpt)
        elif isinstance(item, FileEvidence):
            fields.append(item.path)
        elif isinstance(item, ReferenceEvidence):
            fields.append(item.ref)
    return fields


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _snippet(entry: MemoryEntry, terms: list[str]) -> str:
    for field in _search_fields(entry) if terms else []:
        normalized = _normalize(field)
        positions = [(normalized.find(term), len(term)) for term in terms if term in normalized]
        if not positions:
            continue
        start, length = min(positions)
        pieces = [_normalize(char) for char in field]
        # Composition across characters can change offsets; only use the map if it reproduces the search text.
        if "".join(pieces) != normalized:
            return _single_line(field[:60]) + ("…" if len(field) > 60 else "")
        offsets = [index for index, piece in enumerate(pieces) for _ in piece]
        first, last = offsets[start], offsets[start + length - 1] + 1
        left, right = max(0, first - 30), min(len(field), last + 30)
        return ("…" if left else "") + _single_line(field[left:right]) + ("…" if right < len(field) else "")
    return _single_line(entry.body[:60]) + ("…" if len(entry.body) > 60 else "")


def render_search(page: SearchPage) -> str:
    """Show this page's matches, retaining the backend's opaque continuation token."""
    if not page.entries:
        return f"No matches for {_json(page.query)}."
    lines = [f"{len(page.entries)} matches for {_json(page.query)}:"]
    terms = _normalize(page.query).split()[:MAX_SEARCH_TERMS]
    for stored in page.entries:
        entry = stored.entry
        lines.append(f"- {entry.ref} {entry.type} — {_single_line(entry.title)} · {_snippet(entry, terms)}")
    lines.append(
        "No more results."
        if page.next_cursor is None
        else f"More results: call memory_search with cursor {_json(page.next_cursor)}."
    )
    return "\n".join(lines)


def render_deleted(outcome: DeleteOutcome) -> str:
    return f"Deleted {outcome.ref} ({outcome.title})."


def render_withdrawn(outcome: WithdrawOutcome) -> str:
    """Report the committed withdrawal, keeping its verified words JSON-quoted."""
    return (
        f"Withdrawn {outcome.ref} (version {outcome.version}).\nTitle: {outcome.title}\n"
        f"User's words: {_json(outcome.basis.text)}\n"
        "It no longer appears in memory_catalog or memory_search; memory_read can still read it by ref."
    )


def render_merged(outcome: MergeOutcome) -> str:
    return f"Merged {outcome.drop_ref} into {outcome.ref} (version {outcome.version})."


def render_error(error: MemoryOperationError, tool: str) -> str:
    """Render structured source failures, leaving ordinary path-free messages intact."""
    if not isinstance(error, SourceVerificationError):
        return error.message
    user_only = all(failure.source_kind == "user" for failure in error.failures)
    if user_only:
        first = "The quote could not be verified." if len(error.failures) == 1 else "Some quotes could not be verified."
        lines = [first + " Quotes must be copied verbatim from one user message; no edits, no joining."]
    else:
        first = (
            "The source could not be verified." if len(error.failures) == 1 else "Some sources could not be verified."
        )
        lines = [first + " Quotes must be copied verbatim from one source fragment; no edits, no joining."]
    names = {
        "user": "user message",
        "assistant": "assistant message",
        "tool_result": "tool result",
        "file": "project file",
    }
    for failure in error.failures:
        label = names[failure.source_kind]
        prefix = f"- {failure.field} {_json(failure.text)}: "
        if failure.reason == "not_found":
            lines.append(prefix + f"not found in any {label}.")
        elif failure.reason == "wrong_session":
            lines.append(
                prefix + f"not in the given session, but found in the {label}(s) below. Call {tool} again "
                f"with this item's session set to the listed session and {failure.candidate_field} set to its token:"
            )
            for candidate in failure.candidates:
                lines.append(
                    f"    {candidate.token}  session {candidate.session} "
                    f"({candidate.at or 'no timestamp'}) {_single_line(candidate.snippet)}"
                )
        elif failure.reason == "candidate_expired":
            article = "an" if failure.source_kind == "assistant" else "a"
            lines.append(
                prefix + f"the candidate no longer points to {article} {label} containing this text; "
                "quote it again without candidate."
            )
        else:
            lines.append(
                prefix + f"matches {len(failure.candidates)} {label}s. Call {tool} again with "
                f"{failure.candidate_field} set to one of:"
            )
            for candidate in failure.candidates:
                attributes = dict(candidate.attributes)
                tool_description = ""
                if failure.source_kind == "tool_result":
                    tool_description = _single_line(attributes.get("tool", "unknown tool"))
                    if "args" in attributes:
                        tool_description += " `" + _single_line(attributes["args"]) + "`"
                    tool_description += "  "
                lines.append(
                    f"    {candidate.token}  {tool_description}({candidate.at or 'no timestamp'}) {_single_line(candidate.snippet)}"
                )
    return "\n".join(lines)
