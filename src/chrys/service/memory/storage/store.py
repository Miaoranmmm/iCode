# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Memory entries and the storage interface every backend implements.

An entry is one remembered requirement (``intent``, backed by the user's own
words) or finding (``engineering``, backed by evidence). This module holds the
backend-neutral part: the entry schema with its limits, the scope an entry
lives in, the ``MemoryStore`` protocol and the error kinds callers see. How
entries are laid out, locked and versioned is the backend's business.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Literal, Protocol

if TYPE_CHECKING:
    from chrys.service.memory.basis import BasisResult

SCHEMA_VERSION: Final = 1

MemoryType = Literal["intent", "engineering"]
MemoryStatus = Literal["withdrawn"]
ScopeName = Literal["project", "global"]
WrittenBy = Literal["model", "user", "extraction"]
QuoteVia = Literal["message", "remember", "cli"]
ProjectAnchor = Literal["git", "dir"]
MemoryErrorKind = Literal[
    "memory_invalid_input",
    "memory_quote_not_found",
    "memory_evidence_not_found",
    "memory_evidence_ambiguous",
    "memory_evidence_candidate_expired",
    "memory_source_wrong_session",
    "memory_not_found",
    "memory_version_conflict",
    "memory_full",
    "memory_entry_invalid",
    "memory_store_unavailable",
    "merge_partial",
]

MEMORY_TYPES: Final[tuple[MemoryType, ...]] = ("intent", "engineering")
SCOPE_NAMES: Final[tuple[ScopeName, ...]] = ("project", "global")
WRITERS: Final[tuple[WrittenBy, ...]] = ("model", "user", "extraction")
QUOTE_VIAS: Final[tuple[QuoteVia, ...]] = ("message", "remember", "cli")

MAX_TITLE: Final = 80
MAX_APPLIES_WHEN: Final = 200
MAX_BODY: Final = 1500
MAX_QUOTES: Final = 3
MAX_QUOTE: Final = 500
# A quote the user typed straight into /remember or the CLI is the whole entry, not an excerpt.
MAX_DIRECT_QUOTE: Final = 1500
MAX_CONTEXT: Final = 600
MAX_EVIDENCE: Final = 6
MAX_EXCERPT: Final = 600
MAX_ARGS: Final = 200
MAX_ENTRY_BYTES: Final = 16 * 1024
MAX_ENTRIES_PER_SCOPE: Final = 500
SEARCH_PAGE_SIZE: Final = 10
MAX_SEARCH_TERMS: Final = 8

ENTRY_ID_RE: Final = re.compile(r"[0-9a-f]{12}")
PROJECT_ID_RE: Final = re.compile(r"[0-9a-f]{16}")


class MemoryOperationError(Exception):
    """A memory operation that failed in a way the caller should report; *message* names no absolute path."""

    def __init__(self, kind: MemoryErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind: MemoryErrorKind = kind
        self.message = message


class MemoryInputRejected(MemoryOperationError):
    """A controlled check rejected the caller's input before changing any memory.

    Only the check that owns this guarantee may raise this type. Storage,
    concurrency and unknown failures remain ordinary ``MemoryOperationError``s.
    """


# ---------------------------------------------------------------------------
# Sources: what an entry rests on
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Quote:
    """The user's own words. *session* is ``None`` only when they were typed outside any session (the CLI)."""

    text: str
    via: QuoteVia
    session: str | None = None
    at: str | None = None
    item: str | None = None


@dataclass(frozen=True, slots=True)
class Context:
    """The agent's proposal the user was answering; it shows what was agreed to, not the user's words."""

    text: str
    session: str | None = None
    at: str | None = None
    item: str | None = None


@dataclass(frozen=True, slots=True)
class ToolResultEvidence:
    """A passage from a tool result seen in a session."""

    tool: str
    args: str
    excerpt: str
    session: str | None = None
    at: str | None = None
    item: str | None = None
    kind: Literal["tool_result"] = "tool_result"


@dataclass(frozen=True, slots=True)
class FileEvidence:
    """A project file as it was when the entry was saved; *sha256* is ``None`` for files over 2 MiB."""

    project: str
    path: str
    sha256: str | None
    bytes: int
    kind: Literal["file"] = "file"


@dataclass(frozen=True, slots=True)
class ReferenceEvidence:
    """A URL or document name, recorded as given and never fetched."""

    ref: str
    note: str | None = None
    kind: Literal["reference"] = "reference"


@dataclass(frozen=True, slots=True)
class UserReportEvidence:
    """A fact the user stated, quoted from their message; reported, not independently verified."""

    excerpt: str
    session: str | None = None
    at: str | None = None
    item: str | None = None
    kind: Literal["user_report"] = "user_report"


Evidence = ToolResultEvidence | FileEvidence | ReferenceEvidence | UserReportEvidence


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemoryEntryDraft:
    """An entry before the backend gives it an id and timestamps; its scope comes from the ``MemoryScope``."""

    type: MemoryType
    title: str
    body: str
    written_by: WrittenBy
    applies_when: str | None = None
    quotes: tuple[Quote, ...] = ()
    context: Context | None = None
    evidence: tuple[Evidence, ...] = ()
    status: MemoryStatus | None = None
    status_basis: Quote | None = None
    merged_from: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """A stored entry. Fields mirror the on-disk frontmatter; ``schema`` is always ``SCHEMA_VERSION``."""

    id: str
    scope: ScopeName
    created: str
    updated: str
    type: MemoryType
    title: str
    body: str
    written_by: WrittenBy
    applies_when: str | None = None
    quotes: tuple[Quote, ...] = ()
    context: Context | None = None
    evidence: tuple[Evidence, ...] = ()
    status: MemoryStatus | None = None
    status_basis: Quote | None = None
    merged_from: tuple[str, ...] = ()

    @property
    def ref(self) -> str:
        return format_ref(self.scope, self.id)

    def draft(self) -> MemoryEntryDraft:
        """The entry's content without its identity and timestamps."""
        return MemoryEntryDraft(
            type=self.type,
            title=self.title,
            body=self.body,
            written_by=self.written_by,
            applies_when=self.applies_when,
            quotes=self.quotes,
            context=self.context,
            evidence=self.evidence,
            status=self.status,
            status_basis=self.status_basis,
            merged_from=self.merged_from,
        )


@dataclass(frozen=True, slots=True)
class MemoryScope:
    """Which entries an operation sees: the project's (``project_id`` set) or the global ones.

    ``root`` is used only for display and for resolving file evidence and basis
    paths; it takes no part in identity, deduplication or versions. ``anchor``
    describes whether the id came from a Git repository or a plain directory.
    """

    scope: ScopeName
    project_id: str | None = None
    root: str | None = field(default=None, compare=False)
    anchor: ProjectAnchor | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.scope == "project":
            if self.project_id is None or not PROJECT_ID_RE.fullmatch(self.project_id):
                raise MemoryOperationError("memory_invalid_input", "A project scope needs a 16-character project id.")
        elif self.scope == "global":
            if self.project_id is not None or self.root is not None or self.anchor is not None:
                raise MemoryOperationError("memory_invalid_input", "The global scope has no project.")
        else:
            raise MemoryOperationError("memory_invalid_input", f"Unknown scope {self.scope!r}.")


GLOBAL_SCOPE: Final = MemoryScope("global")


@dataclass(frozen=True, slots=True)
class StoredMemory:
    """An entry and the version token read with it; pass the version back to update or delete it."""

    entry: MemoryEntry
    version: str
    basis: BasisResult | None = None


@dataclass(frozen=True, slots=True)
class CreateResult:
    """``created`` is ``False`` when an identical entry already existed; ``ref`` and ``version`` are then its."""

    ref: str
    version: str
    created: bool
    status: MemoryStatus | None = None


@dataclass(frozen=True, slots=True)
class MergeResult:
    """The kept entry was written; deleted=False is a durable partial merge."""

    version: str
    deleted: bool


@dataclass(frozen=True, slots=True)
class CatalogItem:
    """The fields the memory catalog shows for one entry."""

    ref: str
    type: MemoryType
    title: str
    applies_when: str | None
    updated: str


@dataclass(frozen=True, slots=True)
class UnreadableMemory:
    """A quarantined entry: never shown or searched, only listed and deletable with this ref and version."""

    ref: str
    version: str
    reason: str


@dataclass(frozen=True, slots=True)
class ProjectInfo:
    """A project that has stored memories; ``root`` is ``None`` when its description is missing."""

    project_id: str
    root: str | None


class MemoryStore(Protocol):
    """Where entries live. Sources are verified before an entry gets here; a backend only stores.

    Every method but ``list_projects`` takes the scope it acts in. Errors are ``MemoryOperationError``.
    """

    async def list_entries(self, scope: MemoryScope) -> list[StoredMemory]:
        """Every readable entry, newest ``updated`` first, then by id."""
        ...

    async def get(self, scope: MemoryScope, ref: str) -> StoredMemory: ...

    async def search(
        self, scope: MemoryScope, query: str, cursor: str | None = None, *, with_global: bool = False
    ) -> tuple[list[StoredMemory], str | None]:
        """One page of matches and the cursor of the next page, ``None`` on the last.

        Covered: title, applies_when, body, quote text, context text and
        evidence excerpt, path and ref. A non-empty query is split into words
        and an entry matches when it contains at least one; entries containing
        all of them come first. An empty query lists everything newest first.
        *with_global* on a project scope is the ``all`` scope: the project's and
        the global entries ranked together under one cursor.
        """
        ...

    async def catalog(self, scope: MemoryScope) -> list[CatalogItem]: ...

    async def create(self, scope: MemoryScope, draft: MemoryEntryDraft) -> CreateResult:
        """Store *draft* with a fresh id and timestamps, or return the identical entry already stored."""
        ...

    async def update(self, scope: MemoryScope, ref: str, version: str, draft: MemoryEntryDraft) -> str:
        """Replace the entry's content, keeping its id, type and creation time; returns the new version."""
        ...

    async def delete(self, scope: MemoryScope, ref: str, version: str) -> None:
        """Delete the entry, readable or quarantined, if it is still at *version*."""
        ...

    async def merge(
        self,
        scope: MemoryScope,
        keep_ref: str,
        keep_version: str,
        drop_ref: str,
        drop_version: str,
        keep_draft: MemoryEntryDraft,
    ) -> MergeResult:
        """Write the kept entry before deleting the other under one scope lock.

        A partial result confirms the new kept version, while the requested
        deletion did not complete. A drop that no longer exists once the kept
        entry is written is the intended end state and is not partial. Neither
        outcome rolls back the kept entry.
        """
        ...

    async def unreadable(self, scope: MemoryScope) -> list[UnreadableMemory]: ...

    async def location(self, scope: MemoryScope) -> str | None:
        """Where the scope's entries are kept, for display; ``None`` for a backend with no file layout."""
        ...

    async def list_projects(self) -> list[ProjectInfo]: ...


# ---------------------------------------------------------------------------
# Refs
# ---------------------------------------------------------------------------


def format_ref(scope: ScopeName, name: str) -> str:
    return f"{scope}:{name}"


def parse_ref(ref: str) -> tuple[ScopeName, str]:
    """Split ``project:<name>`` / ``global:<name>``; *name* is an id, or a quarantined file's name."""
    scope, sep, name = ref.partition(":")
    if not sep or scope not in SCOPE_NAMES or not name:
        raise MemoryInputRejected("memory_invalid_input", f"{ref!r} is not a memory ref (project:<id> or global:<id>).")
    return scope, name  # type: ignore[return-value]  # membership checked above


# ---------------------------------------------------------------------------
# Validation and the canonical payload
# ---------------------------------------------------------------------------


def validate_draft(draft: MemoryEntryDraft, *, scope: ScopeName | None = None) -> None:
    """Raise ``memory_invalid_input`` naming every field that breaks the schema."""
    problems = draft_problems(draft, scope=scope)
    if problems:
        raise MemoryInputRejected("memory_invalid_input", "Invalid memory entry: " + "; ".join(problems) + ".")


def draft_problems(draft: MemoryEntryDraft, *, scope: ScopeName | None = None) -> list[str]:
    """Every schema rule *draft* breaks, as short phrases; empty when it is valid."""
    problems: list[str] = []
    if draft.type not in MEMORY_TYPES:
        problems.append(f"type must be one of {', '.join(MEMORY_TYPES)}")
    if draft.written_by not in WRITERS:
        problems.append(f"written_by must be one of {', '.join(WRITERS)}")
    _check_text(problems, "title", draft.title, MAX_TITLE, required=True, one_line=True)
    _check_text(problems, "applies_when", draft.applies_when, MAX_APPLIES_WHEN)
    _check_text(problems, "body", draft.body, MAX_BODY, required=True)
    if len(draft.quotes) > MAX_QUOTES:
        problems.append(f"at most {MAX_QUOTES} quotes")
    for index, quote in enumerate(draft.quotes):
        if quote.via not in QUOTE_VIAS:
            problems.append(f"quotes[{index}].via must be one of {', '.join(QUOTE_VIAS)}")
        limit = MAX_DIRECT_QUOTE if quote.via in ("remember", "cli") else MAX_QUOTE
        _check_text(problems, f"quotes[{index}].text", quote.text, limit, required=True)
        if quote.session is None and quote.via != "cli":
            problems.append(f"quotes[{index}] needs the session it was said in")
        _check_at(problems, f"quotes[{index}].at", quote.at)
    if draft.context is not None:
        _check_text(problems, "context.text", draft.context.text, MAX_CONTEXT, required=True)
        _check_at(problems, "context.at", draft.context.at)
    if len(draft.evidence) > MAX_EVIDENCE:
        problems.append(f"at most {MAX_EVIDENCE} evidence items")
    for index, item in enumerate(draft.evidence):
        _evidence_problems(problems, f"evidence[{index}]", item)
    if draft.status not in (None, "withdrawn"):
        problems.append("status must be withdrawn or omitted")
    if (draft.status is None) != (draft.status_basis is None):
        problems.append("status and status_basis must appear together")
    if draft.status_basis is not None:
        _quote_problems(problems, "status_basis", draft.status_basis)
    _merged_from_problems(problems, draft.merged_from, scope)
    if draft.type == "intent" and draft.written_by != "user" and not draft.quotes:
        problems.append("an intent entry needs the user's words in quotes")
    if draft.type == "engineering" and not draft.evidence:
        problems.append("an engineering entry needs evidence")
    return problems


def _quote_problems(problems: list[str], name: str, quote: Quote) -> None:
    if not isinstance(quote, Quote):
        problems.append(f"{name} must be a user quote")
        return
    if quote.via not in QUOTE_VIAS:
        problems.append(f"{name}.via must be one of {', '.join(QUOTE_VIAS)}")
    limit = MAX_DIRECT_QUOTE if quote.via in ("remember", "cli") else MAX_QUOTE
    _check_text(problems, f"{name}.text", quote.text, limit, required=True)
    if quote.session is None and quote.via != "cli":
        problems.append(f"{name} needs the session it was said in")
    _check_at(problems, f"{name}.at", quote.at)


def _merged_from_problems(problems: list[str], refs: tuple[str, ...], scope: ScopeName | None) -> None:
    if not isinstance(refs, tuple):
        problems.append("merged_from must contain memory refs")
        return
    seen_scopes: set[str] = set()
    for index, ref in enumerate(refs):
        if not isinstance(ref, str):
            problems.append(f"merged_from[{index}] must be a memory ref")
            continue
        ref_scope, separator, entry_id = ref.partition(":")
        if separator != ":" or ref_scope not in SCOPE_NAMES or not ENTRY_ID_RE.fullmatch(entry_id):
            problems.append(f"merged_from[{index}] must be a memory ref")
            continue
        seen_scopes.add(ref_scope)
        if scope is not None and ref_scope != scope:
            problems.append(f"merged_from[{index}] must belong to {scope} scope")
    if len(seen_scopes) > 1:
        problems.append("merged_from refs must belong to one scope")


def is_zoned_timestamp(value: str) -> bool:
    """Whether *value* is an ISO-8601 moment with a time zone (``2026-11-02T08:05:40Z``)."""
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return False
    return moment.tzinfo is not None and moment.utcoffset() is not None


def utc_text(moment: datetime) -> str:
    """*moment* (zoned) in UTC as ``...Z``, with its fraction of a second only when it has one."""
    utc = moment.astimezone(UTC)
    return utc.isoformat(timespec="microseconds" if utc.microsecond else "seconds").replace("+00:00", "Z")


def _check_at(problems: list[str], name: str, value: str | None) -> None:
    # Only the "...Z" form is written, so an entry reads back exactly as it was saved and dedupes against itself.
    if value is not None and not (isinstance(value, str) and value.endswith("Z") and is_zoned_timestamp(value)):
        problems.append(f"{name} must be a UTC timestamp ending in Z")


def _evidence_problems(problems: list[str], name: str, item: Evidence) -> None:
    # Only args and excerpts have budgets in the design; a path, tool name or reference is as long as it is.
    if isinstance(item, ToolResultEvidence):
        _check_text(problems, f"{name}.tool", item.tool, None, required=True)
        _check_text(problems, f"{name}.args", item.args, MAX_ARGS)
        _check_text(problems, f"{name}.excerpt", item.excerpt, MAX_EXCERPT, required=True)
        _check_at(problems, f"{name}.at", item.at)
    elif isinstance(item, FileEvidence):
        if not PROJECT_ID_RE.fullmatch(item.project):
            problems.append(f"{name}.project must be a project id")
        _check_text(problems, f"{name}.path", item.path, None, required=True)
        if item.sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", item.sha256):
            problems.append(f"{name}.sha256 must be 64 hex characters")
        if item.bytes < 0:
            problems.append(f"{name}.bytes must not be negative")
    elif isinstance(item, ReferenceEvidence):
        _check_text(problems, f"{name}.ref", item.ref, None, required=True)
        _check_text(problems, f"{name}.note", item.note, None)
    elif isinstance(item, UserReportEvidence):
        _check_text(problems, f"{name}.excerpt", item.excerpt, MAX_EXCERPT, required=True)
        _check_at(problems, f"{name}.at", item.at)
    else:
        problems.append(f"{name} has an unknown kind")


def _check_text(
    problems: list[str],
    name: str,
    value: str | None,
    limit: int | None,
    *,
    required: bool = False,
    one_line: bool = False,
) -> None:
    if value is None:
        if required:
            problems.append(f"{name} is required")
        return
    if not isinstance(value, str):
        problems.append(f"{name} must be text")
        return
    if required and not value.strip():
        problems.append(f"{name} must not be empty")
    if limit is not None and len(value) > limit:
        problems.append(f"{name} is over {limit} characters")
    if one_line and ("\n" in value or "\r" in value):
        problems.append(f"{name} must be one line")


def payload_key(scope: ScopeName, draft: MemoryEntryDraft) -> str:
    """A canonical serialization of an entry's content; two entries are duplicates when their keys are equal.

    Identity, timestamps and who wrote it are left out on purpose: the same
    words saved again under the same scope, by the user or the model, find the
    entry already stored.
    """
    content = asdict(draft)
    for name in ("written_by", "status", "status_basis", "merged_from"):
        del content[name]
    return json.dumps({"scope": scope, **content}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
