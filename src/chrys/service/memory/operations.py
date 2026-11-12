# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Host-independent memory reads, verified writes and their structured results."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from chrys.service.memory.basis import check_basis
from chrys.service.memory.change_log import ChangeLog, ChangeOperation, MemoryChange
from chrys.service.memory.storage.store import (
    GLOBAL_SCOPE,
    MAX_APPLIES_WHEN,
    MAX_BODY,
    MAX_CONTEXT,
    MAX_EVIDENCE,
    MAX_QUOTE,
    MAX_QUOTES,
    MAX_TITLE,
    MEMORY_TYPES,
    SCOPE_NAMES,
    CatalogItem,
    Context,
    Evidence,
    FileEvidence,
    MemoryEntryDraft,
    MemoryInputRejected,
    MemoryOperationError,
    MemoryScope,
    MemoryStatus,
    MemoryStore,
    MemoryType,
    Quote,
    ScopeName,
    StoredMemory,
    ToolResultEvidence,
    UserReportEvidence,
    WrittenBy,
    parse_ref,
    utc_text,
)
from chrys.service.memory.verification import (
    EVIDENCE_KINDS,
    ConversationView,
    EvidenceClaim,
    EvidenceKind,
    QuoteClaim,
    SourceFailure,
    SourceMessage,
    SourceVerificationError,
    verify_context,
    verify_evidence,
    verify_quotes,
)

logger = logging.getLogger(__name__)

SearchScope = Literal["project", "global", "all"]


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    """One read of the current project's and global catalogs."""

    project: tuple[CatalogItem, ...]
    global_: tuple[CatalogItem, ...]


@dataclass(frozen=True, slots=True)
class SearchPage:
    """A page of matches and the backend's opaque continuation token."""

    query: str
    entries: tuple[StoredMemory, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class SaveRequest:
    """The trial's memory_save arguments; omitted quotes differ from an empty list."""

    type: MemoryType
    title: str
    body: str
    applies_when: str | None = None
    scope: ScopeName | None = None
    user_quotes: tuple[QuoteClaim, ...] | None = None
    ref: str | None = None
    version: str | None = None
    evidence: tuple[EvidenceClaim, ...] | None = None
    context_quote: str | None = None
    context_candidate: str | None = None
    context_session: str | None = None


@dataclass(frozen=True, slots=True)
class SaveOutcome:
    """The submitted content and committed identity, without a competing reread."""

    ref: str
    version: str
    status: Literal["created", "existing", "updated"]
    type: MemoryType
    title: str
    applies_when: str | None
    quotes: tuple[Quote, ...]
    evidence: tuple[Evidence, ...] = ()
    context: Context | None = None
    entry_status: MemoryStatus | None = None
    restored: bool = False


@dataclass(frozen=True, slots=True)
class DeleteOutcome:
    """The identity and title of the entry just deleted."""

    ref: str
    title: str


@dataclass(frozen=True, slots=True)
class WithdrawOutcome:
    """A committed withdrawal and the verified words that authorized it."""

    ref: str
    version: str
    title: str
    basis: Quote


@dataclass(frozen=True, slots=True)
class MergeOutcome:
    """The kept entry's committed identity and the entry merged into it."""

    ref: str
    version: str
    title: str
    drop_ref: str


class MergePartialError(MemoryOperationError):
    """A merge committed the kept entry but did not finish deleting its duplicate."""

    def __init__(self, outcome: MergeOutcome) -> None:
        self.ref = outcome.ref
        self.version = outcome.version
        self.drop_ref = outcome.drop_ref
        super().__init__(
            "merge_partial",
            f"Merged sources into {outcome.ref} (version {outcome.version}), but {outcome.drop_ref} was not "
            "removed by this merge; read both entries before retrying.",
        )


class MemoryReader:
    """Read memories without exposing any write operation to a host."""

    def __init__(self, store: MemoryStore, project: MemoryScope | None, *, project_root: Path | None = None) -> None:
        self._store = store
        self._project = project
        self._project_root = project_root

    async def read(self, ref: str) -> StoredMemory:
        scope = _scope_for_ref(ref, self._project)
        stored = await self._store.get(scope, ref)
        basis = await asyncio.to_thread(check_basis, stored.entry.evidence, self._project, self._project_root)
        return replace(stored, basis=basis)

    async def search(self, query: str, scope: SearchScope = "all", cursor: str | None = None) -> SearchPage:
        if scope == "all":
            target = self._project if self._project is not None else GLOBAL_SCOPE
            entries, next_cursor = await self._store.search(target, query, cursor, with_global=True)
        elif scope == "project":
            entries, next_cursor = await self._store.search(_require_project(self._project), query, cursor)
        elif scope == "global":
            entries, next_cursor = await self._store.search(GLOBAL_SCOPE, query, cursor)
        else:
            raise MemoryInputRejected("memory_invalid_input", "Search scope must be project, global or all.")
        return SearchPage(query=query, entries=tuple(entries), next_cursor=next_cursor)

    async def catalog(self) -> CatalogSnapshot:
        project = () if self._project is None else tuple(await self._store.catalog(self._project))
        global_ = tuple(await self._store.catalog(GLOBAL_SCOPE))
        return CatalogSnapshot(project=project, global_=global_)


class MemoryWriter:
    """Verify the user's words before committing a complete memory draft."""

    def __init__(
        self,
        store: MemoryStore,
        project: MemoryScope | None,
        written_by: WrittenBy,
        *,
        change_log: ChangeLog | None = None,
        source: str | None = None,
        project_root: Path | None = None,
        allowed_types: tuple[MemoryType, ...] = MEMORY_TYPES,
        allowed_evidence: tuple[EvidenceKind, ...] = EVIDENCE_KINDS,
        allow_created_rewrites: bool = False,
    ) -> None:
        self._store = store
        self._project = project
        self._written_by = written_by
        self._change_log = change_log
        self._source = source
        self._project_root = project_root
        self._allowed_types = allowed_types
        self._allowed_evidence = allowed_evidence
        self._allow_created_rewrites = allow_created_rewrites
        self._created_versions: dict[str, str] = {}

    async def save(self, request: SaveRequest, view: ConversationView) -> SaveOutcome:
        _validate_request(request, view, self._allowed_types, self._allowed_evidence)
        if request.ref is None:
            scope = GLOBAL_SCOPE if request.scope == "global" else _require_project(self._project)
            draft = await self._draft(request, view)
            result = await self._store.create(scope, draft)
            if result.created:
                if self._allow_created_rewrites:
                    self._created_versions[result.ref] = result.version
                await self._append_change(
                    scope, result.ref, draft.type, "create", self._written_by, None, result.version
                )
            return _save_outcome(
                result.ref,
                result.version,
                "created" if result.created else "existing",
                draft,
                entry_status=result.status,
            )

        scope = _scope_for_ref(request.ref, self._project)
        current = await self._store.get(scope, request.ref)
        version = request.version
        if version is None:
            raise MemoryInputRejected("memory_invalid_input", "Updating an entry needs version.")
        _check_version(request.ref, current.version, version)
        can_rewrite_created = self._can_rewrite_created(request.ref, version)
        entry = current.entry
        if request.type != entry.type:
            raise MemoryInputRejected("memory_invalid_input", "An entry's type cannot change.")
        title_only = request.body == entry.body and request.applies_when == entry.applies_when
        restoring = entry.status == "withdrawn"
        occurrences: list[SourceMessage] = []
        draft = await self._draft(
            request, view, current=entry.draft(), title_only=title_only, restoring=restoring, occurrences=occurrences
        )
        if restoring:
            if entry.status_basis is None or not any(
                _after_withdrawal(message, entry.status_basis, view) for message in occurrences
            ):
                raise MemoryInputRejected(
                    "memory_invalid_input",
                    "This entry was withdrawn; updating it needs words the user said after withdrawing it.",
                )
            draft = replace(draft, status=None, status_basis=None)
        elif not title_only:
            if request.type == "intent":
                new_basis = any(
                    _quote_basis(quote) not in {_quote_basis(old) for old in entry.quotes} for quote in draft.quotes
                )
            else:
                new_basis = _has_new_evidence(draft.evidence, entry.evidence)
            created_rewrite = can_rewrite_created and (
                bool(draft.quotes) if request.type == "intent" else _has_new_evidence(draft.evidence, ())
            )
            if not new_basis and not created_rewrite:
                raise MemoryInputRejected(
                    "memory_invalid_input",
                    "Nothing was updated. Changing body or applies_when needs a new verified basis for this "
                    "entry's type; resubmitting the same basis will still fail. Supply newly verified user "
                    "words for intent, or newly verified tool results, user reports or changed file evidence "
                    "for engineering. Alternatively, change only the title while keeping body and "
                    "applies_when unchanged, or keep the existing entry and abandon this change.",
                )
        updated_version = await self._store.update(scope, request.ref, version, draft)
        if restoring:
            self._created_versions.pop(request.ref, None)
        elif can_rewrite_created:
            self._created_versions[request.ref] = updated_version
        await self._append_change(
            scope,
            request.ref,
            draft.type,
            "restore" if restoring else "update",
            # The change log records who acted; the entry's written_by stays its content author.
            self._written_by,
            version,
            updated_version,
        )
        return _save_outcome(request.ref, updated_version, "updated", draft, restored=restoring)

    def _can_rewrite_created(self, ref: str, version: str) -> bool:
        if ref not in self._created_versions:
            return False
        if self._created_versions[ref] != version:
            # An observed outside write permanently ends this writer's edit window.
            del self._created_versions[ref]
            return False
        return True

    async def _draft(
        self,
        request: SaveRequest,
        view: ConversationView,
        *,
        current: MemoryEntryDraft | None = None,
        title_only: bool = False,
        restoring: bool = False,
        occurrences: list[SourceMessage] | None = None,
    ) -> MemoryEntryDraft:
        """Retain omitted sources on title-only edits and restores, including context.

        New entries start without omitted sources. Ordinary substantive updates
        require their type's sources and clear omitted optional context; restore
        additionally needs freshly verified user words after the withdrawal.
        """
        old_quotes = current.quotes if current is not None else ()
        old_evidence = current.evidence if current is not None else ()
        old_context = current.context if current is not None else None
        quotes: tuple[Quote, ...] = ()
        evidence: tuple[Evidence, ...] = ()
        context: Context | None = None
        failures: list[SourceFailure] = []
        if request.user_quotes is not None:
            try:
                quotes = verify_quotes(
                    request.user_quotes, view, stored=old_quotes, view_first=restoring, occurrences=occurrences
                )
            except SourceVerificationError as error:
                failures.extend(error.failures)
        elif title_only or restoring:
            quotes = old_quotes
        elif request.type == "intent":
            raise MemoryInputRejected(
                "memory_invalid_input", "A new or substantively changed intent entry needs user_quotes."
            )
        if request.evidence is not None:
            try:
                evidence = await verify_evidence(
                    request.evidence,
                    view,
                    stored=old_evidence,
                    project=self._project,
                    project_root=self._project_root,
                    view_first=restoring,
                    occurrences=occurrences,
                )
            except SourceVerificationError as error:
                failures.extend(error.failures)
        elif title_only or restoring:
            evidence = old_evidence
        elif request.type == "engineering":
            raise MemoryInputRejected(
                "memory_invalid_input", "A new or substantively changed engineering entry needs evidence."
            )
        if request.context_quote is not None:
            claim = QuoteClaim(
                "context_quote",
                "context_candidate",
                request.context_quote,
                request.context_candidate,
                request.context_session,
            )
            try:
                context = verify_context(claim, view, stored=old_context)
            except SourceVerificationError as error:
                failures.extend(error.failures)
        elif title_only or restoring:
            context = old_context
        if failures:
            raise SourceVerificationError(failures)
        unchanged_sources = request.user_quotes is None and request.evidence is None and request.context_quote is None
        written_by = (
            current.written_by if current is not None and title_only and unchanged_sources else self._written_by
        )
        return MemoryEntryDraft(
            request.type,
            request.title,
            request.body,
            written_by,
            request.applies_when,
            quotes,
            context,
            evidence,
            status=current.status if current is not None else None,
            status_basis=current.status_basis if current is not None else None,
            merged_from=current.merged_from if current is not None else (),
        )

    async def withdraw(self, ref: str, version: str, user_quote: QuoteClaim, view: ConversationView) -> WithdrawOutcome:
        scope = _scope_for_ref(ref, self._project)
        current = await self._store.get(scope, ref)
        _check_version(ref, current.version, version)
        if current.entry.status == "withdrawn":
            raise MemoryInputRejected("memory_invalid_input", "This entry is already withdrawn.")
        basis = verify_quotes((user_quote,), view)[0]
        if basis.at is None:
            raise MemoryInputRejected("memory_invalid_input", "Withdrawing an entry needs a timestamped user message.")
        draft = replace(current.entry.draft(), status="withdrawn", status_basis=basis)
        updated_version = await self._store.update(scope, ref, version, draft)
        self._created_versions.pop(ref, None)
        await self._append_change(scope, ref, draft.type, "withdraw", self._written_by, version, updated_version)
        return WithdrawOutcome(ref, updated_version, draft.title, basis)

    async def merge(self, keep_ref: str, keep_version: str, drop_ref: str, drop_version: str) -> MergeOutcome:
        if keep_ref == drop_ref:
            raise MemoryInputRejected("memory_invalid_input", "A memory entry cannot be merged into itself.")
        if parse_ref(keep_ref)[0] != parse_ref(drop_ref)[0]:
            raise MemoryInputRejected("memory_invalid_input", "Merged entries must belong to the same scope.")
        scope = _scope_for_ref(keep_ref, self._project)
        keep = await self._store.get(scope, keep_ref)
        drop = await self._store.get(scope, drop_ref)
        _check_version(keep_ref, keep.version, keep_version)
        _check_version(drop_ref, drop.version, drop_version)
        if keep.entry.type != drop.entry.type:
            raise MemoryInputRejected("memory_invalid_input", "Merged entries must have the same type.")
        if keep.entry.status is not None or drop.entry.status is not None:
            raise MemoryInputRejected("memory_invalid_input", "Withdrawn entries cannot be merged.")
        quotes = _union(keep.entry.quotes, drop.entry.quotes)
        evidence = _union(keep.entry.evidence, drop.entry.evidence)
        if len(quotes) > MAX_QUOTES or len(evidence) > MAX_EVIDENCE:
            raise MemoryInputRejected(
                "memory_invalid_input",
                f"Merging would exceed {MAX_QUOTES} user quotes or {MAX_EVIDENCE} evidence items; "
                "nothing was merged. Keep both entries unchanged.",
            )
        context = keep.entry.context
        if context is not None and drop.entry.context is not None and context != drop.entry.context:
            raise MemoryInputRejected("memory_invalid_input", "Merged entries have different context sources.")
        draft = replace(
            keep.entry.draft(),
            quotes=quotes,
            evidence=evidence,
            context=context if context is not None else drop.entry.context,
            merged_from=_union(keep.entry.merged_from, (drop_ref,)),
        )
        result = await self._store.merge(scope, keep_ref, keep_version, drop_ref, drop_version, draft)
        self._created_versions.pop(keep_ref, None)
        self._created_versions.pop(drop_ref, None)
        outcome = MergeOutcome(keep_ref, result.version, draft.title, drop_ref)
        await self._append_change(scope, keep_ref, draft.type, "merge", self._written_by, keep_version, result.version)
        if not result.deleted:
            raise MergePartialError(outcome)
        # deleted=True also covers a drop that was already gone after keep was
        # written; the delete record then notes the merged end state, not an unlink here.
        await self._append_change(scope, drop_ref, drop.entry.type, "delete", self._written_by, drop_version, None)
        return outcome

    async def forget(self, ref: str, version: str, user_quote: QuoteClaim, view: ConversationView) -> DeleteOutcome:
        scope = _scope_for_ref(ref, self._project)
        problems: list[str] = []
        if not user_quote.text:
            problems.append(f"{user_quote.field}.text must not be empty")
        elif len(user_quote.text) > MAX_QUOTE:
            problems.append(f"{user_quote.field}.text is over {MAX_QUOTE} characters")
        if view.current_session is None and not user_quote.session:
            problems.append(f"{user_quote.field}.session is required in batch mode")
        _raise_invalid(problems)
        current = await self._store.get(scope, ref)
        _check_version(ref, current.version, version)
        verify_quotes((user_quote,), view)
        await self._store.delete(scope, ref, version)
        self._created_versions.pop(ref, None)
        await self._append_change(scope, ref, current.entry.type, "delete", self._written_by, version, None)
        return DeleteOutcome(ref=ref, title=current.entry.title)

    async def _append_change(
        self,
        scope: MemoryScope,
        ref: str,
        type: MemoryType,
        operation: ChangeOperation,
        written_by: WrittenBy,
        before_version: str | None,
        after_version: str | None,
    ) -> None:
        if self._change_log is None:
            return
        if not self._source:
            logger.warning("Memory change logging was skipped because no source was supplied.")
            return
        try:
            change = MemoryChange(
                time=utc_text(datetime.now(UTC)),
                scope=scope,
                ref=ref,
                type=type,
                operation=operation,
                written_by=written_by,
                source=self._source,
                before_version=before_version,
                after_version=after_version,
            )
            await self._change_log.append(change)
        except Exception:
            logger.warning("Memory change logging failed after a successful write.")


def _require_project(project: MemoryScope | None) -> MemoryScope:
    if project is None:
        raise MemoryOperationError(
            "memory_store_unavailable", "The current project could not be determined; project memory is unavailable."
        )
    return project


def _scope_for_ref(ref: str, project: MemoryScope | None) -> MemoryScope:
    scope, _ = parse_ref(ref)
    return GLOBAL_SCOPE if scope == "global" else _require_project(project)


def _validate_request(
    request: SaveRequest,
    view: ConversationView,
    allowed_types: tuple[MemoryType, ...],
    allowed_evidence: tuple[EvidenceKind, ...],
) -> None:
    problems: list[str] = []
    if request.type not in MEMORY_TYPES or request.type not in allowed_types:
        problems.append("this writer does not allow the requested entry type")
    if request.type == "intent" and request.evidence is not None:
        problems.append("intent entries cannot carry evidence")
    if request.type == "engineering" and request.user_quotes is not None:
        problems.append("engineering entries cannot carry user_quotes")
    if request.evidence is not None:
        if not 1 <= len(request.evidence) <= MAX_EVIDENCE:
            problems.append(f"evidence must be omitted or contain 1-{MAX_EVIDENCE} items")
        for index, claim in enumerate(request.evidence):
            if claim.kind not in allowed_evidence:
                problems.append(
                    f"evidence[{index}].kind is not allowed by this writer; file evidence is unavailable during extraction"
                )
    _text_problems(
        problems, "context_quote", request.context_quote, MAX_CONTEXT, required=request.context_quote is not None
    )
    if request.context_quote is None and (request.context_candidate is not None or request.context_session is not None):
        problems.append("context_candidate and context_session need context_quote")
    if request.context_quote is not None and view.current_session is None and not request.context_session:
        problems.append("context_session is required in batch mode")
    _text_problems(problems, "title", request.title, MAX_TITLE, required=True)
    if "\n" in request.title or "\r" in request.title:
        problems.append("title must be one line")
    _text_problems(problems, "body", request.body, MAX_BODY, required=True)
    _text_problems(problems, "applies_when", request.applies_when, MAX_APPLIES_WHEN)
    if request.user_quotes is not None:
        if not 1 <= len(request.user_quotes) <= MAX_QUOTES:
            problems.append(f"user_quotes must be omitted or contain 1-{MAX_QUOTES} quotes")
        if view.current_session is None:
            problems.extend(
                f"{claim.field}.session is required in batch mode" for claim in request.user_quotes if not claim.session
            )
    if request.scope is not None and request.scope not in SCOPE_NAMES:
        problems.append("scope must be project or global")
    if request.ref is None and request.version is not None:
        problems.append("version needs ref")
    if request.ref is not None:
        if request.version is None:
            problems.append("ref needs version")
        try:
            ref_scope, _ = parse_ref(request.ref)
        except MemoryOperationError as error:
            problems.append(error.message)
        else:
            if request.scope is not None and request.scope != ref_scope:
                problems.append("scope must match ref; an entry's scope cannot change")
    _raise_invalid(problems)


def _text_problems(problems: list[str], name: str, value: str | None, limit: int, *, required: bool = False) -> None:
    if value is None:
        if required:
            problems.append(f"{name} is required")
    else:
        if required and not value.strip():
            problems.append(f"{name} must not be empty")
        if len(value) > limit:
            problems.append(f"{name} is over {limit} characters")


def _raise_invalid(problems: list[str]) -> None:
    if problems:
        raise MemoryInputRejected("memory_invalid_input", "Invalid memory input: " + "; ".join(problems) + ".")


def _check_version(ref: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise MemoryOperationError(
            "memory_version_conflict",
            f"{ref} changed since it was read (now version {actual}); read it again before changing it.",
        )


def _save_outcome(
    ref: str,
    version: str,
    status: Literal["created", "existing", "updated"],
    draft: MemoryEntryDraft,
    *,
    entry_status: MemoryStatus | None = None,
    restored: bool = False,
) -> SaveOutcome:
    return SaveOutcome(
        ref=ref,
        version=version,
        status=status,
        type=draft.type,
        title=draft.title,
        applies_when=draft.applies_when,
        quotes=draft.quotes,
        evidence=draft.evidence,
        context=draft.context,
        entry_status=entry_status if entry_status is not None else draft.status,
        restored=restored,
    )


def _after_withdrawal(message: SourceMessage, basis: Quote, view: ConversationView) -> bool:
    if message.session == basis.session and basis.item is not None:
        matches = tuple(
            candidate
            for candidate in view.messages
            if candidate.kind == "user" and candidate.session == basis.session and candidate.item == basis.item
        )
        if len(matches) == 1:
            return message.position > matches[0].position
    if message.at is None or basis.at is None:
        return False
    return datetime.fromisoformat(message.at) > datetime.fromisoformat(basis.at)


def _union[T](first: tuple[T, ...], second: tuple[T, ...]) -> tuple[T, ...]:
    result: list[T] = []
    for item in (*first, *second):
        if item not in result:
            result.append(item)
    return tuple(result)


def _quote_basis(quote: Quote) -> tuple[str, str, str | None, str | None]:
    return "user", quote.text, quote.session, quote.item


def _excerpt_basis(evidence: ToolResultEvidence | UserReportEvidence) -> tuple[str, str, str | None, str | None]:
    return evidence.kind, evidence.excerpt, evidence.session, evidence.item


def _has_new_evidence(current: tuple[Evidence, ...], stored: tuple[Evidence, ...]) -> bool:
    old_excerpts = {
        _excerpt_basis(item) for item in stored if isinstance(item, ToolResultEvidence | UserReportEvidence)
    }
    for item in current:
        if isinstance(item, ToolResultEvidence | UserReportEvidence) and _excerpt_basis(item) not in old_excerpts:
            return True
        if isinstance(item, FileEvidence) and not any(
            isinstance(old, FileEvidence)
            and (old.project, old.path, old.sha256) == (item.project, item.path, item.sha256)
            for old in stored
        ):
            return True
    return False
