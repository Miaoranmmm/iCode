# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Verify verbatim sources against an immutable, host-neutral conversation view."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .names import is_memory_tool
from .storage.store import (
    MAX_ARGS,
    MAX_CONTEXT,
    MAX_DIRECT_QUOTE,
    MAX_EXCERPT,
    MAX_QUOTE,
    Context,
    Evidence,
    MemoryErrorKind,
    MemoryInputRejected,
    MemoryOperationError,
    MemoryScope,
    Quote,
    ReferenceEvidence,
    ToolResultEvidence,
    UserReportEvidence,
    is_zoned_timestamp,
)

SourceKind = Literal["user", "assistant", "tool_result"]
EvidenceKind = Literal["tool_result", "file", "reference", "user_report"]
EVIDENCE_KINDS: tuple[EvidenceKind, ...] = ("tool_result", "file", "reference", "user_report")


@dataclass(frozen=True, slots=True)
class SourceMessage:
    """One original host record or block; kind and part are part of its identity."""

    session: str
    position: int
    fragments: tuple[str, ...]
    kind: SourceKind = "user"
    part: int = 0
    item: str | None = None
    at: str | None = None
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        problems: list[str] = []
        if not isinstance(self.session, str) or not self.session:
            problems.append("session must not be empty")
        if self.kind not in ("user", "assistant", "tool_result"):
            problems.append("kind must be user, assistant or tool_result")
        if not isinstance(self.part, int) or isinstance(self.part, bool) or self.part < 0:
            problems.append("part must be a nonnegative integer")
        if not self.fragments or any(not isinstance(fragment, str) or not fragment for fragment in self.fragments):
            problems.append("fragments must contain at least one nonempty text fragment")
        if self.at is not None and not (
            isinstance(self.at, str) and self.at.endswith("Z") and is_zoned_timestamp(self.at)
        ):
            problems.append("at must be a UTC timestamp ending in Z")
        if not isinstance(self.attributes, tuple) or any(
            not isinstance(pair, tuple) or len(pair) != 2 or any(not isinstance(value, str) for value in pair)
            for pair in self.attributes
        ):
            problems.append("attributes must contain text pairs")
        if problems:
            raise MemoryOperationError("memory_invalid_input", "Invalid source message: " + "; ".join(problems) + ".")


@dataclass(frozen=True, slots=True)
class ConversationView:
    """Original sources in one conversation, or the material shown in an extraction batch."""

    messages: tuple[SourceMessage, ...]
    current_session: str | None

    def __post_init__(self) -> None:
        if self.current_session == "":
            raise MemoryOperationError("memory_invalid_input", "current_session must not be empty.")
        identities: set[tuple[str, SourceKind, str, str | int, int]] = set()
        for message in self.messages:
            if self.current_session is not None and message.session != self.current_session:
                raise MemoryOperationError("memory_invalid_input", "Every message must belong to current_session.")
            identity = (message.session, *_identity(message))
            if identity in identities:
                raise MemoryOperationError(
                    "memory_invalid_input", "Message identities must be unique within a session."
                )
            identities.add(identity)


@dataclass(frozen=True, slots=True)
class QuoteClaim:
    """A quote as the model submitted it."""

    field: str
    candidate_field: str
    text: str
    candidate: str | None = None
    session: str | None = None


@dataclass(frozen=True, slots=True)
class ToolResultClaim:
    quote: str
    candidate: str | None = None
    session: str | None = None
    kind: Literal["tool_result"] = "tool_result"


@dataclass(frozen=True, slots=True)
class UserReportClaim:
    quote: str
    candidate: str | None = None
    session: str | None = None
    kind: Literal["user_report"] = "user_report"


@dataclass(frozen=True, slots=True)
class FileClaim:
    path: str
    kind: Literal["file"] = "file"


@dataclass(frozen=True, slots=True)
class ReferenceClaim:
    ref: str
    note: str | None = None
    kind: Literal["reference"] = "reference"


EvidenceClaim = ToolResultClaim | UserReportClaim | FileClaim | ReferenceClaim


@dataclass(frozen=True, slots=True)
class QuoteCandidate:
    """One possible source of an ambiguous quote."""

    token: str
    session: str
    at: str | None
    snippet: str
    attributes: tuple[tuple[str, str], ...] = ()


FailureReason = Literal["not_found", "candidate_expired", "wrong_session", "ambiguous"]


@dataclass(frozen=True, slots=True)
class SourceFailure:
    """A failed claim, including the source kind and parameter paths."""

    field: str
    candidate_field: str
    text: str
    reason: FailureReason
    candidates: tuple[QuoteCandidate, ...] = ()
    source_kind: SourceKind | Literal["file"] = "user"
    evidence: bool = False


class SourceVerificationError(MemoryInputRejected):
    """All failed claims from one verification, with a stable priority."""

    def __init__(self, failures: Sequence[SourceFailure]) -> None:
        self.failures = tuple(failures)
        for reason in ("not_found", "candidate_expired", "wrong_session", "ambiguous"):
            matching = tuple(failure for failure in self.failures if failure.reason == reason)
            if not matching:
                continue
            kind: MemoryErrorKind
            if reason == "not_found":
                kind = (
                    "memory_quote_not_found"
                    if any(not failure.evidence for failure in matching)
                    else "memory_evidence_not_found"
                )
            elif reason == "candidate_expired":
                kind = "memory_evidence_candidate_expired"
            elif reason == "wrong_session":
                kind = "memory_source_wrong_session"
            else:
                kind = "memory_evidence_ambiguous"
            super().__init__(kind, "Some sources could not be verified.")
            return
        raise MemoryOperationError("memory_invalid_input", "Source verification needs at least one failed claim.")


def _validate_claim(claim: QuoteClaim, view: ConversationView, limit: int) -> None:
    problems: list[str] = []
    if not claim.text:
        problems.append(f"{claim.field}.text must not be empty")
    if view.current_session is None and not claim.session:
        problems.append(f"{claim.field}.session is required in batch mode")
    if len(claim.text) > limit:
        problems.append(f"{claim.field}.text is over {limit} characters")
    if problems:
        raise MemoryInputRejected("memory_invalid_input", "Invalid quote claims: " + "; ".join(problems) + ".")


def _locate(claim: QuoteClaim, view: ConversationView, kind: SourceKind, *, evidence: bool) -> SourceMessage:
    matches = tuple(
        message
        for message in view.messages
        if message.kind == kind
        and (kind != "tool_result" or not is_memory_tool(dict(message.attributes).get("tool", "")))
        and (view.current_session is not None or message.session == claim.session)
        and any(claim.text in fragment for fragment in message.fragments)
        and (claim.candidate is None or _token(message) == claim.candidate)
    )
    if len(matches) == 1:
        return matches[0]
    reason: FailureReason = (
        "candidate_expired" if claim.candidate is not None else ("not_found" if not matches else "ambiguous")
    )
    if reason == "not_found" and view.current_session is None:
        # A long session id is easy to mistype. Text that sits in another shown session is
        # offered back with its real session; nothing is saved until the model resubmits.
        matches = tuple(
            message
            for message in view.messages
            if message.kind == kind
            and (kind != "tool_result" or not is_memory_tool(dict(message.attributes).get("tool", "")))
            and any(claim.text in fragment for fragment in message.fragments)
        )
        if matches:
            reason = "wrong_session"
    candidates = (
        tuple(_candidate(message, claim.text) for message in matches)
        if reason in ("ambiguous", "wrong_session")
        else ()
    )
    raise SourceVerificationError(
        (SourceFailure(claim.field, claim.candidate_field, claim.text, reason, candidates, kind, evidence),)
    )


def verify_quotes(
    claims: Sequence[QuoteClaim],
    view: ConversationView,
    *,
    stored: Sequence[Quote] = (),
    view_first: bool = False,
    occurrences: list[SourceMessage] | None = None,
) -> tuple[Quote, ...]:
    """Verify user words, optionally locating fresh occurrences before retaining old sources."""
    verified: list[Quote] = []
    failures: list[SourceFailure] = []
    retained_indices: set[int] = set()
    for claim in claims:
        retained_index = _next_retained(
            tuple(
                index
                for index, quote in enumerate(stored)
                if quote.text == claim.text and quote.session == claim.session
            ),
            retained_indices,
        )
        retained = stored[retained_index] if retained_index is not None else None
        limit = MAX_DIRECT_QUOTE if retained is not None and retained.via in ("remember", "cli") else MAX_QUOTE
        _validate_claim(claim, view, limit)
        if retained_index is not None and not view_first:
            retained_indices.add(retained_index)
            verified.append(stored[retained_index])
            continue
        try:
            message = _locate(claim, view, "user", evidence=False)
        except SourceVerificationError as error:
            if retained_index is not None and all(_absent_here(failure) for failure in error.failures):
                retained_indices.add(retained_index)
                verified.append(stored[retained_index])
            else:
                failures.extend(error.failures)
        else:
            verified.append(Quote(claim.text, "message", message.session, message.at, message.item))
            if occurrences is not None:
                occurrences.append(message)
    if failures:
        raise SourceVerificationError(failures)
    return tuple(verified)


def _absent_here(failure: SourceFailure) -> bool:
    """The named session no longer shows the text; a copy elsewhere does not void a stored source."""
    return failure.reason in ("not_found", "wrong_session")


def _next_retained(matches: tuple[int, ...], consumed: set[int]) -> int | None:
    """Keep full stored occurrences one-to-one before reusing an identical claim's first match."""
    return next((index for index in matches if index not in consumed), matches[0] if matches else None)


def verify_context(claim: QuoteClaim, view: ConversationView, *, stored: Context | None = None) -> Context:
    """Keep matching stored context, otherwise verify only assistant text."""
    _validate_claim(claim, view, MAX_CONTEXT)
    if stored is not None and stored.text == claim.text and stored.session == claim.session:
        return stored
    source = _locate(claim, view, "assistant", evidence=True)
    return Context(claim.text, source.session, source.at, source.item)


async def verify_evidence(
    claims: Sequence[EvidenceClaim],
    view: ConversationView,
    *,
    stored: Sequence[Evidence] = (),
    project: MemoryScope | None = None,
    project_root: Path | None = None,
    view_first: bool = False,
    occurrences: list[SourceMessage] | None = None,
) -> tuple[Evidence, ...]:
    """Verify each submitted kind without mixing sources or partially saving."""
    from .file_evidence import capture_file

    verified: list[Evidence] = []
    failures: list[SourceFailure] = []
    retained_indices: set[int] = set()
    for index, evidence in enumerate(claims):
        field = f"evidence[{index}]"
        if isinstance(evidence, FileClaim):
            if not evidence.path:
                raise MemoryInputRejected("memory_invalid_input", f"{field}.path must not be empty.")
            try:
                verified.append(await asyncio.to_thread(capture_file, project, project_root, evidence.path))
            except MemoryInputRejected:
                failures.append(
                    SourceFailure(
                        field, field + ".candidate", evidence.path, "not_found", source_kind="file", evidence=True
                    )
                )
            continue
        if isinstance(evidence, ReferenceClaim):
            if not evidence.ref:
                raise MemoryInputRejected("memory_invalid_input", f"{field}.ref must not be empty.")
            previous_index = _next_retained(
                tuple(
                    index
                    for index, item in enumerate(stored)
                    if isinstance(item, ReferenceEvidence) and item.ref == evidence.ref and item.note == evidence.note
                ),
                retained_indices,
            )
            previous = (
                stored[previous_index] if previous_index is not None else ReferenceEvidence(evidence.ref, evidence.note)
            )
            if previous_index is not None:
                retained_indices.add(previous_index)
            if previous not in verified:
                verified.append(previous)
            continue
        claim = QuoteClaim(field, field + ".candidate", evidence.quote, evidence.candidate, evidence.session)
        _validate_claim(claim, view, MAX_EXCERPT if isinstance(evidence, ToolResultClaim) else MAX_QUOTE)
        if isinstance(evidence, ToolResultClaim):
            previous_index = _next_retained(
                tuple(
                    index
                    for index, item in enumerate(stored)
                    if isinstance(item, ToolResultEvidence)
                    and item.excerpt == evidence.quote
                    and item.session == evidence.session
                ),
                retained_indices,
            )
            kind: SourceKind = "tool_result"
        else:
            previous_index = _next_retained(
                tuple(
                    index
                    for index, item in enumerate(stored)
                    if isinstance(item, UserReportEvidence)
                    and item.excerpt == evidence.quote
                    and item.session == evidence.session
                ),
                retained_indices,
            )
            kind = "user"
        if previous_index is not None and not (view_first and isinstance(evidence, UserReportClaim)):
            retained_indices.add(previous_index)
            verified.append(stored[previous_index])
            continue
        try:
            source = _locate(claim, view, kind, evidence=True)
        except SourceVerificationError as error:
            if previous_index is not None and all(_absent_here(failure) for failure in error.failures):
                retained_indices.add(previous_index)
                verified.append(stored[previous_index])
            else:
                failures.extend(error.failures)
        else:
            if isinstance(evidence, ToolResultClaim):
                attributes = dict(source.attributes)
                tool, args = attributes.get("tool", ""), attributes.get("args", "")
                if not tool:
                    failures.append(
                        SourceFailure(
                            field, field + ".candidate", evidence.quote, "not_found", source_kind=kind, evidence=True
                        )
                    )
                    continue
                if len(args) > MAX_ARGS:
                    raise MemoryInputRejected("memory_invalid_input", f"{field}.args is over {MAX_ARGS} characters.")
                verified.append(ToolResultEvidence(tool, args, evidence.quote, source.session, source.at, source.item))
            else:
                verified.append(UserReportEvidence(evidence.quote, source.session, source.at, source.item))
                if occurrences is not None:
                    occurrences.append(source)
    if failures:
        raise SourceVerificationError(failures)
    return tuple(verified)


def _identity(message: SourceMessage) -> tuple[SourceKind, str, str | int, int]:
    return (
        (message.kind, "item", message.item, message.part)
        if message.item is not None
        else (message.kind, "position", message.position, message.part)
    )


def _token(message: SourceMessage) -> str:
    encoded = json.dumps(
        [message.session, *_identity(message), message.fragments], ensure_ascii=True, separators=(",", ":")
    )
    return "q-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _candidate(message: SourceMessage, text: str) -> QuoteCandidate:
    fragment = next(fragment for fragment in message.fragments if text in fragment)
    start = fragment.index(text)
    left, right = max(0, start - 40), min(len(fragment), start + len(text) + 40)
    snippet = " ".join(fragment[left:right].splitlines())
    if left:
        snippet = "…" + snippet
    if right < len(fragment):
        snippet += "…"
    return QuoteCandidate(_token(message), message.session, message.at, snippet, message.attributes)
