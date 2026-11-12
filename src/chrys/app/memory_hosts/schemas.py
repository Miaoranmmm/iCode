# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Strict public memory schemas and typed decoding of model arguments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp.types import Tool

from chrys.service.memory.names import MEMORY_TOOL_NAMES
from chrys.service.memory.operations import SaveRequest, SearchScope
from chrys.service.memory.storage.store import (
    MAX_APPLIES_WHEN,
    MAX_BODY,
    MAX_CONTEXT,
    MAX_DIRECT_QUOTE,
    MAX_EVIDENCE,
    MAX_EXCERPT,
    MAX_QUOTE,
    MAX_QUOTES,
    MAX_TITLE,
    MemoryInputRejected,
    MemoryOperationError,
)
from chrys.service.memory.verification import (
    EvidenceClaim,
    FileClaim,
    QuoteClaim,
    ReferenceClaim,
    ToolResultClaim,
    UserReportClaim,
)

_INTENT_SAVE_DESCRIPTION = (
    "Save a lasting preference, requirement or decision the user stated that later conversations will need. "
    "Save when the user asks you to remember something. Quote their words verbatim. Do not save task progress, "
    "to-dos, one-off instructions, what loaded reference files already say, facts cheap to read from code, "
    "or your own analysis as intent. To change an entry, pass its ref and version."
)
_INTENT_READ_DESCRIPTION = (
    "Read one memory entry in full, including its text and the user's words or evidence behind it. "
    "Read an entry before relying on it."
)
SEARCH_DESCRIPTION = (
    "Find entries by words in their title, applies_when, text, quotes or evidence, including entries absent "
    "from the catalog. An empty query lists entries newest first."
)
SAVE_DESCRIPTION = (
    "Save what later conversations will need, so the user does not repeat it and you do not rediscover it. "
    'Two cases: the user stated a lasting preference, requirement or decision (type "intent", quote their words verbatim); '
    'you established a reusable engineering finding or failure cause from tool results (type "engineering", cite the evidence). '
    "Save when the user asks you to remember something. Do not save task progress, to-dos, one-off instructions, "
    "what the loaded reference files already say, or facts cheap to read from the code. Never turn a user requirement "
    "into an engineering entry or your own analysis into an intent entry. To change an entry, pass its ref and version."
)
READ_DESCRIPTION = (
    "Read one memory entry in full: its text, the user's words or the evidence behind it, and whether the files "
    "it was based on have changed since it was saved. Read an entry before relying on it."
)
FORGET_DESCRIPTION = (
    "Delete a memory entry only when the user asks for it to be permanently removed. Quote the user's request "
    "verbatim in user_quote. If the user says an entry no longer applies, use memory_withdraw. If evidence suggests "
    "an engineering entry is outdated, update it with memory_save instead of deleting it."
)
WITHDRAW_DESCRIPTION = (
    "Withdraw a memory entry when the user says it no longer applies. Quote the user's words verbatim in user_quote. "
    "If the user asks for permanent removal, use memory_forget. If evidence suggests an engineering entry is outdated, "
    "update it with memory_save and new evidence instead."
)
MERGE_DESCRIPTION = (
    "Merge two duplicate active entries of the same type and scope. Keep the first entry's text and combine their "
    "stored sources without loss. Read both entries first; at most three quotes and six evidence items may remain."
)
NOTE_CONFLICT_DESCRIPTION = "Count a conflict you cannot resolve between two entries. Leave both entries unchanged."
EXTRACTION_ONLY_TOOL_NAMES = MEMORY_TOOL_NAMES & {"memory_merge", "memory_note_conflict"}
READ_ONLY_TOOL_NAMES = MEMORY_TOOL_NAMES & {"memory_read", "memory_search", "memory_catalog"}

CATALOG_DESCRIPTION = "List saved memories for this project and global scope. Read an entry before relying on it."
_INTENT_SCOPE_DESCRIPTION = (
    "Default project. Use global only when the user explicitly said it applies across projects; "
    '"from now on" alone does not.'
)
_QUOTE_DESCRIPTION = (
    "New quotes must be at most 500 characters; up to 1,500 is allowed only to retain an unchanged stored "
    "cli/remember quote. Copy one continuous user-provided fragment from one user message verbatim, including "
    "typed or pasted text; no edits or joining across paste boundaries. Pass the JSON-decoded value of a "
    "displayed quote, without its surrounding quotes and with escaped newlines decoded."
)
_CANDIDATE_DESCRIPTION = "Set only to a candidate token returned by this tool for an ambiguous match."
_INTENT_UPDATE_DESCRIPTION = (
    "Omit ref and version to create. For an update submit the complete entry; type and scope cannot change. "
    "Changes to body or applies_when need new words from the user. A title-only edit may omit user_quotes "
    "while keeping body and applies_when unchanged. Pass the version returned by memory_read unchanged."
)

_SCOPE_DESCRIPTION = (
    '"project": visible only in this project (default). "global": visible in every project; for intent only when '
    'the user explicitly said so ("all projects"; "from now on" alone does not), for engineering only when it holds '
    "outside this project, with its preconditions in applies_when."
)
_UPDATE_DESCRIPTION = (
    "Omit ref and version to create. For an update submit the complete entry; type and scope cannot change. "
    "A substantive intent change needs new words from the user, an engineering change new evidence; a title-only "
    "edit may keep the existing basis without widening its meaning. Pass the version returned by memory_read unchanged."
)
_CREATED_UPDATE_DESCRIPTION = (
    _UPDATE_DESCRIPTION
    + " An entry genuinely created in this batch may be reworded using the same verified sources and its "
    "latest version written by this batch; this exception does not apply to pre-existing entries."
)
_REF_DESCRIPTION = (
    "Copy the complete ref returned by a tool, including its project: or global: prefix; never pass only the id."
)


def _string(description: str | None = None, *, limit: int | None = None, minimum: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    if description is not None:
        schema["description"] = description
    if limit is not None:
        schema["maxLength"] = limit
    if minimum is not None:
        schema["minLength"] = minimum
    return schema


def _object(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


def tool_definitions(
    *,
    writable: bool,
    intent_only: bool = False,
    review: bool = False,
    created_rewrites: bool = False,
    describe_refs: bool = False,
) -> list[Tool]:
    """Fresh schema objects prevent a caller from mutating later listings."""
    quote = _object(
        {
            "text": _string(_QUOTE_DESCRIPTION, limit=MAX_DIRECT_QUOTE, minimum=1),
            "candidate": _string(_CANDIDATE_DESCRIPTION),
            "session": _string("Copy the full session from memory_read to retain a stored quote on update."),
        },
        ("text",),
    )
    save = _object(
        {
            "type": {"type": "string", "enum": ["intent"] if intent_only else ["intent", "engineering"]},
            "title": {
                **_string('State the conclusion itself, not "about X".', limit=MAX_TITLE, minimum=1),
                "not": {"pattern": r"[\r\n]"},
            },
            "body": _string(
                "The conclusion and the conditions under which it holds; keep it short.", limit=MAX_BODY, minimum=1
            ),
            "applies_when": _string(
                'When to look at this entry: which task, files, environment or error. Omit rather than write "always".',
                limit=MAX_APPLIES_WHEN,
            ),
            "scope": {
                "type": "string",
                "enum": ["project", "global"],
                "default": "project",
                "description": _INTENT_SCOPE_DESCRIPTION if intent_only else _SCOPE_DESCRIPTION,
            },
            "user_quotes": {"type": "array", "items": quote, "minItems": 1, "maxItems": MAX_QUOTES},
            "ref": _string(_INTENT_UPDATE_DESCRIPTION if intent_only else _UPDATE_DESCRIPTION),
            "version": _string(_INTENT_UPDATE_DESCRIPTION if intent_only else _UPDATE_DESCRIPTION),
        },
        ("type", "title", "body"),
    )
    if not intent_only:
        source = {
            "quote": _string("Copy one original fragment verbatim; no edits or joining.", limit=MAX_EXCERPT, minimum=1),
            "candidate": _string(_CANDIDATE_DESCRIPTION),
            "session": _string("Copy the full source session; required in batch mode and to retain stored evidence."),
        }
        tool_result = _object({"kind": {"const": "tool_result", "type": "string"}, **source}, ("kind", "quote"))
        user_report = _object(
            {
                "kind": {"const": "user_report", "type": "string"},
                **source,
                "quote": _string(
                    "A fact the user stated, copied verbatim from one user fragment.", limit=MAX_QUOTE, minimum=1
                ),
            },
            ("kind", "quote"),
        )
        file = _object(
            {
                "kind": {"const": "file", "type": "string"},
                "path": _string("A file within the current project root, hashed when saved.", minimum=1),
            },
            ("kind", "path"),
        )
        reference = _object(
            {
                "kind": {"const": "reference", "type": "string"},
                "ref": _string("A URL or document name, recorded without fetching it.", minimum=1),
                "note": _string(),
            },
            ("kind", "ref"),
        )
        save["properties"].update(
            evidence={
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_EVIDENCE,
                "items": {"oneOf": [tool_result, file, reference, user_report]},
            },
            context_quote=_string(
                "The proposal of yours the user was answering, copied verbatim; shows what the user agreed to.",
                limit=MAX_CONTEXT,
                minimum=1,
            ),
            context_candidate=_string(_CANDIDATE_DESCRIPTION),
            context_session=_string(
                "The full context source session; required in batch mode. Copy it from memory_read to retain stored context.",
                minimum=1,
            ),
        )
    save["dependentRequired"] = {"ref": ["version"], "version": ["ref"]}
    if not intent_only:
        save["dependentRequired"].update(context_candidate=["context_quote"], context_session=["context_quote"])
    tools = [
        Tool(
            name="memory_read",
            description=_INTENT_READ_DESCRIPTION if intent_only else READ_DESCRIPTION,
            inputSchema=_object({"ref": _string()}, ("ref",)),
        ),
        Tool(
            name="memory_search",
            description=SEARCH_DESCRIPTION,
            inputSchema=_object(
                {
                    "query": _string(),
                    "scope": {"type": "string", "enum": ["project", "global", "all"], "default": "all"},
                    "cursor": _string(),
                },
                ("query",),
            ),
        ),
        Tool(name="memory_catalog", description=CATALOG_DESCRIPTION, inputSchema=_object({})),
    ]
    if writable:
        tools.insert(
            0,
            Tool(
                name="memory_save",
                description=_INTENT_SAVE_DESCRIPTION if intent_only else SAVE_DESCRIPTION,
                inputSchema=save,
            ),
        )
        tools.insert(
            3,
            Tool(
                name="memory_forget",
                description=FORGET_DESCRIPTION,
                inputSchema=_object(
                    {
                        "ref": _string(),
                        "version": _string(),
                        "user_quote": _string(limit=MAX_QUOTE, minimum=1),
                        "candidate": _string(_CANDIDATE_DESCRIPTION),
                    },
                    ("ref", "version", "user_quote"),
                ),
            ),
        )
    if writable and not intent_only:
        tools.append(
            Tool(
                name="memory_withdraw",
                description=WITHDRAW_DESCRIPTION,
                inputSchema=_object(
                    {
                        "ref": _string(),
                        "version": _string(),
                        "user_quote": _string("Copy the user's withdrawal verbatim.", limit=MAX_QUOTE, minimum=1),
                        "session": _string("The full source session; required in batch mode."),
                        "candidate": _string(_CANDIDATE_DESCRIPTION),
                    },
                    ("ref", "version", "user_quote"),
                ),
            )
        )
        if review:
            tools.extend(
                [
                    Tool(
                        name="memory_merge",
                        description=MERGE_DESCRIPTION,
                        inputSchema=_object(
                            {
                                "keep_ref": _string(),
                                "keep_version": _string(),
                                "drop_ref": _string(),
                                "drop_version": _string(),
                            },
                            ("keep_ref", "keep_version", "drop_ref", "drop_version"),
                        ),
                    ),
                    Tool(
                        name="memory_note_conflict",
                        description=NOTE_CONFLICT_DESCRIPTION,
                        inputSchema=_object({"ref_a": _string(), "ref_b": _string()}, ("ref_a", "ref_b")),
                    ),
                ]
            )
    if created_rewrites or describe_refs:
        for tool in tools:
            properties = tool.inputSchema["properties"]
            if created_rewrites and tool.name == "memory_save":
                for name in ("ref", "version"):
                    properties[name]["description"] = _CREATED_UPDATE_DESCRIPTION
            if describe_refs:
                for name in ("ref", "keep_ref", "drop_ref", "ref_a", "ref_b"):
                    if name in properties:
                        previous = properties[name].get("description", "")
                        properties[name]["description"] = (previous + " " if previous else "") + _REF_DESCRIPTION
    return tools


@dataclass(frozen=True, slots=True)
class ReadRequest:
    ref: str


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query: str
    scope: SearchScope = "all"
    cursor: str | None = None


@dataclass(frozen=True, slots=True)
class ForgetRequest:
    ref: str
    version: str
    quote: QuoteClaim


@dataclass(frozen=True, slots=True)
class CatalogRequest:
    pass


@dataclass(frozen=True, slots=True)
class WithdrawRequest:
    ref: str
    version: str
    quote: QuoteClaim


@dataclass(frozen=True, slots=True)
class MergeRequest:
    keep_ref: str
    keep_version: str
    drop_ref: str
    drop_version: str


@dataclass(frozen=True, slots=True)
class NoteConflictRequest:
    ref_a: str
    ref_b: str


Request = (
    SaveRequest
    | ReadRequest
    | SearchRequest
    | ForgetRequest
    | CatalogRequest
    | WithdrawRequest
    | MergeRequest
    | NoteConflictRequest
)


def _invalid(problem: str | None = None) -> MemoryInputRejected:
    # Name only schema-owned field names and lengths; never echo untrusted
    # argument names, values or validation-library exceptions.
    detail = f" {problem}." if problem else ""
    return MemoryInputRejected(
        "memory_invalid_input", f"Invalid memory tool arguments.{detail} Follow the tool's schema."
    )


def _fields(value: object, allowed: set[str], required: set[str], label: str = "arguments") -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise _invalid(f"{label} must be an object")
    if missing := sorted(required - set(value)):
        raise _invalid(f"{label} is missing {', '.join(missing)}")
    if set(value) - allowed:
        raise _invalid(f"{label} has fields the schema does not allow")
    return dict(value)


def _text(
    fields: dict[str, object], key: str, *, maximum: int | None = None, minimum: int = 0, label: str | None = None
) -> str:
    value, name = fields[key], label or key
    if not isinstance(value, str):
        raise _invalid(f"{name} must be a string")
    if len(value) < minimum:
        raise _invalid(f"{name} must not be empty")
    if maximum is not None and len(value) > maximum:
        raise _invalid(f"{name} is {len(value)} characters; the limit is {maximum}")
    return value


def _optional(
    fields: dict[str, object], key: str, *, maximum: int | None = None, label: str | None = None
) -> str | None:
    return _text(fields, key, maximum=maximum, label=label) if key in fields else None


def _evidence(raw: object) -> tuple[EvidenceClaim, ...]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_EVIDENCE:
        raise _invalid(f"evidence must be a list of 1 to {MAX_EVIDENCE} items")
    claims: list[EvidenceClaim] = []
    for index, value in enumerate(raw):
        path = f"evidence[{index}]"
        if not isinstance(value, dict):
            raise _invalid(f"{path} must be an object")
        kind = value.get("kind")
        if kind in ("tool_result", "user_report"):
            item = _fields(value, {"kind", "quote", "candidate", "session"}, {"kind", "quote"}, path)
            quote = _text(
                item,
                "quote",
                minimum=1,
                maximum=MAX_EXCERPT if kind == "tool_result" else MAX_QUOTE,
                label=path + ".quote",
            )
            candidate = _optional(item, "candidate", label=path + ".candidate")
            session = _optional(item, "session", label=path + ".session")
            claims.append(
                ToolResultClaim(quote, candidate, session)
                if kind == "tool_result"
                else UserReportClaim(quote, candidate, session)
            )
        elif kind == "file":
            item = _fields(value, {"kind", "path"}, {"kind", "path"}, path)
            claims.append(FileClaim(_text(item, "path", minimum=1, label=path + ".path")))
        elif kind == "reference":
            item = _fields(value, {"kind", "ref", "note"}, {"kind", "ref"}, path)
            claims.append(
                ReferenceClaim(
                    _text(item, "ref", minimum=1, label=path + ".ref"), _optional(item, "note", label=path + ".note")
                )
            )
        else:
            raise _invalid(f"{path}.kind is not an allowed evidence kind")
    return tuple(claims)


def decode_request(name: str, arguments: object) -> Request:
    """Reject null, extra fields and coercions before invoking any context provider."""
    if name == "memory_read":
        fields = _fields(arguments, {"ref"}, {"ref"})
        return ReadRequest(_text(fields, "ref"))
    if name == "memory_search":
        fields = _fields(arguments, {"query", "scope", "cursor"}, {"query"})
        scope = _text(fields, "scope") if "scope" in fields else "all"
        if scope not in ("project", "global", "all"):
            raise _invalid("scope must be project, global or all")
        return SearchRequest(_text(fields, "query"), scope, _optional(fields, "cursor"))
    if name == "memory_forget":
        fields = _fields(arguments, {"ref", "version", "user_quote", "candidate"}, {"ref", "version", "user_quote"})
        quote = QuoteClaim(
            "user_quote",
            "candidate",
            _text(fields, "user_quote", minimum=1, maximum=MAX_QUOTE),
            _optional(fields, "candidate"),
        )
        return ForgetRequest(_text(fields, "ref"), _text(fields, "version"), quote)
    if name == "memory_withdraw":
        fields = _fields(
            arguments, {"ref", "version", "user_quote", "session", "candidate"}, {"ref", "version", "user_quote"}
        )
        quote = QuoteClaim(
            "user_quote",
            "candidate",
            _text(fields, "user_quote", minimum=1, maximum=MAX_QUOTE),
            _optional(fields, "candidate"),
            _optional(fields, "session"),
        )
        return WithdrawRequest(_text(fields, "ref"), _text(fields, "version"), quote)
    if name == "memory_merge":
        fields = _fields(
            arguments,
            {"keep_ref", "keep_version", "drop_ref", "drop_version"},
            {"keep_ref", "keep_version", "drop_ref", "drop_version"},
        )
        return MergeRequest(
            _text(fields, "keep_ref"),
            _text(fields, "keep_version"),
            _text(fields, "drop_ref"),
            _text(fields, "drop_version"),
        )
    if name == "memory_note_conflict":
        fields = _fields(arguments, {"ref_a", "ref_b"}, {"ref_a", "ref_b"})
        return NoteConflictRequest(_text(fields, "ref_a"), _text(fields, "ref_b"))
    if name == "memory_catalog":
        _fields(arguments, set(), set())
        return CatalogRequest()
    if name != "memory_save":
        raise MemoryOperationError("memory_invalid_input", "Unknown memory tool.")
    fields = _fields(
        arguments,
        {
            "type",
            "title",
            "body",
            "applies_when",
            "scope",
            "user_quotes",
            "ref",
            "version",
            "evidence",
            "context_quote",
            "context_candidate",
            "context_session",
        },
        {"type", "title", "body"},
    )
    type = _text(fields, "type")
    if type not in ("intent", "engineering"):
        raise _invalid("type must be intent or engineering")
    if type == "intent" and "evidence" in fields:
        raise _invalid("an intent entry takes user_quotes, not evidence")
    if type == "engineering" and "user_quotes" in fields:
        raise _invalid("an engineering entry takes evidence, not user_quotes")
    if ("context_candidate" in fields or "context_session" in fields) and "context_quote" not in fields:
        raise _invalid("context_candidate and context_session need context_quote")
    scope = _optional(fields, "scope")
    if scope not in (None, "project", "global"):
        raise _invalid("scope must be project or global")
    title = _text(fields, "title", minimum=1, maximum=MAX_TITLE)
    if "\r" in title or "\n" in title:
        raise _invalid("title must be one line")
    quotes: tuple[QuoteClaim, ...] | None = None
    if "user_quotes" in fields:
        raw = fields["user_quotes"]
        if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_QUOTES:
            raise _invalid(f"user_quotes must be a list of 1 to {MAX_QUOTES} items")
        claims: list[QuoteClaim] = []
        for index, value in enumerate(raw):
            path = f"user_quotes[{index}]"
            item = _fields(value, {"text", "candidate", "session"}, {"text"}, path)
            claims.append(
                QuoteClaim(
                    path,
                    path + ".candidate",
                    _text(item, "text", minimum=1, maximum=MAX_DIRECT_QUOTE, label=path + ".text"),
                    _optional(item, "candidate", label=path + ".candidate"),
                    _optional(item, "session", label=path + ".session"),
                )
            )
        quotes = tuple(claims)
    ref, version = _optional(fields, "ref"), _optional(fields, "version")
    if (ref is None) != (version is None):
        raise _invalid("ref and version must be given together")
    return SaveRequest(
        type,
        title,
        _text(fields, "body", minimum=1, maximum=MAX_BODY),
        _optional(fields, "applies_when", maximum=MAX_APPLIES_WHEN),
        scope,
        quotes,
        ref,
        version,
        _evidence(fields["evidence"]) if "evidence" in fields else None,
        _text(fields, "context_quote", maximum=MAX_CONTEXT, minimum=1) if "context_quote" in fields else None,
        _optional(fields, "context_candidate"),
        _text(fields, "context_session", minimum=1) if "context_session" in fields else None,
    )
