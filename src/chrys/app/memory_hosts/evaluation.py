# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Read-only memory and observed host events shared by evaluation commands."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from chrys.app.memory_hosts.claude_code.chain import InvalidChain
from chrys.app.memory_hosts.claude_code.conversation import _main
from chrys.app.memory_hosts.claude_code.sources import OccurrenceGraph, SourceRecord, blocks
from chrys.app.memory_hosts.codex_history import _ROOT_FIELDS
from chrys.app.memory_hosts.history import HistoryError, JsonObject, object_value, read_snapshot
from chrys.foundation.platform.files import secure_open_regular_binary
from chrys.service.memory.change_log import ChangeOperation, MemoryChange
from chrys.service.memory.extraction import SessionKey
from chrys.service.memory.project import bind_project
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.layout import scope_directory
from chrys.service.memory.storage.store import (
    GLOBAL_SCOPE,
    MEMORY_TYPES,
    WRITERS,
    MemoryOperationError,
    MemoryScope,
    StoredMemory,
    is_zoned_timestamp,
    parse_ref,
)

_OPERATIONS: tuple[ChangeOperation, ...] = ("create", "update", "withdraw", "restore", "merge", "delete")
_CHANGE_FIELDS = frozenset(
    {"time", "scope", "ref", "type", "operation", "written_by", "source", "before_version", "after_version"}
)
_LOG_NAME = re.compile(r"changes(?:\.[1-9][0-9]*)?\.jsonl")
_CATALOG = re.compile(r"<memory_catalog>(.*?)</memory_catalog>", re.DOTALL)
_CATALOG_LINE = re.compile(r"^\s*-\s*((?:project|global):[0-9a-f]{12})(?![0-9a-f])", re.MULTILINE)
_REF = re.compile(r"\b(?:project|global):[0-9a-f]{12}(?![0-9A-Fa-f])")


@dataclass(frozen=True, slots=True)
class EvaluationDiagnostic:
    """A skipped observation's shape and location, never its contents."""

    kind: str
    file: str | None = None
    position: int | None = None
    count: int = 1


@dataclass(frozen=True, slots=True)
class ScopedEntry:
    scope: MemoryScope
    stored: StoredMemory


@dataclass(frozen=True, slots=True)
class StoreSnapshot:
    entries: tuple[ScopedEntry, ...] = ()
    changes: tuple[MemoryChange, ...] = ()
    diagnostics: tuple[EvaluationDiagnostic, ...] = ()


async def scan_store(root: Path, project: MemoryScope | None = None) -> StoreSnapshot:
    """Read valid entries and every numbered change-log archive without creating paths."""
    store = FileMemoryStore(root)
    scopes = [GLOBAL_SCOPE]
    if project is not None:
        scopes.append(project)
    else:
        scopes.extend(MemoryScope("project", item.project_id) for item in await store.list_projects())
    entries: list[ScopedEntry] = []
    changes: list[MemoryChange] = []
    diagnostics: list[EvaluationDiagnostic] = []
    for scope in scopes:
        entries.extend(ScopedEntry(scope, item) for item in await store.list_entries(scope))
        diagnostics.extend(EvaluationDiagnostic("unreadable_entry", item.ref) for item in await store.unreadable(scope))
        records, notes = await asyncio.to_thread(_scope_changes, root, scope)
        changes.extend(records)
        diagnostics.extend(notes)
    return StoreSnapshot(tuple(entries), tuple(changes), tuple(diagnostics))


def _scope_changes(root: Path, scope: MemoryScope) -> tuple[list[MemoryChange], list[EvaluationDiagnostic]]:
    directory = scope_directory(root, scope)
    if not directory.is_dir():
        return [], []
    changes: list[MemoryChange] = []
    diagnostics: list[EvaluationDiagnostic] = []
    for path in sorted(directory.iterdir()):
        if _LOG_NAME.fullmatch(path.name) is None or not path.is_file():
            continue
        try:
            with secure_open_regular_binary(path) as handle:
                for position, line in enumerate(handle, 1):
                    change = _change(line, scope)
                    if change is None:
                        diagnostics.append(EvaluationDiagnostic("invalid_change_record", path.name, position))
                    else:
                        changes.append(change)
        except OSError, ValueError:
            diagnostics.append(EvaluationDiagnostic("change_log_unavailable", path.name))
    return changes, diagnostics


def _change(line: bytes, scope: MemoryScope) -> MemoryChange | None:
    try:
        value = object_value(json.loads(line))
        if value is None or set(value) != _CHANGE_FIELDS:
            return None
        scope_value = object_value(value.get("scope"))
        expected = {"scope": scope.scope}
        if scope.scope == "project":
            expected["project_id"] = scope.project_id
        if scope_value != expected:
            return None
        time, ref, source = value["time"], value["ref"], value["source"]
        type_ = _choice(value["type"], MEMORY_TYPES)
        operation = _choice(value["operation"], _OPERATIONS)
        writer = _choice(value["written_by"], WRITERS)
        before, after = value["before_version"], value["after_version"]
        if not isinstance(time, str) or not is_zoned_timestamp(time):
            return None
        if not isinstance(ref, str) or parse_ref(ref)[0] != scope.scope:
            return None
        if not isinstance(source, str) or not source:
            return None
        if type_ is None or operation is None or writer is None:
            return None
        if (before is not None and not isinstance(before, str)) or (after is not None and not isinstance(after, str)):
            return None
        return MemoryChange(time, scope, ref, type_, operation, writer, source, before, after)
    except ValueError, UnicodeError, RecursionError, MemoryOperationError:
        return None


def _choice[T](value: object, choices: tuple[T, ...]) -> T | None:
    return next((choice for choice in choices if choice == value), None)


@dataclass(frozen=True, slots=True)
class SessionObservation:
    key: SessionKey
    shown: frozenset[str]
    read: frozenset[str]
    search: frozenset[str]


@dataclass(frozen=True, slots=True)
class UsageScan:
    sessions: tuple[SessionObservation, ...] = ()
    diagnostics: tuple[EvaluationDiagnostic, ...] = ()


def _texts(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            nested = json.loads(value)
        except ValueError, RecursionError:
            return (value,)
        return _texts(nested) if isinstance(nested, dict | list) else (value,)
    if isinstance(value, list):
        return tuple(text for item in value for text in _texts(item))
    if isinstance(value, dict):
        return tuple(text for item in value.values() for text in _texts(item))
    return ()


def _catalog_refs(value: object) -> frozenset[str]:
    return frozenset(
        match.group(1)
        for text in _texts(value)
        for catalog in _CATALOG.finditer(text)
        for match in _CATALOG_LINE.finditer(catalog.group(1))
    )


def _read_refs(value: object) -> frozenset[str]:
    return frozenset(match.group() for text in _texts(value) for match in _REF.finditer(text))


def _mapped(cwd: object, project: MemoryScope) -> bool:
    if not isinstance(cwd, str) or not cwd:
        return False
    try:
        return bind_project(cwd).project_id == project.project_id
    except MemoryOperationError:
        return False


def _own_main(record: JsonObject, session: str) -> bool:
    return _main(record) and record.get("sessionId") in (None, session)


@dataclass(slots=True)
class _ObservedCall:
    name: str
    arguments: object
    eligible: bool
    answered: bool = False
    kind: str = ""


def _claude_usage(
    records: Sequence[tuple[int, JsonObject]], session: str
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    source_records: list[SourceRecord] = []
    segment = 0
    for position, record in records:
        if record.get("type") == "system" and record.get("subtype") == "compact_boundary":
            segment += 1
        if isinstance(record.get("uuid"), str) and record.get("uuid"):
            source_records.append(SourceRecord(position, record, segment))
    graph = OccurrenceGraph(source_records)
    declarations: dict[int, list[tuple[str, _ObservedCall]]] = {}
    shown: set[str] = set()
    read: set[str] = set()
    search: set[str] = set()
    for node in source_records:
        record = node.record
        eligible = _own_main(record, session)
        if eligible and record.get("type") == "attachment":
            attachment = object_value(record.get("attachment"))
            if attachment is not None and attachment.get("type") == "hook_additional_context":
                shown.update(_catalog_refs(attachment.get("content")))
        for _, block in blocks(record):
            if record.get("type") == "assistant" and block.get("type") == "tool_use":
                call_id, name = block.get("id"), block.get("name")
                if not isinstance(call_id, str) or not isinstance(name, str):
                    continue
                name = name.rsplit("__", 1)[-1]
                call = _ObservedCall(name, block.get("input"), eligible)
                declarations.setdefault(node.position, []).append((call_id, call))
                if eligible and name in {"memory_read", "memory_search"}:
                    (read if name == "memory_read" else search).update(_read_refs(call.arguments))
            elif eligible and record.get("type") == "user" and block.get("type") == "tool_result":
                call_id = block.get("tool_use_id")
                try:
                    parent = graph.parent(node)
                    seen: set[int] = set()
                    while parent is not None and parent.position not in seen:
                        seen.add(parent.position)
                        matches = [
                            call for identity, call in declarations.get(parent.position, ()) if identity == call_id
                        ]
                        if matches:
                            call = matches[0]
                            if len(matches) == 1 and call.eligible and not call.answered:
                                call.answered = True
                                if call.name in {"memory_read", "memory_search"}:
                                    (read if call.name == "memory_read" else search).update(
                                        _read_refs(block.get("content"))
                                    )
                            break
                        parent = graph.parent(parent)
                except InvalidChain:
                    continue
    return frozenset(shown), frozenset(read), frozenset(search)


def _codex_usage(records: Sequence[tuple[int, JsonObject]]) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    shown: set[str] = set()
    read: set[str] = set()
    search: set[str] = set()
    calls: dict[str, _ObservedCall | None] = {}
    for _, record in records:
        payload = object_value(record.get("payload"))
        if payload is None:
            continue
        if record.get("type") == "event_msg" and payload.get("type") == "task_started":
            calls.clear()
            continue
        if record.get("type") != "response_item":
            continue
        kind, identity = payload.get("type"), payload.get("call_id")
        if not isinstance(identity, str):
            continue
        if kind in {"function_call", "custom_tool_call"}:
            name = payload.get("name")
            name = name.rsplit("__", 1)[-1] if isinstance(name, str) else ""
            arguments = payload.get("arguments" if kind == "function_call" else "input")
            if name in {"memory_read", "memory_search"}:
                (read if name == "memory_read" else search).update(_read_refs(arguments))
            prior = calls.get(identity)
            if identity in calls and (prior is None or not prior.answered):
                calls[identity] = None
            else:
                calls[identity] = _ObservedCall(name, arguments, True, kind=str(kind))
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            call = calls.get(identity)
            if call is None or call.answered:
                continue
            expected = "custom_tool_call" if kind == "custom_tool_call_output" else "function_call"
            if call.kind != expected:
                continue
            call.answered = True
            if call.name == "memory_catalog":
                shown.update(_catalog_refs(payload.get("output")))
            elif call.name in {"memory_read", "memory_search"}:
                (read if call.name == "memory_read" else search).update(_read_refs(payload.get("output")))
    return frozenset(shown), frozenset(read), frozenset(search)


async def scan_usage(
    project: MemoryScope, claude_source: Path | None = None, codex_source: Path | None = None
) -> UsageScan:
    """Count observations, including active/abandoned main-session events, without holder probes."""
    return await asyncio.to_thread(_scan_usage, project, claude_source, codex_source)


def _scan_usage(project: MemoryScope, claude_source: Path | None, codex_source: Path | None) -> UsageScan:
    sessions: dict[SessionKey, SessionObservation] = {}
    diagnostics: list[EvaluationDiagnostic] = []
    for host, source in (("claude-code", claude_source), ("codex", codex_source)):
        if source is None:
            continue
        paths = sorted(source.glob("*.jsonl") if host == "claude-code" else source.rglob("*.jsonl"))
        for path in paths:
            try:
                snapshot = read_snapshot(path)
                if host == "claude-code":
                    UUID(path.stem)
                    session = path.stem
                    start = next(
                        (
                            record
                            for _, record in snapshot.records
                            if _own_main(record, session)
                            and record.get("type") == "user"
                            and record.get("sessionId") == session
                        ),
                        None,
                    )
                    cwd = start.get("cwd") if start is not None else None
                else:
                    metadata = next(
                        (
                            object_value(record.get("payload"))
                            for _, record in snapshot.records
                            if record.get("type") == "session_meta"
                        ),
                        None,
                    )
                    session_value = metadata.get("id") if metadata is not None else None
                    if isinstance(session_value, str):
                        UUID(session_value)
                    session = session_value if isinstance(session_value, str) else ""
                    cwd = metadata.get("cwd") if metadata is not None else None
                    if metadata is not None and any(metadata.get(field) is not None for field in _ROOT_FIELDS):
                        diagnostics.append(EvaluationDiagnostic("not_root", path.name))
                        continue
                if not session:
                    diagnostics.append(EvaluationDiagnostic("unknown_session", path.name))
                    continue
                if not _mapped(cwd, project):
                    diagnostics.append(EvaluationDiagnostic("source_mapping_unconfirmed", path.name))
                    continue
                shown, read, search = (
                    _claude_usage(snapshot.records, session)
                    if host == "claude-code"
                    else _codex_usage(snapshot.records)
                )
                key = SessionKey(host, session)
                previous = sessions.get(key)
                sessions[key] = SessionObservation(
                    key,
                    shown | previous.shown if previous is not None else shown,
                    read | previous.read if previous is not None else read,
                    search | previous.search if previous is not None else search,
                )
                for kind, count in (
                    ("sidechain_records_excluded", sum(not _main(record) for _, record in snapshot.records)),
                    (
                        "inherited_records_excluded",
                        sum(record.get("sessionId") not in (None, session) for _, record in snapshot.records),
                    ),
                ):
                    if count:
                        diagnostics.append(EvaluationDiagnostic(kind, path.name, count=count))
            except HistoryError as error:
                diagnostics.append(EvaluationDiagnostic(error.kind, path.name))
            except ValueError:
                diagnostics.append(EvaluationDiagnostic("unknown_session", path.name))
    return UsageScan(
        tuple(sorted(sessions.values(), key=lambda item: (item.key.host, item.key.session))), tuple(diagnostics)
    )
