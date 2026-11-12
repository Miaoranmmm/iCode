# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Ended Codex Desktop roots with independently verified user text blocks."""

from __future__ import annotations

import asyncio
import json
import ntpath
import os
import re
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from chrys.app.memory_hosts.claude_code.conversation import _utc_timestamp
from chrys.app.memory_hosts.claude_code.holder import Activity
from chrys.app.memory_hosts.claude_code.sources import arguments, result_fragments
from chrys.app.memory_hosts.history import HistoryError, HistorySnapshot, object_value, read_snapshot
from chrys.service.memory.extraction import ScanDiagnostic, ScanResult, SessionKey, SourceRevision, SourceSession
from chrys.service.memory.names import is_memory_tool
from chrys.service.memory.project import bind_project
from chrys.service.memory.related_refs import call_refs, result_refs
from chrys.service.memory.storage.store import MemoryOperationError, MemoryScope
from chrys.service.memory.verification import SourceKind, SourceMessage

_TAIL_UUID = re.compile(r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$")
_ROOT_FIELDS = (
    "parent_thread_id",
    "forked_from_id",
    "agent_path",
    "agent_nickname",
    "subagent_history_start_ordinal",
    "history_base",
    "agent_role",
)
_REPLACEMENTS = frozenset({"compacted", "thread_rolled_back"})


@dataclass(frozen=True, slots=True)
class ProcessExecutable:
    """Only the identity needed for a same-user executable check, never argv."""

    user: str | int | None
    exe: str | None


class ProcessTable(Protocol):
    """A complete process snapshot with an explicit current-user identity."""

    def current_user(self) -> str | int | None: ...

    def snapshot(self) -> tuple[ProcessExecutable, ...]: ...


class SystemProcessTable:
    """Reuse psutil enumeration without inspecting command-line data."""

    def current_user(self) -> str | int | None:
        import psutil

        if sys.platform != "win32":
            return os.getuid()
        try:
            return psutil.Process().username()
        except psutil.Error:
            return None

    def snapshot(self) -> tuple[ProcessExecutable, ...]:
        import psutil

        current = self.current_user()
        result: list[ProcessExecutable] = []
        try:
            for process in psutil.process_iter():
                try:
                    user = process.username() if sys.platform == "win32" else process.uids().real
                    exe = process.exe() if user == current else None
                    result.append(ProcessExecutable(user, exe))
                except psutil.NoSuchProcess:
                    continue
                except psutil.Error:
                    result.append(ProcessExecutable(None, None))
        except psutil.Error:
            raise OSError("Process enumeration unavailable.") from None
        return tuple(result)


class CodexActivity(Protocol):
    """Check all Codex hosts owned by the current user."""

    def check(self) -> Activity: ...


class CodexProcessCheck:
    """A live or uninspectable same-user host prevents history extraction."""

    def __init__(self, table: ProcessTable | None = None) -> None:
        self.table = table or SystemProcessTable()

    def check(self) -> Activity:
        try:
            user = self.table.current_user()
            if user is None:
                return "unknown"
            unknown = False
            for process in self.table.snapshot():
                if process.user is None:
                    unknown = True
                    continue
                if process.user != user:
                    continue
                if not process.exe:
                    unknown = True
                    continue
                components = process.exe.replace("\\", "/").split("/")
                if ntpath.basename(process.exe).casefold() in {"codex", "codex.exe"} or any(
                    component.casefold() == "codex.app" for component in components
                ):
                    return "active"
            return "unknown" if unknown else "inactive"
        except OSError, ValueError:
            return "unknown"


def _user_source(
    payload: dict[str, object],
    record: dict[str, object],
    position: int,
    session: str,
    key: SessionKey,
    diagnostics: list[ScanDiagnostic],
) -> SourceMessage | None:
    content = payload.get("content")
    meta = object_value(payload.get("internal_chat_message_metadata_passthrough"))
    kinds = meta.get("content_item_kinds") if meta is not None else None
    item_metadata = meta.get("content_item_metadata") if meta is not None else None
    if (
        not isinstance(content, list)
        or not isinstance(kinds, list)
        or not isinstance(item_metadata, list)
        or len(content) != len(kinds)
        or len(content) != len(item_metadata)
    ):
        diagnostics.append(ScanDiagnostic("missing_provenance", key, details=(("position", str(position)),)))
        return None
    fragments: list[str] = []
    for block, kind, block_meta in zip(content, kinds, item_metadata, strict=True):
        item = object_value(block)
        information = object_value(block_meta)
        provenance = object_value(information.get("provenance")) if information is not None else None
        provenance_type = provenance.get("type") if provenance is not None else None
        if not isinstance(provenance_type, str) or not provenance_type.strip():
            diagnostics.append(ScanDiagnostic("missing_provenance", key, details=(("position", str(position)),)))
            continue
        if (
            item is not None
            and item.get("type") == "input_text"
            and kind == "user.text"
            and provenance_type == "user"
            and isinstance(text := item.get("text"), str)
            and text
        ):
            fragments.append(text)
    if not fragments:
        return None
    return SourceMessage(
        session, position, tuple(fragments), item=_item(payload, position), at=_utc_timestamp(record.get("timestamp"))
    )


def _item(payload: dict[str, object], position: int) -> str:
    value = payload.get("id")
    return value if isinstance(value, str) and value else f"line:{position}"


@dataclass(frozen=True, slots=True)
class _CodexCall:
    tool: str
    args: str
    kind: str
    answered: bool = False


def _call_arguments(payload: dict[str, object], custom: bool) -> str:
    value = payload.get("input" if custom else "arguments")
    if not custom and isinstance(value, str):
        with suppress(ValueError):
            value = json.loads(value)
    return arguments(value)


def decode_codex_history(
    snapshot: HistorySnapshot, session: str, project: MemoryScope
) -> tuple[SourceSession | None, tuple[ScanDiagnostic, ...]]:
    """The first metadata record and each block's provenance are independent gates."""
    key = SessionKey("codex", session)
    diagnostics: list[ScanDiagnostic] = [ScanDiagnostic("files", key), ScanDiagnostic("signals_unavailable", key)]
    metadata = next(
        (object_value(record.get("payload")) for _, record in snapshot.records if record.get("type") == "session_meta"),
        None,
    )
    if metadata is None:
        return None, (*diagnostics, ScanDiagnostic("missing_session_meta", key))
    rejected = False
    if metadata.get("id") != session:
        diagnostics.append(ScanDiagnostic("session_id_mismatch", key))
        rejected = True
    if metadata.get("source") != "vscode" or metadata.get("originator") != "Codex Desktop":
        diagnostics.append(ScanDiagnostic("unsupported_entrypoint", key))
        rejected = True
    non_root = tuple(field for field in _ROOT_FIELDS if metadata.get(field) is not None)
    if non_root:
        diagnostics.append(ScanDiagnostic("not_root", key, details=tuple(("field", field) for field in non_root)))
        rejected = True
    cwd = metadata.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        diagnostics.append(ScanDiagnostic("missing_project", key))
        rejected = True
    else:
        try:
            if bind_project(cwd).project_id != project.project_id:
                diagnostics.append(ScanDiagnostic("different_project", key))
                rejected = True
        except MemoryOperationError:
            diagnostics.append(ScanDiagnostic("project_unavailable", key))
            rejected = True
    if rejected:
        return None, tuple(diagnostics)
    replacements = {str(record.get("type")) for _, record in snapshot.records if record.get("type") in _REPLACEMENTS}
    replacements.update(
        str(payload.get("type"))
        for _, record in snapshot.records
        if (payload := object_value(record.get("payload"))) is not None and payload.get("type") in _REPLACEMENTS
    )
    if replacements:
        return None, (
            *diagnostics,
            ScanDiagnostic(
                "unsupported_history_replacement", key, details=tuple(("type", kind) for kind in sorted(replacements))
            ),
        )
    retained: dict[tuple[SourceKind, str | None, int], SourceMessage] = {}
    calls: dict[str, _CodexCall | None] = {}
    related: dict[str, None] = {}

    def add(source: SourceMessage) -> None:
        retained.setdefault((source.kind, source.item, source.part), source)

    for position, record in snapshot.records:
        payload = object_value(record.get("payload"))
        if payload is None:
            continue
        if record.get("type") == "event_msg" and payload.get("type") == "task_started":
            calls.clear()
            continue
        if record.get("type") != "response_item":
            continue
        kind = payload.get("type")
        if kind == "message" and payload.get("role") == "user":
            source = _user_source(payload, record, position, session, key, diagnostics)
            if source is not None:
                add(source)
        elif kind == "message" and payload.get("role") == "assistant":
            content = payload.get("content")
            if isinstance(content, list):
                for part, block in enumerate(content):
                    item = object_value(block)
                    if (
                        item is not None
                        and item.get("type") in {"output_text", "text"}
                        and isinstance(value := item.get("text"), str)
                        and value
                    ):
                        add(
                            SourceMessage(
                                session,
                                position,
                                (value,),
                                "assistant",
                                part,
                                _item(payload, position),
                                _utc_timestamp(record.get("timestamp")),
                            )
                        )
        elif kind in {"function_call", "custom_tool_call"}:
            call_id, name = payload.get("call_id"), payload.get("name")
            if not isinstance(call_id, str) or not call_id:
                continue
            raw_arguments = payload.get("input" if kind == "custom_tool_call" else "arguments")
            if isinstance(raw_arguments, str):
                with suppress(ValueError):
                    raw_arguments = json.loads(raw_arguments)
            if isinstance(name, str):
                related.update(dict.fromkeys(call_refs(name, raw_arguments)))
            previous = calls.get(call_id)
            if call_id in calls and (previous is None or not previous.answered):
                # Two unanswered declarations cannot identify an output uniquely.
                calls[call_id] = None
                continue
            calls[call_id] = _CodexCall(
                name if isinstance(name, str) else "", _call_arguments(payload, kind == "custom_tool_call"), str(kind)
            )
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            call_id = payload.get("call_id")
            if not isinstance(call_id, str):
                continue
            call = calls.get(call_id)
            expected = "custom_tool_call" if kind == "custom_tool_call_output" else "function_call"
            if call is None or call.answered or call.kind != expected:
                continue
            calls[call_id] = _CodexCall(call.tool, call.args, call.kind, True)
            fragments = result_fragments(payload.get("output"))
            if is_memory_tool(call.tool):
                related.update(dict.fromkeys(result_refs(fragments)))
            elif call.tool and fragments:
                add(
                    SourceMessage(
                        session,
                        position,
                        fragments,
                        "tool_result",
                        0,
                        _item(payload, position),
                        _utc_timestamp(record.get("timestamp")),
                        (("tool", call.tool), ("args", call.args)),
                    )
                )
    messages = tuple(retained.values())
    diagnostics.append(ScanDiagnostic("display_messages", key, len(messages)))
    if not any(message.kind == "user" for message in messages):
        return None, (*diagnostics, ScanDiagnostic("no_user_messages", key))
    return SourceSession(key, project, snapshot.revision, tuple(messages), related_refs=tuple(related)), tuple(
        diagnostics
    )


class CodexHistoryReader:
    """Read recursive rollout snapshots only after every same-user Codex host exits."""

    def __init__(
        self,
        source: Path,
        *,
        processes: CodexActivity | None = None,
        sessions: frozenset[str] = frozenset(),
        replay: bool = False,
    ) -> None:
        self.source = source
        self.processes = processes or CodexProcessCheck()
        self.sessions = sessions
        self.replay = replay
        self.paths: dict[SessionKey, Path] = {}
        self.revisions: dict[SessionKey, SourceRevision] = {}

    async def scan(self, project: MemoryScope) -> ScanResult:
        return await asyncio.to_thread(self._scan, project, None)

    async def read_files(self, project: MemoryScope, paths: tuple[Path, ...]) -> ScanResult:
        return await asyncio.to_thread(self._scan, project, paths)

    def _scan(self, project: MemoryScope, explicit: tuple[Path, ...] | None) -> ScanResult:
        if not self.replay:
            activity = self.processes.check()
            if activity != "inactive":
                return ScanResult(diagnostics=(ScanDiagnostic("codex_hosts_" + activity),), fatal=True)
        try:
            paths = (
                explicit
                if explicit is not None
                else tuple(sorted(self.source.rglob("*.jsonl")))
                if self.source.is_dir()
                else ()
            )
        except OSError:
            return ScanResult(diagnostics=(ScanDiagnostic("source_unavailable"),), fatal=True)
        if not paths:
            return ScanResult(diagnostics=(ScanDiagnostic("no_session_files"),), fatal=True)
        sessions: list[SourceSession] = []
        diagnostics: list[ScanDiagnostic] = [ScanDiagnostic("source_files", count=len(paths))]
        recognized_files = 0
        for path in paths:
            match = _TAIL_UUID.search(path.stem)
            if match is None:
                diagnostics.append(ScanDiagnostic("unknown_format", details=(("file", path.name),)))
                continue
            recognized_files += 1
            session = match[1]
            if self.sessions and session not in self.sessions:
                continue
            key = SessionKey("codex", session)
            self.paths[key] = path
            try:
                snapshot = read_snapshot(path)
                self.revisions[key] = snapshot.revision
                source, notes = decode_codex_history(snapshot, session, project)
                diagnostics.extend(notes)
                if source is not None:
                    sessions.append(source)
            except HistoryError as error:
                diagnostics.append(ScanDiagnostic(error.kind, key, details=error.details))
        if not recognized_files:
            diagnostics.append(ScanDiagnostic("no_session_files"))
        return ScanResult(tuple(sessions), tuple(diagnostics), not recognized_files)

    async def revalidate(self, source: SourceSession) -> str | None:
        return await asyncio.to_thread(self._revalidate, source)

    def _revalidate(self, source: SourceSession) -> str | None:
        path = self.paths.get(source.key)
        if path is None:
            return "source_unavailable"
        if not self.replay and self.processes.check() != "inactive":
            return "codex_hosts_reopened"
        try:
            revision = read_snapshot(path).revision
        except HistoryError as error:
            return error.kind
        if revision != self.revisions.get(source.key):
            return "source_changed"
        return None
