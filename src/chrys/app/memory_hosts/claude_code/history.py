# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Ended Claude Desktop history, bound by its first self-produced user cwd."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from chrys.app.memory_hosts.history import HistoryError, HistorySnapshot, JsonObject, object_value, read_snapshot
from chrys.service.memory.extraction import (
    ScanDiagnostic,
    ScanResult,
    SessionKey,
    SourceAnnotation,
    SourceRevision,
    SourceSession,
)
from chrys.service.memory.project import bind_project
from chrys.service.memory.storage.store import MemoryOperationError, MemoryScope

from .chain import InvalidChain, walk_parent_chain
from .conversation import _boundary, _main, _message, user_fragments
from .holder import ClaudeHolder, NativeClaudeHolder
from .sources import OccurrenceGraph, SourceRecord, collect_sources, human_resets, tool_failure_signals

_ZERO_UUID = "00000000-0000-0000-0000-000000000000"
_INTERRUPTS = frozenset({"[Request interrupted by user]", "[Request interrupted by user for tool use]"})


@dataclass(frozen=True, slots=True)
class _Occurrence:
    segment: int
    uuid: str


@dataclass(frozen=True, slots=True)
class _HistoryNode:
    key: _Occurrence
    position: int
    record: JsonObject


class _HistoryChain:
    """Bridge compact boundaries only through physical, provable occurrences."""

    def __init__(self, snapshot: HistorySnapshot) -> None:
        self.nodes: dict[_Occurrence, _HistoryNode] = {}
        self.boundaries: dict[int, _HistoryNode] = {}
        self.records: list[SourceRecord] = []
        self.diagnostics: list[ScanDiagnostic] = []
        segment = 0
        contents: dict[str, object] = {}
        for position, record in snapshot.records:
            if _boundary(record):
                segment += 1
            uuid = record.get("uuid")
            if isinstance(uuid, str) and uuid:
                self.records.append(SourceRecord(position, record, segment))
            if not _main(record) or record.get("type") == "progress":
                continue
            if not isinstance(uuid, str) or not uuid:
                if record.get("type") in {"user", "assistant"}:
                    raise HistoryError("unknown_format", details=(("position", str(position)),))
                continue
            key = _Occurrence(segment, uuid)
            node = _HistoryNode(key, position, record)
            previous = self.nodes.setdefault(key, node)
            if previous.record != record:
                raise HistoryError("conflicting_uuid", details=(("position", str(position)),))
            if record.get("type") in {"user", "assistant"}:
                content = _message(record).get("content")
                if uuid in contents and contents[uuid] != content:
                    raise HistoryError("conflicting_uuid", details=(("position", str(position)),))
                contents[uuid] = content
            if _boundary(record):
                self.boundaries[segment] = node
        self.last_segment = segment

    def _plain_parent(self, node: _HistoryNode) -> _Occurrence | None:
        value = node.record.get("parentUuid")
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise HistoryError("invalid_parent", details=(("position", str(node.position)),))
        return _Occurrence(node.key.segment, value)

    def _target_returns(self, target: _HistoryNode, boundary: _HistoryNode) -> str:
        def parent(node: _HistoryNode) -> _Occurrence | None:
            return None if node.key == boundary.key else self._plain_parent(node)

        try:
            chain = walk_parent_chain(target.key, self.nodes.__getitem__, parent)
        except InvalidChain, HistoryError:
            return "unknown"
        return "yes" if any(node.key == boundary.key for node in chain) else "no"

    def parent(self, node: _HistoryNode) -> _Occurrence | None:
        if not _boundary(node.record):
            return self._plain_parent(node)
        logical = node.record.get("logicalParentUuid")
        if logical is None:
            metadata = object_value(node.record.get("compactMetadata")) or {}
            logical = metadata.get("logicalParentUuid")
        if not isinstance(logical, str) or not logical:
            raise HistoryError(
                "missing_compact_bridge",
                details=(
                    ("boundary_position", str(node.position)),
                    ("target_position", "null"),
                    ("returns_to_boundary", "unknown"),
                ),
            )
        targets = [item for item in self.nodes.values() if item.key.uuid == logical]
        before = [item for item in targets if item.position < node.position and item.key.segment < node.key.segment]
        if before:
            return max(before, key=lambda item: item.position).key
        after = [item for item in targets if item.position > node.position]
        target = min(after, key=lambda item: item.position) if after else None
        returns = self._target_returns(target, node) if target is not None else "unknown"
        details = (
            ("boundary_position", str(node.position)),
            ("target_position", str(target.position) if target is not None else "null"),
            ("returns_to_boundary", returns),
        )
        if target is None:
            raise HistoryError("missing_compact_bridge", details=details)
        prior = [item for item in self.nodes.values() if item.position < node.position]
        if prior or target.key.segment != node.key.segment or returns != "yes":
            raise HistoryError("unknown_head_boundary", details=details)
        diagnostic = ScanDiagnostic("head_boundary", details=details)
        if diagnostic not in self.diagnostics:
            self.diagnostics.append(diagnostic)
            self.diagnostics.append(ScanDiagnostic("earlier_history_unavailable"))
        return None

    def walk(self, key: _Occurrence) -> list[_HistoryNode]:
        try:
            return walk_parent_chain(key, self.nodes.__getitem__, self.parent)
        except InvalidChain:
            raise HistoryError("invalid_chain") from None

    def _select_tails(self, tails: list[_HistoryNode]) -> list[_HistoryNode]:
        choices = [self.walk(node.key) for node in tails]
        identities = {
            tuple(node.key.uuid for node in reversed(chain) if user_fragments(node.record)) for chain in choices
        }
        if len(identities) != 1:
            raise HistoryError("ambiguous_final_branch")
        return max(choices, key=lambda chain: chain[0].position)

    def _select_after_anchor(self, anchor: _HistoryNode, later: list[_HistoryNode]) -> list[_HistoryNode]:
        # Prove each later occurrence once; shared suffixes end at an already
        # proved descendant rather than walking the same parent chain again.
        proven = {anchor.key}

        def parent_until_proven(node: _HistoryNode) -> _Occurrence | None:
            return None if node.key in proven else self.parent(node)

        for node in later:
            try:
                path = walk_parent_chain(node.key, self.nodes.__getitem__, parent_until_proven)
            except InvalidChain:
                raise HistoryError("invalid_chain") from None
            if path[-1].key not in proven:
                raise HistoryError("ambiguous_final_branch")
            proven.update(item.key for item in path)
        parents = {self.parent(node) for node in later}
        tails = [node for node in later if node.key not in parents]
        return self._select_tails(tails)

    def select(self, snapshot: HistorySnapshot) -> list[_HistoryNode]:
        checkpoints = [
            (position, record) for position, record in snapshot.records if record.get("type") == "last-prompt"
        ]
        checkpoint = checkpoints[-1] if checkpoints else None
        if checkpoint is not None:
            position, record = checkpoint
            value = record.get("leafUuid")
            later = [node for node in self.nodes.values() if node.position > position]
            if record.get("explicit") is True and value is None and not later:
                return []
            if isinstance(value, str) and value and value != _ZERO_UUID:
                matches = [node for node in self.nodes.values() if node.key.uuid == value and node.position <= position]
                if not matches:
                    raise HistoryError("missing_final_anchor")
                anchor = max(matches, key=lambda node: node.position)
                if not later:
                    return self.walk(anchor.key)
                return self._select_after_anchor(anchor, later)
        last = [node for node in self.nodes.values() if node.key.segment == self.last_segment]
        parents = {self._plain_parent(node) for node in last if not _boundary(node.record)}
        terminals = [node for node in last if node.key not in parents and not _boundary(node.record)]
        if not terminals:
            if self.nodes:
                raise HistoryError("invalid_chain")
            return []
        return self._select_tails(terminals)

    def annotations(self, selected: list[_HistoryNode]) -> tuple[SourceAnnotation, ...]:
        annotations: list[SourceAnnotation] = []
        for node in reversed(selected):
            record = node.record
            content = _message(record).get("content")
            single = content if isinstance(content, str) else None
            if isinstance(content, list) and len(content) == 1:
                block = object_value(content[0])
                single = block.get("text") if block is not None and block.get("type") == "text" else None
            origin = object_value(record.get("origin")) or {}
            human = origin.get("kind") == "human" or record.get("turnOrigin") == "human"
            if isinstance(single, str) and single in _INTERRUPTS and not human:
                if (
                    record.get("type") == "user"
                    and record.get("entrypoint") == "claude-desktop"
                    and record.get("userType") == "external"
                    and not record.get("isMeta")
                    and not record.get("isCompactSummary")
                ):
                    annotations.append(SourceAnnotation("user_interrupt", node.position, node.key.uuid))
                else:
                    self.diagnostics.append(
                        ScanDiagnostic("unsupported_interrupt_shape", details=(("position", str(node.position)),))
                    )
        return tuple(annotations)


def _human_marker(record: JsonObject) -> bool:
    return (object_value(record.get("origin")) or {}).get("kind") == "human" or record.get("turnOrigin") == "human"


def decode_claude_history(
    snapshot: HistorySnapshot, session: str, project: MemoryScope
) -> tuple[SourceSession | None, tuple[ScanDiagnostic, ...]]:
    """Apply source gates before collecting the selected original source kinds."""
    key = SessionKey("claude-code", session)
    diagnostics: list[ScanDiagnostic] = [ScanDiagnostic("files", key)]
    raw_dialogue = [
        (position, record) for position, record in snapshot.records if record.get("type") in {"user", "assistant"}
    ]
    dialogue = [(position, record) for position, record in raw_dialogue if _main(record)]
    users = [record for _, record in dialogue if record.get("type") == "user"]
    raw_users = [record for _, record in raw_dialogue if record.get("type") == "user"]
    diagnostics.append(ScanDiagnostic("raw_human_markers", key, sum(_human_marker(record) for record in raw_users)))
    diagnostics.append(ScanDiagnostic("eligible_human", key, sum(bool(user_fragments(record)) for record in raw_users)))
    if not dialogue:
        return None, (*diagnostics, ScanDiagnostic("empty_session", key))
    if any(not isinstance(record.get("sessionId"), str) or not record["sessionId"] for _, record in dialogue):
        return None, (*diagnostics, ScanDiagnostic("unknown_format", key))
    if not any(record.get("sessionId") == session for _, record in dialogue):
        return None, (*diagnostics, ScanDiagnostic("copied_only", key))
    diagnostics.append(ScanDiagnostic("non_copy", key))
    entries = {value if isinstance(value := record.get("entrypoint"), str) else "<missing>" for record in users}
    if entries != {"claude-desktop"}:
        kind = "mixed_entrypoint" if len(entries) > 1 else "unsupported_entrypoint"
        return None, (
            *diagnostics,
            ScanDiagnostic(kind, key, details=tuple(("entrypoint", str(entry)) for entry in sorted(entries))),
        )
    try:
        chain = _HistoryChain(snapshot)
        selected = chain.select(snapshot)
        annotations = chain.annotations(selected)
        selected_keys = {(node.key.segment, node.key.uuid) for node in selected}
        graph = OccurrenceGraph(chain.records)

        def parent_of(node: SourceRecord) -> SourceRecord | None:
            if not _boundary(node.record):
                return graph.parent(node)
            occurrence = _Occurrence(node.segment, node.key[1])
            if (node.segment, node.key[1]) not in selected_keys:
                raise InvalidChain("An unselected compact boundary is not a proven bridge.")
            try:
                target = chain.parent(chain.nodes[occurrence])
                return None if target is None else graph.by_position[chain.nodes[target].position]
            except KeyError, HistoryError:
                raise InvalidChain("Missing proven compact source bridge.") from None

        collection = collect_sources(chain.records, selected_keys, session, parent_of=parent_of, error_attributes=True)
        messages = collection.messages
        failures = tool_failure_signals(collection.results, human_resets(chain.records, selected_keys))
        annotations = tuple(sorted((*annotations, *failures), key=lambda item: (item.position, item.part)))

    except HistoryError as error:
        return None, (*diagnostics, ScanDiagnostic(error.kind, key, details=error.details))
    diagnostics.extend(ScanDiagnostic(item.kind, key, item.count, item.details) for item in chain.diagnostics)
    diagnostics.append(ScanDiagnostic("display_messages", key, len(messages)))
    if not any(message.kind == "user" for message in messages):
        return None, (*diagnostics, ScanDiagnostic("no_user_messages", key))
    return SourceSession(key, project, snapshot.revision, messages, annotations, collection.related_refs), tuple(
        diagnostics
    )


class ClaudeHistoryReader:
    """Read supplied top-level snapshots; only their starting user cwd binds the project."""

    def __init__(
        self,
        source: Path,
        *,
        holder: ClaudeHolder | None = None,
        sessions: frozenset[str] = frozenset(),
        replay: bool = False,
    ) -> None:
        self.source = source
        self.holder = holder or NativeClaudeHolder(Path.home() / ".claude" / "sessions")
        self.sessions = sessions
        self.replay = replay
        self.paths: dict[SessionKey, Path] = {}
        self.revisions: dict[SessionKey, SourceRevision] = {}

    async def scan(self, project: MemoryScope) -> ScanResult:
        return await asyncio.to_thread(self._scan, project, None)

    async def read_files(self, project: MemoryScope, paths: tuple[Path, ...]) -> ScanResult:
        return await asyncio.to_thread(self._scan, project, paths)

    def _scan(self, project: MemoryScope, explicit: tuple[Path, ...] | None) -> ScanResult:
        try:
            paths = (
                explicit
                if explicit is not None
                else tuple(sorted(self.source.glob("*.jsonl")))
                if self.source.is_dir()
                else ()
            )
        except OSError:
            return ScanResult(diagnostics=(ScanDiagnostic("source_unavailable"),), fatal=True)
        if not paths:
            return ScanResult(diagnostics=(ScanDiagnostic("no_session_files"),), fatal=True)
        sessions: list[SourceSession] = []
        diagnostics: list[ScanDiagnostic] = [ScanDiagnostic("source_files", count=len(paths))]
        dialogue_files = 0
        unidentifiable = 0
        recognized_files = 0
        for path in paths:
            session = path.stem
            try:
                UUID(session)
            except ValueError:
                diagnostics.append(ScanDiagnostic("unknown_format", details=(("file", path.name),)))
                continue
            recognized_files += 1
            if self.sessions and session not in self.sessions:
                continue
            key = SessionKey("claude-code", session)
            self.paths[key] = path
            activity = "inactive" if self.replay else self.holder.check(session)
            if activity != "inactive":
                diagnostics.append(ScanDiagnostic(activity, key))
                continue
            try:
                snapshot = read_snapshot(path)
                self.revisions[key] = snapshot.revision
                if any(_main(record) and record.get("type") in {"user", "assistant"} for _, record in snapshot.records):
                    dialogue_files += 1
                source, notes = decode_claude_history(snapshot, session, project)
                diagnostics.extend(notes)
                if any(note.kind in {"copied_only", "unknown_format"} for note in notes):
                    unidentifiable += 1
                start = next(
                    (
                        record
                        for _, record in snapshot.records
                        if _main(record) and record.get("type") == "user" and record.get("sessionId") == session
                    ),
                    None,
                )
                value = start.get("cwd") if start is not None else None
                start_cwd = value if isinstance(value, str) and value else None
                cwds = sorted(
                    {
                        str(record["cwd"])
                        for _, record in snapshot.records
                        if _main(record) and record.get("type") == "user" and isinstance(record.get("cwd"), str)
                    }
                )
                diagnostics.append(
                    ScanDiagnostic(
                        "source_group",
                        key,
                        details=(tuple(("cwd", cwd) for cwd in cwds) or (("cwd", "<unrecorded>"),))
                        + (("start_cwd", start_cwd if start_cwd is not None else "<unrecorded>"),),
                    )
                )
                if source is None:
                    continue
                reason = None
                if start_cwd is None:
                    reason = "no_start"
                else:
                    try:
                        if bind_project(start_cwd).project_id != project.project_id:
                            reason = "other_project"
                    except MemoryOperationError:
                        reason = "unbound"
                if reason is not None:
                    diagnostics.append(ScanDiagnostic("source_mapping_unconfirmed", key, details=(("reason", reason),)))
                    if session not in self.sessions:
                        continue
                if not self.replay and self.holder.check(session) != "inactive":
                    diagnostics.append(ScanDiagnostic("source_reopened", key))
                    continue
                sessions.append(source)
            except HistoryError as error:
                diagnostics.append(ScanDiagnostic(error.kind, key, details=error.details))
        fatal = not recognized_files or bool(dialogue_files and dialogue_files == unidentifiable)
        if not recognized_files:
            diagnostics.append(ScanDiagnostic("no_session_files"))
        return ScanResult(tuple(sessions), tuple(diagnostics), fatal)

    async def revalidate(self, source: SourceSession) -> str | None:
        return await asyncio.to_thread(self._revalidate, source)

    def _revalidate(self, source: SourceSession) -> str | None:
        path = self.paths.get(source.key)
        if path is None:
            return "source_unavailable"
        if not self.replay and self.holder.check(source.key.session) != "inactive":
            return "source_reopened"
        try:
            revision = read_snapshot(path).revision
        except HistoryError as error:
            return error.kind
        if revision != self.revisions.get(source.key):
            return "source_changed"
        return None
