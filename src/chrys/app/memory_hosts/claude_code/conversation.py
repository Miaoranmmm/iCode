# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Read Claude's current transcript chain without treating host text as user words."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Protocol, cast

from chrys.foundation.platform.files import secure_open_regular_binary
from chrys.service.memory.storage.store import MemoryOperationError
from chrys.service.memory.verification import ConversationView, SourceMessage

from .chain import InvalidChain, walk_parent_chain
from .registry import SOURCE_UNAVAILABLE

if TYPE_CHECKING:
    from .registry import Registration

MAX_SEGMENT_BYTES = 128 * 1024 * 1024
MAX_LINE_BYTES = 16 * 1024 * 1024
STOP_WINDOW_BYTES = 8 * 1024 * 1024
STOP_MAX_RECORDS = 10_000
STOP_BUDGET_SECONDS = 0.100
SEGMENT_TOO_LARGE = (
    "Current transcript segment exceeds the 128 MiB memory-verification limit. "
    "Compact or start a new session; read/search remain available."
)

JsonObject = dict[str, object]


class Clock(Protocol):
    """The monotonic clock used for Stop's soft parsing budget."""

    def monotonic(self) -> float: ...


@dataclass(frozen=True, slots=True)
class _Node:
    record: JsonObject
    position: int


@dataclass(slots=True)
class _Segment:
    nodes: dict[str, _Node] = field(default_factory=dict)
    occurrences: list[_Node] = field(default_factory=list)
    fingerprints: dict[str, bytes] = field(default_factory=dict)
    anchor_ids: set[str] = field(default_factory=set)
    boundary: JsonObject | None = None
    bad: bool = False
    oversized: bool = False
    size: int = 0


def _object(value: object) -> JsonObject | None:
    return cast(JsonObject, value) if isinstance(value, dict) else None


def _source_error(message: str = SOURCE_UNAVAILABLE) -> MemoryOperationError:
    return MemoryOperationError("memory_store_unavailable", message)


def _boundary(record: JsonObject) -> bool:
    return record.get("type") == "system" and record.get("subtype") == "compact_boundary"


def _main(record: JsonObject) -> bool:
    return not record.get("isSidechain") and not record.get("agentId")


def _message(record: JsonObject) -> JsonObject:
    return _object(record.get("message")) or {}


def _tool_call(record: JsonObject, tool_use_id: str) -> bool:
    if record.get("type") != "assistant" or not _main(record):
        return False
    content = _message(record).get("content")
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id") == tool_use_id
        for block in content
    )


def _snapshot_lines(handle: BinaryIO, size: int) -> Iterator[tuple[int, bytes | None, int]]:
    """Yield complete lines of the frozen file extent, bounding each allocation."""
    remaining = size
    position = 0
    while remaining:
        chunk = handle.readline(min(MAX_LINE_BYTES + 1, remaining))
        if not chunk:
            break
        remaining -= len(chunk)
        length = len(chunk)
        too_long = length > MAX_LINE_BYTES
        if not chunk.endswith(b"\n") and remaining:
            too_long = True
            while remaining:
                extra = handle.readline(min(64 * 1024, remaining))
                if not extra:
                    remaining = 0
                    break
                remaining -= len(extra)
                length += len(extra)
                if extra.endswith(b"\n"):
                    break
            else:
                break
            if not extra.endswith(b"\n"):
                break
        elif not chunk.endswith(b"\n"):
            break
        position += 1
        yield position, None if too_long else chunk, length


def _decode(line: bytes) -> JsonObject | None:
    try:
        return _object(json.loads(line))
    except ValueError, UnicodeError, RecursionError:
        return None


def _add_node(segment: _Segment, record: JsonObject, position: int, *, retain: bool) -> None:
    uuid = record.get("uuid")
    if not isinstance(uuid, str) or not uuid:
        return
    try:
        encoded = json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    except RecursionError:
        segment.bad = True
        return
    fingerprint = hashlib.sha256(encoded.encode("utf-8")).digest()
    previous = segment.fingerprints.setdefault(uuid, fingerprint)
    if previous != fingerprint:
        segment.bad = True
    if retain:
        node = _Node(record, position)
        segment.nodes.setdefault(uuid, node)
        segment.occurrences.append(node)


def _scan(registration: Registration, tool_use_id: str, *, retain: bool) -> _Segment:
    segment = _Segment()
    try:
        with secure_open_regular_binary(Path(registration.transcript_path)) as handle:
            size = os.fstat(handle.fileno()).st_size
            for position, line, length in _snapshot_lines(handle, size):
                record = _decode(line) if line is not None else None
                if record is not None and _boundary(record):
                    segment = _Segment(boundary=record)
                segment.size += length
                if retain and segment.size > MAX_SEGMENT_BYTES:
                    segment.oversized = True
                    segment.nodes.clear()
                    segment.occurrences.clear()
                    segment.fingerprints.clear()
                if record is None:
                    segment.bad = True
                    continue
                if segment.oversized:
                    continue
                _add_node(segment, record, position, retain=retain)
                if _tool_call(record, tool_use_id):
                    uuid = record.get("uuid")
                    if not isinstance(uuid, str) or not uuid:
                        segment.bad = True
                    else:
                        segment.anchor_ids.add(uuid)
                        segment.nodes.setdefault(uuid, _Node(record, position))
    except OSError, ValueError:
        raise _source_error() from None
    return segment


def _anchor(segment: _Segment, registration: Registration, tool_use_id: str) -> str:
    if segment.oversized:
        raise _source_error(SEGMENT_TOO_LARGE)
    if not tool_use_id or segment.bad or len(segment.anchor_ids) != 1:
        raise _source_error()
    uuid = next(iter(segment.anchor_ids))
    record = segment.nodes[uuid].record
    if record.get("sessionId") != registration.session_id or "forkedFrom" in record:
        raise _source_error()
    if segment.boundary is not None:
        metadata = _object(segment.boundary.get("compactMetadata"))
        preserved = _object(metadata.get("preservedMessages")) if metadata is not None else None
        all_uuids = preserved.get("allUuids") if preserved is not None else None
        if (
            not isinstance(all_uuids, list)
            or not all_uuids
            or any(not isinstance(item, str) or not item for item in all_uuids)
        ):
            raise _source_error()
        if uuid in all_uuids:
            raise _source_error()
    return uuid


def validate_anchor(registration: Registration, tool_use_id: str) -> None:
    """Validate this call's source without constructing or limiting a write view."""
    _anchor(_scan(registration, tool_use_id, retain=False), registration, tool_use_id)


def _chain(nodes: dict[str, _Node], tail: str) -> list[_Node]:
    def parent(node: _Node) -> str | None:
        if _boundary(node.record):
            return None
        parent = node.record.get("parentUuid")
        if parent is not None and (not isinstance(parent, str) or not parent):
            raise _source_error()
        return parent

    try:
        return walk_parent_chain(tail, nodes.__getitem__, parent)
    except InvalidChain:
        raise _source_error() from None


def _utc_timestamp(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        timestamp = datetime.fromisoformat(value)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            return None
        return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")
    except ValueError, OverflowError:
        return None


_REPLY = re.compile(r"<!--\s*reply(?:\s+\d+)?\s*-->")
_TAG = re.compile(r"<\s*(/?)\s*([A-Za-z_][\w:.-]*)([^<>]*?)>")
_PASTE = re.compile(r"\s+id\s*=\s*([\"'])([^\"']+)\1\s*")
_DROP_BLOCKS = {"system-reminder", "artifact-view-context"}
EXCLUDE_BASH_MESSAGES = True
_BASH_TAGS = {"bash-input", "bash-stdout", "bash-stderr"}
_REJECT_TAGS = {
    "local-command-caveat",
    "local-command-stdout",
    "command-name",
    "command-message",
    "command-args",
    "task-notification",
}


_QUOTE_LINE = re.compile(r"^[ \t]*>[^\n]*(?:\n|$)", re.MULTILINE)


def _closing_host_block(text: str, opening: re.Match[str], name: str) -> re.Match[str] | None:
    depth = 1
    for candidate in _TAG.finditer(text, opening.end()):
        if candidate.group(2) != name:
            continue
        if candidate.group(1):
            depth -= 1
            if depth == 0:
                return candidate
        elif not candidate.group(3).rstrip().endswith("/"):
            depth += 1
    return None


def _bash_block(text: str, opening: re.Match[str]) -> tuple[re.Match[str], str | None] | None:
    """One policy switch owns actual Bash input; command output is never quotable."""
    if EXCLUDE_BASH_MESSAGES or opening.group(1) or opening.group(3).rstrip().endswith("/"):
        return None
    closing = _closing_host_block(text, opening, opening.group(2))
    if closing is None:
        return None
    body = text[opening.end() : closing.start()] if opening.group(2) == "bash-input" else None
    return closing, body


def _structured_fragments(text: str) -> list[str] | None:
    """Respect the outer source: paste bodies are opaque, host blocks are discarded."""
    fragments: list[str] = []
    outside: list[str] = []
    start = 0
    cursor = 0
    unknown_stack: list[str] = []

    def append_outside(end: int) -> None:
        fragment = text[start:end]
        if fragment:
            fragments.append(fragment)
            outside.append(fragment)

    while True:
        candidates = [match for pattern in (_TAG, _REPLY, _QUOTE_LINE) if (match := pattern.search(text, cursor))]
        if not candidates:
            break
        match = min(candidates, key=lambda found: found.start())
        if match.re is not _TAG:
            append_outside(match.start())
            cursor = start = match.end()
            if match.re is _REPLY and text[cursor : cursor + 1] == "\n":
                cursor = start = cursor + 1
            continue
        closing, name, attributes = match.groups()
        self_closing = attributes.rstrip().endswith("/")
        if name in _REJECT_TAGS:
            return None
        if name in _BASH_TAGS:
            bash = _bash_block(text, match)
            if bash is None:
                return None
            end, body = bash
            append_outside(match.start())
            if body:
                fragments.append(body)
            cursor = start = end.end()
            continue
        if name == "pasted_content":
            identifier = _PASTE.fullmatch(attributes)
            if closing or identifier is None:
                return None
            end = next(
                (
                    candidate
                    for candidate in _TAG.finditer(text, match.end())
                    if candidate.group(1)
                    and candidate.group(2) == "pasted_content"
                    and (end_id := _PASTE.fullmatch(candidate.group(3))) is not None
                    and end_id.group(2) == identifier.group(2)
                ),
                None,
            )
            if end is None:
                return None
            append_outside(match.start())
            body = text[match.end() : end.start()]
            if body:
                fragments.append(body)
            cursor = start = end.end()
            continue
        if name in _DROP_BLOCKS:
            if closing or self_closing:
                return None
            end = _closing_host_block(text, match, name)
            if end is None:
                return None
            append_outside(match.start())
            cursor = start = end.end()
            continue
        if name == "artifact-content-authored-by-others":
            if closing or not self_closing or unknown_stack:
                return None
            append_outside(match.start())
            cursor = start = match.end()
            continue
        if closing:
            if unknown_stack and unknown_stack[-1] == name:
                unknown_stack.pop()
        elif not self_closing:
            unknown_stack.append(name)
        cursor = match.end()
    append_outside(len(text))
    # Prefix validation uses this block's outside text; quotable fragments remain separate.
    return None if _TAG.match("".join(outside).lstrip()) is not None else fragments


def user_fragments(record: JsonObject) -> tuple[str, ...]:
    """Return only independently quotable fragments of a normal human message."""
    if record.get("type") != "user" or not _main(record):
        return ()
    if record.get("isMeta") or record.get("isCompactSummary"):
        return ()
    content = _message(record).get("content")
    if isinstance(content, str):
        blocks = [content]
    elif isinstance(content, list):
        blocks: list[str] = []
        for block in content:
            item = _object(block)
            if item is None or item.get("type") not in ("text", "image"):
                return ()
            if item.get("type") == "text":
                text = item.get("text")
                if not isinstance(text, str):
                    return ()
                blocks.append(text)
    else:
        return ()
    fragments: list[str] = []
    for block in blocks:
        parsed = _structured_fragments(block)
        if parsed is None:
            return ()
        fragments.extend(parsed)
    origin = _object(record.get("origin"))
    if (origin or {}).get("kind") != "human" and record.get("turnOrigin") != "human":
        return ()
    return tuple(fragment for fragment in fragments if fragment)


def _collect_sources(segment: _Segment, anchor: str, session: str) -> tuple[SourceMessage, ...]:
    from .sources import SourceRecord, collect_sources

    chain = _chain(segment.nodes, anchor)
    cutoff = segment.nodes[anchor].position
    records = [SourceRecord(node.position, node.record) for node in segment.occurrences]
    selected = {(0, cast(str, node.record["uuid"])) for node in chain}
    return collect_sources(records, selected, session, cutoff=cutoff).messages


def build_conversation(registration: Registration, tool_use_id: str) -> ConversationView:
    """Build original sources strictly before this call in the active segment."""
    segment = _scan(registration, tool_use_id, retain=True)
    anchor = _anchor(segment, registration, tool_use_id)
    try:
        messages = _collect_sources(segment, anchor, registration.session_id)
        return ConversationView(messages=messages, current_session=registration.session_id)
    except MemoryOperationError:
        raise _source_error() from None


def _assistant_text(record: JsonObject) -> str | None:
    content = _message(record).get("content")
    if not isinstance(content, list):
        return None
    text = [
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
    ]
    rendered = "\n".join(text).strip()
    return rendered or None


def catalog_visibility(
    transcript_path: str,
    prompt_id: str | None,
    last_assistant_message: str | None,
    target: str,
    *,
    clock: Clock | None = None,
) -> bool | None:
    """Inspect one bounded tail snapshot; unknown evidence never requests redelivery."""
    if not transcript_path or not prompt_id or not last_assistant_message:
        return None
    timer = clock or time
    started = timer.monotonic()
    try:
        with secure_open_regular_binary(Path(transcript_path)) as handle:
            size = os.fstat(handle.fileno()).st_size
            offset = max(0, size - STOP_WINDOW_BYTES)
            handle.seek(offset)
            payload = handle.read(min(size, STOP_WINDOW_BYTES))
    except OSError, ValueError:
        return None
    if offset:
        newline = payload.find(b"\n")
        payload = payload[newline + 1 :] if newline >= 0 else b""
    if not payload or not payload.endswith(b"\n"):
        return None
    segment = _Segment()
    tail: str | None = None
    for position, line in enumerate(payload.splitlines(), 1):
        if position > STOP_MAX_RECORDS or timer.monotonic() - started > STOP_BUDGET_SECONDS:
            return None
        record = _decode(line)
        if record is None:
            segment.bad = True
            continue
        if _boundary(record):
            segment = _Segment(boundary=record)
            tail = None
        _add_node(segment, record, position, retain=True)
        uuid = record.get("uuid")
        if isinstance(uuid, str) and uuid and _main(record) and record.get("type") != "progress":
            tail = uuid
    if segment.bad or tail is None or timer.monotonic() - started > STOP_BUDGET_SECONDS:
        return None
    try:
        chain = _chain(segment.nodes, tail)
    except MemoryOperationError:
        return None
    assistant_seen = False
    ready = False
    visible = False
    for node in chain:
        if timer.monotonic() - started > STOP_BUDGET_SECONDS:
            return None
        record = node.record
        if not assistant_seen and record.get("type") == "assistant":
            if _assistant_text(record) != last_assistant_message:
                return None
            assistant_seen = True
        if user_fragments(record):
            if not assistant_seen or record.get("promptId") != prompt_id:
                return None
            ready = True
            break
    if not ready:
        return None
    for node in chain:
        if timer.monotonic() - started > STOP_BUDGET_SECONDS:
            return None
        record = node.record
        if record.get("type") != "attachment":
            continue
        attachment = _object(record.get("attachment"))
        if attachment is None:
            return None
        if attachment.get("type") != "hook_additional_context":
            continue
        content = attachment.get("content")
        if not isinstance(content, list) or any(not isinstance(item, str) for item in content):
            return None
        if any(item == target or item == target + "\n" for item in content):
            visible = True
    return visible if timer.monotonic() - started <= STOP_BUDGET_SECONDS else None
