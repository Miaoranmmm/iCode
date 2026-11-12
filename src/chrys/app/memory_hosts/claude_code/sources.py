# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared physical Claude source collection after each caller proves its selected chain."""

from __future__ import annotations

import json
from bisect import bisect_left
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from chrys.service.memory.extraction import SourceAnnotation
from chrys.service.memory.names import is_memory_tool
from chrys.service.memory.related_refs import call_refs, result_refs
from chrys.service.memory.storage.store import MAX_ARGS
from chrys.service.memory.verification import SourceKind, SourceMessage

from .chain import InvalidChain
from .conversation import JsonObject, _boundary, _main, _message, _object, _utc_timestamp, user_fragments

SourceKey = tuple[int, str]


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """A physical occurrence; segment scopes UUID parents and response identities."""

    position: int
    record: JsonObject
    segment: int = 0

    @property
    def key(self) -> SourceKey:
        return self.segment, cast(str, self.record["uuid"])


@dataclass(frozen=True, slots=True)
class ToolObservation:
    """A paired original result, even when it has no independently quotable text."""

    position: int
    part: int
    item: str
    tool: str
    is_error: bool | None


@dataclass(frozen=True, slots=True)
class SourceCollection:
    messages: tuple[SourceMessage, ...]
    results: tuple[ToolObservation, ...]
    related_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Call:
    position: int
    part: int
    call_id: str
    tool: str
    args: str
    eligible: bool
    related_refs: tuple[str, ...]

    @property
    def identity(self) -> tuple[int, int]:
        return self.position, self.part


class OccurrenceGraph:
    """Resolve a parent to its latest preceding occurrence in the same compact segment."""

    def __init__(self, nodes: Sequence[SourceRecord]) -> None:
        self.by_uuid: dict[SourceKey, list[SourceRecord]] = {}
        self.by_position = {node.position: node for node in nodes}
        for node in nodes:
            self.by_uuid.setdefault(node.key, []).append(node)

    def parent(self, node: SourceRecord) -> SourceRecord | None:
        if _boundary(node.record):
            return None
        parent = node.record.get("parentUuid")
        if parent is None:
            return None
        if not isinstance(parent, str) or not parent:
            raise InvalidChain("Invalid transcript parent.")
        occurrences = self.by_uuid.get((node.segment, parent), [])
        index = bisect_left(occurrences, node.position, key=lambda found: found.position) - 1
        if index < 0:
            raise InvalidChain("Missing preceding transcript parent.")
        return occurrences[index]


def blocks(record: JsonObject) -> list[tuple[int, JsonObject]]:
    content = _message(record).get("content")
    if not isinstance(content, list):
        return []
    return [(part, item) for part, block in enumerate(content) if (item := _object(block)) is not None]


def arguments(value: object) -> str:
    """One-line JSON with its ellipsis included in the character budget."""
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return encoded if len(encoded) <= MAX_ARGS else encoded[: MAX_ARGS - 1] + "…"


def result_fragments(content: object) -> tuple[str, ...]:
    if isinstance(content, str):
        return (content,) if content else ()
    if not isinstance(content, list):
        return ()
    return tuple(
        value
        for block in content
        if (item := _object(block)) is not None
        and item.get("type") == "text"
        and isinstance(value := item.get("text"), str)
        and value
    )


def _calls(
    nodes: Sequence[SourceRecord], selected: set[SourceKey], response_ids: set[tuple[int, str]]
) -> dict[int, tuple[_Call, ...]]:
    result: dict[int, tuple[_Call, ...]] = {}
    for node in nodes:
        record = node.record
        if record.get("type") != "assistant":
            continue
        message_id = _message(record).get("id")
        eligible = _main(record) and (
            node.key in selected or (isinstance(message_id, str) and (node.segment, message_id) in response_ids)
        )
        declarations: list[_Call] = []
        for part, block in blocks(record):
            call_id, name = block.get("id"), block.get("name")
            if block.get("type") != "tool_use" or not isinstance(call_id, str) or not call_id:
                continue
            if not isinstance(name, str) or not name:
                name = ""
            declarations.append(
                _Call(
                    node.position,
                    part,
                    call_id,
                    name,
                    arguments(block.get("input")),
                    eligible and bool(name),
                    call_refs(name, block.get("input")),
                )
            )
        if declarations:
            result[node.position] = tuple(declarations)
    return result


def _pair_result(
    node: SourceRecord,
    call_id: str,
    parent_of: Callable[[SourceRecord], SourceRecord | None],
    calls: dict[int, tuple[_Call, ...]],
    answered: set[tuple[int, int]],
) -> _Call | None:
    try:
        parent = parent_of(node)
        seen: set[int] = set()
        while parent is not None:
            if parent.position in seen:
                return None
            seen.add(parent.position)
            matching = [call for call in calls.get(parent.position, ()) if call.call_id == call_id]
            if matching:
                # Excluded or answered nearer declarations block reuse of an older id.
                if len(matching) != 1:
                    return None
                call = matching[0]
                if not call.eligible or call.identity in answered:
                    return None
                answered.add(call.identity)
                return call
            parent = parent_of(parent)
    except InvalidChain:
        return None
    return None


def collect_sources(
    records: Sequence[SourceRecord],
    selected: set[SourceKey],
    session: str,
    *,
    cutoff: int | None = None,
    parent_of: Callable[[SourceRecord], SourceRecord | None] | None = None,
    error_attributes: bool = False,
) -> SourceCollection:
    """Collect all kinds once; callers retain their own strict selection/compact policy."""
    response_ids = {
        (node.segment, message_id)
        for node in records
        if node.key in selected
        and (cutoff is None or node.position <= cutoff)
        and _main(node.record)
        and node.record.get("type") == "assistant"
        and isinstance(message_id := _message(node.record).get("id"), str)
        and message_id
    }
    nodes = [node for node in records if cutoff is None or node.position < cutoff]
    graph = OccurrenceGraph(nodes)
    parent = parent_of or graph.parent
    calls = _calls(nodes, selected, response_ids)
    answered: set[tuple[int, int]] = set()
    sources: dict[tuple[str, SourceKind, int], SourceMessage] = {}
    observations: dict[tuple[str, int], ToolObservation] = {}
    related: dict[str, None] = {}

    def add(
        node: SourceRecord,
        kind: SourceKind,
        part: int,
        fragments: tuple[str, ...],
        attributes: tuple[tuple[str, str], ...] = (),
    ) -> None:
        uuid = node.key[1]
        key = uuid, kind, part
        if fragments and key not in sources:
            sources[key] = SourceMessage(
                session,
                node.position,
                fragments,
                kind,
                part,
                uuid,
                _utc_timestamp(node.record.get("timestamp")),
                attributes,
            )

    for node in nodes:
        record = node.record
        if not _main(record):
            continue
        for declaration in calls.get(node.position, ()):
            if declaration.eligible:
                related.update(dict.fromkeys(declaration.related_refs))
        if node.key in selected:
            if fragments := user_fragments(record):
                add(node, "user", 0, fragments)
            if record.get("type") == "assistant":
                for part, block in blocks(record):
                    value = block.get("text")
                    if block.get("type") == "text" and isinstance(value, str) and value:
                        add(node, "assistant", part, (value,))
        if record.get("type") != "user":
            continue
        for part, block in blocks(record):
            call_id = block.get("tool_use_id")
            if block.get("type") != "tool_result" or not isinstance(call_id, str) or not call_id:
                continue
            call = _pair_result(node, call_id, parent, calls, answered)
            if call is None:
                continue
            if is_memory_tool(call.tool):
                related.update(dict.fromkeys(result_refs(result_fragments(block.get("content")))))
                continue
            flag = block.get("is_error", False)
            error = flag if isinstance(flag, bool) else None
            observations.setdefault(
                (node.key[1], part), ToolObservation(node.position, part, node.key[1], call.tool, error)
            )
            attributes = (("tool", call.tool), ("args", call.args))
            if error_attributes:
                attributes += (("is_error", "true" if error is True else "false" if error is False else "unknown"),)
            add(node, "tool_result", part, result_fragments(block.get("content")), attributes)
    return SourceCollection(tuple(sources.values()), tuple(observations.values()), tuple(related))


def human_resets(records: Sequence[SourceRecord], selected: set[SourceKey]) -> tuple[int, ...]:
    """A real new user input resets streaks even if it contains only an image."""
    positions: dict[str, int] = {}
    for node in records:
        record = node.record
        origin = _object(record.get("origin")) or {}
        if node.key not in selected or not _main(record) or record.get("type") != "user":
            continue
        if record.get("isMeta") or record.get("isCompactSummary"):
            continue
        if origin.get("kind") != "human" and record.get("turnOrigin") != "human":
            continue
        if any(block.get("type") == "tool_result" for _, block in blocks(record)):
            continue
        positions.setdefault(node.key[1], node.position)
    return tuple(sorted(positions.values()))


def tool_failure_signals(results: Sequence[ToolObservation], resets: Sequence[int]) -> tuple[SourceAnnotation, ...]:
    """Use only paired result flags; interleaved tools keep independent streaks."""
    failures: dict[str, list[ToolObservation]] = {}
    reset_index = 0
    annotations: list[SourceAnnotation] = []
    for result in results:
        while reset_index < len(resets) and resets[reset_index] <= result.position:
            failures.clear()
            reset_index += 1
        streak = failures.setdefault(result.tool, [])
        if result.is_error is True:
            streak.append(result)
            continue
        if result.is_error is False and len(streak) >= 2:
            first = streak[0]
            annotations.append(
                SourceAnnotation(
                    "tool_failures_then_success",
                    result.position,
                    result.item,
                    start_position=first.position,
                    start_part=first.part,
                    part=result.part,
                )
            )
        streak.clear()
    return tuple(annotations)
