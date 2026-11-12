# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Small invented host records; no exported transcript or real session identity."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from chrys.app.memory_hosts.claude_code.holder import Activity
from chrys.foundation.platform.files import atomic_write_owner_only_bytes

SESSION = "11111111-1111-4111-8111-111111111111"
PARENT = "22222222-2222-4222-8222-222222222222"
JsonObject = dict[str, object]


def claude_user(item: str, text: object, parent: str | None = None, **fields: object) -> JsonObject:
    return {
        "type": "user",
        "uuid": item,
        "parentUuid": parent,
        "sessionId": SESSION,
        "entrypoint": "claude-desktop",
        "origin": {"kind": "human"},
        "timestamp": "2026-11-01T10:00:00+08:00",
        "message": {"role": "user", "content": text},
        **fields,
    }


def claude_assistant(item: str, parent: str | None, **fields: object) -> JsonObject:
    return {
        "type": "assistant",
        "uuid": item,
        "parentUuid": parent,
        "sessionId": SESSION,
        "message": {"content": [{"type": "text", "text": "Synthetic answer"}]},
        **fields,
    }


def compact(item: str, logical: str) -> JsonObject:
    return {
        "type": "system",
        "subtype": "compact_boundary",
        "uuid": item,
        "parentUuid": None,
        "logicalParentUuid": logical,
        "compactMetadata": {"preservedMessages": {"allUuids": [logical]}},
    }


def write_jsonl(path: Path, records: Sequence[JsonObject], suffix: bytes = b"") -> None:
    atomic_write_owner_only_bytes(
        path, b"".join((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8") for record in records) + suffix
    )


@dataclass
class FixedHolder:
    activity: Activity = "inactive"
    calls: int = 0

    def check(self, session: str) -> Activity:
        self.calls += 1
        return self.activity


@dataclass
class FixedCodex:
    activity: Activity = "inactive"
    calls: int = 0

    def check(self) -> Activity:
        self.calls += 1
        return self.activity
