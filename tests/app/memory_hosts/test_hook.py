# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Catalog delivery state transitions from the host hook specification."""

from __future__ import annotations

import asyncio
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from chrys.app.memory_hosts.claude_code import hook
from chrys.app.memory_hosts.claude_code.registry import DeliveryState, HostKey, ProcessInfo, Registry
from chrys.service.memory.operations import MemoryReader
from chrys.service.memory.render import WITHDRAWN_CATALOG, render_catalog
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import GLOBAL_SCOPE, MemoryEntryDraft, MemoryOperationError, Quote
from tests.support.secure_files import plant_owner_only_bytes

KEY = HostKey(20, 17)


@dataclass
class Tree:
    argv: tuple[str, ...] | None = ("claude",)

    def inspect(self, pid: int) -> ProcessInfo | None:
        if pid == 10:
            return ProcessInfo(10, 20, "python", 18, ())
        if pid == 20:
            return ProcessInfo(20, 0, "claude", 17, self.argv)
        return None


@dataclass
class Harness:
    tree: Tree
    registry: Registry
    store: FileMemoryStore
    environ: dict[str, str]
    output: io.StringIO
    diagnostic: io.StringIO
    transcript: Path

    def event(self, event: str, source: str = "startup") -> dict[str, object]:
        return {
            "hook_event_name": event,
            "source": source,
            "session_id": "full-session-id",
            "transcript_path": str(self.transcript),
            "cwd": "/shell/has/moved",
            "prompt": "Never retained by the hook.",
            "prompt_id": "current-prompt",
            "last_assistant_message": "Done",
        }

    def invoke(self, event: str, source: str = "startup") -> str:
        self.output.seek(0)
        self.output.truncate()
        asyncio.run(
            hook.handle_hook(
                self.event(event, source),
                tree=self.tree,
                registry=self.registry,
                store=self.store,
                environ=self.environ,
                output=self.output,
                diagnostic=self.diagnostic,
                pid=10,
            )
        )
        return self.output.getvalue()

    def seed(self) -> str:
        asyncio.run(
            self.store.create(
                GLOBAL_SCOPE,
                MemoryEntryDraft(
                    "intent",
                    "Use SQLite",
                    "Use SQLite for durable local state.",
                    "model",
                    quotes=(Quote("Use SQLite", "message", "s"),),
                ),
            )
        )
        result = render_catalog(asyncio.run(MemoryReader(self.store, None).catalog()))
        assert result is not None
        return result

    def transcript_rows(self, *, catalog: str | None = None) -> None:
        rows: list[dict[str, object]] = []
        parent: str | None = None
        if catalog is not None:
            rows.append(
                {
                    "uuid": "catalog",
                    "parentUuid": None,
                    "type": "attachment",
                    "attachment": {
                        "type": "hook_additional_context",
                        "content": [catalog],
                    },
                }
            )
            parent = "catalog"
        rows.extend(
            [
                {
                    "uuid": "user",
                    "parentUuid": parent,
                    "type": "user",
                    "promptId": "current-prompt",
                    "origin": {"kind": "human"},
                    "message": {"role": "user", "content": "Use SQLite"},
                },
                {
                    "uuid": "assistant",
                    "parentUuid": "user",
                    "type": "assistant",
                    "message": {"role": "assistant", "content": [{"type": "text", "text": "Done"}]},
                },
                {"uuid": "summary", "parentUuid": "assistant", "type": "system", "subtype": "stop_hook_summary"},
                {"type": "last-prompt"},
            ]
        )
        plant_owner_only_bytes(self.transcript, ("\n".join(json.dumps(row) for row in rows) + "\n").encode("utf-8"))


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(
        Tree(),
        Registry(tmp_path / "memory"),
        FileMemoryStore(tmp_path / "memory"),
        {"CLAUDE_PROJECT_DIR": str(tmp_path)},
        io.StringIO(),
        io.StringIO(),
        tmp_path / "transcript.jsonl",
    )


@pytest.mark.parametrize("source", ["startup", "clear", "resume", "fork", "compact"])
def test_session_start_forces_nonempty_catalog_without_writing_registration(harness: Harness, source: str) -> None:
    text = harness.seed()
    assert harness.invoke("SessionStart", source) == text + "\n"
    assert harness.invoke("SessionStart", source) == text + "\n"
    assert harness.registry.read_registration(KEY) is None
    assert harness.registry.read_delivery(KEY, "full-session-id") == DeliveryState("full-session-id", text, True)


@pytest.mark.parametrize(
    "source,expected",
    [
        ("startup", ""),
        ("clear", ""),
        ("resume", WITHDRAWN_CATALOG + "\n"),
        ("fork", WITHDRAWN_CATALOG + "\n"),
        ("compact", WITHDRAWN_CATALOG + "\n"),
    ],
)
def test_empty_session_start_uses_source_rule(harness: Harness, source: str, expected: str) -> None:
    assert harness.invoke("SessionStart", source) == expected
    assert harness.registry.read_delivery(KEY, "full-session-id") == DeliveryState("full-session-id", None, False)


def test_unknown_session_start_does_not_touch_binding_or_delivery(harness: Harness) -> None:
    harness.invoke("UserPromptSubmit")
    before = harness.registry.registration_path(KEY).read_bytes()
    delivery = harness.registry.delivery_path(KEY, "full-session-id").read_bytes()
    assert harness.invoke("SessionStart", "unknown") == ""
    assert harness.registry.registration_path(KEY).read_bytes() == before
    assert harness.registry.delivery_path(KEY, "full-session-id").read_bytes() == delivery


def test_ups_changed_text_or_redelivery_only_and_stable_twenty_rounds(harness: Harness) -> None:
    text = harness.seed()
    assert harness.invoke("UserPromptSubmit") == text + "\n"
    for _ in range(20):
        assert harness.invoke("UserPromptSubmit") == ""
    harness.registry.mark_redeliver(KEY, "full-session-id")
    assert harness.invoke("UserPromptSubmit") == text + "\n"
    assert harness.registry.needs_redeliver(KEY, "full-session-id") is False
    registration = harness.registry.read_registration(KEY)
    assert registration is not None
    assert registration.project_dir == harness.environ["CLAUDE_PROJECT_DIR"]
    assert "prompt" not in json.loads(harness.registry.registration_path(KEY).read_text(encoding="utf-8"))


def test_ups_empty_never_nonempty_does_not_withdraw_even_with_flag(harness: Harness) -> None:
    harness.registry.mark_redeliver(KEY, "full-session-id")
    assert harness.invoke("UserPromptSubmit") == ""
    assert harness.registry.needs_redeliver(KEY, "full-session-id") is False


def test_ups_empty_after_previous_catalog_withdraws_once(harness: Harness) -> None:
    harness.registry.write_delivery(KEY, DeliveryState("full-session-id", "old catalog", True))
    assert harness.invoke("UserPromptSubmit") == WITHDRAWN_CATALOG + "\n"
    assert harness.invoke("UserPromptSubmit") == ""
    assert harness.registry.read_delivery(KEY, "full-session-id") == DeliveryState("full-session-id", None, True)


@pytest.mark.parametrize("bad_project", ["missing", "relative", "deleted", "override", "argv-unreadable"])
def test_invalid_project_registers_none_and_skips_catalog(harness: Harness, bad_project: str) -> None:
    harness.seed()
    harness.invoke("UserPromptSubmit")
    if bad_project == "missing":
        harness.environ.clear()
    elif bad_project == "relative":
        harness.environ["CLAUDE_PROJECT_DIR"] = "relative"
    elif bad_project == "deleted":
        harness.environ["CLAUDE_PROJECT_DIR"] += "/does-not-exist"
    elif bad_project == "override":
        harness.tree.argv = ("claude", "--project-config-root=other")
    else:
        harness.tree.argv = None
    assert harness.invoke("UserPromptSubmit") == ""
    registration = harness.registry.read_registration(KEY)
    assert registration is not None and registration.project_dir is None
    assert "shell cwd is not used as a fallback" in harness.diagnostic.getvalue()


def test_missing_git_falls_back_to_global_for_catalog(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    text = harness.seed()

    def fail_bind(cwd: str) -> None:
        raise FileNotFoundError("must not appear")

    monkeypatch.setattr(hook, "bind_project", fail_bind)
    assert harness.invoke("UserPromptSubmit") == text + "\n"
    assert "only global" in harness.diagnostic.getvalue()
    assert "must not appear" not in harness.diagnostic.getvalue()


def test_store_failure_cannot_withdraw_or_clear_flag(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    harness.registry.write_delivery(KEY, DeliveryState("full-session-id", "old", True))
    harness.registry.mark_redeliver(KEY, "full-session-id")

    async def fail_catalog(self: MemoryReader) -> None:
        raise MemoryOperationError("memory_store_unavailable", "store unavailable")

    monkeypatch.setattr(MemoryReader, "catalog", fail_catalog)
    with pytest.raises(MemoryOperationError):
        harness.invoke("UserPromptSubmit")
    assert harness.output.getvalue() == ""
    assert harness.registry.needs_redeliver(KEY, "full-session-id") is True
    assert harness.registry.read_delivery(KEY, "full-session-id") == DeliveryState("full-session-id", "old", True)


def test_output_is_flushed_before_delivery_baseline_is_saved(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    text = harness.seed()
    original = harness.registry.write_delivery

    def save(key: HostKey, state: DeliveryState) -> None:
        assert harness.output.getvalue() == text + "\n"
        original(key, state)

    monkeypatch.setattr(harness.registry, "write_delivery", save)
    assert harness.invoke("UserPromptSubmit") == text + "\n"


def test_stop_sets_only_redelivery_flag_and_ups_clears_it(harness: Harness) -> None:
    text = harness.seed()
    harness.invoke("UserPromptSubmit")
    registration = harness.registry.registration_path(KEY).read_bytes()
    delivery = harness.registry.delivery_path(KEY, "full-session-id").read_bytes()
    harness.transcript_rows()
    assert harness.invoke("Stop") == ""
    assert harness.registry.needs_redeliver(KEY, "full-session-id") is True
    assert harness.registry.registration_path(KEY).read_bytes() == registration
    assert harness.registry.delivery_path(KEY, "full-session-id").read_bytes() == delivery
    assert harness.invoke("UserPromptSubmit") == text + "\n"
    assert harness.registry.needs_redeliver(KEY, "full-session-id") is False


@pytest.mark.parametrize("target", ["catalog", "withdrawn"])
def test_stop_visible_catalog_or_withdrawal_does_not_repeat(harness: Harness, target: str) -> None:
    text = harness.seed() if target == "catalog" else None
    harness.registry.write_delivery(KEY, DeliveryState("full-session-id", text, True))
    harness.transcript_rows(catalog=text if text is not None else WITHDRAWN_CATALOG)
    for _ in range(20):
        assert harness.invoke("Stop") == ""
    assert harness.registry.needs_redeliver(KEY, "full-session-id") is False


def test_stop_unknown_or_missing_state_keeps_flags_and_catalog(harness: Harness) -> None:
    assert harness.invoke("Stop") == ""
    harness.registry.write_delivery(KEY, DeliveryState("full-session-id", "catalog", True))
    before = harness.registry.delivery_path(KEY, "full-session-id").read_bytes()
    for _ in range(20):
        assert harness.invoke("Stop") == ""
    assert harness.registry.needs_redeliver(KEY, "full-session-id") is False
    assert harness.registry.delivery_path(KEY, "full-session-id").read_bytes() == before


def test_oversized_stdin_is_fully_drained_without_json_parse() -> None:
    class Input(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(b"{" + b" " * (hook.STDIN_LIMIT * 2))
            self.read_sizes: list[int] = []

        def read(self, size: int = -1) -> bytes:
            self.read_sizes.append(size)
            return super().read(size)

    stream = Input()
    assert hook.read_input(stream) is None
    assert stream.tell() == len(stream.getvalue())
    assert set(stream.read_sizes) == {64 * 1024}
