# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Codex root identity and per-block provenance contracts, using invented rollouts."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from chrys.app.memory_hosts.codex_history import CodexHistoryReader, decode_codex_history
from chrys.app.memory_hosts.history import read_snapshot
from chrys.service.memory.extraction import ScanDiagnostic, SessionKey
from chrys.service.memory.project import bind_project
from tests.app.memory_hosts.history_fixtures import PARENT, SESSION, FixedCodex, write_jsonl


def meta(cwd: Path, **fields: object) -> dict[str, object]:
    return {
        "type": "session_meta",
        "payload": {"id": SESSION, "cwd": str(cwd), "source": "vscode", "originator": "Codex Desktop", **fields},
    }


def message(
    *texts: str, item: str | None = "msg_fixture", provenance: str = "user", kind: str = "user.text"
) -> dict[str, object]:
    return {
        "type": "response_item",
        "timestamp": "2026-11-01T10:00:00+08:00",
        "payload": {
            "type": "message",
            "role": "user",
            "id": item,
            "content": [{"type": "input_text", "text": text} for text in texts],
            "internal_chat_message_metadata_passthrough": {
                "content_item_kinds": [kind for _ in texts],
                "content_item_metadata": [{"provenance": {"type": provenance}} for _ in texts],
            },
        },
    }


def decode(tmp_path: Path, records: list[dict[str, object]]):
    path = tmp_path / f"rollout-2026-11-01T10-00-00-{SESSION}.jsonl"
    write_jsonl(path, records)
    return decode_codex_history(read_snapshot(path), SESSION, bind_project(str(tmp_path)))


def kinds(notes):
    return {note.kind for note in notes}


@pytest.mark.parametrize(
    "field",
    [
        "parent_thread_id",
        "forked_from_id",
        "agent_path",
        "agent_nickname",
        "subagent_history_start_ordinal",
        "history_base",
        "agent_role",
    ],
)
@pytest.mark.parametrize("value", ["", {}, [], 0, "parent"])
def test_each_root_field_must_be_absent_or_null_even_for_empty_values(
    tmp_path: Path, field: str, value: object
) -> None:
    source, notes = decode(tmp_path, [meta(tmp_path, **{field: value}), message("Words")])
    assert source is None and "not_root" in kinds(notes)


def test_null_root_fields_and_thread_source_user_do_not_exclude_desktop_root(tmp_path: Path) -> None:
    fields = dict.fromkeys(
        (
            "parent_thread_id",
            "forked_from_id",
            "agent_path",
            "agent_nickname",
            "subagent_history_start_ordinal",
            "history_base",
            "agent_role",
        )
    )
    source, notes = decode(tmp_path, [meta(tmp_path, thread_source="user", **fields), message("Words")])
    assert source is not None and source.annotations == ()
    assert "signals_unavailable" in kinds(notes)


@pytest.mark.parametrize("source", [{"subagent": "root"}, "cli", "exec", "guardian", None])
def test_only_verified_desktop_entry_is_supported(tmp_path: Path, source: object) -> None:
    result, notes = decode(tmp_path, [meta(tmp_path, source=source), message("Words")])
    assert result is None and "unsupported_entrypoint" in kinds(notes)


def test_first_meta_subagent_cannot_be_rescued_by_last_desktop_meta(tmp_path: Path) -> None:
    source, notes = decode(
        tmp_path,
        [
            meta(tmp_path, source={"subagent": "parent"}, parent_thread_id=PARENT),
            message("Words"),
            meta(tmp_path, id=PARENT),
        ],
    )
    assert source is None and {"unsupported_entrypoint", "not_root"} <= kinds(notes)


def test_seventeen_meta_root_is_still_decided_only_by_first_record(tmp_path: Path) -> None:
    source, _ = decode(
        tmp_path, [meta(tmp_path), message("Words"), *[meta(tmp_path, source="exec", id=PARENT) for _ in range(16)]]
    )
    assert source is not None and source.key.session == SESSION


def test_id_and_nonroot_reasons_are_independently_reported(tmp_path: Path) -> None:
    source, notes = decode(tmp_path, [meta(tmp_path, id=PARENT, history_base={}), message("Words")])
    assert source is None and {"session_id_mismatch", "not_root"} <= kinds(notes)


def test_each_user_text_block_is_independent_and_claude_markup_is_literal(tmp_path: Path) -> None:
    source, _ = decode(
        tmp_path,
        [
            meta(tmp_path),
            message(
                "First",
                '<pasted_content id="p">Literal</pasted_content id="p">',
                "<!-- reply -->\n> Literal model quote",
                item=None,
            ),
            {"type": "event_msg", "payload": {"type": "user_message", "message": "Do not duplicate"}},
        ],
    )
    assert source is not None
    assert len(source.messages) == 1
    assert source.messages[0].fragments == (
        "First",
        '<pasted_content id="p">Literal</pasted_content id="p">',
        "<!-- reply -->\n> Literal model quote",
    )
    assert source.messages[0].item == "line:2" and source.messages[0].position == 2
    assert source.messages[0].at == "2026-11-01T02:00:00Z"


@pytest.mark.parametrize("shape", ["missing", "kinds_short", "metadata_short", "harness", "wrong_kind", "image"])
def test_user_role_alone_cannot_establish_quotable_provenance(tmp_path: Path, shape: str) -> None:
    record = message("User words")
    payload = record["payload"]
    assert isinstance(payload, dict)
    metadata = payload["internal_chat_message_metadata_passthrough"]
    assert isinstance(metadata, dict)
    if shape == "missing":
        del payload["internal_chat_message_metadata_passthrough"]
    elif shape == "kinds_short":
        metadata["content_item_kinds"] = []
    elif shape == "metadata_short":
        metadata["content_item_metadata"] = []
    elif shape == "harness":
        metadata["content_item_metadata"] = [{"provenance": {"type": "harness"}}]
    elif shape == "wrong_kind":
        metadata["content_item_kinds"] = ["environment"]
    else:
        payload["content"] = [{"type": "input_image", "image_url": "synthetic"}]
    source, _ = decode(tmp_path, [meta(tmp_path), record])
    assert source is None


@pytest.mark.parametrize(
    "block_metadata",
    [
        None,
        False,
        "not-an-object",
        [],
        {},
        {"provenance": None},
        {"provenance": "user"},
        {"provenance": []},
        {"provenance": {}},
        {"provenance": {"type": None}},
        {"provenance": {"type": ""}},
        {"provenance": {"type": " "}},
        {"provenance": {"type": 42}},
        {"provenance": {"type": []}},
    ],
)
def test_aligned_block_with_missing_or_malformed_provenance_is_reported_and_not_quoted(
    tmp_path: Path, block_metadata: object
) -> None:
    record = message("Unverifiable block", "Verified neighbor")
    payload = record["payload"]
    assert isinstance(payload, dict)
    metadata = payload["internal_chat_message_metadata_passthrough"]
    assert isinstance(metadata, dict)
    metadata["content_item_metadata"] = [block_metadata, {"provenance": {"type": "user"}}]

    source, notes = decode(tmp_path, [meta(tmp_path), record])

    assert source is not None
    assert source.messages[0].fragments == ("Verified neighbor",)
    assert [note for note in notes if note.kind == "missing_provenance"] == [
        ScanDiagnostic("missing_provenance", SessionKey("codex", SESSION), details=(("position", "2"),))
    ]


@pytest.mark.parametrize("provenance", ["environment", "AGENTS", "harness"])
def test_valid_nonuser_provenance_is_a_silent_exclusion(tmp_path: Path, provenance: str) -> None:
    source, notes = decode(tmp_path, [meta(tmp_path), message("Host material", provenance=provenance)])
    assert source is None
    assert "missing_provenance" not in kinds(notes)


def test_missing_provenance_is_reported_even_when_block_kind_is_excluded(tmp_path: Path) -> None:
    record = message("Host material", kind="environment")
    payload = record["payload"]
    assert isinstance(payload, dict)
    metadata = payload["internal_chat_message_metadata_passthrough"]
    assert isinstance(metadata, dict)
    metadata["content_item_metadata"] = [{}]

    source, notes = decode(tmp_path, [meta(tmp_path), record])

    assert source is None
    assert [note for note in notes if note.kind == "missing_provenance"] == [
        ScanDiagnostic("missing_provenance", SessionKey("codex", SESSION), details=(("position", "2"),))
    ]


@pytest.mark.parametrize("kind", ["compacted", "thread_rolled_back"])
@pytest.mark.parametrize("container", ["top", "event"])
def test_history_replacement_skips_whole_session_instead_of_reading_summaries(
    tmp_path: Path, kind: str, container: str
) -> None:
    replacement = (
        {"type": kind, "replacement_history": [message("Synthetic summary")]}
        if container == "top"
        else {"type": "event_msg", "payload": {"type": kind, "num_turns": 1}}
    )
    source, notes = decode(tmp_path, [meta(tmp_path), message("Original"), replacement])
    assert source is None and "unsupported_history_replacement" in kinds(notes)


async def test_reader_extracts_tail_uuid_recursively_and_revalidates_source(tmp_path: Path) -> None:
    folder = tmp_path / "2026" / "11"
    await asyncio.to_thread(folder.mkdir, parents=True)
    path = folder / f"rollout-2026-11-01T10-00-00-{SESSION}.jsonl"
    write_jsonl(path, [meta(tmp_path), message("Words")])
    project = await asyncio.to_thread(bind_project, str(tmp_path))
    processes = FixedCodex()
    reader = CodexHistoryReader(tmp_path, processes=processes)
    result = await reader.scan(project)
    assert result.fatal is False and len(result.sessions) == 1
    assert result.sessions[0].key.session == SESSION
    assert await reader.revalidate(result.sessions[0]) is None
    processes.activity = "active"
    assert await reader.revalidate(result.sessions[0]) == "codex_hosts_reopened"


@pytest.mark.parametrize("activity", ["active", "unknown"])
async def test_any_live_or_unknown_codex_host_rejects_entire_scan(tmp_path: Path, activity) -> None:
    result = await CodexHistoryReader(tmp_path, processes=FixedCodex(activity)).scan(
        await asyncio.to_thread(bind_project, str(tmp_path))
    )
    assert result.fatal is True and result.sessions == () and "codex_hosts_" + activity in kinds(result.diagnostics)


async def test_replay_uses_only_explicit_snapshot_and_never_checks_current_hosts(tmp_path: Path) -> None:
    path = tmp_path / f"rollout-frozen-{SESSION}.jsonl"
    write_jsonl(path, [meta(tmp_path), message("Frozen words")])
    processes = FixedCodex("active")
    reader = CodexHistoryReader(tmp_path / "missing-live-directory", processes=processes, replay=True)
    result = await reader.read_files(await asyncio.to_thread(bind_project, str(tmp_path)), (path,))
    assert len(result.sessions) == 1 and processes.calls == 0
    assert await reader.revalidate(result.sessions[0]) is None and processes.calls == 0
    write_jsonl(path, [meta(tmp_path), message("Changed snapshot")])
    assert await reader.revalidate(result.sessions[0]) == "source_changed"
