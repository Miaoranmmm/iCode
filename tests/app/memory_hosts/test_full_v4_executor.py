# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Full-v4 refusals, durable batch outcomes, and stopping after real tool failures."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from chrys.app.memory_hosts import executor_icode as module
from chrys.app.memory_hosts.executor_icode import ICodeExecutor
from chrys.app.memory_hosts.extract import ExtractionOptions, run_extraction
from chrys.foundation.tool_result_metadata import TOOL_FAILED_METADATA_KEY, TOOL_RESULT_METADATA_KEY
from chrys.service.memory.extraction import (
    FULL_V3,
    FULL_V4,
    USER_V2,
    MaterialProfile,
    ScanResult,
    SessionKey,
    SourceRevision,
    SourceSession,
    WatermarkStore,
)
from chrys.service.memory.operations import MemoryWriter, SaveOutcome, SaveRequest
from chrys.service.memory.project import bind_project
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import MemoryOperationError, MemoryScope
from chrys.service.memory.tuning import V0
from chrys.service.memory.verification import ConversationView, QuoteClaim, SourceMessage, ToolResultClaim
from tests.app.memory_hosts.executor_fakes import (
    RUN_ID,
    SESSION,
    WORDS,
    ClientFactory,
    RecordingLog,
    ScriptedWire,
    call,
    prepared,
    profile,
    response,
    save_arguments,
)


@dataclass
class FixedReader:
    source: SourceSession

    async def scan(self, project: MemoryScope) -> ScanResult:
        assert project == self.source.project
        return ScanResult((self.source,))

    async def revalidate(self, source: SourceSession) -> str | None:
        assert source.key == self.source.key and source.revision == self.source.revision
        return None


@dataclass
class Trial:
    options: ExtractionOptions
    source: SourceSession
    store: FileMemoryStore
    view: ConversationView
    engineering: SaveOutcome
    intent: SaveOutcome


async def trial(tmp_path: Path) -> Trial:
    project_dir = tmp_path / "project"
    await asyncio.to_thread(project_dir.mkdir)
    project = await asyncio.to_thread(bind_project, str(project_dir))
    messages = (
        SourceMessage(SESSION, 1, (WORDS,), item="user", at="2026-01-01T00:00:00Z"),
        SourceMessage(
            SESSION,
            2,
            ("alpha beta gamma delta epsilon zeta eta theta",),
            kind="tool_result",
            item="tool",
            attributes=(("tool", "shell"), ("args", "{}")),
        ),
    )
    source = SourceSession(SessionKey("claude-code", SESSION), project, SourceRevision(1, "a" * 64), messages)
    options = ExtractionOptions(project_dir, tmp_path / "memory")
    store = FileMemoryStore(options.memory_dir)
    view = ConversationView(messages, None)
    writer = MemoryWriter(store, project, "extraction")
    engineering = await writer.save(
        SaveRequest("engineering", "Finding", "Observed alpha.", evidence=(ToolResultClaim("alpha", session=SESSION),)),
        view,
    )
    intent = await writer.save(
        SaveRequest(
            type="intent",
            title="Preference",
            body=WORDS,
            user_quotes=(QuoteClaim("user_quotes[0]", "user_quotes[0].candidate", WORDS, session=SESSION),),
        ),
        view,
    )
    return Trial(options, source, store, view, engineering, intent)


async def run_trial(
    configured: Trial,
    monkeypatch: pytest.MonkeyPatch,
    script: list[Any],
    *,
    material: MaterialProfile = FULL_V4,
    max_turns: int = 24,
    decoded_hook=None,
):
    wire = ScriptedWire(script, decoded_hook=decoded_hook)
    monkeypatch.setattr(module, "create_client", ClientFactory(wire))
    runner = ICodeExecutor(
        prepared(), configured.store, RecordingLog(), RUN_ID, material_profile=material, max_turns=max_turns
    )
    report = await run_extraction(
        configured.options,
        profile(),
        readers={"claude-code": FixedReader(configured.source)},
        executor=runner,
        material_profile=material,
        tuning=V0,
    )
    record = (await WatermarkStore(configured.options.memory_dir).load()).record(
        configured.source.project.project_id or "", configured.source.key
    )
    return report, record, wire


async def rejected_call(configured: Trial, case: str) -> tuple[str, dict[str, object], str]:
    saved = configured.engineering
    update: dict[str, object] = {
        "type": "engineering",
        "title": "Finding",
        "body": "Reworded alpha.",
        "evidence": [{"kind": "tool_result", "quote": "alpha", "session": SESSION}],
        "ref": saved.ref,
        "version": saved.version,
    }
    writer = MemoryWriter(configured.store, configured.source.project, "extraction")
    if case == "decode":
        return "memory_read", {"ref": 7}, "memory_invalid_input"
    if case == "ref":
        return "memory_read", {"ref": "0123456789ab"}, "memory_invalid_input"
    if case == "missing":
        return "memory_read", {"ref": "project:ffffffffffff"}, "memory_not_found"
    if case == "quote":
        return (
            "memory_save",
            save_arguments(user_quotes=[{"text": "absent", "session": SESSION}]),
            "memory_quote_not_found",
        )
    if case == "evidence":
        update.pop("ref")
        update.pop("version")
        update["evidence"] = [{"kind": "tool_result", "quote": "absent", "session": SESSION}]
        return "memory_save", update, "memory_evidence_not_found"
    if case == "expired":
        return (
            "memory_save",
            save_arguments(user_quotes=[{"text": WORDS, "session": SESSION, "candidate": "expired"}]),
            "memory_evidence_candidate_expired",
        )
    if case == "basis":
        return "memory_save", update, "memory_invalid_input"
    if case == "type":
        update.update(type="intent", user_quotes=[{"text": WORDS, "session": SESSION}])
        update.pop("evidence")
        return "memory_save", update, "memory_invalid_input"
    if case == "scope":
        return "memory_save", {**update, "scope": "global"}, "memory_invalid_input"
    if case in ("withdrawn", "restore"):
        withdrawn = await writer.withdraw(
            saved.ref, saved.version, QuoteClaim("user_quote", "candidate", WORDS, session=SESSION), configured.view
        )
        if case == "restore":
            return "memory_save", {**update, "version": withdrawn.version}, "memory_invalid_input"
        return (
            "memory_withdraw",
            {"ref": saved.ref, "version": withdrawn.version, "user_quote": WORDS, "session": SESSION},
            "memory_invalid_input",
        )
    if case == "withdraw_quote":
        return (
            "memory_withdraw",
            {"ref": saved.ref, "version": saved.version, "user_quote": "absent", "session": SESSION},
            "memory_quote_not_found",
        )
    if case == "merge_type":
        return (
            "memory_merge",
            {
                "keep_ref": saved.ref,
                "keep_version": saved.version,
                "drop_ref": configured.intent.ref,
                "drop_version": configured.intent.version,
            },
            "memory_invalid_input",
        )
    if case == "merge_sources":
        entries = []
        for index, words in enumerate((("alpha", "beta", "gamma", "delta"), ("epsilon", "zeta", "eta", "theta"))):
            entries.append(
                await writer.save(
                    SaveRequest(
                        "engineering",
                        f"Finding {index}",
                        f"Distinct finding {index}.",
                        evidence=tuple(ToolResultClaim(word, session=SESSION) for word in words),
                    ),
                    configured.view,
                )
            )
        left, right = entries
        return (
            "memory_merge",
            {"keep_ref": left.ref, "keep_version": left.version, "drop_ref": right.ref, "drop_version": right.version},
            "memory_invalid_input",
        )
    if case == "conflict":
        return "memory_note_conflict", {"ref_a": saved.ref, "ref_b": saved.ref}, "memory_invalid_input"
    raise ValueError(case)


@pytest.mark.parametrize(
    "case",
    [
        "decode",
        "ref",
        "missing",
        "quote",
        "evidence",
        "expired",
        "basis",
        "type",
        "scope",
        "withdrawn",
        "restore",
        "withdraw_quote",
        "merge_type",
        "merge_sources",
        "conflict",
    ],
)
async def test_explicit_input_refusals_complete_and_commit_the_exact_watermark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    configured = await trial(tmp_path)
    name, arguments, kind = await rejected_call(configured, case)
    before = await configured.store.list_entries(configured.source.project)
    if case == "decode":
        original = module.extraction_tool_schemas

        def permissive_schema(material: MaterialProfile):
            schemas = original(material)
            next(schema for schema in schemas if schema["name"] == "memory_read")["parameters"]["properties"][
                "ref"
            ] = {}
            return schemas

        monkeypatch.setattr(module, "extraction_tool_schemas", permissive_schema)
    report, record, wire = await run_trial(
        configured, monkeypatch, [response(call(name, arguments)), response(text="Done")]
    )
    assert (report.status, report.exit_code) == ("completed", 0)
    assert record.updated == configured.source.revision and record.attempts == 0
    summary = report.batches[0]
    assert (summary["rejection_count"], summary["failure_count"]) == (1, 0)
    assert summary["rejection_kinds"] == ((f"{name}:{kind}", 1),)
    assert summary["failure_kinds"] == ()
    assert await configured.store.list_entries(configured.source.project) == before
    result = next(
        content for message in wire.requests[-1][0] for content in message.contents if content.type == "function_result"
    )
    assert result.additional_properties[TOOL_RESULT_METADATA_KEY][TOOL_FAILED_METADATA_KEY] is True
    assert str(result.result).startswith("Error: Input rejected; no changes were made.")
    assert len(wire.requests) == 2


@pytest.mark.parametrize("material", [USER_V2, FULL_V3])
async def test_old_profiles_keep_input_refusals_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, material: MaterialProfile
) -> None:
    configured = await trial(tmp_path)
    report, record, wire = await run_trial(
        configured,
        monkeypatch,
        [response(call("memory_read", {"ref": "0123456789ab"})), response(text="Done")],
        material=material,
    )
    assert report.status == "failed", report
    assert report.batches[0]["failure_count"] == 1
    assert report.batches[0]["rejection_count"] == 0
    assert record.updated is None and record.attempts == 1
    assert len(wire.requests) == 2


@pytest.mark.parametrize("material", [FULL_V3, FULL_V4])
@pytest.mark.parametrize("source_defect", ["missing_tool", "long_args"])
async def test_unusable_tool_sources_allow_continuation_with_profile_specific_batch_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, material: MaterialProfile, source_defect: str
) -> None:
    configured = await trial(tmp_path)
    attributes = (("args", "{}"),) if source_defect == "missing_tool" else (("tool", "shell"), ("args", "x" * 201))
    source = replace(configured.source.messages[1], attributes=attributes)
    configured.source = replace(configured.source, messages=(configured.source.messages[0], source))
    before = await configured.store.list_entries(configured.source.project)
    report, record, wire = await run_trial(
        configured,
        monkeypatch,
        [
            response(
                call(
                    "memory_save",
                    {
                        "type": "engineering",
                        "title": "Another finding",
                        "body": "Another conclusion.",
                        "evidence": [{"kind": "tool_result", "quote": "alpha", "session": SESSION}],
                    },
                )
            ),
            response(text="Done"),
        ],
        material=material,
    )
    is_rejection = material == FULL_V4
    assert report.status == ("completed" if is_rejection else "failed")
    assert record.updated == (configured.source.revision if is_rejection else None)
    assert record.attempts == (0 if is_rejection else 1)
    summary = report.batches[0]
    assert (summary["rejection_count"], summary["failure_count"]) == ((1, 0) if is_rejection else (0, 1))
    kind = "memory_evidence_not_found" if source_defect == "missing_tool" else "memory_invalid_input"
    assert summary["rejection_kinds" if is_rejection else "failure_kinds"] == ((f"memory_save:{kind}", 1),)
    assert await configured.store.list_entries(configured.source.project) == before
    assert len(wire.requests) == 2
    if source_defect == "long_args":
        result = next(
            content
            for message in wire.requests[-1][0]
            for content in message.contents
            if content.type == "function_result"
        )
        assert "evidence[0].args is over 200 characters." in str(result.result)


@pytest.mark.parametrize("limit", ["output", "turn"])
@pytest.mark.parametrize("rejected", [False, True])
async def test_only_limits_without_any_tool_failure_revoke_the_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str, rejected: bool
) -> None:
    configured = await trial(tmp_path)
    first = call("memory_read", {"ref": "0123456789ab"}) if rejected else call("memory_search", {"query": ""})
    script = [response(first)]
    if limit == "output":
        script.append(response(status="incomplete", reason="max_output_tokens", text="partial"))
    report, record, wire = await run_trial(configured, monkeypatch, script, max_turns=1 if limit == "turn" else 24)
    assert report.status == ("failed" if rejected else "incomplete_limits")
    assert record.updated is None and record.attempts == int(rejected)
    assert report.batches[0]["rejection_count"] == int(rejected)
    assert len(wire.requests) == (1 if limit == "turn" else 2)


@pytest.mark.parametrize(
    "case", ["storage", "lock", "unknown", "unmarked", "version", "corrupt", "kernel", "guard", "partial"]
)
async def test_hard_failures_keep_progress_uncommitted_and_stop_before_another_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    configured = await trial(tmp_path)
    saved = configured.engineering
    item = call("memory_save", save_arguments())
    if case in ("storage", "lock", "unknown", "unmarked"):
        errors = {
            "storage": MemoryOperationError("memory_store_unavailable", "Unavailable."),
            "lock": TimeoutError("lock unavailable"),
            "unknown": RuntimeError("unexpected failure"),
            "unmarked": MemoryOperationError("memory_invalid_input", "An unmarked failure is not a refusal."),
        }

        async def fail_create(scope, draft):
            raise errors[case]

        monkeypatch.setattr(configured.store, "create", fail_create)
    elif case == "version":
        item = call(
            "memory_save",
            {
                "type": "engineering",
                "title": "Finding",
                "body": "Changed.",
                "ref": saved.ref,
                "version": "stale",
                "evidence": [{"kind": "tool_result", "quote": "alpha", "session": SESSION}],
            },
        )
    elif case == "corrupt":
        path = next(configured.options.memory_dir.rglob(saved.ref.split(":")[1] + ".md"))
        await asyncio.to_thread(path.write_text, "broken entry", encoding="utf-8")
        item = call("memory_read", {"ref": saved.ref})
    elif case == "kernel":
        item = call("memory_read", {"ref": 7})
    elif case == "guard":
        item = call("memory_search", {"query": ""}, call_id="")
    elif case == "partial":
        writer = MemoryWriter(configured.store, configured.source.project, "extraction")
        second = await writer.save(
            SaveRequest(
                "engineering", "Second", "Another observation.", evidence=(ToolResultClaim("beta", session=SESSION),)
            ),
            configured.view,
        )
        original = Path.unlink

        def refuse_drop(path: Path, missing_ok: bool = False) -> None:
            if path.name == second.ref.split(":")[1] + ".md":
                raise PermissionError("cannot delete the duplicate")
            original(path, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", refuse_drop)
        item = call(
            "memory_merge",
            {
                "keep_ref": saved.ref,
                "keep_version": saved.version,
                "drop_ref": second.ref,
                "drop_version": second.version,
            },
        )
    report, record, wire = await run_trial(
        configured, monkeypatch, [response(item), response(text="must not be requested")]
    )
    assert report.status == "failed" and report.exit_code == 1
    assert report.batches[0]["failure_count"] >= 1 and report.batches[0]["rejection_count"] == 0
    assert record.updated is None and record.attempts == 1
    assert len(wire.requests) == 1
    if case == "partial":
        assert report.batches[0]["merged_count"] == 1
        assert report.batches[0]["failure_kinds"] == (("memory_merge:merge_partial", 1),)


async def test_new_entry_can_be_reworded_from_its_saved_sources_in_full_v4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = await trial(tmp_path)

    async def reword(messages):
        entries = await configured.store.list_entries(configured.source.project)
        created = next(entry for entry in entries if entry.entry.title == "New finding")
        return response(
            call(
                "memory_save",
                {
                    "type": "engineering",
                    "title": "New finding",
                    "body": "Improved wording of beta.",
                    "ref": created.entry.ref,
                    "version": created.version,
                    "evidence": [{"kind": "tool_result", "quote": "beta", "session": SESSION}],
                },
            )
        )

    script = [
        response(
            call(
                "memory_save",
                {
                    "type": "engineering",
                    "title": "New finding",
                    "body": "Initial beta wording.",
                    "evidence": [{"kind": "tool_result", "quote": "beta", "session": SESSION}],
                },
            )
        ),
        reword,
        response(text="Done"),
    ]
    report, record, _ = await run_trial(configured, monkeypatch, script)
    assert report.status == "completed" and record.updated == configured.source.revision
    assert (
        report.batches[0]["created_count"],
        report.batches[0]["updated_count"],
        report.batches[0]["rejection_count"],
    ) == (1, 1, 0)
    assert any(
        entry.entry.body == "Improved wording of beta."
        for entry in await configured.store.list_entries(configured.source.project)
    )


async def test_unsettled_admitted_call_is_hard_and_cannot_request_another_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = await trial(tmp_path)
    held = []

    def echo(decoded):
        calls = [
            content for message in decoded.messages for content in message.contents if content.type == "function_call"
        ]
        if held:
            decoded.messages[0].contents = [held[0]]
        else:
            held.extend(calls)

    report, record, wire = await run_trial(
        configured,
        monkeypatch,
        [
            response(call("memory_search", {"query": ""})),
            response(call("memory_search", {"query": ""}), text="Done"),
            response(text="must not be requested"),
        ],
        decoded_hook=echo,
    )
    assert report.status == "failed" and report.batches[0]["unsettled_count"] == 1
    assert ("executor:unsettled", 1) in report.batches[0]["failure_kinds"]
    assert record.updated is None and record.attempts == 1
    assert len(wire.requests) == 2


async def test_hard_failure_retains_other_admitted_writes_before_stopping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = await trial(tmp_path)

    async def fail_read(scope, ref):
        raise MemoryOperationError("memory_store_unavailable", "Read unavailable.")

    monkeypatch.setattr(configured.store, "get", fail_read)
    report, record, wire = await run_trial(
        configured,
        monkeypatch,
        [
            response(
                call("memory_read", {"ref": configured.engineering.ref}, "read"),
                call("memory_save", save_arguments(), "save"),
            ),
            response(text="must not be requested"),
        ],
    )
    assert report.status == "failed" and record.updated is None
    assert (
        report.batches[0]["created_count"],
        report.batches[0]["failure_count"],
        report.batches[0]["unsettled_count"],
    ) == (1, 1, 0)
    assert any(
        entry.entry.title == "Keep changes small"
        for entry in await configured.store.list_entries(configured.source.project)
    )
    assert len(wire.requests) == 1


async def test_correcting_a_ref_preserves_the_rejection_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = await trial(tmp_path)
    report, record, wire = await run_trial(
        configured,
        monkeypatch,
        [
            response(call("memory_read", {"ref": configured.engineering.ref.split(":")[1]})),
            response(call("memory_read", {"ref": configured.engineering.ref})),
            response(text="Done"),
        ],
    )
    assert report.status == "completed" and record.updated == configured.source.revision
    assert (
        report.batches[0]["rejection_count"],
        report.batches[0]["failure_count"],
        report.batches[0]["success_count"],
    ) == (1, 0, 1)
    assert len(wire.requests) == 3


async def test_full_v3_continues_after_failure_keeps_later_writes_and_fails_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = await trial(tmp_path)
    report, record, wire = await run_trial(
        configured,
        monkeypatch,
        [
            response(call("memory_read", {"ref": "0123456789ab"})),
            response(call("memory_save", save_arguments())),
            response(text="Done"),
        ],
        material=FULL_V3,
    )
    assert len(wire.requests) == 3
    assert report.status == "failed" and report.batches[0]["failure_count"] == 1
    assert report.batches[0]["created_count"] == 1 and report.batches[0]["unsettled_count"] == 0
    assert report.batches[0]["rejection_count"] == 0
    assert record.updated is None and record.attempts == 1
    assert any(
        entry.entry.title == "Keep changes small"
        for entry in await configured.store.list_entries(configured.source.project)
    )


async def test_full_v4_full_store_is_hard_failure_and_never_commits_the_watermark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chrys.service.memory.storage import file_store

    configured = await trial(tmp_path)
    # The fixture's two real entries fill a smaller cap; admission and persistence are unchanged.
    monkeypatch.setattr(file_store, "MAX_ENTRIES_PER_SCOPE", 2)
    report, record, wire = await run_trial(
        configured,
        monkeypatch,
        [response(call("memory_save", save_arguments())), response(text="Done")],
    )
    assert (report.status, report.exit_code) == ("failed", 1)
    assert report.batches[0]["failure_count"] == 1
    assert report.batches[0]["failure_kinds"] == (("memory_save:memory_full", 1),)
    assert report.batches[0]["rejection_count"] == report.batches[0]["created_count"] == 0
    assert record.updated is None and record.attempts == 1
    assert len(await configured.store.list_entries(configured.source.project)) == 2
    assert len(wire.requests) == 1


@pytest.mark.parametrize("material", [USER_V2, FULL_V3, FULL_V4])
async def test_guard_rejection_still_stops_every_profile_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, material: MaterialProfile
) -> None:
    configured = await trial(tmp_path)
    report, record, wire = await run_trial(
        configured,
        monkeypatch,
        [response(call("memory_search", {"query": ""}, call_id="")), response(text="must not be requested")],
        material=material,
    )
    assert report.status == "failed" and report.batches[0]["created_count"] == 0
    assert report.batches[0]["failure_kinds"] == (("guard:invalid_call_id", 1),)
    assert record.updated is None and record.attempts == 1
    assert len(wire.requests) == 1
