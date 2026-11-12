# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Versioned extraction requests preserve frozen user material and admit engineering."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from chrys.app.memory_hosts import executor_icode as executor_module
from chrys.app.memory_hosts.executor_icode import ICodeExecutor, extraction_tool_schemas
from chrys.app.memory_hosts.extraction_replay import load_manifest, text_digest
from chrys.kernel import Message
from chrys.service.memory import extraction as core
from chrys.service.memory.extraction import build_batches, render_prompt
from chrys.service.memory.operations import MemoryWriter, SaveOutcome, SaveRequest
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import MemoryOperationError, MemoryScope
from chrys.service.memory.verification import ConversationView, SourceMessage
from tests.app.memory_hosts.executor_fakes import (
    PROJECT,
    RUN_ID,
    SESSION,
    ClientFactory,
    RecordingLog,
    ScriptedWire,
    batch,
    call,
    prepared,
    response,
)

FIXTURE = Path(__file__).parent / "fixtures" / "user_v2_regression"


def test_full_v3_request_bytes_stay_frozen() -> None:
    assert text_digest(core.FULL_V3.prompt) == "f6e92ac39275e8eefc2ef2088555fd6ef5fa61c3b13a427d0e5ed3025e114d3a"
    assert text_digest(core.FULL_V3.instructions) == "e0a02658d8350da98aa25bf6abb964b69f3c2dd181cf2f88df5f4ce45e495c83"
    assert (
        text_digest(json.dumps(extraction_tool_schemas(core.FULL_V3), ensure_ascii=True, sort_keys=True))
        == "d7dab4607d83d2e67e2807ac323a57703bcfa3e7a5ca8c0df575c33f00646421"
    )


def test_new_extraction_schema_accepts_engineering_without_file_evidence() -> None:
    schemas = extraction_tool_schemas()
    assert {schema["name"] for schema in schemas} == {
        "memory_search",
        "memory_read",
        "memory_save",
        "memory_withdraw",
        "memory_merge",
        "memory_note_conflict",
    }
    save = next(schema for schema in schemas if schema["name"] == "memory_save")["parameters"]
    assert save["properties"]["type"]["enum"] == ["intent", "engineering"]
    evidence = save["properties"]["evidence"]["items"]["oneOf"]
    assert {item["properties"]["kind"]["const"] for item in evidence} == {"tool_result", "reference", "user_report"}
    for item in evidence:
        if item["properties"]["kind"]["const"] != "reference":
            assert "session" in item["required"]
    assert save["dependentRequired"]["context_quote"] == ["context_session"]


def test_full_v4_describes_every_memory_ref_without_changing_legacy_schemas() -> None:
    schemas = {item["name"]: item["parameters"] for item in extraction_tool_schemas(core.FULL_V4)}
    guidance = (
        "Copy the complete ref returned by a tool, including its project: or global: prefix; never pass only the id."
    )
    for tool, parameter in (
        ("memory_read", "ref"),
        ("memory_save", "ref"),
        ("memory_withdraw", "ref"),
        ("memory_merge", "keep_ref"),
        ("memory_merge", "drop_ref"),
        ("memory_note_conflict", "ref_a"),
        ("memory_note_conflict", "ref_b"),
    ):
        assert schemas[tool]["properties"][parameter]["description"].count(guidance) == 1
    for parameter in ("ref", "version"):
        description = schemas["memory_save"]["properties"][parameter]["description"]
        assert all(term in description for term in ("same", "sources", "this batch", "latest version"))
    for legacy in (core.USER_V2, core.FULL_V3):
        old = {item["name"]: item["parameters"] for item in extraction_tool_schemas(legacy)}
        assert "description" not in old["memory_read"]["properties"]["ref"]
        assert "this batch" not in old["memory_save"]["properties"]["ref"]["description"]


@pytest.mark.parametrize("describe_refs", [False, True])
@pytest.mark.parametrize("created_rewrites", [False, True])
def test_ref_guidance_is_independent_of_created_rewrite_guidance(describe_refs: bool, created_rewrites: bool) -> None:
    material = replace(core.FULL_V4, describe_refs=describe_refs, allow_created_rewrites=created_rewrites)
    schemas = {item["name"]: item["parameters"] for item in extraction_tool_schemas(material)}
    assert ("description" in schemas["memory_read"]["properties"]["ref"]) is describe_refs
    for parameter in ("ref", "keep_ref", "drop_ref", "ref_a", "ref_b"):
        for schema in schemas.values():
            if parameter in schema["properties"]:
                description = schema["properties"][parameter].get("description", "")
                assert ("never pass only the id" in description) is describe_refs
    description = schemas["memory_save"]["properties"]["version"]["description"]
    assert ("created in this batch" in description) is created_rewrites


async def test_historical_v1_manifest_uses_original_user_v2_material_and_request() -> None:
    from chrys.app.memory_hosts.extraction_replay import load_frozen_sources

    manifest = await asyncio.to_thread(load_manifest, FIXTURE / "manifest.json")
    expected = json.loads(await asyncio.to_thread((FIXTURE / "expected.json").read_text, encoding="utf-8"))
    assert manifest.material_profile.name == "user-v2"
    assert (manifest.render_version, manifest.prompt_version, manifest.tuning_version) == ("2", "1", "v0")
    project = MemoryScope("project", manifest.project_id, root="/synthetic/trial-project", anchor="dir")
    scanned = await load_frozen_sources(manifest, project)
    batches = build_batches(project, scanned.sessions, manifest.material_budget, profile=manifest.material_profile)
    manifest.verify_batches(batches)
    assert [
        text_digest(
            render_prompt(item, "No saved memories.", manifest.read_references(), profile=manifest.material_profile)
        )
        for item in batches
    ] == expected["prompt_sha256"]
    assert text_digest(manifest.material_profile.instructions) == expected["instructions_sha256"]
    assert (
        text_digest(json.dumps(extraction_tool_schemas(manifest.material_profile), ensure_ascii=True, sort_keys=True))
        == expected["tool_schemas_sha256"]
    )
    assert all(message.kind == "user" for item in batches for message in item.source_index.view.messages)


@pytest.mark.parametrize("tuning", [None, "v0", "candidate-version"])
def test_manifest_accepts_optional_tuning_version(tmp_path: Path, tuning: str | None) -> None:
    raw = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    if tuning is not None:
        raw["tuning_version"] = tuning
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    manifest = load_manifest(path)
    assert manifest.tuning_version == (tuning or "v0")


async def test_full_v3_runs_engineering_with_real_decoder_and_core(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = batch(
        messages=(
            SourceMessage(SESSION, 1, ("Investigate this expensive failure.",), item="user"),
            SourceMessage(
                SESSION, 2, ("The adapter must retain the request boundary.",), kind="assistant", item="assistant"
            ),
            SourceMessage(
                SESSION,
                3,
                ("Synthetic diagnostic: request body was truncated.",),
                kind="tool_result",
                item="result",
                attributes=(("tool", "shell"), ("args", '{"command":"synthetic-check"}')),
            ),
        )
    )
    wire = ScriptedWire([])
    monkeypatch.setattr(executor_module, "create_client", ClientFactory(wire))
    store, log = FileMemoryStore(tmp_path / "memory"), RecordingLog()

    async def read_saved(messages: Sequence[Message]):
        stored = (await store.list_entries(PROJECT))[0]
        return response(call("memory_read", {"ref": stored.entry.ref}), text=None)

    wire.script = [
        response(call("memory_search", {"query": "request"})),
        response(
            call(
                "memory_save",
                {
                    "type": "engineering",
                    "title": "Retain request boundaries",
                    "body": "The adapter must retain the request boundary.",
                    "evidence": [{"kind": "tool_result", "quote": "request body was truncated.", "session": SESSION}],
                    "context_quote": "The adapter must retain the request boundary.",
                    "context_session": SESSION,
                },
            )
        ),
        read_saved,
        response(text="Done"),
    ]
    result = await ICodeExecutor(prepared(), store, log, RUN_ID, material_profile=core.FULL_V3).run(source)
    assert (result.termination, result.call_count, result.failure_count, result.created_count) == ("completed", 3, 0, 1)
    stored = (await store.list_entries(PROJECT))[0]
    assert stored.entry.type == "engineering"
    assert stored.entry.evidence[0].kind == "tool_result"
    assert stored.entry.evidence[0].tool == "shell"
    assert stored.entry.context is not None and stored.entry.context.session == SESSION
    assert log.changes[0].source == RUN_ID


@pytest.mark.parametrize("profile_name,kind", [("USER_V2", "engineering"), ("FULL_V3", "file")])
async def test_profile_writer_rejects_disallowed_type_or_file_after_schema_is_bypassed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, profile_name: str, kind: str
) -> None:
    arguments: dict[str, object] = {
        "type": "engineering",
        "title": "Unverified snapshot",
        "body": "Never capture a present-day snapshot.",
        "evidence": [{"kind": "reference", "ref": "synthetic-document"}],
    }
    if kind == "file":
        arguments["evidence"] = [{"kind": "file", "path": "source.py"}]
    wire = ScriptedWire(
        [
            response(call("memory_search", {"query": ""})),
            response(call("memory_save", arguments)),
            response(text="Done"),
        ]
    )
    monkeypatch.setattr(executor_module, "create_client", ClientFactory(wire))
    failures: list[str] = []

    class ObservedWriter(MemoryWriter):
        async def save(self, request: SaveRequest, view: ConversationView) -> SaveOutcome:
            try:
                return await super().save(request, view)
            except MemoryOperationError as error:
                failures.append(error.kind)
                raise

    monkeypatch.setattr(executor_module, "MemoryWriter", ObservedWriter)
    # A schema bypass is deliberate here: the writer remains the admission authority.
    schema_builder = extraction_tool_schemas

    def permissive_schema(profile: core.MaterialProfile = core.FULL_V3):
        schemas = list(schema_builder(core.FULL_V3))
        next(item for item in schemas if item["name"] == "memory_save")["parameters"] = {
            "type": "object",
            "additionalProperties": True,
        }
        return tuple(schemas)

    monkeypatch.setattr(executor_module, "extraction_tool_schemas", permissive_schema)
    store, log = FileMemoryStore(tmp_path / "memory"), RecordingLog()
    result = await ICodeExecutor(prepared(), store, log, RUN_ID, material_profile=getattr(core, profile_name)).run(
        batch()
    )
    assert result.failure_count == 1
    assert failures == ["memory_invalid_input"]
    assert await store.list_entries(PROJECT) == [] and log.changes == []


@pytest.mark.parametrize("versions", [("2", "2"), ("3", "1"), ("4", "2")])
def test_manifest_rejects_unknown_profile_pairs(tmp_path: Path, versions: tuple[str, str]) -> None:
    raw = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    raw["render_version"], raw["prompt_version"] = versions
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(core.ExtractionError) as caught:
        load_manifest(path)
    assert caught.value.kind == "invalid_replay"


@pytest.mark.parametrize("profile_name", ["USER_V2", "FULL_V3", "FULL_V4"])
def test_preflight_counts_the_same_profile_instructions_prompt_and_schema(
    monkeypatch: pytest.MonkeyPatch, profile_name: str
) -> None:
    from chrys.app.memory_hosts import extract
    from tests.app.memory_hosts.executor_fakes import profile

    material_profile = getattr(core, profile_name)
    fixed_inputs: list[str] = []

    def budget(context_window: int, output_cap: int, fixed_input: str, *, max_material_tokens: int) -> int:
        assert max_material_tokens == 40_000
        fixed_inputs.append(fixed_input)
        return 40_000

    monkeypatch.setattr(extract, "material_budget", budget)
    assert extract._budget(profile(), PROJECT, "CATALOG", (), material_profile) == 40_000
    empty = core.ExtractionBatch(PROJECT, (), "", core.SourceIndex(core.ConversationView((), None)))
    assert fixed_inputs == [
        material_profile.instructions
        + render_prompt(empty, "CATALOG", (), profile=material_profile)
        + json.dumps(extraction_tool_schemas(material_profile), ensure_ascii=False)
    ]


async def test_run_extraction_replays_v1_with_manifest_profile_instead_of_new_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from chrys.app.memory_hosts import extract
    from chrys.app.memory_hosts.extract import ExtractionOptions, run_extraction
    from tests.app.memory_hosts.executor_fakes import profile

    manifest = await asyncio.to_thread(load_manifest, FIXTURE / "manifest.json")
    project = MemoryScope("project", manifest.project_id, root="/synthetic/trial-project", anchor="dir")

    def binding(cwd: str) -> MemoryScope:
        return project

    monkeypatch.setattr(extract, "bind_project", binding)
    options = ExtractionOptions(tmp_path, tmp_path / "isolated-memory", dry_run=True, replay=FIXTURE / "manifest.json")
    report = await run_extraction(options, profile())
    assert (report.status, report.exit_code) == ("replay_dry_run", 0)
    assert report.profile["material_profile"] == "user-v2"
    assert (report.profile["render_version"], report.profile["prompt_version"]) == ("2", "1")
    assert [batch["sessions"] for batch in report.batches] == [
        [{"host": key.host, "session": key.session} for key in batch.keys] for batch in manifest.batches
    ]
    assert not await asyncio.to_thread(options.memory_dir.exists)
