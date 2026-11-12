# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Only the full material contract gains review tools, targets, and instructions."""

from __future__ import annotations

from dataclasses import replace

from chrys.service.memory.extraction import FULL_V3, USER_V2, build_batches, project_source, render_prompt
from tests.service.memory.test_extraction_material_profiles import PROJECT, source, user


def test_review_tool_allowlist_belongs_to_the_material_profile() -> None:
    assert USER_V2.tool_names == ("memory_search", "memory_read", "memory_save")
    assert FULL_V3.tool_names == (
        "memory_search",
        "memory_read",
        "memory_save",
        "memory_withdraw",
        "memory_merge",
        "memory_note_conflict",
    )


def test_full_review_prompt_reuses_the_batch_and_keeps_target_lists_outside_quotable_sources() -> None:
    original = replace(source((user(1),)), related_refs=("project:111111111111",))
    batch = build_batches(PROJECT, (original,), 40_000, profile=FULL_V3)[0]
    prompt = render_prompt(
        batch,
        "CATALOG",
        profile=FULL_V3,
        related_entries="project:111111111111 intent | Related title",
        withdrawn_entries="global:222222222222 Withdrawn title",
    )
    assert "<related_entries>\nproject:111111111111 intent | Related title\n</related_entries>" in prompt
    assert "<withdrawn_entries>\nglobal:222222222222 Withdrawn title\n</withdrawn_entries>" in prompt
    assert "Related title" not in batch.material and "Withdrawn title" not in batch.material
    assert batch.source_index.view.messages == original.messages
    assert batch.sources[0].related_refs == original.related_refs
    assert "<related_entries>\nnone\n</related_entries>" in render_prompt(batch, "CATALOG", profile=FULL_V3)


def test_legacy_projection_and_prompt_ignore_review_metadata_byte_for_byte() -> None:
    plain = source((user(1),))
    with_refs = replace(plain, related_refs=("project:111111111111",))
    assert project_source(with_refs, USER_V2) == plain
    batch = build_batches(PROJECT, (with_refs,), 40_000, profile=USER_V2)[0]
    expected = render_prompt(batch, "CATALOG", (("AGENTS.md", "REFERENCE"),), profile=USER_V2)
    actual = render_prompt(
        batch,
        "CATALOG",
        (("AGENTS.md", "REFERENCE"),),
        profile=USER_V2,
        related_entries="PRIVATE RELATED",
        withdrawn_entries="PRIVATE WITHDRAWN",
    )
    assert actual == expected
    assert "related_entries" not in actual and "withdrawn_entries" not in actual


def test_full_review_instructions_require_explicit_words_and_lossless_merge() -> None:
    prompt = FULL_V3.prompt
    for required in (
        "Read related_entries first",
        "memory_withdraw",
        "memory_note_conflict",
        "memory_merge",
        "incidental",
        "three quotes",
        "six evidence",
        "after withdrawing",
        "both entries unchanged",
    ):
        assert required in prompt
    assert "No delete, withdrawal, merge, or review operation is available" not in prompt
    assert "No delete, withdrawal, or review operation is available" in USER_V2.prompt
