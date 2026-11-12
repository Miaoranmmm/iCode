# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Catalog tuning changes policy while the built-in v0 stays byte exact."""

from __future__ import annotations

from dataclasses import replace

from chrys.service.memory.operations import CatalogSnapshot
from chrys.service.memory.render import render_catalog
from chrys.service.memory.storage.store import CatalogItem
from chrys.service.memory.tuning import V0


def snapshot() -> CatalogSnapshot:
    return CatalogSnapshot(
        (
            CatalogItem("project:111111111111", "intent", "First choice", None, "2026-11-01T00:00:00Z"),
            CatalogItem("project:222222222222", "engineering", "New observation", None, "2026-11-02T00:00:00Z"),
        ),
        (),
    )


def test_v0_catalog_is_identical_and_sorting_is_configurable() -> None:
    original = render_catalog(snapshot())
    assert original is not None
    assert render_catalog(snapshot(), tuning=V0) == original
    assert original.index("First choice") < original.index("New observation")
    updated = render_catalog(snapshot(), tuning=replace(V0, catalog_order="updated_first"))
    assert updated is not None
    assert updated.index("New observation") < updated.index("First choice")


def test_describe_guidance_removes_only_the_read_first_instruction() -> None:
    original = render_catalog(snapshot())
    assert original is not None
    describe = render_catalog(snapshot(), tuning=replace(V0, guidance_strength="describe"))
    assert describe == original.replace("Read an entry with memory_read before relying on it. ", "")


def test_catalog_budget_comes_from_the_selected_tuning() -> None:
    items = tuple(
        CatalogItem(f"project:{index:012x}", "intent", "A detailed title " * 20, None, "2026-11-01T00:00:00Z")
        for index in range(20)
    )
    data = CatalogSnapshot(items, ())
    original = render_catalog(data)
    limited = render_catalog(data, tuning=replace(V0, catalog_budget=300))
    assert original is not None and limited is not None
    assert limited.count("- project:") < original.count("- project:")
    assert "more entries not listed" in limited


def test_tiny_catalog_budget_does_not_publish_an_over_budget_header() -> None:
    assert render_catalog(snapshot(), tuning=replace(V0, catalog_budget=1)) is None
