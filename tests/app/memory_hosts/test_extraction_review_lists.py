# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Review coverage is bounded metadata and budgeted with the same line renderer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from chrys.app.memory_hosts.extraction_review import load_review_snapshot
from chrys.foundation.text.tokenizer import MixedLanguageTokenizer
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import GLOBAL_SCOPE, MemoryEntryDraft, Quote
from tests.app.memory_hosts.executor_fakes import PROJECT, batch


async def test_review_lists_include_related_and_withdrawn_metadata_only(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    refs = []
    for index in range(24):
        withdrawn = index % 2 == 0
        scope = PROJECT if index % 3 else GLOBAL_SCOPE
        result = await store.create(
            scope,
            MemoryEntryDraft(
                "intent",
                f"Conclusion {index}",
                "PRIVATE_BODY",
                "model",
                quotes=(Quote(f"PRIVATE_QUOTE {index}", "message", "synthetic-source"),),
                status="withdrawn" if withdrawn else None,
                status_basis=Quote(
                    "PRIVATE_WITHDRAWAL",
                    "message",
                    "synthetic-source",
                    at="2026-11-11T00:00:00Z",
                    item=f"withdraw-{index}",
                )
                if withdrawn
                else None,
            ),
        )
        refs.append(result.ref)
    snapshot = await load_review_snapshot(store, PROJECT)
    source = replace(batch().sources[0], related_refs=(*refs, *refs, "project:ffffffffffff"))
    lists = snapshot.lists((source,))
    assert len(lists.related_entries.splitlines()) == 12
    assert len(lists.withdrawn_entries.splitlines()) == 12
    assert all(ref in lists.related_entries for ref in refs[1::2])
    assert all(ref not in lists.related_entries for ref in refs[::2])
    assert "PRIVATE_" not in lists.related_entries + lists.withdrawn_entries
    assert "global:" in lists.withdrawn_entries and "project:" in lists.withdrawn_entries
    assert all(" — Conclusion " in line for line in lists.withdrawn_entries.splitlines())


async def test_withdrawn_list_caps_more_than_twenty_rows(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path)
    for index in range(23):
        await store.create(
            PROJECT,
            MemoryEntryDraft(
                "intent",
                f"Withdrawn {index}",
                "body",
                "model",
                quotes=(Quote("words", "message", "s"),),
                status="withdrawn",
                status_basis=Quote("stop", "message", "s", "2026-11-11T00:00:00Z"),
            ),
        )
    lists = (await load_review_snapshot(store, PROJECT)).lists(())
    assert len(lists.withdrawn_entries.splitlines()) == 20


async def test_budget_bound_covers_any_batch_members_even_when_first_twenty_refs_are_short(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    refs = []
    for index in range(26):
        saved = await store.create(
            PROJECT,
            MemoryEntryDraft(
                "intent",
                ("短" * 70 if index >= 20 else "x") + str(index),
                "body",
                "model",
                quotes=(Quote(str(index), "message", "source"),),
            ),
        )
        refs.append(saved.ref)
    snapshot = await load_review_snapshot(store, PROJECT)
    first = replace(batch().sources[0], related_refs=tuple(refs))
    bound = snapshot.lists((first,), budget_bound=True)
    later = snapshot.lists((replace(first, related_refs=tuple(refs[10:])),))
    counter = MixedLanguageTokenizer()
    assert len(bound.related_entries.splitlines()) == 20
    assert counter.count_token_units(bound.related_entries) >= counter.count_token_units(later.related_entries)
