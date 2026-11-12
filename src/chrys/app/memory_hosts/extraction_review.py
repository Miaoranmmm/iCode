# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bounded review lists shared by extraction preflight and the native executor."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from chrys.foundation.text.tokenizer import MixedLanguageTokenizer
from chrys.service.memory.extraction import SourceSession
from chrys.service.memory.render import render_catalog_line
from chrys.service.memory.storage.store import GLOBAL_SCOPE, CatalogItem, MemoryScope, MemoryStore, StoredMemory

REVIEW_ENTRY_LIMIT = 20


@dataclass(frozen=True, slots=True)
class ReviewLists:
    related_entries: str = ""
    withdrawn_entries: str = ""


def _catalog_item(stored: StoredMemory) -> CatalogItem:
    entry = stored.entry
    return CatalogItem(entry.ref, entry.type, entry.title, entry.applies_when, entry.updated)


@dataclass(frozen=True, slots=True)
class ReviewSnapshot:
    """A single read snapshot; metadata lists never include bodies or source quotes."""

    entries: tuple[StoredMemory, ...] = ()

    def lists(
        self, sources: Sequence[SourceSession], *, budget_bound: bool = False, related_limit: int = REVIEW_ENTRY_LIMIT
    ) -> ReviewLists:
        refs = tuple(dict.fromkeys(ref for source in sources for ref in source.related_refs))
        by_ref = {stored.entry.ref: stored for stored in self.entries if stored.entry.status is None}
        related = [render_catalog_line(_catalog_item(by_ref[ref])).rstrip("\n") for ref in refs if ref in by_ref]
        if budget_bound:
            # Token units are additive. The largest permitted lines bound every
            # possible member subset without planning a second set of batches.
            related.sort(key=MixedLanguageTokenizer().count_token_units, reverse=True)
        withdrawn = []
        for stored in self.entries:
            if stored.entry.status == "withdrawn":
                title = (
                    render_catalog_line(replace(_catalog_item(stored), applies_when=None))
                    .split(" — ", 1)[1]
                    .rstrip("\n")
                )
                withdrawn.append(f"- {stored.entry.ref} — {title}")
        return ReviewLists("\n".join(related[:related_limit]), "\n".join(withdrawn[:REVIEW_ENTRY_LIMIT]))


async def load_review_snapshot(store: MemoryStore, project: MemoryScope | None) -> ReviewSnapshot:
    scopes = (project, GLOBAL_SCOPE) if project is not None and project.scope == "project" else (GLOBAL_SCOPE,)
    entries = [stored for scope in scopes for stored in await store.list_entries(scope)]
    entries.sort(key=lambda stored: stored.entry.ref)
    entries.sort(key=lambda stored: datetime.fromisoformat(stored.entry.updated), reverse=True)
    return ReviewSnapshot(tuple(entries))
