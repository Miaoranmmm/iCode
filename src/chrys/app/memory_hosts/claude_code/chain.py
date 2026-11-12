# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cycle-checked parent traversal shared by interactive and historical views."""

from __future__ import annotations

from collections.abc import Callable, Hashable


class InvalidChain(ValueError):
    """A parent is missing or one physical occurrence is visited twice."""


def walk_parent_chain[K: Hashable, N](tail: K, resolve: Callable[[K], N], parent: Callable[[N], K | None]) -> list[N]:
    """Return tail-to-root occurrences, with no assumptions about compact policy."""
    result: list[N] = []
    seen: set[K] = set()
    current: K | None = tail
    while current is not None:
        if current in seen:
            raise InvalidChain("Repeated physical transcript occurrence.")
        seen.add(current)
        try:
            node = resolve(current)
        except KeyError:
            raise InvalidChain("Missing transcript parent.") from None
        result.append(node)
        current = parent(node)
    return result
