# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Extract review targets from proven memory calls without admitting their text as evidence."""

from __future__ import annotations

import re
from collections.abc import Sequence

_REF = re.compile(r"(?:project|global):[0-9a-f]{12}")
_REF_ARGUMENT_TOOLS = frozenset({"memory_read", "memory_save", "memory_forget", "memory_withdraw"})


def call_refs(tool: str, arguments: object) -> tuple[str, ...]:
    """Only the four specified memory operations name a review target in ref."""
    if tool.rsplit("__", 1)[-1] not in _REF_ARGUMENT_TOOLS or not isinstance(arguments, dict):
        return ()
    ref = arguments.get("ref")
    return (ref,) if isinstance(ref, str) and _REF.fullmatch(ref) else ()


def result_refs(fragments: Sequence[str]) -> tuple[str, ...]:
    """The caller proves this is a paired memory result before scanning its display text."""
    return tuple(dict.fromkeys(match.group() for fragment in fragments for match in _REF.finditer(fragment)))
