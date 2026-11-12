# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""In-process tests refuse an absent call anchor at once; the wait has its own tests."""

from __future__ import annotations

import pytest

from chrys.app.memory_hosts.claude_code import conversation


@pytest.fixture(autouse=True)
def _no_anchor_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conversation, "ANCHOR_WAIT_SECONDS", 0.0)
