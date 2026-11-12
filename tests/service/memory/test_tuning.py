# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Immutable memory tuning versions and explicit activation from step 9."""

from __future__ import annotations

import json
import stat
import sys
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import pytest

from chrys.service.memory.extraction import (
    DEFAULT_BATCH_TIMEOUT_SECONDS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_TURNS,
    FULL_EXTRACTION_PROMPT,
    MAX_BATCHES,
    MAX_MATERIAL_TOKENS,
)
from chrys.service.memory.render import CATALOG_TOKEN_BUDGET
from chrys.service.memory.tuning import (
    V0,
    MemoryTuning,
    apply_version,
    list_versions,
    load_active,
    load_candidate,
    load_version,
    propose,
)
from tests.support.secure_files import plant_owner_only_bytes


def test_v0_preserves_the_current_defaults_and_is_immutable(tmp_path: Path) -> None:
    assert load_active(tmp_path) is V0
    assert not tuple(tmp_path.iterdir())
    assert V0.version == "v0"
    assert V0.extraction_types == ("intent", "engineering")
    assert V0.catalog_budget == CATALOG_TOKEN_BUDGET == 2000
    assert V0.material_budget == MAX_MATERIAL_TOKENS == 40000
    assert V0.batch_cap == MAX_BATCHES == 3
    assert V0.max_turns == DEFAULT_MAX_TURNS == 24
    assert V0.max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS == 16384
    assert V0.batch_timeout_seconds == DEFAULT_BATCH_TIMEOUT_SECONDS == 600
    assert V0.related_entry_cap == 20
    assert V0.oversize_selection == "chronological"
    assert V0.catalog_order == "intent_first"
    assert V0.guidance_strength == "read_first"
    assert FULL_EXTRACTION_PROMPT.startswith(V0.task_description)
    assert V0.do_not_save == "Do not save one-off tasks, mere questions, narratives, or untested hypotheses."
    assert V0.pasted_text is None and V0.body_limits is None
    with pytest.raises(FrozenInstanceError):
        V0.catalog_budget = 1  # type: ignore[misc]


def test_proposal_does_not_activate_and_versions_are_never_overwritten(tmp_path: Path) -> None:
    first = propose(tmp_path, replace(V0, oversize_selection="signal_first"))
    path = tmp_path / "memory-tuning" / "versions" / "1.json"
    before = path.read_bytes()
    assert first.version == "v1"
    assert load_active(tmp_path) == V0
    assert not (tmp_path / "memory-tuning" / "active").exists()
    second = propose(tmp_path, replace(V0, extraction_types=("intent",)))
    assert second.version == "v2"
    assert path.read_bytes() == before
    assert load_version(tmp_path, "1") == load_version(tmp_path, "v1") == first
    assert list_versions(tmp_path) == (V0, first, second)
    apply_version(tmp_path, "1")
    assert (tmp_path / "memory-tuning" / "active").read_bytes() == b"1\n"
    assert load_active(tmp_path) == first
    apply_version(tmp_path, "v0")
    assert (tmp_path / "memory-tuning" / "active").read_bytes() == b"0\n"
    assert load_active(tmp_path) == V0
    if stat.S_IMODE(path.stat().st_mode) != 0o600 and sys.platform != "win32":
        pytest.fail("A tuning version must be owner-only.")


def test_version_written_before_the_prompt_slots_still_loads_with_both_prompts_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"version": "v1", "oversize_selection": "signal_first"}), encoding="utf-8")
    candidate = load_candidate(tmp_path, path)
    assert (candidate.pasted_text, candidate.body_limits) == (None, None)


def test_json_candidate_and_version_use_the_same_validator(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    payload = asdict(replace(V0, material_budget=45000, catalog_budget=1000))
    path.write_text(json.dumps(payload), encoding="utf-8")
    candidate = load_candidate(tmp_path, path)
    assert candidate.material_budget == 45000
    assert candidate.extraction_types == ("intent", "engineering")
    version = propose(tmp_path, candidate)
    assert load_candidate(tmp_path, version.version) == version
    payload["verbatim"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown tuning fields: verbatim"):
        load_candidate(tmp_path, path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("catalog_budget", 0),
        ("material_budget", -1),
        ("batch_cap", True),
        ("max_turns", 0),
        ("max_output_tokens", 1.5),
        ("batch_timeout_seconds", float("inf")),
        ("related_entry_cap", -1),
        ("task_description", ""),
        ("do_not_save", "  "),
        ("pasted_text", ""),
        ("body_limits", 3),
        ("extraction_types", ()),
        ("extraction_types", ("intent", "intent")),
        ("extraction_types", ("review",)),
        ("catalog_order", "random"),
        ("guidance_strength", "execute"),
        ("oversize_selection", "random"),
        ("version", "../v1"),
    ],
)
def test_invalid_tuning_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(V0, **{field: value})


def test_broken_active_is_not_silently_replaced_by_default(tmp_path: Path) -> None:
    root = tmp_path / "memory-tuning"
    root.mkdir()
    plant_owner_only_bytes(root / "active", b"v93\n")
    with pytest.raises(ValueError, match="version v93 is unavailable"):
        load_active(tmp_path)
    assert MemoryTuning is type(V0)
