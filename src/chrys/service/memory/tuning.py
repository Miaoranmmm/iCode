# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Immutable, explicitly activated memory tuning outside the memory store."""

from __future__ import annotations

import errno
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Literal

from chrys.foundation.platform.files import atomic_write_owner_only_bytes, secure_open_owner_only_binary
from chrys.foundation.util.lock import FileLock
from chrys.service.memory.storage.store import MemoryType

CatalogOrder = Literal["intent_first", "updated_first"]
GuidanceStrength = Literal["describe", "read_first"]
OversizeSelection = Literal["chronological", "head_tail", "signal_first"]
TASK_DESCRIPTION_V0 = "You curate durable user principles and reusable engineering findings across sessions."
DO_NOT_SAVE_V0 = "Do not save one-off tasks, mere questions, narratives, or untested hypotheses."
_VERSION = re.compile(r"v?(0|[1-9][0-9]*)\Z")


def _version(value: str) -> str:
    match = _VERSION.fullmatch(value)
    if match is None:
        raise ValueError("A tuning version must be v0 or a positive version number.")
    return "v" + match[1]


@dataclass(frozen=True, slots=True)
class MemoryTuning:
    """Only policy and resource choices; source verification is never configurable."""

    version: str = "v0"
    extraction_types: tuple[MemoryType, ...] = ("intent", "engineering")
    task_description: str = TASK_DESCRIPTION_V0
    do_not_save: str = DO_NOT_SAVE_V0
    # None keeps each prompt's own sentence; the two prompts word pasted_text differently.
    pasted_text: str | None = None
    body_limits: str | None = None
    catalog_budget: int = 2000
    catalog_order: CatalogOrder = "intent_first"
    guidance_strength: GuidanceStrength = "read_first"
    material_budget: int = 40000
    batch_cap: int = 3
    max_turns: int = 24
    max_output_tokens: int = 16384
    batch_timeout_seconds: float = 600
    related_entry_cap: int = 20
    oversize_selection: OversizeSelection = "chronological"

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or _version(self.version) != self.version:
            raise ValueError("The stored tuning version must use its canonical v<number> spelling.")
        if (
            not isinstance(self.extraction_types, tuple)
            or not self.extraction_types
            or any(
                not isinstance(value, str) or value not in {"intent", "engineering"} for value in self.extraction_types
            )
            or len(set(self.extraction_types)) != len(self.extraction_types)
        ):
            raise ValueError("extraction_types must be a nonempty unique tuple of intent and engineering.")
        for name, value in (("task_description", self.task_description), ("do_not_save", self.do_not_save)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty text.")
        for name, value in (("pasted_text", self.pasted_text), ("body_limits", self.body_limits)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be null or nonempty text.")
        positive = (
            ("catalog_budget", self.catalog_budget),
            ("material_budget", self.material_budget),
            ("batch_cap", self.batch_cap),
            ("max_turns", self.max_turns),
            ("max_output_tokens", self.max_output_tokens),
        )
        for name, value in positive:
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.related_entry_cap) is not int or self.related_entry_cap < 0:
            raise ValueError("related_entry_cap must be a nonnegative integer.")
        if (
            isinstance(self.batch_timeout_seconds, bool)
            or not isinstance(self.batch_timeout_seconds, (int, float))
            or not math.isfinite(self.batch_timeout_seconds)
            or self.batch_timeout_seconds <= 0
        ):
            raise ValueError("batch_timeout_seconds must be a finite positive number.")
        if not isinstance(self.catalog_order, str) or self.catalog_order not in {"intent_first", "updated_first"}:
            raise ValueError("catalog_order must be intent_first or updated_first.")
        if not isinstance(self.guidance_strength, str) or self.guidance_strength not in {"describe", "read_first"}:
            raise ValueError("guidance_strength must be describe or read_first.")
        if not isinstance(self.oversize_selection, str) or self.oversize_selection not in {
            "chronological",
            "head_tail",
            "signal_first",
        }:
            raise ValueError("oversize_selection must be chronological, head_tail or signal_first.")


# This literal version remains the baseline even as future candidates are introduced.
V0 = MemoryTuning()


def _decode(value: object) -> MemoryTuning:
    if not isinstance(value, dict):
        raise ValueError("A tuning file must contain a JSON object.")
    unknown = set(value) - {field.name for field in fields(MemoryTuning)}
    if unknown:
        raise ValueError("Unknown tuning fields: " + ", ".join(sorted(unknown)))
    values = dict(value)
    if "extraction_types" in values:
        types = values["extraction_types"]
        if not isinstance(types, (list, tuple)):
            raise ValueError("extraction_types must be an array.")
        values["extraction_types"] = tuple(types)
    try:
        return MemoryTuning(**values)
    except TypeError as error:
        raise ValueError("A tuning file contains fields with invalid types.") from error


def _read(path: Path, *, secure: bool = True) -> MemoryTuning:
    try:
        if secure:
            with secure_open_owner_only_binary(path) as stream:
                return _decode(json.load(stream))
        with path.open(encoding="utf-8") as stream:
            return _decode(json.load(stream))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Tuning file {path.name} is unavailable or invalid.") from error


def load_version(config_dir: Path, version: str) -> MemoryTuning:
    """Read one immutable version, with v0 always supplied by code."""
    canonical = _version(version)
    if canonical == "v0":
        return V0
    path = config_dir / "memory-tuning" / "versions" / f"{canonical[1:]}.json"
    try:
        tuning = _read(path)
    except ValueError as error:
        raise ValueError(f"Tuning version {canonical} is unavailable or invalid.") from error
    if tuning.version != canonical:
        raise ValueError(f"Tuning version {canonical} has a mismatched identity.")
    return tuning


def load_active(config_dir: Path) -> MemoryTuning:
    """Take one startup snapshot; proposals alone never change the active version."""
    path = config_dir / "memory-tuning" / "active"
    try:
        with secure_open_owner_only_binary(path) as stream:
            version = stream.read().decode("utf-8")
    except OSError as error:
        if error.errno == errno.ENOENT:
            return V0
        raise ValueError("The active tuning version is unavailable.") from error
    except UnicodeError as error:
        raise ValueError("The active tuning version is unavailable.") from error
    if not version.endswith("\n") or len(version.splitlines()) != 1:
        raise ValueError("The active tuning file must contain exactly one version line.")
    return load_version(config_dir, version.rstrip("\n"))


def load_candidate(config_dir: Path, selector: str | Path) -> MemoryTuning:
    """Resolve a version or a candidate JSON file without activating it."""
    if isinstance(selector, str) and _VERSION.fullmatch(selector):
        return load_version(config_dir, selector)
    return load_candidate_file(Path(selector))


def load_candidate_file(path: Path) -> MemoryTuning:
    """Read a proposal file, even when its filename looks like a version number."""
    return _read(path, secure=False)


def list_versions(config_dir: Path) -> tuple[MemoryTuning, ...]:
    """List the baseline and all installed version files in numeric order."""
    directory = config_dir / "memory-tuning" / "versions"
    versions = sorted(
        (int(path.stem) for path in directory.glob("*.json") if re.fullmatch(r"[1-9][0-9]*", path.stem)),
    )
    return (V0, *(load_version(config_dir, str(version)) for version in versions))


def propose(config_dir: Path, candidate: MemoryTuning | Mapping[str, object]) -> MemoryTuning:
    """Allocate a new immutable version under one cross-process lock."""
    checked = candidate if isinstance(candidate, MemoryTuning) else _decode(dict(candidate))
    directory = config_dir / "memory-tuning"
    versions = directory / "versions"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    versions.mkdir(mode=0o700, exist_ok=True)
    with FileLock(directory / ".lock", timeout=5):
        number = max((int(path.stem) for path in versions.glob("*.json") if path.stem.isdecimal()), default=0) + 1
        result = replace(checked, version=f"v{number}")
        payload = (json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        atomic_write_owner_only_bytes(versions / f"{number}.json", payload)
    return result


def apply_version(config_dir: Path, version: str) -> MemoryTuning:
    """Publish only an existing version; the CLI owns human confirmation."""
    tuning = load_version(config_dir, version)
    directory = config_dir / "memory-tuning"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with FileLock(directory / ".lock", timeout=5):
        atomic_write_owner_only_bytes(directory / "active", (tuning.version[1:] + "\n").encode("utf-8"))
    return tuning
