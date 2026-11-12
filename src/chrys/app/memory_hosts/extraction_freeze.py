# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Publish a private regression set only after the existing replay path verifies it."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from chrys.app.memory_hosts.extraction_replay import FrozenSource, _state_file, load_manifest, text_digest
from chrys.app.memory_hosts.history import MAX_HISTORY_BYTES, object_value
from chrys.foundation.platform import get_platform
from chrys.foundation.platform.files import (
    _rename_file_no_replace,
    _validate_link_free_parent,
    atomic_write_owner_only_bytes,
    fsync_directory,
    secure_open_regular_binary,
    strip_inherited_directory_acl,
)
from chrys.foundation.trajectory.keys import ensure_owner_only_directory
from chrys.service.memory.extraction import (
    DEFAULT_MATERIAL_PROFILE,
    ExtractionBatch,
    ExtractionError,
    MaterialProfile,
)


@dataclass(frozen=True, slots=True)
class FreezeInputs:
    target: Path
    memory_dir: Path
    baseline_dir: Path | None
    protected_roots: tuple[Path, ...]
    project_id: str
    material_budget: int
    max_output_tokens: int
    sources: tuple[FrozenSource, ...]
    references: tuple[tuple[str, str], ...]
    batches: tuple[ExtractionBatch, ...]
    material_profile: MaterialProfile = DEFAULT_MATERIAL_PROFILE
    tuning_version: str = "v0"


@dataclass(frozen=True, slots=True)
class FreezePublication:
    """An exclusively created directory, removable only while its identity matches."""

    path: Path
    device: int
    inode: int

    def discard(self) -> None:
        info = self.path.lstat()
        if self.path.is_symlink() or (info.st_dev, info.st_ino) != (self.device, self.inode):
            raise ExtractionError("freeze_cleanup_failed", "The published regression set could not be removed safely.")
        shutil.rmtree(self.path)


def validate_destination(target: Path, memory_dir: Path, protected_roots: tuple[Path, ...]) -> None:
    """Validate the as-supplied parent chain before comparing resolved locations."""
    try:
        _validate_link_free_parent(target / ".freeze-check")
        if target.exists() or target.is_symlink():
            raise ExtractionError("freeze_target_exists", "The regression-set directory must not already exist.")
        resolved = target.resolve()
        for root in protected_roots:
            protected = root.resolve()
            if resolved.is_relative_to(protected) or protected.is_relative_to(resolved):
                raise ExtractionError(
                    "invalid_freeze_target", "The regression-set directory overlaps a protected root."
                )
        require_empty_store(memory_dir)
    except OSError, ValueError, RuntimeError:
        raise ExtractionError(
            "invalid_freeze_target", "The regression-set destination is not a safe new directory."
        ) from None


def require_empty_store(memory_dir: Path) -> None:
    """Use exactly the replay exclusion for state files; never create the store."""
    if memory_dir.exists() and not memory_dir.is_dir():
        raise ExtractionError(
            "freeze_store_not_empty", "Freeze requires an empty memory directory, excluding state files."
        )
    for path in memory_dir.rglob("*"):
        if path.is_symlink() or (path.is_file() and not _state_file(path.relative_to(memory_dir))):
            raise ExtractionError(
                "freeze_store_not_empty", "Freeze requires an empty memory directory, excluding state files."
            )


def _private_directory(path: Path) -> None:
    ensure_owner_only_directory(path)
    strip_inherited_directory_acl(path)


def _create_parents(path: Path) -> None:
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        _private_directory(directory)


def _put(root: Path, relative: Path, payload: bytes) -> None:
    directory = root
    for part in relative.parts[:-1]:
        directory /= part
        if not directory.exists():
            _private_directory(directory)
    atomic_write_owner_only_bytes(root / relative, payload, create_parents=False)


def _read_source(source: FrozenSource) -> bytes:
    try:
        with secure_open_regular_binary(source.path) as handle:
            payload = handle.read(MAX_HISTORY_BYTES + 1)
    except OSError:
        raise ExtractionError("source_unavailable", "A planned source could not be copied.") from None
    if len(payload) > MAX_HISTORY_BYTES or hashlib.sha256(payload).hexdigest() != source.sha256:
        raise ExtractionError("source_changed", "A planned source changed after its snapshot was read.")
    return payload


def _baseline_kind(payload: bytes) -> str | None:
    try:
        lines = payload.removeprefix(b"\xef\xbb\xbf").splitlines()
        if not lines or lines[0] != b"---":
            return None
        end = next((index for index, line in enumerate(lines[1:], 1) if line == b"---"), None)
        if end is None:
            raise ValueError("Unclosed frontmatter")
        front = object_value(yaml.safe_load(b"\n".join(lines[1:end])))
        metadata = object_value(front.get("metadata")) if front is not None else None
        kind = metadata.get("type") if metadata is not None else None
        return kind if isinstance(kind, str) and kind in ("feedback", "user") else None
    except UnicodeError, ValueError, yaml.YAMLError:
        raise ExtractionError("baseline_invalid", "Baseline frontmatter could not be read safely.") from None


def _stage(inputs: FreezeInputs) -> tuple[Path, dict[str, int]]:
    validate_destination(inputs.target, inputs.memory_dir, inputs.protected_roots)
    if not inputs.batches:
        raise ExtractionError("freeze_empty", "There are no planned batches to freeze.")
    if inputs.baseline_dir is not None and not inputs.baseline_dir.is_dir():
        raise ExtractionError("baseline_unavailable", "The baseline directory is unavailable.")
    _create_parents(inputs.target.parent)
    stage = Path(tempfile.mkdtemp(prefix=f".{inputs.target.name}.", dir=inputs.target.parent))
    try:
        _private_directory(stage)
        for name in ("sources", "references", "baseline"):
            _private_directory(stage / name)
        sources: list[dict[str, object]] = []
        for source in inputs.sources:
            if (
                source.key.host not in {"claude-code", "codex"}
                or Path(source.key.session).name != source.key.session
                or source.key.session in {".", "..", ""}
            ):
                raise ExtractionError("invalid_batch", "A planned source has an invalid storage identity.")
            relative = Path("sources") / source.key.host / f"{source.key.session}.jsonl"
            _put(stage, relative, _read_source(source))
            sources.append(
                {
                    "host": source.key.host,
                    "session": source.key.session,
                    "path": relative.as_posix(),
                    "sha256": source.sha256,
                    "updated_ns": source.updated_ns,
                }
            )
        references: list[dict[str, str]] = []
        for name, text in inputs.references:
            if name not in {"AGENTS.md", "CLAUDE.md", "CLAUDE.local.md"}:
                raise ExtractionError("invalid_batch", "A reference has an unsupported name.")
            relative = Path("references") / name
            payload = text.encode("utf-8")
            _put(stage, relative, payload)
            references.append(
                {"name": name, "path": relative.as_posix(), "sha256": hashlib.sha256(payload).hexdigest()}
            )
        baseline: list[dict[str, str]] = []
        counts = {"feedback": 0, "user": 0}
        if inputs.baseline_dir is not None:
            for path in sorted(inputs.baseline_dir.glob("*.md")):
                if not path.is_file():
                    continue
                with secure_open_regular_binary(path) as handle:
                    payload = handle.read()
                kind = _baseline_kind(payload)
                if kind is None:
                    continue
                relative = Path("baseline") / path.name
                _put(stage, relative, payload)
                baseline.append(
                    {"name": path.name, "path": relative.as_posix(), "sha256": hashlib.sha256(payload).hexdigest()}
                )
                counts[kind] += 1
        manifest = {
            "version": 1,
            "prompt_version": inputs.material_profile.prompt_version,
            "render_version": inputs.material_profile.render_version,
            "tuning_version": inputs.tuning_version,
            "project_id": inputs.project_id,
            "material_budget": inputs.material_budget,
            "max_output_tokens": inputs.max_output_tokens,
            "sources": sources,
            "references": references,
            "baseline": baseline,
            "initial_store": [],
            "batches": [
                {
                    "sessions": [asdict(source.key) for source in batch.sources],
                    "material_sha256": text_digest(batch.material),
                }
                for batch in inputs.batches
            ],
        }
        _put(stage, Path("manifest.json"), (json.dumps(manifest, ensure_ascii=True) + "\n").encode("utf-8"))
        return stage, counts
    except BaseException:
        shutil.rmtree(stage)
        raise


def _publish(stage: Path, inputs: FreezeInputs, report: Mapping[str, object]) -> FreezePublication:
    validate_destination(inputs.target, inputs.memory_dir, inputs.protected_roots)
    _put(stage, Path("report.json"), (json.dumps(report, ensure_ascii=True) + "\n").encode("utf-8"))
    info = stage.stat()
    _publish_directory(stage, inputs.target)
    fsync_directory(inputs.target.parent)
    return FreezePublication(inputs.target, info.st_dev, info.st_ino)


def _publish_directory(stage: Path, target: Path) -> None:
    # Native no-replace rename also refuses a racing empty target directory.
    if get_platform().is_windows:
        os.rename(stage, target)
    else:
        _rename_file_no_replace(stage, target)


async def freeze_regression_set(
    inputs: FreezeInputs,
    verify_replay: Callable[[Path], Awaitable[None]],
    report_snapshot: Callable[[dict[str, int]], Mapping[str, object]],
) -> FreezePublication:
    """Keep the temporary tree private until the normal replay planner accepts it."""
    stage: Path | None = None
    try:
        stage, counts = await asyncio.to_thread(_stage, inputs)
        manifest = stage / "manifest.json"
        frozen = await asyncio.to_thread(load_manifest, manifest)
        await asyncio.to_thread(frozen.verify_inputs, inputs.memory_dir)
        await verify_replay(manifest)
        report = report_snapshot(counts)
        return await asyncio.to_thread(_publish, stage, inputs, report)
    except FileExistsError:
        raise ExtractionError("freeze_target_exists", "The regression-set directory must not already exist.") from None
    except OSError, ValueError:
        raise ExtractionError("freeze_write_failed", "The regression set could not be written.") from None
    finally:
        if stage is not None:
            await asyncio.to_thread(_remove_stage, stage)


def _remove_stage(path: Path) -> None:
    with suppress(FileNotFoundError):
        shutil.rmtree(path)
