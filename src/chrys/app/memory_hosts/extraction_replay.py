# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Frozen extraction inputs, independent of live host directories and watermarks."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from chrys.service.memory.extraction import (
    DEFAULT_MATERIAL_PROFILE,
    ExtractionError,
    MaterialProfile,
    ScanResult,
    SessionKey,
    SourceRevision,
    conflicting_sessions,
    material_profile_for_versions,
)

if TYPE_CHECKING:
    from chrys.service.memory.extraction import ExtractionBatch
    from chrys.service.memory.storage.store import MemoryScope


def _error() -> ExtractionError:
    return ExtractionError("invalid_replay", "The frozen replay manifest or input does not match.")


@dataclass(frozen=True, slots=True)
class FrozenSource:
    key: SessionKey
    path: Path
    sha256: str
    updated_ns: int


@dataclass(frozen=True, slots=True)
class FrozenFile:
    name: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class FrozenBatch:
    keys: tuple[SessionKey, ...]
    material_sha256: str


@dataclass(frozen=True, slots=True)
class ReplayManifest:
    """Version 1 uses paths relative to the manifest, except initial-store paths."""

    project_id: str
    material_budget: int
    max_output_tokens: int
    sources: tuple[FrozenSource, ...]
    references: tuple[FrozenFile, ...]
    baseline: tuple[FrozenFile, ...]
    initial_store: tuple[tuple[str, str], ...]
    batches: tuple[FrozenBatch, ...]
    render_version: str = DEFAULT_MATERIAL_PROFILE.render_version
    prompt_version: str = DEFAULT_MATERIAL_PROFILE.prompt_version
    tuning_version: str = "v0"

    @property
    def material_profile(self) -> MaterialProfile:
        return material_profile_for_versions(self.render_version, self.prompt_version)

    def read_references(self) -> tuple[tuple[str, str], ...]:
        result: list[tuple[str, str]] = []
        for file in self.references:
            payload = file.path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != file.sha256:
                raise _error()
            result.append((file.name, payload.decode("utf-8")))
        return tuple(result)

    def verify_inputs(self, memory_dir: Path) -> None:
        self.verify_source_files()
        actual = tuple(
            sorted(
                (path.relative_to(memory_dir).as_posix(), digest(path))
                for path in memory_dir.rglob("*")
                if path.is_file() and not _state_file(path.relative_to(memory_dir))
            )
        )
        if actual != tuple(sorted(self.initial_store)):
            raise _error()

    def verify_source_files(self) -> None:
        """Verify frozen content without requiring a memory store or live host paths."""
        try:
            for file in (*self.sources, *self.references, *self.baseline):
                if digest(file.path) != file.sha256:
                    raise _error()
        except OSError as error:
            raise _error() from error

    def verify_batches(self, batches: tuple[ExtractionBatch, ...]) -> None:
        actual = tuple(
            FrozenBatch(tuple(source.key for source in batch.sources), text_digest(batch.material)) for batch in batches
        )
        if actual != self.batches:
            raise _error()


def _state_file(path: Path) -> bool:
    return path.name.endswith(".lock") or path.as_posix() == "extraction.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _object(value: object, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise _error()
    return value


def _string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise _error()
    return value


def _number(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _error()
    return value


def _array(value: object) -> list[object]:
    if not isinstance(value, list):
        raise _error()
    return value


def _hash(value: object) -> str:
    text = _string(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise _error()
    return text


def _key(value: object) -> SessionKey:
    item = _object(value, {"host", "session"})
    host = _string(item["host"])
    if host not in {"claude-code", "codex"}:
        raise _error()
    return SessionKey(host, _string(item["session"]))


def load_manifest(path: Path) -> ReplayManifest:
    """Read only frozen locations; an invalid manifest never falls back to live inputs."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        keys = {
            "version",
            "prompt_version",
            "render_version",
            "project_id",
            "material_budget",
            "max_output_tokens",
            "sources",
            "references",
            "baseline",
            "initial_store",
            "batches",
        }
        if isinstance(raw, dict) and "tuning_version" in raw:
            keys.add("tuning_version")
        root = _object(raw, keys)
        if type(root["version"]) is not int or root["version"] != 1:
            raise _error()
        render_version, prompt_version = _string(root["render_version"]), _string(root["prompt_version"])
        material_profile_for_versions(render_version, prompt_version)
        tuning_version = _string(root.get("tuning_version", "v0"))
        sources: list[FrozenSource] = []
        for raw in _array(root["sources"]):
            item = _object(raw, {"host", "session", "path", "sha256", "updated_ns"})
            key = _key({"host": item["host"], "session": item["session"]})
            sources.append(
                FrozenSource(
                    key, path.parent / _string(item["path"]), _hash(item["sha256"]), _number(item["updated_ns"])
                )
            )
        if not sources or len({source.key for source in sources}) != len(sources):
            raise _error()

        def files(field: str) -> tuple[FrozenFile, ...]:
            result: list[FrozenFile] = []
            for raw in _array(root[field]):
                item = _object(raw, {"name", "path", "sha256"})
                result.append(
                    FrozenFile(_string(item["name"]), path.parent / _string(item["path"]), _hash(item["sha256"]))
                )
            return tuple(result)

        references = files("references")
        if len({file.name for file in references}) != len(references) or any(
            file.name not in {"AGENTS.md", "CLAUDE.md", "CLAUDE.local.md"} for file in references
        ):
            raise _error()
        initial: list[tuple[str, str]] = []
        for raw in _array(root["initial_store"]):
            item = _object(raw, {"path", "sha256"})
            name = _string(item["path"])
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or _state_file(relative):
                raise _error()
            initial.append((name, _hash(item["sha256"])))
        if len(dict(initial)) != len(initial):
            raise _error()
        batches: list[FrozenBatch] = []
        for raw in _array(root["batches"]):
            item = _object(raw, {"sessions", "material_sha256"})
            batches.append(
                FrozenBatch(tuple(_key(key) for key in _array(item["sessions"])), _hash(item["material_sha256"]))
            )
        if not batches:
            raise _error()
        return ReplayManifest(
            _string(root["project_id"]),
            _number(root["material_budget"]),
            _number(root["max_output_tokens"]),
            tuple(sources),
            references,
            files("baseline"),
            tuple(initial),
            tuple(batches),
            render_version,
            prompt_version,
            tuning_version,
        )
    except (OSError, ValueError, TypeError) as error:
        raise _error() from error


def verify_frozen_scan(manifest: ReplayManifest, scanned: ScanResult) -> ScanResult:
    """Keep the read-time digest and frozen ordering revision checks shared."""
    frozen = {source.key: source for source in manifest.sources}
    if scanned.fatal or conflicting_sessions(scanned.sessions):
        raise _error()
    if {source.key for source in scanned.sessions} != set(frozen):
        raise ExtractionError("invalid_replay", "Frozen sources no longer reproduce the accepted sessions.")
    if any(source.revision.sha256 != frozen[source.key].sha256 for source in scanned.sessions):
        raise ExtractionError("invalid_replay", "A frozen source changed while its snapshot was read.")
    return replace(
        scanned,
        sessions=tuple(
            replace(source, revision=SourceRevision(frozen[source.key].updated_ns, source.revision.sha256))
            for source in scanned.sessions
        ),
    )


async def load_frozen_sources(manifest: ReplayManifest, project: MemoryScope) -> ScanResult:
    """Read a closed frozen session set without scanning or probing a live host."""
    from chrys.app.memory_hosts.claude_code.history import ClaudeHistoryReader
    from chrys.app.memory_hosts.codex_history import CodexHistoryReader

    await asyncio.to_thread(manifest.verify_source_files)
    sessions = frozenset(source.key.session for source in manifest.sources)
    claude = ClaudeHistoryReader(Path("."), sessions=sessions, replay=True)
    codex = CodexHistoryReader(Path("."), sessions=sessions, replay=True)
    scans = []
    for host, reader in (("claude-code", claude), ("codex", codex)):
        selected = tuple(source.path for source in manifest.sources if source.key.host == host)
        if selected:
            scans.append(await reader.read_files(project, selected))
    return verify_frozen_scan(
        manifest,
        ScanResult(
            tuple(source for scan in scans for source in scan.sessions),
            tuple(note for scan in scans for note in scan.diagnostics),
            any(scan.fatal for scan in scans),
        ),
    )
