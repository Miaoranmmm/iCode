# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The file backend: one Markdown file per entry under the user's config directory.

Layout::

    <root>/global/<id>.md
    <root>/projects/<project id>/<id>.md
    <root>/projects/<project id>/project.json   # where the project was, for display

Each scope directory has a ``.lock`` that every writer holds for the whole
check-and-write; reads take no lock because files are replaced atomically. A
version is the first 16 hex digits of the file's SHA-256, so a hand edit
changes it. A file that does not parse is quarantined: listed with its reason
and deletable, never repaired, overwritten or shown as an entry.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import stat
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import yaml

from chrys.foundation.platform.files import atomic_write_owner_only_bytes
from chrys.foundation.util.lock import FileLock
from chrys.service.memory.storage.layout import scope_directory
from chrys.service.memory.storage.store import (
    ENTRY_ID_RE,
    GLOBAL_SCOPE,
    MAX_ENTRIES_PER_SCOPE,
    MAX_ENTRY_BYTES,
    MAX_SEARCH_TERMS,
    MEMORY_TYPES,
    PROJECT_ID_RE,
    QUOTE_VIAS,
    SCHEMA_VERSION,
    SCOPE_NAMES,
    SEARCH_PAGE_SIZE,
    WRITERS,
    CatalogItem,
    Context,
    CreateResult,
    Evidence,
    FileEvidence,
    MemoryEntry,
    MemoryEntryDraft,
    MemoryInputRejected,
    MemoryOperationError,
    MemoryScope,
    MergeResult,
    ProjectInfo,
    Quote,
    ReferenceEvidence,
    ScopeName,
    StoredMemory,
    ToolResultEvidence,
    UnreadableMemory,
    UserReportEvidence,
    draft_problems,
    format_ref,
    is_zoned_timestamp,
    parse_ref,
    payload_key,
    utc_text,
    validate_draft,
)

logger = logging.getLogger(__name__)

LOCK_TIMEOUT_SECONDS: Final = 5.0
_ENTRY_SUFFIX: Final = ".md"
_PROJECT_FILE: Final = "project.json"
_FIELDS: Final = (
    "schema",
    "id",
    "type",
    "scope",
    "title",
    "applies_when",
    "created",
    "updated",
    "written_by",
    "quotes",
    "context",
    "evidence",
    "status",
    "status_basis",
    "merged_from",
)


class _Unreadable(Exception):
    """Why one file is quarantined; the text is shown to the user as is."""


class FileMemoryStore:
    """``MemoryStore`` over Markdown files under *root* (normally ``<config dir>/memory``)."""

    def __init__(self, root: Path) -> None:
        self._root = root

    # -- reads ---------------------------------------------------------------

    async def list_entries(self, scope: MemoryScope) -> list[StoredMemory]:
        valid, _ = await _run(lambda: _scan(self._scope_dir(scope), scope.scope))
        return _newest_first(valid)

    async def get(self, scope: MemoryScope, ref: str) -> StoredMemory:
        name = _ref_name(scope, ref)
        return await _run(lambda: _read_stored(self._scope_dir(scope) / f"{name}{_ENTRY_SUFFIX}", scope.scope, ref))

    async def search(
        self, scope: MemoryScope, query: str, cursor: str | None = None, *, with_global: bool = False
    ) -> tuple[list[StoredMemory], str | None]:
        offset = _cursor_offset(cursor)
        scopes = [scope, GLOBAL_SCOPE] if with_global and scope.scope == "project" else [scope]

        def scan_all() -> list[StoredMemory]:
            found: list[StoredMemory] = []
            for each in scopes:
                valid, _ = _scan(self._scope_dir(each), each.scope)
                if len(valid) > MAX_ENTRIES_PER_SCOPE:
                    raise MemoryOperationError(
                        "memory_full",
                        f"The {each.scope} memory holds more than {MAX_ENTRIES_PER_SCOPE} entries, "
                        "too many to search; delete some first.",
                    )
                found.extend(stored for stored in valid if stored.entry.status is None)
            return found

        ranked = _rank(await _run(scan_all), query)
        page = ranked[offset : offset + SEARCH_PAGE_SIZE]
        following = offset + SEARCH_PAGE_SIZE
        return page, (str(following) if following < len(ranked) else None)

    async def catalog(self, scope: MemoryScope) -> list[CatalogItem]:
        return [
            CatalogItem(
                ref=stored.entry.ref,
                type=stored.entry.type,
                title=stored.entry.title,
                applies_when=stored.entry.applies_when,
                updated=stored.entry.updated,
            )
            for stored in await self.list_entries(scope)
            if stored.entry.status is None
        ]

    async def unreadable(self, scope: MemoryScope) -> list[UnreadableMemory]:
        _, invalid = await _run(lambda: _scan(self._scope_dir(scope), scope.scope))
        return sorted(invalid, key=lambda item: item.ref)

    async def location(self, scope: MemoryScope) -> str | None:
        return str(self._scope_dir(scope))

    async def list_projects(self) -> list[ProjectInfo]:
        return await _run(self._list_projects)

    # -- writes --------------------------------------------------------------

    async def create(self, scope: MemoryScope, draft: MemoryEntryDraft) -> CreateResult:
        validate_draft(draft, scope=scope.scope)
        return await _run(lambda: self._create(scope, draft))

    async def update(self, scope: MemoryScope, ref: str, version: str, draft: MemoryEntryDraft) -> str:
        name = _ref_name(scope, ref)
        validate_draft(draft, scope=scope.scope)
        return await _run(lambda: self._update(scope, name, ref, version, draft))

    async def delete(self, scope: MemoryScope, ref: str, version: str) -> None:
        name = _ref_name(scope, ref)
        await _run(lambda: self._delete(scope, name, ref, version))

    async def merge(
        self,
        scope: MemoryScope,
        keep_ref: str,
        keep_version: str,
        drop_ref: str,
        drop_version: str,
        keep_draft: MemoryEntryDraft,
    ) -> MergeResult:
        keep_name = _ref_name(scope, keep_ref)
        drop_name = _ref_name(scope, drop_ref)
        if keep_name == drop_name:
            raise MemoryInputRejected("memory_invalid_input", "An entry cannot be merged with itself.")
        validate_draft(keep_draft, scope=scope.scope)
        return await _run(
            lambda: self._merge(scope, keep_name, keep_ref, keep_version, drop_name, drop_ref, drop_version, keep_draft)
        )

    # -- transactions (run in a worker thread, under the scope's lock) ------

    def _create(self, scope: MemoryScope, draft: MemoryEntryDraft) -> CreateResult:
        directory = self._scope_dir(scope)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with FileLock(directory / ".lock", timeout=LOCK_TIMEOUT_SECONDS):
            valid, _ = _scan(directory, scope.scope)
            key = payload_key(scope.scope, draft)
            for stored in valid:
                if payload_key(scope.scope, stored.entry.draft()) == key:
                    return CreateResult(
                        ref=stored.entry.ref, version=stored.version, created=False, status=stored.entry.status
                    )
            if len(valid) >= MAX_ENTRIES_PER_SCOPE:
                raise MemoryOperationError(
                    "memory_full",
                    f"The {scope.scope} memory is full ({MAX_ENTRIES_PER_SCOPE} entries); delete some first.",
                )
            entry_id = _unused_id(directory)
            now = _now()
            entry = MemoryEntry(id=entry_id, scope=scope.scope, created=now, updated=now, **_draft_fields(draft))
            data = _encode(entry)
            atomic_write_owner_only_bytes(directory / f"{entry_id}{_ENTRY_SUFFIX}", data)
            self._describe_project(scope, directory)
            return CreateResult(ref=entry.ref, version=_version(data), created=True, status=entry.status)

    def _update(self, scope: MemoryScope, name: str, ref: str, version: str, draft: MemoryEntryDraft) -> str:
        directory = self._scope_dir(scope)
        path = directory / f"{name}{_ENTRY_SUFFIX}"
        _require_directory(directory, ref)
        with FileLock(directory / ".lock", timeout=LOCK_TIMEOUT_SECONDS):
            current = _read_stored(path, scope.scope, ref)
            _check_version(ref, current.version, version)
            if draft.type != current.entry.type:
                raise MemoryInputRejected("memory_invalid_input", "An entry's type cannot change.")
            entry = replace(current.entry, updated=_now(), **_draft_fields(draft))
            data = _encode(entry)
            atomic_write_owner_only_bytes(path, data, create_parents=False)
            self._describe_project(scope, directory)
            return _version(data)

    def _delete(self, scope: MemoryScope, name: str, ref: str, version: str) -> None:
        directory = self._scope_dir(scope)
        path = directory / f"{name}{_ENTRY_SUFFIX}"
        _require_directory(directory, ref)
        with FileLock(directory / ".lock", timeout=LOCK_TIMEOUT_SECONDS):
            # Quarantined files are deleted the same way, so only the bytes are read here.
            data = _read_bytes(path, ref)
            _check_version(ref, _version(data), version)
            path.unlink()

    def _merge(
        self,
        scope: MemoryScope,
        keep_name: str,
        keep_ref: str,
        keep_version: str,
        drop_name: str,
        drop_ref: str,
        drop_version: str,
        keep_draft: MemoryEntryDraft,
    ) -> MergeResult:
        if keep_name == drop_name:
            raise MemoryInputRejected("memory_invalid_input", "An entry cannot be merged with itself.")
        directory = self._scope_dir(scope)
        keep_path = directory / f"{keep_name}{_ENTRY_SUFFIX}"
        drop_path = directory / f"{drop_name}{_ENTRY_SUFFIX}"
        _require_directory(directory, keep_ref)
        with FileLock(directory / ".lock", timeout=LOCK_TIMEOUT_SECONDS):
            kept = _read_stored(keep_path, scope.scope, keep_ref)
            dropped = _read_stored(drop_path, scope.scope, drop_ref)
            _check_version(keep_ref, kept.version, keep_version)
            _check_version(drop_ref, dropped.version, drop_version)
            if kept.entry.type != dropped.entry.type or keep_draft.type != kept.entry.type:
                raise MemoryInputRejected("memory_invalid_input", "Merged entries must have the same type.")
            if kept.entry.status is not None or dropped.entry.status is not None:
                raise MemoryInputRejected("memory_invalid_input", "Withdrawn entries cannot be merged.")
            keep_draft = replace(keep_draft, written_by=kept.entry.written_by)
            entry = replace(kept.entry, updated=_now(), **_draft_fields(keep_draft))
            data = _encode(entry)
            atomic_write_owner_only_bytes(keep_path, data, create_parents=False)
            self._describe_project(scope, directory)
            version = _version(data)
            try:
                # A manual edit does not honor the advisory scope lock. Recheck
                # the drop's bytes after publishing keep before deleting it.
                _check_version(drop_ref, _version(_read_bytes(drop_path, drop_ref)), drop_version)
                drop_path.unlink()
            except FileNotFoundError:
                return MergeResult(version, deleted=True)
            except MemoryOperationError as error:
                return MergeResult(version, deleted=error.kind == "memory_not_found")
            except OSError:
                return MergeResult(version, deleted=False)
            return MergeResult(version, deleted=True)

    def _describe_project(self, scope: MemoryScope, directory: Path) -> None:
        """Record where the project lives the first time it gets an entry; an existing record is kept."""
        if scope.scope != "project" or scope.root is None:
            return
        target = directory / _PROJECT_FILE
        if target.exists():
            return
        record = {"root": scope.root, "anchor": scope.anchor}
        try:
            atomic_write_owner_only_bytes(target, (json.dumps(record, indent=2) + "\n").encode())
        except OSError:
            # The record only labels the project in listings; the entry just written stands without it.
            logger.warning("Could not record the memory project's location", exc_info=True)

    # -- layout --------------------------------------------------------------

    def _scope_dir(self, scope: MemoryScope) -> Path:
        return scope_directory(self._root, scope)

    def _list_projects(self) -> list[ProjectInfo]:
        projects_dir = self._root / "projects"
        if not projects_dir.is_dir():
            return []
        return [
            ProjectInfo(project_id=child.name, root=_project_root(child / _PROJECT_FILE))
            for child in sorted(projects_dir.iterdir())
            if child.is_dir() and PROJECT_ID_RE.fullmatch(child.name)
        ]


# ---------------------------------------------------------------------------
# Thread hop and error mapping
# ---------------------------------------------------------------------------


async def _run[T](work: Callable[[], T]) -> T:
    """Run a file transaction off the event loop; low-level failures become ``memory_store_unavailable``."""
    try:
        return await asyncio.to_thread(work)
    except MemoryOperationError:
        raise
    except TimeoutError as exc:  # before OSError: it is one
        raise MemoryOperationError(
            "memory_store_unavailable", "Another process is writing to the memory store; try again."
        ) from exc
    except OSError as exc:
        raise MemoryOperationError(
            "memory_store_unavailable", "The memory store could not be read or written."
        ) from exc


# ---------------------------------------------------------------------------
# Scanning and reading
# ---------------------------------------------------------------------------


def _scan(directory: Path, scope: ScopeName) -> tuple[list[StoredMemory], list[UnreadableMemory]]:
    if not directory.is_dir():
        return [], []
    valid: list[StoredMemory] = []
    invalid: list[UnreadableMemory] = []
    for path in sorted(directory.iterdir()):
        # Dot files are locks and in-flight atomic writes, never entries.
        if path.name.startswith(".") or not path.name.endswith(_ENTRY_SUFFIX) or not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            continue  # deleted while scanning
        name = path.name.removesuffix(_ENTRY_SUFFIX)
        try:
            entry = parse_entry(data, scope=scope, name=name)
        except _Unreadable as exc:
            invalid.append(UnreadableMemory(ref=format_ref(scope, name), version=_version(data), reason=str(exc)))
            continue
        valid.append(StoredMemory(entry=entry, version=_version(data)))
    return valid, invalid


def _require_directory(directory: Path, ref: str) -> None:
    try:
        info = directory.stat()
    except FileNotFoundError:
        raise MemoryInputRejected("memory_not_found", f"No memory entry {ref}.") from None
    if not stat.S_ISDIR(info.st_mode):
        raise MemoryOperationError("memory_store_unavailable", "The memory store could not be read or written.")


def _read_bytes(path: Path, ref: str) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise MemoryInputRejected("memory_not_found", f"No memory entry {ref}.") from exc


def _read_stored(path: Path, scope: ScopeName, ref: str) -> StoredMemory:
    data = _read_bytes(path, ref)
    try:
        entry = parse_entry(data, scope=scope, name=path.name.removesuffix(_ENTRY_SUFFIX))
    except _Unreadable as exc:
        raise MemoryOperationError("memory_entry_invalid", f"{ref} cannot be read: {exc}.") from exc
    return StoredMemory(entry=entry, version=_version(data))


def _project_root(path: Path) -> str | None:
    """The root recorded in ``project.json``; a missing or damaged record only loses the display."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    root = record.get("root") if isinstance(record, dict) else None
    return root if isinstance(root, str) else None


def _ref_name(scope: MemoryScope, ref: str) -> str:
    """The file stem *ref* names in *scope*; it must stay inside the scope's directory."""
    ref_scope, name = parse_ref(ref)
    if ref_scope != scope.scope:
        raise MemoryInputRejected("memory_invalid_input", f"{ref} is not a {scope.scope} entry.")
    if name.startswith(".") or any(char in name for char in "/\\\0") or name != name.strip():
        raise MemoryInputRejected("memory_invalid_input", f"{ref!r} is not a memory ref.")
    return name


def _check_version(ref: str, current: str, expected: str) -> None:
    if current != expected:
        raise MemoryOperationError(
            "memory_version_conflict",
            f"{ref} changed since it was read (now version {current}); read it again before changing it.",
        )


def _cursor_offset(cursor: str | None) -> int:
    if cursor is None:
        return 0
    try:
        # isdigit() alone also accepts digits such as "²" that int() rejects.
        if cursor.isascii() and cursor.isdigit():
            return int(cursor)
    except ValueError:
        pass  # past the interpreter's integer-string digit limit
    raise MemoryInputRejected("memory_invalid_input", "This is not a search cursor returned by memory_search.")


def _version(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _unused_id(directory: Path) -> str:
    # A quarantined file keeps its name, so a new id must avoid every existing file, valid or not.
    while True:
        candidate = secrets.token_hex(6)
        if not (directory / f"{candidate}{_ENTRY_SUFFIX}").exists():
            return candidate


def _now() -> str:
    return utc_text(datetime.now(UTC).replace(microsecond=0))


def _draft_fields(draft: MemoryEntryDraft) -> dict[str, Any]:
    return {
        "type": draft.type,
        "title": draft.title,
        "body": draft.body,
        "written_by": draft.written_by,
        "applies_when": draft.applies_when,
        "quotes": draft.quotes,
        "context": draft.context,
        "evidence": draft.evidence,
        "status": draft.status,
        "status_basis": draft.status_basis,
        "merged_from": draft.merged_from,
    }


# ---------------------------------------------------------------------------
# Ordering and search
# ---------------------------------------------------------------------------


def _newest_first(stored: Iterable[StoredMemory]) -> list[StoredMemory]:
    by_ref = sorted(stored, key=lambda item: (item.entry.id, item.entry.scope))
    # By time, not by text: "…:00Z" and "…:00.5Z" sort the wrong way round as strings.
    return sorted(by_ref, key=lambda item: datetime.fromisoformat(item.entry.updated), reverse=True)


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _search_text(entry: MemoryEntry) -> str:
    parts = [entry.title, entry.applies_when or "", entry.body, *(quote.text for quote in entry.quotes)]
    if entry.context is not None:
        parts.append(entry.context.text)
    for item in entry.evidence:
        if isinstance(item, ToolResultEvidence | UserReportEvidence):
            parts.append(item.excerpt)
        elif isinstance(item, FileEvidence):
            parts.append(item.path)
        elif isinstance(item, ReferenceEvidence):
            parts.append(item.ref)
    return _normalize("\n".join(parts))


def _rank(stored: list[StoredMemory], query: str) -> list[StoredMemory]:
    """Matches for *query*: entries with every word first, then by words matched, then newest first."""
    ordered = _newest_first(stored)
    terms = _normalize(query).split()[:MAX_SEARCH_TERMS]
    if not terms:
        return ordered
    scored = []
    for item in ordered:
        text = _search_text(item.entry)
        hits = sum(term in text for term in terms)
        if hits:
            scored.append((hits, item))
    # sorted is stable, so entries with equal scores stay newest first.
    scored.sort(key=lambda pair: (pair[0] == len(terms), pair[0]), reverse=True)
    return [item for _, item in scored]


# ---------------------------------------------------------------------------
# The file format: YAML frontmatter, then the body
# ---------------------------------------------------------------------------


class _EntryDumper(yaml.SafeDumper):
    """Writes multi-line text as ``|`` blocks; PyYAML falls back to quoting when a block would not round-trip."""


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|" if "\n" in data else None)


_EntryDumper.add_representer(str, _represent_str)


def serialize_entry(entry: MemoryEntry) -> str:
    front = {
        "schema": SCHEMA_VERSION,
        "id": entry.id,
        "type": entry.type,
        "scope": entry.scope,
        "title": entry.title,
        "applies_when": entry.applies_when,
        "created": entry.created,
        "updated": entry.updated,
        "written_by": entry.written_by,
        "quotes": [_quote_record(quote) for quote in entry.quotes],
        "context": None if entry.context is None else _context_record(entry.context),
        "evidence": [_evidence_record(item) for item in entry.evidence],
    }
    if entry.status is not None:
        front["status"] = entry.status
    if entry.status_basis is not None:
        front["status_basis"] = _quote_record(entry.status_basis)
    if entry.merged_from:
        front["merged_from"] = list(entry.merged_from)
    header = yaml.dump(front, Dumper=_EntryDumper, allow_unicode=True, sort_keys=False, width=4096)
    return f"---\n{header}---\n{entry.body}\n"


def _encode(entry: MemoryEntry) -> bytes:
    try:
        data = serialize_entry(entry).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise MemoryInputRejected(
            "memory_invalid_input", "The entry contains characters that cannot be saved."
        ) from exc
    if len(data) > MAX_ENTRY_BYTES:
        raise MemoryInputRejected(
            "memory_invalid_input", f"The entry is over {MAX_ENTRY_BYTES // 1024} KiB once saved; shorten it."
        )
    return data


def _optional(record: dict[str, Any], **fields: Any) -> dict[str, Any]:
    record.update({key: value for key, value in fields.items() if value is not None})
    return record


def _quote_record(quote: Quote) -> dict[str, Any]:
    return _optional({"text": quote.text, "via": quote.via, "session": quote.session}, at=quote.at, item=quote.item)


def _context_record(context: Context) -> dict[str, Any]:
    return _optional({"text": context.text, "session": context.session}, at=context.at, item=context.item)


def _evidence_record(item: Evidence) -> dict[str, Any]:
    if isinstance(item, ToolResultEvidence):
        return _optional(
            {"kind": item.kind, "tool": item.tool, "args": item.args, "excerpt": item.excerpt, "session": item.session},
            at=item.at,
            item=item.item,
        )
    if isinstance(item, FileEvidence):
        return {
            "kind": item.kind,
            "project": item.project,
            "path": item.path,
            "sha256": item.sha256,
            "bytes": item.bytes,
        }
    if isinstance(item, ReferenceEvidence):
        return _optional({"kind": item.kind, "ref": item.ref}, note=item.note)
    return _optional({"kind": item.kind, "excerpt": item.excerpt, "session": item.session}, at=item.at, item=item.item)


def parse_entry(data: bytes, *, scope: ScopeName, name: str) -> MemoryEntry:
    """Parse one entry file; raises ``_Unreadable`` with the reason it is quarantined."""
    if len(data) > MAX_ENTRY_BYTES:
        raise _Unreadable(f"the file is over {MAX_ENTRY_BYTES // 1024} KiB")
    if not ENTRY_ID_RE.fullmatch(name):
        raise _Unreadable("the file name is not a memory id")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _Unreadable("the file is not UTF-8 text") from exc
    text = text.removeprefix("\ufeff")
    # A file saved with Windows line endings is read as one with Unix ones; a file this backend wrote has
    # "\n" line ends, so a "\r" in its body is the body's own and stays.
    if text.startswith("---\r\n"):
        text = text.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        raise _Unreadable("the file does not start with a --- frontmatter line")
    end = text.find("\n---\n", 3)
    if end == -1:
        if not text.endswith("\n---"):
            raise _Unreadable("the frontmatter has no closing --- line")
        end = len(text) - 4
    try:
        front = yaml.safe_load(text[4 : end + 1])
    except (yaml.YAMLError, ValueError) as exc:
        # PyYAML raises ValueError for a timestamp-shaped value that is no real date (month 13).
        raise _Unreadable("the frontmatter is not valid YAML") from exc
    # The writer ends the file with one newline after the body; anything before it is the body's own.
    body = text[end + 5 :].removesuffix("\n")
    if not isinstance(front, dict):
        raise _Unreadable("the frontmatter is not a set of fields")
    entry = _entry_from_front(front, body, scope=scope, name=name)
    problems = draft_problems(entry.draft(), scope=scope)
    if problems:
        raise _Unreadable("; ".join(problems))
    return entry


def _entry_from_front(front: dict[Any, Any], body: str, *, scope: ScopeName, name: str) -> MemoryEntry:
    unknown = sorted(str(key) for key in front if key not in _FIELDS)
    if unknown:
        raise _Unreadable(f"unknown field {unknown[0]!r}")
    schema = front.get("schema")
    if isinstance(schema, int) and not isinstance(schema, bool) and schema > SCHEMA_VERSION:
        raise _Unreadable("it was written by a newer version of iCode")
    if schema != SCHEMA_VERSION or isinstance(schema, bool):
        raise _Unreadable(f"schema must be {SCHEMA_VERSION}")
    entry_id = front.get("id")
    if entry_id != name:
        raise _Unreadable("its id does not match the file name")
    stored_scope = front.get("scope")
    if stored_scope not in SCOPE_NAMES or stored_scope != scope:
        raise _Unreadable(f"its scope is not {scope}")
    entry_type = _choice(front, "type", MEMORY_TYPES)
    written_by = _choice(front, "written_by", WRITERS)
    quotes = _list(front, "quotes")
    evidence = _list(front, "evidence")
    context = front.get("context")
    status_basis = front.get("status_basis")
    if context is not None and not isinstance(context, dict):
        raise _Unreadable("context must be a set of fields or null")
    return MemoryEntry(
        id=name,
        scope=scope,
        created=_timestamp(front, "created"),
        updated=_timestamp(front, "updated"),
        type=entry_type,
        title=_text(front, "title"),
        body=body,
        written_by=written_by,
        applies_when=_text(front, "applies_when", optional=True),
        quotes=tuple(_quote(record, index) for index, record in enumerate(quotes)),
        context=None if context is None else _context(context),
        evidence=tuple(_evidence(record, index) for index, record in enumerate(evidence)),
        status=_text(front, "status", optional=True),
        status_basis=None if status_basis is None else _source_quote(status_basis, "status_basis"),
        merged_from=tuple(_list(front, "merged_from")),
    )


def _fields(record: Any, name: str, allowed: set[str], required: set[str]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise _Unreadable(f"{name} must be a set of fields")
    unknown = sorted(str(key) for key in record if key not in allowed)
    if unknown:
        raise _Unreadable(f"{name} has an unknown field {unknown[0]!r}")
    missing = sorted(required - record.keys())
    if missing:
        raise _Unreadable(f"{name} is missing {missing[0]!r}")
    return record


def _quote(record: Any, index: int) -> Quote:
    return _source_quote(record, f"quotes[{index}]")


def _source_quote(record: Any, name: str) -> Quote:
    fields = _fields(record, name, {"text", "via", "session", "at", "item"}, {"text", "via"})
    via = fields["via"]
    if via not in QUOTE_VIAS:
        raise _Unreadable(f"{name}.via must be one of {', '.join(QUOTE_VIAS)}")
    return Quote(
        text=_text(fields, "text", name=name),
        via=via,
        session=_text(fields, "session", optional=True, name=name),
        at=_optional_timestamp(fields, "at", name),
        item=_text(fields, "item", optional=True, name=name),
    )


def _context(record: dict[Any, Any]) -> Context:
    fields = _fields(record, "context", {"text", "session", "at", "item"}, {"text"})
    return Context(
        text=_text(fields, "text", name="context"),
        session=_text(fields, "session", optional=True, name="context"),
        at=_optional_timestamp(fields, "at", "context"),
        item=_text(fields, "item", optional=True, name="context"),
    )


def _evidence(record: Any, index: int) -> Evidence:
    name = f"evidence[{index}]"
    if not isinstance(record, dict):
        raise _Unreadable(f"{name} must be a set of fields")
    kind = record.get("kind")
    if kind == "tool_result":
        fields = _fields(
            record, name, {"kind", "tool", "args", "excerpt", "session", "at", "item"}, {"tool", "args", "excerpt"}
        )
        return ToolResultEvidence(
            tool=_text(fields, "tool", name=name),
            args=_text(fields, "args", name=name),
            excerpt=_text(fields, "excerpt", name=name),
            session=_text(fields, "session", optional=True, name=name),
            at=_optional_timestamp(fields, "at", name),
            item=_text(fields, "item", optional=True, name=name),
        )
    if kind == "file":
        fields = _fields(record, name, {"kind", "project", "path", "sha256", "bytes"}, {"project", "path", "bytes"})
        size = fields["bytes"]
        if not isinstance(size, int) or isinstance(size, bool):
            raise _Unreadable(f"{name}.bytes must be a number")
        return FileEvidence(
            project=_text(fields, "project", name=name),
            path=_text(fields, "path", name=name),
            sha256=_text(fields, "sha256", optional=True, name=name),
            bytes=size,
        )
    if kind == "reference":
        fields = _fields(record, name, {"kind", "ref", "note"}, {"ref"})
        return ReferenceEvidence(
            ref=_text(fields, "ref", name=name), note=_text(fields, "note", optional=True, name=name)
        )
    if kind == "user_report":
        fields = _fields(record, name, {"kind", "excerpt", "session", "at", "item"}, {"excerpt"})
        return UserReportEvidence(
            excerpt=_text(fields, "excerpt", name=name),
            session=_text(fields, "session", optional=True, name=name),
            at=_optional_timestamp(fields, "at", name),
            item=_text(fields, "item", optional=True, name=name),
        )
    raise _Unreadable(f"{name}.kind must be tool_result, file, reference or user_report")


def _list(front: dict[Any, Any], key: str) -> list[Any]:
    value = front.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise _Unreadable(f"{key} must be a list")
    return value


def _choice(front: dict[Any, Any], key: str, allowed: tuple[str, ...]) -> Any:
    value = front.get(key)
    if value not in allowed:
        raise _Unreadable(f"{key} must be one of {', '.join(allowed)}")
    return value


def _text(record: dict[Any, Any], key: str, *, optional: bool = False, name: str = "") -> Any:
    label = f"{name}.{key}" if name else key
    if key not in record and not optional:
        raise _Unreadable(f"{label} is missing")
    value = record.get(key)
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise _Unreadable(f"{label} must be text")
    return value


def _timestamp(record: dict[Any, Any], key: str, name: str = "") -> str:
    value = _optional_timestamp(record, key, name)
    if value is None:
        raise _Unreadable(f"{key} is missing")
    return value


def _optional_timestamp(record: dict[Any, Any], key: str, name: str) -> str | None:
    """A timestamp with a zone, in UTC: ``...Z`` text as written, any other zone or form converted."""
    value = record.get(key)
    label = f"{name}.{key}" if name else key
    if value is None:
        return None
    if isinstance(value, str):
        if not is_zoned_timestamp(value):
            raise _Unreadable(f"{label} must be a timestamp with a time zone")
        # This backend writes UTC as "...Z"; such a value is kept exactly, fractions of a second included.
        return value if value.endswith("Z") else utc_text(datetime.fromisoformat(value))
    # YAML turns an unquoted timestamp into a datetime; one with no zone is no moment to order by.
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise _Unreadable(f"{label} must be a timestamp with a time zone")
    return utc_text(value)
