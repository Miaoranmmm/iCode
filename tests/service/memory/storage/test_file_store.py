# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ruff: noqa: RUF001  # the entries are written in Chinese, full-width punctuation included

"""The file backend keeps the storage contract: versions, dedupe, limits, quarantine, search and project records."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from chrys.foundation.util.lock import FileLock
from chrys.service.memory.storage import file_store
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import (
    GLOBAL_SCOPE,
    Context,
    FileEvidence,
    MemoryEntryDraft,
    MemoryOperationError,
    MemoryScope,
    Quote,
    ReferenceEvidence,
    ToolResultEvidence,
)

PROJECT = MemoryScope("project", "5e1f0c9a7b2d4e61", root="/work/shop-installer", anchor="git")
OTHER_PROJECT = MemoryScope("project", "0123456789abcdef", root="/work/other", anchor="dir")

# Sample ② of the design, as a user would have it on disk: unquoted timestamps.
SAMPLE = """---
schema: 1
id: 3f6c2a9e41b0
type: intent
scope: project
title: 安装器必须支持完全离线部署
applies_when: 安装器、打包与依赖获取流程
created: 2026-11-02T08:05:40Z
updated: 2026-11-02T08:05:40Z
written_by: model
quotes:
  - text: 安装器必须支持离线部署，客户机房没有外网，这个是硬要求。
    via: message
    session: a1b2c3d4
    at: 2026-11-02T08:05:31Z
context: null
evidence: []
---
安装过程不能访问任何网络（包括包索引和 git 远端）。改动安装器、锁文件或依赖来源时，要验证断网下能装成功。
"""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "memory"


@pytest.fixture
def store(root: Path) -> FileMemoryStore:
    return FileMemoryStore(root)


def intent(title: str = "回答用中文", body: str = "回答默认用中文。", **changes: object) -> MemoryEntryDraft:
    fields: dict[str, object] = {
        "type": "intent",
        "title": title,
        "body": body,
        "written_by": "model",
        "quotes": (Quote(text="以后都用中文回答", via="message", session="a1b2c3d4"),),
    }
    fields.update(changes)
    return MemoryEntryDraft(**fields)  # type: ignore[arg-type]


def project_dir(root: Path, scope: MemoryScope = PROJECT) -> Path:
    return root / "projects" / str(scope.project_id)


def entry_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*.md") if not path.name.startswith("."))


async def expect_error(kind: str, call) -> MemoryOperationError:
    with pytest.raises(MemoryOperationError) as caught:
        await call
    assert caught.value.kind == kind
    return caught.value


# ---------------------------------------------------------------------------
# Create, read, dedupe
# ---------------------------------------------------------------------------


async def test_create_writes_one_markdown_file_named_by_its_id(store: FileMemoryStore, root: Path) -> None:
    result = await store.create(PROJECT, intent())

    assert result.created
    scope, _, entry_id = result.ref.partition(":")
    assert scope == "project"
    assert len(entry_id) == 12 and all(char in "0123456789abcdef" for char in entry_id)
    [path] = entry_files(project_dir(root))
    assert path.name == f"{entry_id}.md"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\nschema: 1\n")
    assert text.endswith("---\n回答默认用中文。\n")
    stored = await store.get(PROJECT, result.ref)
    assert stored.version == result.version
    assert stored.entry.created == stored.entry.updated
    assert stored.entry.created.endswith("Z")


async def test_identical_content_returns_the_stored_entry(store: FileMemoryStore, root: Path) -> None:
    first = await store.create(PROJECT, intent())
    again = await store.create(PROJECT, intent())
    different = await store.create(PROJECT, intent(body="回答默认用中文，简洁。"))

    assert again == type(again)(ref=first.ref, version=first.version, created=False)
    assert different.created and different.ref != first.ref
    assert len(entry_files(project_dir(root))) == 2


async def test_the_same_content_in_another_scope_is_a_separate_entry(store: FileMemoryStore) -> None:
    in_project = await store.create(PROJECT, intent())
    in_global = await store.create(GLOBAL_SCOPE, intent())
    elsewhere = await store.create(OTHER_PROJECT, intent())

    assert in_global.created and in_global.ref.startswith("global:")
    assert elsewhere.created
    assert [s.entry.ref for s in await store.list_entries(PROJECT)] == [in_project.ref]
    assert [s.entry.ref for s in await store.list_entries(OTHER_PROJECT)] == [elsewhere.ref]


async def test_a_hand_written_file_reads_and_its_version_follows_its_bytes(store: FileMemoryStore, root: Path) -> None:
    path = project_dir(root) / "3f6c2a9e41b0.md"
    path.parent.mkdir(parents=True)
    path.write_text(SAMPLE, encoding="utf-8")

    stored = await store.get(PROJECT, "project:3f6c2a9e41b0")
    assert stored.entry.created == "2026-11-02T08:05:40Z"
    assert stored.entry.quotes[0].at == "2026-11-02T08:05:31Z"
    assert stored.entry.body.startswith("安装过程不能访问任何网络")

    path.write_text(SAMPLE.replace("硬要求", "硬性要求"), encoding="utf-8")
    assert (await store.get(PROJECT, "project:3f6c2a9e41b0")).version != stored.version


async def test_every_source_kind_survives_a_round_trip(store: FileMemoryStore) -> None:
    draft = MemoryEntryDraft(
        type="engineering",
        title="集成测试必须 -n0 串行跑",
        body="fixture 固定绑定 127.0.0.1:8765。\n  如果改成端口 0，这条不再适用。  ",
        written_by="model",
        applies_when="运行 tests/integration 时",
        quotes=(Quote(text="多行\n  原话 ", via="message", session="s1", at="2026-11-02T11:31:02Z", item="01J9"),),
        context=Context(text="方案 B：串行", session="s1"),
        evidence=(
            ToolResultEvidence(tool="shell", args="uv run pytest -x", excerpt="OSError: [Errno 48]", session="s1"),
            FileEvidence(project=PROJECT.project_id or "", path="tests/conftest.py", sha256="a" * 64, bytes=2381),
            FileEvidence(project=PROJECT.project_id or "", path="big.bin", sha256=None, bytes=3_000_000),
            ReferenceEvidence(ref="https://docs.example/sync", note="只禁用网络"),
        ),
    )
    result = await store.create(PROJECT, draft)

    assert (await store.get(PROJECT, result.ref)).entry.draft() == draft


# ---------------------------------------------------------------------------
# Update and delete
# ---------------------------------------------------------------------------


async def test_update_keeps_identity_and_needs_the_current_version(store: FileMemoryStore) -> None:
    created = await store.create(PROJECT, intent())
    before = await store.get(PROJECT, created.ref)

    new_version = await store.update(PROJECT, created.ref, created.version, intent(title="回答用中文、简洁"))
    after = await store.get(PROJECT, created.ref)

    assert after.version == new_version != created.version
    assert after.entry.title == "回答用中文、简洁"
    assert (after.entry.id, after.entry.created) == (before.entry.id, before.entry.created)
    error = await expect_error(
        "memory_version_conflict", store.update(PROJECT, created.ref, created.version, intent(title="旧版本"))
    )
    assert new_version in error.message
    assert (await store.get(PROJECT, created.ref)).entry.title == "回答用中文、简洁"


async def test_update_cannot_change_the_type(store: FileMemoryStore) -> None:
    created = await store.create(PROJECT, intent())
    engineering = MemoryEntryDraft(
        type="engineering",
        title="t",
        body="b",
        written_by="model",
        evidence=(ReferenceEvidence(ref="doc"),),
    )
    await expect_error("memory_invalid_input", store.update(PROJECT, created.ref, created.version, engineering))


async def test_delete_needs_the_current_version(store: FileMemoryStore, root: Path) -> None:
    created = await store.create(PROJECT, intent())
    [path] = entry_files(project_dir(root))
    path.write_text(path.read_text(encoding="utf-8").replace("回答默认用中文。", "改过了。"), encoding="utf-8")

    await expect_error("memory_version_conflict", store.delete(PROJECT, created.ref, created.version))
    current = await store.get(PROJECT, created.ref)
    await store.delete(PROJECT, created.ref, current.version)

    assert entry_files(project_dir(root)) == []
    await expect_error("memory_not_found", store.get(PROJECT, created.ref))
    await expect_error("memory_not_found", store.delete(PROJECT, created.ref, current.version))


async def test_a_ref_from_another_scope_or_outside_the_directory_is_rejected(store: FileMemoryStore) -> None:
    created = await store.create(PROJECT, intent())

    await expect_error("memory_invalid_input", store.get(GLOBAL_SCOPE, created.ref))
    await expect_error("memory_invalid_input", store.get(PROJECT, "project:../global/x"))
    await expect_error("memory_invalid_input", store.get(PROJECT, "project:.lock"))
    await expect_error("memory_invalid_input", store.get(PROJECT, "3f6c2a9e41b0"))


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


async def test_a_full_scope_refuses_new_entries_without_archiving(store: FileMemoryStore, root: Path) -> None:
    seed = await store.create(PROJECT, intent())
    directory = project_dir(root)
    template = (directory / f"{seed.ref.partition(':')[2]}.md").read_text(encoding="utf-8")
    seed_id = seed.ref.partition(":")[2]
    for number in range(1, 500):
        entry_id = f"a{number:011x}"
        text = template.replace(seed_id, entry_id).replace("回答默认用中文。", f"条目 {number}")
        (directory / f"{entry_id}.md").write_text(text, encoding="utf-8")

    await expect_error("memory_full", store.create(PROJECT, intent(body="第 501 条")))
    assert len(entry_files(directory)) == 500
    # An exact duplicate is still answered: it adds nothing.
    assert not (await store.create(PROJECT, intent())).created


async def test_an_entry_too_large_to_store_is_refused(store: FileMemoryStore, root: Path) -> None:
    excerpt = "错" * 600
    draft = MemoryEntryDraft(
        type="engineering",
        title="大条目",
        body="正" * 1500,
        written_by="model",
        context=Context(text="提" * 600, session="s"),
        evidence=tuple(ToolResultEvidence(tool="shell", args="x", excerpt=excerpt, session="s") for _ in range(6)),
    )
    await expect_error("memory_invalid_input", store.create(PROJECT, draft))
    assert not project_dir(root).exists() or entry_files(project_dir(root)) == []


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"title": "t" * 81}, "title"),
        ({"title": "two\nlines"}, "title"),
        ({"body": ""}, "body"),
        ({"applies_when": "a" * 201}, "applies_when"),
        ({"quotes": ()}, "quotes"),
        ({"quotes": (Quote(text="x" * 501, via="message", session="s"),)}, "quotes[0].text"),
        ({"quotes": (Quote(text="no session", via="message"),)}, "session"),
        ({"quotes": tuple(Quote(text=f"q{i}", via="message", session="s") for i in range(4))}, "quotes"),
    ],
)
async def test_drafts_that_break_the_schema_are_refused(
    store: FileMemoryStore, changes: dict[str, object], reason: str
) -> None:
    error = await expect_error("memory_invalid_input", store.create(PROJECT, intent(**changes)))
    assert reason in error.message


async def test_engineering_needs_evidence(store: FileMemoryStore) -> None:
    draft = MemoryEntryDraft(type="engineering", title="t", body="b", written_by="model")
    await expect_error("memory_invalid_input", store.create(PROJECT, draft))


async def test_a_typed_quote_needs_no_session_and_may_be_long(store: FileMemoryStore) -> None:
    long_words = "记住" * 700
    draft = intent(written_by="user", body=long_words, quotes=(Quote(text=long_words, via="cli"),))
    result = await store.create(PROJECT, draft)
    assert (await store.get(PROJECT, result.ref)).entry.quotes[0].session is None


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "text", "reason"),
    [
        ("notes", "free-form notes\n", "file name"),
        ("3f6c2a9e41b0", "no frontmatter at all\n", "frontmatter"),
        ("3f6c2a9e41b0", "---\ntitle: [unclosed\n---\nbody\n", "YAML"),
        ("3f6c2a9e41b0", SAMPLE.replace("schema: 1", "schema: 2"), "newer version"),
        ("3f6c2a9e41b0", SAMPLE.replace("id: 3f6c2a9e41b0", "id: 000000000000"), "id"),
        ("3f6c2a9e41b0", SAMPLE.replace("scope: project", "scope: global"), "scope"),
        ("3f6c2a9e41b0", SAMPLE.replace("context: null", "context: null\nmood: happy"), "mood"),
        ("3f6c2a9e41b0", SAMPLE.replace("title: 安装器", "title: " + "长" * 81 + "安装器"), "title"),
        ("3f6c2a9e41b0", SAMPLE + "x" * (16 * 1024), "16 KiB"),
    ],
)
async def test_unreadable_files_are_quarantined_and_deletable(
    store: FileMemoryStore, root: Path, name: str, text: str, reason: str
) -> None:
    directory = project_dir(root)
    directory.mkdir(parents=True)
    path = directory / f"{name}.md"
    path.write_text(text, encoding="utf-8")
    ref = f"project:{name}"

    assert await store.list_entries(PROJECT) == []
    assert await store.catalog(PROJECT) == []
    assert (await store.search(PROJECT, ""))[0] == []
    [quarantined] = await store.unreadable(PROJECT)
    assert quarantined.ref == ref
    assert reason in quarantined.reason
    error = await expect_error("memory_entry_invalid", store.get(PROJECT, ref))
    assert reason in error.message
    assert path.read_text(encoding="utf-8") == text  # never repaired

    await store.delete(PROJECT, ref, quarantined.version)
    assert not path.exists()


async def test_a_new_id_never_lands_on_an_existing_file(
    store: FileMemoryStore, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = project_dir(root)
    directory.mkdir(parents=True)
    taken = directory / "aaaaaaaaaaaa.md"
    taken.write_text("not an entry\n", encoding="utf-8")
    candidates = iter(["aaaaaaaaaaaa", "bbbbbbbbbbbb"])
    real_token_hex = file_store.secrets.token_hex
    # Six bytes is an entry id; other sizes (atomic-write temp names) stay random.
    monkeypatch.setattr(
        file_store.secrets, "token_hex", lambda n=None: next(candidates) if n == 6 else real_token_hex(n)
    )

    result = await store.create(PROJECT, intent())

    assert result.ref == "project:bbbbbbbbbbbb"
    assert taken.read_text(encoding="utf-8") == "not an entry\n"


# ---------------------------------------------------------------------------
# Search and catalog
# ---------------------------------------------------------------------------


async def test_search_covers_every_text_field(store: FileMemoryStore) -> None:
    draft = MemoryEntryDraft(
        type="engineering",
        title="标题词",
        body="正文词",
        written_by="model",
        applies_when="条件词",
        quotes=(Quote(text="原话词", via="message", session="s"),),
        context=Context(text="提议词", session="s"),
        evidence=(
            ToolResultEvidence(tool="shell", args="参数不搜", excerpt="摘录词", session="s"),
            FileEvidence(project=PROJECT.project_id or "", path="src/路径词.py", sha256=None, bytes=1),
            ReferenceEvidence(ref="https://出处词.example", note="备注不搜"),
        ),
    )
    created = await store.create(PROJECT, draft)

    for word in ("标题词", "正文词", "条件词", "原话词", "提议词", "摘录词", "路径词", "出处词"):
        page, _ = await store.search(PROJECT, word)
        assert [s.entry.ref for s in page] == [created.ref], word
    assert (await store.search(PROJECT, "参数不搜"))[0] == []


async def test_search_ranks_entries_with_every_word_first(store: FileMemoryStore) -> None:
    only_one = await store.create(PROJECT, intent(title="离线安装", body="只提离线"))
    both = await store.create(PROJECT, intent(title="离线 索引", body="离线与索引都提"))
    await store.create(PROJECT, intent(title="无关", body="什么都不提"))

    page, cursor = await store.search(PROJECT, "离线 索引")

    assert [s.entry.ref for s in page] == [both.ref, only_one.ref]
    assert cursor is None


async def test_search_normalizes_width_and_case(store: FileMemoryStore) -> None:
    created = await store.create(PROJECT, intent(title="Use SQLite WAL", body="ＷＡＬ mode"))
    assert [s.entry.ref for s in (await store.search(PROJECT, "sqlite"))[0]] == [created.ref]
    assert [s.entry.ref for s in (await store.search(PROJECT, "wal mode"))[0]] == [created.ref]


async def test_search_pages_ten_at_a_time_newest_first(store: FileMemoryStore, root: Path) -> None:
    for number in range(12):
        await store.create(PROJECT, intent(body=f"第 {number} 条"))
    # Give every file a distinct updated time so "newest first" is observable.
    for number, path in enumerate(entry_files(project_dir(root))):
        text = path.read_text(encoding="utf-8")
        stamp = f"2026-11-0{1 + number % 9}T{number:02d}:00:00Z"
        lines = [f"updated: '{stamp}'" if line.startswith("updated:") else line for line in text.splitlines()]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    first, cursor = await store.search(PROJECT, "")
    assert len(first) == 10 and cursor is not None
    second, last = await store.search(PROJECT, "", cursor)
    assert len(second) == 2 and last is None
    everything = [s.entry.updated for s in first + second]
    assert everything == sorted(everything, reverse=True)
    assert {s.entry.ref for s in first}.isdisjoint({s.entry.ref for s in second})
    await expect_error("memory_invalid_input", store.search(PROJECT, "", "not-a-cursor"))


async def test_search_with_global_ranks_both_scopes_together(store: FileMemoryStore) -> None:
    mine = await store.create(PROJECT, intent(title="项目里的离线"))
    shared = await store.create(GLOBAL_SCOPE, intent(title="全局的离线"))
    await store.create(OTHER_PROJECT, intent(title="别的项目的离线"))

    page, _ = await store.search(PROJECT, "离线", with_global=True)
    assert {s.entry.ref for s in page} == {mine.ref, shared.ref}
    assert [s.entry.ref for s in (await store.search(PROJECT, "离线"))[0]] == [mine.ref]


async def test_catalog_lists_what_the_catalog_shows(store: FileMemoryStore) -> None:
    created = await store.create(PROJECT, intent(applies_when="回答问题时"))
    [item] = await store.catalog(PROJECT)
    stored = await store.get(PROJECT, created.ref)
    assert (item.ref, item.type, item.title, item.applies_when, item.updated) == (
        created.ref,
        "intent",
        "回答用中文",
        "回答问题时",
        stored.entry.updated,
    )


# ---------------------------------------------------------------------------
# Projects and locations
# ---------------------------------------------------------------------------


async def test_the_first_write_records_where_the_project_is(store: FileMemoryStore, root: Path) -> None:
    await store.create(PROJECT, intent())
    moved = MemoryScope("project", PROJECT.project_id, root="/somewhere/else", anchor="git")
    await store.create(moved, intent(body="第二条"))

    record = json.loads((project_dir(root) / "project.json").read_text(encoding="utf-8"))
    assert record == {"root": "/work/shop-installer", "anchor": "git"}
    await store.create(GLOBAL_SCOPE, intent())
    projects = await store.list_projects()
    assert [(info.project_id, info.root) for info in projects] == [(PROJECT.project_id, "/work/shop-installer")]


async def test_a_project_without_its_record_still_lists(store: FileMemoryStore, root: Path) -> None:
    await store.create(MemoryScope("project", OTHER_PROJECT.project_id), intent())
    assert not (project_dir(root, OTHER_PROJECT) / "project.json").exists()
    assert [(info.project_id, info.root) for info in await store.list_projects()] == [(OTHER_PROJECT.project_id, None)]
    assert len(await store.list_entries(MemoryScope("project", OTHER_PROJECT.project_id))) == 1


async def test_locations_are_the_scope_directories(store: FileMemoryStore, root: Path) -> None:
    assert await store.location(PROJECT) == str(project_dir(root))
    assert await store.location(GLOBAL_SCOPE) == str(root / "global")


def test_a_scope_must_be_well_formed() -> None:
    with pytest.raises(MemoryOperationError):
        MemoryScope("project")
    with pytest.raises(MemoryOperationError):
        MemoryScope("project", "not-hex")
    with pytest.raises(MemoryOperationError):
        MemoryScope("global", root="/x")
    assert MemoryScope("project", PROJECT.project_id, root="/a") == MemoryScope("project", PROJECT.project_id)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


async def test_a_writer_holding_the_lock_makes_the_store_unavailable(
    store: FileMemoryStore, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await store.create(PROJECT, intent())
    monkeypatch.setattr(file_store, "LOCK_TIMEOUT_SECONDS", 0.2)
    held = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with FileLock(project_dir(root) / ".lock"):
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert held.wait(5)
        await expect_error("memory_store_unavailable", store.create(PROJECT, intent(body="并发")))
    finally:
        release.set()
        holder.join()
    assert len(await store.list_entries(PROJECT)) == 1


# ---------------------------------------------------------------------------
# Fidelity: what was saved is what is read back
# ---------------------------------------------------------------------------


async def test_text_round_trips_exactly_so_a_resave_is_a_duplicate(store: FileMemoryStore) -> None:
    for body in ("正文\n", "正文\n\n", "\n开头空行", "行尾空格  \n下一行\r\n"):
        draft = intent(title=f"正文 {body!r}", body=body)
        created = await store.create(PROJECT, draft)
        assert (await store.get(PROJECT, created.ref)).entry.body == body
        assert not (await store.create(PROJECT, draft)).created


async def test_a_file_saved_with_windows_line_endings_reads(store: FileMemoryStore, root: Path) -> None:
    path = project_dir(root) / "3f6c2a9e41b0.md"
    path.parent.mkdir(parents=True)
    path.write_bytes(SAMPLE.replace("\n", "\r\n").encode("utf-8"))

    stored = await store.get(PROJECT, "project:3f6c2a9e41b0")
    assert stored.entry.title == "安装器必须支持完全离线部署"
    assert "\r" not in stored.entry.body


@pytest.mark.parametrize(
    ("written", "read"),
    [
        ("2026-11-02T08:05:40Z", "2026-11-02T08:05:40Z"),
        ("'2026-11-02T08:05:40Z'", "2026-11-02T08:05:40Z"),
        ("2026-11-02T16:05:40+08:00", "2026-11-02T08:05:40Z"),
        ("'2026-11-02T16:05:40+08:00'", "2026-11-02T08:05:40Z"),
    ],
)
async def test_timestamps_read_back_in_utc(store: FileMemoryStore, root: Path, written: str, read: str) -> None:
    path = project_dir(root) / "3f6c2a9e41b0.md"
    path.parent.mkdir(parents=True)
    path.write_text(SAMPLE.replace("updated: 2026-11-02T08:05:40Z", f"updated: {written}"), encoding="utf-8")
    assert (await store.get(PROJECT, "project:3f6c2a9e41b0")).entry.updated == read


@pytest.mark.parametrize(
    "written",
    ["2026-13-45T00:00:00Z", "2026-11-02T08:05:40", "'2026-11-02T08:05:40'", "2026-11-02", "yesterday", "12"],
)
async def test_a_timestamp_that_is_no_moment_quarantines_only_its_file(
    store: FileMemoryStore, root: Path, written: str
) -> None:
    good = await store.create(PROJECT, intent())
    path = project_dir(root) / "3f6c2a9e41b0.md"
    path.write_text(SAMPLE.replace("updated: 2026-11-02T08:05:40Z", f"updated: {written}"), encoding="utf-8")

    assert [s.entry.ref for s in await store.list_entries(PROJECT)] == [good.ref]
    assert [s.entry.ref for s in (await store.search(PROJECT, ""))[0]] == [good.ref]
    assert [item.ref for item in await store.unreadable(PROJECT)] == ["project:3f6c2a9e41b0"]
    assert not (await store.create(PROJECT, intent())).created


async def test_paths_and_references_have_no_length_budget(store: FileMemoryStore) -> None:
    deep = "/".join(["很深的目录"] * 60) + "/file.py"
    url = "https://example.com/" + "a" * 500
    draft = MemoryEntryDraft(
        type="engineering",
        title="长路径",
        body="b",
        written_by="model",
        evidence=(
            FileEvidence(project=PROJECT.project_id or "", path=deep, sha256=None, bytes=1),
            ReferenceEvidence(ref=url, note="n" * 400),
            ToolResultEvidence(tool="mcp__server__" + "x" * 120, args="a", excerpt="e", session="s"),
        ),
    )
    created = await store.create(PROJECT, draft)
    assert (await store.get(PROJECT, created.ref)).entry.draft() == draft


# ---------------------------------------------------------------------------
# Writers racing each other
# ---------------------------------------------------------------------------


async def test_racing_identical_creates_store_one_entry(store: FileMemoryStore, root: Path) -> None:
    results = await asyncio.gather(*(store.create(PROJECT, intent()) for _ in range(8)))

    assert sum(result.created for result in results) == 1
    assert len({result.ref for result in results}) == 1
    assert len(entry_files(project_dir(root))) == 1


async def test_racing_updates_from_one_version_let_only_one_win(store: FileMemoryStore) -> None:
    created = await store.create(PROJECT, intent())
    outcomes = await asyncio.gather(
        *(store.update(PROJECT, created.ref, created.version, intent(title=f"第 {n} 版")) for n in range(6)),
        return_exceptions=True,
    )

    winners = [outcome for outcome in outcomes if isinstance(outcome, str)]
    losers = [outcome for outcome in outcomes if isinstance(outcome, MemoryOperationError)]
    assert len(winners) == 1
    assert len(losers) == 5 and {error.kind for error in losers} == {"memory_version_conflict"}
    assert (await store.get(PROJECT, created.ref)).version == winners[0]


async def test_who_wrote_an_entry_does_not_make_it_different(store: FileMemoryStore) -> None:
    by_model = await store.create(PROJECT, intent())
    by_extraction = await store.create(PROJECT, intent(written_by="extraction"))
    assert by_extraction == type(by_extraction)(ref=by_model.ref, version=by_model.version, created=False)


async def test_a_lost_project_record_does_not_fail_the_write(
    store: FileMemoryStore, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write = file_store.atomic_write_owner_only_bytes

    def write(path: Path, payload: bytes, **kwargs: object) -> None:
        if path.name == "project.json":
            raise PermissionError("read-only")
        real_write(path, payload, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(file_store, "atomic_write_owner_only_bytes", write)
    created = await store.create(PROJECT, intent())

    assert created.created
    assert (await store.get(PROJECT, created.ref)).version == created.version
    assert [(info.project_id, info.root) for info in await store.list_projects()] == [(PROJECT.project_id, None)]


async def test_a_quote_without_a_session_key_follows_the_schema_rule(store: FileMemoryStore, root: Path) -> None:
    directory = project_dir(root)
    directory.mkdir(parents=True)
    typed = SAMPLE.replace("written_by: model", "written_by: user").replace("via: message", "via: cli")
    (directory / "3f6c2a9e41b0.md").write_text(typed.replace("    session: a1b2c3d4\n", ""), encoding="utf-8")
    (directory / "4f6c2a9e41b0.md").write_text(
        SAMPLE.replace("3f6c2a9e41b0", "4f6c2a9e41b0").replace("    session: a1b2c3d4\n", ""), encoding="utf-8"
    )

    assert (await store.get(PROJECT, "project:3f6c2a9e41b0")).entry.quotes[0].session is None
    [quarantined] = await store.unreadable(PROJECT)
    assert quarantined.ref == "project:4f6c2a9e41b0"
    assert "session" in quarantined.reason


async def test_fractions_of_a_second_round_trip_and_order(store: FileMemoryStore, root: Path) -> None:
    quote = Quote(text="以后都用中文回答", via="message", session="s", at="2026-11-03T00:00:00.123Z")
    draft = intent(quotes=(quote,))
    created = await store.create(PROJECT, draft)
    assert (await store.get(PROJECT, created.ref)).entry.quotes[0].at == "2026-11-03T00:00:00.123Z"
    assert not (await store.create(PROJECT, draft)).created

    later = await store.create(PROJECT, intent(body="同一秒里更晚"))
    for ref, stamp in ((created.ref, "2026-11-03T00:00:00Z"), (later.ref, "2026-11-03T00:00:00.5Z")):
        path = project_dir(root) / f"{ref.partition(':')[2]}.md"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text(
            "\n".join(f"updated: '{stamp}'" if line.startswith("updated:") else line for line in lines) + "\n",
            encoding="utf-8",
        )
    assert [s.entry.ref for s in await store.list_entries(PROJECT)] == [later.ref, created.ref]


async def test_a_draft_timestamp_without_a_zone_is_refused(store: FileMemoryStore) -> None:
    quote = Quote(text="以后都用中文回答", via="message", session="s", at="2026-11-03T00:00:00")
    error = await expect_error("memory_invalid_input", store.create(PROJECT, intent(quotes=(quote,))))
    assert "quotes[0].at" in error.message


async def test_a_draft_timestamp_in_another_zone_is_refused_so_rereads_match(store: FileMemoryStore) -> None:
    quote = Quote(text="以后都用中文回答", via="message", session="s", at="2026-11-03T08:00:00.123+08:00")
    error = await expect_error("memory_invalid_input", store.create(PROJECT, intent(quotes=(quote,))))
    assert "UTC" in error.message
