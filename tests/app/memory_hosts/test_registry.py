# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Host-process and atomic state contracts, independent of a local Claude install."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType

import pytest

from chrys.app.memory_hosts.claude_code import registry as registry_module
from chrys.app.memory_hosts.claude_code.registry import (
    DELIVERY_LIMIT,
    REGISTRATION_LIMIT,
    DeliveryState,
    HostKey,
    ProcessInfo,
    Registration,
    Registry,
    find_host,
    project_directory,
    project_source_supported,
    verify_host,
)
from chrys.service.memory.storage.store import MemoryOperationError
from tests.support.secure_files import plant_owner_only_bytes


@dataclass
class Tree:
    records: dict[int, ProcessInfo | None]
    inspected: list[int] = field(default_factory=list)

    def inspect(self, pid: int) -> ProcessInfo | None:
        self.inspected.append(pid)
        return self.records.get(pid)


def process(pid: int, ppid: int, exe: str, *, started: int = 17, argv: tuple[str, ...] | None = ()) -> ProcessInfo:
    return ProcessInfo(pid, ppid, exe, started, argv)


@pytest.mark.parametrize("exe", ["/custom/bin/claude", "C:\\Apps\\CLAUDE.EXE", "claude"])
def test_nearest_executable_ancestor_crosses_uv_and_shell_wrappers(exe: str) -> None:
    tree = Tree(
        {
            10: process(10, 20, "python"),
            20: process(20, 30, "uv"),
            30: process(30, 40, exe),
            40: process(40, 0, "claude"),
        }
    )
    assert find_host(tree, 10) == HostKey(30, 17)
    assert tree.inspected == [10, 20, 30]


@pytest.mark.parametrize("exe", ["/bin/not-claude", "/bin/claude-wrapper", "/bin/python"])
def test_argv_or_comm_does_not_establish_a_host(exe: str) -> None:
    tree = Tree({10: process(10, 20, "python"), 20: process(20, 0, exe, argv=("claude",))})
    assert find_host(tree, 10) is None


@pytest.mark.parametrize(
    "records", [{}, {10: process(10, 20, "python")}, {10: process(10, 20, "python"), 20: process(20, 10, "uv")}]
)
def test_missing_inaccessible_or_cyclic_parent_chain_is_unbound(records: dict[int, ProcessInfo | None]) -> None:
    assert find_host(Tree(records), 10) is None


def test_nearest_host_without_registration_does_not_use_outer_host(tmp_path: Path) -> None:
    tree = Tree({10: process(10, 20, "python"), 20: process(20, 30, "claude"), 30: process(30, 0, "claude")})
    registry = Registry(tmp_path)
    registry.write_registration(HostKey(30, 17), Registration("outer", "/outer.jsonl", str(tmp_path)))
    nearest = find_host(tree, 10)
    assert nearest == HostKey(20, 17)
    assert registry.read_registration(nearest) is None


@pytest.mark.parametrize("replacement", [None, process(20, 0, "claude", started=18), process(20, 0, "python")])
def test_dead_inaccessible_and_reused_pid_cannot_validate(replacement: ProcessInfo | None) -> None:
    assert verify_host(Tree({20: replacement}), HostKey(20, 17)) is False


def test_frozen_host_validation_does_not_walk_to_outer_host() -> None:
    tree = Tree({20: process(20, 30, "claude"), 30: process(30, 0, "claude")})
    assert verify_host(tree, HostKey(20, 17)) is True
    assert tree.inspected == [20]


@pytest.mark.parametrize(
    "argv", [None, ("claude", "--project-config-root", "private"), ("claude", "--project-config-root=private")]
)
def test_unsupported_project_source_is_not_replaced_by_environment(
    tmp_path: Path, argv: tuple[str, ...] | None
) -> None:
    tree = Tree({20: process(20, 0, "claude", argv=argv)})
    assert project_source_supported(tree, HostKey(20, 17)) is False
    assert project_directory(tree, HostKey(20, 17), {"CLAUDE_PROJECT_DIR": str(tmp_path)}) is None


def test_project_directory_uses_only_current_environment_and_checks_existence(tmp_path: Path) -> None:
    tree = Tree({20: process(20, 0, "claude", argv=("claude", "--not-project-config-root"))})
    assert project_directory(tree, HostKey(20, 17), {"CLAUDE_PROJECT_DIR": str(tmp_path)}) == str(tmp_path)
    for environ in ({}, {"CLAUDE_PROJECT_DIR": "relative"}, {"CLAUDE_PROJECT_DIR": str(tmp_path / "deleted")}):
        assert project_directory(tree, HostKey(20, 17), environ) is None


def test_registration_is_exact_atomic_last_writer_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = Registry(tmp_path)
    key = HostKey(20, 17)
    old = Registration("parent-full-id", "/parent.jsonl", str(tmp_path))
    new = Registration("child-full-id", "/child.jsonl", None)
    replaced: list[Path] = []
    original = registry_module.atomic_write_owner_only_bytes

    def replace(path: Path, payload: bytes, *, create_parents: bool = True) -> None:
        replaced.append(path)
        original(path, payload, create_parents=create_parents)

    monkeypatch.setattr(registry_module, "atomic_write_owner_only_bytes", replace)
    registry.write_registration(key, new)
    registry.write_registration(key, old)
    assert registry.read_registration(key) == old
    assert json.loads(registry.registration_path(key).read_text(encoding="utf-8")) == {
        "session_id": old.session_id,
        "transcript_path": old.transcript_path,
        "project_dir": old.project_dir,
    }
    assert replaced == [registry.registration_path(key)] * 2
    assert registry.registration_path(HostKey(20, 18)) != registry.registration_path(key)


@pytest.mark.parametrize(
    "payload",
    [
        b"not JSON",
        b"{}",
        b'{"session_id":"s","transcript_path":"/t","project_dir":null,"prompt":"private"}',
        b"x" * (REGISTRATION_LIMIT + 1),
    ],
)
def test_bad_registration_is_source_failure(tmp_path: Path, payload: bytes) -> None:
    registry = Registry(tmp_path)
    key = HostKey(20, 17)
    registry.registration_path(key).parent.mkdir(parents=True, exist_ok=True)
    plant_owner_only_bytes(registry.registration_path(key), payload)
    with pytest.raises(MemoryOperationError) as failure:
        registry.read_registration(key)
    assert failure.value.kind == "memory_store_unavailable"
    assert "private" not in str(failure.value)


def test_delivery_flag_is_independent_of_last_text(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    key = HostKey(20, 17)
    state = DeliveryState("full-session-id", "catalog", True)
    registry.write_delivery(key, state)
    before = registry.delivery_path(key, state.session_id).read_bytes()
    registry.mark_redeliver(key, state.session_id)
    assert registry.needs_redeliver(key, state.session_id) is True
    assert registry.delivery_path(key, state.session_id).read_bytes() == before
    assert registry.read_delivery(key, state.session_id) == state
    registry.clear_redeliver(key, state.session_id)
    assert registry.needs_redeliver(key, state.session_id) is False
    assert registry.read_delivery(key, "other") is None
    assert state.session_id not in registry.delivery_path(key, state.session_id).name


@pytest.mark.parametrize(
    "payload", [b"bad", b"x" * (DELIVERY_LIMIT + 1), b'{"session_id":"other","last_text":null,"ever_nonempty":true}']
)
def test_untrustworthy_delivery_state_is_first_delivery(tmp_path: Path, payload: bytes) -> None:
    registry = Registry(tmp_path)
    key = HostKey(20, 17)
    registry.delivery_path(key, "session").parent.mkdir(parents=True, exist_ok=True)
    plant_owner_only_bytes(registry.delivery_path(key, "session"), payload)
    assert registry.read_delivery(key, "session") is None


def test_corrupt_last_text_can_retain_confirmed_ever_nonempty(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    key = HostKey(20, 17)
    registry.delivery_path(key, "session").parent.mkdir(parents=True, exist_ok=True)
    plant_owner_only_bytes(
        registry.delivery_path(key, "session"), b'{"session_id":"session","last_text":7,"ever_nonempty":true}'
    )
    assert registry.read_delivery(key, "session") == replace(DeliveryState("session", None, True), valid=False)


def test_redelivery_flag_is_presence_only(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    key = HostKey(20, 17)
    registry.redeliver_path(key, "session").parent.mkdir(parents=True)
    plant_owner_only_bytes(registry.redeliver_path(key, "session"), b"presence is sufficient")
    assert registry.needs_redeliver(key, "session") is True


def test_registration_json_recursion_failure_is_a_source_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = Registry(tmp_path)
    key = HostKey(20, 17)
    registry.registration_path(key).parent.mkdir(parents=True)
    plant_owner_only_bytes(registry.registration_path(key), b"{}")
    monkeypatch.setattr(registry_module, "json", _recursive_json_parser())
    with pytest.raises(MemoryOperationError) as failure:
        registry.read_registration(key)
    assert failure.value.kind == "memory_store_unavailable"
    assert "memory source is unavailable" in failure.value.message


def test_delivery_json_recursion_failure_falls_back_to_first_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = Registry(tmp_path)
    key = HostKey(20, 17)
    registry.delivery_path(key, "session").parent.mkdir(parents=True)
    plant_owner_only_bytes(registry.delivery_path(key, "session"), b"{}")
    monkeypatch.setattr(registry_module, "json", _recursive_json_parser())
    assert registry.read_delivery(key, "session") is None


def _recursive_json_parser() -> ModuleType:
    parser = ModuleType("json")

    def loads(payload: bytes) -> object:
        raise RecursionError("private parser details")

    parser.loads = loads
    return parser
