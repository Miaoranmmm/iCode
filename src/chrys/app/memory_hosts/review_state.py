# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Private review checkpoints shared by the hook and review runner."""

from __future__ import annotations

import errno
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from chrys.app.memory_hosts.runtime_files import ensure_private_directory
from chrys.foundation.platform.files import atomic_write_owner_only_bytes, secure_open_owner_only_binary


@dataclass
class ReviewState:
    snapshot: dict[str, str] = field(default_factory=dict)
    policy: str = ""
    success_at: float = 0
    failures: int = 0
    failed_snapshot: dict[str, str] = field(default_factory=dict)
    failed_policy: str = ""
    paused: bool = False
    failed_writes: dict[str, str] = field(default_factory=dict)


def state_path(root: Path, project_id: str) -> Path:
    return root / "review" / "states" / f"{project_id}.json"


def load_state(root: Path, project_id: str) -> ReviewState:
    try:
        with secure_open_owner_only_binary(state_path(root, project_id)) as handle:
            data = json.load(handle)
        state = ReviewState(**data)
        for snapshot in (state.snapshot, state.failed_snapshot, state.failed_writes):
            if not isinstance(snapshot, dict) or any(
                not isinstance(k, str) or not isinstance(v, str) for k, v in snapshot.items()
            ):
                raise ValueError("Invalid snapshot")
        if (
            not isinstance(state.success_at, (int, float))
            or isinstance(state.success_at, bool)
            or not math.isfinite(state.success_at)
            or not isinstance(state.failures, int)
            or state.failures < 0
        ):
            raise ValueError("Invalid review clock")
        if not isinstance(state.paused, bool):
            raise ValueError("Invalid review pause")
        if not isinstance(state.policy, str) or not isinstance(state.failed_policy, str):
            raise ValueError("Invalid review policy")
        return state
    except OSError as error:
        if error.errno != errno.ENOENT:
            raise
        return ReviewState()
    except TypeError, ValueError:
        # Corrupt checkpoints cannot certify that a library has been reviewed.
        return ReviewState()


def save_state(root: Path, project_id: str, state: ReviewState) -> None:
    ensure_private_directory(root / "review")
    ensure_private_directory(root / "review" / "states")
    atomic_write_owner_only_bytes(
        state_path(root, project_id), json.dumps(asdict(state)).encode("utf-8"), create_parents=False
    )
