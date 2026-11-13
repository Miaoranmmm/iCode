# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Private runtime directories and verified sidecar locks shared by memory hosts."""

from __future__ import annotations

import os
from pathlib import Path

from chrys.foundation.platform.files import secure_open_owner_only_append, strip_inherited_directory_acl
from chrys.foundation.util.lock import FileLock


def ensure_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink():
        raise OSError("Automatic extraction directory must not be a link.")
    path.chmod(0o700)
    strip_inherited_directory_acl(path)


def private_file_lock(path: Path, timeout: float) -> FileLock:
    # Pre-create and verify the lock with the same owner-only gates as the state.
    os.close(secure_open_owner_only_append(path).fd)
    return FileLock(path, timeout=timeout)
