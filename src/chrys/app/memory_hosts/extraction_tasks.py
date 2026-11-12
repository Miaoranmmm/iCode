# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Own complete extraction operations across cancellation of their waiters."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from contextlib import suppress
from typing import Any

from chrys.foundation.util.once_close import finish_close


class OperationNotAdmitted(asyncio.CancelledError):
    """Cancellation arrived before another operation could be admitted."""


class OperationOwner:
    """Keep admitted operations alive, retaining their actual durable results.

    Cancellation closes admission and is reported through ``cancel_requested``.
    It never cancels an operation already handed to this owner. Callers classify
    its actual return value after draining instead of inferring a rollback.
    """

    def __init__(self) -> None:
        self.cancel_requested = False
        self._tasks: set[asyncio.Task[Any]] = set()

    @property
    def pending_count(self) -> int:
        return sum(not task.done() for task in self._tasks)

    async def run[T](self, operation: Coroutine[Any, Any, T]) -> T:
        current = asyncio.current_task()
        if self.cancel_requested or (current is not None and current.cancelling()):
            self.cancel_requested = True
            operation.close()
            raise OperationNotAdmitted()
        task = asyncio.create_task(operation)
        self._tasks.add(task)
        try:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                self.cancel_requested = True
                # Close admission immediately, then drain even repeated waiter
                # cancellations. The operation's actual result still wins.
                with suppress(asyncio.CancelledError):
                    await finish_close(task)
                return task.result()
        finally:
            self._tasks.discard(task)

    async def drain(self) -> None:
        for task in tuple(self._tasks):
            try:
                await finish_close(task)
            except asyncio.CancelledError:
                self.cancel_requested = True
                task.result()
