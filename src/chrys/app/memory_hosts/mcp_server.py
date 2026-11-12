# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared stdio memory tools with explicit, isolated host context providers."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from chrys.app.memory_hosts.codex import CodexProvider
from chrys.app.memory_hosts.context import (
    CLAUDE_INSTRUCTIONS,
    CLAUDE_READ_ONLY_INSTRUCTIONS,
    CODEX_INSTRUCTIONS,
    EMPTY_CATALOG,
    READ_ONLY_REASON,
    SOURCE_UNAVAILABLE,
    MemoryContextProvider,
    Mode,
)
from chrys.app.memory_hosts.schemas import (
    EXTRACTION_ONLY_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    CatalogRequest,
    ForgetRequest,
    ReadRequest,
    SearchRequest,
    WithdrawRequest,
    decode_request,
    tool_definitions,
)
from chrys.foundation.platform import get_platform
from chrys.orchestration.startup import bootstrap_runtime
from chrys.service.memory.names import MEMORY_TOOL_NAMES
from chrys.service.memory.operations import MemoryReader, MemoryWriter, SaveRequest
from chrys.service.memory.render import (
    render_catalog,
    render_deleted,
    render_entry,
    render_error,
    render_saved,
    render_search,
    render_withdrawn,
)
from chrys.service.memory.storage.file_change_log import FileChangeLog
from chrys.service.memory.storage.file_store import FileMemoryStore
from chrys.service.memory.storage.store import MemoryOperationError, MemoryStore
from chrys.service.memory.tuning import V0, MemoryTuning, load_active

if TYPE_CHECKING:
    from chrys.service.memory.change_log import ChangeLog


def _result(text: str, *, error: bool = False) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=error)


class MemoryTools:
    """One dispatch boundary for the SDK and typed in-process tests."""

    def __init__(
        self,
        store: MemoryStore,
        mode: Mode,
        context: MemoryContextProvider,
        *,
        change_log: ChangeLog | None = None,
        tuning: MemoryTuning = V0,
    ) -> None:
        if isinstance(context, CodexProvider) and mode != "read-only":
            raise ValueError("Codex memory connections must be read-only.")
        self.store = store
        self.mode = mode
        self.context = context
        self.change_log = change_log
        self.tuning = tuning

    async def call(self, name: str, arguments: object, metadata: Mapping[str, object] | None) -> CallToolResult:
        """Keep claim errors separate from source construction and unexpected failures."""
        try:
            if self.mode == "read-only" and name in MEMORY_TOOL_NAMES - READ_ONLY_TOOL_NAMES:
                return _result("Error: " + READ_ONLY_REASON, error=True)
            if name in EXTRACTION_ONLY_TOOL_NAMES:
                raise MemoryOperationError(
                    "memory_invalid_input", "This tool is available only during extraction review."
                )
            request = decode_request(name, arguments)
            if isinstance(request, (SaveRequest, ForgetRequest, WithdrawRequest)):
                try:
                    context = await self.context.write_context(metadata)
                except MemoryOperationError as error:
                    if error.kind == "memory_invalid_input":
                        raise MemoryOperationError("memory_store_unavailable", SOURCE_UNAVAILABLE) from error
                    raise
                writer = MemoryWriter(
                    self.store,
                    context.project,
                    "model",
                    change_log=self.change_log,
                    source=context.view.current_session,
                    project_root=Path(context.project.root)
                    if context.project is not None and context.project.root is not None
                    else None,
                )
                if isinstance(request, SaveRequest):
                    text = render_saved(await writer.save(request, context.view))
                elif isinstance(request, WithdrawRequest):
                    text = render_withdrawn(
                        await writer.withdraw(request.ref, request.version, request.quote, context.view)
                    )
                else:
                    text = render_deleted(
                        await writer.forget(request.ref, request.version, request.quote, context.view)
                    )
            else:
                reading = await self.context.read_context(metadata)
                reader = MemoryReader(
                    self.store,
                    reading.project,
                    project_root=Path(reading.project.root)
                    if reading.project is not None and reading.project.root is not None
                    else None,
                )
                if isinstance(request, ReadRequest):
                    text = render_entry(await reader.read(request.ref))
                elif isinstance(request, SearchRequest):
                    text = render_search(await reader.search(request.query, request.scope, request.cursor))
                elif isinstance(request, CatalogRequest):
                    text = render_catalog(await reader.catalog(), tuning=self.tuning) or EMPTY_CATALOG
                else:
                    raise RuntimeError("Unhandled memory request.")
            return _result(text)
        except MemoryOperationError as error:
            return _result("Error: " + render_error(error, name), error=True)
        except Exception:
            return _result("Error: Memory operation failed. Try again later.", error=True)


def create_server(
    store: MemoryStore,
    mode: Mode,
    context: MemoryContextProvider,
    *,
    change_log: ChangeLog | None = None,
    tuning: MemoryTuning = V0,
) -> Server:
    """Build the shared protocol; no writer exists on a read-only connection."""
    dispatch = MemoryTools(store, mode, context, change_log=change_log, tuning=tuning)
    instructions = (
        CODEX_INSTRUCTIONS
        if isinstance(context, CodexProvider)
        else CLAUDE_READ_ONLY_INSTRUCTIONS
        if mode == "read-only"
        else CLAUDE_INSTRUCTIONS
    )
    if tuning.guidance_strength == "describe":
        instructions = instructions.replace("Read relevant entries with memory_read. ", "").replace(
            ", and memory_read before relying on an entry.", "."
        )
    server = Server("chrys-memory", instructions=instructions)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return tool_definitions(writable=mode == "read-write")

    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict[str, object]) -> CallToolResult:
        # Copy custom _meta now; None and an explicitly empty object have different contracts.
        meta = server.request_context.meta
        metadata = None if meta is None else meta.model_dump(by_alias=True, exclude_unset=True)
        return await dispatch.call(name, arguments, metadata)

    return server


async def _run(host: str, mode: Mode) -> None:
    config_dir = get_platform().config_dir
    try:
        tuning = await asyncio.to_thread(load_active, config_dir)
    except ValueError:
        sys.stderr.write("Memory tuning unavailable; using v0.\n")
        tuning = V0
    root = config_dir / "memory"
    if host == "codex":
        provider: MemoryContextProvider = await asyncio.to_thread(CodexProvider)
    else:
        from chrys.app.memory_hosts.claude_code.provider import ClaudeCodeProvider
        from chrys.app.memory_hosts.claude_code.registry import Registry

        provider = await asyncio.to_thread(ClaudeCodeProvider, Registry(root))
    change_log = FileChangeLog(root) if mode == "read-write" else None
    server = create_server(FileMemoryStore(root), mode, provider, change_log=change_log, tuning=tuning)
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


def main(argv: Sequence[str] | None = None) -> int:
    """Run only the explicitly selected host and public interactive mode."""
    parser = argparse.ArgumentParser(description="Memory MCP stdio server")
    parser.add_argument("--host", required=True, choices=("claude-code", "codex"))
    parser.add_argument("--mode", required=True, choices=("read-write", "read-only"))
    args = parser.parse_args(argv)
    if args.host == "codex" and args.mode != "read-only":
        parser.error("Codex memory connections must be read-only.")
    try:
        bootstrap_runtime(configure_stdio=True, apply_patches=False, setup_telemetry=False, project_root=None)
        asyncio.run(_run(args.host, args.mode))
    except Exception:
        sys.stderr.write("Memory MCP server could not start.\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
