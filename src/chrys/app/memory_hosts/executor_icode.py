# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Run one fixed extraction batch with profile-scoped native memory tools."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from chrys.app.memory_hosts.extraction_review import load_review_snapshot
from chrys.app.memory_hosts.extraction_tasks import OperationNotAdmitted, OperationOwner
from chrys.app.memory_hosts.schemas import (
    MergeRequest,
    NoteConflictRequest,
    ReadRequest,
    SearchRequest,
    WithdrawRequest,
    decode_request,
    tool_definitions,
)
from chrys.foundation.errors import ProviderResponseError
from chrys.foundation.tool_result_metadata import (
    TOOL_ERROR_KIND_METADATA_KEY,
    TOOL_FAILED_METADATA_KEY,
    TOOL_INTERRUPTED_METADATA_KEY,
    TOOL_RESULT_METADATA_KEY,
    tool_result_metadata_failure_state,
)
from chrys.foundation.util.once_close import OnceClose
from chrys.kernel import (
    EXHAUSTION_TAIL_KWARG,
    Agent,
    ChatContext,
    ChatMiddleware,
    ChatResponse,
    Content,
    FunctionInvocationContext,
    FunctionTool,
    LoopRecorder,
    Message,
    ToolLoopLayer,
)
from chrys.kernel.client import OUTPUT_CAP_OPTION_ALIASES
from chrys.kernel.exchanges import (
    TOOL_CALL_CONTENT_TYPES,
    EmptyIdPolicy,
    LiveAccessor,
    NoneIdPolicy,
    PairingPolicy,
    iter_exchanges,
    pair_results,
)
from chrys.service.agent_middleware.response_validation import ResponseValidationMiddleware
from chrys.service.llm.clients import create_client
from chrys.service.memory.extraction import (
    DEFAULT_MATERIAL_PROFILE,
    CallOutcome,
    ExtractionBatch,
    ExtractionError,
    ExtractionResult,
    MaterialProfile,
    SaveReceipt,
    Termination,
    UsageField,
    UsageSummary,
    render_prompt,
)
from chrys.service.memory.names import MEMORY_TOOL_NAMES
from chrys.service.memory.operations import MemoryReader, MemoryWriter, MergePartialError, SaveRequest
from chrys.service.memory.render import (
    render_catalog,
    render_entry,
    render_error,
    render_merged,
    render_saved,
    render_search,
    render_withdrawn,
)
from chrys.service.memory.storage.store import ENTRY_ID_RE, MemoryInputRejected, MemoryOperationError, parse_ref
from chrys.service.memory.tuning import V0, MemoryTuning
from chrys.service.memory.verification import SourceVerificationError
from chrys.service.profiles.models.options import (
    AUTO_INTERLEAVED_THINKING_OPTION,
    THINKING_BLOCK_BINDING_OPTION,
    effective_chat_options,
)
from chrys.service.profiles.models.schema import ModelProfile, uses_responses_wire_dialect

if TYPE_CHECKING:
    from chrys.service.memory.change_log import ChangeLog
    from chrys.service.memory.storage.store import MemoryStore

_TOOL_NAMES = MEMORY_TOOL_NAMES & {
    "memory_search",
    "memory_read",
    "memory_save",
    "memory_withdraw",
    "memory_merge",
    "memory_note_conflict",
}
_KNOWN_RAW_TOOLS = frozenset(
    {
        "custom_tool_call",
        "apply_patch_call",
        "computer_call",
        "web_search_call",
        "file_search_call",
        "code_interpreter_call",
        "image_generation_call",
        "mcp_call",
        "mcp_list_tools",
        "mcp_approval_request",
        "local_shell_call",
        "shell_call",
        "tool_search_call",
        "tool_search_output",
        "shell_call_output",
    }
)
_OUTCOME_KEY = "extraction_outcome"
_RECORD_KEY = "extraction_record"
_SAFE_FAILURE = "Error: Memory operation failed."
_EMPTY_CATALOG = "No saved memories."
_USAGE_FIELDS = (
    "input_token_count",
    "output_token_count",
    "total_token_count",
    "reasoning_output_token_count",
    "cache_read_input_token_count",
    "cache_creation_input_token_count",
)
_PAIRING = PairingPolicy(
    call_types=frozenset({"function_call"}),
    include_informational_calls=False,
    result_types=frozenset({"function_result"}),
    none_id=NoneIdPolicy.UNPAIRABLE_OCCURRENCE,
    empty_id=EmptyIdPolicy.UNPAIRABLE_OCCURRENCE,
    malformed_id="treat_as_none",
)


Wire = Literal["responses", "messages"]
_PREPARED_KEY = object()
_COMMON_OPTIONS = frozenset(
    {"temperature", "top_p", "metadata", "service_tier", "allow_multiple_tool_calls", *OUTPUT_CAP_OPTION_ALIASES}
)
_RESPONSES_OPTIONS = _COMMON_OPTIONS | {"reasoning", "store", "parallel_tool_calls", "prompt_cache_key"}
_MESSAGES_OPTIONS = _COMMON_OPTIONS | {
    "top_k",
    "thinking",
    "betas",
    "additional_beta_flags",
    THINKING_BLOCK_BINDING_OPTION,
    AUTO_INTERLEAVED_THINKING_OPTION,
    "output_config",
}


@dataclass(frozen=True, slots=True, init=False)
class PreparedProfile:
    """A run-local validated profile, constructed only by prepare_profile."""

    profile: ModelProfile
    wire: Wire

    def __init__(self, profile: ModelProfile, wire: Wire, *, _key: object) -> None:
        if _key is not _PREPARED_KEY:
            raise TypeError("Use prepare_profile to construct PreparedProfile.")
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "wire", wire)


def _validate_options(options: dict[str, Any], allowed: frozenset[str], prefix: str = "") -> None:
    """Allow-list top-level chat_options/extra_body keys, not arbitrary nested dictionaries.

    output_config has a restricted shape; thinking's effective budget is checked
    separately after environment resolution. Other values are not recursed.
    """
    for key, value in options.items():
        path = prefix + key
        if key not in allowed:
            raise ExtractionError("memory_invalid_input", f"Extraction profile option is not allowed: {path}.")
        if key == "output_config":
            if not isinstance(value, dict):
                raise ExtractionError("memory_invalid_input", f"Invalid extraction profile option: {path}.")
            for nested in value:
                if nested != "effort":
                    raise ExtractionError(
                        "memory_invalid_input", f"Extraction profile option is not allowed: {path}.{nested}."
                    )


def _validate_thinking(options: dict[str, Any], cap: int, prefix: str = "") -> None:
    value = options.get("thinking")
    if isinstance(value, dict):
        if value.get("type") == "enabled" and "budget_tokens" not in value:
            raise ExtractionError("memory_invalid_input", f"{prefix}thinking.budget_tokens is required when enabled.")
        if "budget_tokens" in value:
            budget = value["budget_tokens"]
            if not isinstance(budget, int) or isinstance(budget, bool) or not 1024 <= budget < cap:
                raise ExtractionError(
                    "memory_invalid_input",
                    f"{prefix}thinking.budget_tokens must be at least 1024 and below the output cap.",
                )


def prepare_profile(profile: ModelProfile, max_output_tokens: int = 16_384) -> PreparedProfile:
    """Validate only user-supplied options, then normalize one run-local copy."""
    wire: Wire
    if profile.provider == "anthropic":
        wire = "messages"
    elif uses_responses_wire_dialect(profile):
        wire = "responses"
    else:
        raise ExtractionError("unsupported_extraction_protocol", "Extraction requires Responses or Anthropic Messages.")
    if max_output_tokens <= 0:
        raise ExtractionError("memory_invalid_input", "max_output_tokens must be positive.")
    try:
        raw = json.loads(profile.chat_options) if profile.chat_options else {}
    except ValueError, TypeError:
        raise ExtractionError("memory_invalid_input", "Invalid extraction profile chat options.") from None
    if not isinstance(raw, dict):
        raise ExtractionError("memory_invalid_input", "Invalid extraction profile chat options.")
    extra = raw.get("extra_body", {})
    if not isinstance(extra, dict):
        raise ExtractionError("memory_invalid_input", "Invalid extraction profile extra_body.")
    cap = min(max_output_tokens, profile.max_output_tokens)
    allowed = _MESSAGES_OPTIONS if wire == "messages" else _RESPONSES_OPTIONS
    _validate_options({key: value for key, value in raw.items() if key != "extra_body"}, allowed)
    _validate_options(extra, allowed | ({"cache_control"} if wire == "messages" else set()), "extra_body.")
    copied = replace(profile, max_output_tokens=cap)
    options = dict(effective_chat_options(copied) or {})
    clean_extra = dict(options.get("extra_body") or {})
    _validate_thinking(options, cap)
    _validate_thinking(clean_extra, cap, "extra_body.")
    for key in OUTPUT_CAP_OPTION_ALIASES:
        options.pop(key, None)
        clean_extra.pop(key, None)
    options.update(max_tokens=cap, tool_choice="auto")
    if wire == "responses":
        options["store"] = False
        if "store" in clean_extra:
            clean_extra["store"] = False
    if clean_extra:
        options["extra_body"] = clean_extra
    else:
        options.pop("extra_body", None)
    normalized = replace(copied, chat_options=json.dumps(options, ensure_ascii=True))
    return PreparedProfile(normalized, wire, _key=_PREPARED_KEY)


def extraction_tool_schemas(
    profile: MaterialProfile = DEFAULT_MATERIAL_PROFILE,
) -> tuple[dict[str, Any], ...]:
    """Use the profile's native tools, requiring verifiable batch sessions."""
    definitions = [
        definition
        for definition in tool_definitions(
            writable=True,
            intent_only=profile.tool_schema == "intent",
            review=profile.tool_schema == "engineering",
            created_rewrites=profile.allow_created_rewrites,
            describe_refs=profile.describe_refs,
        )
        if definition.name in _TOOL_NAMES & MEMORY_TOOL_NAMES & set(profile.tool_names)
    ]
    result: list[dict[str, Any]] = []
    for definition in definitions:
        schema = definition.inputSchema
        if definition.name == "memory_save":
            schema["properties"]["type"]["enum"] = list(profile.allowed_types)
            quote = schema["properties"]["user_quotes"]["items"]
            quote["required"] = ["text", "session"]
            quote["properties"]["session"]["description"] = (
                "Copy the full source session id shown in source_sessions, or from memory_read when retaining a stored quote."
            )
            if profile.tool_schema == "engineering":
                evidence = schema["properties"]["evidence"]["items"]
                evidence["oneOf"] = [
                    item
                    for item in evidence["oneOf"]
                    if item["properties"]["kind"]["const"] in profile.allowed_evidence
                ]
                for item in evidence["oneOf"]:
                    if "session" in item["properties"]:
                        item["required"].append("session")
                schema["dependentRequired"]["context_quote"] = ["context_session"]
        elif definition.name == "memory_withdraw":
            schema["required"].append("session")
        result.append({"name": definition.name, "description": definition.description, "parameters": schema})
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CallRecord:
    """One settled native call; no arguments, source text or exception detail."""

    identity: int
    name: str
    outcome: CallOutcome
    kind: str | None = None
    saved: SaveReceipt | None = None
    conflict: bool = False


class ExtractionStopped(Exception):
    """Stop before another wire request without rewriting a landed response."""

    def __init__(self, reason: Termination) -> None:
        super().__init__(reason)
        self.reason = reason


class RawUsage(ChatMiddleware):
    """Keep actual fields once per wire response, including failed responses."""

    def __init__(self) -> None:
        self.observations: list[Mapping[str, Any] | None] = []

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        try:
            await call_next()
        except ProviderResponseError as error:
            self.observations.append(error.usage_details)
            raise
        except asyncio.CancelledError:
            self.observations.append(None)
            raise
        except Exception:
            self.observations.append(None)
            raise
        if not isinstance(context.result, ChatResponse):
            self.observations.append(None)
        else:
            self.observations.append(context.result.usage_details)

    def summary(self) -> UsageSummary:
        names = set(_USAGE_FIELDS)
        for observation in self.observations:
            if observation is not None:
                names.update(observation)
        fields: list[UsageField] = []
        for name in sorted(names):
            values = [None if row is None else row.get(name) for row in self.observations]
            actual = [value for value in values if isinstance(value, int | float) and not isinstance(value, bool)]
            fields.append(UsageField(name, sum(actual), len(values) - len(actual)))
        return UsageSummary(tuple(fields), calls=len(self.observations))


@dataclass(frozen=True, slots=True)
class RawTurn:
    """Protocol facts inspected before decoded calls can be admitted."""

    limited: bool = False
    failure: str | None = None
    unsupported: tuple[tuple[str, int], ...] = ()
    hosted: tuple[tuple[str, int], ...] = ()


def _raw_items(items: list[Any], response: ChatResponse, wire: Wire, *, limited: bool) -> RawTurn:
    ids: set[str] = set()
    unknown: Counter[str] = Counter()
    hosted_types: Counter[str] = Counter()
    call_failure: str | None = None
    for item in items:
        kind = getattr(item, "type", None)
        if kind == ("tool_use" if wire == "messages" else "function_call"):
            call_id = getattr(item, "id" if wire == "messages" else "call_id", None)
            if not isinstance(call_id, str) or not call_id:
                call_failure = call_failure or "invalid_call_id"
            elif call_id in ids:
                call_failure = call_failure or "duplicate_call_id"
            else:
                ids.add(call_id)
        elif kind not in (
            ("text", "thinking", "redacted_thinking") if wire == "messages" else ("message", "reasoning")
        ):
            if wire == "messages":
                hosted = isinstance(kind, str) and (
                    kind in {"server_tool_use", "mcp_tool_use", "mcp_tool_result"} or kind.endswith("_tool_result")
                )
            else:
                hosted = kind in _KNOWN_RAW_TOOLS or any(
                    content.provider_hosted and content.raw_representation is item
                    for message in response.messages
                    for content in message.contents
                )
            if hosted:
                hosted_types[kind if isinstance(kind, str) else "<missing>"] += 1
            else:
                unknown[kind if isinstance(kind, str) else "<missing>"] += 1
    failure = call_failure or (
        "unexpected_tool_call" if hosted_types else "unsupported_raw_output" if unknown else None
    )
    return RawTurn(
        limited=limited,
        failure=failure,
        unsupported=tuple(sorted(unknown.items())),
        hosted=tuple(sorted(hosted_types.items())),
    )


def _read_responses(response: ChatResponse) -> RawTurn:
    # Item-level status is not admission evidence: a response-level limit stops
    # dispatch of every item, including otherwise complete function calls.
    raw = response.raw_representation
    status = getattr(raw, "status", None)
    limited = (
        status == "incomplete"
        and getattr(getattr(raw, "incomplete_details", None), "reason", None) == "max_output_tokens"
    )
    if status == "incomplete" and not limited:
        return RawTurn(failure="unknown_incomplete")
    if status not in ("completed", "incomplete"):
        return RawTurn(failure="invalid_response_status")
    if (status == "completed" and response.finish_reason not in (None, "stop")) or (
        limited and response.finish_reason != "length"
    ):
        return RawTurn(failure="contradictory_response_status")
    output = getattr(raw, "output", None)
    if not isinstance(output, list):
        return RawTurn(failure="invalid_raw_output")
    return _raw_items(output, response, "responses", limited=limited)


def _read_messages(response: ChatResponse) -> RawTurn:
    raw = response.raw_representation
    if getattr(raw, "type", None) != "message":
        return RawTurn(failure="invalid_response_status")
    blocks = getattr(raw, "content", None)
    if not isinstance(blocks, list):
        return RawTurn(failure="invalid_raw_output")
    reason = getattr(raw, "stop_reason", None)
    has_calls = any(getattr(block, "type", None) == "tool_use" for block in blocks)
    if reason == "refusal":
        return RawTurn(failure="refusal")
    if reason == "model_context_window_exceeded":
        return RawTurn(failure="context_window_exceeded")
    if reason in ("pause_turn", "stop_sequence", "compaction"):
        return RawTurn(failure="contradictory_response_status")
    if reason not in ("end_turn", "tool_use", "max_tokens"):
        return RawTurn(failure="invalid_response_status")
    turn = _raw_items(blocks, response, "messages", limited=reason == "max_tokens")
    if turn.hosted:
        return turn
    if (reason == "end_turn" and has_calls) or (reason == "tool_use" and not has_calls):
        return RawTurn(failure="contradictory_response_status")
    return turn


class ExtractionGuard(ChatMiddleware):
    """Observe raw calls before filtering, landing, echo removal or dispatch."""

    def __init__(
        self,
        owner: OperationOwner,
        max_turns: int,
        wire: Wire,
        *,
        tool_names: Sequence[str] | None = None,
        stop_on_hard_failure: bool = False,
    ) -> None:
        self.owner = owner
        self.tool_names = frozenset(_TOOL_NAMES if tool_names is None else tool_names)
        self.stop_on_hard_failure = stop_on_hard_failure
        self.wire = wire
        self.max_turns = max_turns
        self.requests = 0
        self.tail_seen = False
        self.termination: Termination = "failed"
        self.stop_reason: str | None = None
        self.failures: list[str] = []
        self.unsupported: Counter[str] = Counter()
        self.accepted: list[tuple[int, Content]] = []
        self.current_calls: dict[str, tuple[int, Content]] = {}
        self.records: dict[int, CallRecord] = {}

    def fail(self, kind: str) -> None:
        self.failures.append(kind)
        self.termination, self.stop_reason = "failed", kind

    def _reject(self, kind: str) -> NoReturn:
        self.fail(kind)
        raise ExtractionStopped("failed")

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        nested = context.kwargs.get("client_kwargs")
        tail = isinstance(nested, Mapping) and nested.get(EXHAUSTION_TAIL_KWARG) is True
        if tail:
            self.tail_seen = True
        if self.owner.cancel_requested:
            self.termination, self.stop_reason = "cancelled", "cancelled"
            raise ExtractionStopped("cancelled")
        observed = _observe_calls(self, context.messages)
        if self.failures or (self.stop_on_hard_failure and observed.has_hard_failures):
            await self.owner.drain()
            self.termination = "failed"
            self.stop_reason = "tool_error_stop" if tail else "tool_failure"
            raise ExtractionStopped("failed")
        if tail:
            if observed.has_tool_failures:
                self.termination = "failed"
                self.stop_reason = "tool_error_stop"
            elif self.requests >= self.max_turns:
                self.termination, self.stop_reason = "turn_limited", "max_turns"
            else:
                self.termination, self.stop_reason = "failed", "tool_error_stop"
                self.failures.append("tool_error_stop")
            raise ExtractionStopped(self.termination)
        self.requests += 1
        await call_next()
        response = context.result
        if not isinstance(response, ChatResponse):
            self._reject("invalid_response")
        if response.continuation_token is not None:
            self._reject("continuation")
        turn = _read_messages(response) if self.wire == "messages" else _read_responses(response)
        self.unsupported.update(dict(turn.unsupported))
        if turn.failure is not None:
            self._reject(turn.failure)
        calls = [
            content
            for message in response.messages
            for content in message.contents
            if content.type in TOOL_CALL_CONTENT_TYPES
        ]
        for content in calls:
            if (
                content.type != "function_call"
                or content.name not in self.tool_names & _TOOL_NAMES & MEMORY_TOOL_NAMES
                or content.informational_only
                or content.provider_hosted
            ):
                self._reject("unexpected_tool_call")
            if not isinstance(content.call_id, str) or not content.call_id:
                self._reject("invalid_call_id")
        if turn.limited:
            if observed.has_tool_failures:
                self.termination, self.stop_reason = "failed", "tool_failure"
            else:
                self.termination, self.stop_reason = "output_limited", "max_output_tokens"
            raise ExtractionStopped(self.termination)
        admitted = [(len(self.accepted) + index, content) for index, content in enumerate(calls)]
        self.current_calls = {
            content.call_id: (identity, content) for identity, content in admitted if isinstance(content.call_id, str)
        }
        self.accepted.extend(admitted)
        if calls:
            self.termination = "failed"
        else:
            self.termination, self.stop_reason = "completed", None


class ICodeExecutor:
    """Create one owned client and a fresh fixed-view Agent for each batch."""

    def __init__(
        self,
        profile: PreparedProfile,
        store: MemoryStore,
        change_log: ChangeLog,
        run_id: str,
        *,
        references: tuple[tuple[str, str], ...] = (),
        material_profile: MaterialProfile = DEFAULT_MATERIAL_PROFILE,
        max_turns: int = 24,
        timeout_seconds: float = 600,
        tuning: MemoryTuning = V0,
    ) -> None:
        if not isinstance(profile, PreparedProfile):
            raise MemoryOperationError(
                "memory_invalid_input", "Extraction requires a PreparedProfile from prepare_profile."
            )
        if max_turns <= 0 or timeout_seconds <= 0:
            raise MemoryOperationError("memory_invalid_input", "Extraction limits must be positive.")
        self.profile = profile.profile
        self.wire = profile.wire
        self.store, self.change_log, self.run_id = store, change_log, run_id
        self.references = references
        self.material_profile = material_profile
        self.max_turns, self.timeout_seconds = max_turns, timeout_seconds
        self.tuning = tuning

    async def run(self, batch: ExtractionBatch) -> ExtractionResult:
        owner = OperationOwner()
        guard = ExtractionGuard(
            owner,
            self.max_turns,
            self.wire,
            tool_names=self.material_profile.tool_names,
            stop_on_hard_failure=self.material_profile.allow_input_rejections,
        )
        usage = RawUsage()
        recorder = LoopRecorder()
        messages: Sequence[Message] | None = None
        reader = MemoryReader(
            self.store,
            batch.project,
            project_root=Path(batch.project.root) if batch.project.root is not None else None,
        )
        writer = MemoryWriter(
            self.store,
            batch.project,
            "extraction",
            change_log=self.change_log,
            source=self.run_id,
            allowed_types=self.material_profile.allowed_types,
            allowed_evidence=self.material_profile.allowed_evidence,
            allow_created_rewrites=self.material_profile.allow_created_rewrites,
        )
        tools = self._tools(reader, writer, batch, owner, guard)
        # The factory hands this stack to exactly one close owner before any
        # further await. Cleanup drains full operations before closing it.
        try:
            client = await create_client(self.profile)
        except asyncio.CancelledError:
            owner.cancel_requested = True
            guard.termination, guard.stop_reason = "cancelled", "cancelled"
            return _settle(guard, usage, owner, ())
        except Exception:
            guard.fail("client_build_error")
            return _settle(guard, usage, owner, ())

        async def cleanup() -> None:
            await owner.drain()
            await client.aclose()

        close = OnceClose(cleanup)
        try:
            if not isinstance(client, ToolLoopLayer):
                raise MemoryOperationError("memory_invalid_input", "Extraction requires the factory tool-loop stack.")
            client.max_iterations = self.max_turns
            catalog = render_catalog(await owner.run(reader.catalog()), tuning=self.tuning) or _EMPTY_CATALOG
            review = (
                (await owner.run(load_review_snapshot(self.store, batch.project))).lists(
                    batch.sources, related_limit=self.tuning.related_entry_cap
                )
                if self.material_profile.tool_schema == "engineering"
                else None
            )
            prompt = render_prompt(
                batch,
                catalog,
                self.references,
                profile=self.material_profile,
                related_entries=review.related_entries if review is not None else "",
                withdrawn_entries=review.withdrawn_entries if review is not None else "",
            )
            agent = Agent(
                client,
                instructions=self.material_profile.instructions,
                tools=tools,
                context_providers=[],
                middleware=[ResponseValidationMiddleware(), guard, usage],
            )
            try:
                timeout = asyncio.timeout(self.timeout_seconds)
                async with timeout:
                    response = await agent.run(
                        prompt,
                        stream=False,
                        session=None,
                        options=effective_chat_options(self.profile),
                        client_kwargs={"loop_recorder": recorder},
                    )
                messages = response.messages
                if timeout.expired():
                    guard.fail("timeout")
            except ExtractionStopped:
                if timeout.expired():
                    guard.fail("timeout")
            except TimeoutError:
                guard.fail("timeout")
            except asyncio.CancelledError:
                owner.cancel_requested = True
                guard.termination, guard.stop_reason = "cancelled", "cancelled"
            except Exception:
                guard.fail("client_or_validation_error")
        except OperationNotAdmitted:
            owner.cancel_requested = True
            guard.termination, guard.stop_reason = "cancelled", "cancelled"
        except asyncio.CancelledError:
            owner.cancel_requested = True
            guard.termination, guard.stop_reason = "cancelled", "cancelled"
        except Exception:
            guard.fail("executor_setup_error")
        finally:
            try:
                await close()
            except asyncio.CancelledError:
                owner.cancel_requested = True
            except Exception:
                guard.fail("client_close_error")
        if messages is None:
            messages = recorder.loop_messages or ()
        return _settle(guard, usage, owner, messages)

    async def _note_conflict(self, request: NoteConflictRequest) -> str:
        for ref in (request.ref_a, request.ref_b):
            _, entry_id = parse_ref(ref)
            if not ENTRY_ID_RE.fullmatch(entry_id):
                raise MemoryInputRejected("memory_invalid_input", "A conflict needs two valid entry refs.")
        if request.ref_a == request.ref_b:
            raise MemoryInputRejected("memory_invalid_input", "A conflict needs two different entry refs.")
        return "Conflict noted; both entries were left unchanged."

    def _tools(
        self,
        reader: MemoryReader,
        writer: MemoryWriter,
        batch: ExtractionBatch,
        owner: OperationOwner,
        guard: ExtractionGuard,
    ) -> list[FunctionTool]:
        def bind(name: str) -> Callable[..., Awaitable[str]]:
            async def invoke(ctx: FunctionInvocationContext, **arguments: Any) -> str:
                call_id = ctx.metadata.get("call_id")
                admitted = guard.current_calls.get(call_id) if isinstance(call_id, str) else None
                identity = admitted[0] if admitted is not None else id(ctx)

                async def operation() -> str:
                    saved: SaveReceipt | None = None
                    kind: str | None = None
                    conflict = False
                    outcome: CallOutcome = "success"
                    try:
                        request = decode_request(name, arguments)
                        if isinstance(request, SearchRequest):
                            result = render_search(await reader.search(request.query, request.scope, request.cursor))
                        elif isinstance(request, ReadRequest):
                            result = render_entry(await reader.read(request.ref))
                        elif isinstance(request, SaveRequest):
                            result_saved = await writer.save(request, batch.source_index.view)
                            saved = SaveReceipt(
                                result_saved.ref,
                                result_saved.version,
                                "restored" if result_saved.restored else result_saved.status,
                            )
                            result = render_saved(result_saved)
                        elif isinstance(request, WithdrawRequest):
                            withdrawn = await writer.withdraw(
                                request.ref, request.version, request.quote, batch.source_index.view
                            )
                            saved = SaveReceipt(withdrawn.ref, withdrawn.version, "withdrawn")
                            result = render_withdrawn(withdrawn)
                        elif isinstance(request, MergeRequest):
                            merged = await writer.merge(
                                request.keep_ref, request.keep_version, request.drop_ref, request.drop_version
                            )
                            saved = SaveReceipt(merged.ref, merged.version, "merged")
                            result = render_merged(merged)
                        elif isinstance(request, NoteConflictRequest):
                            result = await self._note_conflict(request)
                            conflict = True
                        else:
                            raise MemoryOperationError("memory_invalid_input", "Unknown extraction tool.")
                    except SourceVerificationError as error:
                        if (
                            name == "memory_save"
                            and error.failures
                            and all(
                                failure.reason in ("ambiguous", "wrong_session") and failure.candidates
                                for failure in error.failures
                            )
                        ):
                            outcome = "candidate"
                            result = "Not saved. " + render_error(error, "memory_save")
                        else:
                            outcome = "rejection" if self.material_profile.allow_input_rejections else "failure"
                            kind, result = error.kind, render_error(error, name)
                    except MergePartialError as error:
                        saved = SaveReceipt(error.ref, error.version, "merged")
                        outcome, kind = "failure", error.kind
                        result = render_error(error, name)
                    except MemoryInputRejected as error:
                        outcome = "rejection" if self.material_profile.allow_input_rejections else "failure"
                        kind, result = error.kind, render_error(error, name)
                    except MemoryOperationError as error:
                        outcome, kind = "failure", error.kind
                        result = render_error(error, name)
                    except Exception:
                        outcome, kind, result = "failure", "memory_store_unavailable", _SAFE_FAILURE
                    if outcome == "rejection":
                        result = "Error: Input rejected; no changes were made. " + result
                    guard.records[identity] = CallRecord(identity, name, outcome, kind, saved, conflict)
                    metadata: dict[str, Any] = {
                        _OUTCOME_KEY: outcome,
                        _RECORD_KEY: identity,
                        TOOL_FAILED_METADATA_KEY: outcome in ("failure", "rejection"),
                    }
                    if kind is not None:
                        metadata[TOOL_ERROR_KIND_METADATA_KEY] = kind
                    ctx.metadata[TOOL_RESULT_METADATA_KEY] = metadata
                    return result

                return await owner.run(operation())

            return invoke

        return [
            FunctionTool(
                name=schema["name"],
                description=schema["description"],
                input_model=schema["parameters"],
                func=bind(schema["name"]),
            )
            for schema in extraction_tool_schemas(self.material_profile)
        ]


@dataclass(frozen=True, slots=True)
class CallObservations:
    """One interpretation of native records and their kernel results for every decision."""

    records: dict[int, CallRecord]
    unsettled: set[int]

    @property
    def has_hard_failures(self) -> bool:
        return bool(self.unsettled) or any(record.outcome == "failure" for record in self.records.values())

    @property
    def has_tool_failures(self) -> bool:
        return self.has_hard_failures or any(record.outcome == "rejection" for record in self.records.values())


def _observe_calls(guard: ExtractionGuard, messages: Sequence[Message]) -> CallObservations:
    records = dict(guard.records)
    unsettled: set[int] = set()
    seen: set[int] = set()
    accessor = LiveAccessor()
    for exchange in iter_exchanges(messages, accessor):
        paired = pair_results(messages, exchange, accessor, _PAIRING)
        for assignments in paired.truthy_assignments.values():
            for call, result in assignments:
                content = messages[call.message_index].contents[call.content_index]
                identities = [
                    identity for identity, accepted in guard.accepted if accepted is content and identity not in seen
                ]
                if not identities:
                    continue
                identity = identities[0]
                seen.add(identity)
                if result is None:
                    if identity not in records:
                        unsettled.add(identity)
                    continue
                answered = messages[result.message_index].contents[result.content_index]
                metadata = answered.additional_properties
                carried = metadata.get(TOOL_RESULT_METADATA_KEY)
                failure_metadata = carried if isinstance(carried, Mapping) else metadata
                if identity in records:
                    # A cancellation fill cannot overwrite a full native save
                    # operation that has already settled in its owned Task.
                    if (
                        not metadata.get(TOOL_INTERRUPTED_METADATA_KEY)
                        and (
                            answered.exception is not None
                            or tool_result_metadata_failure_state(failure_metadata) is True
                        )
                        and records[identity].outcome != "failure"
                    ):
                        # Only this call's own rejection result keeps the rejection; any
                        # other failed result for it fails closed as a hard failure.
                        if (
                            answered.exception is None
                            and records[identity].outcome == "rejection"
                            and failure_metadata.get(_OUTCOME_KEY) == "rejection"
                            and failure_metadata.get(_RECORD_KEY) == identity
                        ):
                            continue
                        records[identity] = CallRecord(
                            identity,
                            content.name or "",
                            "failure",
                            "kernel_result_failure",
                            records[identity].saved,
                            records[identity].conflict,
                        )
                    continue
                if answered.exception is not None or tool_result_metadata_failure_state(failure_metadata) is True:
                    kind = failure_metadata.get(TOOL_ERROR_KIND_METADATA_KEY)
                    records[identity] = CallRecord(
                        identity, content.name or "", "failure", kind if isinstance(kind, str) else "kernel_tool_error"
                    )
                else:
                    unsettled.add(identity)
    for identity, _content in guard.accepted:
        if identity not in seen and identity not in records:
            unsettled.add(identity)
    return CallObservations(records, unsettled)


def _settle(
    guard: ExtractionGuard, usage: RawUsage, owner: OperationOwner, messages: Sequence[Message]
) -> ExtractionResult:
    observed = _observe_calls(guard, messages)
    records, unsettled = observed.records, observed.unsettled
    outcomes = Counter(record.outcome for record in records.values())
    receipts = tuple(record.saved for record in records.values() if record.saved is not None)
    failure_count = outcomes["failure"] + len(guard.failures) + len(unsettled)
    failure_kinds = Counter(
        f"{record.name}:{record.kind or 'unknown'}" for record in records.values() if record.outcome == "failure"
    )
    rejection_kinds = Counter(
        f"{record.name}:{record.kind or 'unknown'}" for record in records.values() if record.outcome == "rejection"
    )
    failure_kinds.update(f"guard:{kind}" for kind in guard.failures)
    if unsettled:
        failure_kinds["executor:unsettled"] += len(unsettled)
    termination = guard.termination
    reason = guard.stop_reason
    if guard.stop_reason == "timeout":
        termination, reason = "failed", "timeout"
    elif owner.cancel_requested:
        termination, reason = "cancelled", "saved_then_cancelled" if receipts else "cancelled"
    elif failure_count:
        termination = "failed"
        if reason is None:
            reason = "unseen_call" if unsettled else "tool_failure"
    elif termination == "completed" and not records:
        termination, reason, failure_count = "failed", "zero_calls", 1
        failure_kinds["guard:zero_calls"] += 1
    return ExtractionResult(
        termination=termination,
        call_count=len(records) + len(unsettled),
        failure_count=failure_count,
        candidate_count=outcomes["candidate"],
        unsettled_count=len(unsettled) + owner.pending_count,
        stop_reason=reason,
        usage=usage.summary(),
        success_count=outcomes["success"],
        created_count=sum(receipt.outcome == "created" for receipt in receipts),
        updated_count=sum(receipt.outcome == "updated" for receipt in receipts),
        existing_count=sum(receipt.outcome == "existing" for receipt in receipts),
        restored_count=sum(receipt.outcome == "restored" for receipt in receipts),
        withdrawn_count=sum(receipt.outcome == "withdrawn" for receipt in receipts),
        merged_count=sum(receipt.outcome == "merged" for receipt in receipts),
        conflict_count=sum(record.conflict for record in records.values()),
        request_count=len(usage.observations),
        turn_count=guard.requests,
        tail_seen=guard.tail_seen,
        unsupported_raw_output=tuple(sorted(guard.unsupported.items())),
        failure_kinds=tuple(sorted(failure_kinds.items())),
        rejection_count=outcomes["rejection"],
        rejection_kinds=tuple(sorted(rejection_kinds.items())),
        cancel_requested=owner.cancel_requested and reason != "timeout",
        saved=receipts,
        first_input_tokens=_first_input_tokens(usage),
    )


def _first_input_tokens(usage: RawUsage) -> int | float | None:
    if not usage.observations or usage.observations[0] is None:
        return None
    value = usage.observations[0].get("input_token_count")
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None
