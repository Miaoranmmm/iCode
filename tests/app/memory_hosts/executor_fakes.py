# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Synthetic SDK Responses crossing the actual decoder and kernel stack."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from openai.types.responses import Response, ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

from chrys.kernel import ChatMiddlewareLayer, ChatResponse, Message, ToolLoopLayer
from chrys.service.llm.anthropic_messages.decode import decode_message
from chrys.service.llm.openai_responses.client import DEEPSEEK_RESPONSES
from chrys.service.llm.openai_responses.decode import decode_response
from chrys.service.memory.change_log import MemoryChange
from chrys.service.memory.extraction import ExtractionBatch, SessionKey, SourceIndex, SourceRevision, SourceSession
from chrys.service.memory.storage.store import MemoryScope
from chrys.service.memory.verification import ConversationView, SourceMessage
from chrys.service.profiles.models.schema import ModelProfile
from tests.support.transcript_invariants import InvariantCheckedToolLoopLayer

if TYPE_CHECKING:
    from chrys.app.memory_hosts.executor_icode import PreparedProfile

SESSION = "00000000-0000-4000-8000-000000000001"
PROJECT = MemoryScope("project", "0123456789abcdef")
RUN_ID = "extract-" + "a" * 32
WORDS = "Keep changes small and explain their scope."


def profile(**changes: Any) -> ModelProfile:
    return ModelProfile(
        **{
            "id": "test-responses",
            "name": "Synthetic Responses",
            "provider": "deepseek-openai",
            "api_style": "responses",
            "model_id": "synthetic-model",
            "max_context_tokens": 200_000,
            "max_output_tokens": 64_000,
            "chat_options": '{"reasoning":{"effort":"high"}}',
            **changes,
        }
    )


def messages_profile(**changes: Any) -> ModelProfile:
    return profile(
        **{
            "provider": "anthropic",
            "api_style": "chat_completions",
            "chat_options": '{"extra_body":{"cache_control":{"type":"ephemeral"}}}',
            **changes,
        }
    )


def prepared(**changes: Any) -> PreparedProfile:
    from chrys.app.memory_hosts.executor_icode import prepare_profile

    return prepare_profile(profile(**changes))


def messages_call(name: str, arguments: object, call_id: object = "call-one") -> BetaToolUseBlock:
    return BetaToolUseBlock.model_construct(type="tool_use", id=call_id, name=name, input=arguments)


def messages_response(
    *items: Any, reason: str | None = "end_turn", text: str | None = None, usage: Any = None
) -> BetaMessage:
    if text is not None:
        items = (*items, BetaTextBlock.model_construct(type="text", text=text, citations=None))
    return BetaMessage.model_construct(
        type="message",
        id="synthetic-message",
        model="synthetic-model",
        role="assistant",
        content=list(items),
        stop_reason=reason,
        stop_sequence=None,
        usage=usage,
    )


def batch(*, messages: tuple[SourceMessage, ...] | None = None) -> ExtractionBatch:
    if messages is None:
        messages = (SourceMessage(SESSION, 1, (WORDS,), item="user-one", at="2026-01-01T00:00:00Z"),)
    source = SourceSession(SessionKey("claude-code", SESSION), PROJECT, SourceRevision(1, "a" * 64), messages)
    return ExtractionBatch(PROJECT, (source,), "synthetic material", SourceIndex(ConversationView(messages, None)))


def call(name: str, arguments: object, call_id: object = "call-one") -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall.model_construct(
        id="function-item",
        type="function_call",
        status="completed",
        name=name,
        arguments=json.dumps(arguments),
        call_id=call_id,
    )


def response(
    *items: Any, status: str = "completed", reason: str | None = None, text: str | None = None, usage: Any = None
) -> Response:
    if text is not None:
        items = (
            *items,
            ResponseOutputMessage.model_construct(
                id="message-item",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText.model_construct(type="output_text", text=text, annotations=[], logprobs=[])
                ],
            ),
        )
    details = None
    if reason is not None:
        from openai.types.responses.response import IncompleteDetails

        details = IncompleteDetails.model_construct(reason=reason)
    return Response.model_construct(
        id="response-item",
        created_at=0,
        model="synthetic-model",
        object="response",
        output=list(items),
        status=status,
        incomplete_details=details,
        metadata={},
        usage=usage,
        conversation=None,
        error=None,
    )


def save_arguments(**changes: object) -> dict[str, object]:
    return {
        "type": "intent",
        "title": "Keep changes small",
        "body": "Keep changes small and explain their scope.",
        "user_quotes": [{"text": WORDS, "session": SESSION}],
        **changes,
    }


ScriptStep = Response | BetaMessage | BaseException | Callable[[Sequence[Message]], Awaitable[Response | BetaMessage]]


class ScriptedWire:
    """Only the wire is a double; every returned SDK object is really decoded."""

    def __init__(
        self, script: Sequence[ScriptStep], *, decoded_hook: Callable[[ChatResponse], None] | None = None
    ) -> None:
        self.script = list(script)
        self.decoded_hook = decoded_hook
        self.requests: list[tuple[Sequence[Message], Mapping[str, Any]]] = []
        self.closed = 0
        self.model = "synthetic-model"
        self.FORCES_STATELESS = True

    async def aclose(self) -> None:
        self.closed += 1

    async def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        options: Mapping[str, Any] | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        compaction_strategy: object = None,
        tokenizer: object = None,
    ) -> ChatResponse:
        assert stream is False
        self.requests.append((messages, dict(options or {})))
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        raw = await step(messages) if callable(step) else step
        decoded = (
            decode_message(raw, response_format=None)
            if isinstance(raw, BetaMessage)
            else decode_response(raw, options or {}, variant=DEEPSEEK_RESPONSES)
        )
        if self.decoded_hook is not None:
            self.decoded_hook(decoded)
        return decoded


class ClientFactory:
    def __init__(self, wire: ScriptedWire) -> None:
        self.wire = wire
        self.profiles: list[ModelProfile] = []
        self.client: ToolLoopLayer | None = None

    async def __call__(
        self,
        profile: ModelProfile,
        on_intermediate_text: Callable[[str], None] | None = None,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        use_route_session_context: bool = False,
        session_dir: Any = None,
        tool_result_ceiling_tokens: int | None = None,
    ) -> ToolLoopLayer:
        self.profiles.append(profile)
        self.client = InvariantCheckedToolLoopLayer(
            ChatMiddlewareLayer(self.wire), max_iterations=7777, max_consecutive_errors=10
        )
        return self.client


@dataclass
class RecordingLog:
    changes: list[MemoryChange] = field(default_factory=list)

    async def append(self, change: MemoryChange) -> None:
        self.changes.append(change)
