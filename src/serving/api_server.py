"""
API Server with SSE Streaming.

Implements the Messages API endpoint compatible with the Anthropic API spec:

    POST /v1/messages
    Content-Type: application/json

    {
        "model": "claude-opus-4-6-20260301",
        "max_tokens": 16384,
        "messages": [{"role": "user", "content": "..."}],
        "stream": true,
        "thinking": {"type": "enabled", "budget_tokens": 10000}
    }

SSE (Server-Sent Events) streaming format:
    event: message_start
    data: {"type": "message_start", "message": {...}}

    event: content_block_start
    data: {"type": "content_block_start", "index": 0, ...}

    event: content_block_delta
    data: {"type": "content_block_delta", "index": 0, "delta": {"text": "..."}}

    event: content_block_stop
    data: {"type": "content_block_stop", "index": 0}

    event: message_delta
    data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, ...}

    event: message_stop
    data: {"type": "message_stop"}

Rate limits (estimated):
    Free tier    :    5 RPM,   20K TPM,    1 concurrent
    Pro          :   50 RPM,  100K TPM,    5 concurrent
    Team         :  100 RPM,  400K TPM,   10 concurrent
    Enterprise   : 4000 RPM, 2000K TPM,  256 concurrent

Pricing (per million tokens, March 2026):
    Model                    Input    Output   Batch-Input  Batch-Output
    opus-4-6-20260301        $5.00    $25.00   $2.50        $12.50
    opus-4-6 + thinking      $5.00    $25.00   (same)
    opus-4-6 prompt caching  $0.50/w  $5.00/r  N/A          N/A

Beta headers:
    anthropic-beta: prompt-caching-2024-07-31
    anthropic-beta: computer-use-2025-01-24
    anthropic-beta: extended-thinking-2025-01-24

References:
    - Anthropic API reference: docs.anthropic.com
    - SSE spec: html.spec.whatwg.org/multipage/server-sent-events.html
"""

import json
import time
import uuid
import logging
import asyncio
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, AsyncIterator
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_VERSION = "2024-01-01"
MODEL_ID = "claude-opus-4-6-20260301"
MAX_CONTEXT_TOKENS = 1_000_000
MAX_OUTPUT_TOKENS = 128_000


# ---------------------------------------------------------------------------
# API Data Models
# ---------------------------------------------------------------------------

class StopReason(str, Enum):
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"


class ContentBlockType(str, Enum):
    TEXT = "text"
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    IMAGE = "image"


@dataclass
class Usage:
    """Token usage for billing."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        """Compute cost in USD at standard pricing."""
        input_cost = self.input_tokens * 5.0 / 1_000_000
        output_cost = self.output_tokens * 25.0 / 1_000_000
        cache_write = self.cache_creation_input_tokens * 0.50 / 1_000_000
        cache_read = self.cache_read_input_tokens * 5.0 / 1_000_000
        return input_cost + output_cost + cache_write + cache_read


@dataclass
class ContentBlock:
    """A content block in the response."""
    type: str = "text"
    text: Optional[str] = None
    thinking: Optional[str] = None
    id: Optional[str] = None       # for tool_use
    name: Optional[str] = None     # for tool_use
    input: Optional[Dict] = None   # for tool_use

    def to_dict(self) -> Dict[str, Any]:
        d = {"type": self.type}
        if self.type == "text":
            d["text"] = self.text or ""
        elif self.type == "thinking":
            d["thinking"] = self.thinking or ""
        elif self.type == "tool_use":
            d["id"] = self.id
            d["name"] = self.name
            d["input"] = self.input or {}
        return d


@dataclass
class Message:
    """Complete API response message."""
    id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:24]}")
    type: str = "message"
    role: str = "assistant"
    content: List[ContentBlock] = field(default_factory=list)
    model: str = MODEL_ID
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "role": self.role,
            "content": [b.to_dict() for b in self.content],
            "model": self.model,
            "stop_reason": self.stop_reason,
            "stop_sequence": self.stop_sequence,
            "usage": asdict(self.usage),
        }


# ---------------------------------------------------------------------------
# SSE Event Builder
# ---------------------------------------------------------------------------

class SSEEventBuilder:
    """
    Build Server-Sent Events for streaming responses.

    SSE format:
        event: {event_type}\n
        data: {json_data}\n\n
    """

    @staticmethod
    def format_event(event_type: str, data: Dict[str, Any]) -> str:
        """Format a single SSE event."""
        json_data = json.dumps(data, ensure_ascii=False)
        return f"event: {event_type}\ndata: {json_data}\n\n"

    @staticmethod
    def message_start(message: Message) -> str:
        return SSEEventBuilder.format_event("message_start", {
            "type": "message_start",
            "message": {
                "id": message.id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": message.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": message.usage.input_tokens, "output_tokens": 0},
            },
        })

    @staticmethod
    def content_block_start(index: int, block: ContentBlock) -> str:
        data = {
            "type": "content_block_start",
            "index": index,
            "content_block": block.to_dict(),
        }
        return SSEEventBuilder.format_event("content_block_start", data)

    @staticmethod
    def text_delta(index: int, text: str) -> str:
        return SSEEventBuilder.format_event("content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        })

    @staticmethod
    def thinking_delta(index: int, thinking: str) -> str:
        return SSEEventBuilder.format_event("content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "thinking_delta", "thinking": thinking},
        })

    @staticmethod
    def content_block_stop(index: int) -> str:
        return SSEEventBuilder.format_event("content_block_stop", {
            "type": "content_block_stop",
            "index": index,
        })

    @staticmethod
    def message_delta(
        stop_reason: str,
        output_tokens: int,
    ) -> str:
        return SSEEventBuilder.format_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        })

    @staticmethod
    def message_stop() -> str:
        return SSEEventBuilder.format_event("message_stop", {
            "type": "message_stop",
        })

    @staticmethod
    def ping() -> str:
        return SSEEventBuilder.format_event("ping", {"type": "ping"})


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Token-bucket rate limiter for API requests.

    Tracks:
        - Requests per minute (RPM)
        - Tokens per minute (TPM)
        - Concurrent requests
    """

    def __init__(
        self,
        rpm: int = 50,
        tpm: int = 100_000,
        max_concurrent: int = 5,
    ):
        self.rpm = rpm
        self.tpm = tpm
        self.max_concurrent = max_concurrent

        self.request_timestamps: List[float] = []
        self.token_usage_window: List[tuple] = []  # (timestamp, token_count)
        self.current_concurrent = 0

    def check(self, estimated_tokens: int = 1000) -> tuple:
        """
        Check if a request is allowed.

        Returns: (allowed: bool, retry_after_ms: Optional[int])
        """
        now = time.time()
        window = 60.0  # 1 minute window

        # Clean old entries
        self.request_timestamps = [
            t for t in self.request_timestamps if now - t < window
        ]
        self.token_usage_window = [
            (t, c) for t, c in self.token_usage_window if now - t < window
        ]

        # Check RPM
        if len(self.request_timestamps) >= self.rpm:
            oldest = self.request_timestamps[0]
            retry_ms = int((oldest + window - now) * 1000)
            return False, retry_ms

        # Check TPM
        current_tpm = sum(c for _, c in self.token_usage_window)
        if current_tpm + estimated_tokens > self.tpm:
            oldest = self.token_usage_window[0][0] if self.token_usage_window else now
            retry_ms = int((oldest + window - now) * 1000)
            return False, retry_ms

        # Check concurrent
        if self.current_concurrent >= self.max_concurrent:
            return False, 1000  # retry in 1s

        return True, None

    def acquire(self, estimated_tokens: int = 1000) -> None:
        """Record a request acquisition."""
        self.request_timestamps.append(time.time())
        self.token_usage_window.append((time.time(), estimated_tokens))
        self.current_concurrent += 1

    def release(self, actual_tokens: int = 0) -> None:
        """Release a request slot."""
        self.current_concurrent = max(0, self.current_concurrent - 1)


# ---------------------------------------------------------------------------
# Request Handler
# ---------------------------------------------------------------------------

@dataclass
class APIRequest:
    """Parsed API request."""
    model: str = MODEL_ID
    max_tokens: int = 4096
    messages: List[Dict[str, Any]] = field(default_factory=list)
    system: Optional[str] = None
    stream: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop_sequences: List[str] = field(default_factory=list)
    # Thinking mode
    thinking_enabled: bool = False
    thinking_budget: int = 0
    # Tool use
    tools: List[Dict] = field(default_factory=list)
    # Prompt caching
    cache_control: Optional[Dict] = None
    # Beta features
    betas: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIRequest":
        """Parse from JSON request body."""
        thinking = data.get("thinking", {})
        return cls(
            model=data.get("model", MODEL_ID),
            max_tokens=data.get("max_tokens", 4096),
            messages=data.get("messages", []),
            system=data.get("system"),
            stream=data.get("stream", False),
            temperature=data.get("temperature", 1.0),
            top_p=data.get("top_p", 1.0),
            top_k=data.get("top_k", -1),
            stop_sequences=data.get("stop_sequences", []),
            thinking_enabled=thinking.get("type") == "enabled",
            thinking_budget=thinking.get("budget_tokens", 0),
            tools=data.get("tools", []),
            cache_control=data.get("cache_control"),
            betas=data.get("betas", []),
        )


class APIServer:
    """
    Main API server handling /v1/messages endpoint.

    In production, this wraps a FastAPI/Starlette server.
    Here we implement the core logic independently of framework.
    """

    def __init__(
        self,
        model_fn=None,       # inference function
        tokenizer=None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.model_fn = model_fn
        self.tokenizer = tokenizer
        self.rate_limiter = rate_limiter or RateLimiter()
        self.sse = SSEEventBuilder()

    def validate_request(self, request: APIRequest) -> Optional[Dict]:
        """
        Validate API request. Returns error dict or None if valid.
        """
        # Model check
        if request.model != MODEL_ID:
            return {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": f"Unknown model: {request.model}",
                },
            }

        # Max tokens check
        if request.max_tokens < 1 or request.max_tokens > MAX_OUTPUT_TOKENS:
            return {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": (
                        f"max_tokens must be between 1 and {MAX_OUTPUT_TOKENS}, "
                        f"got {request.max_tokens}"
                    ),
                },
            }

        # Messages check
        if not request.messages:
            return {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "messages must not be empty",
                },
            }

        # Thinking budget check
        if request.thinking_enabled and request.thinking_budget < 1024:
            return {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": (
                        "thinking.budget_tokens must be >= 1024 "
                        "when thinking is enabled"
                    ),
                },
            }

        return None

    async def handle_request(
        self, request_data: Dict[str, Any]
    ) -> Any:
        """
        Handle a /v1/messages request.

        Returns either a Message dict (non-streaming) or an
        async generator of SSE events (streaming).
        """
        request = APIRequest.from_dict(request_data)

        # Validate
        error = self.validate_request(request)
        if error:
            return error

        # Rate limit check
        allowed, retry_after = self.rate_limiter.check()
        if not allowed:
            return {
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": f"Rate limited. Retry after {retry_after}ms",
                },
            }

        self.rate_limiter.acquire()

        try:
            if request.stream:
                return self._stream_response(request)
            else:
                return await self._complete_response(request)
        finally:
            self.rate_limiter.release()

    async def _complete_response(
        self, request: APIRequest
    ) -> Dict[str, Any]:
        """Generate a complete (non-streaming) response."""
        message = Message()
        message.usage.input_tokens = self._count_input_tokens(request)

        # Generate thinking block if enabled
        if request.thinking_enabled:
            thinking_text = await self._generate_thinking(request)
            message.content.append(ContentBlock(
                type="thinking",
                thinking=thinking_text,
            ))
            message.usage.output_tokens += len(
                self.tokenizer.encode(thinking_text)
            ) if self.tokenizer else len(thinking_text.split())

        # Generate response text
        response_text = await self._generate_response(request)
        message.content.append(ContentBlock(
            type="text",
            text=response_text,
        ))
        message.usage.output_tokens += len(
            self.tokenizer.encode(response_text)
        ) if self.tokenizer else len(response_text.split())

        message.stop_reason = StopReason.END_TURN.value

        return message.to_dict()

    async def _stream_response(
        self, request: APIRequest
    ) -> AsyncIterator[str]:
        """Generate a streaming SSE response."""
        message = Message()
        message.usage.input_tokens = self._count_input_tokens(request)

        # message_start
        yield self.sse.message_start(message)

        block_index = 0
        total_output_tokens = 0

        # Thinking block (if enabled)
        if request.thinking_enabled:
            thinking_block = ContentBlock(type="thinking", thinking="")
            yield self.sse.content_block_start(block_index, thinking_block)

            async for chunk in self._stream_thinking(request):
                yield self.sse.thinking_delta(block_index, chunk)
                total_output_tokens += 1  # approximate

            yield self.sse.content_block_stop(block_index)
            block_index += 1

        # Text block
        text_block = ContentBlock(type="text", text="")
        yield self.sse.content_block_start(block_index, text_block)

        async for chunk in self._stream_text(request):
            yield self.sse.text_delta(block_index, chunk)
            total_output_tokens += 1

        yield self.sse.content_block_stop(block_index)

        # message_delta + message_stop
        yield self.sse.message_delta(
            StopReason.END_TURN.value, total_output_tokens
        )
        yield self.sse.message_stop()

    def _count_input_tokens(self, request: APIRequest) -> int:
        """Count input tokens from request messages."""
        total = 0
        if request.system:
            total += len(self.tokenizer.encode(request.system)) if self.tokenizer else len(request.system.split())
        for msg in request.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(self.tokenizer.encode(content)) if self.tokenizer else len(content.split())
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        total += len(self.tokenizer.encode(text)) if self.tokenizer else len(text.split())
                    elif block.get("type") == "image":
                        total += 2304  # default visual tokens at 672×672
        return total

    async def _generate_thinking(self, request: APIRequest) -> str:
        """Generate thinking tokens. Placeholder for model integration."""
        return "[Thinking placeholder — integrate with ThinkingModeEngine]"

    async def _generate_response(self, request: APIRequest) -> str:
        """Generate response text. Placeholder for model integration."""
        return "[Response placeholder — integrate with FastModeEngine]"

    async def _stream_thinking(
        self, request: APIRequest
    ) -> AsyncIterator[str]:
        """Stream thinking tokens. Placeholder."""
        chunks = ["[Thinking", " placeholder", " — integrate", " with ThinkingModeEngine]"]
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.01)

    async def _stream_text(
        self, request: APIRequest
    ) -> AsyncIterator[str]:
        """Stream response text. Placeholder."""
        chunks = ["[Response", " placeholder", " — integrate", " with FastModeEngine]"]
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.01)
