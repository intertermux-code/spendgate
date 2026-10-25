"""Translation between OpenAI's chat shape and Anthropic's Messages API.

spendgate's public surface stays OpenAI-compatible all the way down: an
Anthropic upstream is a dialect, not a second API. Requests leave translated,
answers come back translated, and the tokens Anthropic reports are metered into
the same ledger columns as everyone else's.

Scope is text. Anything that cannot be expressed honestly — tool definitions,
tool results, image parts — raises ``UnsupportedRequest`` so the caller gets a
400 instead of a silently degraded answer.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .upstream import Usage

DEFAULT_MAX_TOKENS = 4096

# OpenAI roles that carry instructions rather than a conversational turn.
SYSTEM_ROLES = ("system", "developer")

# Keys that would ask the model to call something back.
TOOL_KEYS = ("tools", "tool_choice", "functions", "function_call")

FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


class UnsupportedRequest(Exception):
    """A request this translator will not fake an answer for."""


def to_request(body: dict[str, Any]) -> dict[str, Any]:
    """An OpenAI chat-completions body as an Anthropic Messages request."""
    for key in TOOL_KEYS:
        if body.get(key):
            raise UnsupportedRequest(
                f"Anthropic upstreams are proxied for text only: '{key}' is not supported"
            )
    n = body.get("n")
    if isinstance(n, int) and not isinstance(n, bool) and n > 1:
        raise UnsupportedRequest(
            "Anthropic upstreams answer once per request: 'n' greater than 1 is not supported"
        )

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise UnsupportedRequest("'messages' must be a non-empty list")

    system_parts: list[str] = []
    turns: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise UnsupportedRequest("every message must be an object")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise UnsupportedRequest("every message needs a 'role'")
        if message.get("tool_calls") or message.get("function_call"):
            raise UnsupportedRequest(
                "Anthropic upstreams are proxied for text only: tool calls in 'messages' are not supported"
            )
        if role in ("tool", "function"):
            raise UnsupportedRequest(
                "Anthropic upstreams are proxied for text only: tool results in 'messages' are not supported"
            )
        text = _text_of(message.get("content"))
        if role in SYSTEM_ROLES:
            system_parts.append(text)
        elif role in ("user", "assistant"):
            turns.append({"role": role, "text": text})
        else:
            raise UnsupportedRequest(f"message role {role!r} cannot be translated")

    if not turns:
        raise UnsupportedRequest("expected at least one 'user' or 'assistant' message")

    # Anthropic wants alternating roles; same-role neighbours become one turn
    # with one text block each rather than a request the API would reject.
    merged: list[dict[str, Any]] = []
    for turn in turns:
        if not turn["text"]:
            # Anthropic rejects empty text blocks, and a turn that carries no
            # text is a turn this translation cannot represent.
            raise UnsupportedRequest("message 'content' must not be empty")
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1]["content"].append({"type": "text", "text": turn["text"]})
        else:
            merged.append({"role": turn["role"], "content": [{"type": "text", "text": turn["text"]}]})

    out: dict[str, Any] = {
        "model": body["model"],
        "messages": merged,
        "max_tokens": _positive_int(body.get("max_tokens"), DEFAULT_MAX_TOKENS),
    }
    system = "\n\n".join(part for part in system_parts if part)
    if system:
        out["system"] = system
    for key in ("temperature", "top_p"):
        value = body.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = float(value)
    stop = body.get("stop")
    if isinstance(stop, str) and stop:
        out["stop_sequences"] = [stop]
    elif isinstance(stop, list):
        sequences = [entry for entry in stop if isinstance(entry, str) and entry]
        if sequences:
            out["stop_sequences"] = sequences
    if body.get("stream") is True:
        out["stream"] = True
    return out


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                pieces.append(str(part.get("text", "")))
            elif isinstance(part, dict) and part.get("type") == "input_audio":
                raise UnsupportedRequest(
                    "Anthropic upstreams are proxied for text only: audio parts are not supported"
                )
            else:
                raise UnsupportedRequest(
                    "Anthropic upstreams are proxied for text only: "
                    "only {\"type\": \"text\"} content parts are supported"
                )
        return "".join(pieces)
    if content is None:
        return ""
    raise UnsupportedRequest("message 'content' must be a string or a list of text parts")


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value if value > 0 else default


def usage_of(payload: Any) -> Usage | None:
    """Anthropic's input_tokens/output_tokens in the ledger's own terms."""
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = _count(usage.get("input_tokens"))
    completion = _count(usage.get("output_tokens"))
    if prompt is None and completion is None:
        return None
    model = payload.get("model")
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        model=model if isinstance(model, str) and model else None,
    )


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def to_completion(payload: dict[str, Any]) -> dict[str, Any]:
    """A Messages response as a chat.completion, usage included."""
    blocks = payload.get("content")
    text = "".join(
        str(block.get("text", ""))
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ) if isinstance(blocks, list) else ""
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    prompt = _count(usage.get("input_tokens"))
    completion = _count(usage.get("output_tokens"))
    return {
        "id": payload.get("id") or "chatcmpl-anthropic",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model") or "",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": FINISH_REASONS.get(str(payload.get("stop_reason")), "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt or 0,
            "completion_tokens": completion or 0,
            "total_tokens": (prompt or 0) + (completion or 0),
        },
    }


def to_error(raw: bytes, status: int) -> bytes | None:
    """An Anthropic error body in the shape OpenAI clients display. None if the
    body is not a recognisable Anthropic error, so odd bytes still get relayed."""
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("type") != "error":
        return None
    inner = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    message = str(inner.get("message") or "Upstream returned an error")
    err_type = str(inner.get("type") or "api_error")
    envelope = {
        "error": {
            "message": message,
            "type": err_type,
            "param": None,
            "code": err_type if status >= 400 else "upstream_error",
        }
    }
    return json.dumps(envelope, separators=(",", ":")).encode("utf-8")


def _event_bytes(payload: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n\n"


class StreamTranslator:
    """Anthropic SSE in, OpenAI chunk SSE out, one socket block at a time.

    Fed raw bytes as they arrive; every call returns whatever is translatable so
    far, which is what keeps the relay incremental. Usage is read out of the
    events on the side and lands in ``usage`` once the stream has reported it.
    """

    def __init__(self, model_hint: str = "") -> None:
        self.usage: Usage | None = None
        self._model = model_hint or ""
        self._id = ""
        self._created = int(time.time())
        self._buffer = b""
        self._event_name = ""
        self._data: list[str] = []
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._stop_reason: str | None = None
        self.finished = False

    def feed(self, block: bytes) -> bytes:
        out = bytearray()
        lines = (self._buffer + block).split(b"\n")
        self._buffer = lines.pop()
        for line in lines:
            if line.endswith(b"\r"):
                line = line[:-1]
            if not line.strip():
                out += self._dispatch()
                continue
            if line.startswith(b"data:"):
                self._data.append(line[len(b"data:"):].decode("utf-8", "replace").lstrip())
            elif line.startswith(b"event:"):
                self._event_name = line[len(b"event:"):].decode("utf-8", "replace").strip()
        return bytes(out)

    def close(self) -> bytes:
        """Flush a trailing event that arrived without its blank separator."""
        self._buffer = b""
        return self._dispatch() if self._data else b""

    def _dispatch(self) -> bytes:
        name = self._event_name
        if not self._data:
            return b""
        raw = "\n".join(self._data)
        self._data = []
        self._event_name = ""
        try:
            payload = json.loads(raw)
        except ValueError:
            return b""
        if not isinstance(payload, dict):
            return b""
        kind = str(payload.get("type") or name)
        handler = {
            "message_start": self._message_start,
            "content_block_delta": self._content_block_delta,
            "message_delta": self._message_delta,
            "message_stop": self._message_stop,
        }.get(kind)
        return handler(payload) if handler else b""

    def _chunk(self, delta: dict[str, Any], finish_reason: str | None = None, with_usage: bool = False) -> bytes:
        obj: dict[str, Any] = {
            "id": self._id or "chatcmpl-anthropic",
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self._model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if with_usage:
            obj["usage"] = {
                "prompt_tokens": self._input_tokens or 0,
                "completion_tokens": self._output_tokens or 0,
                "total_tokens": (self._input_tokens or 0) + (self._output_tokens or 0),
            }
        return _event_bytes(obj)

    def _message_start(self, payload: dict[str, Any]) -> bytes:
        message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
        self._id = str(message.get("id") or self._id)
        model = message.get("model")
        if isinstance(model, str) and model:
            self._model = model
        usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
        self._input_tokens = _count(usage.get("input_tokens"))
        self._output_tokens = _count(usage.get("output_tokens")) or self._output_tokens
        # OpenAI streams open with a role-only chunk; mirror it so clients that
        # wait for the delta shape before rendering see the same thing.
        return self._chunk({"role": "assistant", "content": ""})

    def _content_block_delta(self, payload: dict[str, Any]) -> bytes:
        delta = payload.get("delta") if isinstance(payload.get("delta"), dict) else {}
        if delta.get("type") != "text_delta":
            return b""
        return self._chunk({"content": str(delta.get("text", ""))})

    def _message_delta(self, payload: dict[str, Any]) -> bytes:
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        tokens = _count(usage.get("output_tokens"))
        if tokens is not None:
            self._output_tokens = tokens
        delta = payload.get("delta") if isinstance(payload.get("delta"), dict) else {}
        if isinstance(delta.get("stop_reason"), str):
            self._stop_reason = delta["stop_reason"]
        self._capture_usage()
        return b""

    def _message_stop(self, payload: dict[str, Any]) -> bytes:
        self._capture_usage()
        self.finished = True
        reason = FINISH_REASONS.get(str(self._stop_reason), "stop")
        return self._chunk({}, finish_reason=reason, with_usage=True) + b"data: [DONE]\n\n"

    def _capture_usage(self) -> None:
        if self._input_tokens is None and self._output_tokens is None:
            return
        self.usage = Usage(
            prompt_tokens=self._input_tokens,
            completion_tokens=self._output_tokens,
            model=self._model or None,
        )
