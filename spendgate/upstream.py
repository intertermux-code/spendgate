"""Upstream request construction and usage extraction.

The proxy speaks HTTP with the stdlib only (``urllib.request``), so the pieces
that are easy to get wrong — streaming usage, SSE framing, which model name to
bill, which auth header a provider wants — live here where they can be tested
without a socket.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any

from . import __version__
from .config import ANTHROPIC_KIND, UpstreamConfig

UPSTREAM_SUFFIX = "chat/completions"
ANTHROPIC_SUFFIX = "messages"

# The documented version header for Anthropic's Messages API. Requests without
# it are rejected outright, and the client never sees the value from here.
ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class Usage:
    prompt_tokens: int | None
    completion_tokens: int | None
    model: str | None


def build_request(upstream: UpstreamConfig, payload: bytes, stream: bool) -> urllib.request.Request:
    """Build the forwarded request. The upstream key is attached here only."""
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(payload)),
        "Accept": "text/event-stream" if stream else "application/json",
        "User-Agent": f"spendgate/{__version__}",
    }
    if upstream.kind == ANTHROPIC_KIND:
        # Anthropic authenticates with x-api-key and no bearer scheme.
        headers["x-api-key"] = upstream.api_key
        headers["anthropic-version"] = ANTHROPIC_VERSION
        suffix = ANTHROPIC_SUFFIX
    else:
        headers["Authorization"] = f"Bearer {upstream.api_key}"
        suffix = UPSTREAM_SUFFIX
    return urllib.request.Request(upstream.url + suffix, data=payload, headers=headers, method="POST")


def inject_include_usage(payload: bytes) -> bytes:
    """Ask a streaming upstream to report usage in its final chunk.

    Only when the client did not set ``stream_options`` itself: a client that
    already chose its own options knows what it wants.
    """
    body = json.loads(payload)
    if isinstance(body, dict) and body.get("stream") is True and "stream_options" not in body:
        body["stream_options"] = {"include_usage": True}
        return json.dumps(body, separators=(",", ":")).encode("utf-8")
    return payload


def usage_from_json(obj: Any) -> Usage | None:
    """Read a ``usage`` object out of a parsed chat completion response."""
    if not isinstance(obj, dict):
        return None
    usage = obj.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = _token_count(usage.get("prompt_tokens"))
    completion = _token_count(usage.get("completion_tokens"))
    if prompt is None and completion is None:
        return None
    model = obj.get("model")
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        model=model if isinstance(model, str) and model else None,
    )


def _token_count(value: Any) -> int | None:
    # bool is an int subclass but is never a token count.
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


class SSEUsageScanner:
    """Incremental SSE reader that finds the usage chunk without buffering.

    Fed raw bytes as they arrive from upstream. It keeps at most one partial
    line, never the stream, so a long completion costs a bounded amount of
    memory while still being relayed to the client untouched.
    """

    def __init__(self) -> None:
        self._partial = b""
        self.usage: Usage | None = None
        self.model: str | None = None

    def feed(self, block: bytes) -> None:
        data = self._partial + block
        *lines, self._partial = data.split(b"\n")
        for line in lines:
            self._line(line)

    def close(self) -> None:
        if self._partial:
            self._line(self._partial)
            self._partial = b""

    def _line(self, line: bytes) -> None:
        line = line.strip()
        if not line.startswith(b"data:"):
            return
        payload = line[len(b"data:"):].strip()
        if not payload or payload == b"[DONE]":
            return
        try:
            obj = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            return
        if isinstance(obj, dict) and isinstance(obj.get("model"), str):
            self.model = self.model or obj["model"]
        found = usage_from_json(obj)
        if found is not None:
            # Later chunks win: include_usage puts usage in the final one.
            self.usage = found
