"""Shared test scaffolding: a mock OpenAI-compatible upstream and a live proxy.

The mock upstream is deliberately able to stall *mid-stream*. That is the only
honest way to prove the proxy relays SSE incrementally instead of buffering: a
buffering proxy deadlocks against it.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from pathlib import Path

from spendgate.config import Config, KeyConfig
from spendgate.ledger import Ledger
from spendgate.proxy import Gate, ProxyServer

UPSTREAM_KEY = "sk-upstream-secret-DO-NOT-LEAK"


def sse_chunk(text: str, model: str, finish: bool = False) -> str:
    obj = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": "stop" if finish else None}],
    }
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"


def sse_usage(prompt: int, completion: int, model: str) -> str:
    obj = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
    }
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"


def completion_json(model: str, prompt: int, completion: int, content: str = "hello") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
    }


class _MockHandler(BaseHTTPRequestHandler):
    server: "MockUpstream"
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args) -> None:  # silence the test noise
        pass

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.loads(raw)
        self.server.received.append({"body": body, "authorization": self.headers.get("Authorization", "")})

        if body.get("stream"):
            self._stream()
        else:
            payload = json.dumps(self.server.json_response).encode()
            self.send_response(self.server.json_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        for index, block in enumerate(self.server.stream_blocks):
            self.wfile.write(block.encode())
            self.wfile.flush()
            if index == self.server.pause_after_chunk:
                # Hold the rest of the stream until the *test* proves the proxy
                # already handed these bytes on to someone.
                self.server.paused.set()
                if not self.server.client_ack.wait(timeout=self.server.pause_timeout):
                    self.server.pause_expired = True
        self.server.streamed.set()


class MockUpstream(ThreadingHTTPServer):
    """Minimal /v1/chat/completions stand-in with scripted bodies."""

    daemon_threads = True

    def __init__(self, stream_blocks: list[str], json_response: dict, pause_after_chunk: int = -1):
        """pause_after_chunk: stall after writing that many blocks (-1 = never)."""
        super().__init__(("127.0.0.1", 0), _MockHandler)
        self.received: list[dict] = []
        self.stream_blocks = stream_blocks
        self.json_response = json_response
        self.json_status = 200
        self.pause_after_chunk = pause_after_chunk
        self.pause_timeout = 5.0
        self.paused = threading.Event()
        self.client_ack = threading.Event()
        self.streamed = threading.Event()
        self.pause_expired = False

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        self.shutdown()
        self.server_close()


def make_config(tmp: Path, keys: dict[str, KeyConfig], upstream: str = "http://127.0.0.1:1/v1", **overrides) -> Config:
    values = {
        "upstream_base_url": upstream,
        "upstream_api_key": UPSTREAM_KEY,
        "listen_host": "127.0.0.1",
        "listen_port": 0,
        "ledger_path": str(tmp / "ledger.sqlite"),
        "prices_path": str(Path(__file__).resolve().parent.parent / "spendgate" / "prices.json"),
        "keys": keys,
    }
    values.update(overrides)
    return Config(**values)


class ProxyHandle:
    """A live proxy on an ephemeral port, plus the mock upstream behind it."""

    def __init__(self, gate: Gate):
        self.server = ProxyServer(gate)
        self.gate = gate
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    @property
    def port(self) -> int:
        return self.server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self.server.stop()


def post_json(url: str, body: dict, token: str, timeout: float = 10.0) -> tuple[int, dict]:
    """Non-streaming client helper; returns (status, parsed json)."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        with error:
            return error.code, json.loads(error.read())


def raw_stream_request(url: str, body: dict, token: str, timeout: float = 10.0) -> socket.socket:
    """Open a socket-level POST so a test can read the reply chunk by chunk."""
    target = urlsplit(url)
    host, port, path = target.hostname, target.port, target.path
    payload = json.dumps(body).encode()
    sock = socket.create_connection((host, int(port)), timeout=timeout)
    sock.sendall(
        b"\r\n".join(
            [
                f"POST {path} HTTP/1.1".encode(),
                f"Host: {host}:{port}".encode(),
                b"Content-Type: application/json",
                f"Authorization: Bearer {token}".encode(),
                f"Content-Length: {len(payload)}".encode(),
                b"Connection: close",
                b"",
                b"",
            ]
        )
        + payload
    )
    return sock


class SocketReader:
    """Socket reader that preserves leftovers, so tests can read exactly one
    response segment at a time. Reading past what a proxy has relayed is the
    only way to prove it is relaying as data arrives."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.buffer = b""

    def _fill(self, deadline: float) -> bool:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        self.sock.settimeout(remaining)
        try:
            chunk = self.sock.recv(65536)
        except (TimeoutError, socket.timeout):
            return False
        if not chunk:
            return False
        self.buffer += chunk
        return True

    def read_until(self, needle: bytes, timeout: float = 10.0) -> bytes:
        """Consume up to and including `needle`; the rest stays readable."""
        deadline = time.monotonic() + timeout
        index = self.buffer.find(needle)
        while index < 0:
            if not self._fill(deadline):
                raise TimeoutError(f"never saw {needle!r} in {self.buffer!r}")
            index = self.buffer.find(needle)
        consumed, self.buffer = self.buffer[: index + len(needle)], self.buffer[index + len(needle):]
        return consumed

    def read_all(self, timeout: float = 10.0) -> bytes:
        deadline = time.monotonic() + timeout
        while self._fill(deadline):
            pass
        return self.buffer

def seed_ledger(path: str, rows: list[dict]) -> Ledger:
    ledger = Ledger(path)
    for row in rows:
        ledger.record(**row)
    return ledger


class _WebhookHandler(BaseHTTPRequestHandler):
    server: "WebhookCatcher"
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args) -> None:
        pass

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.server.payloads.append({"body": json.loads(raw), "authorization": self.headers.get("Authorization", "")})
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


class WebhookCatcher(ThreadingHTTPServer):
    """Local listener that captures budget alerts."""

    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _WebhookHandler)
        self.payloads: list[dict] = []
        self.thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/alerts"

    def wait_for_payload(self, timeout: float = 5.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.payloads:
                return self.payloads[0]
            time.sleep(0.02)
        raise AssertionError("no webhook alert was delivered")

    def stop(self) -> None:
        self.shutdown()
        self.server_close()
