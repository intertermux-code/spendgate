"""Acceptance (b): streaming passthrough is incremental, and usage comes from the last chunk.

The mock upstream stalls after its first chunk and refuses to send the rest
until the test signals that bytes have already arrived at the client socket. A
proxy that buffered the response would sit out the mock's pause timeout, and
each test asserts that the stall was released by the client, not by a timeout.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from tests import support as s
from spendgate.config import KeyConfig
from spendgate.proxy import build_gate

MODEL = "gpt-4o-mini"
BLOCKS = [
    s.sse_chunk("Hel", MODEL),
    s.sse_chunk("lo ", MODEL),
    s.sse_chunk("world", MODEL),
    s.sse_usage(11, 7, MODEL),
    "data: [DONE]\n\n",
]
WHOLE_STREAM = "".join(BLOCKS).encode()


class StreamingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mock = s.MockUpstream(stream_blocks=list(BLOCKS), json_response={}, pause_after_chunk=0)
        self.mock.start()
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.key = "vk-alice"
        self.config = s.make_config(self.tmp, keys={self.key: KeyConfig(label="alice")}, upstream=self.mock.base_url)
        self.proxy = s.ProxyHandle(build_gate(self.config))

    def tearDown(self) -> None:
        self.proxy.stop()
        self.mock.stop()

    def stream(self, extra: dict | None = None, stalled: bool = True) -> tuple[bytes, bytes]:
        """One streaming call. Returns (first SSE event, complete body)."""
        body = {"model": MODEL, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        body.update(extra or {})
        sock = s.raw_stream_request(self.proxy.base_url + "/v1/chat/completions", body, self.key)
        reader = s.SocketReader(sock)

        head = reader.read_until(b"\r\n\r\n")
        self.assertIn(b"200 OK", head.split(b"\r\n", 1)[0])
        self.assertIn(b"text/event-stream", head)

        first_event = reader.read_until(b"\n\n")
        if stalled:
            self.assertTrue(self.mock.paused.wait(2.0), "the upstream never reached its stall point")
            self.assertFalse(self.mock.streamed.is_set(), "the client has data while the upstream is still mid-stream")
            self.mock.client_ack.set()
        rest = reader.read_all()
        sock.close()
        if stalled:
            self.assertFalse(self.mock.pause_expired, "the proxy held the stream and the stall timed out")
        return first_event, first_event + rest

    def test_the_first_event_arrives_while_the_upstream_is_still_blocked(self) -> None:
        first_event, whole = self.stream()
        self.assertEqual(first_event, BLOCKS[0].encode(), "exactly one SSE event, relayed early")
        self.assertLess(len(first_event), len(WHOLE_STREAM))
        self.assertEqual(whole, WHOLE_STREAM)

    def test_usage_is_taken_from_the_final_chunk(self) -> None:
        self.stream()
        row = s.await_rows(self.proxy.gate.ledger, 1)[0]
        self.assertEqual((row.key_label, row.model, row.prompt_tokens, row.completion_tokens), ("alice", MODEL, 11, 7))
        self.assertAlmostEqual(row.cost, (11 * 0.15 + 7 * 0.60) / 1_000_000, places=12)

    def test_include_usage_is_injected_when_the_client_did_not_ask(self) -> None:
        self.stream()
        self.assertEqual(self.mock.received[0]["body"]["stream_options"], {"include_usage": True})

    def test_a_client_that_set_stream_options_is_not_overridden(self) -> None:
        self.stream({"stream_options": {"include_usage": False}})
        self.assertEqual(self.mock.received[0]["body"]["stream_options"], {"include_usage": False})

    def test_upstream_that_reports_no_usage_records_nulls(self) -> None:
        self.mock.stream_blocks = [s.sse_chunk("hi", MODEL), "data: [DONE]\n\n"]
        self.mock.pause_after_chunk = -1
        _, whole = self.stream(stalled=False)
        self.assertEqual(whole, "".join(self.mock.stream_blocks).encode())
        row = s.await_rows(self.proxy.gate.ledger, 1)[0]
        self.assertEqual((row.model, row.prompt_tokens, row.completion_tokens, row.cost), (MODEL, None, None, None))

    def test_sse_framing_survives_the_relay(self) -> None:
        _, whole = self.stream()
        text = whole.decode()
        self.assertEqual(text.count("\n\n"), len(BLOCKS))
        for line in text.splitlines():
            if line.startswith("data: ") and line[len("data: "):].strip() != "[DONE]":
                json.loads(line[len("data: "):])


if __name__ == "__main__":
    unittest.main()


class TruncatedStreamTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.key = "vk-alice"
        self.mock = s.MockUpstream(stream_blocks=[], json_response={})
        self.mock.start()
        self.config = s.make_config(self.tmp, keys={self.key: KeyConfig(label="alice")}, upstream=self.mock.base_url)
        self.proxy = s.ProxyHandle(build_gate(self.config))

    def tearDown(self) -> None:
        try:
            self.proxy.stop()
        except Exception:
            pass
        try:
            self.mock.stop()
        except Exception:
            pass

    def test_truncated_chunked_stream_ends_cleanly_with_error_and_ledger_row(self) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading

        class TruncHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                _ = json.loads(raw)
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunk1 = s.sse_chunk("Hel", MODEL).encode()
                self.wfile.write(f"{len(chunk1):X}\r\n".encode())
                self.wfile.write(chunk1)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                chunk2 = s.sse_chunk("lo", MODEL).encode()
                self.wfile.write(f"{len(chunk2):X}\r\n".encode())
                self.wfile.write(chunk2[:3])
                self.wfile.flush()
                self.connection.close()

        class TruncUpstream(ThreadingHTTPServer):
            daemon_threads = True

            def __init__(self):
                super().__init__(("127.0.0.1", 0), TruncHandler)

            @property
            def base_url(self):
                return f"http://127.0.0.1:{self.server_address[1]}/v1"

            def start(self):
                t = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
                t.start()
                return t

            def stop(self):
                self.shutdown()
                self.server_close()

        trunc = TruncUpstream()
        trunc.start()
        try:
            try:
                self.proxy.stop()
            except Exception:
                pass
            try:
                self.mock.stop()
            except Exception:
                pass
            self.config = s.make_config(self.tmp, keys={self.key: KeyConfig(label="alice")}, upstream=trunc.base_url)
            self.proxy = s.ProxyHandle(build_gate(self.config))

            body = {"model": MODEL, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
            sock = s.raw_stream_request(self.proxy.base_url + "/v1/chat/completions", body, self.key)
            reader = s.SocketReader(sock)
            try:
                head = reader.read_until(b"\r\n\r\n", timeout=2)
                self.assertIn(b"200 OK", head.split(b"\r\n", 1)[0])
                rest = reader.read_all(timeout=3)
                self.assertTrue(len(rest) > 0)
                self.assertIn(b"truncated", rest)
            finally:
                try:
                    sock.close()
                except Exception:
                    pass

            rows = s.await_rows(self.proxy.gate.ledger, 1, timeout=3)
            self.assertEqual(len(rows), 1)
        finally:
            trunc.stop()
