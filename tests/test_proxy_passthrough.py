"""Acceptance (a), (c), (d): passthrough, metering, and authentication."""

from __future__ import annotations

import pathlib
import tempfile
import unittest
import urllib.error
import urllib.request

from tests import support as s
from spendgate.config import KeyConfig
from spendgate.proxy import build_gate

MODEL = "gpt-4o-mini"
# 1000 prompt tokens at $0.15/M plus 250 completion at $0.60/M.
EXPECTED_COST = (1000 * 0.15 + 250 * 0.60) / 1_000_000


class ProxyTestCase(unittest.TestCase):
    """Subclasses only decide what the mock upstream answers with."""

    virtual_key = "vk-alice"

    def mock_upstream(self) -> s.MockUpstream:
        raise NotImplementedError

    def setUp(self) -> None:
        self.mock = self.mock_upstream()
        self.mock.start()
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.config = s.make_config(self.tmp, keys={self.virtual_key: KeyConfig(label="alice")}, upstream=self.mock.base_url)
        self.proxy = s.ProxyHandle(build_gate(self.config))

    def tearDown(self) -> None:
        self.proxy.stop()
        self.mock.stop()

    def call(self, body: dict | None = None, key: str | None = "vk-alice") -> tuple[int, dict]:
        payload = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
        payload.update(body or {})
        return s.post_json(self.proxy.base_url + "/v1/chat/completions", payload, key or self.virtual_key)

    def post_raw(self, data: bytes, headers: dict[str, str]) -> int:
        """Send a hand-built request; return the status it drew."""
        request = urllib.request.Request(self.proxy.base_url + "/v1/chat/completions", data=data, method="POST")
        for name, value in headers.items():
            request.add_header(name, value)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status
        except urllib.error.HTTPError as error:
            with error:
                return error.code


class PassthroughTest(ProxyTestCase):
    def mock_upstream(self) -> s.MockUpstream:
        return s.MockUpstream(stream_blocks=[], json_response=s.completion_json(MODEL, 1000, 250))

    def test_response_is_relayed_intact(self) -> None:
        status, payload = self.call()
        self.assertEqual(status, 200)
        self.assertEqual(payload, self.mock.json_response)

    def test_upstream_gets_the_server_key_not_the_virtual_one(self) -> None:
        self.call()
        self.assertEqual(self.mock.received[0]["authorization"], f"Bearer {s.UPSTREAM_KEY}")

    def test_ledger_row_carries_the_usage_the_upstream_reported(self) -> None:
        self.call()
        rows = s.await_rows(self.proxy.gate.ledger, 1)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row.key_label, row.model, row.prompt_tokens, row.completion_tokens), ("alice", MODEL, 1000, 250))
        self.assertAlmostEqual(row.cost, EXPECTED_COST, places=12)
        self.assertTrue(row.timestamp.endswith("+00:00"))

    def test_upstream_status_and_body_are_passed_through(self) -> None:
        self.mock.json_status = 400
        self.mock.json_response = {"error": {"message": "bad request", "type": "invalid_request_error"}}
        status, payload = self.call()
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["message"], "bad request")

    def test_stream_options_are_not_injected_for_non_streaming_calls(self) -> None:
        self.call()
        self.assertNotIn("stream_options", self.mock.received[0]["body"])


class UnknownModelTest(ProxyTestCase):
    def mock_upstream(self) -> s.MockUpstream:
        return s.MockUpstream(stream_blocks=[], json_response=s.completion_json("vision-pro-9000", 120, 40))

    def test_unknown_model_succeeds_with_null_cost(self) -> None:
        status, _ = self.call({"model": "vision-pro-9000"})
        self.assertEqual(status, 200)
        row = s.await_rows(self.proxy.gate.ledger, 1)[0]
        self.assertIsNone(row.cost)
        self.assertEqual((row.model, row.prompt_tokens, row.completion_tokens), ("vision-pro-9000", 120, 40))


class AuthTest(ProxyTestCase):
    def mock_upstream(self) -> s.MockUpstream:
        return s.MockUpstream(stream_blocks=[], json_response=s.completion_json(MODEL, 1, 1))

    def test_unknown_key_is_rejected_without_forwarding_or_logging(self) -> None:
        status, payload = self.call(key="vk-who")
        self.assertEqual(status, 401)
        self.assertEqual(payload["error"]["type"], "authentication_error")
        self.assertNotIn(s.UPSTREAM_KEY, str(payload))
        self.assertEqual(self.mock.received, [])
        self.assertEqual(self.proxy.gate.ledger.rows(), [])

    def test_missing_authorization_header_is_rejected(self) -> None:
        self.assertEqual(self.post_raw(b'{"model": "gpt-4o-mini", "messages": []}', {"Content-Type": "application/json"}), 401)
        self.assertEqual(self.mock.received, [])

    def test_virtual_key_never_travels_upstream(self) -> None:
        self.call()
        self.assertNotIn(self.virtual_key, str(self.mock.received))

    def test_unknown_path_is_not_proxied(self) -> None:
        status, _ = s.post_json(self.proxy.base_url + "/v1/completions", {"model": MODEL, "messages": []}, self.virtual_key)
        self.assertEqual(status, 404)
        self.assertEqual(self.mock.received, [])

    def test_malformed_json_is_rejected_before_the_upstream(self) -> None:
        status = self.post_raw(b"{not json", {"Content-Type": "application/json", "Authorization": f"Bearer {self.virtual_key}"})
        self.assertEqual(status, 400)
        self.assertEqual(self.mock.received, [])

    def test_request_without_a_model_is_rejected(self) -> None:
        status, payload = self.call({"model": None})
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "missing_model")
        self.assertEqual(self.mock.received, [])


if __name__ == "__main__":
    unittest.main()
