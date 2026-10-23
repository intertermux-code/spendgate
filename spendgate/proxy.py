"""The proxy: an OpenAI-compatible endpoint that meters what it forwards.

Single process, threaded HTTP server from the stdlib. Every accepted call is
written to the spend ledger exactly once, after the upstream has answered.
"""

from __future__ import annotations

import hmac
import json
import logging
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import budgets, __version__
from .config import Config, KeyConfig, default_prices_path
from .ledger import Ledger
from .prices import compute_cost, load_prices
from .upstream import SSEUsageScanner, Usage, build_request, inject_include_usage, usage_from_json

logger = logging.getLogger("spendgate.proxy")

COMPLETIONS_PATH = "/v1/chat/completions"
UPSTREAM_TIMEOUT_SECONDS = 300.0
RELAY_BLOCK_BYTES = 8192


def openai_error(message: str, err_type: str, code: str, param: str | None = None) -> bytes:
    """Error body in the shape OpenAI clients already know how to display."""
    return json.dumps(
        {"error": {"message": message, "type": err_type, "param": param, "code": code}}
    ).encode("utf-8")


class Gate:
    """Everything the handler needs: config, ledger, prices, alert bookkeeping."""

    def __init__(self, config: Config, ledger: Ledger, prices: dict[str, dict[str, float]]):
        self.config = config
        self.ledger = ledger
        self.prices = prices
        self._alerts_lock = threading.Lock()
        self._alerted: set[tuple[str, str]] = set()

    def authenticate(self, token: str) -> KeyConfig | None:
        """Look up a virtual key in constant time; unknown keys get None."""
        found: KeyConfig | None = None
        candidate = token.encode("utf-8")
        for secret, key in self.config.keys.items():
            # Keep comparing every key so a miss costs as long as a hit.
            if hmac.compare_digest(candidate, secret.encode("utf-8")):
                found = key
        return found

    def note_soft_limit(self, key: KeyConfig, decision: budgets.Decision, month: str) -> None:
        """Warn, and alert at most once per key per month.

        Without the dedupe a key that trips the soft limit would post one
        webhook per request for the rest of the month.
        """
        payload = budgets.alert_payload(key, decision, month)
        logger.warning(
            "key %r at %s of its monthly budget: spent %.4f of %.4f",
            key.label,
            decision.level,
            decision.spent_usd,
            decision.cap_usd or 0.0,
        )
        if self.config.webhook_url is None:
            return
        with self._alerts_lock:
            marker = (key.label, month)
            if marker in self._alerted:
                return
            self._alerted.add(marker)
        if not budgets.post_alert(self.config.webhook_url, payload):
            with self._alerts_lock:
                self._alerted.discard(marker)

    def record(self, key: KeyConfig, usage: Usage | None, fallback_model: str) -> None:
        """Append one call to the ledger. Unknown models leave cost NULL."""
        model = (usage.model if usage else None) or fallback_model
        prompt = usage.prompt_tokens if usage else None
        completion = usage.completion_tokens if usage else None
        self.ledger.record(
            key_label=key.label,
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cost=compute_cost(model, prompt, completion, self.prices),
        )


class ProxyHandler(BaseHTTPRequestHandler):
    server: "ProxyServer"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s %s", self.address_string(), format % args)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/healthz":
            self._respond(200, json.dumps({"status": "ok", "version": __version__}).encode())
        else:
            self._respond(404, openai_error("Not found", "invalid_request_error", "path_not_found"))

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != COMPLETIONS_PATH:
            self._reject(404, "Not found", "invalid_request_error", "path_not_found")
            return

        key = self._authorise()
        if key is None:
            # Rejected before the ledger and before the upstream: an unknown
            # key must not be able to grow anyone's spend.
            self._reject(401, "Invalid API key", "authentication_error", "invalid_api_key")
            return

        # A body without a Content-Length cannot be framed by http.server; real
        # OpenAI clients always send one, so this is a bad request, not a gap.
        body = self._read_json_body()
        if body is None:
            self._reject(400, "Request body must be a JSON object with a Content-Length", "invalid_request_error", "invalid_json")
            return
        model = body.get("model")
        if not isinstance(model, str) or not model:
            self._reject(400, "Request body must name a 'model'", "invalid_request_error", "missing_model", param="model")
            return

        gate = self.server.gate
        month = budgets.current_month()
        decision = budgets.check(gate.ledger, key, gate.config.soft_limit_fraction)
        if decision.level == budgets.SOFT:
            gate.note_soft_limit(key, decision, month)
        if not decision.allowed:
            self._reject(
                429,
                f"Monthly budget of ${decision.cap_usd:.2f} for key {key.label!r} is exhausted "
                f"(${decision.spent_usd:.2f} spent in {month})",
                "rate_limit_error",
                "budget_exceeded",
            )
            return

        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        streaming = body.get("stream") is True
        if streaming:
            payload = inject_include_usage(payload)
        self._forward(gate, key, payload, streaming, model)

    # -- plumbing ---------------------------------------------------------

    def _authorise(self) -> KeyConfig | None:
        header = self.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return None
        return self.server.gate.authenticate(token.strip())

    def _read_json_body(self) -> dict[str, Any] | None:
        length = self.headers.get("Content-Length")
        if length is None:
            return None
        try:
            raw = self.rfile.read(int(length))
        except (ValueError, OSError):
            return None
        try:
            parsed = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def _forward(self, gate: Gate, key: KeyConfig, payload: bytes, streaming: bool, model: str) -> None:
        request = build_request(gate.config, payload, streaming)
        try:
            response = urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as error:
            # An upstream refusal is relayed as-is; nothing was metered because
            # nothing was generated.
            with error:
                self._respond(error.code, error.read(), content_type=_content_type(error.headers))
            return
        except (urllib.error.URLError, OSError) as error:
            logger.error("upstream unreachable: %s", error.reason if isinstance(error, urllib.error.URLError) else error)
            self._reject(502, "Upstream provider unreachable", "api_error", "upstream_unreachable")
            return

        usage: Usage | None = None
        try:
            if streaming and _is_event_stream(response.headers):
                usage = self._relay_stream(response)
            else:
                usage = self._relay_buffered(response)
        except (BrokenPipeError, ConnectionResetError):
            # The client hung up before the usage chunk, so there is nothing to
            # record but the fact that the call happened. Inventing a count
            # would be worse than a NULL.
            logger.info("client disconnected before the response finished")
        finally:
            response.close()

        gate.record(key, usage, model)

    def _relay_buffered(self, response: Any) -> Usage | None:
        body = response.read()
        self._respond(response.status, body, content_type=_content_type(response.headers))
        try:
            return usage_from_json(json.loads(body))
        except (ValueError, UnicodeDecodeError):
            return None

    def _relay_stream(self, response: Any) -> Usage | None:
        """Copy SSE bytes to the client as they land, metering on the side."""
        self.send_response(response.status)
        self.send_header("Content-Type", _content_type(response.headers))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        scanner = SSEUsageScanner()
        while True:
            # read1 hands back whatever the socket holds instead of waiting to
            # fill a whole block — that is what keeps this unbuffered.
            block = response.read1(RELAY_BLOCK_BYTES)
            if not block:
                break
            self.wfile.write(block)
            self.wfile.flush()
            scanner.feed(block)
        scanner.close()
        self.wfile.flush()
        return scanner.usage

    def _respond(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _reject(self, status: int, message: str, err_type: str, code: str, param: str | None = None) -> None:
        """Refuse a request, and do not keep the connection open for it."""
        self.close_connection = True
        self._respond(status, openai_error(message, err_type, code, param))


def _content_type(headers: Any) -> str:
    value = headers.get("Content-Type") if headers else None
    return value or "application/octet-stream"


def _is_event_stream(headers: Any) -> bool:
    return "text/event-stream" in (headers.get("Content-Type") or "").lower()


class ProxyServer(ThreadingHTTPServer):
    """Threaded server so a long stream never blocks the calls beside it."""

    daemon_threads = True

    def __init__(self, gate: Gate):
        super().__init__((gate.config.listen_host, gate.config.listen_port), ProxyHandler)
        self.gate = gate

    def serve(self) -> None:
        logger.info("spendgate %s listening on http://%s:%d", __version__, self.server_address[0], self.server_address[1])
        self.serve_forever(poll_interval=0.2)

    def stop(self) -> None:
        self.shutdown()
        self.server_close()


def build_gate(config: Config, prices: dict[str, dict[str, float]] | None = None) -> Gate:
    """Wire up ledger + price table from a config. Tests inject `prices`."""
    if prices is None:
        prices = load_prices(config.prices_path or default_prices_path())
    return Gate(config, Ledger(config.ledger_path), prices)


def run(config: Config) -> None:
    """Serve until interrupted. This is what `spendgate serve` calls."""
    gate = build_gate(config)
    server = ProxyServer(gate)
    print(f"spendgate: forwarding {config.upstream_base_url} on http://{server.server_address[0]}:{server.server_address[1]}")
    print(f"spendgate: {len(config.keys)} virtual key(s), ledger at {config.ledger_path}")
    try:
        server.serve()
    except KeyboardInterrupt:
        print("\nspendgate: shutting down")
    finally:
        server.stop()
