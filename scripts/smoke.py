#!/usr/bin/env python3
"""End-to-end smoke test: a real proxy process, real curl clients, a real report.

    python scripts/smoke.py

Starts a mock OpenAI-compatible upstream and an actual `spendgate serve`
subprocess, drives one non-streaming and one streaming request at it with curl,
then reads the resulting ledger with the report CLI. Exits non-zero, with the
captured output, if anything does not line up.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests import support as s  # noqa: E402  (reuses the mock upstream)
from spendgate.prices import load_prices  # noqa: E402

MODEL = "gpt-4o-mini"
VIRTUAL_KEY = "vk-alice"
UPSTREAM = s.MockUpstream(
    stream_blocks=[s.sse_chunk("streamed", MODEL), s.sse_usage(3, 2, MODEL), "data: [DONE]\n\n"],
    json_response=s.completion_json(MODEL, 1000, 250, content="plain"),
)


class Failure(Exception):
    pass


def check(condition: bool, message: str, *context: str) -> None:
    if not condition:
        raise Failure(message + "\n" + "\n".join(context))


def start_proxy(workdir: pathlib.Path, port: int) -> subprocess.Popen:
    config = {
        "upstream_base_url": UPSTREAM.base_url,
        "upstream_api_key": "sk-mock-upstream-key",
        "listen_host": "127.0.0.1",
        "listen_port": port,
        "ledger_path": str(workdir / "spendgate.sqlite"),
        "webhook_url": None,
        "keys": {VIRTUAL_KEY: {"label": "alice", "monthly_budget_usd": 5.0}},
    }
    path = workdir / "spendgate.json"
    path.write_text(json.dumps(config, indent=2))
    env = dict(os.environ, PYTHONPATH=str(ROOT), SPENDGATE_PRICES_PATH=str(ROOT / "spendgate" / "prices.json"))
    return subprocess.Popen(
        [sys.executable, "-m", "spendgate", "serve", "-c", str(path)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_for_health(port: int, proxy: subprocess.Popen, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proxy.poll() is not None:
            raise Failure(f"proxy exited early with code {proxy.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1.0) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.05)
    raise Failure("proxy never answered /healthz")


def curl(port: int, body: dict, stream: bool = False) -> str:
    command = [
        "curl",
        "-sS",
        "--fail-with-body" if not stream else "-sS",
        "-X",
        "POST",
        f"http://127.0.0.1:{port}/v1/chat/completions",
        "-H",
        f"Authorization: Bearer {VIRTUAL_KEY}",
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(body),
    ]
    if stream:
        command.insert(1, "-N")  # do not buffer the event stream
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    check(result.returncode == 0, f"curl failed: {result.stderr}")
    return result.stdout


def run_report(workdir: pathlib.Path, *args: str) -> str:
    env = dict(os.environ, PYTHONPATH=str(ROOT), SPENDGATE_LEDGER_PATH=str(workdir / "spendgate.sqlite"))
    result = subprocess.run(
        [sys.executable, "-m", "spendgate", "report", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    check(result.returncode == 0, f"report failed: {result.stderr}")
    return result.stdout


def main() -> int:
    if shutil.which("curl") is None:
        print("smoke: curl is required", file=sys.stderr)
        return 2

    UPSTREAM.start()
    workdir = pathlib.Path(tempfile.mkdtemp())
    proxy = None
    prices = load_prices(ROOT / "spendgate" / "prices.json")
    try:
        port = int(os.environ.get("SPENDGATE_SMOKE_PORT", "18080"))
        proxy = start_proxy(workdir, port)
        wait_for_health(port, proxy)
        print(f"smoke: proxy live on 127.0.0.1:{port}, upstream on {UPSTREAM.base_url}")

        plain = curl(port, {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
        print("smoke: non-streaming reply:", plain[:78], "...")
        answer = json.loads(plain)
        check(answer["choices"][0]["message"]["content"] == "plain", "non-streaming body was not relayed intact", plain)

        streamed = curl(port, {"model": MODEL, "stream": True, "messages": [{"role": "user", "content": "hi"}]}, stream=True)
        print("smoke: streaming reply:", repr(streamed[:70]), "...")
        check("streamed" in streamed and "[DONE]" in streamed, "stream was not relayed", streamed)

        text = run_report(workdir)
        json_out = run_report(workdir, "--json")
        print("\n" + text)

        report = json.loads(json_out)
        check(report["calls"] == 2, "expected two ledger rows", json_out)
        by_key = {row["group"]: row for row in report["by_key"]}
        check(by_key["alice"]["calls"] == 2, "both calls should belong to alice", json_out)
        expected = (1000 * prices[MODEL]["input"] + 250 * prices[MODEL]["output"]) / 1e6
        expected += (3 * prices[MODEL]["input"] + 2 * prices[MODEL]["output"]) / 1e6
        check(abs(report["total_cost"] - expected) < 1e-9, f"total cost should be {expected}", json_out)
        check(len(UPSTREAM.received) == 2, "upstream should have seen exactly two calls")
        check(UPSTREAM.received[1]["body"]["stream_options"] == {"include_usage": True}, "include_usage was not injected")

        month = time.strftime("%Y-%m", time.gmtime())
        check("No recorded calls" not in run_report(workdir, "--month", month), "current month should not be empty")
        print("smoke: OK — 2 calls metered, costs computed, report readable offline")
        return 0
    except Failure as failure:
        print(f"smoke: FAILED — {failure}", file=sys.stderr)
        if proxy is not None and proxy.stdout is not None:
            print("--- proxy output ---", file=sys.stderr)
            proxy.terminate()
            print(proxy.stdout.read(), file=sys.stderr)
        return 1
    finally:
        if proxy is not None and proxy.poll() is None:
            proxy.terminate()
            proxy.wait(timeout=10)
        UPSTREAM.stop()


if __name__ == "__main__":
    sys.exit(main())
