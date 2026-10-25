"""The scrape: format, label hygiene, and agreement with the ledger.

Counters here are only honest if they say what the SQLite file says — including
after a restart, which is exactly when an in-memory tally is caught lying.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest
import urllib.error
import urllib.request

from tests import support as s
from spendgate.config import KeyConfig, UpstreamConfig
from spendgate.metrics import CONTENT_TYPE, exposition
from spendgate.proxy import build_gate

MODEL = "gpt-4o-mini"
CHEAP = "vision-pro-9000"
ALICE_COST = (1000 * 0.15 + 250 * 0.60) / 1_000_000


def scrape(url: str, token: str | None = None) -> tuple[int, str, str]:
    """GET the endpoint; returns (status, content type, body)."""
    request = urllib.request.Request(url, method="GET")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read().decode()
    except urllib.error.HTTPError as error:
        with error:
            return error.code, error.headers.get("Content-Type", ""), error.read().decode()


def samples(body: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    """Parse an exposition into {(metric, labels): value}."""
    out: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition(" ")
        labels: list[tuple[str, str]] = []
        if "{" in name:
            metric, _, raw = name.partition("{")
            for pair in raw.rstrip("}").split(","):
                key, _, value = pair.partition("=")
                labels.append((key, value.strip('"')))
            name = metric
        out[(name, tuple(labels))] = float(rest)
    return out


def value(body: str, metric: str, **labels: str) -> float:
    """One sample, addressed by metric and its exact label set."""
    wanted = tuple(sorted(labels.items()))
    for (name, series), sample in samples(body).items():
        if name == metric and tuple(sorted(series)) == wanted:
            return sample
    raise AssertionError(f"{metric} {labels} is not in the scrape")


class MetricsTestCase(unittest.TestCase):
    """One proxy, two keys (one budgeted), a mock upstream that answers with usage."""

    def setUp(self) -> None:
        self.mock = s.MockUpstream(stream_blocks=[], json_response=s.completion_json(MODEL, 1000, 250))
        self.mock.start()
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.keys = {
            "vk-alice-secret": KeyConfig(label="alice", monthly_budget_usd=20.0),
            "vk-bob-secret": KeyConfig(label="bob"),
        }
        self.config = s.make_config(self.tmp, keys=dict(self.keys), upstream=self.mock.base_url)
        self.proxy = s.ProxyHandle(build_gate(self.config))
        self.metrics_url = self.proxy.base_url + "/metrics"

    def tearDown(self) -> None:
        self.proxy.stop()
        self.mock.stop()

    def call(self, key: str = "vk-alice-secret", model: str = MODEL) -> int:
        status, _ = s.post_json(
            self.proxy.base_url + "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "hi"}]},
            key,
        )
        return status

    def body(self) -> str:
        status, content_type, text = scrape(self.metrics_url)
        self.assertEqual(status, 200)
        self.assertEqual(content_type, CONTENT_TYPE)
        return text


class FormatTest(MetricsTestCase):
    def test_the_scrape_is_infrastructure_not_a_client_call(self) -> None:
        status, content_type, _ = scrape(self.metrics_url)
        self.assertEqual(status, 200)
        self.assertEqual(content_type, "text/plain; version=0.0.4")

    def test_every_family_declares_its_help_type_and_samples(self) -> None:
        self.call()
        body = self.body()
        for name in ("spendgate_requests_total", "spendgate_tokens_total", "spendgate_cost_usd_total", "spendgate_budget_fraction"):
            self.assertIn(f"# HELP {name} ", body)
            self.assertIn(f"# TYPE {name} ", body)
            self.assertTrue(any(line.startswith(name + "{") for line in body.splitlines()), f"no {name} sample")

    def test_an_idle_proxy_scrapes_only_its_budgets(self) -> None:
        # No calls yet: the counters are legitimately absent, the budget gauge is not.
        body = self.body()
        self.assertNotIn("spendgate_requests_total{", body)
        self.assertIn('spendgate_budget_fraction{key="alice"} 0', body)
        self.assertNotIn("spendgate_budget_fraction{key=\"bob\"}", body, "an uncapped key has no fraction")

    def test_the_budget_gauge_moves_with_the_ledger(self) -> None:
        self.call()
        fraction = value(self.body(), "spendgate_budget_fraction", key="alice")
        self.assertAlmostEqual(fraction, ALICE_COST / 20.0, places=12)


class LedgerAgreementTest(MetricsTestCase):
    def test_counters_match_the_ledger_after_proxied_calls(self) -> None:
        self.call()
        self.call(key="vk-bob-secret")
        body = self.body()
        self.assertEqual(value(body, "spendgate_requests_total", key="alice", model=MODEL, status="200"), 1)
        self.assertEqual(value(body, "spendgate_requests_total", key="bob", model=MODEL, status="200"), 1)
        self.assertEqual(value(body, "spendgate_tokens_total", key="alice", model=MODEL, kind="prompt"), 1000)
        self.assertEqual(value(body, "spendgate_tokens_total", key="alice", model=MODEL, kind="completion"), 250)
        self.assertAlmostEqual(value(body, "spendgate_cost_usd_total", key="alice", model=MODEL), ALICE_COST, places=12)

        totals = self.proxy.gate.ledger.key_model_totals()
        self.assertEqual(sum(total.calls for total in totals), 2)
        self.assertEqual(sum(total.prompt_tokens or 0 for total in totals), 1000 + 1000)

    def test_a_scrape_after_a_restart_still_reports_the_spend(self) -> None:
        self.call()
        self.call(key="vk-bob-secret")
        before = self.body()

        self.proxy.stop()
        # A fresh gate over the same file: nothing in memory survives the restart.
        restarted = s.ProxyHandle(build_gate(self.config))
        self.addCleanup(restarted.stop)
        status, _, after = scrape(restarted.base_url + "/metrics")
        self.assertEqual(status, 200)

        for metric in ("spendgate_requests_total", "spendgate_tokens_total", "spendgate_cost_usd_total"):
            self.assertEqual(
                {key: v for key, v in samples(before).items() if key[0] == metric},
                {key: v for key, v in samples(after).items() if key[0] == metric},
                f"{metric} moved after a restart",
            )
        self.assertEqual(value(after, "spendgate_requests_total", key="alice", model=MODEL, status="200"), 1)
        self.assertEqual(restarted.gate.ledger.row_count(), 2)

    def test_a_fresh_call_still_grows_the_counter_after_a_restart(self) -> None:
        self.call()
        self.proxy.stop()
        restarted = s.ProxyHandle(build_gate(self.config))
        self.addCleanup(restarted.stop)
        s.post_json(
            restarted.base_url + "/v1/chat/completions",
            {"model": MODEL, "messages": []},
            "vk-alice-secret",
        )
        status, _, body = scrape(restarted.base_url + "/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(value(body, "spendgate_requests_total", key="alice", model=MODEL, status="200"), 2)


class UnknownModelTest(MetricsTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.mock.json_response = s.completion_json(CHEAP, 120, 40)

    def test_an_unpriceable_model_counts_requests_and_tokens_but_has_no_cost(self) -> None:
        self.call(model=CHEAP)
        body = self.body()
        self.assertEqual(value(body, "spendgate_requests_total", key="alice", model=CHEAP, status="200"), 1)
        self.assertEqual(value(body, "spendgate_tokens_total", key="alice", model=CHEAP, kind="prompt"), 120)
        self.assertEqual(value(body, "spendgate_tokens_total", key="alice", model=CHEAP, kind="completion"), 40)
        self.assertNotIn("spendgate_cost_usd_total", "".join(line for line in body.splitlines() if not line.startswith("#")))
        self.assertIsNone(self.proxy.gate.ledger.rows()[0].cost)


class RefusalTest(MetricsTestCase):
    def test_a_refused_key_is_counted_anonymously(self) -> None:
        self.assertEqual(self.call(key="vk-nope"), 401)
        body = self.body()
        self.assertEqual(value(body, "spendgate_requests_total", key="(anonymous)", model="(unknown)", status="401"), 1)
        self.assertNotIn("vk-nope", body)

    def test_a_model_no_upstream_serves_is_counted_and_costs_nothing(self) -> None:
        config = s.make_config(
            self.tmp,
            keys={"vk-alice-secret": KeyConfig(label="alice")},
            upstreams=[UpstreamConfig(name="only", base_url=self.mock.base_url, api_key="k", models=["gpt-4"])],
        )
        proxy = s.ProxyHandle(build_gate(config))
        self.addCleanup(proxy.stop)
        status, _ = s.post_json(
            proxy.base_url + "/v1/chat/completions",
            {"model": "deepseek-chat", "messages": []},
            "vk-alice-secret",
        )
        self.assertEqual(status, 404)
        _, _, body = scrape(proxy.base_url + "/metrics")
        self.assertEqual(value(body, "spendgate_requests_total", key="alice", model="deepseek-chat", status="404"), 1)
        self.assertNotIn("deepseek-chat", "".join(line for line in body.splitlines() if line.startswith("spendgate_cost")))
        self.assertEqual(proxy.gate.ledger.rows(), [])

    def test_a_budget_refusal_is_counted_at_its_status(self) -> None:
        self.proxy.gate.ledger.record(key_label="alice", model=MODEL, prompt_tokens=1, completion_tokens=1, cost=20.0)
        self.assertEqual(self.call(), 429)
        body = self.body()
        self.assertEqual(value(body, "spendgate_requests_total", key="alice", model=MODEL, status="429"), 1)
        self.assertAlmostEqual(value(body, "spendgate_budget_fraction", key="alice"), 1.0, places=9)

    def test_an_unreachable_upstream_is_counted_but_not_costed(self) -> None:
        config = s.make_config(
            self.tmp,
            keys={"vk-alice-secret": KeyConfig(label="alice")},
            upstream="http://127.0.0.1:1/v1",
        )
        proxy = s.ProxyHandle(build_gate(config))
        self.addCleanup(proxy.stop)
        status, _ = s.post_json(proxy.base_url + "/v1/chat/completions", {"model": MODEL, "messages": []}, "vk-alice-secret")
        self.assertEqual(status, 502)
        _, _, body = scrape(proxy.base_url + "/metrics")
        self.assertEqual(value(body, "spendgate_requests_total", key="alice", model=MODEL, status="502"), 1)
        self.assertNotIn("spendgate_cost_usd_total{", body)
        self.assertEqual(proxy.gate.ledger.rows(), [])


class LabelHygieneTest(MetricsTestCase):
    def test_no_key_material_ever_appears_in_the_scrape(self) -> None:
        self.call()
        self.call(key="vk-bob-secret")
        self.mock.json_response = s.completion_json("claude-x", 1, 1)
        self.assertEqual(self.call(key="vk-nope"), 401)
        body = self.body()
        for secret in ("vk-alice-secret", "vk-bob-secret", "vk-nope", s.UPSTREAM_KEY):
            self.assertNotIn(secret, body)
            self.assertNotIn(secret, repr(self.proxy.gate.outcomes.snapshot()))

    def test_a_label_needing_escaping_is_escaped_not_injected(self) -> None:
        self.proxy.gate.ledger.record(key_label='we"ird', model="model\nname", prompt_tokens=1, completion_tokens=1, cost=0.5)
        body = self.body()
        self.assertIn('key="we\\"ird"', body)
        self.assertIn('model="model\\nname"', body)
        seen = {dict(series).get("key") for _, series in samples(body)}
        self.assertNotIn('we"ird', seen, "the quote must not close the label early")
        self.assertIn('we\\"ird', seen)

    def test_an_exposition_with_no_calls_still_declares_every_family(self) -> None:
        config = s.make_config(pathlib.Path(tempfile.mkdtemp()), keys={"vk-alice-secret": KeyConfig(label="alice")})
        gate = build_gate(config)
        body = exposition(gate.config, gate.ledger, gate.outcomes).decode()
        for name in ("requests_total", "tokens_total", "cost_usd_total", "budget_fraction"):
            self.assertIn(f"# TYPE spendgate_{name} ", body)


if __name__ == "__main__":
    unittest.main()
