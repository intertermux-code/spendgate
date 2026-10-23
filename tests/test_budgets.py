"""Acceptance (e) and (f): soft alerts and the hard monthly cutoff."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from tests import support as s
from spendgate import budgets
from spendgate.config import KeyConfig
from spendgate.ledger import Ledger
from spendgate.proxy import build_gate

MODEL = "gpt-4o-mini"


class BudgetTestCase(unittest.TestCase):
    """Seeds the ledger with a known spend, then points a live proxy at it."""

    cap: float | None = 1.0
    spent: float = 0.0

    def mock_upstream(self) -> s.MockUpstream:
        return s.MockUpstream(stream_blocks=[], json_response=s.completion_json(MODEL, 10, 10))

    def setUp(self) -> None:
        self.mock = self.mock_upstream()
        self.mock.start()
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.webhook = s.WebhookCatcher()
        ledger = Ledger(str(self.tmp / "ledger.sqlite"))
        if self.spent:
            ledger.record(key_label="alice", model=MODEL, prompt_tokens=1, completion_tokens=1, cost=self.spent)
        self.key = "vk-alice"
        self.config = s.make_config(
            self.tmp,
            keys={self.key: KeyConfig(label="alice", monthly_budget_usd=self.cap)},
            upstream=self.mock.base_url,
            webhook_url=self.webhook.url,
        )
        self.proxy = s.ProxyHandle(build_gate(self.config))

    def tearDown(self) -> None:
        self.proxy.stop()
        self.mock.stop()
        self.webhook.stop()

    def call(self) -> tuple[int, dict]:
        return s.post_json(self.proxy.base_url + "/v1/chat/completions", {"model": MODEL, "messages": []}, self.key)


class NearCapTest(BudgetTestCase):
    cap = 1.0
    spent = 0.85

    def test_key_near_its_cap_is_still_served(self) -> None:
        status, _ = self.call()
        self.assertEqual(status, 200)
        self.assertEqual(len(self.mock.received), 1)

    def test_crossing_the_soft_limit_fires_one_webhook_alert(self) -> None:
        self.call()
        alert = self.webhook.wait_for_payload()["body"]
        self.assertEqual(alert["key_label"], "alice")
        self.assertEqual(alert["level"], "soft")
        self.assertEqual(alert["budget_usd"], 1.0)
        self.assertAlmostEqual(alert["spent_usd"], 0.85, places=6)
        self.call()
        self.call()
        self.assertEqual(len(self.webhook.payloads), 1, "one alert per key per month, not one per request")

    def test_alert_carries_no_credentials(self) -> None:
        self.call()
        self.webhook.wait_for_payload()
        captured = str(self.webhook.payloads)
        self.assertNotIn(s.UPSTREAM_KEY, captured)
        self.assertNotIn(self.key, captured)


class OverCapTest(BudgetTestCase):
    cap = 1.0
    spent = 1.0

    def test_over_cap_returns_429_without_calling_the_upstream(self) -> None:
        status, payload = self.call()
        self.assertEqual(status, 429)
        self.assertEqual(self.mock.received, [])
        self.assertEqual(payload["error"]["code"], "budget_exceeded")
        self.assertIn("budget", payload["error"]["message"].lower())
        self.assertNotIn(s.UPSTREAM_KEY, str(payload))
        self.assertEqual(len(self.proxy.gate.ledger.rows()), 1, "a refused call is not a spend row")


class ExactlyAtSoftFractionTest(BudgetTestCase):
    cap = 10.0
    spent = 8.0

    def test_default_soft_limit_is_eighty_percent(self) -> None:
        self.call()
        self.webhook.wait_for_payload()


class UncappedKeyTest(BudgetTestCase):
    cap = None
    spent = 500.0

    def test_uncapped_key_is_tracked_but_never_limited(self) -> None:
        status, _ = self.call()
        self.assertEqual(status, 200)
        self.assertEqual(self.webhook.payloads, [])
        self.assertEqual(len(self.proxy.gate.ledger.rows()), 2)


class BudgetUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = Ledger(pathlib.Path(tempfile.mkdtemp()) / "l.sqlite")

    def record(self, label: str, cost: float, timestamp: str) -> None:
        self.ledger.record(key_label=label, model=MODEL, prompt_tokens=1, completion_tokens=1, cost=cost, timestamp=timestamp)

    def test_only_the_current_calendar_month_counts(self) -> None:
        self.record("alice", 5.0, "2025-12-31T23:59:59+00:00")
        self.record("alice", 2.0, "2026-01-01T00:00:00+00:00")
        self.assertEqual(self.ledger.sum_month("alice", 2026, 1), 2.0)

    def test_levels_and_boundaries(self) -> None:
        from datetime import datetime, timezone

        now = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
        key = KeyConfig(label="bob", monthly_budget_usd=10.0)
        for spent, level in ((0.0, budgets.OK), (7.99, budgets.OK), (8.0, budgets.SOFT), (9.99, budgets.SOFT), (10.0, budgets.HARD), (12.0, budgets.HARD)):
            ledger = Ledger(pathlib.Path(tempfile.mkdtemp()) / "l.sqlite")
            ledger.record(key_label="bob", model=MODEL, prompt_tokens=1, completion_tokens=1, cost=spent)
            self.assertEqual(budgets.check(ledger, key, 0.8, now).level, level, f"spent={spent}")

    def test_uncapped_key_reports_ok_with_a_running_total(self) -> None:
        ledger = Ledger(pathlib.Path(tempfile.mkdtemp()) / "l.sqlite")
        ledger.record(key_label="carol", model=MODEL, prompt_tokens=1, completion_tokens=1, cost=99.0)
        decision = budgets.check(ledger, KeyConfig(label="carol"), 0.8)
        self.assertEqual((decision.level, decision.cap_usd, decision.spent_usd), (budgets.OK, None, 99.0))

    def test_unknown_model_spend_cannot_limit_a_key(self) -> None:
        # A NULL cost is not zero spend the proxy can enforce on: it is an
        # absence of information. Documented in the README.
        ledger = Ledger(pathlib.Path(tempfile.mkdtemp()) / "l.sqlite")
        ledger.record(key_label="dave", model="mystery", prompt_tokens=5, completion_tokens=5, cost=None)
        decision = budgets.check(ledger, KeyConfig(label="dave", monthly_budget_usd=1.0), 0.8)
        self.assertEqual(decision.level, budgets.OK)


if __name__ == "__main__":
    unittest.main()
