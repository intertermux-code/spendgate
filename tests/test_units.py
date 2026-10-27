"""Unit tests for the parts that do not need a socket."""

from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import tempfile
import unittest

from spendgate import budgets
from spendgate.config import KeyConfig, default_prices_path, load_config
from spendgate.ledger import GROUP_COLUMNS, Ledger
from spendgate.prices import compute_cost, load_prices
from spendgate.upstream import SSEUsageScanner, inject_include_usage, usage_from_json

ROOT = pathlib.Path(__file__).resolve().parent.parent


class SchemaTest(unittest.TestCase):
    """The ledger table is the project's only persistent contract."""

    def setUp(self) -> None:
        self.path = str(pathlib.Path(tempfile.mkdtemp()) / "l.sqlite")
        Ledger(self.path)
        self.conn = sqlite3.connect(self.path)

    def tearDown(self) -> None:
        self.conn.close()

    def test_columns_and_types(self) -> None:
        columns = {row[1]: row[2] for row in self.conn.execute("PRAGMA table_info(spend)")}
        self.assertEqual(
            columns,
            {
                "id": "INTEGER",
                "timestamp": "TEXT",
                "key_label": "TEXT",
                "model": "TEXT",
                "prompt_tokens": "INTEGER",
                "completion_tokens": "INTEGER",
                "cost": "REAL",
            },
        )

    def test_every_index_in_the_schema_is_created(self) -> None:
        # A malformed CREATE INDEX used to be swallowed silently here: the
        # schema still opened, the model index just never existed.
        indexed = {row[1] for row in self.conn.execute("PRAGMA index_list(spend)")}
        self.assertEqual(indexed, {"idx_spend_timestamp", "idx_spend_key_label", "idx_spend_model"})

    def test_schema_is_idempotent_and_timestamps_are_not_null(self) -> None:
        Ledger(self.path)  # reopening must not choke on the existing table
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute("INSERT INTO spend (key_label, model) VALUES ('a', 'm')")

    def test_group_columns_are_the_only_names_reaching_the_sql(self) -> None:
        self.assertEqual(sorted(GROUP_COLUMNS), ["day", "key", "model"])
        ledger = Ledger(self.path)
        with self.assertRaises(ValueError):
            ledger.summarize("cost) FROM spend --")


class LedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = Ledger(str(pathlib.Path(tempfile.mkdtemp()) / "l.sqlite"))

    def test_record_then_read_back(self) -> None:
        row_id = self.ledger.record(
            key_label="alice", model="gpt-4o", prompt_tokens=3, completion_tokens=4, cost=0.25, timestamp="2026-10-01T00:00:00+00:00"
        )
        self.assertEqual(row_id, 1)
        row = self.ledger.rows()[0]
        self.assertEqual((row.key_label, row.model, row.cost), ("alice", "gpt-4o", 0.25))

    def test_nulls_round_trip(self) -> None:
        self.ledger.record(key_label="a", model="m", prompt_tokens=None, completion_tokens=None, cost=None)
        row = self.ledger.rows()[0]
        self.assertEqual((row.prompt_tokens, row.completion_tokens, row.cost), (None, None, None))

    def test_append_only_means_ids_only_ever_grow(self) -> None:
        ids = [
            self.ledger.record(key_label="a", model="m", prompt_tokens=i, completion_tokens=i, cost=0.0) for i in range(4)
        ]
        self.assertEqual(ids, [1, 2, 3, 4])
        self.assertEqual(self.ledger.row_count(), 4)

    def test_sum_month_ignores_other_months(self) -> None:
        self.ledger.record(key_label="a", model="m", prompt_tokens=1, completion_tokens=1, cost=1.0, timestamp="2026-09-30T23:00:00+00:00")
        self.ledger.record(key_label="a", model="m", prompt_tokens=1, completion_tokens=1, cost=2.0, timestamp="2026-10-01T00:00:00+00:00")
        self.ledger.record(key_label="b", model="m", prompt_tokens=1, completion_tokens=1, cost=4.0, timestamp="2026-10-02T00:00:00+00:00")
        self.assertEqual(self.ledger.sum_month("a", 2026, 10), 2.0)
        self.assertEqual(self.ledger.sum_month("a", 2026, 9), 1.0)
        self.assertEqual(self.ledger.sum_month("nobody", 2026, 10), 0.0)

    def test_read_only_ledger_never_creates_a_file(self) -> None:
        path = pathlib.Path(tempfile.mkdtemp()) / "gone.sqlite"
        with self.assertRaises(FileNotFoundError):
            Ledger(str(path), create=False)
        self.assertFalse(path.exists())

    def test_read_only_ledger_refuses_writes(self) -> None:
        path = str(pathlib.Path(tempfile.mkdtemp()) / "l.sqlite")
        Ledger(path).record(key_label="a", model="m", prompt_tokens=1, completion_tokens=1, cost=1.0)
        readonly = Ledger(path, create=False)
        with self.assertRaises(RuntimeError):
            readonly.record(key_label="a", model="m", prompt_tokens=1, completion_tokens=1, cost=1.0)
        self.assertEqual(readonly.row_count(), 1)

    def test_threads_can_append_without_losing_rows(self) -> None:
        import threading

        def append(start: int) -> None:
            for i in range(start, start + 25):
                self.ledger.record(key_label=f"k{i % 3}", model="m", prompt_tokens=i, completion_tokens=i, cost=0.001)

        threads = [threading.Thread(target=append, args=(n * 25,)) for n in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(self.ledger.row_count(), 150)


class PricesTest(unittest.TestCase):
    def test_shipped_price_table_loads_and_names_current_models(self) -> None:
        prices = load_prices(default_prices_path())
        self.assertGreaterEqual(len(prices), 6)
        for name in ("gpt-4o", "gpt-4o-mini", "gpt-4.1", "deepseek-chat"):
            self.assertIn(name, prices)
            self.assertGreater(prices[name]["input"], 0)
            self.assertGreater(prices[name]["output"], 0)

    def test_price_file_is_valid_json_without_a_model_named_like_its_comment(self) -> None:
        raw = json.loads((ROOT / "spendgate" / "prices.json").read_text())
        self.assertNotIn("_comment", load_prices(default_prices_path()))
        self.assertIn("_comment", raw)

    def test_compute_cost(self) -> None:
        prices = {"m": {"input": 1.0, "output": 2.0}}
        self.assertAlmostEqual(compute_cost("m", 1_000_000, 500_000, prices), 2.0)
        self.assertAlmostEqual(compute_cost("m", 0, 0, prices), 0.0, places=12)
        self.assertAlmostEqual(compute_cost("m", None, 10, prices), 2e-05, places=12)
        self.assertIsNone(compute_cost("unknown", 10, 10, prices))
        self.assertIsNone(compute_cost("m", None, None, prices))
        self.assertIsNone(compute_cost(None, 10, 10, prices))
        self.assertIsNone(compute_cost("m", 10, 10, {}))

    def test_missing_or_ragged_price_files_degrade_to_no_prices(self) -> None:
        tmp = pathlib.Path(tempfile.mkdtemp())
        self.assertEqual(load_prices(tmp / "absent.json"), {})
        bad = tmp / "bad.json"
        bad.write_text('{"ok": {"input": 1, "output": 2}, "junk": 5, "half": {"input": 1}}')
        self.assertEqual(load_prices(bad), {"ok": {"input": 1.0, "output": 2.0}})

    def test_overridden_price_path_is_honoured(self) -> None:
        custom = pathlib.Path(tempfile.mkdtemp()) / "mine.json"
        custom.write_text('{"homegrown": {"input": 7, "output": 14}}')
        self.assertEqual(load_prices(custom)["homegrown"], {"input": 7.0, "output": 14.0})


class UpstreamParsingTest(unittest.TestCase):
    def test_include_usage_is_injected_only_when_absent(self) -> None:
        injected = json.loads(inject_include_usage(b'{"model": "m", "stream": true}'))
        self.assertEqual(injected["stream_options"], {"include_usage": True})

        kept = b'{"model":"m","stream":true,"stream_options":{"include_usage":false}}'
        self.assertEqual(inject_include_usage(kept), kept)

        not_streaming = b'{"model": "m"}'
        self.assertEqual(inject_include_usage(not_streaming), not_streaming)

    def test_usage_from_a_completion_body(self) -> None:
        usage = usage_from_json({"model": "m", "usage": {"prompt_tokens": 5, "completion_tokens": 6}})
        self.assertEqual((usage.prompt_tokens, usage.completion_tokens, usage.model), (5, 6, "m"))
        self.assertIsNone(usage_from_json({"model": "m"}))
        self.assertIsNone(usage_from_json([]))
        self.assertIsNone(usage_from_json({"usage": {"prompt_tokens": "many"}}))

    def test_scanner_survives_bytes_split_mid_line(self) -> None:
        stream = 'data: {"model":"m","choices":[]}\n\ndata: {"model":"m","usage":{"prompt_tokens":9,"completion_tokens":8}}\n\ndata: [DONE]\n\n'
        scanner = SSEUsageScanner()
        for i in range(0, len(stream), 7):  # deliberately ugly boundaries
            scanner.feed(stream[i : i + 7].encode())
        scanner.close()
        self.assertEqual((scanner.usage.prompt_tokens, scanner.usage.completion_tokens), (9, 8))
        self.assertEqual(scanner.model, "m")

    def test_scanner_ignores_noise_and_keeps_only_the_last_usage(self) -> None:
        scanner = SSEUsageScanner()
        scanner.feed(b"data: not json\n\n: keep-alive comment\n\n")
        scanner.feed(b'data: {"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n')
        scanner.feed(b'data: {"usage":{"prompt_tokens":2,"completion_tokens":2}}\n\n')
        scanner.close()
        self.assertEqual((scanner.usage.prompt_tokens, scanner.usage.completion_tokens), (2, 2))

    def test_scanner_does_not_grow_with_the_stream(self) -> None:
        scanner = SSEUsageScanner()
        for _ in range(2000):
            scanner.feed(b"data: " + b"x" * 200 + b"\n\n")
        self.assertLess(len(scanner._partial), 20)


class ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = pathlib.Path(tempfile.mkdtemp())

    def write(self, payload: dict) -> str:
        path = self.dir / "spendgate.json"
        path.write_text(json.dumps(payload))
        return str(path)

    def test_full_round_trip(self) -> None:
        path = self.write(
            {
                "upstream_base_url": "https://api.example.com/v1",
                "upstream_api_key": "sk-real-but-test-only",
                "listen_port": 9999,
                "ledger_path": "var/spend.sqlite",
                "webhook_url": "https://hooks.example.com/llm",
                "soft_limit_fraction": 0.5,
                "keys": {
                    "vk-one": {"label": "alice", "monthly_budget_usd": 20},
                    "vk-two": {"label": "bob"},
                    "vk-three": "plain-label",
                },
            }
        )
        config = load_config(path)
        self.assertEqual(config.upstream_url, "https://api.example.com/v1/")
        self.assertEqual(config.listen_port, 9999)
        self.assertEqual(config.soft_limit_fraction, 0.5)
        self.assertEqual(config.virtual_key_for("vk-one").monthly_budget_usd, 20.0)
        self.assertIsNone(config.virtual_key_for("vk-two").monthly_budget_usd)
        self.assertEqual(config.virtual_key_for("vk-three").label, "plain-label")
        self.assertIsNone(config.virtual_key_for("vk-nope"))

    def test_required_fields(self) -> None:
        for missing in ("upstream_base_url", "upstream_api_key"):
            with self.subTest(missing=missing):
                payload = {"upstream_base_url": "http://x/v1", "upstream_api_key": "k", "keys": {"a": "b"}}
                del payload[missing]
                with self.assertRaises(ValueError):
                    load_config(self.write(payload))

    def test_a_config_with_no_keys_is_refused_rather_than_serving_nobody(self) -> None:
        with self.assertRaises(ValueError):
            load_config(self.write({"upstream_base_url": "http://x/v1", "upstream_api_key": "k"}))

    def test_keys_must_be_an_object_and_says_so(self) -> None:
        # Hand-writing a config invites a list of keys; dereferencing .items()
        # on one used to be an AttributeError on the user's terminal.
        for wrong in (
            ["vk-one", "vk-two"],
            [{"label": "alice"}],
            "vk-one",
            7,
            None,
        ):
            with self.subTest(keys=wrong):
                with self.assertRaises(ValueError) as caught:
                    load_config(self.write({"upstream_base_url": "http://x/v1", "upstream_api_key": "k", "keys": wrong}))
                self.assertIn("'keys'", str(caught.exception))

    def test_bad_soft_limit_fraction(self) -> None:
        with self.assertRaises(ValueError):
            load_config(self.write({"upstream_base_url": "http://x/v1", "upstream_api_key": "k", "keys": {"a": "b"}, "soft_limit_fraction": 1.5}))

    def test_missing_file_names_the_template(self) -> None:
        with self.assertRaises(FileNotFoundError) as caught:
            load_config(str(self.dir / "nope.json"))
        self.assertIn("config.example.json", str(caught.exception))

    def test_environment_overrides(self) -> None:
        path = self.write({"upstream_base_url": "http://file/v1", "upstream_api_key": "file-key", "keys": {"a": "b"}})
        os.environ["SPENDGATE_UPSTREAM_API_KEY"] = "env-key"
        os.environ["SPENDGATE_LISTEN_PORT"] = "1234"
        self.addCleanup(os.environ.pop, "SPENDGATE_UPSTREAM_API_KEY", None)
        self.addCleanup(os.environ.pop, "SPENDGATE_LISTEN_PORT", None)
        config = load_config(path)
        self.assertEqual(config.listen_port, 1234)
        self.assertEqual(config.upstream_api_key, "env-key")


class BudgetUnitTests(unittest.TestCase):
    def test_month_of_is_utc(self) -> None:
        from datetime import datetime, timezone

        self.assertEqual(budgets.month_of(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)), (2026, 1, "2026-01"))

    def test_alert_payload_shape_and_no_secrets(self) -> None:
        payload = budgets.alert_payload(KeyConfig(label="alice", monthly_budget_usd=10.0), budgets.Decision(budgets.SOFT, 8.5, 10.0), "2026-10")
        self.assertEqual(
            payload,
            {
                "event": "budget_soft_limit",
                "key_label": "alice",
                "month": "2026-10",
                "spent_usd": 8.5,
                "budget_usd": 10.0,
                "level": "soft",
            },
        )
        self.assertNotIn("upstream", json.dumps(payload))

    def test_post_alert_survives_a_dead_webhook(self) -> None:
        self.assertFalse(budgets.post_alert("http://127.0.0.1:1/nope", {"a": 1}))


if __name__ == "__main__":
    unittest.main()
