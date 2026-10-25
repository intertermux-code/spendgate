"""Model routing across several upstreams, and the refusal when none matches."""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import unittest

from tests import support as s
from spendgate.config import KeyConfig, UpstreamConfig, load_config, pattern_matches
from spendgate.proxy import build_gate


class RoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fast = s.MockUpstream(stream_blocks=[], json_response=s.completion_json("gpt-4o-mini", 10, 5))
        self.deep = s.MockUpstream(stream_blocks=[], json_response=s.completion_json("deepseek-chat", 20, 7))
        for mock in (self.fast, self.deep):
            mock.start()
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.key = "vk-alice"
        self.config = s.make_config(
            self.tmp,
            keys={self.key: KeyConfig(label="alice")},
            upstreams=[
                UpstreamConfig(name="fast", base_url=self.fast.base_url, api_key="fast-key", models=["gpt-4", "o3-"]),
                UpstreamConfig(name="deep", base_url=self.deep.base_url, api_key="deep-key", models=["deepseek-*"]),
            ],
        )
        self.proxy = s.ProxyHandle(build_gate(self.config))

    def tearDown(self) -> None:
        self.proxy.stop()
        self.fast.stop()
        self.deep.stop()

    def call(self, model: str) -> tuple[int, dict]:
        return s.post_json(
            self.proxy.base_url + "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "hi"}]},
            self.key,
        )

    def test_a_model_goes_to_the_upstream_that_serves_it(self) -> None:
        status, _ = self.call("deepseek-chat")
        self.assertEqual(status, 200)
        self.assertEqual(self.deep.received[0]["body"]["model"], "deepseek-chat")
        self.assertEqual(self.fast.received, [])

    def test_the_first_matching_upstream_wins(self) -> None:
        status, _ = self.call("gpt-4o-mini")
        self.assertEqual(status, 200)
        self.assertEqual(len(self.fast.received), 1)
        self.assertEqual(self.deep.received, [])

    def test_prefixes_and_globs_both_select(self) -> None:
        for model, name in (("gpt-4.1-nano", "fast"), ("o3-mini", "fast"), ("deepseek-reasoner", "deep")):
            with self.subTest(model=model):
                self.assertEqual(self.config.route(model).name, name)

    def test_each_upstream_uses_its_own_key(self) -> None:
        self.call("gpt-4o-mini")
        self.call("deepseek-chat")
        self.assertEqual(self.fast.received[0]["authorization"], "Bearer fast-key")
        self.assertEqual(self.deep.received[0]["authorization"], "Bearer deep-key")
        self.assertNotIn("deep-key", str(self.fast.received))
        self.assertNotIn("fast-key", str(self.deep.received))

    def test_no_upstream_serves_the_model_means_4xx_nothing_forwarded_nothing_ledged(self) -> None:
        status, payload = self.call("llama-3.1-70b-versatile")
        self.assertGreaterEqual(status, 400)
        self.assertLess(status, 500)
        self.assertEqual(payload["error"]["code"], "model_not_found")
        self.assertEqual(self.fast.received, [])
        self.assertEqual(self.deep.received, [])
        self.assertEqual(self.proxy.gate.ledger.rows(), [])

    def test_the_refusal_names_the_model_but_no_key(self) -> None:
        status, payload = self.call("llama-3.1-70b-versatile")
        self.assertEqual(status, 404)
        self.assertIn("llama-3.1-70b-versatile", payload["error"]["message"])
        self.assertNotIn("fast-key", json.dumps(payload))
        self.assertNotIn(self.key, json.dumps(payload))

    def test_an_empty_upstreams_field_routes_through_the_single_upstream(self) -> None:
        # The shipped configuration, unchanged: one base URL, every model.
        single = s.MockUpstream(stream_blocks=[], json_response=s.completion_json("gpt-4o", 3, 4))
        single.start()
        self.addCleanup(single.stop)
        config = s.make_config(
            pathlib.Path(tempfile.mkdtemp()),
            keys={self.key: KeyConfig(label="alice")},
            upstream=single.base_url,
        )
        proxy = s.ProxyHandle(build_gate(config))
        self.addCleanup(proxy.stop)
        status, _ = s.post_json(
            proxy.base_url + "/v1/chat/completions",
            {"model": "anything-at-all", "messages": []},
            self.key,
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(single.received), 1)
        self.assertEqual(single.received[0]["authorization"], f"Bearer {s.UPSTREAM_KEY}")
        self.assertEqual(s.await_rows(proxy.gate.ledger, 1)[0].model, "gpt-4o")

    def test_the_upstreams_list_takes_over_the_single_upstream(self) -> None:
        # make_config points the legacy fields at a closed port: if routing ever
        # fell back to them, this call would fail instead of reaching `fast`.
        status, _ = self.call("gpt-4o")
        self.assertEqual(status, 200)
        self.assertEqual(len(self.fast.received), 1)


class PatternTest(unittest.TestCase):
    def test_a_pattern_without_glob_characters_is_a_prefix(self) -> None:
        self.assertTrue(pattern_matches("gpt-4", "gpt-4o-mini"))
        self.assertFalse(pattern_matches("gpt-4", "chatgpt-4"))
        self.assertTrue(pattern_matches("", "anything"))

    def test_glob_patterns(self) -> None:
        self.assertTrue(pattern_matches("claude-*-sonnet-*", "claude-3-5-sonnet-20241022"))
        self.assertFalse(pattern_matches("claude-*-sonnet-*", "claude-3-5-haiku-20241022"))
        self.assertTrue(pattern_matches("o[34]-mini", "o3-mini"))


class UpstreamConfigFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = pathlib.Path(tempfile.mkdtemp())

    def write(self, payload: dict) -> str:
        path = self.dir / "spendgate.json"
        path.write_text(json.dumps(payload))
        return str(path)

    def base(self, **extra) -> dict:
        return {
            "upstream_base_url": "http://single.invalid/v1",
            "upstream_api_key": "single-key",
            "keys": {"vk-one": {"label": "one"}},
            **extra,
        }

    def test_the_list_is_parsed_in_order(self) -> None:
        path = self.write(
            self.base(
                upstreams=[
                    {"name": "fast", "base_url": "http://a.invalid/v1", "api_key": "k-a", "models": ["gpt-4"]},
                    {"name": "brain", "base_url": "http://b.invalid/v1", "api_key": "k-b", "models": ["o1", "o3"], "kind": "openai"},
                ]
            )
        )
        config = load_config(path)
        self.assertEqual([u.name for u in config.upstreams], ["fast", "brain"])
        self.assertEqual(config.route("gpt-4o").base_url, "http://a.invalid/v1")
        self.assertEqual(config.route("o3-mini").api_key, "k-b")
        self.assertEqual(config.route("gpt-4o").kind, "openai")

    def test_a_single_upstream_config_needs_no_list(self) -> None:
        config = load_config(self.write(self.base()))
        self.assertEqual(config.upstreams, [])
        self.assertEqual(config.route("whatever").name, "default")
        self.assertEqual(config.route("whatever").api_key, "single-key")

    def test_the_list_makes_the_legacy_fields_optional(self) -> None:
        payload = self.base(
            upstreams=[{"name": "only", "base_url": "http://a.invalid/v1", "api_key": "k", "models": ["gpt-4"]}]
        )
        del payload["upstream_base_url"]
        del payload["upstream_api_key"]
        config = load_config(self.write(payload))
        self.assertEqual(config.upstream_base_url, "")
        self.assertEqual(config.route("gpt-4o").name, "only")

    def test_per_upstream_key_env_override(self) -> None:
        path = self.write(
            self.base(
                upstreams=[{"name": "work-bot", "base_url": "http://a.invalid/v1", "api_key": "from-file", "models": ["gpt-4"]}]
            )
        )
        os.environ["SPENDGATE_WORK_BOT_API_KEY"] = "from-env"
        self.addCleanup(os.environ.pop, "SPENDGATE_WORK_BOT_API_KEY", None)
        self.assertEqual(load_config(path).route("gpt-4o").api_key, "from-env")

    def test_a_per_upstream_key_can_come_from_the_environment_alone(self) -> None:
        path = self.write(
            self.base(upstreams=[{"name": "primary", "base_url": "http://a.invalid/v1", "models": ["gpt-4"]}])
        )
        os.environ["SPENDGATE_PRIMARY_API_KEY"] = "from-env"
        self.addCleanup(os.environ.pop, "SPENDGATE_PRIMARY_API_KEY", None)
        self.assertEqual(load_config(path).route("gpt-4o").api_key, "from-env")

    def test_a_missing_upstream_key_names_the_variable_to_set(self) -> None:
        path = self.write(self.base(upstreams=[{"name": "primary", "base_url": "http://a.invalid/v1", "models": ["gpt-4"]}]))
        with self.assertRaises(ValueError) as caught:
            load_config(path)
        self.assertIn("SPENDGATE_PRIMARY_API_KEY", str(caught.exception))

    def test_a_broken_upstream_entry_is_refused_at_load(self) -> None:
        cases = {
            "not a list": self.base(upstreams={"name": "x"}),
            "empty list": self.base(upstreams=[]),
            "entry not an object": self.base(upstreams=["fast"]),
            "no name": self.base(upstreams=[{"base_url": "http://a/v1", "api_key": "k", "models": ["gpt-4"]}]),
            "no base_url": self.base(upstreams=[{"name": "a", "api_key": "k", "models": ["gpt-4"]}]),
            "no models": self.base(upstreams=[{"name": "a", "base_url": "http://a/v1", "api_key": "k"}]),
            "empty models": self.base(upstreams=[{"name": "a", "base_url": "http://a/v1", "api_key": "k", "models": []}]),
            "ragged models": self.base(upstreams=[{"name": "a", "base_url": "http://a/v1", "api_key": "k", "models": [7]}]),
            "unknown kind": self.base(upstreams=[{"name": "a", "base_url": "http://a/v1", "api_key": "k", "models": ["gpt-4"], "kind": "mistral"}]),
            "duplicate name": self.base(
                upstreams=[
                    {"name": "a", "base_url": "http://1/v1", "api_key": "k", "models": ["gpt-4"]},
                    {"name": "a", "base_url": "http://2/v1", "api_key": "k", "models": ["o1"]},
                ]
            ),
        }
        for label, payload in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    load_config(self.write(payload))


if __name__ == "__main__":
    unittest.main()
