"""Anthropic as an upstream kind: what goes out, what comes back, what is metered.

The unit tests cover the translation itself; the proxy tests drive the same
translation through a real socket against a local stand-in for the Messages
API, because a dialect that only works in a unit test is not a dialect.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from tests import support as s
from spendgate import anthropic
from spendgate.anthropic import DEFAULT_MAX_TOKENS, UnsupportedRequest, to_request
from spendgate.config import KeyConfig, UpstreamConfig
from spendgate.proxy import build_gate
from spendgate.upstream import ANTHROPIC_VERSION

MODEL = "claude-3-5-sonnet-20241022"
# 1000 prompt tokens at $3.00/M plus 250 completion at $15.00/M.
EXPECTED_COST = (1000 * 3.00 + 250 * 15.00) / 1_000_000


def chat(**extra) -> dict:
    body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
    body.update(extra)
    return body


class RequestTranslationTest(unittest.TestCase):
    def test_system_messages_become_the_system_field(self) -> None:
        translated = to_request(
            chat(
                messages=[
                    {"role": "system", "content": "Be brief."},
                    {"role": "developer", "content": "Be kind."},
                    {"role": "user", "content": "hi"},
                ]
            )
        )
        self.assertEqual(translated["system"], "Be brief.\n\nBe kind.")
        self.assertEqual([turn["role"] for turn in translated["messages"]], ["user"])

    def test_no_system_messages_means_no_system_field(self) -> None:
        self.assertNotIn("system", to_request(chat()))

    def test_content_parts_are_flattened_to_text(self) -> None:
        translated = to_request(chat(messages=[{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]))
        self.assertEqual(translated["messages"][0]["content"], [{"type": "text", "text": "ab"}])

    def test_consecutive_turns_for_one_role_are_merged(self) -> None:
        # Anthropic wants alternating roles; two user messages in a row are one
        # turn with two blocks, not a request the API would reject.
        translated = to_request(
            chat(messages=[{"role": "user", "content": "one"}, {"role": "user", "content": "two"}, {"role": "assistant", "content": "three"}, {"role": "user", "content": "four"}])
        )
        self.assertEqual([turn["role"] for turn in translated["messages"]], ["user", "assistant", "user"])
        self.assertEqual(len(translated["messages"][0]["content"]), 2)

    def test_max_tokens_defaults_when_the_client_omits_it(self) -> None:
        self.assertEqual(to_request(chat())["max_tokens"], DEFAULT_MAX_TOKENS)

    def test_max_tokens_is_passed_through_when_given(self) -> None:
        self.assertEqual(to_request(chat(max_tokens=77))["max_tokens"], 77)
        self.assertEqual(to_request(chat(max_tokens=0))["max_tokens"], DEFAULT_MAX_TOKENS)
        self.assertEqual(to_request(chat(max_tokens="lots"))["max_tokens"], DEFAULT_MAX_TOKENS)

    def test_sampling_options_carry_over(self) -> None:
        translated = to_request(chat(temperature=0.2, top_p=0.9, stop="END", stream=True))
        self.assertEqual((translated["temperature"], translated["top_p"]), (0.2, 0.9))
        self.assertEqual(translated["stop_sequences"], ["END"])
        self.assertIs(translated["stream"], True)
        self.assertNotIn("stream", to_request(chat()))

    def test_a_list_of_stop_sequences_is_kept_as_a_list(self) -> None:
        self.assertEqual(to_request(chat(stop=["a", "b"]))["stop_sequences"], ["a", "b"])
        self.assertNotIn("stop_sequences", to_request(chat(stop=[None, ""])))

    def test_the_original_body_is_not_mutated(self) -> None:
        body = chat(max_tokens=10)
        to_request(body)
        self.assertEqual(body, chat(max_tokens=10))

    def test_an_empty_conversation_is_refused(self) -> None:
        with self.assertRaises(UnsupportedRequest):
            to_request(chat(messages=[{"role": "system", "content": "only instructions"}]))
        with self.assertRaises(UnsupportedRequest):
            to_request(chat(messages=[]))

    def test_tool_calls_are_refused_not_dropped(self) -> None:
        cases = [
            chat(tools=[{"type": "function", "function": {"name": "lookup"}}]),
            chat(tool_choice="auto", tools=[{"name": "x"}]),
            chat(functions=[{"name": "x"}]),
            chat(messages=[{"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]}]),
            chat(messages=[{"role": "tool", "tool_call_id": "1", "content": "result"}]),
        ]
        for index, body in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(UnsupportedRequest) as caught:
                    to_request(body)
                self.assertIn("text only", str(caught.exception))

    def test_a_request_for_several_answers_is_refused(self) -> None:
        with self.assertRaises(UnsupportedRequest):
            to_request(chat(n=3))
        to_request(chat(n=1))

    def test_an_empty_turn_is_refused(self) -> None:
        for messages in ([{"role": "user", "content": ""}], [{"role": "assistant", "content": None}, {"role": "user", "content": "hi"}]):
            with self.subTest(messages=messages):
                with self.assertRaises(UnsupportedRequest):
                    to_request(chat(messages=messages))

    def test_non_text_content_is_refused(self) -> None:
        with self.assertRaises(UnsupportedRequest):
            to_request(chat(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://x/y.png"}}]}]))


class ResponseTranslationTest(unittest.TestCase):
    def test_a_message_becomes_a_chat_completion(self) -> None:
        completion = anthropic.to_completion(
            {"id": "msg_1", "type": "message", "model": MODEL, "content": [{"type": "text", "text": "Hel"}, {"type": "text", "text": "lo"}], "stop_reason": "end_turn", "usage": {"input_tokens": 11, "output_tokens": 5}}
        )
        self.assertEqual(completion["object"], "chat.completion")
        self.assertEqual(completion["model"], MODEL)
        self.assertEqual(completion["choices"][0]["message"], {"role": "assistant", "content": "Hello"})
        self.assertEqual(completion["choices"][0]["finish_reason"], "stop")
        self.assertEqual(completion["usage"], {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16})

    def test_stop_reasons_map_to_openai_ones(self) -> None:
        def finish(stop_reason):
            payload = {"type": "message", "model": MODEL, "content": [], "usage": {"input_tokens": 1, "output_tokens": 1}}
            if stop_reason is not None:
                payload["stop_reason"] = stop_reason
            return anthropic.to_completion(payload)["choices"][0]["finish_reason"]

        self.assertEqual(finish("end_turn"), "stop")
        self.assertEqual(finish("max_tokens"), "length")
        self.assertEqual(finish("stop_sequence"), "stop")
        self.assertEqual(finish(None), "stop")

    def test_usage_reads_anthropics_own_names(self) -> None:
        usage = anthropic.usage_of({"model": MODEL, "usage": {"input_tokens": 30, "output_tokens": 7}})
        self.assertEqual((usage.prompt_tokens, usage.completion_tokens, usage.model), (30, 7, MODEL))
        self.assertIsNone(anthropic.usage_of({"model": MODEL}))
        self.assertIsNone(anthropic.usage_of({"usage": {"input_tokens": "many"}}))

    def test_an_error_body_is_translated(self) -> None:
        raw = json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": "boom"}}).encode()
        translated = json.loads(anthropic.to_error(raw, 400))
        self.assertEqual(translated["error"]["message"], "boom")
        self.assertEqual(translated["error"]["type"], "invalid_request_error")

    def test_a_body_that_is_not_an_anthropic_error_is_left_alone(self) -> None:
        self.assertIsNone(anthropic.to_error(b"<html>502</html>", 502))
        self.assertIsNone(anthropic.to_error(b'{"status": "ok"}', 200))


class StreamTranslatorTest(unittest.TestCase):
    def translate(self, blocks: list[str], bite: int = 0) -> str:
        translator = anthropic.StreamTranslator()
        out = []
        for block in blocks:
            raw = block.encode()
            if bite:
                for i in range(0, len(raw), bite):
                    out.append(translator.feed(raw[i : i + bite]))
            else:
                out.append(translator.feed(raw))
        out.append(translator.close())
        return b"".join(out).decode()

    def chunks(self, text: str) -> list[dict]:
        return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]

    def test_text_deltas_become_openai_chunks(self) -> None:
        body = self.translate(s.anthropic_stream(MODEL, ["Hel", "lo"], 12, 4))
        chunks = self.chunks(body)
        self.assertEqual("".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks), "Hello")
        self.assertTrue(all(chunk["object"] == "chat.completion.chunk" for chunk in chunks))
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")

    def test_message_stop_becomes_a_finish_reason_and_a_usage_chunk(self) -> None:
        chunks = self.chunks(self.translate(s.anthropic_stream(MODEL, ["hi"], 12, 4)))
        last = chunks[-1]
        self.assertEqual(last["choices"][0]["finish_reason"], "stop")
        self.assertEqual(last["usage"], {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16})
        self.assertTrue(self.translate(s.anthropic_stream(MODEL, ["hi"], 12, 4)).endswith("data: [DONE]\n\n"))

    def test_usage_is_captured_for_the_ledger(self) -> None:
        translator = anthropic.StreamTranslator()
        for block in s.anthropic_stream(MODEL, ["hi"], 12, 4):
            translator.feed(block.encode())
        self.assertEqual((translator.usage.prompt_tokens, translator.usage.completion_tokens, translator.usage.model), (12, 4, MODEL))

    def test_no_anthropic_event_names_reach_the_client(self) -> None:
        body = self.translate(s.anthropic_stream(MODEL, ["hi"], 12, 4))
        for name in ("content_block_delta", "message_start", "message_stop", "input_json_delta", "ping"):
            self.assertNotIn(name, body)

    def test_events_survive_being_split_mid_line(self) -> None:
        body = self.translate(s.anthropic_stream(MODEL, ["Hel", "lo"], 12, 4), bite=3)
        chunks = self.chunks(body)
        self.assertEqual("".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks), "Hello")

    def test_a_stream_without_usage_reports_none(self) -> None:
        translator = anthropic.StreamTranslator()
        translator.feed(b"event: ping\ndata: {\"type\": \"ping\"}\n\n")
        self.assertIsNone(translator.usage)


class AnthropicProxyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mock = s.MockAnthropic(stream_blocks=[], json_response=s.anthropic_message(MODEL, 1000, 250))
        self.mock.start()
        self.key = "vk-alice"
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.config = s.make_config(
            self.tmp,
            keys={self.key: KeyConfig(label="alice")},
            upstreams=[
                UpstreamConfig(name="claude", base_url=self.mock.base_url, api_key=s.ANTHROPIC_KEY, models=["claude-"], kind="anthropic")
            ],
        )
        self.proxy = s.ProxyHandle(build_gate(self.config))

    def tearDown(self) -> None:
        self.proxy.stop()
        self.mock.stop()

    def stream(self, body: dict) -> str:
        sock = s.raw_stream_request(self.proxy.base_url + "/v1/chat/completions", body, self.key)
        reader = s.SocketReader(sock)
        head = reader.read_until(b"\r\n\r\n")
        self.assertIn(b"200 OK", head.split(b"\r\n", 1)[0])
        self.assertIn(b"text/event-stream", head)
        text = reader.read_all().decode()
        sock.close()
        return text

    def test_a_plain_call_is_translated_both_ways(self) -> None:
        status, payload = s.post_json(self.proxy.base_url + "/v1/chat/completions", chat(), self.key)
        self.assertEqual(status, 200)
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"]["content"], "hello")
        self.assertEqual(payload["usage"], {"prompt_tokens": 1000, "completion_tokens": 250, "total_tokens": 1250})

    def test_the_request_reaches_the_messages_api_with_its_own_headers(self) -> None:
        s.post_json(self.proxy.base_url + "/v1/chat/completions", chat(), self.key)
        seen = self.mock.received[0]
        self.assertEqual(seen["path"], "/v1/messages")
        self.assertEqual(seen["x_api_key"], s.ANTHROPIC_KEY)
        self.assertEqual(seen["anthropic_version"], ANTHROPIC_VERSION)
        self.assertEqual(seen["authorization"], "")

    def test_the_openai_body_arrives_in_anthropic_shape(self) -> None:
        s.post_json(
            self.proxy.base_url + "/v1/chat/completions",
            chat(messages=[{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}], max_tokens=64, temperature=0.5),
            self.key,
        )
        seen = self.mock.received[0]["body"]
        self.assertEqual(seen["system"], "be nice")
        self.assertEqual(seen["messages"], [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        self.assertEqual(seen["max_tokens"], 64)
        self.assertEqual(seen["temperature"], 0.5)

    def test_a_client_that_omits_max_tokens_gets_the_default(self) -> None:
        s.post_json(self.proxy.base_url + "/v1/chat/completions", chat(), self.key)
        self.assertEqual(self.mock.received[0]["body"]["max_tokens"], DEFAULT_MAX_TOKENS)

    def test_the_tokens_anthropic_reports_are_ledged_and_priceable(self) -> None:
        s.post_json(self.proxy.base_url + "/v1/chat/completions", chat(), self.key)
        row = s.await_rows(self.proxy.gate.ledger, 1)[0]
        self.assertEqual((row.key_label, row.model, row.prompt_tokens, row.completion_tokens), ("alice", MODEL, 1000, 250))
        self.assertAlmostEqual(row.cost, EXPECTED_COST, places=12)

    def test_a_streamed_answer_arrives_as_openai_chunks(self) -> None:
        self.mock.stream_blocks = s.anthropic_stream(MODEL, ["Hel", "lo ", "world"], 30, 9)
        body = self.stream(chat(stream=True))
        chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
        self.assertEqual("".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks), "Hello world")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[-1]["usage"]["prompt_tokens"], 30)
        self.assertTrue(body.endswith("data: [DONE]\n\n"))
        self.assertNotIn("content_block_delta", body)

    def test_a_streamed_answer_is_metered_from_the_events(self) -> None:
        self.mock.stream_blocks = s.anthropic_stream(MODEL, ["hi"], 30, 9)
        self.stream(chat(stream=True))
        row = s.await_rows(self.proxy.gate.ledger, 1)[0]
        self.assertEqual((row.model, row.prompt_tokens, row.completion_tokens), (MODEL, 30, 9))
        self.assertAlmostEqual(row.cost, (30 * 3.00 + 9 * 15.00) / 1_000_000, places=12)

    def test_a_streaming_request_asks_anthropic_for_a_stream(self) -> None:
        self.mock.stream_blocks = s.anthropic_stream(MODEL, ["hi"], 1, 1)
        self.stream(chat(stream=True))
        seen = self.mock.received[0]["body"]
        self.assertIs(seen["stream"], True)
        # stream_options is an OpenAI-only field; Anthropic would reject it.
        self.assertNotIn("stream_options", seen)

    def test_tool_calls_are_refused_before_anything_is_forwarded(self) -> None:
        status, payload = s.post_json(
            self.proxy.base_url + "/v1/chat/completions",
            chat(tools=[{"type": "function", "function": {"name": "lookup"}}]),
            self.key,
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "unsupported_for_upstream")
        self.assertIn("text only", payload["error"]["message"])
        self.assertEqual(self.mock.received, [])
        self.assertEqual(self.proxy.gate.ledger.rows(), [])

    def test_an_upstream_error_comes_back_in_the_shape_openai_clients_expect(self) -> None:
        self.mock.json_status = 400
        self.mock.json_response = {"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens too large"}}
        status, payload = s.post_json(self.proxy.base_url + "/v1/chat/completions", chat(), self.key)
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["message"], "max_tokens too large")
        self.assertEqual(self.proxy.gate.ledger.rows(), [])

    def test_the_upstream_key_never_surfaces_in_an_answer(self) -> None:
        status, payload = s.post_json(self.proxy.base_url + "/v1/chat/completions", chat(), self.key)
        self.assertEqual(status, 200)
        self.assertNotIn(s.ANTHROPIC_KEY, json.dumps(payload))
        self.assertNotIn(s.ANTHROPIC_KEY, self.stream(chat(stream=True)))


if __name__ == "__main__":
    unittest.main()
