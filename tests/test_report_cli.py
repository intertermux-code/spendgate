"""Acceptance (g): the offline report CLI against a seeded ledger."""

from __future__ import annotations

import io
import json
import pathlib
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from spendgate.cli import main
from spendgate.ledger import Ledger

# Hand-computed by hand, from these six rows:
#   alice  gpt-4o-mini   1000p/500c   $0.00045   2026-10-03
#   alice  gpt-4o-mini    200p/100c   $0.00021   2026-10-04
#   alice  gpt-4o         400p/200c   $0.00300   2026-10-03
#   bob    gpt-4o-mini    100p/100c   $0.00021   2026-10-04
#   carol  mystery-llm    500p/500c   NULL       2026-10-04
#   alice  gpt-4o-mini   1000p/500c   $0.00045   2026-09-30   <- other month
ROWS = [
    dict(key_label="alice", model="gpt-4o-mini", prompt_tokens=1000, completion_tokens=500, cost=0.00045, timestamp="2026-10-03T08:00:00+00:00"),
    dict(key_label="alice", model="gpt-4o-mini", prompt_tokens=200, completion_tokens=100, cost=0.00021, timestamp="2026-10-04T09:00:00+00:00"),
    dict(key_label="alice", model="gpt-4o", prompt_tokens=400, completion_tokens=200, cost=0.003, timestamp="2026-10-03T10:00:00+00:00"),
    dict(key_label="bob", model="gpt-4o-mini", prompt_tokens=100, completion_tokens=100, cost=0.00021, timestamp="2026-10-04T11:00:00+00:00"),
    dict(key_label="carol", model="mystery-llm", prompt_tokens=500, completion_tokens=500, cost=None, timestamp="2026-10-04T12:00:00+00:00"),
    dict(key_label="alice", model="gpt-4o-mini", prompt_tokens=1000, completion_tokens=500, cost=0.00045, timestamp="2026-09-30T23:00:00+00:00"),
]

OCTOBER = {
    "calls": 5,
    "prompt_tokens": 2200,
    "completion_tokens": 1400,
    "total_cost": 0.00045 + 0.00021 + 0.003 + 0.00021,
    "uncosted_calls": 1,
}


class ReportTest(unittest.TestCase):
    def setUp(self) -> None:
        path = pathlib.Path(tempfile.mkdtemp()) / "ledger.sqlite"
        ledger = Ledger(str(path))
        for row in ROWS:
            ledger.record(**row)
        self.path = str(path)

    def run_cli(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["report", "-d", self.path, *argv])
        return code, out.getvalue()

    def report_json(self, *argv: str) -> dict:
        code, text = self.run_cli("--json", *argv)
        self.assertEqual(code, 0, text)
        return json.loads(text)

    def test_totals_by_key(self) -> None:
        report = self.report_json("--month", "2026-10")
        by_key = {row["group"]: row for row in report["by_key"]}
        self.assertEqual(by_key["alice"]["calls"], 3)
        self.assertEqual(by_key["alice"]["prompt_tokens"], 1600)
        self.assertEqual(by_key["alice"]["completion_tokens"], 800)
        self.assertAlmostEqual(by_key["alice"]["cost"], 0.00045 + 0.00021 + 0.003, places=12)
        self.assertAlmostEqual(by_key["bob"]["cost"], 0.00021, places=12)
        self.assertIsNone(by_key["carol"]["cost"])
        self.assertEqual(by_key["carol"]["uncosted_calls"], 1)

    def test_totals_by_model(self) -> None:
        report = self.report_json("--month", "2026-10")
        by_model = {row["group"]: row for row in report["by_model"]}
        self.assertEqual(by_model["gpt-4o-mini"]["calls"], 3)
        self.assertEqual(by_model["gpt-4o-mini"]["prompt_tokens"], 1300)
        self.assertAlmostEqual(by_model["gpt-4o-mini"]["cost"], 0.00045 + 0.00021 + 0.00021, places=12)
        self.assertAlmostEqual(by_model["gpt-4o"]["cost"], 0.003, places=12)
        self.assertIsNone(by_model["mystery-llm"]["cost"])

    def test_totals_by_day(self) -> None:
        report = self.report_json("--month", "2026-10")
        by_day = {row["group"]: row for row in report["by_day"]}
        self.assertEqual(sorted(by_day), ["2026-10-03", "2026-10-04"])
        self.assertEqual(by_day["2026-10-03"]["calls"], 2)
        self.assertAlmostEqual(by_day["2026-10-03"]["cost"], 0.00045 + 0.003, places=12)
        self.assertEqual(by_day["2026-10-04"]["calls"], 3)
        self.assertEqual(by_day["2026-10-04"]["prompt_tokens"], 800)

    def test_month_option_excludes_other_months(self) -> None:
        october = self.report_json("--month", "2026-10")
        self.assertEqual(october["calls"], OCTOBER["calls"])
        self.assertEqual(october["prompt_tokens"], OCTOBER["prompt_tokens"])
        self.assertEqual(october["completion_tokens"], OCTOBER["completion_tokens"])
        self.assertAlmostEqual(october["total_cost"], OCTOBER["total_cost"], places=12)
        self.assertAlmostEqual(self.report_json()["total_cost"], OCTOBER["total_cost"] + 0.00045, places=12)
        self.assertEqual(self.report_json("--month", "2026-09")["calls"], 1)

    def test_json_output_parses_and_is_self_describing(self) -> None:
        text = self.run_cli("--json")[1]
        report = json.loads(text)
        self.assertEqual(report["month"], None)
        self.assertEqual(set(report), {"month", "calls", "uncosted_calls", "prompt_tokens", "completion_tokens", "total_cost", "by_key", "by_model", "by_day"})

    def test_text_report_shows_every_grouping(self) -> None:
        code, text = self.run_cli("--month", "2026-10")
        self.assertEqual(code, 0)
        for heading in ("By key", "By model", "By day", "total cost"):
            self.assertIn(heading, text)
        self.assertIn("alice", text)
        self.assertIn("2026-10-03", text)
        self.assertIn("n/a", text, "carol's unknown-model spend must read as unknown, not zero")
        self.assertIn("1 call(s) have no cost", text)

    def test_empty_month_says_so(self) -> None:
        code, text = self.run_cli("--month", "2030-01")
        self.assertEqual(code, 0)
        self.assertIn("No recorded calls", text)

    def test_missing_ledger_file_fails_loudly(self) -> None:
        err = io.StringIO()
        with redirect_stderr(err):
            code = main(["report", "-d", str(pathlib.Path(tempfile.mkdtemp()) / "absent.sqlite")])
        self.assertEqual(code, 1)
        self.assertIn("no ledger", err.getvalue())

    def test_bad_month_is_rejected(self) -> None:
        err = io.StringIO()
        with redirect_stderr(err):
            code = main(["report", "-d", self.path, "--month", "october"])
        self.assertEqual(code, 1)
        self.assertIn("2026-01", err.getvalue())

    def test_report_can_take_its_path_from_a_config(self) -> None:
        config = pathlib.Path(self.path).parent / "spendgate.json"
        config.write_text(
            json.dumps(
                {
                    "upstream_base_url": "https://api.openai.com/v1",
                    "upstream_api_key": "placeholder",
                    "ledger_path": self.path,
                    "keys": {"vk": {"label": "alice"}},
                }
            )
        )
        err = io.StringIO()
        with redirect_stdout(io.StringIO()) as out, redirect_stderr(err):
            code = main(["report", "-c", str(config)])
        self.assertEqual(code, 0, err.getvalue())
        self.assertIn("By key", out.getvalue())

    def test_a_broken_config_fails_cleanly_rather_than_tracing_back(self) -> None:
        err = io.StringIO()
        with redirect_stderr(err):
            code = main(["report", "-c", str(pathlib.Path(tempfile.mkdtemp()) / "absent.json")])
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", err.getvalue())


    def test_reading_a_ledger_never_creates_or_migrates_one(self) -> None:
        # The pitch is a file you scp off the box: reporting on it must be a
        # purely read operation.
        other = Ledger(self.path)
        other.record(key_label="zoe", model="gpt-4o", prompt_tokens=1, completion_tokens=1, cost=0.0)
        self.assertEqual(self.run_cli()[1].count("zoe"), 1)
        missing = str(pathlib.Path(tempfile.mkdtemp()) / "deep" / "absent.sqlite")
        with self.assertRaises(FileNotFoundError):
            Ledger(missing, create=False)
        self.assertFalse(pathlib.Path(missing).exists())


if __name__ == "__main__":
    unittest.main()
