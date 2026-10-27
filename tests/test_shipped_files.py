"""The files a user reads and installs must be present and internally true."""

from __future__ import annotations

import json
import pathlib
import re
import unittest

from spendgate.cli import build_parser
from spendgate.config import load_config
from spendgate.prices import load_prices

ROOT = pathlib.Path(__file__).resolve().parent.parent

SHIPPED = (
    "README.md",
    "LICENSE",
    ".gitignore",
    "config.example.json",
    "pyproject.toml",
    "spendgate/prices.json",
    ".github/workflows/test.yml",
    "scripts/smoke.py",
)


class ShippedFilesTest(unittest.TestCase):
    def test_every_promised_file_exists(self) -> None:
        for relative in SHIPPED:
            self.assertTrue((ROOT / relative).is_file(), f"missing {relative}")

    def test_the_example_config_is_a_config(self) -> None:
        # A template that will not load is worse than no template.
        config = load_config(str(ROOT / "config.example.json"))
        self.assertTrue(config.upstream_url.endswith("/v1/"))
        self.assertEqual(len(config.keys), 3)
        self.assertEqual(config.soft_limit_fraction, 0.8)

    def test_the_example_config_holds_placeholders_only(self) -> None:
        raw = (ROOT / "config.example.json").read_text()
        self.assertIn("REPLACE_ME", raw)
        self.assertNotIn("sk-", raw)
        self.assertNotIn(str(pathlib.Path.home()), raw)

    def test_the_quickstart_key_is_a_key_the_template_ships(self) -> None:
        # The quickstart tells the reader to copy config.example.json, then
        # hands them a client snippet: a virtual key the template does not
        # define is refused at the gate, and nothing in the error says why.
        readme = (ROOT / "README.md").read_text()
        keys = json.loads((ROOT / "config.example.json").read_text())["keys"]
        used = re.findall(r'api_key="([^"]+)"', readme) + re.findall(r"Bearer ([A-Za-z0-9_-]+)", readme)
        self.assertTrue(used)
        for key in used:
            self.assertIn(key, keys)

    def test_the_models_the_readme_shows_are_priceable(self) -> None:
        prices = load_prices(ROOT / "spendgate" / "prices.json")
        readme = (ROOT / "README.md").read_text()
        for model in ("gpt-4o-mini", "gpt-4o"):
            self.assertIn(model, prices)
            self.assertIn(model, readme)

    def test_the_ci_workflow_tests_what_this_suite_is(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text()
        self.assertIn("python -m unittest discover", workflow)
        self.assertIn("pip install -e .", workflow)
        for version in ("3.11", "3.12", "3.13"):
            self.assertIn(version, workflow)

    def test_no_source_file_hardcodes_a_home_directory(self) -> None:
        for path in (ROOT / "spendgate").glob("*.py"):
            self.assertNotIn("/Users/", path.read_text(), path)


class CliTest(unittest.TestCase):
    def test_both_subcommands_are_documented_and_parse(self) -> None:
        parser = build_parser()
        serve = parser.parse_args(["serve", "-c", "x.json"])
        self.assertEqual(serve.config, "x.json")
        report = parser.parse_args(["report", "--month", "2026-10", "--json"])
        self.assertEqual((report.month, report.json), ("2026-10", True))

    def test_a_command_is_required(self) -> None:
        import io
        from contextlib import redirect_stderr

        errors = io.StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit):
            build_parser().parse_args([])
        self.assertIn("the following arguments are required: command", errors.getvalue())

    def test_serve_reports_a_bad_config_instead_of_traceback(self) -> None:
        import io
        from contextlib import redirect_stderr

        from spendgate.cli import main

        errors = io.StringIO()
        with redirect_stderr(errors):
            code = main(["serve", "-c", str(ROOT / "does-not-exist.json")])
        self.assertEqual(code, 1)
        self.assertIn("config.example.json", errors.getvalue())

    def test_serve_reports_malformed_keys_instead_of_traceback(self) -> None:
        import io
        import tempfile
        from contextlib import redirect_stderr

        from spendgate.cli import main

        path = pathlib.Path(tempfile.mkdtemp()) / "spendgate.json"
        path.write_text(
            json.dumps(
                {
                    "upstream_base_url": "http://x/v1",
                    "upstream_api_key": "k",
                    "keys": ["vk-one", "vk-two"],
                }
            )
        )
        errors = io.StringIO()
        with redirect_stderr(errors):
            code = main(["serve", "-c", str(path)])
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", errors.getvalue())
        self.assertIn("spendgate: config: 'keys' must be an object", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
