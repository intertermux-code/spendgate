"""The version a user installs must be the version the tool reports.

Three files claim a version — pyproject.toml, the package, and what the CLI
prints — and the tag that triggers a publish is a fourth. They drift silently
and a wheel then ships labelled with a number the code disagrees with, so the
agreement is asserted rather than trusted.
"""

from __future__ import annotations

import io
import pathlib
import tomllib
import unittest
from contextlib import redirect_stdout

import spendgate
from spendgate.cli import main

ROOT = pathlib.Path(__file__).resolve().parent.parent


def pyproject_version() -> str:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return config["project"]["version"]


class VersionSyncTest(unittest.TestCase):
    def _reported_by_the_cli(self) -> tuple[str, int]:
        """Capture `spendgate --version`, which prints and then exits."""
        out = io.StringIO()
        code = None
        with redirect_stdout(out):
            try:
                main(["--version"])
            except SystemExit as stop:
                code = stop.code
        return out.getvalue().strip(), 0 if code is None else code

    def test_pyproject_and_the_package_are_the_same_version(self) -> None:
        self.assertEqual(pyproject_version(), spendgate.__version__)

    def test_the_cli_reports_the_package_version(self) -> None:
        printed, code = self._reported_by_the_cli()
        self.assertEqual(code, 0)
        self.assertEqual(printed, f"spendgate {spendgate.__version__}")

    def test_all_three_names_agree(self) -> None:
        printed, _ = self._reported_by_the_cli()
        number = pyproject_version()
        self.assertEqual(
            {number, spendgate.__version__, printed.removeprefix("spendgate ")},
            {number},
        )


if __name__ == "__main__":
    unittest.main()
