"""Command line entry point: `spendgate serve` and `spendgate report`."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Sequence

from . import __version__
from .config import DEFAULT_LEDGER_PATH, Config, load_config
from .ledger import Ledger
from .report import MONTH_PATTERN, collect, emit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spendgate",
        description="OpenAI-compatible proxy with a SQLite spend ledger and per-key budgets.",
    )
    parser.add_argument("--version", action="version", version=f"spendgate {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the proxy (Ctrl-C to stop)")
    serve.add_argument("-c", "--config", help="config file (default: $SPENDGATE_CONFIG or ./spendgate.json)")
    serve.add_argument("-v", "--verbose", action="store_true", help="log every forwarded request")

    report = sub.add_parser("report", help="summarise a ledger file; the proxy does not need to be running")
    report.add_argument(
        "-d",
        "--db",
        default=None,
        help=f"ledger file (default: $SPENDGATE_LEDGER_PATH or ./{DEFAULT_LEDGER_PATH})",
    )
    report.add_argument("-c", "--config", help="read the ledger path from this config file instead")
    report.add_argument("--month", help="restrict to one calendar month, as YYYY-MM")
    report.add_argument("--json", action="store_true", help="emit JSON instead of tables")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    return _report(args)


def _serve(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as error:
        print(f"spendgate: {error}", file=sys.stderr)
        return 1
    from .proxy import run  # imported here so `report` stays free of the HTTP stack

    try:
        run(config)
    except OSError as error:
        print(f"spendgate: cannot listen on {config.listen_host}:{config.listen_port}: {error}", file=sys.stderr)
        return 1
    return 0


def _report(args: argparse.Namespace) -> int:
    if args.month and not MONTH_PATTERN.match(args.month):
        print(f"spendgate: --month must look like 2026-01, got {args.month!r}", file=sys.stderr)
        return 1
    try:
        path = _report_ledger_path(args)
    except (OSError, ValueError) as error:
        print(f"spendgate: {error}", file=sys.stderr)
        return 1
    try:
        ledger = Ledger(path, create=False)
    except FileNotFoundError:
        print(f"spendgate: no ledger at {path}. Has the proxy written any calls yet?", file=sys.stderr)
        return 1
    print(emit(collect(ledger, args.month), path, args.json))
    return 0


def _report_ledger_path(args: argparse.Namespace) -> str:
    if args.db:
        return args.db
    if args.config:
        config: Config = load_config(args.config)
        return config.ledger_path
    return os.environ.get("SPENDGATE_LEDGER_PATH", DEFAULT_LEDGER_PATH)
