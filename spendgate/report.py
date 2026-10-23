"""Offline spend reports, straight from the ledger file.

No server, no daemon, no database service: the ledger is a SQLite file, so this
reads it the way ``sqlite3`` does and prints totals per key, per model and per
UTC day.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .ledger import GroupTotal, Ledger

MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

GROUPS = ("key", "model", "day")


def collect(ledger: Ledger, month: str | None = None) -> dict[str, Any]:
    """Everything the report shows, as plain data (this is what --json emits)."""
    totals = {group: ledger.summarize(group, month) for group in GROUPS}
    costed = [total.cost for total in totals["key"] if total.cost is not None]
    return {
        "month": month,
        "calls": ledger.row_count(month),
        "uncosted_calls": sum(total.uncosted_calls for total in totals["key"]),
        "prompt_tokens": sum(total.prompt_tokens for total in totals["key"]),
        "completion_tokens": sum(total.completion_tokens for total in totals["key"]),
        "total_cost": sum(costed) if costed else None,
        "by_key": [_as_dict(total) for total in totals["key"]],
        "by_model": [_as_dict(total) for total in totals["model"]],
        "by_day": [_as_dict(total) for total in totals["day"]],
    }


def _as_dict(total: GroupTotal) -> dict[str, Any]:
    return {
        "group": total.group,
        "calls": total.calls,
        "prompt_tokens": total.prompt_tokens,
        "completion_tokens": total.completion_tokens,
        "cost": total.cost,
        "uncosted_calls": total.uncosted_calls,
    }


def render(report: dict[str, Any], ledger_path: str) -> str:
    """Human-readable text: one table per grouping."""
    if report["calls"] == 0:
        scope = f" in {report['month']}" if report["month"] else ""
        return f"No recorded calls{scope} in {ledger_path}."

    lines = [
        f"spendgate report — {ledger_path}",
        f"month: {report['month'] or 'all time'}   calls: {report['calls']}",
        f"tokens: {report['prompt_tokens']} prompt / {report['completion_tokens']} completion",
        f"total cost: {_money(report['total_cost'])}",
    ]
    if report["uncosted_calls"]:
        lines.append(
            f"({report['uncosted_calls']} call(s) have no cost: unknown model or the "
            "upstream returned no usage)"
        )

    for title, field in (("By key", "by_key"), ("By model", "by_model"), ("By day", "by_day")):
        lines.append("")
        lines.append(title)
        lines.append(f"  {'':<24}{'calls':>7}{'prompt':>10}{'complet':>10}{'cost usd':>12}")
        for row in report[field]:
            lines.append(
                f"  {_clip(row['group']):<24}{row['calls']:>7}"
                f"{row['prompt_tokens']:>10}{row['completion_tokens']:>10}{_money(row['cost']):>12}"
            )
    return "\n".join(lines) + "\n"


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _clip(text: str) -> str:
    return text if len(text) <= 24 else text[:23] + "…"


def emit(report: dict[str, Any], ledger_path: str, as_json: bool) -> str:
    if as_json:
        return json.dumps(report, indent=2, sort_keys=True)
    return render(report, ledger_path)
