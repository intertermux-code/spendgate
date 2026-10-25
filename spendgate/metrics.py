"""Prometheus text exposition, read straight off the ledger.

A counter kept only in memory restarts at zero while the money is still in the
SQLite file, and a dashboard that dips on every deploy is worse than none. So
the families that describe *completed* calls are aggregated from the ledger at
scrape time; the in-memory part holds only the outcomes the ledger never learns
about — refusals, unroutable models, unreachable upstreams. The two sets do not
overlap, so they cannot double count.

Label values are key *labels*, never key material. Nothing here can name a
virtual key or an upstream token.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .budgets import month_of
from .config import Config
from .ledger import Ledger

CONTENT_TYPE = "text/plain; version=0.0.4"

# The status a ledger row stands for: a call the proxy completed and metered.
LEDGER_STATUS = "200"

# Outcomes with no attributable key or model still count, under names that say
# so instead of echoing whatever the client presented.
ANONYMOUS_KEY = "(anonymous)"
UNKNOWN_MODEL = "(unknown)"

# name -> (type, help, label names)
FAMILIES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "requests_total": (
        "counter",
        "Chat completions the proxy handled, by key label, model and HTTP status.",
        ("key", "model", "status"),
    ),
    "tokens_total": (
        "counter",
        "Tokens the upstream reported, by key label, model and kind.",
        ("key", "model", "kind"),
    ),
    "cost_usd_total": (
        "counter",
        "Ledger cost in USD, by key label and model; absent where no price is known.",
        ("key", "model"),
    ),
    "budget_fraction": (
        "gauge",
        "Spend in the current UTC month divided by the key's monthly budget.",
        ("key",),
    ),
}

PREFIX = "spendgate_"


@dataclass(frozen=True)
class Outcome:
    """One request outcome that never became a ledger row."""

    key_label: str
    model: str
    status: int


@dataclass
class Outcomes:
    """Tally of the outcomes the ledger cannot report: refusals and failures."""

    _counts: dict[Outcome, int] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def note(self, key_label: str | None, model: str | None, status: int) -> None:
        outcome = Outcome(key_label or ANONYMOUS_KEY, model or UNKNOWN_MODEL, status)
        with self._lock:
            self._counts[outcome] = self._counts.get(outcome, 0) + 1

    def snapshot(self) -> dict[Outcome, int]:
        with self._lock:
            return dict(self._counts)


def exposition(config: Config, ledger: Ledger, outcomes: Outcomes, now: datetime | None = None) -> bytes:
    """The whole scrape, as Prometheus text format 0.0.4."""
    requests: dict[tuple[str, str, str], int] = {}
    tokens: dict[tuple[str, str, str], int] = {}
    costs: dict[tuple[str, str], float] = {}

    for total in ledger.key_model_totals():
        pair = (total.key_label, total.model)
        requests[pair + (LEDGER_STATUS,)] = total.calls
        if total.prompt_tokens is not None:
            tokens[pair + ("prompt",)] = total.prompt_tokens
        if total.completion_tokens is not None:
            tokens[pair + ("completion",)] = total.completion_tokens
        if total.cost is not None:
            costs[pair] = total.cost

    for outcome, count in outcomes.snapshot().items():
        labelled = (outcome.key_label, outcome.model, str(outcome.status))
        requests[labelled] = requests.get(labelled, 0) + count

    series: dict[str, dict[Any, Any]] = {
        "requests_total": requests,
        "tokens_total": tokens,
        "cost_usd_total": costs,
        "budget_fraction": _budgets(config, ledger, now),
    }

    lines: list[str] = []
    for name, samples in series.items():
        kind, help_text, labels = FAMILIES[name]
        full = PREFIX + name
        lines.append(f"# HELP {full} {help_text}")
        lines.append(f"# TYPE {full} {kind}")
        for values in sorted(samples):
            lines.append(f"{full}{_label_text(labels, values)} {_number(samples[values])}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _budgets(config: Config, ledger: Ledger, now: datetime | None) -> dict[tuple[str, ...], float]:
    """Where each budgeted key stands, from the same sums the gate enforces."""
    year, month, _ = month_of(now)
    out: dict[tuple[str, ...], float] = {}
    for key in config.keys.values():
        cap = key.monthly_budget_usd
        if cap is None or cap <= 0 or key.label in out:
            # Budgets are counted per label, so a repeated label is one series.
            continue
        out[(key.label,)] = ledger.sum_month(key.label, year, month) / cap
    return out


def _label_text(names: tuple[str, ...], values: tuple) -> str:
    pairs = ",".join(f"{name}={_quote(value)}" for name, value in zip(names, values))
    return "{" + pairs + "}"


def _quote(value: Any) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _number(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    number = float(value)
    return str(int(number)) if number.is_integer() else repr(number)
