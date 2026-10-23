"""Budget checks and soft-limit alerts.

Spend for a key is whatever the ledger says it is for the current UTC calendar
month, so a restart never loses track and two proxies over one file agree.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import KeyConfig
from .ledger import Ledger

logger = logging.getLogger("spendgate.budget")

OK = "ok"
SOFT = "soft"
HARD = "hard"

ALERT_TIMEOUT_SECONDS = 5.0


@dataclass
class Decision:
    level: str
    spent_usd: float
    cap_usd: float | None

    @property
    def allowed(self) -> bool:
        return self.level != HARD


def month_of(now: datetime | None = None) -> tuple[int, int, str]:
    """Current (year, month, 'YYYY-MM') in UTC — the accounting calendar."""
    now = now or datetime.now(timezone.utc)
    return now.year, now.month, f"{now.year:04d}-{now.month:02d}"


def current_month(now: datetime | None = None) -> str:
    """The 'YYYY-MM' bucket spend is counted against."""
    return month_of(now)[2]


def check(
    ledger: Ledger,
    key: KeyConfig,
    soft_limit_fraction: float,
    now: datetime | None = None,
) -> Decision:
    """Where a key stands against its monthly cap, read from the ledger.

    A key without a cap is always allowed: it is still accounted for, just
    never limited.
    """
    year, month, _ = month_of(now)
    spent = ledger.sum_month(key.label, year, month)
    cap = key.monthly_budget_usd
    if cap is None:
        return Decision(OK, spent, None)
    if spent >= cap:
        return Decision(HARD, spent, cap)
    if cap > 0 and spent >= cap * soft_limit_fraction:
        return Decision(SOFT, spent, cap)
    return Decision(OK, spent, cap)


def alert_payload(key: KeyConfig, decision: Decision, month: str) -> dict:
    """Body of a budget alert. Deliberately carries no upstream credentials."""
    return {
        "event": "budget_soft_limit",
        "key_label": key.label,
        "month": month,
        "spent_usd": round(decision.spent_usd, 6),
        "budget_usd": decision.cap_usd,
        "level": decision.level,
    }


def post_alert(webhook_url: str, payload: dict) -> bool:
    """Fire a budget alert at a webhook. Returns True if it was accepted.

    A failing alert must never fail the user's request, so every error is
    swallowed into the log.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=ALERT_TIMEOUT_SECONDS) as response:
            return 200 <= response.status < 300
    except urllib.error.HTTPError as exc:
        logger.warning("budget alert rejected by %s: HTTP %s", webhook_url, exc.code)
    except (urllib.error.URLError, OSError) as exc:
        logger.warning("budget alert failed for %s: %s", webhook_url, exc)
    return False
