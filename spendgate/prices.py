"""Price table loading and cost computation.

Prices are stored as USD per 1,000,000 tokens in a JSON data file. The path is
overridable via config ``prices_path`` or the ``SPENDGATE_PRICES_PATH``
environment variable. Unknown models cost nothing (cost is NULL) but usage is
still recorded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_prices(path: str | Path | None) -> dict[str, dict[str, float]]:
    """Load the price table. Returns a dict keyed by model name.

    Each value is ``{"input": usd_per_1m, "output": usd_per_1m}``. An empty
    or missing file yields an empty table (all costs NULL).
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for model, prices in data.items():
        if not isinstance(prices, dict):
            continue
        inp = prices.get("input")
        outp = prices.get("output")
        if not isinstance(inp, (int, float)) or not isinstance(outp, (int, float)):
            continue
        out[str(model)] = {"input": float(inp), "output": float(outp)}
    return out


def compute_cost(
    model: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    prices: dict[str, dict[str, float]],
) -> float | None:
    """Return cost in USD for a call, or None if it cannot be computed.

    Returns None when the model is unknown or when token counts are missing;
    never raises and never invents numbers.
    """
    if model is None or not prices:
        return None
    price = prices.get(model)
    if price is None:
        return None
    if prompt_tokens is None and completion_tokens is None:
        return None
    return ((prompt_tokens or 0) * price["input"] + (completion_tokens or 0) * price["output"]) / 1_000_000.0