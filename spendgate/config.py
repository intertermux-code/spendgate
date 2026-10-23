"""Configuration loading for spendgate.

The config is a single JSON file. All runtime behaviour is driven from it;
nothing is hardcoded. See ``config.example.json`` for the template.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = "spendgate.json"
DEFAULT_LEDGER_PATH = "spendgate.sqlite"
DEFAULT_PRICES_FILENAME = "prices.json"


@dataclass
class KeyConfig:
    label: str
    monthly_budget_usd: float | None = None


@dataclass
class Config:
    upstream_base_url: str
    upstream_api_key: str
    listen_host: str = "127.0.0.1"
    listen_port: int = 8080
    ledger_path: str = DEFAULT_LEDGER_PATH
    prices_path: str | None = None
    webhook_url: str | None = None
    keys: dict[str, KeyConfig] = field(default_factory=dict)
    soft_limit_fraction: float = 0.8

    @property
    def upstream_url(self) -> str:
        """Return the upstream base URL with a single trailing slash."""
        return self.upstream_base_url.rstrip("/") + "/"

    def virtual_key_for(self, key: str) -> KeyConfig | None:
        return self.keys.get(key)


def _coerce_keys(raw: dict[str, Any]) -> dict[str, KeyConfig]:
    out: dict[str, KeyConfig] = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            budget = v.get("monthly_budget_usd")
            out[k] = KeyConfig(
                label=str(v.get("label", k)),
                monthly_budget_usd=float(budget) if budget is not None else None,
            )
        else:
            out[k] = KeyConfig(label=str(v))
    if not out:
        raise ValueError("config: 'keys' must define at least one virtual key")
    return out


def load_config(path: str | None = None) -> Config:
    """Load and validate a spendgate JSON config file.

    Precedence is environment, then file, then default, so a deployment can
    override one setting without rewriting the config:
      SPENDGATE_UPSTREAM_BASE_URL
      SPENDGATE_UPSTREAM_API_KEY
      SPENDGATE_LISTEN_HOST
      SPENDGATE_LISTEN_PORT
      SPENDGATE_LEDGER_PATH
      SPENDGATE_PRICES_PATH
      SPENDGATE_WEBHOOK_URL
      SPENDGATE_SOFT_LIMIT_FRACTION
    """
    cfg_path = Path(path or os.environ.get("SPENDGATE_CONFIG", DEFAULT_CONFIG_PATH))
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {cfg_path}. Copy config.example.json and fill it in."
        )

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON object")

    def value(name: str, default: Any = None) -> Any:
        env = os.environ.get(f"SPENDGATE_{name}")
        if env not in (None, ""):
            return env
        return raw.get(name.lower(), default)

    upstream_base_url = value("UPSTREAM_BASE_URL")
    upstream_api_key = value("UPSTREAM_API_KEY")
    if not upstream_base_url:
        raise ValueError("config: 'upstream_base_url' is required")
    if not upstream_api_key:
        raise ValueError("config: 'upstream_api_key' is required")

    listen_host = value("LISTEN_HOST", "127.0.0.1")
    listen_port = int(value("LISTEN_PORT", 8080))
    ledger_path = value("LEDGER_PATH", DEFAULT_LEDGER_PATH)
    prices_path = value("PRICES_PATH")
    webhook_url = value("WEBHOOK_URL")
    soft_limit_fraction = float(value("SOFT_LIMIT_FRACTION", 0.8))

    if not 0.0 <= soft_limit_fraction <= 1.0:
        raise ValueError("soft_limit_fraction must be between 0.0 and 1.0")

    return Config(
        upstream_base_url=str(upstream_base_url),
        upstream_api_key=str(upstream_api_key),
        listen_host=str(listen_host),
        listen_port=int(listen_port),
        ledger_path=str(ledger_path),
        prices_path=str(prices_path) if prices_path else None,
        webhook_url=str(webhook_url) if webhook_url else None,
        keys=_coerce_keys(raw.get("keys", {})),
        soft_limit_fraction=soft_limit_fraction,
    )


def default_prices_path() -> Path:
    """Return the path to the checked-in prices.json inside the package."""
    return Path(__file__).resolve().parent / DEFAULT_PRICES_FILENAME