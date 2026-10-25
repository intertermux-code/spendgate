"""Configuration loading for spendgate.

The config is a single JSON file. All runtime behaviour is driven from it;
nothing is hardcoded. See ``config.example.json`` for the template.

Either one OpenAI-compatible upstream (``upstream_base_url`` +
``upstream_api_key``) or an ``upstreams`` list that splits the models across
several providers. The list wins when both are present.
"""

from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = "spendgate.json"
DEFAULT_LEDGER_PATH = "spendgate.sqlite"
DEFAULT_PRICES_FILENAME = "prices.json"

OPENAI_KIND = "openai"
ANTHROPIC_KIND = "anthropic"
UPSTREAM_KINDS = (OPENAI_KIND, ANTHROPIC_KIND)

# Name of a synthetic upstream that fronts the single-upstream config; it never
# collides with a user-supplied name because names are required to be non-empty.
SINGLE_UPSTREAM_NAME = "default"


@dataclass
class KeyConfig:
    label: str
    monthly_budget_usd: float | None = None


@dataclass
class UpstreamConfig:
    """One provider spendgate can forward to, and the models it answers for."""

    name: str
    base_url: str
    api_key: str
    models: list[str] = field(default_factory=list)
    kind: str = OPENAI_KIND

    @property
    def url(self) -> str:
        """Base URL with a single trailing slash, ready for path suffixes."""
        return self.base_url.rstrip("/") + "/"

    def serves(self, model: str) -> bool:
        return any(pattern_matches(pattern, model) for pattern in self.models)


@dataclass
class Config:
    upstream_base_url: str = ""
    upstream_api_key: str = ""
    listen_host: str = "127.0.0.1"
    listen_port: int = 8080
    ledger_path: str = DEFAULT_LEDGER_PATH
    prices_path: str | None = None
    webhook_url: str | None = None
    keys: dict[str, KeyConfig] = field(default_factory=dict)
    soft_limit_fraction: float = 0.8
    upstreams: list[UpstreamConfig] = field(default_factory=list)

    @property
    def upstream_url(self) -> str:
        """Return the upstream base URL with a single trailing slash."""
        return self.upstream_base_url.rstrip("/") + "/"

    def virtual_key_for(self, key: str) -> KeyConfig | None:
        return self.keys.get(key)

    @property
    def active_upstreams(self) -> list[UpstreamConfig]:
        """Every upstream routing may consider, in config order."""
        if self.upstreams:
            return list(self.upstreams)
        if self.upstream_base_url and self.upstream_api_key:
            # "" as a model pattern matches everything: one catch-all upstream
            # is exactly what the single-upstream config always was.
            return [
                UpstreamConfig(
                    name=SINGLE_UPSTREAM_NAME,
                    base_url=self.upstream_base_url,
                    api_key=self.upstream_api_key,
                    models=[""],
                )
            ]
        return []

    def route(self, model: str) -> UpstreamConfig | None:
        """First upstream whose model patterns match, or None if nothing does."""
        for upstream in self.active_upstreams:
            if upstream.serves(model):
                return upstream
        return None


def pattern_matches(pattern: str, model: str) -> bool:
    """A pattern with glob characters is a glob; anything else is a prefix.

    Prefixes are the common case ("gpt-4" for every gpt-4 model), globs cover
    the names that cannot be expressed as one ("claude-*-sonnet-*").
    """
    if any(char in pattern for char in "*?["):
        return fnmatch.fnmatchcase(model, pattern)
    return model.startswith(pattern)


def env_var_for(name: str) -> str:
    """The env var that overrides an upstream's key: SPENDGATE_<NAME>_API_KEY."""
    slug = "".join(char if char.isalnum() else "_" for char in name).upper()
    return f"SPENDGATE_{slug}_API_KEY"


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


def _coerce_upstreams(raw: Any) -> list[UpstreamConfig]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not raw:
        raise ValueError("config: 'upstreams' must be a non-empty list")

    out: list[UpstreamConfig] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"config: upstreams[{index}] must be an object")
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError(f"config: upstreams[{index}] needs a 'name'")
        if name in seen:
            raise ValueError(f"config: duplicate upstream name {name!r}")
        seen.add(name)

        base_url = entry.get("base_url")
        if not base_url:
            raise ValueError(f"config: upstream {name!r} needs a 'base_url'")

        models = entry.get("models")
        if not isinstance(models, list) or not models:
            raise ValueError(f"config: upstream {name!r} needs a non-empty 'models' list")
        if not all(isinstance(model, str) and model.strip() for model in models):
            raise ValueError(f"config: upstream {name!r} has a 'models' entry that is not a name")

        kind = str(entry.get("kind", OPENAI_KIND)).strip().lower()
        if kind not in UPSTREAM_KINDS:
            raise ValueError(
                f"config: upstream {name!r} has kind {kind!r}; expected one of {', '.join(UPSTREAM_KINDS)}"
            )

        env_name = env_var_for(name)
        api_key = os.environ.get(env_name) or entry.get("api_key")
        if api_key in (None, ""):
            raise ValueError(f"config: upstream {name!r} needs an 'api_key' or ${env_name}")

        out.append(
            UpstreamConfig(
                name=name,
                base_url=str(base_url),
                api_key=str(api_key),
                models=[str(model).strip() for model in models],
                kind=kind,
            )
        )
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

    With an ``upstreams`` list, each entry's key is overridden by
    ``SPENDGATE_<NAME>_API_KEY`` instead.
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

    upstreams = _coerce_upstreams(raw.get("upstreams"))

    upstream_base_url = value("UPSTREAM_BASE_URL")
    upstream_api_key = value("UPSTREAM_API_KEY")
    if not upstreams:
        # The single-upstream path keeps its own contract: both values required.
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
        upstream_base_url=str(upstream_base_url or ""),
        upstream_api_key=str(upstream_api_key or ""),
        listen_host=str(listen_host),
        listen_port=int(listen_port),
        ledger_path=str(ledger_path),
        prices_path=str(prices_path) if prices_path else None,
        webhook_url=str(webhook_url) if webhook_url else None,
        keys=_coerce_keys(raw.get("keys", {})),
        soft_limit_fraction=soft_limit_fraction,
        upstreams=upstreams,
    )


def default_prices_path() -> Path:
    """Return the path to the checked-in prices.json inside the package."""
    return Path(__file__).resolve().parent / DEFAULT_PRICES_FILENAME
