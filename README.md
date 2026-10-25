# spendgate

[![PyPI](https://img.shields.io/pypi/v/spendgate)](https://pypi.org/project/spendgate/)
![Tests](https://github.com/intertermux-code/spendgate/actions/workflows/test.yml/badge.svg)

An OpenAI-compatible proxy that answers "who spent what, and are we over
budget?" with one Python process and one SQLite file. No Postgres, no Redis,
no ClickHouse, no Docker.

`spendgate` sits between your apps and one or more upstreams: an
OpenAI-compatible endpoint, or several of them split by model name, or Anthropic
behind a translation layer. Every developer or agent gets their own virtual key.
Each call is routed by model, forwarded, metered from the `usage` the upstream
reports, and appended to a spend ledger. Keys can carry a monthly budget: a
warning at 80% of the cap, a hard `429` at 100%. Because the ledger is a plain
file, the report is `scp` plus a command — the server does not have to be
running.

## Why this exists

The established LLM metering tools are good products that ask for
infrastructure first. LiteLLM wants a database for its spend logs; Langfuse
wants Postgres and ClickHouse; both want a service you keep alive. If you have
eight people and a shared API key and you only want to know who is burning
money, that is a lot of moving parts to stand up and patch.

`spendgate` is the narrow version of the problem: spend control first,
observability second. One file records everything, budgets are enforced from
that file, and the whole thing is the Python standard library.

## Install

```sh
pip install spendgate
```

Or from a checkout:

```sh
git clone https://github.com/intertermux-code/spendgate
cd spendgate
pip install -e .
```

Requires Python 3.11 or newer. There are no third-party dependencies.

## Quickstart

**1. Write a config.** Copy the example and fill in the two upstream values.

```sh
cp config.example.json spendgate.json
$EDITOR spendgate.json
```

```json
{
  "upstream_base_url": "https://api.openai.com/v1",
  "upstream_api_key": "sk-your-real-upstream-key",
  "listen_port": 8080,
  "ledger_path": "spendgate.sqlite",
  "keys": {
    "vk-alice-dev-key-000000": { "label": "alice", "monthly_budget_usd": 20.0 },
    "vk-bob-agent-key-000000": { "label": "bob", "monthly_budget_usd": 5.0 },
    "vk-ci-bot-key-000000000": { "label": "ci-bot" }
  }
}
```

`upstream_api_key` never leaves the server. The strings under `keys` are the
virtual keys your team uses; they are labels for accounting, not credentials
for any provider. Make them long and random.

**2. Run it.**

```sh
spendgate serve -c spendgate.json
```

**3. Point a client at it.** Same base URL shape, same request bodies — this is
the OpenAI SDK, with `base_url` aimed at the proxy and a virtual key:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="vk-alice-dev-key-000000")
client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
```

Or with curl:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer vk-alice-dev-key-000000" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}'
```

Streaming works the same way (`"stream": true`), and events are relayed as they
arrive rather than after the completion finishes.

**4. Read the report.** Offline, against the file:

```sh
spendgate report                       # everything, by key, model, and day
spendgate report --month 2026-10
spendgate report --json | jq .by_key
spendgate report -d /tmp/spendgate.sqlite   # a copy pulled off the box
```

```
spendgate report — spendgate.sqlite
month: all time   calls: 2
tokens: 1003 prompt / 252 completion
total cost: 0.0003

By key
                            calls    prompt   complet    cost usd
  alice                         2      1003       252      0.0003
```

## Configuration

One JSON file. Default location is `./spendgate.json`, or `-c/--config`, or
`$SPENDGATE_CONFIG`. A `SPENDGATE_`-prefixed environment variable overrides its
own setting, which keeps secrets out of files on shared boxes.

| Key | Env | Default | Meaning |
| --- | --- | --- | --- |
| `upstream_base_url` | `SPENDGATE_UPSTREAM_BASE_URL` | required without `upstreams` | OpenAI-compatible base URL, including its `/v1` suffix |
| `upstream_api_key` | `SPENDGATE_UPSTREAM_API_KEY` | required without `upstreams` | the real provider key; only ever sent upstream |
| `upstreams` | — | none | list of named upstreams, each serving a set of models; wins over the two fields above |
| `listen_host` | `SPENDGATE_LISTEN_HOST` | `127.0.0.1` | bind address; `0.0.0.0` exposes the proxy to your network |
| `listen_port` | `SPENDGATE_LISTEN_PORT` | `8080` | bind port |
| `ledger_path` | `SPENDGATE_LEDGER_PATH` | `spendgate.sqlite` | the SQLite file; created on first write |
| `prices_path` | `SPENDGATE_PRICES_PATH` | bundled `prices.json` | your own price table |
| `webhook_url` | `SPENDGATE_WEBHOOK_URL` | none | where soft-limit alerts are POSTed |
| `soft_limit_fraction` | `SPENDGATE_SOFT_LIMIT_FRACTION` | `0.8` | warning threshold, as a fraction of the cap |
| `keys` | — | required | virtual key → `{ "label", "monthly_budget_usd"? }`, or a bare label string |

`monthly_budget_usd` is optional per key. A key without one is still metered,
never limited.

### Multiple upstreams

`upstreams` splits the models over several providers. Each entry names itself,
points at a base URL, carries its own key, and lists the model names it serves:

```json
{
  "listen_port": 8080,
  "keys": { "REPLACE_ME-vk-alice": { "label": "alice", "monthly_budget_usd": 20.0 } },
  "upstreams": [
    {
      "name": "openai",
      "base_url": "https://api.openai.com/v1",
      "api_key": "REPLACE_ME-openai-key",
      "models": ["gpt-4", "o3-"]
    },
    {
      "name": "deepseek",
      "base_url": "https://api.deepseek.com/v1",
      "api_key": "REPLACE_ME-deepseek-key",
      "models": ["deepseek-*"]
    },
    {
      "name": "claude",
      "kind": "anthropic",
      "base_url": "https://api.anthropic.com/v1",
      "api_key": "REPLACE_ME-anthropic-key",
      "models": ["claude-"]
    }
  ]
}
```

Routing rule: a request goes to the **first** entry in the list whose `models`
match the requested model. A pattern is a prefix unless it contains glob
characters (`gpt-4` matches `gpt-4o-mini`; `deepseek-*` matches
`deepseek-chat`), so order matters — put the narrow entries first. `kind`
defaults to `"openai"`; names must be unique.

With `upstreams` present, `upstream_base_url` and `upstream_api_key` are
ignored, and each entry's key can come from `SPENDGATE_<NAME>_API_KEY` instead
of the file (`SPENDGATE_CLAUDE_API_KEY` for the entry named `claude`). Without
it, the single-upstream configuration behaves exactly as it always has: one
base URL, every model.

A model no upstream serves is refused with `404` and
`{"error": {"code": "model_not_found"}}` — nothing is forwarded, nothing is
written to the ledger, and the call costs nothing. Per-upstream keys are sent
only to the upstream they belong to and appear in no log line, no error body
and no metric.

### Anthropic upstreams

An upstream with `"kind": "anthropic"` is reached through Anthropic's Messages
API (`POST <base_url>/messages`, `x-api-key`, `anthropic-version: 2023-06-01`)
while your clients keep speaking OpenAI. On the way out: `system` and
`developer` messages become the `system` field, consecutive turns for one role
are merged, `temperature`, `top_p` and `stop` carry over, and a client that
omits `max_tokens` gets `4096` because Anthropic requires the field. On the way
back: the answer is rebuilt as a `chat.completion`, and a streamed answer
becomes OpenAI chunks — text deltas as they arrive, and a final chunk carrying
`finish_reason: "stop"` plus the usage.

The tokens Anthropic reports (`input_tokens`, `output_tokens`) are metered
through the same price table and the same ledger, so an Anthropic-backed call
shows up in `spendgate report` and in `/metrics` like any other.

Scope is text. `tools`, `tool_choice`, `functions`, tool calls or tool results
in `messages`, `n` greater than 1, empty turns and non-text content parts are
refused with `400` and `unsupported_for_upstream` rather than quietly dropped.

### Prometheus metrics

`GET /metrics` on the proxy port answers in Prometheus text format
(`text/plain; version=0.0.4`). It is infrastructure, so it needs no virtual key:

| Metric | Type | Labels |
| --- | --- | --- |
| `spendgate_requests_total` | counter | `key`, `model`, `status` |
| `spendgate_tokens_total` | counter | `key`, `model`, `kind` (`prompt` / `completion`) |
| `spendgate_cost_usd_total` | counter | `key`, `model` |
| `spendgate_budget_fraction` | gauge | `key` — spend this UTC month over the cap |

```yaml
# prometheus.yml
scrape_configs:
  - job_name: spendgate
    static_configs:
      - targets: ["127.0.0.1:8080"]
```

The completed-call families are aggregated from the ledger on every scrape, so
a restart does not zero the spend it can still read from the file; requests
that never reached the ledger (a `401`, a `429`, an unreachable upstream, an
unroutable model) are counted in memory alongside it. `key` carries the key's
*label*, never the key string, and an unknown model still counts requests and
tokens while simply having no `spendgate_cost_usd_total` series — the price is
never guessed.

### Prices

Cost is `prompt_tokens * input + completion_tokens * output`, divided by a
million, from `spendgate/prices.json` (USD per 1M tokens). That file is meant to
be edited: point `prices_path` at your own copy when your provider's numbers
move, or when you use a model that is not in the table. An unknown model is
recorded with its token counts and `cost NULL` — spendgate never guesses a
price.

### Budget alerts

At `soft_limit_fraction` of a cap, spendgate logs a warning and, if
`webhook_url` is set, posts:

```json
{
  "event": "budget_soft_limit",
  "key_label": "alice",
  "month": "2026-10",
  "spent_usd": 8.5,
  "budget_usd": 10.0,
  "level": "soft"
}
```

One alert per key per month, so a busy key does not flood a channel. A failed
webhook is logged and never fails the user's request. At 100% the request is
refused before it reaches the provider, with `429` and an OpenAI-style error
body.

## Development

```sh
pip install -e .
python -m unittest discover      # stdlib unittest, no pytest — 150 tests
python scripts/smoke.py          # live proxy process + curl + report
```

`scripts/smoke.py` starts a mock upstream and a real `spendgate serve`, drives
one streaming and one non-streaming call through it with curl, and checks the
report against hand-computed costs.

## Releasing

`.github/workflows/publish.yml` does the publishing: push a `v*` tag, or run the
workflow by hand, and it builds an sdist and a wheel, checks them with
`twine check`, and uploads both to PyPI over OIDC trusted publishing. There is
no API token in the repository to keep secret, and nothing to rotate.

A release is two edits and a tag:

1. Bump `version` in `pyproject.toml` **and** `__version__` in
   `spendgate/__init__.py`. They have to agree; `tests/test_version.py` fails if
   they drift, and the workflow additionally refuses a tag whose version does
   not match `pyproject.toml`.
2. Tag the commit `v` plus that version, and push the tag:

   ```sh
   git tag v0.2.0 && git push origin v0.2.0
   ```

**Once, before the first release:** PyPI has to be told which workflow is
allowed to publish this project. On PyPI, open the **spendgate** project →
**Manage** → **Publishing** → **Add a trusted publisher**, and enter:

| Field | Value |
| --- | --- |
| Publisher | GitHub Actions |
| Account name | `intertermux-code` |
| Repository name | `spendgate` |
| Workflow name | `publish.yml` |

PyPI matches the workflow *file* name, so `.github/workflows/publish.yml` is
registered as `publish.yml` — rename that file and this registration has to
change with it. Until the publisher is registered, the publish job fails when
PyPI rejects the OIDC token; that is the intended refusal, not something to
work around.

Running the workflow from `workflow_dispatch` has no tag to check, so it
publishes whatever version `pyproject.toml` says. The index refuses a
re-upload of a version that already exists, so an accidental repeat cannot
overwrite a release that is out there.

## Limitations, honestly

- **Single process, no HA.** One `spendgate serve` per ledger file. There is no
  replication, no clustering, no failover. WAL mode means a crash does not
  corrupt the file, but two proxies writing one ledger over a network mount is
  not a supported setup.
- **Budgets are calendar-monthly, in UTC,** and per key label. No daily caps,
  no per-model limits, no rollover windows, no team rollups.
- **Enforcement is best-effort at the margin.** The check reads what the ledger
  already holds, so a key can overshoot its cap by the cost of whatever is in
  flight, and concurrent requests can each slip past a check made before either
  finished. It stops the bleeding; it is not an exact pre-debit.
- **Streaming usage depends on the upstream.** spendgate asks for
  `stream_options.include_usage`, but if the provider ignores it, the call is
  recorded with `NULL` tokens and `NULL` cost. Nothing is estimated.
- **Unknown models cost `NULL`,** and `NULL` spend cannot trip a budget. Add the
  model to your price table if you need it capped.
- **Only `POST /v1/chat/completions` is proxied.** Embeddings, images and
  assistants are out of scope and answer `404`.
- **The Anthropic translation is text-only.** Messages requests are built from
  text turns and answered with text. Tool definitions, tool calls, tool results,
  images and audio are refused with a `400` that says so — spendgate never
  pretends a tool round trip happened, and it does not translate any other part
  of Anthropic's API surface.
- **Prices are a maintained text file,** not a live feed. Check them against
  your provider.
- **Virtual keys are labels, not real authentication.** They keep teammates and
  agents apart in the accounting; they are not a defence against someone on the
  network with the proxy's address. Bind to `127.0.0.1`, or put it behind
  something that does TLS and real auth.
- **No request or response bodies are stored.** You get token counts and cost,
  not a conversation history. That is deliberate — it is what keeps the ledger
  small enough to email.
- **A scrape aggregates the whole ledger.** `/metrics` is a grouped read over
  every row, so counters are lifetime totals and the read grows with the file.
  Budgets stay month-scoped; the metrics do not.

## What spendgate is not

This is a spend-control proxy. It is **not** a router in the sense of choosing
an answer for you: `upstreams` is a fixed map from model names to providers that
you write, and a request for a model nothing serves is refused rather than
substituted. It does not replace
[llmroutes](https://github.com/intertermux-code/llmroutes), a benchmark-driven
router CLI that picks the best model for a task. Different tools for different
jobs: llmroutes chooses *which* model answers; spendgate decides *whose* calls
these are, which upstream they may reach, and what they may cost. Use both if
you want both.

## License

MIT. See [LICENSE](LICENSE).
