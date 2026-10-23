# spendgate

An OpenAI-compatible proxy that answers "who spent what, and are we over
budget?" with one Python process and one SQLite file. No Postgres, no Redis,
no ClickHouse, no Docker.

`spendgate` sits between your apps and an OpenAI-compatible upstream. Every
developer or agent gets their own virtual key. Each call is forwarded, metered
from the `usage` the upstream reports, and appended to a spend ledger. Keys can
carry a monthly budget: a warning at 80% of the cap, a hard `429` at 100%.
Because the ledger is a plain file, the report is `scp` plus a command — the
server does not have to be running.

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
| `upstream_base_url` | `SPENDGATE_UPSTREAM_BASE_URL` | required | OpenAI-compatible base URL, including its `/v1` suffix |
| `upstream_api_key` | `SPENDGATE_UPSTREAM_API_KEY` | required | the real provider key; only ever sent upstream |
| `listen_host` | `SPENDGATE_LISTEN_HOST` | `127.0.0.1` | bind address; `0.0.0.0` exposes the proxy to your network |
| `listen_port` | `SPENDGATE_LISTEN_PORT` | `8080` | bind port |
| `ledger_path` | `SPENDGATE_LEDGER_PATH` | `spendgate.sqlite` | the SQLite file; created on first write |
| `prices_path` | `SPENDGATE_PRICES_PATH` | bundled `prices.json` | your own price table |
| `webhook_url` | `SPENDGATE_WEBHOOK_URL` | none | where soft-limit alerts are POSTed |
| `soft_limit_fraction` | `SPENDGATE_SOFT_LIMIT_FRACTION` | `0.8` | warning threshold, as a fraction of the cap |
| `keys` | — | required | virtual key → `{ "label", "monthly_budget_usd"? }`, or a bare label string |

`monthly_budget_usd` is optional per key. A key without one is still metered,
never limited.

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
python -m unittest discover      # stdlib unittest, no pytest
python scripts/smoke.py          # live proxy process + curl + report
```

`scripts/smoke.py` starts a mock upstream and a real `spendgate serve`, drives
one streaming and one non-streaming call through it with curl, and checks the
report against hand-computed costs.

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
- **Only `POST /v1/chat/completions` is proxied.** Embeddings, images,
  assistants, tool-calling round trips beyond what the endpoint carries, and
  non-OpenAI providers' native shapes are out of scope for v1.
- **Prices are a maintained text file,** not a live feed. Check them against
  your provider.
- **Virtual keys are labels, not real authentication.** They keep teammates and
  agents apart in the accounting; they are not a defence against someone on the
  network with the proxy's address. Bind to `127.0.0.1`, or put it behind
  something that does TLS and real auth.
- **No request or response bodies are stored.** You get token counts and cost,
  not a conversation history. That is deliberate — it is what keeps the ledger
  small enough to email.

## What spendgate is not

This is a spend-control proxy. It is **not** a router, and it does not replace
[llmroutes](https://github.com/intertermux-code/llmroutes), a benchmark-driven
router CLI that picks the best model for a task. Different tools for different
jobs: llmroutes chooses *which* model answers; spendgate decides *whose* calls
these are and what they may cost. Use both if you want both.

## License

MIT. See [LICENSE](LICENSE).
