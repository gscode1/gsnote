# gsnote

[![Tests](https://github.com/gscode1/gsnote/actions/workflows/test.yml/badge.svg)](https://github.com/gscode1/gsnote/actions/workflows/test.yml)

A personal memory bot for Telegram. Send notes, ask questions later, get a weekly nudge so nothing is forgotten.

Self-hosted · single-user · your own LLM key · Apache-2.0

---

## What you need

1. A [Telegram bot token](https://t.me/BotFather)
2. Your Telegram user id ([@userinfobot](https://t.me/userinfobot))
3. An LLM API key (OpenRouter, Anthropic, or any OpenAI-compatible provider)

---

## Setup

```bash
cp .env.example .env
```

Fill in at least:

```bash
LLM_API_KEY=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_ALLOWED_USER_IDS=123456789   # your numeric id; comma-separate for more
```

Then:

```bash
docker compose pull   # prebuilt multi-arch image from GHCR
docker compose up -d
```

### Verify it works

Open your bot in Telegram. You should see an exchange like:

> **You:** I told the landlord I'd fix the leaky tap by Friday
>
> **Bot:** Noted — saved that.
>
> **You:** what did I promise the landlord?
>
> **Bot:** You promised to fix the leaky tap by Friday.

If the bot answers your question from the note you just sent, everything is wired up. No reply? Check `docker compose logs -f` — the bot refuses to start with a clear error if a required variable is missing.

### Other ways to run (optional)

Build from source instead of pulling:

```bash
docker compose up --build
```

Without Docker:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # same required vars
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Kubernetes (Helm):

```bash
cp charts/gsnote/values.yaml my-secrets.yaml   # fill in image + secrets (gitignored)
helm install gsnote charts/gsnote -n gsnote --create-namespace -f my-secrets.yaml
```

Optional in-cluster whisper STT: set `whisper.enabled=true` and point `config.STT_BASE_URL` at `http://whisper:9000/v1`.

---

## How to use

| You send | What happens |
|---|---|
| A note or idea | Saved to your memory (classified + searchable) |
| A question | Searches your notes and answers from them |
| A voice note | Transcribed, then treated like text *(needs STT — see below)* |
| `/space <name>` | Switch to (or create) a named space |
| `/space` | Show active space and your spaces |
| `/work` or `/personal` | Space shortcuts |
| Weekly digest buttons | Engaged / Dismiss / Snooze |

Spaces keep notes apart. New notes and questions use the active space, and every reply ends with a `[space]` tag showing where it was saved or searched.

---

## Config

Minimum (required when `CHANNEL=telegram`):

| Variable | What it is |
|---|---|
| `LLM_API_KEY` | Your provider key |
| `LLM_PROVIDER` | `openrouter` (default), `anthropic`, or any OpenAI-compatible name |
| `TELEGRAM_BOT_TOKEN` | From BotFather |
| `TELEGRAM_ALLOWED_USER_IDS` | Your Telegram id(s) — empty = bot refuses to start |

Useful optional vars:

| Variable | Default | What it does |
|---|---|---|
| `API_TOKEN` | empty = HTTP data API off | Bearer token for `/capture`, `/search`, `/report`, `/export` (`Authorization: Bearer <token>`) |
| `CLASSIFIER_MODEL` / `ANSWER_MODEL` | cheap / strong | Models for tagging notes vs answering |
| `CHANNEL=none` | — | HTTP only (no Telegram) |
| `RESURFACING_CRON` | `0 9 * * MON` | Weekly digest schedule |
| `STT_ENABLED` / `STT_BASE_URL` | off | Voice notes via an OpenAI-compatible STT endpoint |

Full list: [`.env.example`](.env.example).

---

## Export

Your notes are yours. Download everything as JSON (needs `API_TOKEN`):

```bash
curl -H "Authorization: Bearer $API_TOKEN" http://localhost:8000/export -o gsnote-export.json
```

The file is `{"version": 1, "exported_at": ..., "notes": [...]}` — each note carries its content, category, source, space, timestamps, and access stats.

---

## Privacy

Notes live in local SQLite. When the bot reasons, content goes to **your** LLM provider. You pick the key and endpoint.

---

## Backup (optional)

Off by default. Set `LITESTREAM_ENABLED=true` plus S3 vars in `.env`, then restart. Restore runs on container start when Litestream is on and no local DB exists.

---

## Develop

```bash
pip install -e ".[dev]"
pytest
```

Contributing a channel or fix? See [`CONTRIBUTING.md`](CONTRIBUTING.md). Security issues: [`SECURITY.md`](SECURITY.md).

---

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
