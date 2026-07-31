# Security

## Reporting a vulnerability

Use GitHub's private vulnerability reporting: open the repository's **Security** tab and click **Report a vulnerability**. This creates a private advisory only maintainers can see.

Do **not** open a public issue for security problems, and never include note contents, `.env` values, bot tokens, or API keys in any report — describe the issue, redact the secrets.

## Threat model

gsnote is **single-user, self-hosted** software, not a multi-tenant SaaS. The trust boundary is your own machine and the channels you configure:

- **Telegram**: `TELEGRAM_ALLOWED_USER_IDS` is required when `CHANNEL=telegram`; an empty allowlist refuses to start. Anyone not on the list gets nothing.
- **HTTP API**: `/capture`, `/search`, `/report`, and `/export` require `Authorization: Bearer $API_TOKEN`. An empty `API_TOKEN` disables these routes entirely. `/health` is intentionally unauthenticated and exposes nothing.
- **AI providers**: note content leaves your machine when sent to your configured LLM / embedding / STT endpoints. You choose the provider and hold the key — treat that provider as trusted with your notes.
- **Voice**: audio goes to your configured STT endpoint when `STT_ENABLED=true`.

## What to protect

- `.env` and `my-secrets.yaml` — contain bot token, LLM key, API token (both gitignored; keep them that way)
- the `data/` directory — SQLite database with every note in plaintext
- Litestream/S3 credentials and the backup bucket, if backups are enabled — backups are only as safe as the bucket
- the host itself — gsnote assumes a trusted machine and runs without sandboxing

## Deployment guidance

- Don't expose the HTTP port beyond localhost unless you need the API, and if you do, put it behind TLS (e.g. a reverse proxy) — the bearer token is sent on every request.
- Generate `API_TOKEN` with `openssl rand -hex 32`, never reuse a password.
- Rotate the Telegram token and LLM key immediately if `.env` leaks; revoke and reissue rather than auditing first.
