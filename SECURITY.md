# Security

## Reporting

Email or message the maintainer privately. Do **not** open a public issue that includes notes, `.env` contents, bot tokens, or API keys.

## Assumptions

- Single-user, self-hosted. Not a multi-tenant SaaS.
- `TELEGRAM_ALLOWED_USER_IDS` must be set when using Telegram; empty allowlist refuses to start.
- Note text may leave the machine when sent to your configured LLM / STT provider (BYOM).
- Keep `k8s/secret.yaml` and `.env` out of git (already gitignored).
