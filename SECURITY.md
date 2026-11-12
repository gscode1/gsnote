# Security

## Reporting a vulnerability

Use GitHub's private vulnerability reporting: open the repository's **Security** tab and click **Report a vulnerability**. This creates a private advisory only maintainers can see.

Do **not** open a public issue for security problems, and never include note contents, `.env` values or tokens in a report. Describe the issue and redact secrets.

## Threat model

gsnote is **single-user, self-hosted** software, not a multi-tenant service. One bearer token protects everything:

- **API and MCP**: `/capture`, `/search`, `/spaces`, `/notes/{id}`, `/export`, `/import` and `/mcp` require `Authorization: Bearer $API_TOKEN`. Anyone with the token can read, write, export and delete all notes.
- **Fail closed**: if `API_TOKEN` is empty these routes return 503. `/health` is unauthenticated and exposes nothing.
- **DNS-rebinding protection is off** on `/mcp`, because the token is required on every request.
- **No third parties by default**: embeddings run locally and there is no LLM. The only outbound traffic is the embedding model download on first start, or your own endpoint if `EMBEDDING_PROVIDER=api` (then note text is sent there; treat it as trusted).

## What to protect

- `.env` and `my-secrets.yaml` (API token, optional embedding key, S3 credentials; both are gitignored)
- the `data/` directory or volume: the SQLite database holds every note in plaintext
- Litestream/S3 credentials and the backup bucket, if backups are enabled
- the host itself: gsnote assumes a trusted machine

## Deployment guidance

- Keep the port on localhost unless you need remote access. If exposed beyond localhost, run behind TLS (e.g. a reverse proxy); the token is sent on every request.
- Generate `API_TOKEN` with `openssl rand -hex 32`; never reuse a password.
- If the token leaks, change it and restart; if S3 credentials leak, revoke and reissue them.
