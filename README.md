# gsnote

[![Tests](https://github.com/gscode1/gsnote/actions/workflows/test.yml/badge.svg)](https://github.com/gscode1/gsnote/actions/workflows/test.yml)

A self-hosted personal notes memory that any coding agent can use over MCP (there is also an HTTP API and a CLI). You tell your agent to remember something; later you or any other agent can search for it. It is for people who want their own agent memory without handing notes to a third party.

- **No LLM, no keys.** Embeddings run locally (fastembed, `bge-large` by default). The only secret is your own `API_TOKEN`.
- **One SQLite file.** [sqlite-vec](https://github.com/asg017/sqlite-vec) for vectors, FTS5 for keywords, fused with Reciprocal Rank Fusion. Easy to back up and export.
- **Time-aware search.** `since` / `until` filters (whole days in your `TIMEZONE`) let the calling agent answer "what did I note last week?".
- **Spaces.** Notes belong to a space (`work`, `personal`, a project name) and have a free-form category.

Apache-2.0.

## Quick start

```bash
cp .env.example .env
# set API_TOKEN in .env, e.g.: openssl rand -hex 32
docker compose up -d
curl localhost:8000/health
```

The first start downloads the embedding model (about 1.3 GB). Data lives in the `gsnote-data` volume.

Without Docker: `uv sync && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000`.

## Connect your agent

The MCP endpoint is `http://localhost:8000/mcp` (streamable HTTP, same bearer token as the API).

Claude Code:

```bash
claude mcp add --transport http gsnote http://localhost:8000/mcp \
  --header "Authorization: Bearer $API_TOKEN"
```

Add `--scope user` to make it available in every project.

Other MCP clients: add an HTTP/streamable-HTTP server with the URL `http://localhost:8000/mcp` and the header `Authorization: Bearer <your API_TOKEN>`. The exact config format depends on the client; for example:

```json
{
  "mcpServers": {
    "gsnote": {
      "url": "http://localhost:8000/mcp",
      "headers": { "Authorization": "Bearer <your API_TOKEN>" }
    }
  }
}
```

### Tools

| Tool | What it does |
|---|---|
| `save_note` | Save a note (`content`, optional `space`, `category`) |
| `search_notes` | Hybrid search; optional `query`, `space`, `category`, `since`, `until`, `limit`. No query lists newest matching notes. Returns today's date and timezone so the agent can resolve "last week" |
| `list_spaces` | List spaces with note counts |
| `delete_note` | Permanently delete a note by id |

### Example

> **You:** save that we decided to move the API to Postgres in work
>
> **Agent:** *(calls `save_note` with space `work`, category `decision`)* Saved.
>
> *a week later, in any agent:*
>
> **You:** what did I note last week?
>
> **Agent:** *(calls `search_notes` with `since`/`until` for last week)* On Tuesday you noted that you decided to move the API to Postgres.

## CLI

`gsnote` is a thin client over the HTTP API (installed with the package; `uv run gsnote ...` from a checkout). It reads `API_TOKEN` and `GSNOTE_URL` (default `http://localhost:8000`); `--url`, `--token` and `--json` work on every command.

```bash
gsnote health
gsnote add "Decided to move the API to Postgres" --space work --category decision
echo "from a pipe" | gsnote add
gsnote search "postgres" --space work --since 2026-11-01 --until 2026-11-07 --top-k 5
gsnote search --since 2026-11-01          # no query: newest matching notes
gsnote spaces
gsnote delete <id>
gsnote export -o gsnote-export.json
gsnote import gsnote-export.json
```

`search` also takes `--category`. `since`/`until` accept `YYYY-MM-DD` or an ISO-8601 datetime, inclusive.

## HTTP API

All routes except `/health` need `Authorization: Bearer $API_TOKEN`. If `API_TOKEN` is empty they return 503.

| Route | Purpose |
|---|---|
| `GET /health` | Liveness (no auth) |
| `POST /capture` | Save a note: `{content, space?, category?, source?}` |
| `GET /search` | `q`, `space`, `category`, `since`, `until`, `top_k` |
| `GET /spaces` | Spaces with counts |
| `DELETE /notes/{id}` | Delete a note |
| `GET /export` | All notes as JSON |
| `POST /import` | Import an export file (skips existing ids) |
| `/mcp` | MCP endpoint |

## Config

Environment variables (or `.env`); full list in [`.env.example`](.env.example).

| Variable | Default | What it does |
|---|---|---|
| `API_TOKEN` | empty | Bearer token for the API and `/mcp`. Empty disables them (fail closed) |
| `TIMEZONE` | `UTC` | IANA name; date-only `since`/`until` mean whole days here |
| `DB_PATH` | `./data/gsnote.db` | SQLite file (`/data/gsnote.db` in Compose) |
| `EMBEDDING_PROVIDER` | `local` | `local` (in-process) or `api` (OpenAI-compatible `/embeddings` endpoint) |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Model name |
| `EMBEDDING_DIM` | `1024` | Must match the model; do not change on a populated DB without re-embedding |
| `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` | empty | For `EMBEDDING_PROVIDER=api` (key optional) |
| `RETRIEVAL_TOP_K` | `8` | Default number of results |
| `RETRIEVAL_CANDIDATE_N` | `20` | Candidates per ranker before fusion |
| `RRF_K` | `60` | RRF constant |
| `LITESTREAM_ENABLED`, `S3_*` | off | Optional backup, see below |

## Kubernetes (Helm)

```bash
cp charts/gsnote/values.yaml my-secrets.yaml   # set image and secrets.API_TOKEN (gitignored)
helm install gsnote charts/gsnote -n gsnote --create-namespace -f my-secrets.yaml
```

Single replica (SQLite is single-writer) with a PVC for the DB and the model cache.

## Backup (optional)

Off by default. Set `LITESTREAM_ENABLED=true` plus the `S3_*` variables and restart; [Litestream](https://litestream.io) replicates the DB to S3. On container start with Litestream enabled and no local DB, it restores from the replica. `scripts/test_backup_restore.sh` exercises this against a local replica.

## Upgrading from the bot version

The old Telegram/Slack bot is gone; its last version is git tag `v0-bot`. Its database cannot be opened by this version (it refuses with an explanatory error). To keep your notes:

1. With the old version running, export: `curl -H "Authorization: Bearer $API_TOKEN" http://localhost:8000/export -o gsnote-export.json`
2. Start the new version with a new `DB_PATH` (or a fresh volume).
3. `gsnote import gsnote-export.json`

Both export formats (v1 from the bot, v2) are accepted; soft-deleted notes are skipped.

## Privacy

Everything stays on your machine. The only outbound traffic is downloading the embedding model on first start, or calls to your own endpoint if `EMBEDDING_PROVIDER=api`. See [`SECURITY.md`](SECURITY.md).

## Develop

```bash
uv sync --extra dev
uv run pytest
uv run python -m eval.run_eval   # retrieval quality eval
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Apache-2.0, see [`LICENSE`](LICENSE).
