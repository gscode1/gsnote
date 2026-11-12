# Contributing

Small, focused PRs are easiest to review. Run the tests before opening one:

```bash
uv sync --extra dev
uv run pytest    # CI runs the same suite
```

- Keep it simple. The core path (capture, embed, search) makes no LLM calls and uses no third-party services; please keep it that way.
- Add or update tests for behavior changes. Real network calls and model downloads do not belong in tests.
- Retrieval changes: run `uv run python -m eval.run_eval` and compare.
- Never commit `.env`, tokens, or note data (including database files).

Security issues: see [`SECURITY.md`](SECURITY.md) instead of opening a public issue.
