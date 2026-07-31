# Contributing

Small, focused PRs are easiest to review. Run the test suite before opening one:

```bash
pip install -e ".[dev]"
pytest    # same command CI runs
```

Never commit `.env`, tokens, or note data. If you found a security issue, see [`SECURITY.md`](SECURITY.md) instead of opening a public issue.

---

## Adding a channel

Channels are the extension point for new messaging surfaces (Slack, Matrix, Signal, …). The core pipeline — capture, classification, retrieval, resurfacing — lives behind the `Channel` ABC in [`app/channels/__init__.py`](app/channels/__init__.py) and knows nothing about any specific messenger. [`app/channels/telegram.py`](app/channels/telegram.py) is the reference adapter; copy its shape.

### The contract

Create `app/channels/<name>.py` with a class implementing:

| Member | Required | What the core expects |
|---|---|---|
| `send(user_id, message, *, with_nudge_buttons=False, notification_id=None)` | yes | Push a message to a user. Used for replies *and* proactive digests, so the adapter must be able to initiate messages, not just answer webhooks. When `with_nudge_buttons=True`, attach engaged/dismiss/snooze affordances if the platform has them, encoding `notification_id` into the payload so the response survives a restart. |
| `on_message(handler)` | yes | Call `handler(user_id: str, text: str)` for every inbound text message. This is the entry point into the whole note/question pipeline — don't duplicate that logic in the adapter. |
| `start()` / `stop()` | yes | Start/stop receiving (polling loop or webhook server). `start()` is awaited during app startup, `stop()` on shutdown. |
| `recipients()` | no | Who gets proactive digests. Default `[]` = no digests. |
| `on_command(handler)` | no | Slash commands like `/work`. `handler(user_id, command, args) -> str` returns the reply text. Default no-op. |
| `on_response(handler)` | no | Nudge button responses: `handler(user_id, response, notification_id)`. Default no-op. |

Rules of thumb:

- **All ids are `str`.** Convert platform ids at the boundary (Telegram casts `int -> str`, see reference adapter).
- **Authorization is the adapter's job, and it fails closed.** An empty allowlist must refuse to start, never "allow all" — mirror `TelegramChannel.__init__`.
- **Inbound text goes to `on_message`, replies go through `send`.** Don't call agents, capture, or retrieval directly from the adapter.

### Wiring it in

1. Add your settings to `app/config.py` (`Settings`), documented in `.env.example`.
2. Register the adapter in `_build_channel()` in `app/main.py` under a new `CHANNEL=<name>` value.
3. Document setup in the README config table.

### Verifying locally

- The core pipeline is channel-independent: run with `CHANNEL=none` and drive it over the HTTP API (`/capture`, `/search`) to test note logic without any messenger credentials.
- Unit-test the adapter with its platform client mocked — see `tests/test_telegram_allowlist.py` for the fail-closed pattern. Real network calls don't belong in the test suite.
- Then run the full suite: `pytest`.
