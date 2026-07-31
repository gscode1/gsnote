"""gsnote CLI: thin client over the HTTP API (see app/main.py). Stdlib only."""
import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_URL = "http://localhost:8000"


class CliError(Exception):
    pass


def _request(method: str, path: str, *, base_url: str, token: str | None,
             body: dict | None = None, params: dict | None = None) -> dict:
    url = base_url.rstrip("/") + path
    if params:
        url += "?" + urlencode({k: v for k, v in params.items() if v is not None})
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except ValueError:
            pass
        raise CliError(f"{e.code} from server: {detail}") from e
    except URLError as e:
        raise CliError(f"cannot reach {base_url}: {e.reason}") from e
    except OSError as e:
        raise CliError(f"connection to {base_url} failed: {e}") from e


def _require_token(args) -> str:
    token = args.token or os.environ.get("API_TOKEN", "")
    if not token:
        raise CliError("no API token: set API_TOKEN (or pass --token)")
    return token


def _base_url(args) -> str:
    return args.url or os.environ.get("GSNOTE_URL", DEFAULT_URL)


def _emit(args, payload, human) -> None:
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        human(payload)


def cmd_health(args) -> None:
    payload = _request("GET", "/health", base_url=_base_url(args), token=None)
    _emit(args, payload, lambda p: print(p.get("status", "ok")))


def cmd_add(args) -> None:
    content = args.content
    if content in (None, "-"):
        content = sys.stdin.read().strip()
    if not content:
        raise CliError("empty note: pass content as an argument or via stdin")
    payload = _request("POST", "/capture", base_url=_base_url(args),
                       token=_require_token(args),
                       body={"content": content, "source": args.source, "space": args.space})
    _emit(args, payload, lambda p: print(f"saved {p['id']} [{p['space']}]"))


def cmd_search(args) -> None:
    payload = _request("GET", "/search", base_url=_base_url(args),
                       token=_require_token(args),
                       params={"q": args.query, "top_k": args.top_k, "space": args.space})

    def human(p):
        for note in p["notes"]:
            print(f"{note['content']} [{note.get('space', 'default')}]")
        if not p["notes"]:
            print("no notes found")

    _emit(args, payload, human)


def cmd_report(args) -> None:
    payload = _request("POST", "/report", base_url=_base_url(args),
                       token=_require_token(args),
                       body={"query": args.query, "category": args.category, "space": args.space})
    _emit(args, payload, lambda p: print(p["summary"]))


def cmd_export(args) -> None:
    payload = _request("GET", "/export", base_url=_base_url(args),
                       token=_require_token(args))
    out = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out + "\n")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(out)


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", help=f"server URL (env GSNOTE_URL, default {DEFAULT_URL})")
    common.add_argument("--token", help="API token (env API_TOKEN)")
    common.add_argument("--json", action="store_true", help="print raw API response")

    parser = argparse.ArgumentParser(prog="gsnote", description="CLI for the gsnote HTTP API")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("health", parents=[common], help="check server health")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("add", parents=[common], help="capture a note")
    p.add_argument("content", nargs="?", help="note text ('-' or omitted reads stdin)")
    p.add_argument("--space", default="default")
    p.add_argument("--source", default="cli")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("search", parents=[common], help="search notes")
    p.add_argument("query")
    p.add_argument("--space", default="default")
    p.add_argument("--top-k", type=int, default=None)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("report", parents=[common], help="summarize notes matching a query")
    p.add_argument("query")
    p.add_argument("--space", default="default")
    p.add_argument("--category", default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("export", parents=[common], help="export all notes as JSON")
    p.add_argument("-o", "--output", help="output file (default stdout)")
    p.set_defaults(func=cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except CliError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
