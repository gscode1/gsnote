import io
import json
from urllib.error import HTTPError

import pytest

from app import cli


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def api(monkeypatch):
    """Intercept urlopen and record requests; returns a dict of path -> payload."""
    calls = []
    routes = {
        "GET /health": {"status": "ok"},
        "POST /capture": {"id": "n1", "content": "x", "space": "default"},
        "GET /search": {"notes": [{"content": "fix the tap", "space": "default"}]},
        "POST /report": {"summary": "you promised things"},
        "GET /export": {"version": 1, "notes": []},
    }

    def fake_urlopen(req):
        key = f"{req.get_method()} {req.full_url.split('?')[0].replace('http://test', '')}"
        body = json.loads(req.data) if req.data else None
        calls.append((key, req.full_url, dict(req.headers), body))
        return FakeResponse(routes[key])

    monkeypatch.setattr(cli, "urlopen", fake_urlopen)
    return calls


def run(argv, monkeypatch):
    monkeypatch.setenv("GSNOTE_URL", "http://test")
    monkeypatch.setenv("API_TOKEN", "tok")
    return cli.main(argv)


def test_health_no_token(api, monkeypatch, capsys):
    monkeypatch.setenv("GSNOTE_URL", "http://test")
    monkeypatch.delenv("API_TOKEN", raising=False)
    assert cli.main(["health"]) == 0
    assert capsys.readouterr().out.strip() == "ok"


def test_add(api, monkeypatch, capsys):
    assert run(["add", "fix the tap"], monkeypatch) == 0
    assert "saved n1 [default]" in capsys.readouterr().out
    key, url, headers, body = api[0]
    assert key == "POST /capture"
    assert body == {"content": "fix the tap", "source": "cli", "space": "default"}
    assert headers["Authorization"] == "Bearer tok"


def test_add_stdin_and_space(api, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("from stdin\n"))
    assert run(["add", "--space", "work"], monkeypatch) == 0
    assert api[0][3] == {"content": "from stdin", "source": "cli", "space": "work"}


def test_add_missing_token(api, monkeypatch, capsys):
    monkeypatch.setenv("GSNOTE_URL", "http://test")
    monkeypatch.delenv("API_TOKEN", raising=False)
    assert cli.main(["add", "hi"]) == 1
    assert "API_TOKEN" in capsys.readouterr().err


def test_search(api, monkeypatch, capsys):
    assert run(["search", "tap", "--top-k", "3"], monkeypatch) == 0
    assert "fix the tap [default]" in capsys.readouterr().out
    assert "q=tap" in api[0][1] and "top_k=3" in api[0][1]


def test_report(api, monkeypatch, capsys):
    assert run(["report", "promises", "--category", "intention"], monkeypatch) == 0
    assert "you promised things" in capsys.readouterr().out
    assert api[0][3]["category"] == "intention"


def test_search_json(api, monkeypatch, capsys):
    assert run(["search", "tap", "--json"], monkeypatch) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["notes"][0]["content"] == "fix the tap"


def test_export_stdout(api, monkeypatch, capsys):
    assert run(["export"], monkeypatch) == 0
    assert json.loads(capsys.readouterr().out)["version"] == 1


def test_export_file(api, monkeypatch, tmp_path):
    out = tmp_path / "export.json"
    assert run(["export", "-o", str(out)], monkeypatch) == 0
    assert json.loads(out.read_text())["version"] == 1


def test_http_error(monkeypatch, capsys):
    monkeypatch.setenv("GSNOTE_URL", "http://test")
    monkeypatch.setenv("API_TOKEN", "bad")

    def raise_401(req):
        raise HTTPError(req.full_url, 401, "Unauthorized", {},
                        io.BytesIO(b'{"detail": "Unauthorized"}'))

    monkeypatch.setattr(cli, "urlopen", raise_401)
    assert cli.main(["search", "x"]) == 1
    assert "401" in capsys.readouterr().err
