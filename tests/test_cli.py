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
        "GET /search": {"notes": [{"id": "n1", "content": "fix the tap", "space": "default",
                                   "category": "task", "created_at": "2026-03-02T10:00:00+00:00"}]},
        "GET /spaces": [{"space": "default", "notes": 2}, {"space": "work", "notes": 1}],
        "GET /export": {"version": 2, "notes": []},
        "DELETE /notes/abc": {"deleted": "abc"},
        "POST /import": {"imported": 3, "skipped": 1},
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
    assert body == {"content": "fix the tap", "source": "cli", "space": "default",
                    "category": "note"}
    assert headers["Authorization"] == "Bearer tok"


def test_add_stdin_and_space(api, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("from stdin\n"))
    assert run(["add", "--space", "work"], monkeypatch) == 0
    assert api[0][3] == {"content": "from stdin", "source": "cli", "space": "work",
                          "category": "note"}


def test_add_missing_token(api, monkeypatch, capsys):
    monkeypatch.setenv("GSNOTE_URL", "http://test")
    monkeypatch.delenv("API_TOKEN", raising=False)
    assert cli.main(["add", "hi"]) == 1
    assert "API_TOKEN" in capsys.readouterr().err


def test_search(api, monkeypatch, capsys):
    assert run(["search", "tap", "--top-k", "3"], monkeypatch) == 0
    assert "2026-03-02  [default/task]  fix the tap  (n1)" in capsys.readouterr().out
    assert "q=tap" in api[0][1] and "top_k=3" in api[0][1]


def test_search_filters_passed_through(api, monkeypatch, capsys):
    argv = ["search", "tap", "--space", "work", "--category", "task",
            "--since", "2026-03-01", "--until", "2026-03-31", "--top-k", "5"]
    assert run(argv, monkeypatch) == 0
    url = api[0][1]
    for part in ("q=tap", "space=work", "category=task", "since=2026-03-01",
                 "until=2026-03-31", "top_k=5"):
        assert part in url


def test_search_broad_flag(api, monkeypatch, capsys):
    assert run(["search", "tap", "--broad"], monkeypatch) == 0
    assert "broad=true" in api[0][1]
    assert run(["search", "tap"], monkeypatch) == 0
    assert "broad" not in api[1][1]


def test_search_without_query_lists_and_omits_unset_params(api, monkeypatch, capsys):
    assert run(["search", "--since", "2026-03-01"], monkeypatch) == 0
    url = api[0][1]
    assert "since=2026-03-01" in url
    assert "q=" not in url and "space=" not in url and "category=" not in url
    assert "fix the tap" in capsys.readouterr().out


def test_add_category(api, monkeypatch, capsys):
    assert run(["add", "ship it", "--category", "task"], monkeypatch) == 0
    assert api[0][3]["category"] == "task"


def test_spaces(api, monkeypatch, capsys):
    assert run(["spaces"], monkeypatch) == 0
    out = capsys.readouterr().out
    assert "default\t2" in out and "work\t1" in out
    assert api[0][0] == "GET /spaces"


def test_spaces_json(api, monkeypatch, capsys):
    assert run(["spaces", "--json"], monkeypatch) == 0
    assert json.loads(capsys.readouterr().out)[0] == {"space": "default", "notes": 2}


def test_search_json(api, monkeypatch, capsys):
    assert run(["search", "tap", "--json"], monkeypatch) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["notes"][0]["content"] == "fix the tap"


def test_export_stdout(api, monkeypatch, capsys):
    assert run(["export"], monkeypatch) == 0
    assert json.loads(capsys.readouterr().out)["version"] == 2


def test_export_file(api, monkeypatch, tmp_path):
    out = tmp_path / "export.json"
    assert run(["export", "-o", str(out)], monkeypatch) == 0
    assert json.loads(out.read_text())["version"] == 2


def test_delete(api, monkeypatch, capsys):
    assert run(["delete", "abc"], monkeypatch) == 0
    assert "deleted abc" in capsys.readouterr().out
    assert api[0][0] == "DELETE /notes/abc"
    assert api[0][2]["Authorization"] == "Bearer tok"


def test_import_file(api, monkeypatch, capsys, tmp_path):
    f = tmp_path / "export.json"
    f.write_text(json.dumps({"version": 2, "notes": [{"id": "x"}]}))
    assert run(["import", str(f)], monkeypatch) == 0
    assert "imported 3, skipped 1" in capsys.readouterr().out
    assert api[0][0] == "POST /import"
    assert api[0][3] == {"version": 2, "notes": [{"id": "x"}]}


def test_import_stdin(api, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO('{"version": 1, "notes": []}'))
    assert run(["import", "-"], monkeypatch) == 0
    assert api[0][3] == {"version": 1, "notes": []}


def test_import_invalid_json(api, monkeypatch, capsys, tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("{nope")
    assert run(["import", str(f)], monkeypatch) == 1
    assert "not valid JSON" in capsys.readouterr().err


def test_http_error(monkeypatch, capsys):
    monkeypatch.setenv("GSNOTE_URL", "http://test")
    monkeypatch.setenv("API_TOKEN", "bad")

    def raise_401(req):
        raise HTTPError(req.full_url, 401, "Unauthorized", {},
                        io.BytesIO(b'{"detail": "Unauthorized"}'))

    monkeypatch.setattr(cli, "urlopen", raise_401)
    assert cli.main(["search", "x"]) == 1
    assert "401" in capsys.readouterr().err
