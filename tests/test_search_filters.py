"""Search filters: space, category, since/until (timezone-aware) and list mode."""
import pytest
from fastapi.testclient import TestClient

from app import capture
from app.config import get_settings
from app.db import get_conn
from app.main import app
from app.retrieval import Filters, _vector_candidates, search


def note(content, created_at=None, **kw):
    n = capture.capture_note(content, **kw)
    if created_at:
        get_conn().execute("UPDATE notes SET created_at = ? WHERE id = ?", (created_at, n["id"]))
        get_conn().commit()
    return n["id"]


def ids(**kw):
    return [n["id"] for n in search(**kw)]


@pytest.fixture
def warsaw(monkeypatch):
    monkeypatch.setenv("TIMEZONE", "Europe/Warsaw")  # UTC+1 in early March
    get_settings.cache_clear()


def test_date_bounds_are_inclusive_whole_local_days(warsaw):
    # Warsaw day 2026-03-05 = 2026-03-04T23:00:00Z .. 2026-03-05T22:59:59Z
    before = note("before", "2026-03-04T22:59:59.999999+00:00")  # 23:59:59.9 local on the 4th
    first = note("first", "2026-03-04T23:00:00+00:00")  # 00:00:00 local on the 5th
    micro = note("micro", "2026-03-05T12:00:00.123456+00:00")
    last = note("last", "2026-03-05T22:59:59+00:00")  # 23:59:59 local on the 5th
    last_micro = note("last micro", "2026-03-05T22:59:59.999999+00:00")
    after = note("after", "2026-03-05T23:00:00+00:00")  # 00:00:00 local on the 6th

    day = set(ids(since="2026-03-05", until="2026-03-05"))
    assert day == {first, micro, last, last_micro}
    assert before not in day and after not in day


def test_since_only_and_until_only(warsaw):
    a = note("a", "2026-03-01T10:00:00+00:00")
    b = note("b", "2026-03-05T10:00:00.5+00:00")
    c = note("c", "2026-03-09T10:00:00+00:00")
    assert ids(since="2026-03-05") == [c, b]
    assert ids(until="2026-03-05") == [b, a]
    assert ids(since="2026-03-02", until="2026-03-08") == [b]


def test_utc_default_timezone():
    inside = note("in", "2026-03-05T23:59:59+00:00")
    note("out", "2026-03-06T00:00:00+00:00")
    assert ids(since="2026-03-05", until="2026-03-05") == [inside]


def test_datetime_bounds(warsaw):
    a = note("a", "2026-03-05T09:00:00+00:00")  # 10:00 local
    b = note("b", "2026-03-05T11:00:00+00:00")  # 12:00 local
    # naive datetime = Warsaw local; with offset/Z = absolute
    assert ids(since="2026-03-05T11:00:00") == [b]  # 11:00 local = 10:00Z
    assert ids(since="2026-03-05T10:00:00Z") == [b]
    assert ids(since="2026-03-05T11:00:00+01:00", until="2026-03-05T12:00:00+01:00") == [b]
    assert ids(until="2026-03-05T10:00:00Z") == [a]
    assert ids(until="2026-03-05T11:00:00+02:00") == [a]  # 09:00Z, inclusive


def test_imported_offsets_and_formats_compare_correctly(warsaw):
    # created_at stored with other offsets / Z still compares as the same instant
    x = note("x", "2026-03-05T00:30:00+01:00")  # 2026-03-04T23:30:00Z -> 00:30 local on 5th
    y = note("y", "2026-03-05T23:30:00Z")  # 00:30 local on the 6th
    assert ids(since="2026-03-05", until="2026-03-05") == [x]
    assert ids(since="2026-03-06", until="2026-03-06") == [y]


def test_list_mode_newest_first_limited(warsaw):
    a = note("a", "2026-03-01T10:00:00+00:00")
    b = note("b", "2026-03-03T10:00:00.250000+00:00")
    c = note("c", "2026-03-03T10:00:00+00:00")  # same second as b, earlier (no fraction)
    d = note("d", "2026-03-04T10:00:00+00:00")
    assert ids() == [d, b, c, a]
    assert ids(top_k=2) == [d, b]
    assert ids(query="   ") == [d, b, c, a]
    assert all(n["score"] is None for n in search())


def test_category_filter_in_list_and_query_mode():
    task = note("buy milk and eggs", category="task")
    idea = note("buy a new idea notebook", category="idea")
    assert ids(category="task") == [task]
    assert ids(category="Idea") == [idea]  # normalized
    assert ids(query="buy", category="task") == [task]
    assert set(ids(query="buy")) == {task, idea}


def test_space_none_searches_all_spaces_and_space_filters():
    w = note("quarterly plan", space="work")
    p = note("weekend plan", space="personal")
    assert set(ids(query="plan")) == {w, p}
    assert ids(query="plan", space="work") == [w]
    assert ids(space="Personal") == [p]
    assert set(ids()) == {w, p}


def test_query_mode_applies_date_filter_and_scores(warsaw):
    old = note("budget review", "2026-02-01T10:00:00+00:00")
    new = note("budget planning", "2026-03-05T10:00:00+00:00")
    results = search("budget", since="2026-03-01")
    assert [n["id"] for n in results] == [new]
    assert isinstance(results[0]["score"], float)
    assert set(ids(query="budget")) == {old, new}


def test_vector_candidates_oversample_for_date_filter():
    n = get_settings().retrieval_candidate_n
    for i in range(n + 5):  # newer near-duplicates dominate the global top-N
        note(f"quarterly roadmap planning meeting draft {i}", "2026-06-01T10:00:00+00:00")
    old = note("quarterly roadmap planning meeting for the org", "2026-01-01T10:00:00+00:00")
    cands = _vector_candidates(
        get_conn(), "quarterly roadmap planning meeting", n, Filters(until="2026-02-01 00:00:00")
    )
    assert old in cands


@pytest.mark.parametrize(
    "kw",
    [
        {"since": "yesterday"},
        {"until": "2026-13-01"},
        {"since": "2026-02-30"},
        {"since": "03/05/2026"},
        {"since": "2026-03-06", "until": "2026-03-05"},
        {"since": "2026-03-05T12:00:00Z", "until": "2026-03-05T11:00:00Z"},
        {"space": "!!bad!!"},
        {"category": "!!bad!!"},
    ],
)
def test_invalid_filters_raise_value_error(kw):
    with pytest.raises(ValueError):
        search("x", **kw)
    with pytest.raises(ValueError):
        search(**kw)


def test_same_day_since_until_is_valid(warsaw):
    assert search(since="2026-03-05", until="2026-03-05") == []


def test_invalid_timezone_rejected(monkeypatch):
    monkeypatch.setenv("TIMEZONE", "Mars/Olympus")
    get_settings.cache_clear()
    with pytest.raises(ValueError):
        get_settings()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "s3cret")
    get_settings.cache_clear()
    with TestClient(app, headers={"Authorization": "Bearer s3cret"}) as c:
        yield c


def test_api_search_filters_and_list_mode(client, warsaw):
    a = note("alpha", "2026-03-05T10:00:00+00:00", space="work", category="task")
    note("beta", "2026-03-06T10:00:00+00:00", space="home")
    resp = client.get("/search", params={"space": "work", "category": "task",
                                         "since": "2026-03-05", "until": "2026-03-05"})
    assert resp.status_code == 200
    assert [n["id"] for n in resp.json()["notes"]] == [a]
    assert len(client.get("/search").json()["notes"]) == 2  # no q: newest notes of all spaces
    assert len(client.get("/search", params={"q": "alpha", "top_k": 1}).json()["notes"]) == 1


def test_api_search_invalid_input_422(client):
    assert client.get("/search", params={"since": "nope"}).status_code == 422
    assert client.get("/search", params={"since": "2026-03-06", "until": "2026-03-05"}).status_code == 422
    assert client.get("/search", params={"space": "!!"}).status_code == 422
