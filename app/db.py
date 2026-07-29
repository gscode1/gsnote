import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import sqlite_vec

from app.config import get_settings

_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


_conn: sqlite3.Connection | None = None


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        settings = get_settings()
        _conn = _connect(settings.db_path)
    return _conn


@contextmanager
def cursor():
    conn = get_conn()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def run_migrations() -> None:
    """Idempotent migration runner: applies any .sql file in migrations/ not yet recorded."""
    conn = get_conn()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}

    for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        version = path.stem
        if version in applied:
            continue
        sql = path.read_text()
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (version, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()

    _ensure_vector_table(conn)


def _ensure_vector_table(conn: sqlite3.Connection) -> None:
    settings = get_settings()
    dim = settings.embedding_dim
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='note_vectors'"
    ).fetchone()
    if row is None:
        conn.execute(
            f"CREATE VIRTUAL TABLE note_vectors USING vec0(note_id TEXT PRIMARY KEY, embedding FLOAT[{dim}])"
        )
        conn.commit()


def reset_db_for_tests(db_path: str) -> None:
    """Test helper: point at a fresh db file and run migrations."""
    global _conn
    if os.path.exists(db_path):
        os.remove(db_path)
    _conn = _connect(db_path)
    run_migrations()
