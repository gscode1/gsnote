import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import sqlite_vec

from app.config import get_settings
from app.embeddings import document_prefix

logger = logging.getLogger(__name__)

REINDEX_COMMAND = "python -m app.reindex"

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

    known = {path.stem for path in _MIGRATIONS_DIR.glob("*.sql")}
    legacy = sorted(applied - known)
    if legacy:
        raise RuntimeError(
            f"This database was created by gsnote <= v0-bot (unknown migrations: {', '.join(legacy)}) "
            "and cannot be opened by this version. Export your notes with the old version "
            "(GET /export), point DB_PATH at a new file, start this version, then run "
            "`gsnote import <file>`."
        )

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
    check_embedding_config(conn)


class EmbeddingConfigMismatch(RuntimeError):
    """The vectors in the database were built with a different embedding config than the current one."""


# Set by run_migrations() at startup: why vector features are disabled, or None when they are fine.
# While set, capture/import and query search refuse (require_vectors_ready); everything that does
# not touch vectors (listing, delete, export, spaces) keeps working.
_embedding_mismatch: str | None = None


def require_vectors_ready() -> None:
    if _embedding_mismatch is not None:
        raise EmbeddingConfigMismatch(_embedding_mismatch)


def current_embedding_config() -> dict[str, str]:
    """The embedding config the running process would embed with (document prefix is the effective one)."""
    settings = get_settings()
    return {
        "embedding_model": settings.embedding_model,
        "embedding_dim": str(settings.embedding_dim),
        "embedding_document_prefix": document_prefix(),
    }


def recorded_embedding_config(conn: sqlite3.Connection) -> dict[str, str]:
    """The embedding config recorded in the meta table ({} when nothing is recorded or no meta table)."""
    try:
        rows = conn.execute("SELECT key, value FROM meta WHERE key LIKE 'embedding_%'").fetchall()
    except sqlite3.OperationalError:  # database predates the meta migration
        return {}
    return {row[0]: row[1] for row in rows}


def record_embedding_config(conn: sqlite3.Connection, config: dict[str, str]) -> None:
    conn.executemany("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", list(config.items()))


def declared_vector_dim(conn: sqlite3.Connection) -> int | None:
    """Dimension declared by the note_vectors vec0 table, or None if the table does not exist."""
    row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='note_vectors'").fetchone()
    match = re.search(r"FLOAT\[(\d+)\]", row[0], re.IGNORECASE) if row else None
    return int(match.group(1)) if match else None


def _describe(config: dict[str, str]) -> str:
    if not config:
        return "not recorded"
    return ", ".join(f"{k.removeprefix('embedding_')}={v!r}" for k, v in sorted(config.items()))


def check_embedding_config(conn: sqlite3.Connection) -> None:
    """Record the embedding config on first start; flag a mismatch with the current settings."""
    global _embedding_mismatch
    _embedding_mismatch = None
    current = current_embedding_config()
    recorded = recorded_embedding_config(conn)
    declared = declared_vector_dim(conn)
    dim_ok = declared is None or declared == int(current["embedding_dim"])

    if not recorded and dim_ok:
        # New database, or one created before the config was recorded: assume it matches.
        record_embedding_config(conn, current)
        conn.commit()
        return
    if recorded == current and dim_ok:
        return

    why = f"recorded: {_describe(recorded)}; vector table dimension: {declared}; current: {_describe(current)}"
    _embedding_mismatch = (
        f"The embedding configuration changed since the stored notes were embedded ({why}). Mixing "
        "vector spaces gives silently wrong search results, so capture/import and query search are "
        f"disabled. Re-embed all notes with the current settings: `{REINDEX_COMMAND}` "
        "(stop the server first; see README, 'Changing the embedding model'), then restart. "
        "Alternatively restore the previous EMBEDDING_* settings."
    )
    logger.error(_embedding_mismatch)


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
