"""Re-embed every note with the current EMBEDDING_* settings: `python -m app.reindex`.

Runs locally against DB_PATH (not over HTTP) and meant to run with the server stopped. Changing
EMBEDDING_MODEL / EMBEDDING_DIM / EMBEDDING_DOCUMENT_PREFIX on a populated database makes the
stored vectors incompatible with new queries; this rebuilds them.

Atomic swap: vectors are first computed into a plain staging table (committed per batch). Only
when all notes are embedded does one transaction drop the old vec0 table, create a new one with
the current dimension, copy the staged vectors in and update the recorded config. The live
note_vectors table is untouched until then, so an interrupted run changes nothing and can simply
be rerun (it restarts from scratch). `ALTER TABLE ... RENAME` is not used: as of sqlite-vec 0.1.9
it renames a vec0 table but not its shadow tables, which leaves the table unusable.
"""
import argparse
import os
import sys
from collections.abc import Callable

from sqlite_vec import serialize_float32

from app import db
from app.config import get_settings
from app.embeddings import embed_documents

STAGING = "note_vectors_reindex"
DEFAULT_BATCH_SIZE = 32


def _note_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]


def _stage(conn, batch_size: int, progress: Callable[[int, int], None]) -> int:
    """Embed all notes into the staging table (replacing any leftover from an interrupted run)."""
    conn.execute(f"DROP TABLE IF EXISTS {STAGING}")
    conn.execute(f"CREATE TABLE {STAGING} (note_id TEXT PRIMARY KEY, embedding BLOB NOT NULL)")
    conn.commit()
    total = _note_count(conn)
    done = 0
    cursor = conn.execute("SELECT id, content FROM notes ORDER BY created_at, id")
    while batch := cursor.fetchmany(batch_size):
        vectors = embed_documents([row["content"] for row in batch])
        conn.executemany(
            f"INSERT INTO {STAGING} (note_id, embedding) VALUES (?, ?)",
            [(row["id"], serialize_float32(vec)) for row, vec in zip(batch, vectors)],
        )
        conn.commit()
        done += len(batch)
        progress(done, total)
    return done


def _swap(conn, config: dict[str, str]) -> int:
    """One transaction: reconcile staging with notes, replace note_vectors, record the config."""
    dim = int(config["embedding_dim"])
    conn.execute("BEGIN IMMEDIATE")
    try:
        # Notes added/deleted while we were embedding (the server should be stopped; be safe anyway).
        conn.execute(f"DELETE FROM {STAGING} WHERE note_id NOT IN (SELECT id FROM notes)")
        missing = conn.execute(
            f"SELECT id, content FROM notes WHERE id NOT IN (SELECT note_id FROM {STAGING})"
        ).fetchall()
        for start in range(0, len(missing), DEFAULT_BATCH_SIZE):
            chunk = missing[start:start + DEFAULT_BATCH_SIZE]
            vectors = embed_documents([row["content"] for row in chunk])
            conn.executemany(
                f"INSERT INTO {STAGING} (note_id, embedding) VALUES (?, ?)",
                [(row["id"], serialize_float32(vec)) for row, vec in zip(chunk, vectors)],
            )
        conn.execute("DROP TABLE IF EXISTS note_vectors")
        conn.execute(
            f"CREATE VIRTUAL TABLE note_vectors USING vec0(note_id TEXT PRIMARY KEY, embedding FLOAT[{dim}])"
        )
        conn.execute(
            f"INSERT INTO note_vectors (note_id, embedding) SELECT note_id, embedding FROM {STAGING}"
        )
        count = conn.execute("SELECT COUNT(*) FROM note_vectors").fetchone()[0]
        conn.execute(f"DROP TABLE {STAGING}")
        db.record_embedding_config(conn, config)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return count


def reindex(batch_size: int = DEFAULT_BATCH_SIZE, progress: Callable[[int, int], None] | None = None) -> dict:
    """Re-embed all notes with the current settings and swap the vector table. Returns a summary."""
    conn = db.get_conn()
    previous = db.recorded_embedding_config(conn)
    config = db.current_embedding_config()
    db.run_migrations()  # schema + meta table; a config mismatch is flagged, not raised
    staged = _stage(conn, batch_size, progress or (lambda done, total: None))
    count = _swap(conn, config)
    db.check_embedding_config(conn)  # clears the mismatch flag in this process
    return {"notes": count, "staged": staged, "previous": previous, "current": config}


def describe(conn) -> dict:
    """Recorded vs current config and note count, without changing anything."""
    recorded = db.recorded_embedding_config(conn)
    current = db.current_embedding_config()
    declared = db.declared_vector_dim(conn)
    unchanged = bool(recorded) and recorded == current and declared in (None, int(current["embedding_dim"]))
    return {
        "recorded": recorded,
        "current": current,
        "vector_table_dim": declared,
        "notes": _note_count(conn),
        "reindex_needed": not unchanged,
    }


def _fmt(config: dict[str, str]) -> str:
    if not config:
        return "(not recorded)"
    return ", ".join(f"{k.removeprefix('embedding_')}={v!r}" for k, v in sorted(config.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.reindex",
        description="Re-embed all notes with the current EMBEDDING_* settings (run with the server stopped).",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="report recorded vs current embedding config and the note count; change nothing")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="notes embedded per call")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    settings = get_settings()
    print(f"Database: {settings.db_path}")
    try:
        if args.dry_run:
            if not os.path.exists(settings.db_path):
                print(f"error: database not found: {settings.db_path}", file=sys.stderr)
                return 1
            info = describe(db.get_conn())
            print(f"Recorded config: {_fmt(info['recorded'])}")
            print(f"Vector table dim: {info['vector_table_dim']}")
            print(f"Current config:  {_fmt(info['current'])}")
            print(f"Notes: {info['notes']}")
            print("Reindex needed: " + ("yes" if info["reindex_needed"] else "no (config unchanged)"))
            print("Dry run: nothing changed.")
            return 0

        def progress(done: int, total: int) -> None:
            print(f"\rEmbedding {done}/{total}", end="" if done < total else "\n", flush=True)

        summary = reindex(args.batch_size, progress)
    except Exception as e:  # operator-facing CLI: message, not a traceback
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Reindexed {summary['notes']} notes.")
    print(f"  before: {_fmt(summary['previous'])}")
    print(f"  after:  {_fmt(summary['current'])}")
    print("Start the server again to use the new embeddings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
