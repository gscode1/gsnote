"""gsnote as an Agent Memory Benchmark (AMB) memory provider.

Runs gsnote's real capture and retrieval code paths (``app.capture.import_notes`` for
ingest, ``app.retrieval.search`` for hybrid vector + FTS5 retrieval) against a throwaway
SQLite file under the run's store directory. Nothing in ``app/`` is patched.

Mapping:
  AMB user_id    -> gsnote space (deterministic sanitiser, see ``space_for_user``)
  AMB Document   -> one note per 512-token chunk (AMB's ``chunk_text``, same as bm25),
                    note id ``"<doc.id>::<chunk index>"``
  Document.timestamp -> note ``created_at`` (when it parses as ISO-8601)
  retrieve()     -> ``search(query, space=..., top_k=k)``: the top-k distinct CHUNKS, like AMB's
                    bm25 baseline (no per-document dedup, so several chunks of one session may
                    come back). Each result is a Document whose ``id`` is the original doc id
                    (so ids can repeat), ``source_ids=[id]`` and ``content`` the chunk text.
                    Deduping to one chunk per document dropped evidence chunks: another chunk
                    of the same session often outranks the one holding the answer.
  GSNOTE_AMB_GATE -> "on"/"off" forces gsnote's relevance gate for retrieve() (default: follow
                    the RETRIEVAL_GATE setting).

Process-global state: gsnote keeps its settings and SQLite connection in module globals,
so one provider instance per process owns them. ``concurrency = 1`` accordingly.

Embedding model defaults to bge-small-en-v1.5 (384d) for speed. gsnote production uses
bge-large-en-v1.5 (1024d); override with EMBEDDING_MODEL + EMBEDDING_DIM.
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from memory_bench.memory.base import MemoryProvider
from memory_bench.models import Document
from memory_bench.utils import chunk_text

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_DIM = "384"
DB_FILENAME = "gsnote.db"
IMPORT_BATCH = 64

_SPACE_MAX = 32  # app.spaces: [a-z0-9][a-z0-9-]{0,31}
_ID_SEP = "::"


def space_for_user(user_id: str | None) -> str:
    """Map an AMB user_id to a valid gsnote space name, deterministically and injectively-ish.

    Names that are already valid spaces pass through unchanged. Anything else is slugified
    and suffixed with 8 hex chars of the SHA-1 of the original, so distinct user ids never
    share a space (the suffix is the isolation guarantee, the slug is just for readability).
    """
    if user_id is None:
        return "default"
    slug = re.sub(r"[^a-z0-9]+", "-", user_id.lower()).strip("-")
    if slug == user_id and 0 < len(slug) <= _SPACE_MAX:
        return slug
    digest = hashlib.sha1(user_id.encode("utf-8")).hexdigest()[:8]
    prefix = slug[: _SPACE_MAX - 9].rstrip("-")
    return f"{prefix}-{digest}" if prefix else f"u-{digest}"


def _created_at(timestamp: str | None) -> str | None:
    """Normalise an AMB timestamp to ISO-8601 for gsnote, or None to let gsnote use 'now'."""
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.strip().replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


def _gate_from_env() -> bool | None:
    """GSNOTE_AMB_GATE: "on"/"off" (also true/false/1/0) -> force the gate; unset/empty -> None."""
    raw = os.environ.get("GSNOTE_AMB_GATE", "").strip().lower()
    if raw in ("", "default"):
        return None
    if raw in ("on", "true", "1", "yes"):
        return True
    if raw in ("off", "false", "0", "no"):
        return False
    raise ValueError(f"GSNOTE_AMB_GATE must be 'on' or 'off', got {raw!r}")


def _original_id(note_id: str) -> str:
    return note_id.rsplit(_ID_SEP, 1)[0]


class GsnoteMemoryProvider(MemoryProvider):
    name = "gsnote"
    description = (
        "gsnote: self-hosted notes memory. Hybrid search (sqlite-vec vectors + FTS5 BM25, fused "
        "with RRF) over a single SQLite file. Documents are split into 512-token chunks, one "
        "note per chunk; retrieval returns the top-k chunks; user_id maps to a gsnote space. Default embedding bge-small-en-v1.5 (384d)."
    )
    kind = "local"
    link = "https://github.com/gscode1/gsnote"
    concurrency = 1  # gsnote shares one SQLite connection and global settings per process

    def __init__(self) -> None:
        self._ready = False
        self._tmpdir: tempfile.TemporaryDirectory | None = None

    # -- storage -------------------------------------------------------------------

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        store_dir = Path(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)
        self._open(str(store_dir / DB_FILENAME), reset=reset)

    def _open(self, db_path: str, reset: bool) -> None:
        # Settings are lru_cached and the connection is a module global in app.db, so set the
        # environment first, then drop both caches so the new DB_PATH is picked up.
        os.environ.setdefault("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        os.environ.setdefault("EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM)
        os.environ["DB_PATH"] = db_path

        from app import db
        from app.config import get_settings

        get_settings.cache_clear()
        if db._conn is not None:
            db._conn.close()
            db._conn = None
        if reset:
            for suffix in ("", "-wal", "-shm"):
                if os.path.exists(db_path + suffix):
                    os.remove(db_path + suffix)
        db.get_conn()
        db.run_migrations()
        self._ready = True

    def _ensure_ready(self) -> None:
        if not self._ready:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="gsnote_amb_")
            self._open(os.path.join(self._tmpdir.name, DB_FILENAME), reset=True)

    def cleanup(self) -> None:
        from app import db

        if db._conn is not None:
            db._conn.close()
            db._conn = None
        self._ready = False
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    # -- ingest --------------------------------------------------------------------

    def ingest(self, documents: list[Document]) -> None:
        self._ensure_ready()
        from app.capture import import_notes
        from app.db import get_conn

        notes: list[dict] = []
        for doc in documents:
            space = space_for_user(doc.user_id)
            created_at = _created_at(doc.timestamp)
            for i, chunk in enumerate(chunk_text(doc.content)):
                note = {
                    "id": f"{doc.id}{_ID_SEP}{i}",
                    "content": chunk,
                    "space": space,
                    "category": "note",
                    "source": "amb",
                }
                if created_at:
                    note["created_at"] = created_at
                notes.append(note)

        conn = get_conn()
        for start in range(0, len(notes), IMPORT_BATCH):
            batch = notes[start:start + IMPORT_BATCH]
            # import_notes silently skips ids that already exist. That is right for re-ingesting
            # the same document, but the same doc id under a different user would be dropped
            # without a trace, so fail loudly instead.
            marks = ",".join("?" for _ in batch)
            existing = dict(
                conn.execute(f"SELECT id, space FROM notes WHERE id IN ({marks})", [n["id"] for n in batch])
            )
            for n in batch:
                if n["id"] in existing and existing[n["id"]] != n["space"]:
                    raise ValueError(
                        f"document id {_original_id(n['id'])!r} already ingested under another "
                        f"user (space {existing[n['id']]!r}, now {n['space']!r})"
                    )
            import_notes({"version": 2, "notes": batch})

    # -- retrieve ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        self._ensure_ready()
        from app.retrieval import search

        space = space_for_user(user_id) if user_id is not None else None
        results = search(query, space=space, top_k=max(k, 1), gate=_gate_from_env())

        # Top-k distinct chunks, no per-document dedup (same as AMB's bm25 baseline). Chunk note
        # ids are unique, so the same doc id may appear several times.
        docs = [
            Document(
                id=(orig := _original_id(note["id"])),
                content=note["content"],
                user_id=user_id,
                timestamp=note.get("created_at"),
                source_ids=[orig],
            )
            for note in results[:k]
        ]
        # Some AMB datasets (e.g. LongMemEval) put raw_response into the answer prompt instead of
        # the rendered docs, so raw must hold exactly the k returned memories (with their dates),
        # never the over-fetched candidate pool.
        raw = [{"id": d.id, "date": d.timestamp, "content": d.content} for d in docs]
        return docs, {"results": raw}
