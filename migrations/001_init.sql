-- core note
CREATE TABLE IF NOT EXISTS notes (
  id            TEXT PRIMARY KEY,
  content       TEXT NOT NULL,
  category      TEXT DEFAULT 'note',
  importance    INTEGER DEFAULT 3,
  source        TEXT DEFAULT 'user',
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL,
  last_accessed_at TEXT,
  access_count  INTEGER DEFAULT 0,
  deleted_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_notes_created   ON notes(created_at);
CREATE INDEX IF NOT EXISTS idx_notes_category  ON notes(category);
CREATE INDEX IF NOT EXISTS idx_notes_accessed  ON notes(last_accessed_at);

-- keyword search (BM25)
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(note_id UNINDEXED, content);

-- knowledge graph
CREATE TABLE IF NOT EXISTS edges (
  id        TEXT PRIMARY KEY,
  from_id   TEXT NOT NULL,
  to_id     TEXT NOT NULL,
  type      TEXT NOT NULL,
  weight    REAL DEFAULT 1.0,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id);
CREATE INDEX IF NOT EXISTS idx_edges_to   ON edges(to_id);

-- proactive resurfacing log
CREATE TABLE IF NOT EXISTS notifications (
  id            TEXT PRIMARY KEY,
  note_ids      TEXT NOT NULL,
  kind          TEXT NOT NULL,
  channel       TEXT NOT NULL,
  sent_at       TEXT NOT NULL,
  user_response TEXT,
  snooze_until  TEXT
);
CREATE INDEX IF NOT EXISTS idx_notif_sent ON notifications(sent_at);

-- migration bookkeeping
CREATE TABLE IF NOT EXISTS schema_migrations (
  version TEXT PRIMARY KEY,
  applied_at TEXT NOT NULL
);
