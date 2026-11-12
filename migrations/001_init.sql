CREATE TABLE notes (
  id         TEXT PRIMARY KEY,
  content    TEXT NOT NULL,
  category   TEXT NOT NULL DEFAULT 'note',
  source     TEXT NOT NULL DEFAULT 'user',
  space      TEXT NOT NULL DEFAULT 'default',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX idx_notes_space_created ON notes(space, created_at);
CREATE INDEX idx_notes_category ON notes(category);

-- keyword search (BM25)
CREATE VIRTUAL TABLE notes_fts USING fts5(note_id UNINDEXED, content);
