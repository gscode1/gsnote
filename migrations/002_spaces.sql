-- Work/personal separation (Option B): a per-note "space" plus a per-user active space.
ALTER TABLE notes ADD COLUMN space TEXT NOT NULL DEFAULT 'personal';
CREATE INDEX IF NOT EXISTS idx_notes_space ON notes(space, created_at);

-- Active space per user (which space new notes/queries scope to). Toggled via /work, /personal.
CREATE TABLE IF NOT EXISTS user_settings (
  user_id    TEXT PRIMARY KEY,
  space      TEXT NOT NULL DEFAULT 'personal',
  updated_at TEXT NOT NULL
);
