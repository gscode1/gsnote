-- Opt-in daily commitment briefing (#34): notes carry a nullable due date
-- (YYYY-MM-DD, app-local), users opt in via /briefing on (default off).
ALTER TABLE notes ADD COLUMN due_date TEXT;
CREATE INDEX IF NOT EXISTS idx_notes_due ON notes(due_date);
ALTER TABLE user_settings ADD COLUMN briefing_enabled INTEGER NOT NULL DEFAULT 0;
