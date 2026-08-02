-- Per-reminder local schedule. Existing reminders retain the old 08:00 UTC behavior
-- until their first worker tick initializes next_run_at.
ALTER TABLE reminders ADD COLUMN local_time TEXT NOT NULL DEFAULT '08:00';
ALTER TABLE reminders ADD COLUMN timezone TEXT NOT NULL DEFAULT 'UTC';
ALTER TABLE reminders ADD COLUMN next_run_at TEXT;
ALTER TABLE reminders ADD COLUMN claim_token TEXT;
ALTER TABLE reminders ADD COLUMN claim_until TEXT;
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(deleted_at, next_run_at);
