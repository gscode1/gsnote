-- Scheduled reviews are the only user-facing schedules.
-- Keep legacy rows readable, but migrate query digests and retire fixed messages.
UPDATE reminders
SET action_type = 'review'
WHERE action_type = 'digest'
   OR (
        action_type = 'notify'
        AND (window_mode IS NOT NULL OR window_days IS NOT NULL OR window_value IS NOT NULL)
      );

UPDATE reminders
SET deleted_at = CURRENT_TIMESTAMP
WHERE action_type = 'notify' AND deleted_at IS NULL;

ALTER TABLE reminders ADD COLUMN last_run_status TEXT;
ALTER TABLE reminders ADD COLUMN last_run_at TEXT;
ALTER TABLE reminders ADD COLUMN last_run_notes_count INTEGER;
ALTER TABLE reminders ADD COLUMN last_run_error TEXT;

CREATE TABLE IF NOT EXISTS schedule_runs (
  id            TEXT PRIMARY KEY,
  schedule_id   TEXT NOT NULL,
  fired_at      TEXT NOT NULL,
  status        TEXT NOT NULL,
  notes_count   INTEGER NOT NULL DEFAULT 0,
  result_summary TEXT,
  error_message TEXT,
  FOREIGN KEY (schedule_id) REFERENCES reminders(id)
);
CREATE INDEX IF NOT EXISTS idx_schedule_runs_schedule_id ON schedule_runs(schedule_id);
CREATE INDEX IF NOT EXISTS idx_schedule_runs_fired_at ON schedule_runs(fired_at);
