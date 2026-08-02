-- Explicit query-window semantics for reminder digests.
-- NULL preserves plain message reminders and legacy window_days rows.
ALTER TABLE reminders ADD COLUMN window_mode TEXT;
ALTER TABLE reminders ADD COLUMN window_value INTEGER;
