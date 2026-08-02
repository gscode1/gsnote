-- Add action_type column to reminders table (notify | digest), defaulting to notify.
ALTER TABLE reminders ADD COLUMN action_type TEXT NOT NULL DEFAULT 'notify';
