-- User-defined reminders: message pings and recurring query digests.
-- One daily tick job scans this table; no per-reminder scheduler jobs,
-- so firing state survives restarts by construction.
CREATE TABLE IF NOT EXISTS reminders (
  id            TEXT PRIMARY KEY,
  user_id       TEXT NOT NULL,     -- notes.source / channel user id
  space         TEXT NOT NULL,     -- active space at creation; scopes query digests
  message       TEXT NOT NULL,     -- what to say, in the user's own words
  kind          TEXT NOT NULL,     -- once | daily | weekly
  weekday       INTEGER,           -- 0=Mon..6=Sun (Python date.weekday()), for weekly
  fire_date     TEXT,              -- YYYY-MM-DD, for once
  window_days   INTEGER,           -- non-NULL = query digest: attach notes from last N days
  category      TEXT,              -- optional note filter for the query
  created_at    TEXT NOT NULL,
  last_fired_on TEXT,              -- YYYY-MM-DD guard: at most one fire per day
  deleted_at    TEXT               -- cancelled, or consumed 'once'
);
CREATE INDEX IF NOT EXISTS idx_reminders_active ON reminders(user_id, deleted_at);
