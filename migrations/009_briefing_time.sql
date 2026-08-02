-- User-local daily briefing time; the shared minute worker evaluates this in the
-- user's IANA timezone. Existing opted-in users retain an 08:00 default.
ALTER TABLE user_settings ADD COLUMN briefing_time TEXT NOT NULL DEFAULT '08:00';
