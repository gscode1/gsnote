-- Per-user IANA timezone preference. NULL means the user has not configured one.
ALTER TABLE user_settings ADD COLUMN timezone TEXT;
