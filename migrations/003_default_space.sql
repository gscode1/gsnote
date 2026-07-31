-- Named spaces replaced the work/personal pair: the built-in space is now "default".
-- Existing installs keep any spaces they created; only the old built-in "personal" is renamed.
UPDATE notes SET space = 'default' WHERE space = 'personal';
UPDATE user_settings SET space = 'default' WHERE space = 'personal';

-- ponytail: the notes.space column DEFAULT is still 'personal' (SQLite can't alter it
-- without a table rebuild). Harmless: all inserts pass space explicitly. Rebuild the
-- table here if a code path ever relies on the column default.
