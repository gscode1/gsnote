-- key/value settings the database itself depends on. Currently the embedding config its vectors
-- were built with (embedding_model, embedding_dim, embedding_document_prefix), so a changed
-- EMBEDDING_* setting is detected at startup instead of silently mixing vector spaces.
CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
