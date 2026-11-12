#!/usr/bin/env bash
# Run gsnote locally (no Docker) for fast iteration with a small embedding model
# and a throwaway local DB. No third-party keys needed.
#
# Usage: API_TOKEN=devtoken ./scripts/dev-local.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export API_TOKEN=${API_TOKEN:-devtoken}
export EMBEDDING_MODEL=${EMBEDDING_MODEL:-BAAI/bge-small-en-v1.5}
export EMBEDDING_DIM=${EMBEDDING_DIM:-384}
export DB_PATH=${DB_PATH:-./data/gsnote-dev.db}
export LITESTREAM_ENABLED=false

mkdir -p ./data
echo ">> Starting uvicorn (db=$DB_PATH, model=$EMBEDDING_MODEL)…"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
