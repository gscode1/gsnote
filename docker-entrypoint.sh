#!/bin/sh
set -e

DB_PATH="${DB_PATH:-./data/gsnote.db}"
mkdir -p "$(dirname "$DB_PATH")"

if [ "${LITESTREAM_ENABLED:-false}" = "true" ]; then
    if [ ! -f "$DB_PATH" ]; then
        echo "Litestream enabled, no local DB found — attempting restore from S3..."
        litestream restore -if-replica-exists -config /etc/litestream.yml "$DB_PATH" || \
            echo "No existing replica found; starting fresh."
    fi
    echo "Starting with Litestream replication..."
    exec litestream replicate -config /etc/litestream.yml -exec "$*"
fi

exec "$@"
