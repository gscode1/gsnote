#!/bin/sh
# Exercises the Litestream backup/restore runbook end-to-end against a local
# file-based replica (functionally identical mechanics to the S3 replica used
# in production — Litestream's replicate/restore commands are backend-agnostic).
#
# Usage: ./scripts/test_backup_restore.sh [path-to-litestream-binary]
set -e

LITESTREAM="${1:-litestream}"
WORKDIR="$(mktemp -d)"
DB="$WORKDIR/gsnote.db"
REPLICA="$WORKDIR/replica"
CONFIG="$WORKDIR/litestream.yml"

cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

cat > "$CONFIG" <<EOF
dbs:
  - path: $DB
    replicas:
      - type: file
        path: $REPLICA
EOF

echo "1. Creating DB with sample data..."
sqlite3 "$DB" "CREATE TABLE notes (id TEXT PRIMARY KEY, content TEXT);"
sqlite3 "$DB" "INSERT INTO notes VALUES ('n1', 'Backup runbook smoke test note');"

echo "2. Starting replication in background..."
"$LITESTREAM" replicate -config "$CONFIG" &
REPLICATE_PID=$!
sleep 2

echo "3. Writing more data while replicating..."
sqlite3 "$DB" "INSERT INTO notes VALUES ('n2', 'Second note written during replication');"
sleep 2

echo "4. Killing the replicator (simulates container death)..."
kill "$REPLICATE_PID"
wait "$REPLICATE_PID" 2>/dev/null || true

EXPECTED_COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM notes;")
echo "5. Original DB has $EXPECTED_COUNT note(s). Deleting it..."
rm -f "$DB" "$DB-wal" "$DB-shm"

echo "6. Restoring from replica..."
"$LITESTREAM" restore -if-replica-exists -config "$CONFIG" "$DB"

RESTORED_COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM notes;")
echo "7. Restored DB has $RESTORED_COUNT note(s)."

if [ "$RESTORED_COUNT" -ne "$EXPECTED_COUNT" ]; then
    echo "FAIL: restored row count ($RESTORED_COUNT) != original ($EXPECTED_COUNT)"
    exit 1
fi

sqlite3 "$DB" "SELECT * FROM notes;" | grep -q "n2" || {
    echo "FAIL: note written just before the kill was lost"
    exit 1
}

echo "PASS: restore recovered all data with zero loss."
