#!/usr/bin/env bash
# Run gsnote locally (no Docker) for fast iteration, using the real LLM + Telegram
# credentials pulled from the k8s Secret at runtime (nothing written to disk).
#
# IMPORTANT: Telegram allows only ONE poller per bot token. This script scales the
# cluster Deployment to 0 first so the local process can take over the bot, and
# scales it back to 1 on exit. Use scripts/dev-stop.sh or Ctrl-C to restore.
#
# Usage: ./scripts/dev-local.sh
set -euo pipefail
cd "$(dirname "$0")/.."

NS=gsnote
echo ">> Scaling cluster deployment to 0 (so local can poll the bot)…"
kubectl -n "$NS" scale deploy/gsnote --replicas=0 >/dev/null
trap 'echo ">> Restoring cluster deployment to 1…"; kubectl -n '"$NS"' scale deploy/gsnote --replicas=1 >/dev/null' EXIT

echo ">> Loading config from k8s ConfigMap + Secret…"
export LLM_PROVIDER=$(kubectl -n "$NS" get configmap gsnote-config -o jsonpath='{.data.LLM_PROVIDER}')
export LLM_BASE_URL=$(kubectl -n "$NS" get configmap gsnote-config -o jsonpath='{.data.LLM_BASE_URL}')
export CLASSIFIER_MODEL=$(kubectl -n "$NS" get configmap gsnote-config -o jsonpath='{.data.CLASSIFIER_MODEL}')
export ANSWER_MODEL=$(kubectl -n "$NS" get configmap gsnote-config -o jsonpath='{.data.ANSWER_MODEL}')
export CHANNEL=${CHANNEL:-telegram}

# Secrets
export LLM_API_KEY=$(kubectl -n "$NS" get secret gsnote-secrets -o jsonpath='{.data.LLM_API_KEY}' | base64 -d)
export TELEGRAM_BOT_TOKEN=$(kubectl -n "$NS" get secret gsnote-secrets -o jsonpath='{.data.TELEGRAM_BOT_TOKEN}' | base64 -d)
export TELEGRAM_ALLOWED_USER_IDS=$(kubectl -n "$NS" get secret gsnote-secrets -o jsonpath='{.data.TELEGRAM_ALLOWED_USER_IDS}' | base64 -d)

# Local-only overrides: fast embedding model, local DB, no Litestream.
export EMBEDDING_MODEL=${EMBEDDING_MODEL:-BAAI/bge-small-en-v1.5}
export EMBEDDING_DIM=${EMBEDDING_DIM:-384}
export DB_PATH=${DB_PATH:-./data/gsnote-dev.db}
export LITESTREAM_ENABLED=false

mkdir -p ./data
echo ">> Starting uvicorn (CHANNEL=$CHANNEL, model=$ANSWER_MODEL, db=$DB_PATH)…"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
