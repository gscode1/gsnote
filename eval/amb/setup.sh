#!/usr/bin/env bash
# Clone AMB at the pinned commit into eval/amb/.vendor/amb and build its env with gsnote installed.
# Override AMB_REPO to use a mirror. Safe to re-run.
set -euo pipefail

AMB_REPO="${AMB_REPO:-https://github.com/vectorize-io/agent-memory-benchmark}"
AMB_SHA="28b644bd4dd653c0207c7507280439390c829155"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
VENDOR="$HERE/.vendor/amb"

if [ ! -d "$VENDOR/.git" ]; then
  mkdir -p "$HERE/.vendor"
  git clone "$AMB_REPO" "$VENDOR"
fi
git -C "$VENDOR" fetch --quiet origin "$AMB_SHA" 2>/dev/null || true
git -C "$VENDOR" checkout --quiet "$AMB_SHA"

# AMB's own pinned environment (python 3.11+; torch, qdrant, mem0, ... is a large download).
(cd "$VENDOR" && uv sync --frozen)

# gsnote goes in with --no-deps: its `mcp>=2.3` pin conflicts with AMB's locked mcp 1.x, and
# the provider only imports app.{config,db,capture,retrieval,embeddings,spaces}, which need
# just sqlite-vec on top of what AMB already ships (fastembed, pydantic-settings, httpx).
uv pip install --python "$VENDOR/.venv/bin/python" --no-deps -e "$REPO_ROOT"
uv pip install --python "$VENDOR/.venv/bin/python" "sqlite-vec>=0.1.6" pytest

echo "ok. run: $VENDOR/.venv/bin/python $HERE/run_amb.py providers"
