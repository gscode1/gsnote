# gsnote on the Agent Memory Benchmark (AMB)

A memory provider that lets [AMB](https://github.com/vectorize-io/agent-memory-benchmark)
benchmark gsnote against its other providers (bm25, qdrant hybrid search, mem0, ...).
AMB is not forked: `run_amb.py` registers the provider at runtime and calls AMB's CLI.

AMB is pinned to commit `28b644bd4dd653c0207c7507280439390c829155`.

## Setup

```bash
eval/amb/setup.sh
```

Clones AMB into `eval/amb/.vendor/` (gitignored), builds its env with `uv sync --frozen`
(a large download: torch, qdrant, mem0, cognee, ...) and installs gsnote into it editable.

gsnote is installed with `--no-deps`: its `mcp>=2.3` requirement conflicts with the
`mcp` 1.x that AMB's lockfile resolves to (the benchmark does not use the MCP server).
The provider only imports `app.config/db/capture/retrieval/embeddings/spaces`, which need
only `sqlite-vec` on top of what AMB ships. AMB locks fastembed 0.6 (gsnote's own lock has
0.8); both serve `bge-small-en-v1.5`.

## Run

```bash
PY=eval/amb/.vendor/amb/.venv/bin/python

$PY eval/amb/run_amb.py providers                  # lists gsnote among the providers
$PY eval/amb/run_amb.py run --dataset precisionmembench --split single-turn \
    --memory gsnote --mode retrieval               # no LLM, no API keys
```

Any AMB `run`/`splits`/`dataset-stats` arguments work unchanged. Results go to
`eval/amb/outputs/` (gitignored) unless you pass `-o`. For `--mode retrieval` the shim sets a
placeholder `GEMINI_API_KEY` and `GROQ_API_KEY`, because AMB's `run` command demands them even
though that mode never calls an LLM (real keys in your env win).

Compare against the baselines with the same arguments: `--memory bm25`, `--memory qdrant`
(its results land under `outputs/.../hybrid-search/`; pass `--name` to choose the directory).
At the pinned commit AMB's qdrant provider crashes in every mode (its `async_retrieve` lacks the
`filters` argument), so the shim patches that signature; scoring is untouched.

precisionmembench returns up to 20 memories per query and penalises noise. The baselines have
no relevance cutoff, so at the default limit they score ~0 active passes; gsnote has a
relevance gate (`RETRIEVAL_GATE`, `RETRIEVAL_MIN_SIMILARITY`, `RETRIEVAL_RELATIVE_GAP`). `AMB_PMB_RETURN_CAP=1` (keep only the top N distinct results) separates them better.

### QA modes (need LLM keys)

`--mode rag` (and the agentic modes) use an answer LLM and a judge LLM, configured with env
vars, or a `.env` in `eval/amb/.vendor/amb/`:

```bash
export OMB_ANSWER_LLM=openai OMB_ANSWER_MODEL=gpt-4.1-mini   # providers: gemini, openai, groq, ...
export OMB_JUDGE_LLM=openai   OMB_JUDGE_MODEL=gpt-4.1-mini
export OPENAI_API_KEY=...                                    # or GEMINI_API_KEY / GROQ_API_KEY
# local models via any OpenAI-compatible server (Ollama, vLLM, LM Studio):
export OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=unused

$PY eval/amb/run_amb.py run --dataset locomo --split locomo10 --memory gsnote --query-limit 20
```

(`$PY eval/amb/run_amb.py splits --dataset locomo` lists a dataset's splits; AMB's defaults
are answer `groq`, judge `gemini`.)

## How the provider maps AMB onto gsnote

- **Storage**: `prepare()` points `DB_PATH` at `<store_dir>/gsnote.db` and resets it when
  `reset=True`. It clears gsnote's cached settings and closes the module-level connection so
  the new path takes effect.
- **user_id -> space**: already-valid names pass through; anything else is slugified and
  gets an 8-char SHA-1 suffix so distinct users never share a space. `None` searches all spaces.
- **Ingest**: each document is split with AMB's `chunk_text` (512 tokens, same as bm25 and
  qdrant), one note per chunk, id `<doc.id>::<i>`, via gsnote's real `import_notes` path
  (production embedding + FTS5 + sqlite-vec code). `Document.timestamp` becomes `created_at`
  when it parses as ISO-8601, else the ingest time. Re-ingesting is idempotent; the same
  document id under a different user raises.
- **Retrieve**: `app.retrieval.search(query, space=..., top_k=4*k)`, then chunk hits are
  deduplicated to the original document id in rank order and cut to `k`. Returned
  `Document.id` is the original id, `source_ids` is `[id]`, `content` is the matching chunk.
  `raw` is `{"results": [...gsnote result dicts with scores...]}`.
- `concurrency = 1`: gsnote keeps one shared SQLite connection and global settings.

## Configuration

| env var | default here | note |
|---|---|---|
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `BAAI/bge-small-en-v1.5` / `384` | fast. **gsnote production uses `bge-large-en-v1.5` / `1024`**; set both to benchmark that |

Any other gsnote setting (`RETRIEVAL_CANDIDATE_N`, `RETRIEVAL_GATE`, `RRF_K`, ...) is honoured from
the environment; the provider does not override gsnote's defaults (candidate pool 50 per branch).
Set `RETRIEVAL_GATE=false` and `EMBEDDING_QUERY_PREFIX=` to reproduce the pre-gate behaviour.
Note the first run downloads the embedding model (fastembed cache) and tiktoken's
`cl100k_base` (used by AMB's chunker).

## Tests

```bash
eval/amb/.vendor/amb/.venv/bin/python -m pytest eval/amb/tests -q
```

The file does `pytest.importorskip("memory_bench")`, so the plain gsnote suite
(`uv run pytest`) skips it when AMB is not installed.
