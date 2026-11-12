"""Run the Agent Memory Benchmark CLI with gsnote registered as a memory provider.

    eval/amb/.vendor/amb/.venv/bin/python eval/amb/run_amb.py run \
        --dataset precisionmembench --split single-turn --memory gsnote --mode retrieval

Takes exactly AMB's `amb` CLI arguments; AMB itself stays unmodified.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
OUTPUT_DIR = HERE / "outputs"

for path in (str(HERE), str(REPO_ROOT)):  # gsnote_provider, and `app` if not pip-installed
    if path not in sys.path:
        sys.path.insert(0, path)


def _patch_qdrant_signature(cls) -> None:
    """At the pinned AMB commit, the qdrant provider's async_retrieve() lacks the `filters`
    parameter that every mode now passes, so `--memory qdrant` raises TypeError. Accept and
    ignore it (the provider declares supports_filters = False, so it is always None)."""
    original = cls.async_retrieve

    async def async_retrieve(self, query, k=50, user_id=None, query_timestamp=None, filters=None):
        return await original(self, query, k, user_id, query_timestamp)

    cls.async_retrieve = async_retrieve


def main(argv: list[str]) -> None:
    from memory_bench.cli import app
    from memory_bench.memory import REGISTRY

    from gsnote_provider import GsnoteMemoryProvider

    REGISTRY["gsnote"] = GsnoteMemoryProvider
    _patch_qdrant_signature(REGISTRY["qdrant"])

    if argv and argv[0] == "run":
        if not any(a in ("-o", "--output-dir") or a.startswith("--output-dir=") for a in argv):
            argv = [*argv, "--output-dir", str(OUTPUT_DIR)]
        # AMB's `run` insists on a Gemini key and builds a Groq answer client up front even though
        # --mode retrieval never calls an LLM, so give both a placeholder (real keys win).
        if "retrieval" in argv and "--mode" in argv and argv[argv.index("--mode") + 1] == "retrieval":
            os.environ.setdefault("GEMINI_API_KEY", "unused-in-retrieval-mode")
            os.environ.setdefault("GROQ_API_KEY", "unused-in-retrieval-mode")
    app(args=argv, prog_name="amb")


if __name__ == "__main__":
    main(sys.argv[1:])
