import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
os.environ.setdefault("EMBEDDING_DIM", "384")
os.environ.setdefault("DB_PATH", "./data/test_gsnote.db")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("CHANNEL", "none")  # lifespan must not need a Telegram token

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic_ai import models  # noqa: E402

# Block accidental real LLM calls in the unit-test suite. Tests that exercise agents
# must use TestModel/FunctionModel (which are exempt from this flag) via agent.override().
models.ALLOW_MODEL_REQUESTS = False

from app import db  # noqa: E402
from app.config import get_settings  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_db():
    from app.turn import reset_for_tests

    get_settings.cache_clear()
    reset_for_tests()
    db_path = get_settings().db_path
    db.reset_db_for_tests(db_path)
    yield
    db.get_conn().close()
    db._conn = None
    if os.path.exists(db_path):
        os.remove(db_path)
    for ext in ("-wal", "-shm"):
        p = db_path + ext
        if os.path.exists(p):
            os.remove(p)
