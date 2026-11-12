import json
import logging
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel

from app.capture import capture_note
from app.config import get_settings
from app.db import cursor, get_conn, run_migrations
from app.retrieval import search
from app.spaces import list_spaces, normalize_space

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_migrations()
    yield


app = FastAPI(title="gsnote", lifespan=lifespan)


def require_api_token(authorization: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.api_token:
        raise HTTPException(status_code=503, detail="HTTP API disabled: set API_TOKEN to enable it")
    expected = f"Bearer {settings.api_token}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health():
    get_conn()  # ensures DB connection is alive
    return {"status": "ok"}


class CaptureRequest(BaseModel):
    content: str
    source: str = "user"
    space: str = "default"
    category: str = "note"


@app.post("/capture", dependencies=[Depends(require_api_token)])
def capture_endpoint(req: CaptureRequest):
    try:
        return capture_note(req.content, source=req.source, space=req.space, category=req.category)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


class SearchResponse(BaseModel):
    notes: list[dict]


@app.get("/search", response_model=SearchResponse, dependencies=[Depends(require_api_token)])
def search_endpoint(q: str, top_k: int | None = None, space: str | None = "default"):
    if space is not None:
        try:
            space = normalize_space(space)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
    return {"notes": search(q, top_k=top_k, space=space)}


@app.get("/spaces", dependencies=[Depends(require_api_token)])
def spaces_endpoint():
    return list_spaces()


@app.get("/export", dependencies=[Depends(require_api_token)])
def export_endpoint():
    # Admin surface: behind API_TOKEN, exports every space.
    with cursor() as cur:
        rows = cur.execute(
            "SELECT id, content, category, importance, source, space,"
            " created_at, updated_at, last_accessed_at, access_count"
            " FROM notes WHERE deleted_at IS NULL ORDER BY created_at"
        ).fetchall()
    payload = {
        "version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "notes": [dict(row) for row in rows],
    }
    return Response(
        content=json.dumps(payload, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="gsnote-export.json"'},
    )
