import json
import logging
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.capture import capture_note, delete_note, import_notes
from app.config import get_settings
from app.db import EmbeddingConfigMismatch, cursor, get_conn, run_migrations
from app.mcp_server import new_session_manager
from app.retrieval import search
from app.spaces import list_spaces

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_migrations()
    app.state.mcp_manager = new_session_manager()
    async with app.state.mcp_manager.run():
        yield


app = FastAPI(title="gsnote", lifespan=lifespan)


@app.exception_handler(EmbeddingConfigMismatch)
def embedding_mismatch_handler(request: Request, exc: EmbeddingConfigMismatch):
    return JSONResponse({"detail": str(exc)}, status_code=503)


def require_api_token(authorization: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.api_token:
        raise HTTPException(status_code=503, detail="HTTP API disabled: set API_TOKEN to enable it")
    expected = f"Bearer {settings.api_token}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


class MCPEndpoint:
    """ASGI endpoint for the MCP server: same bearer token as the API, then the session manager."""

    async def __call__(self, scope, receive, send) -> None:
        request = Request(scope, receive)
        try:
            require_api_token(request.headers.get("authorization"))
        except HTTPException as e:
            await JSONResponse({"detail": e.detail}, status_code=e.status_code)(scope, receive, send)
            return
        await request.app.state.mcp_manager.handle_request(scope, receive, send)


app.add_route("/mcp", MCPEndpoint())


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
def search_endpoint(
    q: str | None = None,
    space: str | None = None,
    category: str | None = None,
    since: str | None = None,
    until: str | None = None,
    top_k: int | None = None,
    broad: bool = False,
):
    try:
        notes = search(q, space=space, category=category, since=since, until=until, top_k=top_k,
                       gate=False if broad else None)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"notes": notes}


@app.get("/spaces", dependencies=[Depends(require_api_token)])
def spaces_endpoint():
    return list_spaces()


@app.delete("/notes/{note_id}", dependencies=[Depends(require_api_token)])
def delete_endpoint(note_id: str):
    if not delete_note(note_id):
        raise HTTPException(status_code=404, detail="Note not found")
    return {"deleted": note_id}


@app.get("/export", dependencies=[Depends(require_api_token)])
def export_endpoint():
    # Admin surface: behind API_TOKEN, exports every space.
    with cursor() as cur:
        rows = cur.execute(
            "SELECT id, content, category, source, space, created_at, updated_at"
            " FROM notes ORDER BY created_at"
        ).fetchall()
    payload = {
        "version": 2,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "notes": [dict(row) for row in rows],
    }
    return Response(
        content=json.dumps(payload, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="gsnote-export.json"'},
    )


@app.post("/import", dependencies=[Depends(require_api_token)])
def import_endpoint(payload: dict = Body(...)):
    try:
        return import_notes(payload)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
