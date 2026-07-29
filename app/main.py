import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from app.capture import capture_note
from app.channels import Channel
from app.config import get_settings
from app.db import get_conn, run_migrations
from app.reporting import report
from app.retrieval import search
from app.scheduler import start_scheduler, stop_scheduler
from app.turn import handle_command, handle_message, handle_response, make_digest_sender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_channel: Channel | None = None


async def _build_channel() -> Channel | None:
    settings = get_settings()
    if settings.channel == "none":
        return None
    if settings.channel == "telegram":
        from app.channels.telegram import TelegramChannel

        return TelegramChannel()
    raise ValueError(f"Unknown channel: {settings.channel}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _channel
    run_migrations()

    _channel = await _build_channel()
    if _channel is not None:
        channel = _channel

        async def on_message(user_id: str, text: str) -> None:
            await handle_message(channel, user_id, text)

        _channel.on_message(on_message)
        _channel.on_response(handle_response)
        _channel.on_command(handle_command)
        await _channel.start()

    scheduler = start_scheduler(make_digest_sender(_channel))

    yield

    stop_scheduler()
    if _channel is not None:
        await _channel.stop()


app = FastAPI(title="gsnote", lifespan=lifespan)


@app.get("/health")
def health():
    get_conn()  # ensures DB connection is alive
    return {"status": "ok"}


class CaptureRequest(BaseModel):
    content: str
    source: str = "user"
    space: str = "personal"


@app.post("/capture")
async def capture_endpoint(req: CaptureRequest):
    return await capture_note(req.content, source=req.source, space=req.space)


class SearchResponse(BaseModel):
    notes: list[dict]


@app.get("/search", response_model=SearchResponse)
def search_endpoint(q: str, top_k: int | None = None, space: str | None = "personal"):
    return {"notes": search(q, top_k=top_k, space=space)}


class ReportRequest(BaseModel):
    query: str
    category: str | None = None
    space: str | None = "personal"


@app.post("/report")
async def report_endpoint(req: ReportRequest):
    return {"summary": await report(req.query, category=req.category, space=req.space)}
