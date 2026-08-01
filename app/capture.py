"""Capture pipeline (PRD §5): classify -> store -> embed -> graph edges -> ack.

Classification lives here: it is the capture pipeline's own first step.
"""
import json
import uuid
from datetime import datetime, timezone
from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app import graph
from app.config import get_settings
from app.db import cursor
from app.embeddings import embed
from app.llm import build_model
from app.prompts import CLASSIFIER, prompt

VALID_CATEGORIES = {"idea", "intention", "meeting", "task", "note"}


class Classification(BaseModel):
    category: str = Field(description="One of: idea, intention, meeting, task, note")
    importance: int = Field(ge=1, le=5, description="1 (trivial) to 5 (critical)")
    normalized_content: str = Field(description="The note content, lightly cleaned up")


@lru_cache
def classifier_agent() -> Agent:
    settings = get_settings()
    return Agent(
        build_model(settings.classifier_model),
        output_type=Classification,
        # Callable so data/prompts/classifier.md overrides are re-read each run.
        # NOTE: this prompt backs structured Classification output — an override must
        # still demand category (idea|intention|meeting|task|note) + importance 1-5.
        instructions=lambda: prompt("classifier", CLASSIFIER),
    )


async def classify_note(content: str) -> Classification:
    result = await classifier_agent().run(content)
    output = result.output
    if output.category not in VALID_CATEGORIES:
        output.category = "note"
    return output


async def capture_note(content: str, source: str = "user", space: str = "default") -> dict:
    classification = await classify_note(content)

    note_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    vector = embed(classification.normalized_content)

    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO notes (id, content, category, importance, source, space, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                note_id,
                classification.normalized_content,
                classification.category,
                classification.importance,
                source,
                space,
                now,
                now,
            ),
        )
        cur.execute(
            "INSERT INTO notes_fts (note_id, content) VALUES (?, ?)",
            (note_id, classification.normalized_content),
        )
        cur.execute(
            "INSERT INTO note_vectors (note_id, embedding) VALUES (?, ?)",
            (note_id, json.dumps(vector)),
        )
        graph.build_temporal_edges(cur.connection, note_id, source, now)
        graph.build_semantic_edges(cur.connection, note_id, vector)

    return {
        "id": note_id,
        "content": classification.normalized_content,
        "category": classification.category,
        "importance": classification.importance,
        "space": space,
        "created_at": now,
    }
