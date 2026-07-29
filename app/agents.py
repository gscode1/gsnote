"""Pydantic AI agents.

Architecture (per pydantic-ai v2 guidance):
- A single `memory_agent` owns the store-vs-retrieve decision via function tools
  (save_note / search_notes / list_recent_notes). The model picks the right tool;
  there is no brittle pre-router.
- Dependencies (the current user id) are injected via `deps_type` + RunContext.
- The classifier and nudge agents remain separate single-purpose agents.
"""
from dataclasses import dataclass
from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models import Model

from app.config import get_settings

VALID_CATEGORIES = {"idea", "intention", "meeting", "task", "note"}

# Short-term conversation context window (in pydantic-ai messages, not user turns —
# one tool-using turn is ~4 messages). Long-term memory lives in the DB via tools.
HISTORY_KEEP_MESSAGES = 20


def trim_history(messages: list[ModelMessage], keep: int = HISTORY_KEEP_MESSAGES) -> list[ModelMessage]:
    """Keep only the most recent turns, trimming on a clean boundary.

    A naive `messages[-keep:]` can start the window in the middle of a tool
    call/return pair, leaving an orphaned ToolReturnPart that some providers reject.
    We slice to the last `keep`, then advance to the first message that begins a
    fresh user turn (a ModelRequest carrying a UserPromptPart), so history always
    re-enters cleanly.
    """
    if len(messages) <= keep:
        return messages
    # Forward-search from the start of the kept window for the first clean user-turn
    # boundary. If the window contains none (only a partial tool sequence), return no
    # history — empty is always valid, malformed (orphan tool-return) is not.
    for i in range(len(messages) - keep, len(messages)):
        m = messages[i]
        if isinstance(m, ModelRequest) and any(isinstance(p, UserPromptPart) for p in m.parts):
            return messages[i:]
    return []


class Classification(BaseModel):
    category: str = Field(description="One of: idea, intention, meeting, task, note")
    importance: int = Field(ge=1, le=5, description="1 (trivial) to 5 (critical)")
    normalized_content: str = Field(description="The note content, lightly cleaned up")


@dataclass
class NoteDeps:
    """Per-request dependencies injected into the memory agent's tools."""

    user_id: str
    space: str = "personal"  # active work/personal space for this user's turn


def _model(model_name: str) -> Model:
    """Build a Model for the configured provider.

    OpenRouter has a dedicated provider/profile (handles model-prefix routing,
    reasoning fields, schema transforms). Anthropic (and Anthropic-compatible
    gateways) use AnthropicModel + AnthropicProvider. Other OpenAI-compatible
    providers fall back to OpenAIChatModel + OpenAIProvider(base_url=...).
    """
    settings = get_settings()
    if settings.llm_provider == "openrouter":
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        return OpenRouterModel(model_name, provider=OpenRouterProvider(api_key=settings.llm_api_key))

    if settings.llm_provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        # base_url is optional — pass it only when set, so an Anthropic-compatible
        # gateway can be targeted while the default (api.anthropic.com) still works.
        provider_kwargs: dict = {"api_key": settings.llm_api_key}
        if settings.llm_base_url:
            provider_kwargs["base_url"] = settings.llm_base_url
        return AnthropicModel(model_name, provider=AnthropicProvider(**provider_kwargs))

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=settings.llm_base_url, api_key=settings.llm_api_key),
    )


@lru_cache
def classifier_agent() -> Agent:
    settings = get_settings()
    return Agent(
        _model(settings.classifier_model),
        output_type=Classification,
        instructions=(
            "You classify short personal notes. Assign a category from "
            "{idea, intention, meeting, task, note} and an importance 1-5. "
            "Lightly clean up the content (fix obvious typos) but preserve meaning and language. "
            "Do not invent information."
        ),
    )


@lru_cache
def memory_agent() -> Agent[NoteDeps, str]:
    """Single agent with tools; the model decides whether to store or retrieve.

    Instantiated once and reused; safe to run concurrently with different `deps`.
    """
    settings = get_settings()
    agent: Agent[NoteDeps, str] = Agent(
        _model(settings.answer_model),
        deps_type=NoteDeps,
        output_type=str,
        # Trim each model request to recent turns on a clean boundary, so the context
        # window stays bounded regardless of how long the conversation runs.
        capabilities=[ProcessHistory(lambda msgs: trim_history(msgs))],
        instructions=(
            "You are the user's personal memory assistant. You have tools to store and query "
            "the user's notes. Decide from each message what to do:\n"
            "- If the user is recording something new (a thought, idea, task, plan, reminder), "
            "call save_note.\n"
            "- If the user asks a question, or asks you to check / look up / list / print / recall "
            "what they saved, call search_notes (topical) or list_recent_notes (time-window or "
            "'everything I saved').\n"
            "ALWAYS use your tools — never claim you lack access to memory or tools. You may call "
            "tools more than once. Answer concisely based on what the tools return; never fabricate notes.\n"
            "After saving a note, confirm back to the user the exact text that was stored (the tool "
            "reports it) and its category, so they can verify what was recorded."
        ),
    )

    @agent.tool
    async def save_note(ctx: RunContext[NoteDeps], content: str) -> str:
        """Store a new note for the user. Use when the user gives you something to remember.

        Args:
            content: The note text to save (the user's new information).
        """
        from app.capture import capture_note

        if not content.strip():
            raise ModelRetry("Cannot save an empty note. Ask the user what to record.")
        note = await capture_note(content, source=ctx.deps.user_id, space=ctx.deps.space)
        # Return the stored (normalized) text so the model can confirm exactly what
        # went into memory — important since we lightly normalize, and for voice input.
        return (
            f"Saved as {note['category']} (importance {note['importance']}). "
            f"Stored text: \"{note['content']}\""
        )

    @agent.tool
    async def search_notes(ctx: RunContext[NoteDeps], query: str, limit: int = 8) -> str:
        """Search the user's stored notes by topic/meaning (hybrid semantic + keyword).

        Args:
            query: What to look for, in natural language.
            limit: Max number of notes to return.
        """
        from app.retrieval import search

        if not query.strip():
            raise ModelRetry("Search query is empty. Provide a topic or keyword to search for.")
        notes = search(query, top_k=limit, space=ctx.deps.space)
        if not notes:
            return "No matching notes found."
        return "\n".join(f"- [{n['category']}] {n['content']} ({n['created_at']})" for n in notes)

    @agent.tool
    async def list_recent_notes(ctx: RunContext[NoteDeps], days: int = 7, category: str | None = None) -> str:
        """List all notes saved in the last N days. Use for 'everything saved' or time-window requests.

        Args:
            days: How many days back to include.
            category: Optional filter — one of idea, intention, meeting, task, note.
        """
        from app.reporting import notes_in_window

        notes = notes_in_window(days, category, space=ctx.deps.space)
        if not notes:
            return f"No notes found in the last {days} day(s)."
        return "\n".join(f"- [{n['category']}] {n['content']} ({n['created_at']})" for n in notes)

    return agent


@lru_cache
def answer_agent() -> Agent:
    settings = get_settings()
    return Agent(
        _model(settings.answer_model),
        output_type=str,
        instructions=(
            "You answer questions about the user's personal notes using only the provided "
            "context notes. Be concise. If the notes don't contain the answer, say so plainly. "
            "Never fabricate notes that weren't given to you."
        ),
    )


@lru_cache
def nudge_agent() -> Agent:
    settings = get_settings()
    return Agent(
        _model(settings.answer_model),
        output_type=str,
        instructions=(
            "You phrase a short, friendly proactive nudge message reminding the user about "
            "notes they captured and haven't revisited. One or two sentences, warm but brief. "
            "Reference the actual content given, don't be generic."
        ),
    )


async def classify_note(content: str) -> Classification:
    result = await classifier_agent().run(content)
    output = result.output
    if output.category not in VALID_CATEGORIES:
        output.category = "note"
    return output


async def phrase_nudge(notes: list[dict]) -> str:
    context = "\n".join(f"- {n['content']}" for n in notes)
    result = await nudge_agent().run(f"Notes to nudge about:\n{context}")
    return result.output
