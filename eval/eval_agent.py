"""Agent evals with pydantic_evals (PRD §6, M2).

Two datasets, both run against the REAL configured model (needs LLM_API_KEY):

1. Routing accuracy — does the single memory agent pick the right tool
   (save_note vs search_notes vs list_recent_notes) for each kind of message?
   Deterministic evaluator inspects the actual tool calls via result.all_messages().

2. Retrieval quality — are the agent's answers grounded in the seeded notes?
   Graded by an LLMJudge rubric.

Run with: python -m eval.eval_agent
"""
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
os.environ.setdefault("EMBEDDING_DIM", "384")
os.environ.setdefault("DB_PATH", "./data/eval_agent.db")

from pydantic_ai.messages import ModelResponse, ToolCallPart  # noqa: E402
from pydantic_evals import Case, Dataset  # noqa: E402
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge  # noqa: E402
from pydantic_evals.evaluators.llm_as_a_judge import set_default_judge_model  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.llm import build_model  # noqa: E402
from app.turn import NoteDeps, memory_agent  # noqa: E402
from eval.run_eval import EVAL_SET_PATH, seed  # noqa: E402


# --- task functions (what each Case runs) --------------------------------------------------------

async def route_task(message: str) -> dict:
    """Run the agent on a single message and report which tool it called first."""
    result = await memory_agent().run(message, deps=NoteDeps(user_id="eval"))
    tool_calls = [
        part.tool_name
        for msg in result.all_messages()
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    ]
    return {"tool_called": tool_calls[0] if tool_calls else None, "output": result.output}


async def answer_task(query: str) -> str:
    """Run the agent on a retrieval query and return its final answer."""
    result = await memory_agent().run(query, deps=NoteDeps(user_id="eval"))
    return result.output


# --- evaluators ----------------------------------------------------------------------------------

@dataclass
class CorrectToolCalled(Evaluator):
    expected_tool: str

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.output.get("tool_called") == self.expected_tool


# --- datasets ------------------------------------------------------------------------------------

def routing_dataset() -> Dataset:
    return Dataset(
        name="routing_accuracy",
        cases=[
            Case(name="store_task", inputs="Remember to buy milk on the way home",
                 evaluators=[CorrectToolCalled(expected_tool="save_note")]),
            Case(name="store_idea", inputs="Idea: build a parametric CAD enclosure for the cluster",
                 evaluators=[CorrectToolCalled(expected_tool="save_note")]),
            Case(name="ask_topical", inputs="What did I note about my home lab?",
                 evaluators=[CorrectToolCalled(expected_tool="search_notes")]),
            Case(name="ask_list_all", inputs="Check all saved in memory and print",
                 evaluators=[CorrectToolCalled(expected_tool="list_recent_notes")]),
            Case(name="ask_question_mark", inputs="do I have any tasks?",
                 evaluators=[CorrectToolCalled(expected_tool="search_notes")]),
        ],
    )


def retrieval_dataset() -> Dataset:
    return Dataset(
        name="retrieval_quality",
        cases=[
            Case(
                name="errands",
                inputs="what errands do I still need to run?",
                evaluators=[LLMJudge(
                    rubric="The answer references errand/task notes that were actually retrieved and "
                           "does not invent notes. If relevant notes exist it should mention them.",
                    include_input=True,
                )],
            ),
            Case(
                name="ideas",
                inputs="what ideas have I been thinking about?",
                evaluators=[LLMJudge(
                    rubric="The answer summarizes idea-category notes grounded in the user's stored notes, "
                           "without fabricating content.",
                    include_input=True,
                )],
            ),
        ],
    )


async def main() -> None:
    if not get_settings().llm_api_key or get_settings().llm_api_key == "test-key":
        print("ERROR: set LLM_API_KEY to a real key to run agent evals.")
        sys.exit(1)

    # Judge with the configured provider (we run on OpenRouter, not the OpenAI default).
    set_default_judge_model(build_model(get_settings().answer_model))

    eval_set = json.loads(EVAL_SET_PATH.read_text())

    # Routing cases include store_* which persist notes via save_note; seed first, and
    # re-seed before retrieval so the two datasets don't contaminate each other's DB state.
    seed(eval_set)
    print("\n=== Routing accuracy ===")
    routing_report = await routing_dataset().evaluate(route_task)
    routing_report.print(include_input=True, include_output=True)

    seed(eval_set)
    print("\n=== Retrieval quality ===")
    retrieval_report = await retrieval_dataset().evaluate(answer_task)
    retrieval_report.print(include_input=True, include_output=True)


if __name__ == "__main__":
    asyncio.run(main())
