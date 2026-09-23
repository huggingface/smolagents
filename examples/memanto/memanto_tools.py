"""smolagents Tool wrappers for Memanto long-term memory."""

from __future__ import annotations

from smolagents import Tool

from memanto_client import MemantoClient, MemantoError


MEMORY_TYPES = (
    "fact",
    "preference",
    "goal",
    "decision",
    "artifact",
    "learning",
    "event",
    "instruction",
    "relationship",
    "context",
    "observation",
    "commitment",
    "error",
)


def _format_memories(memories: list[dict]) -> str:
    if not memories:
        return "No relevant memories found."
    lines = []
    for memory in memories:
        memory_type = memory.get("type", "fact")
        title = memory.get("title")
        content = memory.get("content", "")
        if title:
            lines.append(f"- [{memory_type}] {title}: {content}")
        else:
            lines.append(f"- [{memory_type}] {content}")
    return "Retrieved memories:\n" + "\n".join(lines)


class MemantoRecallTool(Tool):
    name = "recall_memory"
    description = (
        "Search long-term memory for user facts, preferences, decisions, and past context. "
        "Use this before asking the user to repeat stable information they may have shared before."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Natural-language search query, e.g. 'what does the user prefer for code style?'",
        }
    }
    output_type = "string"

    def __init__(self, client: MemantoClient, **kwargs):
        super().__init__(**kwargs)
        self.client = client

    def forward(self, query: str) -> str:
        try:
            memories = self.client.recall(query)
        except MemantoError as exc:
            return f"Memory recall failed: {exc}"
        return _format_memories(memories)


class MemantoRememberTool(Tool):
    name = "remember"
    description = (
        "Store a stable fact, preference, decision, goal, or instruction in long-term memory "
        "so it can be recalled in future conversations."
    )
    inputs = {
        "content": {
            "type": "string",
            "description": "One atomic statement to remember (max ~10000 chars).",
        },
        "memory_type": {
            "type": "string",
            "description": (
                "Semantic type: fact, preference, goal, decision, instruction, event, "
                "observation, commitment, error, or other Memanto memory types."
            ),
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, client: MemantoClient, **kwargs):
        super().__init__(**kwargs)
        self.client = client

    def forward(self, content: str, memory_type: str | None = "fact") -> str:
        memory_type = memory_type or "fact"
        try:
            memory_id = self.client.remember(content, memory_type=memory_type)
        except MemantoError as exc:
            return f"Memory storage failed: {exc}"
        return f"Stored memory {memory_id} ({memory_type})."


class MemantoAnswerTool(Tool):
    name = "answer_from_memory"
    description = (
        "Ask a question and get an answer synthesized from long-term memory (RAG). "
        "Prefer recall_memory when you only need raw memory hits."
    )
    inputs = {
        "question": {
            "type": "string",
            "description": "The question to answer using stored memories.",
        }
    }
    output_type = "string"

    def __init__(self, client: MemantoClient, **kwargs):
        super().__init__(**kwargs)
        self.client = client

    def forward(self, question: str) -> str:
        try:
            answer = self.client.answer(question)
        except MemantoError as exc:
            return f"Memory answer failed: {exc}"
        return answer or "No answer could be generated from memory."


def create_memanto_tools(
    client: MemantoClient | None = None,
    *,
    include_answer: bool = False,
) -> list[Tool]:
    """Build Memanto tools sharing one client instance."""
    client = client or MemantoClient()
    tools: list[Tool] = [MemantoRecallTool(client), MemantoRememberTool(client)]
    if include_answer:
        tools.append(MemantoAnswerTool(client))
    return tools
