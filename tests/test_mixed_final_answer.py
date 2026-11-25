"""A mixed final_answer call must be rejected before other tools execute."""

from smolagents import ToolCallingAgent, tool
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    Model,
)

writes = []


@tool
def save_record(record: str) -> str:
    """Save a record.

    Args:
        record: Record to save.
    """
    writes.append(record)
    return "saved"


def _call(name, arguments):
    return ChatMessageToolCall(ChatMessageToolCallFunction(arguments=arguments, name=name), name, "function")


class StubModel(Model):
    def generate(self, messages, **kwargs):
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            tool_calls=[
                _call("final_answer", {"answer": "done"}),
                _call("save_record", {"record": "A"}),
            ],
        )


def test_other_tool_does_not_run_when_mixed_with_final_answer():
    writes.clear()
    agent = ToolCallingAgent(model=StubModel(), tools=[save_record], verbosity_level=-1)

    agent.run("Save record A")

    error = agent.memory.steps[1].error
    assert error is not None
    assert "do not perform any other tool calls" in str(error)
    assert writes == []
