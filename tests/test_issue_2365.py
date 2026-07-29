from unittest.mock import MagicMock

from smolagents.agents import ToolCallingAgent
from smolagents.memory import ActionStep
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole
from smolagents.monitoring import Timing
from smolagents.tools import tool


def test_issue_2365():
    @tool
    def echo(value: str) -> str:
        """Return the supplied value.

        Args:
            value: The value to return.
        """
        return value

    agent = ToolCallingAgent(model=MagicMock(), tools=[echo])
    memory_step = ActionStep(step_number=1, timing=Timing(start_time=0.0, end_time=1.0))
    message = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="",
        tool_calls=[
            ChatMessageToolCall(
                id="call_1",
                type="function",
                function=ChatMessageToolCallFunction(name="echo", arguments={"value": "first result"}),
            ),
            ChatMessageToolCall(
                id="call_2",
                type="function",
                function=ChatMessageToolCallFunction(name="echo", arguments={"value": "second result"}),
            ),
        ],
    )

    list(agent.process_tool_calls(message, memory_step))

    assert memory_step.observations == ["first result", "second result"]
