from unittest.mock import MagicMock

from smolagents.agents import ToolCallingAgent
from smolagents.memory import ActionStep
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole


def test_issue_2365(test_tool):
    agent = ToolCallingAgent(tools=[test_tool], model=MagicMock())
    chat_message = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="",
        tool_calls=[
            ChatMessageToolCall(
                id="call_1",
                type="function",
                function=ChatMessageToolCallFunction(name="test_tool", arguments={"input": "value1"}),
            ),
            ChatMessageToolCall(
                id="call_2",
                type="function",
                function=ChatMessageToolCallFunction(name="test_tool", arguments={"input": "value2"}),
            ),
        ],
    )
    memory_step = ActionStep(step_number=1, timing="mock_timing", model_output="")

    list(agent.process_tool_calls(chat_message, memory_step))

    assert memory_step.tool_calls[0].id == "call_1"
    assert memory_step.tool_calls[1].id == "call_2"
    assert memory_step.observations == ["Processed: value1", "Processed: value2"]
