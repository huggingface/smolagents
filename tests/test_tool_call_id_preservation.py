from unittest.mock import MagicMock

from smolagents import ToolCallingAgent
from smolagents.memory import ActionStep, ToolCall
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction


def _agent():
    model = MagicMock()
    return ToolCallingAgent(tools=[], model=model)


def _message(ids):
    return ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            ChatMessageToolCall(
                id=tool_id,
                function=ChatMessageToolCallFunction(name="echo", arguments={"value": i}),
            )
            for i, tool_id in enumerate(ids)
        ],
    )


def test_duplicate_tool_call_ids_are_not_dropped():
    agent = _agent()
    agent.tools["echo"] = MagicMock(side_effect=lambda value, sanitize_inputs_outputs=True: value)

    message = _message(["same-id", "same-id"])
    step = ActionStep(step_number=0, timing=MagicMock())
    outputs = list(agent.process_tool_calls(message, step))

    calls = [item for item in outputs if isinstance(item, ToolCall)]
    results = [item for item in outputs if hasattr(item, "output")]

    assert len(calls) == 2
    assert [call.id for call in calls] == ["same-id", "same-id"]
    assert len(results) == 2
    assert [result.output for result in results] == [0, 1]
    assert len(step.tool_calls) == 2


def test_missing_tool_call_ids_are_not_dropped():
    agent = _agent()
    agent.tools["echo"] = MagicMock(side_effect=lambda value, sanitize_inputs_outputs=True: value)

    message = _message([None, None])
    step = ActionStep(step_number=0, timing=MagicMock())
    outputs = list(agent.process_tool_calls(message, step))

    calls = [item for item in outputs if isinstance(item, ToolCall)]
    results = [item for item in outputs if hasattr(item, "output")]

    assert len(calls) == 2
    assert len(results) == 2
    assert [result.output for result in results] == [0, 1]
