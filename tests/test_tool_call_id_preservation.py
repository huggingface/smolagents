from unittest.mock import MagicMock

import pytest

from smolagents import ToolCallingAgent
from smolagents.agents import ToolOutput
from smolagents.memory import ActionStep, ToolCall
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction
from smolagents.tools import Tool


class EchoTool(Tool):
    name = "echo"
    description = "Return the provided integer unchanged."
    inputs = {"value": {"type": "integer", "description": "Integer to return."}}
    output_type = "integer"

    def forward(self, value: int) -> int:
        return value


def _agent():
    return ToolCallingAgent(tools=[EchoTool()], model=MagicMock())


def _message(ids):
    return ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            ChatMessageToolCall(
                id=tool_id,
                type="function",
                function=ChatMessageToolCallFunction(name="echo", arguments={"value": i}),
            )
            for i, tool_id in enumerate(ids)
        ],
    )


@pytest.mark.parametrize(
    "ids",
    [
        ["first", "second", "third"],
        ["same-id", "same-id", "same-id"],
        [None, None, None],
        ["one-id"],
    ],
    ids=["distinct-ids", "duplicate-ids", "missing-ids", "single-call"],
)
def test_tool_calls_preserve_order_and_do_not_drop_duplicate_ids(ids):
    agent = _agent()
    step = ActionStep(step_number=0, timing=MagicMock())
    events = list(agent.process_tool_calls(_message(ids), step))

    calls = [event for event in events if isinstance(event, ToolCall)]
    results = [event for event in events if isinstance(event, ToolOutput)]

    assert len(calls) == len(ids)
    assert len(results) == len(ids)
    assert [call.id for call in calls] == ids
    assert [call.arguments["value"] for call in calls] == list(range(len(ids)))

    # Tool outputs are yielded in completion order, not necessarily call order.
    # Verify every result, rather than imposing a scheduler-dependent ordering.
    assert sorted(result.output for result in results) == list(range(len(ids)))
    for result in results:
        assert result.output == result.tool_call.arguments["value"]

    # Memory must retain the original provider IDs and deterministic call order.
    assert [call.id for call in step.tool_calls] == ids
    assert [call.arguments["value"] for call in step.tool_calls] == list(range(len(ids)))
    assert step.observations == "\n".join(str(i) for i in range(len(ids)))
