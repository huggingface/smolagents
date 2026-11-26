"""Falsy scalar tool arguments must reach single-input tools unchanged."""

import json
import time

from smolagents import Tool, ToolCallingAgent
from smolagents.memory import ActionStep
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    Model,
)
from smolagents.monitoring import Timing

received = []


class Echo(Tool):
    name = "echo"
    description = "Record a single value."
    inputs = {"value": {"type": "any", "description": "Value to echo"}}
    output_type = "any"

    def forward(self, value):
        received.append((value, type(value)))
        return value


class StubModel(Model):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.emitted = False

    def generate(self, messages, **kwargs):
        if self.emitted:
            return ChatMessage(role=MessageRole.ASSISTANT, content="Final answer: done")
        self.emitted = True
        call = ChatMessageToolCall(
            function=ChatMessageToolCallFunction(name="echo", arguments=json.dumps(self.value)),
            id="call_0",
            type="function",
        )
        return ChatMessage(role=MessageRole.ASSISTANT, tool_calls=[call])


def test_falsy_scalar_reaches_tool():
    for value in (False, 0, ""):
        received.clear()
        agent = ToolCallingAgent(tools=[Echo()], model=StubModel(value), verbosity_level=-1)
        list(
            agent._step_stream(
                ActionStep(step_number=1, timing=Timing(start_time=time.time()))
            )
        )
        assert received == [(value, type(value))], (value, received)
