# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock

import pytest

from smolagents.agents import AgentMaxStepsError, ToolCallingAgent
from smolagents.memory import ActionStep, ToolCall
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole, Model
from smolagents.tools import Tool


class ReturnDirectTool(Tool):
    name = "return_direct_tool"
    description = "A tool that returns its output directly as the final answer."
    inputs = {"value": {"type": "string", "description": "The value to return."}}
    output_type = "string"
    return_direct = True

    def forward(self, value: str) -> str:
        return value


class NormalTool(Tool):
    name = "normal_tool"
    description = "A normal tool."
    inputs = {"value": {"type": "string", "description": "A value."}}
    output_type = "string"

    def forward(self, value: str) -> str:
        return value


def _make_model_response(tool_name: str, arguments: dict, call_id: str = "call_1") -> ChatMessage:
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        content="",
        tool_calls=[
            ChatMessageToolCall(
                id=call_id,
                type="function",
                function=ChatMessageToolCallFunction(name=tool_name, arguments=arguments),
            )
        ],
    )


class TestReturnDirectToolCallingAgent:
    def test_return_direct_tool_output_is_final(self):
        """When a return_direct tool is called, its output is the agent's final answer."""
        model = MagicMock(spec=Model)
        model.model_id = "fake"
        model.generate.return_value = _make_model_response("return_direct_tool", {"value": "direct result"})

        agent = ToolCallingAgent(tools=[ReturnDirectTool()], model=model, max_steps=3)
        result = agent.run("Return something directly")

        assert result == "direct result"

    def test_normal_tool_not_final(self):
        """A tool without return_direct does not produce a final answer."""
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return _make_model_response("normal_tool", {"value": "intermediate"})
            return _make_model_response("final_answer", {"answer": "done"})

        model = MagicMock(spec=Model)
        model.model_id = "fake"
        model.generate.side_effect = side_effect

        agent = ToolCallingAgent(tools=[NormalTool()], model=model, max_steps=5)
        result = agent.run("Do something normal")

        assert result == "done"
        assert call_count > 1

    def test_multiple_tools_one_return_direct(self):
        """When mixing return_direct and normal tools, only return_direct triggers final answer."""
        model = MagicMock(spec=Model)
        model.model_id = "fake"
        model.generate.return_value = _make_model_response("return_direct_tool", {"value": "early exit"})

        agent = ToolCallingAgent(tools=[ReturnDirectTool(), NormalTool()], model=model, max_steps=3)
        result = agent.run("Use tools")

        assert result == "early exit"

    def test_return_direct_attribute_default(self):
        """Tools have return_direct=False by default."""
        tool = NormalTool()
        assert tool.return_direct is False

    def test_return_direct_attribute_missing(self):
        """getattr fallback works for objects without return_direct attribute."""

        class BareObject:
            pass

        obj = BareObject()
        assert getattr(obj, "return_direct", False) is False

    def test_return_direct_validation_rejects_non_bool(self):
        """return_direct must be a bool; non-bool values are rejected at validation time."""
        with pytest.raises(TypeError, match="return_direct should have type bool"):

            class BadTool(Tool):
                name = "bad_tool"
                description = "Bad tool."
                inputs = {"x": {"type": "string", "description": "x"}}
                output_type = "string"
                return_direct = "yes"

                def forward(self, x: str) -> str:
                    return x

            BadTool()
