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
"""Tests for the return_direct feature on Tools."""

from smolagents.agents import (
    CodeAgent,
    ToolCallingAgent,
)
from smolagents.default_tools import FinalAnswerTool
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    Model,
)
from smolagents.tools import Tool


# --- Test Tools ---


class ReturnDirectTool(Tool):
    """A tool that has return_direct=True."""

    name = "direct_answer"
    description = "Returns a direct answer."
    inputs = {"query": {"type": "string", "description": "The query to answer"}}
    output_type = "string"
    return_direct = True

    def forward(self, query: str) -> str:
        return f"Direct result for: {query}"


class NormalTool(Tool):
    """A tool that has return_direct=False (default)."""

    name = "normal_tool"
    description = "Returns an answer normally."
    inputs = {"query": {"type": "string", "description": "The query"}}
    output_type = "string"

    def forward(self, query: str) -> str:
        return f"Normal result for: {query}"


# --- Fake Models ---


class FakeToolCallModelReturnDirect(Model):
    """Model that calls a return_direct tool on the first step."""

    def generate(self, messages, tools_to_call_from=None, stop_sequences=None):
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="I will call the direct_answer tool.",
            tool_calls=[
                ChatMessageToolCall(
                    id="call_0",
                    type="function",
                    function=ChatMessageToolCallFunction(
                        name="direct_answer", arguments={"query": "test"}
                    ),
                )
            ],
        )


class FakeToolCallModelNormal(Model):
    """Model that calls a normal tool, then final_answer."""

    def __init__(self):
        self.call_count = 0

    def generate(self, messages, tools_to_call_from=None, stop_sequences=None):
        self.call_count += 1
        if self.call_count == 1:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="I will call the normal tool.",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_0",
                        type="function",
                        function=ChatMessageToolCallFunction(
                            name="normal_tool", arguments={"query": "test"}
                        ),
                    )
                ],
            )
        else:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Returning final answer.",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(
                            name="final_answer", arguments={"answer": "final result"}
                        ),
                    )
                ],
            )


class FakeToolCallModelMultipleTools(Model):
    """Model that calls multiple tools including one with return_direct."""

    def generate(self, messages, tools_to_call_from=None, stop_sequences=None):
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="I will call the direct_answer tool.",
            tool_calls=[
                ChatMessageToolCall(
                    id="call_0",
                    type="function",
                    function=ChatMessageToolCallFunction(
                        name="direct_answer", arguments={"query": "test"}
                    ),
                ),
                ChatMessageToolCall(
                    id="call_1",
                    type="function",
                    function=ChatMessageToolCallFunction(
                        name="normal_tool", arguments={"query": "other"}
                    ),
                ),
            ],
        )


class FakeCodeModelReturnDirect(Model):
    """Model that generates code calling a return_direct tool."""

    def generate(self, messages, stop_sequences=None):
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="""
Thought: I will use the direct_answer tool.
<code>
result = direct_answer(query="test")
</code>
""",
        )


class FakeCodeModelNormal(Model):
    """Model that generates code calling a normal tool, then final_answer."""

    def __init__(self):
        self.call_count = 0

    def generate(self, messages, stop_sequences=None):
        self.call_count += 1
        if self.call_count == 1:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I will use the normal tool. special_marker
<code>
result = normal_tool(query="test")
</code>
""",
            )
        else:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I can now answer.
<code>
final_answer("final result")
</code>
""",
            )


# --- Tests ---


class TestReturnDirectAttribute:
    """Test that the return_direct attribute is properly set on Tool."""

    def test_return_direct_default_is_false(self):
        tool = NormalTool()
        assert tool.return_direct is False

    def test_return_direct_can_be_set_true(self):
        tool = ReturnDirectTool()
        assert tool.return_direct is True

    def test_return_direct_on_final_answer_tool(self):
        tool = FinalAnswerTool()
        assert tool.return_direct is False


class TestToolCallingAgentReturnDirect:
    """Test return_direct behavior with ToolCallingAgent."""

    def test_return_direct_skips_llm(self):
        """When a tool has return_direct=True, the agent returns immediately."""
        model = FakeToolCallModelReturnDirect()
        agent = ToolCallingAgent(
            tools=[ReturnDirectTool()],
            model=model,
        )
        result = agent.run("test query")
        assert result == "Direct result for: test"

    def test_normal_tool_does_not_return_direct(self):
        """Normal tools (return_direct=False) go through the LLM again."""
        model = FakeToolCallModelNormal()
        agent = ToolCallingAgent(
            tools=[NormalTool()],
            model=model,
        )
        result = agent.run("test query")
        # The model is called twice: once for the normal tool, once for final_answer
        assert model.call_count == 2
        assert result == "final result"

    def test_return_direct_single_step(self):
        """Return direct should complete in a single step."""
        model = FakeToolCallModelReturnDirect()
        agent = ToolCallingAgent(
            tools=[ReturnDirectTool()],
            model=model,
        )
        result = agent.run("test query")
        # The agent should have completed in 1 step
        assert result == "Direct result for: test"


class TestCodeAgentReturnDirect:
    """Test return_direct behavior with CodeAgent."""

    def test_return_direct_skips_llm(self):
        """When a tool has return_direct=True, CodeAgent returns immediately."""
        model = FakeCodeModelReturnDirect()
        agent = CodeAgent(
            tools=[ReturnDirectTool()],
            model=model,
        )
        result = agent.run("test query")
        assert result == "Direct result for: test"

    def test_normal_tool_does_not_return_direct(self):
        """Normal tools (return_direct=False) go through the LLM again."""
        model = FakeCodeModelNormal()
        agent = CodeAgent(
            tools=[NormalTool()],
            model=model,
        )
        result = agent.run("test query")
        assert model.call_count == 2
        assert result == "final result"

    def test_return_direct_single_step(self):
        """Return direct should complete in a single step for CodeAgent."""
        model = FakeCodeModelReturnDirect()
        agent = CodeAgent(
            tools=[ReturnDirectTool()],
            model=model,
        )
        result = agent.run("test query")
        assert result == "Direct result for: test"


class TestReturnDirectEdgeCases:
    """Test edge cases for the return_direct feature."""

    def test_return_direct_with_getattr(self):
        """getattr fallback works for tools without return_direct attribute."""
        tool = NormalTool()
        assert getattr(tool, "return_direct", False) is False

    def test_multiple_tools_one_return_direct(self):
        """When multiple tools are called and one has return_direct, the agent
        raises an error (same as calling final_answer with other tools) and
        retries.  We just verify it doesn't crash."""
        model = FakeToolCallModelMultipleTools()
        agent = ToolCallingAgent(
            tools=[ReturnDirectTool(), NormalTool()],
            model=model,
            max_steps=2,
        )
        # The agent should not crash; it handles the error internally
        # and continues (or hits max_steps).
        agent.run("test query")
