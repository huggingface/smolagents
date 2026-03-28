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

import pytest

from smolagents.agents import CodeAgent, ToolCallingAgent
from smolagents.guardrails import (
    AllowlistGuardrail,
    BlocklistGuardrail,
    CompositeGuardrail,
    GuardrailDecision,
    GuardrailProvider,
)
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole, Model
from smolagents.tools import tool
from smolagents.utils import AgentToolExecutionError


# --- Fake models ---


class FakeToolCallModelWithTool(Model):
    """Calls a specific tool on step 1, then final_answer on step 2."""

    def __init__(self, tool_name, tool_args):
        self._tool_name = tool_name
        self._tool_args = tool_args

    def generate(self, messages, tools_to_call_from=None, stop_sequences=None):
        if len(messages) < 3:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="I will call the tool.",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_0",
                        type="function",
                        function=ChatMessageToolCallFunction(name=self._tool_name, arguments=self._tool_args),
                    )
                ],
            )
        else:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Final answer.",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="final_answer", arguments={"answer": "done"}),
                    )
                ],
            )


class FakeCodeModelWithTool(Model):
    """Generates code that calls a tool on step 1, then final_answer on step 2."""

    def __init__(self, tool_call_code):
        self._tool_call_code = tool_call_code

    def generate(self, messages, stop_sequences=None):
        prompt = str(messages)
        if "special_marker" not in prompt:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=f"""
Thought: I will call the tool. special_marker
<code>
{self._tool_call_code}
</code>
""",
            )
        else:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: Return the result.
<code>
final_answer("done")
</code>
""",
            )


# --- Test tools ---


@tool
def sample_tool(query: str) -> str:
    """A sample tool that echoes back.

    Args:
        query: The input query.
    """
    return f"echo: {query}"


@tool
def another_tool(value: str) -> str:
    """Another tool.

    Args:
        value: A value.
    """
    return f"another: {value}"


# --- GuardrailDecision tests ---


class TestGuardrailDecision:
    def test_allowed_decision(self):
        d = GuardrailDecision(allowed=True)
        assert d.allowed is True
        assert d.reason == ""

    def test_denied_decision(self):
        d = GuardrailDecision(allowed=False, reason="not authorized")
        assert d.allowed is False
        assert d.reason == "not authorized"


# --- AllowlistGuardrail tests ---


class TestAllowlistGuardrail:
    def test_allowed_tool(self):
        g = AllowlistGuardrail(["sample_tool"])
        decision = g.before_tool_call("sample_tool", {"query": "test"})
        assert decision.allowed is True

    def test_denied_tool(self):
        g = AllowlistGuardrail(["sample_tool"])
        decision = g.before_tool_call("another_tool", {"value": "test"})
        assert decision.allowed is False
        assert "another_tool" in decision.reason

    def test_final_answer_always_allowed(self):
        g = AllowlistGuardrail(["sample_tool"])
        decision = g.before_tool_call("final_answer", {"answer": "42"})
        assert decision.allowed is True

    def test_empty_allowlist_still_allows_final_answer(self):
        g = AllowlistGuardrail([])
        decision = g.before_tool_call("final_answer", {"answer": "42"})
        assert decision.allowed is True


# --- BlocklistGuardrail tests ---


class TestBlocklistGuardrail:
    def test_allowed_tool(self):
        g = BlocklistGuardrail(["dangerous_tool"])
        decision = g.before_tool_call("sample_tool", {"query": "test"})
        assert decision.allowed is True

    def test_blocked_tool(self):
        g = BlocklistGuardrail(["sample_tool"])
        decision = g.before_tool_call("sample_tool", {"query": "test"})
        assert decision.allowed is False


# --- CompositeGuardrail tests ---


class TestCompositeGuardrail:
    def test_all_allow(self):
        g = CompositeGuardrail(
            [
                AllowlistGuardrail(["sample_tool", "another_tool"]),
                BlocklistGuardrail(["dangerous_tool"]),
            ]
        )
        decision = g.before_tool_call("sample_tool", {"query": "test"})
        assert decision.allowed is True

    def test_first_denies(self):
        g = CompositeGuardrail(
            [
                AllowlistGuardrail(["sample_tool"]),
                BlocklistGuardrail([]),
            ]
        )
        decision = g.before_tool_call("another_tool", {"value": "test"})
        assert decision.allowed is False

    def test_second_denies(self):
        g = CompositeGuardrail(
            [
                AllowlistGuardrail(["sample_tool"]),
                BlocklistGuardrail(["sample_tool"]),
            ]
        )
        decision = g.before_tool_call("sample_tool", {"query": "test"})
        assert decision.allowed is False


# --- Custom GuardrailProvider via protocol ---


class TestCustomGuardrailProvider:
    def test_custom_provider(self):
        class ArgCheckGuardrail:
            def before_tool_call(self, tool_name, arguments):
                if isinstance(arguments, dict) and "forbidden" in str(arguments.values()):
                    return GuardrailDecision(allowed=False, reason="forbidden argument detected")
                return GuardrailDecision(allowed=True)

        g = ArgCheckGuardrail()
        assert isinstance(g, GuardrailProvider)
        assert g.before_tool_call("sample_tool", {"query": "hello"}).allowed is True
        assert g.before_tool_call("sample_tool", {"query": "forbidden"}).allowed is False


# --- ToolCallingAgent integration tests ---


class TestToolCallingAgentGuardrail:
    def test_execute_tool_call_allowed(self):
        guardrail = AllowlistGuardrail(["sample_tool"])
        agent = ToolCallingAgent(
            model=FakeToolCallModelWithTool("sample_tool", {"query": "hi"}),
            tools=[sample_tool],
            guardrail=guardrail,
        )
        result = agent.execute_tool_call("sample_tool", {"query": "hi"})
        assert result == "echo: hi"

    def test_execute_tool_call_denied(self):
        guardrail = AllowlistGuardrail(["another_tool"])
        agent = ToolCallingAgent(
            model=FakeToolCallModelWithTool("sample_tool", {"query": "hi"}),
            tools=[sample_tool, another_tool],
            guardrail=guardrail,
        )
        with pytest.raises(AgentToolExecutionError, match="denied by guardrail"):
            agent.execute_tool_call("sample_tool", {"query": "hi"})

    def test_full_run_with_allowed_tool(self):
        guardrail = AllowlistGuardrail(["sample_tool"])
        agent = ToolCallingAgent(
            model=FakeToolCallModelWithTool("sample_tool", {"query": "hi"}),
            tools=[sample_tool],
            guardrail=guardrail,
        )
        result = agent.run("test task")
        assert result == "done"

    def test_no_guardrail_still_works(self):
        agent = ToolCallingAgent(
            model=FakeToolCallModelWithTool("sample_tool", {"query": "hi"}),
            tools=[sample_tool],
        )
        result = agent.execute_tool_call("sample_tool", {"query": "hi"})
        assert result == "echo: hi"


# --- CodeAgent integration tests ---


class TestCodeAgentGuardrail:
    def test_code_agent_allowed_tool(self):
        guardrail = AllowlistGuardrail(["sample_tool"])
        agent = CodeAgent(
            model=FakeCodeModelWithTool('result = sample_tool(query="hello")'),
            tools=[sample_tool],
            guardrail=guardrail,
        )
        result = agent.run("test task")
        assert result == "done"

    def test_code_agent_denied_tool(self):
        guardrail = AllowlistGuardrail(["another_tool"])
        agent = CodeAgent(
            model=FakeCodeModelWithTool('result = sample_tool(query="hello")'),
            tools=[sample_tool, another_tool],
            guardrail=guardrail,
        )
        # The agent should still complete (the denied tool surfaces as an error observation,
        # and the agent adapts by calling final_answer on the next step)
        result = agent.run("test task")
        assert result == "done"

    def test_code_agent_no_guardrail(self):
        agent = CodeAgent(
            model=FakeCodeModelWithTool('result = sample_tool(query="hello")'),
            tools=[sample_tool],
        )
        result = agent.run("test task")
        assert result == "done"


# --- Shared tool instance tests ---


class TestGuardrailSharedToolInstances:
    def test_shared_tool_not_mutated_by_guardrail(self):
        """Ensure that wrapping tools for the code executor does not mutate the original tool."""
        shared_tool = sample_tool

        guardrail = AllowlistGuardrail(["sample_tool"])
        agent = CodeAgent(
            model=FakeCodeModelWithTool('result = sample_tool(query="hello")'),
            tools=[shared_tool],
            guardrail=guardrail,
        )
        agent.run("test task")

        # Original tool should still work without guardrail interference
        assert shared_tool(query="direct") == "echo: direct"
        # The tool class should not have been replaced or wrapped
        assert type(shared_tool).__name__ == "SimpleTool"

    def test_different_guardrails_on_shared_tool(self):
        """Two agents with different guardrails sharing the same tool instance should not interfere."""
        shared_tool = sample_tool

        agent_allow = ToolCallingAgent(
            model=FakeToolCallModelWithTool("sample_tool", {"query": "hi"}),
            tools=[shared_tool],
            guardrail=AllowlistGuardrail(["sample_tool"]),
        )
        agent_deny = ToolCallingAgent(
            model=FakeToolCallModelWithTool("sample_tool", {"query": "hi"}),
            tools=[shared_tool],
            guardrail=AllowlistGuardrail(["other_tool"]),
        )

        # First agent should allow
        result = agent_allow.execute_tool_call("sample_tool", {"query": "hi"})
        assert result == "echo: hi"

        # Second agent should deny
        with pytest.raises(AgentToolExecutionError, match="denied by guardrail"):
            agent_deny.execute_tool_call("sample_tool", {"query": "hi"})

        # First agent should still allow (not affected by second)
        result = agent_allow.execute_tool_call("sample_tool", {"query": "hi"})
        assert result == "echo: hi"


# --- Managed agent guardrail tests ---


class TestGuardrailWithManagedAgents:
    def test_guardrail_blocks_managed_agent(self):
        """Guardrail should be able to block calls to managed agents too."""
        managed = CodeAgent(
            model=FakeCodeModelWithTool('final_answer("sub-result")'),
            tools=[],
            name="sub_agent",
            description="A sub agent.",
        )
        guardrail = AllowlistGuardrail(["sample_tool"])  # sub_agent not in allowlist

        agent = ToolCallingAgent(
            model=FakeToolCallModelWithTool("sub_agent", {"task": "do something"}),
            tools=[sample_tool],
            managed_agents=[managed],
            guardrail=guardrail,
        )
        with pytest.raises(AgentToolExecutionError, match="denied by guardrail"):
            agent.execute_tool_call("sub_agent", {"task": "do something"})
