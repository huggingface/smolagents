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
"""Tests for async functionality in smolagents."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smolagents.agents import CodeAgent, ToolCallingAgent
from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage


class MockAsyncModel(Model):
    """Mock model for testing async functionality."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generate_called = False
        self.agenerate_called = False
        self.generate_stream_called = False
        self.agenerate_stream_called = False

    def generate(self, messages, **kwargs):
        """Sync generate for comparison."""
        self.generate_called = True
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=[{"type": "text", "text": "Sync response"}],
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    async def agenerate(self, messages, **kwargs):
        """Async generate implementation."""
        self.agenerate_called = True
        # Simulate async I/O
        await asyncio.sleep(0.01)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=[{"type": "text", "text": "Async response"}],
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    async def agenerate_stream(self, messages, **kwargs):
        """Async generate stream implementation."""
        self.agenerate_stream_called = True
        # Simulate streaming
        for i in range(3):
            await asyncio.sleep(0.01)
            from smolagents.models import ChatMessageStreamDelta

            yield ChatMessageStreamDelta(
                content=f"chunk_{i}",
                token_usage=TokenUsage(input_tokens=10 if i == 0 else 0, output_tokens=1),
            )


class TestAsyncModels:
    """Test async model functionality."""

    @pytest.mark.asyncio
    async def test_model_has_agenerate_method(self):
        """Test that Model base class has agenerate method."""
        model = Model()
        assert hasattr(model, "agenerate")
        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            await model.agenerate([])

    @pytest.mark.asyncio
    async def test_async_model_agenerate(self):
        """Test async model generate."""
        model = MockAsyncModel()
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}])]

        response = await model.agenerate(messages)

        assert model.agenerate_called
        assert not model.generate_called
        assert response.role == MessageRole.ASSISTANT
        assert response.content == [{"type": "text", "text": "Async response"}]
        assert response.token_usage.input_tokens == 10
        assert response.token_usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_async_model_agenerate_stream(self):
        """Test async model streaming."""
        model = MockAsyncModel()
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}])]

        chunks = []
        async for chunk in model.agenerate_stream(messages):
            chunks.append(chunk)

        assert model.agenerate_stream_called
        assert len(chunks) == 3
        assert chunks[0].content == "chunk_0"
        assert chunks[1].content == "chunk_1"
        assert chunks[2].content == "chunk_2"
        # First chunk should have input tokens
        assert chunks[0].token_usage.input_tokens == 10
        assert chunks[0].token_usage.output_tokens == 1

    @pytest.mark.asyncio
    async def test_concurrent_model_calls(self):
        """Test concurrent async model calls."""
        model = MockAsyncModel()
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}])]

        # Run 3 concurrent calls
        tasks = [model.agenerate(messages) for _ in range(3)]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 3
        for response in responses:
            assert response.role == MessageRole.ASSISTANT
            assert response.content == [{"type": "text", "text": "Async response"}]

    @pytest.mark.asyncio
    async def test_model_concurrent_performance(self):
        """Test that concurrent calls are actually concurrent."""
        import time

        model = MockAsyncModel()
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}])]

        # Time 3 sequential calls
        start = time.time()
        for _ in range(3):
            await model.agenerate(messages)
        sequential_time = time.time() - start

        # Time 3 concurrent calls
        start = time.time()
        await asyncio.gather(*[model.agenerate(messages) for _ in range(3)])
        concurrent_time = time.time() - start

        # Concurrent should be significantly faster (we sleep 0.01s per call)
        # Sequential: ~0.03s, Concurrent: ~0.01s
        assert concurrent_time < sequential_time * 0.5


class TestAsyncAgents:
    """Test async agent functionality."""

    @pytest.mark.asyncio
    async def test_agent_has_arun_method(self):
        """Test that Agent has arun method."""
        model = MockAsyncModel()
        agent = ToolCallingAgent(model=model, tools=[])
        assert hasattr(agent, "arun")

    @pytest.mark.asyncio
    async def test_agent_arun_basic(self):
        """Test basic async agent run."""
        model = MockAsyncModel()

        # Mock the agent's internal methods to avoid full execution
        with patch.object(ToolCallingAgent, "_arun_stream") as mock_stream:
            # Mock the async generator
            async def mock_gen(*args, **kwargs):
                from smolagents.memory import FinalAnswerStep

                yield FinalAnswerStep(output="test result")

            mock_stream.return_value = mock_gen()

            agent = ToolCallingAgent(model=model, tools=[])
            result = await agent.arun("test task")

            assert result == "test result"
            assert mock_stream.called

    @pytest.mark.asyncio
    async def test_agent_arun_with_stream(self):
        """Test async agent run with streaming."""
        model = MockAsyncModel()

        with patch.object(ToolCallingAgent, "_arun_stream") as mock_stream:
            from smolagents.memory import ActionStep, FinalAnswerStep

            async def mock_gen(*args, **kwargs):
                yield ActionStep(step_number=1)
                yield FinalAnswerStep(output="test result")

            mock_stream.return_value = mock_gen()

            agent = ToolCallingAgent(model=model, tools=[])
            stream = agent.arun("test task", stream=True)

            steps = []
            async for step in stream:
                steps.append(step)

            assert len(steps) == 2
            assert isinstance(steps[0], ActionStep)
            assert isinstance(steps[1], FinalAnswerStep)

    @pytest.mark.asyncio
    async def test_concurrent_agent_execution_separate_instances(self):
        """Test concurrent execution with SEPARATE agent instances (correct pattern)."""
        model = MockAsyncModel()

        # Mock _arun_stream for each agent
        with patch.object(ToolCallingAgent, "_arun_stream") as mock_stream:
            from smolagents.memory import FinalAnswerStep

            async def mock_gen(*args, **kwargs):
                await asyncio.sleep(0.01)  # Simulate work
                yield FinalAnswerStep(output=f"result for {args[0]}")

            mock_stream.return_value = mock_gen()

            # ✅ CORRECT: Separate agent instances
            tasks = ["task1", "task2", "task3"]
            agents = [ToolCallingAgent(model=model, tools=[]) for _ in tasks]

            results = await asyncio.gather(*[agent.arun(task) for agent, task in zip(agents, tasks)])

            assert len(results) == 3
            # All should complete without memory corruption

    @pytest.mark.asyncio
    async def test_agent_state_isolation(self):
        """Test that separate agent instances maintain isolated state."""
        model = MockAsyncModel()

        with patch.object(ToolCallingAgent, "_arun_stream") as mock_stream:
            from smolagents.memory import FinalAnswerStep

            # Track which agent instance is being used
            call_count = {"agent1": 0, "agent2": 0}

            async def mock_gen_factory(agent_id):
                async def mock_gen(*args, **kwargs):
                    call_count[agent_id] += 1
                    await asyncio.sleep(0.01)
                    yield FinalAnswerStep(output=f"result from {agent_id}")

                return mock_gen()

            agent1 = ToolCallingAgent(model=model, tools=[])
            agent2 = ToolCallingAgent(model=model, tools=[])

            # Run both agents concurrently
            mock_stream.side_effect = [mock_gen_factory("agent1"), mock_gen_factory("agent2")]

            results = await asyncio.gather(agent1.arun("task1"), agent2.arun("task2"))

            assert len(results) == 2
            # Each agent should have been called once
            assert call_count["agent1"] == 1
            assert call_count["agent2"] == 1


class TestAsyncHelperMethods:
    """Test async helper methods on agents."""

    @pytest.mark.asyncio
    async def test_aprovide_final_answer(self):
        """Test async provide_final_answer method."""
        model = MockAsyncModel()
        agent = ToolCallingAgent(model=model, tools=[])

        # Mock memory
        from smolagents.memory import Memory

        agent.memory = Memory()

        result = await agent.aprovide_final_answer("test task")

        assert model.agenerate_called
        assert result.role == MessageRole.ASSISTANT

    @pytest.mark.asyncio
    async def test_ahandle_max_steps_reached(self):
        """Test async handle_max_steps_reached method."""
        model = MockAsyncModel()
        agent = ToolCallingAgent(model=model, tools=[])

        # Mock memory
        from smolagents.memory import Memory

        agent.memory = Memory()
        agent.step_number = 10

        result = await agent._ahandle_max_steps_reached("test task")

        assert model.agenerate_called
        assert result is not None

    @pytest.mark.asyncio
    async def test_agenerate_planning_step(self):
        """Test async generate_planning_step method."""
        model = MockAsyncModel()
        agent = ToolCallingAgent(model=model, tools=[])

        # Mock memory
        from smolagents.memory import Memory

        agent.memory = Memory()

        steps = []
        async for step in agent._agenerate_planning_step("test task", is_first_step=True, step=1):
            steps.append(step)

        assert model.agenerate_called
        assert len(steps) > 0


class TestAsyncIntegration:
    """Integration tests for async functionality."""

    @pytest.mark.asyncio
    async def test_end_to_end_async_workflow(self):
        """Test complete async workflow from model to agent."""
        model = MockAsyncModel()

        # This is a simplified integration test
        # In reality, would need full agent setup with executors, etc.
        with patch.object(ToolCallingAgent, "_arun_stream") as mock_stream:
            from smolagents.memory import ActionStep, FinalAnswerStep

            async def mock_gen(*args, **kwargs):
                # Simulate multi-step execution
                yield ActionStep(step_number=1)
                await asyncio.sleep(0.01)
                yield ActionStep(step_number=2)
                await asyncio.sleep(0.01)
                yield FinalAnswerStep(output="Final result")

            mock_stream.return_value = mock_gen()

            agent = ToolCallingAgent(model=model, tools=[])

            # Test with streaming
            steps = []
            async for step in agent.arun("complex task", stream=True):
                steps.append(step)

            assert len(steps) == 3
            assert isinstance(steps[-1], FinalAnswerStep)
            assert steps[-1].output == "Final result"

    @pytest.mark.asyncio
    async def test_model_compatibility(self):
        """Test that sync and async interfaces coexist."""
        model = MockAsyncModel()
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}])]

        # Both sync and async should work
        sync_result = model.generate(messages)
        async_result = await model.agenerate(messages)

        assert model.generate_called
        assert model.agenerate_called
        assert sync_result.role == MessageRole.ASSISTANT
        assert async_result.role == MessageRole.ASSISTANT


class TestAsyncErrorHandling:
    """Test error handling in async methods."""

    @pytest.mark.asyncio
    async def test_agenerate_not_implemented(self):
        """Test that base Model raises NotImplementedError for agenerate."""
        model = Model()
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}])]

        with pytest.raises(NotImplementedError, match="must be implemented in child classes"):
            await model.agenerate(messages)

    @pytest.mark.asyncio
    async def test_astep_stream_not_implemented(self):
        """Test that base Agent raises NotImplementedError for _astep_stream."""
        model = MockAsyncModel()
        agent = ToolCallingAgent(model=model, tools=[])

        from smolagents.memory import ActionStep

        step = ActionStep(step_number=1)

        with pytest.raises(NotImplementedError, match="must be implemented in child classes"):
            async for _ in agent._astep_stream(step):
                pass

    @pytest.mark.asyncio
    async def test_async_exception_propagation(self):
        """Test that exceptions in async methods are properly propagated."""

        class FailingModel(Model):
            async def agenerate(self, messages, **kwargs):
                raise ValueError("Test error")

        model = FailingModel()
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}])]

        with pytest.raises(ValueError, match="Test error"):
            await model.agenerate(messages)


class TestAsyncTools:
    """Test async tool functionality."""

    def test_sync_tool(self):
        """Test that sync tools work normally."""
        from smolagents.tools import Tool

        class SyncTool(Tool):
            name = "sync_test"
            description = "Test sync tool"
            inputs = {"message": {"type": "string", "description": "A message"}}
            output_type = "string"

            def forward(self, message: str):
                return f"sync: {message}"

        tool = SyncTool()
        result = tool("hello")
        assert result == "sync: hello"

    @pytest.mark.asyncio
    async def test_async_tool(self):
        """Test that async tools work and return coroutines."""
        import inspect

        from smolagents.tools import Tool

        class AsyncTool(Tool):
            name = "async_test"
            description = "Test async tool"
            inputs = {"message": {"type": "string", "description": "A message"}}
            output_type = "string"

            async def forward(self, message: str):
                await asyncio.sleep(0.01)
                return f"async: {message}"

        tool = AsyncTool()
        result = tool("world")

        # Should return a coroutine
        assert inspect.iscoroutine(result)

        # Await the coroutine
        result_value = await result
        assert result_value == "async: world"

    def test_agent_execute_sync_tool(self):
        """Test agent executing sync tool."""
        from smolagents import ToolCallingAgent
        from smolagents.tools import Tool

        class SyncTool(Tool):
            name = "sync_test"
            description = "Test sync tool"
            inputs = {"message": {"type": "string", "description": "A message"}}
            output_type = "string"

            def forward(self, message: str):
                return f"sync: {message}"

        model = MockAsyncModel()
        tool = SyncTool()
        agent = ToolCallingAgent(model=model, tools=[tool])

        result = agent.execute_tool_call("sync_test", {"message": "hello"})
        assert result == "sync: hello"

    def test_agent_execute_async_tool_in_sync_context(self):
        """Test agent executing async tool in sync context (uses asyncio.run)."""
        from smolagents import ToolCallingAgent
        from smolagents.tools import Tool

        class AsyncTool(Tool):
            name = "async_test"
            description = "Test async tool"
            inputs = {"message": {"type": "string", "description": "A message"}}
            output_type = "string"

            async def forward(self, message: str):
                await asyncio.sleep(0.01)
                return f"async: {message}"

        model = MockAsyncModel()
        tool = AsyncTool()
        agent = ToolCallingAgent(model=model, tools=[tool])

        # execute_tool_call should detect async tool and use asyncio.run()
        result = agent.execute_tool_call("async_test", {"message": "world"})
        assert result == "async: world"

    @pytest.mark.asyncio
    async def test_agent_async_execute_async_tool(self):
        """Test async agent executing async tool (true async await)."""
        from smolagents import ToolCallingAgent
        from smolagents.tools import Tool

        class AsyncTool(Tool):
            name = "async_test"
            description = "Test async tool"
            inputs = {"message": {"type": "string", "description": "A message"}}
            output_type = "string"

            async def forward(self, message: str):
                await asyncio.sleep(0.01)
                return f"async: {message}"

        model = MockAsyncModel()
        tool = AsyncTool()
        agent = ToolCallingAgent(model=model, tools=[tool])

        # async_execute_tool_call should await async tool directly
        result = await agent.async_execute_tool_call("async_test", {"message": "world"})
        assert result == "async: world"

    @pytest.mark.asyncio
    async def test_human_in_the_loop_async_tool(self):
        """Test human-in-the-loop pattern with async tool."""
        from smolagents.tools import Tool

        class HumanApprovalTool(Tool):
            name = "human_approval"
            description = "Wait for human approval"
            inputs = {"action": {"type": "string", "description": "Action to approve"}}
            output_type = "string"

            async def forward(self, action: str):
                # Simulated async wait for human input
                # In real app: await message_queue.get(), await db.poll(), etc.
                await asyncio.sleep(0.05)
                return f"approved: {action}"

        tool = HumanApprovalTool()
        result = await tool("delete important file")
        assert result == "approved: delete important file"


class TestAsyncCodeAgent:
    """Test async CodeAgent functionality with async tools."""

    @pytest.mark.asyncio
    async def test_async_executor_with_async_tool(self):
        """Test that async executor can run async tools transparently."""
        from smolagents.local_python_executor import LocalPythonExecutor
        from smolagents.tools import Tool

        class AsyncTestTool(Tool):
            name = "async_tool"
            description = "Test async tool"
            inputs = {"msg": {"type": "string", "description": "Message"}}
            output_type = "string"

            async def forward(self, msg: str):
                await asyncio.sleep(0.01)
                return f"async: {msg}"

        tool = AsyncTestTool()
        executor = LocalPythonExecutor(additional_authorized_imports=[])
        executor.send_tools({"async_tool": tool, "final_answer": lambda x: x})

        # Test direct call
        code = 'async_tool("hello")'
        output = await executor.async_call(code)
        assert output.output == "async: hello"

    @pytest.mark.asyncio
    async def test_async_executor_with_assignment(self):
        """Test async executor with assignment pattern (most common in CodeAgent)."""
        from smolagents.local_python_executor import LocalPythonExecutor
        from smolagents.tools import Tool

        class AsyncApprovalTool(Tool):
            name = "human_approval"
            description = "Get human approval"
            inputs = {"action": {"type": "string", "description": "Action"}}
            output_type = "string"

            async def forward(self, action: str):
                await asyncio.sleep(0.01)
                return f"approved: {action}"

        tool = AsyncApprovalTool()
        executor = LocalPythonExecutor(additional_authorized_imports=[])
        executor.send_tools({"human_approval": tool, "final_answer": lambda x: x})

        # Assignment pattern - generated code doesn't need await!
        code = '''
result = human_approval("delete file")
final_answer(result)
'''
        output = await executor.async_call(code)
        assert output.output == "approved: delete file"
        assert output.is_final_answer == True

    @pytest.mark.asyncio
    async def test_async_code_agent_has_astep_stream(self):
        """Test that CodeAgent has async _astep_stream method."""
        import inspect

        from smolagents import CodeAgent

        model = MockAsyncModel()
        agent = CodeAgent(model=model, tools=[])

        assert hasattr(agent, "_astep_stream")
        # _astep_stream is an async generator function
        assert inspect.isasyncgenfunction(agent._astep_stream)


# Note: These tests mock most functionality to avoid requiring actual API calls
# For full integration testing with real models, use separate integration test suite
# with appropriate API keys and markers (e.g., @pytest.mark.requires_api_key)
