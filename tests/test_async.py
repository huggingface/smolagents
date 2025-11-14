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


    @pytest.mark.asyncio
    async def test_async_executor_with_mixed_sync_async_tools(self):
        """Test executor with both sync and async tools (backward compatibility)."""
        from smolagents.local_python_executor import LocalPythonExecutor
        from smolagents.tools import Tool

        class SyncTool(Tool):
            name = "sync_tool"
            description = "Sync tool"
            inputs = {"msg": {"type": "string", "description": "Message"}}
            output_type = "string"

            def forward(self, msg: str):
                return f"sync: {msg}"

        class AsyncTool(Tool):
            name = "async_tool"
            description = "Async tool"
            inputs = {"msg": {"type": "string", "description": "Message"}}
            output_type = "string"

            async def forward(self, msg: str):
                await asyncio.sleep(0.01)
                return f"async: {msg}"

        sync_tool = SyncTool()
        async_tool = AsyncTool()
        executor = LocalPythonExecutor(additional_authorized_imports=[])
        executor.send_tools({
            "sync_tool": sync_tool,
            "async_tool": async_tool,
            "final_answer": lambda x: x
        })

        # Test code using both sync and async tools
        code = '''
sync_result = sync_tool("hello")
async_result = async_tool("world")
final_answer(f"{sync_result}, {async_result}")
'''
        output = await executor.async_call(code)
        assert output.output == "sync: hello, async: world"

    @pytest.mark.asyncio
    async def test_concurrent_agents_non_blocking_demo(self):
        """
        Demonstrate non-blocking I/O benefit with concurrent agents.

        This test shows the key advantage of async support:
        - While one agent waits for an async tool (e.g., human approval),
          the event loop can switch to other agents
        - This is more efficient than threading (each thread = 1-8MB memory)
        """
        import time
        from smolagents.tools import Tool

        class SlowAsyncTool(Tool):
            name = "slow_approval"
            description = "Simulates slow approval process"
            inputs = {"action": {"type": "string", "description": "Action to approve"}}
            output_type = "string"

            async def forward(self, action: str):
                # Simulate waiting for human approval
                await asyncio.sleep(0.1)
                return f"approved: {action}"

        tool = SlowAsyncTool()

        # Run 5 tool calls concurrently
        start = time.time()
        results = await asyncio.gather(*[
            tool(f"action_{i}") for i in range(5)
        ])
        elapsed = time.time() - start

        # All 5 approvals should complete in ~0.1s (concurrent)
        # vs ~0.5s if sequential
        assert elapsed < 0.2  # Should be close to 0.1s
        assert len(results) == 5
        assert all("approved:" in r for r in results)


class TestAsyncRealWorldPatterns:
    """Test real-world async patterns and use cases."""

    @pytest.mark.asyncio
    async def test_human_approval_workflow(self):
        """
        Test human-in-the-loop approval workflow.

        This pattern is common for:
        - Sensitive operations requiring approval
        - Interactive debugging tools
        - User confirmation before actions
        """
        from smolagents.local_python_executor import LocalPythonExecutor
        from smolagents.tools import Tool

        class HumanApprovalTool(Tool):
            name = "request_approval"
            description = "Request human approval for sensitive operations"
            inputs = {
                "operation": {"type": "string", "description": "Operation to approve"},
                "risk_level": {"type": "string", "description": "Risk level: low, medium, high"}
            }
            output_type = "string"

            async def forward(self, operation: str, risk_level: str):
                # In production: await approval_queue.get(), await websocket.receive(), etc.
                await asyncio.sleep(0.02)  # Simulate waiting for human
                return f"APPROVED: {operation} (risk: {risk_level})"

        tool = HumanApprovalTool()
        executor = LocalPythonExecutor(additional_authorized_imports=[])
        executor.send_tools({"request_approval": tool, "final_answer": lambda x: x})

        # Generated code requests approval before sensitive operation
        code = '''
# Agent wants to delete files
approval = request_approval("delete /data/users", "high")
final_answer(approval)
'''
        output = await executor.async_call(code)
        assert "APPROVED: delete /data/users" in output.output
        assert "risk: high" in output.output

    @pytest.mark.asyncio
    async def test_external_api_calls_pattern(self):
        """
        Test external API call pattern with async tools.

        This pattern is useful for:
        - Fetching data from external services
        - Database queries
        - Network requests
        """
        from smolagents.local_python_executor import LocalPythonExecutor
        from smolagents.tools import Tool

        class ExternalAPITool(Tool):
            name = "fetch_user_data"
            description = "Fetch user data from external API"
            inputs = {"user_id": {"type": "string", "description": "User ID to fetch"}}
            output_type = "string"

            async def forward(self, user_id: str):
                # Simulate API call with network latency
                await asyncio.sleep(0.03)
                return f'{{"user_id": "{user_id}", "name": "John Doe", "status": "active"}}'

        tool = ExternalAPITool()
        executor = LocalPythonExecutor(additional_authorized_imports=[])
        executor.send_tools({"fetch_user_data": tool, "final_answer": lambda x: x})

        code = '''
user_data = fetch_user_data("user_123")
final_answer(user_data)
'''
        output = await executor.async_call(code)
        assert "user_123" in output.output
        assert "John Doe" in output.output


class TestAsyncRateLimiter:
    """Test async rate limiter functionality."""

    @pytest.mark.asyncio
    async def test_rate_limiter_athrottle(self):
        """Test that async rate limiter doesn't block event loop."""
        import time
        from smolagents.utils import RateLimiter

        # Create rate limiter: 60 requests/minute = 1 req/second
        rate_limiter = RateLimiter(requests_per_minute=60)

        # First call should be immediate
        start = time.time()
        await rate_limiter.athrottle()
        first_call_time = time.time() - start
        assert first_call_time < 0.1  # Should be nearly instant

        # Second call should wait ~1 second but not block
        start = time.time()
        await rate_limiter.athrottle()
        second_call_time = time.time() - start
        assert 0.9 < second_call_time < 1.2  # Should wait ~1 second

    @pytest.mark.asyncio
    async def test_rate_limiter_non_blocking(self):
        """Test that async rate limiter allows other tasks to run during wait."""
        import time
        from smolagents.utils import RateLimiter

        # Create rate limiter: 120 requests/minute = 0.5 second delay
        rate_limiter = RateLimiter(requests_per_minute=120)

        # Track execution order
        execution_order = []

        async def task1():
            execution_order.append("task1_start")
            await rate_limiter.athrottle()
            execution_order.append("task1_throttle1")
            await rate_limiter.athrottle()  # Will wait 0.5s
            execution_order.append("task1_throttle2")

        async def task2():
            await asyncio.sleep(0.1)
            execution_order.append("task2_during_wait")

        # Run both tasks concurrently
        await asyncio.gather(task1(), task2())

        # task2 should have executed during task1's throttle wait
        assert "task1_throttle1" in execution_order
        assert "task2_during_wait" in execution_order
        # task2 should run during the wait (proof of non-blocking)
        task1_throttle1_idx = execution_order.index("task1_throttle1")
        task2_idx = execution_order.index("task2_during_wait")
        task1_throttle2_idx = execution_order.index("task1_throttle2")
        assert task1_throttle1_idx < task2_idx < task1_throttle2_idx

    @pytest.mark.asyncio
    async def test_rate_limiter_disabled(self):
        """Test that rate limiter with None is disabled."""
        from smolagents.utils import RateLimiter

        rate_limiter = RateLimiter(requests_per_minute=None)

        # Should be instant even with multiple calls
        import time
        start = time.time()
        for _ in range(10):
            await rate_limiter.athrottle()
        elapsed = time.time() - start

        # All 10 calls should complete nearly instantly
        assert elapsed < 0.1


# Note: These tests mock most functionality to avoid requiring actual API calls
# For full integration testing with real models, use separate integration test suite
# with appropriate API keys and markers (e.g., @pytest.mark.requires_api_key)
