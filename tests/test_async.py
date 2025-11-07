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


# Note: These tests mock most functionality to avoid requiring actual API calls
# For full integration testing with real models, use separate integration test suite
# with appropriate API keys and markers (e.g., @pytest.mark.requires_api_key)
