#!/usr/bin/env python
# coding=utf-8
"""Tests for the stigmergy module."""

import json
import pytest
from unittest.mock import MagicMock, patch

from smolagents.stigmergy import SharedState, StigmergyAgent, StigmergyOrchestrator


class TestSharedState:
    """Tests for the SharedState class."""

    def test_init_defaults(self):
        """Test SharedState initializes with correct defaults."""
        state = SharedState()
        assert state.task == ""
        assert state.status == "pending"
        assert state.observations == []
        assert state.artifacts == {}
        assert state.signals == {"ready_for": [], "blockers": []}
        assert state.token_usage == {}

    def test_to_context(self):
        """Test SharedState converts to JSON context correctly."""
        state = SharedState(task="test task", status="in_progress")
        context = state.to_context()
        parsed = json.loads(context)
        assert parsed["task"] == "test task"
        assert parsed["status"] == "in_progress"

    def test_add_observation(self):
        """Test adding observations to shared state."""
        state = SharedState()
        state.add_observation("researcher", "Found important data", "research")
        assert len(state.observations) == 1
        assert state.observations[0]["agent"] == "researcher"
        assert state.observations[0]["content"] == "Found important data"
        assert state.observations[0]["type"] == "research"
        assert "timestamp" in state.observations[0]

    def test_set_and_get_artifact(self):
        """Test setting and getting artifacts."""
        state = SharedState()
        state.set_artifact("research", {"findings": ["a", "b"]}, agent_name="researcher")

        artifact = state.get_artifact("research")
        assert artifact == {"findings": ["a", "b"]}

        # Test non-existent artifact
        assert state.get_artifact("nonexistent") is None

    def test_ready_for_signals(self):
        """Test ready_for signal management."""
        state = SharedState()

        assert not state.is_ready_for("analyst")

        state.mark_ready_for("analyst")
        assert state.is_ready_for("analyst")
        assert "analyst" in state.signals["ready_for"]

        # Marking again should not duplicate
        state.mark_ready_for("analyst")
        assert state.signals["ready_for"].count("analyst") == 1

    def test_blocker_management(self):
        """Test blocker signal management."""
        state = SharedState()

        assert not state.has_blockers()

        state.add_blocker("missing_data")
        assert state.has_blockers()
        assert "missing_data" in state.signals["blockers"]

        state.remove_blocker("missing_data")
        assert not state.has_blockers()

    def test_token_tracking(self):
        """Test token usage tracking."""
        state = SharedState()

        state.add_tokens("researcher", 100)
        state.add_tokens("analyst", 150)
        state.add_tokens("researcher", 50)  # Add more to existing

        assert state.token_usage["researcher"] == 150
        assert state.token_usage["analyst"] == 150
        assert state.total_tokens() == 300

    def test_reset(self):
        """Test state reset clears everything except token_usage."""
        state = SharedState()
        state.task = "test"
        state.status = "completed"
        state.add_observation("agent", "obs")
        state.set_artifact("test", "value")
        state.add_tokens("agent", 100)

        state.reset()

        assert state.task == ""
        assert state.status == "pending"
        assert state.observations == []
        assert state.artifacts == {}
        # Token usage is preserved for tracking
        assert state.token_usage == {"agent": 100}

    def test_to_context_limits_observations(self):
        """Test that to_context limits number of observations."""
        state = SharedState()
        for i in range(10):
            state.add_observation("agent", f"observation {i}")

        context = state.to_context(max_observations=3)
        parsed = json.loads(context)
        assert len(parsed["recent_observations"]) == 3
        # Should be the last 3
        assert parsed["recent_observations"][0]["content"] == "observation 7"


class TestStigmergyAgent:
    """Tests for the StigmergyAgent class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock()
        model.generate.return_value = MagicMock(
            content="Test response\nFINAL_ANSWER: test answer",
            token_usage=MagicMock(input_tokens=10, output_tokens=20),
        )
        return model

    def test_agent_init(self, mock_model):
        """Test StigmergyAgent initialization."""
        state = SharedState()
        agent = StigmergyAgent(
            tools=[],
            model=mock_model,
            shared_state=state,
            role="researcher",
        )
        assert agent.role == "researcher"
        assert agent.shared_state is state

    def test_agent_uses_shared_state(self, mock_model):
        """Test that agent uses shared state instead of message history."""
        state = SharedState(task="test task")
        state.set_artifact("previous_work", "some data")

        agent = StigmergyAgent(
            tools=[],
            model=mock_model,
            shared_state=state,
            role="analyst",
        )

        # Get the messages that would be sent to the model
        messages = agent.write_memory_to_messages()

        # Should include current state as context
        state_message = messages[1]  # After system prompt
        assert "CURRENT SHARED STATE" in state_message.content[0]["text"]
        assert "test task" in state_message.content[0]["text"]


class TestStigmergyOrchestrator:
    """Tests for the StigmergyOrchestrator class."""

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        agents = []
        for role in ["researcher", "analyst", "writer"]:
            agent = MagicMock(spec=StigmergyAgent)
            agent.role = role
            agent.run = MagicMock()
            agents.append(agent)
        return agents

    def test_orchestrator_init(self, mock_agents):
        """Test orchestrator initialization."""
        state = SharedState()
        orchestrator = StigmergyOrchestrator(agents=mock_agents, shared_state=state)

        assert orchestrator.shared_state is state
        assert len(orchestrator.agents) == 3
        # All agents should share the same state
        for agent in mock_agents:
            assert agent.shared_state is state

    def test_orchestrator_run(self, mock_agents):
        """Test orchestrator runs all agents sequentially."""
        state = SharedState()
        orchestrator = StigmergyOrchestrator(agents=mock_agents, shared_state=state)

        result = orchestrator.run("test task")

        assert result["task"] == "test task"
        assert state.status == "completed"

        # Each agent should have been run
        for agent in mock_agents:
            agent.run.assert_called_once()

    def test_orchestrator_token_tracking(self, mock_agents):
        """Test orchestrator tracks tokens across agents."""
        state = SharedState()
        state.add_tokens("researcher", 100)
        state.add_tokens("analyst", 150)

        orchestrator = StigmergyOrchestrator(agents=mock_agents, shared_state=state)

        assert orchestrator.total_tokens() == 250

    def test_token_savings_estimate(self, mock_agents):
        """Test token savings estimation."""
        state = SharedState()
        state.add_tokens("researcher", 100)
        state.add_tokens("analyst", 100)

        orchestrator = StigmergyOrchestrator(agents=mock_agents, shared_state=state)
        savings = orchestrator.token_savings_estimate()

        assert savings["actual_tokens"] == 200
        assert savings["estimated_traditional_tokens"] == 500  # 200 * 2.5
        assert savings["savings_percentage"] == 60.0
        assert savings["tokens_saved"] == 300


class TestIntegration:
    """Integration tests for stigmergy workflow."""

    def test_full_workflow_simulation(self):
        """Test a simulated full workflow without actual model calls."""
        state = SharedState()
        state.task = "Analyze market trends"

        # Simulate researcher phase
        state.status = "in_progress"
        state.add_observation("researcher", "Found 3 key trends", "research")
        state.set_artifact(
            "research",
            {"trends": ["AI", "Cloud", "Remote"]},
            agent_name="researcher",
        )
        state.add_tokens("researcher", 150)
        state.mark_ready_for("analyst")

        # Simulate analyst phase
        state.add_observation("analyst", "Identified correlation between trends", "analysis")
        state.set_artifact(
            "analysis",
            {"insight": "AI drives cloud adoption"},
            agent_name="analyst",
        )
        state.add_tokens("analyst", 120)
        state.mark_ready_for("writer")

        # Simulate writer phase
        state.add_observation("writer", "Created executive summary", "writing")
        state.set_artifact(
            "draft",
            "Executive summary: AI is driving...",
            agent_name="writer",
        )
        state.add_tokens("writer", 100)
        state.status = "completed"

        # Verify final state
        assert state.status == "completed"
        assert len(state.artifacts) == 3
        assert state.total_tokens() == 370
        assert state.get_artifact("research")["trends"] == ["AI", "Cloud", "Remote"]
        assert "insight" in state.get_artifact("analysis")
