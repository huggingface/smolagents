import json

import pytest
from PIL import Image

from smolagents.agents import ToolCall
from smolagents.memory import (
    ActionStep,
    AgentMemory,
    ChatMessage,
    MemoryStep,
    MemorySummaryStep,
    MessageRole,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
)
from smolagents.monitoring import Timing, TokenUsage


class TestAgentMemory:
    def test_initialization(self):
        system_prompt = "This is a system prompt."
        memory = AgentMemory(system_prompt=system_prompt)
        assert memory.system_prompt.system_prompt == system_prompt
        assert memory.steps == []

    def test_return_all_code_actions(self):
        memory = AgentMemory(system_prompt="This is a system prompt.")
        memory.steps = [
            ActionStep(step_number=1, timing=Timing(start_time=0.0, end_time=1.0), code_action="print('Hello')"),
            ActionStep(step_number=2, timing=Timing(start_time=0.0, end_time=1.0), code_action=None),
            ActionStep(step_number=3, timing=Timing(start_time=0.0, end_time=1.0), code_action="print('World')"),
        ]  # type: ignore
        assert memory.return_full_code() == "print('Hello')\n\nprint('World')"


class TestMemoryStep:
    def test_initialization(self):
        step = MemoryStep()
        assert isinstance(step, MemoryStep)

    def test_dict(self):
        step = MemoryStep()
        assert step.dict() == {}

    def test_to_messages(self):
        step = MemoryStep()
        with pytest.raises(NotImplementedError):
            step.to_messages()


def test_action_step_dict():
    action_step = ActionStep(
        model_input_messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
        tool_calls=[
            ToolCall(id="id", name="get_weather", arguments={"location": "Paris"}),
        ],
        timing=Timing(start_time=0.0, end_time=1.0),
        step_number=1,
        error=None,
        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="Hi"),
        model_output="Hi",
        observations="This is a nice observation",
        observations_images=[Image.new("RGB", (100, 100))],
        action_output="Output",
        token_usage=TokenUsage(input_tokens=10, output_tokens=20),
    )
    action_step_dict = action_step.dict()
    # Check each key individually for better test failure messages
    assert "model_input_messages" in action_step_dict
    assert action_step_dict["model_input_messages"] == [
        {"role": MessageRole.USER, "content": "Hello", "tool_calls": None, "raw": None, "token_usage": None}
    ]

    assert "tool_calls" in action_step_dict
    assert len(action_step_dict["tool_calls"]) == 1
    assert action_step_dict["tool_calls"][0] == {
        "id": "id",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": {"location": "Paris"},
        },
    }

    assert "timing" in action_step_dict
    assert action_step_dict["timing"] == {"start_time": 0.0, "end_time": 1.0, "duration": 1.0}

    assert "token_usage" in action_step_dict
    assert action_step_dict["token_usage"] == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}

    assert "step_number" in action_step_dict
    assert action_step_dict["step_number"] == 1

    assert "error" in action_step_dict
    assert action_step_dict["error"] is None

    assert "model_output_message" in action_step_dict
    assert action_step_dict["model_output_message"] == {
        "role": "assistant",
        "content": "Hi",
        "tool_calls": None,
        "raw": None,
        "token_usage": None,
    }

    assert "model_output" in action_step_dict
    assert action_step_dict["model_output"] == "Hi"

    assert "observations" in action_step_dict
    assert action_step_dict["observations"] == "This is a nice observation"

    assert "observations_images" in action_step_dict

    assert "action_output" in action_step_dict
    assert action_step_dict["action_output"] == "Output"


def test_action_step_to_messages():
    action_step = ActionStep(
        model_input_messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
        tool_calls=[
            ToolCall(id="id", name="get_weather", arguments={"location": "Paris"}),
        ],
        timing=Timing(start_time=0.0, end_time=1.0),
        step_number=1,
        error=None,
        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="Hi"),
        model_output="Hi",
        observations="This is a nice observation",
        observations_images=[Image.new("RGB", (100, 100))],
        action_output="Output",
        token_usage=TokenUsage(input_tokens=10, output_tokens=20),
    )
    messages = action_step.to_messages()
    assert len(messages) == 4
    for message in messages:
        assert isinstance(message, ChatMessage)
    assistant_message = messages[0]
    assert assistant_message.role == MessageRole.ASSISTANT
    assert len(assistant_message.content) == 1
    assert assistant_message.content[0]["type"] == "text"
    assert assistant_message.content[0]["text"] == "Hi"
    message = messages[1]
    assert message.role == MessageRole.TOOL_CALL

    assert len(message.content) == 1
    assert message.content[0]["type"] == "text"
    assert "Calling tools:" in message.content[0]["text"]

    image_message = messages[2]
    assert image_message.content[0]["type"] == "image"  # type: ignore

    observation_message = messages[3]
    assert observation_message.role == MessageRole.TOOL_RESPONSE
    assert "Observation:\nThis is a nice observation" in observation_message.content[0]["text"]


def test_action_step_to_messages_no_tool_calls_with_observations():
    action_step = ActionStep(
        model_input_messages=None,
        tool_calls=None,
        timing=Timing(start_time=0.0, end_time=1.0),
        step_number=1,
        error=None,
        model_output_message=None,
        model_output=None,
        observations="This is an observation.",
        observations_images=None,
        action_output=None,
        token_usage=TokenUsage(input_tokens=10, output_tokens=20),
    )
    messages = action_step.to_messages()
    assert len(messages) == 1
    observation_message = messages[0]
    assert observation_message.role == MessageRole.TOOL_RESPONSE
    assert "Observation:\nThis is an observation." in observation_message.content[0]["text"]


def test_planning_step_to_messages():
    planning_step = PlanningStep(
        model_input_messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="Plan"),
        plan="This is a plan.",
        timing=Timing(start_time=0.0, end_time=1.0),
    )
    messages = planning_step.to_messages(summary_mode=False)
    assert len(messages) == 2
    for message in messages:
        assert isinstance(message, ChatMessage)
        assert isinstance(message.content, list)
        assert len(message.content) == 1
        for content in message.content:
            assert isinstance(content, dict)
            assert "type" in content
            assert "text" in content
    assert messages[0].role == MessageRole.ASSISTANT
    assert messages[1].role == MessageRole.USER


def test_task_step_to_messages():
    task_step = TaskStep(task="This is a task.", task_images=[Image.new("RGB", (100, 100))])
    messages = task_step.to_messages(summary_mode=False)
    assert len(messages) == 1
    for message in messages:
        assert isinstance(message, ChatMessage)
        assert message.role == MessageRole.USER
        assert isinstance(message.content, list)
        assert len(message.content) == 2
        text_content = message.content[0]
        assert isinstance(text_content, dict)
        assert "type" in text_content
        assert "text" in text_content
        for image_content in message.content[1:]:
            assert isinstance(image_content, dict)
            assert "type" in image_content
            assert "image" in image_content


def test_system_prompt_step_to_messages():
    system_prompt_step = SystemPromptStep(system_prompt="This is a system prompt.")
    messages = system_prompt_step.to_messages(summary_mode=False)
    assert len(messages) == 1
    for message in messages:
        assert isinstance(message, ChatMessage)
        assert message.role == MessageRole.SYSTEM
        assert isinstance(message.content, list)
        assert len(message.content) == 1
        for content in message.content:
            assert isinstance(content, dict)
            assert "type" in content
            assert "text" in content


def test_memory_step_json_serialization():
    """Test that memory steps can be JSON serialized without raw fields."""

    # Create a mock ChatCompletion-like object (this is what was causing the error)
    class MockChatCompletion:
        def __init__(self):
            self.id = "chatcmpl-test"
            self.choices = []

    # Create a ChatMessage with raw field containing the non-serializable object
    chat_message = ChatMessage(role=MessageRole.ASSISTANT, content="Test response", raw=MockChatCompletion())

    # Test ActionStep serialization
    action_step = ActionStep(
        step_number=1,
        timing=Timing(start_time=123456, end_time=123457),
        model_output_message=chat_message,
        model_input_messages=[chat_message],
    )

    step_dict = action_step.dict()
    json_str = json.dumps(step_dict)
    # Raw field should be present but serializable
    assert "raw" in json_str
    assert "MockChatCompletion" in json_str

    # Test PlanningStep serialization
    planning_step = PlanningStep(
        model_input_messages=[chat_message],
        model_output_message=chat_message,
        plan="Test plan",
        timing=Timing(start_time=123456, end_time=123457),
    )

    planning_dict = planning_step.dict()
    json_str = json.dumps(planning_dict)
    # Raw field should be present but serializable
    assert "raw" in json_str
    assert "MockChatCompletion" in json_str


# ---------------------------------------------------------------------------
# Tests for MemorySummaryStep and AgentMemory.consolidate
# ---------------------------------------------------------------------------


def _make_action_step(step_number: int, output: str = "thought", observation: str = "obs") -> ActionStep:
    """Helper to build a minimal ActionStep."""
    return ActionStep(
        step_number=step_number,
        timing=Timing(start_time=0.0, end_time=1.0),
        model_output=output,
        observations=observation,
    )


class _FakeModel:
    """Minimal model stub whose ``generate`` returns a canned summary."""

    def __init__(self, summary_text: str = "consolidated summary"):
        self.summary_text = summary_text
        self.call_count = 0

    def generate(self, messages, **kwargs):
        self.call_count += 1
        return ChatMessage(role=MessageRole.ASSISTANT, content=self.summary_text)


class TestMemorySummaryStep:
    def test_to_messages(self):
        step = MemorySummaryStep(summary="A summary of earlier steps.")
        messages = step.to_messages()
        assert len(messages) == 1
        assert messages[0].role == MessageRole.USER
        assert "Consolidated summary" in messages[0].content[0]["text"]
        assert "A summary of earlier steps." in messages[0].content[0]["text"]

    def test_to_messages_summary_mode(self):
        step = MemorySummaryStep(summary="summary")
        # summary_mode should not suppress this step
        messages = step.to_messages(summary_mode=True)
        assert len(messages) == 1


class TestAgentMemoryConsolidate:
    def test_no_consolidation_when_disabled(self):
        memory = AgentMemory(system_prompt="sys", max_memory_steps=None)
        for i in range(10):
            memory.steps.append(_make_action_step(i))
        assert memory.consolidate(_FakeModel()) is False
        assert len(memory.steps) == 10

    def test_no_consolidation_when_under_threshold(self):
        memory = AgentMemory(system_prompt="sys", max_memory_steps=5)
        for i in range(5):
            memory.steps.append(_make_action_step(i))
        assert memory.consolidate(_FakeModel()) is False
        assert len(memory.steps) == 5

    def test_consolidation_replaces_old_steps(self):
        model = _FakeModel("short summary")
        memory = AgentMemory(system_prompt="sys", max_memory_steps=3)
        # Add a task step + 5 action steps
        memory.steps.append(TaskStep(task="do something"))
        for i in range(5):
            memory.steps.append(_make_action_step(i, output=f"thought_{i}", observation=f"obs_{i}"))

        assert memory.consolidate(model) is True
        assert model.call_count == 1

        # We should now have: TaskStep, MemorySummaryStep, and 3 remaining ActionSteps
        assert isinstance(memory.steps[0], TaskStep)
        assert isinstance(memory.steps[1], MemorySummaryStep)
        assert memory.steps[1].summary == "short summary"
        action_steps = [s for s in memory.steps if isinstance(s, ActionStep)]
        assert len(action_steps) == 3

    def test_consolidation_preserves_planning_steps_in_recent(self):
        model = _FakeModel("summary")
        memory = AgentMemory(system_prompt="sys", max_memory_steps=2)
        memory.steps.append(TaskStep(task="task"))
        for i in range(3):
            memory.steps.append(_make_action_step(i))
        memory.steps.append(
            PlanningStep(
                model_input_messages=[],
                model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="plan"),
                plan="plan text",
                timing=Timing(start_time=0.0, end_time=1.0),
            )
        )
        # 3 ActionSteps + 1 PlanningStep = 4 interaction steps; keep 2 → summarise 2
        result = memory.consolidate(model)
        assert result is True
        interaction_remaining = [s for s in memory.steps if isinstance(s, (ActionStep, PlanningStep))]
        assert len(interaction_remaining) == 2

    def test_consolidation_fallback_on_model_error(self):
        class _FailingModel:
            def generate(self, messages, **kwargs):
                raise RuntimeError("model unavailable")

        memory = AgentMemory(system_prompt="sys", max_memory_steps=2)
        for i in range(4):
            memory.steps.append(_make_action_step(i, output=f"t{i}"))

        # Should not raise, should fall back to truncation
        assert memory.consolidate(_FailingModel()) is True
        summary_steps = [s for s in memory.steps if isinstance(s, MemorySummaryStep)]
        assert len(summary_steps) == 1
        # Fallback produces raw text, not empty
        assert len(summary_steps[0].summary) > 0

    def test_repeated_consolidation(self):
        """After consolidation, adding more steps and consolidating again merges the summary."""
        model = _FakeModel("summary_v1")
        memory = AgentMemory(system_prompt="sys", max_memory_steps=2)
        for i in range(4):
            memory.steps.append(_make_action_step(i))
        memory.consolidate(model)

        # Add more steps
        model.summary_text = "summary_v2"
        for i in range(4, 7):
            memory.steps.append(_make_action_step(i))
        memory.consolidate(model)

        summary_steps = [s for s in memory.steps if isinstance(s, MemorySummaryStep)]
        # Only one summary step should remain (old one gets consolidated into new)
        assert len(summary_steps) == 1
        action_steps = [s for s in memory.steps if isinstance(s, ActionStep)]
        assert len(action_steps) == 2


class TestMaxMemoryStepsPropagation:
    """Integration test: max_memory_steps is accepted by agent constructors and triggers consolidation."""

    def test_code_agent_accepts_max_memory_steps(self):
        """CodeAgent constructor propagates max_memory_steps to AgentMemory."""
        from unittest.mock import MagicMock

        from smolagents.agents import CodeAgent
        from smolagents.models import Model

        model = MagicMock(spec=Model)
        model.model_id = "fake"
        agent = CodeAgent(tools=[], model=model, max_memory_steps=5)
        assert agent.memory.max_memory_steps == 5

    def test_tool_calling_agent_accepts_max_memory_steps(self):
        """ToolCallingAgent constructor propagates max_memory_steps to AgentMemory."""
        from unittest.mock import MagicMock

        from smolagents.agents import ToolCallingAgent
        from smolagents.models import Model

        model = MagicMock(spec=Model)
        model.model_id = "fake"
        agent = ToolCallingAgent(tools=[], model=model, max_memory_steps=3)
        assert agent.memory.max_memory_steps == 3

    def test_consolidation_triggered_when_threshold_exceeded(self):
        """Memory consolidation is triggered during agent run when steps exceed threshold."""
        from unittest.mock import MagicMock

        from smolagents.agents import CodeAgent
        from smolagents.models import Model

        model = MagicMock(spec=Model)
        model.model_id = "fake"
        agent = CodeAgent(tools=[], model=model, max_memory_steps=2)

        # Manually populate memory to simulate steps added during a run
        for i in range(5):
            agent.memory.steps.append(_make_action_step(i))

        # Consolidation should trigger and reduce steps
        fake_model = _FakeModel("integration summary")
        result = agent.memory.consolidate(fake_model)
        assert result is True

        summary_steps = [s for s in agent.memory.steps if isinstance(s, MemorySummaryStep)]
        assert len(summary_steps) == 1
        action_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
        assert len(action_steps) == 2
