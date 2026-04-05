import json

import pytest
from PIL import Image

from smolagents.agents import ToolCall
from smolagents.memory import (
    ActionStep,
    AgentMemory,
    ChatMessage,
    MemoryStep,
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
