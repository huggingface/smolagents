import pytest

from smolagents.agents import ToolCall
from smolagents.memory import (
    ActionStep,
    AgentMemory,
    ChatMessage,
    MemoryStep,
    Message,
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
        model_input_messages=[Message(role=MessageRole.USER, content="Hello")],
        tool_calls=[
            ToolCall(id="id", name="get_weather", arguments={"location": "Paris"}),
        ],
        timing=Timing(start_time=0.0, end_time=1.0),
        step_number=1,
        error=None,
        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="Hi"),
        model_output="Hi",
        observations="This is a nice observation",
        observations_images=["image1.png"],
        action_output="Output",
        token_usage=TokenUsage(input_tokens=10, output_tokens=20),
    )
    action_step_dict = action_step.dict()
    # Check each key individually for better test failure messages
    assert "model_input_messages" in action_step_dict
    assert action_step_dict["model_input_messages"] == [Message(role=MessageRole.USER, content="Hello")]

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

    assert "step" in action_step_dict
    assert action_step_dict["step"] == 1

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

    assert "action_output" in action_step_dict
    assert action_step_dict["action_output"] == "Output"


def test_action_step_to_messages():
    action_step = ActionStep(
        model_input_messages=[Message(role=MessageRole.USER, content="Hello")],
        tool_calls=[
            ToolCall(id="id", name="get_weather", arguments={"location": "Paris"}),
        ],
        timing=Timing(start_time=0.0, end_time=1.0),
        step_number=1,
        error=None,
        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="Hi"),
        model_output="Hi",
        observations="This is a nice observation",
        observations_images=["image1.png"],
        action_output="Output",
        token_usage=TokenUsage(input_tokens=10, output_tokens=20),
    )
    messages = action_step.to_messages()
    assert len(messages) == 4
    for message in messages:
        assert isinstance(message, dict)
        assert "role" in message
        assert "content" in message
        assert isinstance(message["role"], MessageRole)
        assert isinstance(message["content"], list)
    assistant_message = messages[0]
    assert assistant_message["role"] == MessageRole.ASSISTANT
    assert len(assistant_message["content"]) == 1
    for content in assistant_message["content"]:
        assert isinstance(content, dict)
        assert "type" in content
        assert "text" in content
    message = messages[1]
    assert message["role"] == MessageRole.TOOL_CALL

    assert len(message["content"]) == 1
    text_content = message["content"][0]
    assert isinstance(text_content, dict)
    assert "type" in text_content
    assert "text" in text_content

    image_message = messages[2]
    image_content = image_message["content"][0]
    assert isinstance(image_content, dict)
    assert "type" in image_content
    assert "image" in image_content

    observation_message = messages[3]
    assert observation_message["role"] == MessageRole.TOOL_RESPONSE
    assert "Observation:\nThis is a nice observation" in observation_message["content"][0]["text"]


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
    assert observation_message["role"] == MessageRole.TOOL_RESPONSE
    assert "Observation:\nThis is an observation." in observation_message["content"][0]["text"]


def test_planning_step_to_messages():
    planning_step = PlanningStep(
        model_input_messages=[Message(role=MessageRole.USER, content="Hello")],
        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="Plan"),
        plan="This is a plan.",
        timing=Timing(start_time=0.0, end_time=1.0),
    )
    messages = planning_step.to_messages(summary_mode=False)
    assert len(messages) == 2
    for message in messages:
        assert isinstance(message, dict)
        assert "role" in message
        assert "content" in message
        assert isinstance(message["content"], list)
        assert len(message["content"]) == 1
        for content in message["content"]:
            assert isinstance(content, dict)
            assert "type" in content
            assert "text" in content
    assert messages[0]["role"] == MessageRole.ASSISTANT
    assert messages[1]["role"] == MessageRole.USER


def test_task_step_to_messages():
    task_step = TaskStep(task="This is a task.", task_images=["task_image1.png"])
    messages = task_step.to_messages(summary_mode=False)
    assert len(messages) == 1
    for message in messages:
        assert isinstance(message, dict)
        assert "role" in message
        assert "content" in message
        assert isinstance(message["role"], MessageRole)
        assert message["role"] == MessageRole.USER
        assert isinstance(message["content"], list)
        assert len(message["content"]) == 2
        text_content = message["content"][0]
        assert isinstance(text_content, dict)
        assert "type" in text_content
        assert "text" in text_content
        for image_content in message["content"][1:]:
            assert isinstance(image_content, dict)
            assert "type" in image_content
            assert "image" in image_content


def test_system_prompt_step_to_messages():
    system_prompt_step = SystemPromptStep(system_prompt="This is a system prompt.")
    messages = system_prompt_step.to_messages(summary_mode=False)
    assert len(messages) == 1
    for message in messages:
        assert isinstance(message, dict)
        assert "role" in message
        assert "content" in message
        assert isinstance(message["role"], MessageRole)
        assert message["role"] == MessageRole.SYSTEM
        assert isinstance(message["content"], list)
        assert len(message["content"]) == 1
        for content in message["content"]:
            assert isinstance(content, dict)
            assert "type" in content
            assert "text" in content
