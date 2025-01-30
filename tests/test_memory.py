import pytest

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


def test_action_step_to_messages():
    action_step = ActionStep(
        model_input_messages=[Message(role=MessageRole.USER, content="Hello")],
        tool_calls=None,
        start_time=0.0,
        end_time=1.0,
        step_number=1,
        error=None,
        duration=1.0,
        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="Hi"),
        model_output="Hi",
        observations="Observation",
        observations_images=["image1.png"],
        action_output="Output",
    )
    messages = action_step.to_messages()
    assert len(messages) == 2
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
    user_message = messages[1]
    assert user_message["role"] == MessageRole.USER
    assert len(user_message["content"]) == 2
    text_content = user_message["content"][0]
    assert isinstance(text_content, dict)
    assert "type" in text_content
    assert "text" in text_content
    for image_content in user_message["content"][1:]:
        assert isinstance(image_content, dict)
        assert "type" in image_content
        assert "image" in image_content


def test_planning_step_to_messages():
    planning_step = PlanningStep(
        model_input_messages=[Message(role=MessageRole.USER, content="Hello")],
        model_output_message_facts=ChatMessage(role=MessageRole.ASSISTANT, content="Facts"),
        facts="These are facts.",
        model_output_message_plan=ChatMessage(role=MessageRole.ASSISTANT, content="Plan"),
        plan="This is a plan.",
    )
    messages = planning_step.to_messages(summary_mode=False)
    assert len(messages) == 2
    for message in messages:
        assert isinstance(message, dict)
        assert "role" in message
        assert "content" in message
        assert isinstance(message["role"], MessageRole)
        assert message["role"] == MessageRole.ASSISTANT
        assert isinstance(message["content"], list)
        assert len(message["content"]) == 1
        for content in message["content"]:
            assert isinstance(content, dict)
            assert "type" in content
            assert "text" in content


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
