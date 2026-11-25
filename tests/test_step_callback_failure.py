"""A failing step callback must not erase an already completed step."""

import pytest

from smolagents import CodeAgent
from smolagents.memory import ActionStep
from smolagents.models import ChatMessage, MessageRole, Model


class ScriptedModel(Model):
    def generate(self, messages, **kwargs):
        return ChatMessage(role=MessageRole.ASSISTANT, content="Final Answer: done")


def test_action_step_recorded_before_callback_failure():
    def bad_callback(step, **kwargs):
        raise OSError("screenshot I/O failed")

    agent = CodeAgent(tools=[], model=ScriptedModel(), step_callbacks=[bad_callback])

    with pytest.raises(OSError):
        agent.run("task")

    assert any(isinstance(step, ActionStep) for step in agent.memory.steps)
