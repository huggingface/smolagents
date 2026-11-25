"""Interrupt raised from a planning callback must stop before the action."""

import pytest

from smolagents import CodeAgent
from smolagents.memory import ActionStep, PlanningStep
from smolagents.models import ChatMessage, MessageRole, Model


class ScriptedModel(Model):
    def generate(self, messages, **kwargs):
        return ChatMessage(role=MessageRole.ASSISTANT, content="Final Answer: done")


def test_interrupt_from_planning_callback_stops_before_action():
    agent = CodeAgent(tools=[], model=ScriptedModel(), planning_interval=1)

    def interrupt_on_planning(step, agent, **kwargs):
        if isinstance(step, PlanningStep):
            agent.interrupt()

    agent.step_callbacks.register(PlanningStep, interrupt_on_planning)

    with pytest.raises(Exception, match="interrupted"):
        agent.run("task")

    assert not any(isinstance(step, ActionStep) for step in agent.memory.steps)
