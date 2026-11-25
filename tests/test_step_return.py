"""step() must return the final answer or None, not an ActionOutput wrapper."""

import time

from smolagents import CodeAgent
from smolagents.memory import ActionStep, TaskStep
from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import Timing


class ScriptedModel(Model):
    def __init__(self, replies):
        super().__init__(model_id="scripted")
        self.replies = iter(replies)

    def generate(self, messages, **kwargs):
        return ChatMessage(role=MessageRole.ASSISTANT, content=next(self.replies))


def _make_agent(reply):
    agent = CodeAgent(tools=[], model=ScriptedModel([reply]), verbosity_level=0)
    agent.python_executor.send_tools({**agent.tools})
    agent.memory.steps.append(TaskStep(task="What is 6 * 7?"))
    return agent


def _step(agent):
    memory_step = ActionStep(step_number=1, timing=Timing(start_time=time.time()))
    return agent.step(memory_step)


def test_step_returns_final_answer():
    agent = _make_agent("Thought: done.\n<code>\nfinal_answer('42')\n</code>")
    assert _step(agent) == "42"


def test_non_final_step_returns_none():
    agent = _make_agent("Thought: compute.\n<code>\nx = 6 * 7\nprint(x)\n</code>")
    assert _step(agent) is None
