"""Completed steps survive failing callbacks without changing successful callback order."""

import pytest

from smolagents import CodeAgent, Tool, ToolCallingAgent
from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, Model


class ScriptedModel(Model):
    def __init__(self, replies):
        super().__init__(model_id="local-fixture")
        self.replies = iter(replies)
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        return next(self.replies)


class Ledger(Tool):
    name = "record_item"
    description = "Record an item in a local test ledger."
    inputs = {}
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.effects = 0

    def forward(self) -> str:
        self.effects += 1
        return "recorded"


def make_agent(agent_class, callbacks, planning=False):
    ledger = Ledger()
    if agent_class is CodeAgent:
        replies = [ChatMessage(role="assistant", content="<code>\nrecord_item()\nfinal_answer('done')\n</code>")]
    else:
        replies = [
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="record_item", arguments={}),
                    )
                ],
            ),
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_2",
                        type="function",
                        function=ChatMessageToolCallFunction(name="final_answer", arguments={"answer": "done"}),
                    )
                ],
            ),
        ]
    if planning:
        replies.insert(0, ChatMessage(role="assistant", content="A local test plan"))
    model = ScriptedModel(replies)
    agent = agent_class(
        model=model,
        tools=[ledger],
        max_steps=2,
        planning_interval=1 if planning else None,
        step_callbacks=callbacks,
    )
    return agent, ledger, model


@pytest.mark.parametrize("agent_class", [CodeAgent, ToolCallingAgent])
@pytest.mark.parametrize("step_class", [ActionStep, PlanningStep])
@pytest.mark.parametrize("error_class", [OSError, KeyboardInterrupt, GeneratorExit])
def test_callback_failure_keeps_completed_step(agent_class, step_class, error_class):
    failure = error_class("controlled callback failure")
    seen = []
    later_calls = []

    def fail(step, agent):
        assert all(previous is not step for previous in agent.memory.steps)
        seen.append(step)
        raise failure

    def later(step, agent):
        later_calls.append(step)

    agent, ledger, model = make_agent(agent_class, {step_class: [fail, later]}, step_class is PlanningStep)
    with pytest.raises(error_class) as caught:
        agent.run("Record one test item.")
    assert caught.value is failure
    assert len(seen) == 1
    assert sum(step is seen[0] for step in agent.memory.steps) == 1
    assert seen[0].timing.end_time is not None
    assert model.calls == 1
    assert ledger.effects == (1 if step_class is ActionStep else 0)
    assert later_calls == []
    if step_class is ActionStep:
        if agent_class is CodeAgent:
            assert seen[0].action_output == "done"
        else:
            assert "recorded" in seen[0].observations
    else:
        assert "A local test plan" in seen[0].plan


@pytest.mark.parametrize("agent_class", [CodeAgent, ToolCallingAgent])
def test_successful_callbacks_keep_order_and_annotations(agent_class):
    observed = []

    def annotate(step, agent):
        assert all(previous is not step for previous in agent.memory.steps)
        step.observations = "callback annotation"
        observed.append(step)

    def inspect_annotation(step, agent):
        assert step.observations == "callback annotation"
        assert observed[-1] is step

    agent, ledger, _ = make_agent(agent_class, {ActionStep: [annotate, inspect_annotation]})
    assert agent.run("Record one test item.") == "done"
    assert ledger.effects == 1
    assert observed
    for step in observed:
        assert sum(previous is step for previous in agent.memory.steps) == 1
        assert step.observations == "callback annotation"


@pytest.mark.parametrize("agent_class", [CodeAgent, ToolCallingAgent])
def test_final_answer_callback_failure_does_not_record_final_answer_step(agent_class):
    failure = OSError("final callback failure")

    def fail(step, agent):
        raise failure

    agent, ledger, _ = make_agent(agent_class, {FinalAnswerStep: fail})
    with pytest.raises(OSError) as caught:
        agent.run("Record one test item.")
    assert caught.value is failure
    assert ledger.effects == 1
    assert any(isinstance(step, ActionStep) for step in agent.memory.steps)
    assert not any(isinstance(step, FinalAnswerStep) for step in agent.memory.steps)


def test_max_steps_callback_failure_keeps_fallback_step():
    failure = OSError("fallback callback failure")
    seen = []

    def fail(step, agent):
        seen.append(step)
        raise failure

    agent = ToolCallingAgent(
        model=ScriptedModel([ChatMessage(role="assistant", content="fallback")]),
        tools=[],
        max_steps=0,
        step_callbacks={ActionStep: fail},
    )
    with pytest.raises(OSError) as caught:
        agent.run("Return a fallback.")
    assert caught.value is failure
    assert len(seen) == 1
    assert seen[0].action_output == "fallback"
    assert sum(step is seen[0] for step in agent.memory.steps) == 1
