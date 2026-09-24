import time
import pytest
from smolagents.memory import AgentMemory, TaskStep, ActionStep
from smolagents.monitoring import Timing


def make_action_step(step_number: int, observation: str) -> ActionStep:
    return ActionStep(
        step_number=step_number,
        timing=Timing(start_time=time.time(), end_time=time.time()),
        observations=observation,
    )


def test_truncate_steps_removes_oldest_first():
    mem = AgentMemory("You are a helpful agent.")
    mem.steps.append(TaskStep(task="What is 2+2?"))
    for i in range(3):
        mem.steps.append(make_action_step(i, "long observation " * 200))

    removed = mem.truncate_steps(max_chars=100)
    assert removed > 0
    # TaskStep must always be kept
    assert any(isinstance(s, TaskStep) for s in mem.steps)


def test_truncate_steps_no_removal_when_under_limit():
    mem = AgentMemory("You are a helpful agent.")
    mem.steps.append(TaskStep(task="hi"))
    mem.steps.append(make_action_step(0, "short"))

    removed = mem.truncate_steps(max_chars=999999)
    assert removed == 0
    assert len(mem.steps) == 2


def test_truncate_steps_preserves_task_step():
    mem = AgentMemory("You are a helpful agent.")
    mem.steps.append(TaskStep(task="What is 2+2?"))
    mem.steps.append(make_action_step(0, "x" * 10000))

    mem.truncate_steps(max_chars=1)
    # Only TaskStep should remain
    assert len(mem.steps) == 1
    assert isinstance(mem.steps[0], TaskStep)
