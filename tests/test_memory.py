import pytest

from smolagents.memory import (
    AgentMemory,
    MemoryStep,
    MessageRole,
    TaskStep,
)
from smolagents.memory_providers import DictMemory, FilteredMemory


class TestAgentMemory:
    def test_initialization(self):
        system_prompt = "This is a system prompt."
        memory = AgentMemory(system_prompt=system_prompt)
        assert memory.system_prompt.system_prompt == system_prompt
        assert memory.steps == []

    def test_write_to_messages(self):
        system_prompt = "This is a system prompt."
        memory = AgentMemory(system_prompt=system_prompt)

        # Add a task step
        task_step = TaskStep(task="This is a task.")
        memory.steps.append(task_step)

        # Get messages
        messages = memory.write_to_messages()

        # Check that the system prompt and task step are included
        assert len(messages) == 2
        assert messages[0]["role"] == MessageRole.SYSTEM
        assert messages[1]["role"] == MessageRole.USER


class TestMemoryStep:
    def test_initialization(self):
        # MemoryStep is now an abstract class, so we can't instantiate it directly
        with pytest.raises(TypeError):
            MemoryStep()
class TestDictMemory:
    def test_initialization(self):
        system_prompt = "This is a system prompt."
        memory = DictMemory(system_prompt=system_prompt)
        assert memory.system_prompt.system_prompt == system_prompt
        assert len(memory.steps) == 0

    def test_add_step(self):
        system_prompt = "This is a system prompt."
        memory = DictMemory(system_prompt=system_prompt)

        # Add a task step
        task_step = TaskStep(task="This is a task.")
        step_id = memory.add_step(task_step)

        # Check that the step was added
        assert len(memory.steps) == 1
        assert memory.steps[step_id] == task_step

    def test_get_step(self):
        system_prompt = "This is a system prompt."
        memory = DictMemory(system_prompt=system_prompt)

        # Add a task step
        task_step = TaskStep(task="This is a task.")
        step_id = memory.add_step(task_step)

        # Get the step
        retrieved_step = memory.get_step(step_id)

        # Check that the retrieved step is the same as the added step
        assert retrieved_step == task_step

    def test_write_to_messages(self):
        system_prompt = "This is a system prompt."
        memory = DictMemory(system_prompt=system_prompt)

        # Add a task step
        task_step = TaskStep(task="This is a task.")
        memory.add_step(task_step)

        # Get messages
        messages = memory.write_to_messages()

        # Check that the system prompt and task step are included
        assert len(messages) == 2
        assert messages[0]["role"] == MessageRole.SYSTEM
        assert messages[1]["role"] == MessageRole.USER


class TestFilteredMemory:
    def test_initialization(self):
        system_prompt = "This is a system prompt."
        base_memory = AgentMemory(system_prompt=system_prompt)

        # Create a filtered memory that includes all steps
        def include_all(step):
            return True

        filtered_memory = FilteredMemory(base_memory, include_all)

        # Check that the filtered memory has the same base memory
        assert filtered_memory.base_memory == base_memory

    def test_get_succinct_steps(self):
        system_prompt = "This is a system prompt."
        base_memory = AgentMemory(system_prompt=system_prompt)

        # Add a task step
        task_step = TaskStep(task="This is a task.")
        base_memory.steps.append(task_step)

        # Create a filtered memory that excludes all steps
        def exclude_all(step):
            return False

        filtered_memory = FilteredMemory(base_memory, exclude_all)

        # Get succinct steps
        steps = filtered_memory.get_succinct_steps()

        # Check that no steps are included
        assert len(steps) == 0

    def test_get_full_steps(self):
        system_prompt = "This is a system prompt."
        base_memory = AgentMemory(system_prompt=system_prompt)

        # Add a task step
        task_step = TaskStep(task="This is a task.")
        base_memory.steps.append(task_step)

        # Create a filtered memory that includes all steps
        def include_all(step):
            return True

        filtered_memory = FilteredMemory(base_memory, include_all)

        # Get full steps
        steps = filtered_memory.get_full_steps()

        # Check that all steps are included
        assert len(steps) == 1
