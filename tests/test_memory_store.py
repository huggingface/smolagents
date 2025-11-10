# Standard library imports
import base64
import io

# Third-party imports
import pytest
from PIL import Image

# Local imports
from smolagents.memory import AgentMemory, PlanningStep, SystemPromptStep, TaskStep
from smolagents.memory_store import AgentMemoryStore, load_agent_state, save_agent_state


class MockModel:
    def __init__(self):
        self.model_id = "test-model-123"


class MockAgent:
    def __init__(self):
        self.model = MockModel()
        # Initialize with a default system prompt
        self.memory = AgentMemory(system_prompt="test system prompt")


def create_test_image():
    """Create a simple test image."""
    img = Image.new("RGB", (60, 30), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def test_basic_memory_persistence():
    """Test basic save and restore functionality."""
    store = AgentMemoryStore()
    agent = MockAgent()
    agent.memory.system_prompt = SystemPromptStep("test prompt")

    # Save state
    state = store.save_memory_state(agent)

    # Verify saved state
    assert state["system_prompt"] == "test prompt"
    assert "steps" in state
    assert len(state["steps"]) == 0
    assert "metadata" in state
    assert state["metadata"]["model_id"] == "test-model-123"

    # Restore to new agent
    new_agent = MockAgent()
    store.restore_memory_state(new_agent, state)
    assert new_agent.memory.system_prompt.system_prompt == "test prompt"


def test_task_step_persistence():
    """Test persistence of task steps with images."""
    store = AgentMemoryStore()
    agent = MockAgent()

    # Create task step with image
    test_image = create_test_image()  # This now returns a base64 data URI
    task_step = TaskStep(task="test task", task_images=[test_image])
    agent.memory.steps.append(task_step)

    # Save and restore
    state = store.save_memory_state(agent)
    new_agent = MockAgent()
    store.restore_memory_state(new_agent, state)

    # Verify
    restored_step = new_agent.memory.steps[0]
    assert isinstance(restored_step, TaskStep)
    assert restored_step.task == "test task"
    assert len(restored_step.task_images) == 1


def test_planning_step_persistence():
    """Test persistence of planning steps."""
    store = AgentMemoryStore()
    agent = MockAgent()

    # Create planning step with all required arguments
    planning_step = PlanningStep(
        model_input_messages=[{"role": "system", "content": "test input"}],
        model_output_message_facts={"role": "assistant", "content": "test facts output"},
        facts="test facts",
        model_output_message_plan={"role": "assistant", "content": "test plan output"},
        plan="test plan"
    )
    agent.memory.steps.append(planning_step)

    # Save and restore
    state = store.save_memory_state(agent)
    new_agent = MockAgent()
    store.restore_memory_state(new_agent, state)

    # Verify
    restored_step = new_agent.memory.steps[0]
    assert isinstance(restored_step, PlanningStep)
    assert restored_step.plan == "test plan"
    assert restored_step.facts == "test facts"
    assert restored_step.model_input_messages == [{"role": "system", "content": "test input"}]
    assert restored_step.model_output_message_facts == {"role": "assistant", "content": "test facts output"}
    assert restored_step.model_output_message_plan == {"role": "assistant", "content": "test plan output"}


def test_error_handling():
    """Test error handling for invalid memory states"""
    store = AgentMemoryStore()

    # Test missing required fields
    with pytest.raises(ValueError, match="Missing required field 'system_prompt'"):
        store.restore_memory_state(MockAgent(), {"steps": []})

    with pytest.raises(ValueError, match="Missing required field 'steps'"):
        store.restore_memory_state(MockAgent(), {"system_prompt": "test"})

    # Test invalid types
    with pytest.raises(ValueError, match="System prompt must be a string"):
        store.restore_memory_state(MockAgent(), {"system_prompt": 123, "steps": []})

    with pytest.raises(ValueError, match="Steps must be a list"):
        store.restore_memory_state(MockAgent(), {"system_prompt": "test", "steps": "not a list"})


def test_helper_functions():
    """Test the helper functions save_agent_state and load_agent_state"""
    agent = MockAgent()
    agent.memory.steps.append(TaskStep(task="test task"))

    # Test save_agent_state
    state = save_agent_state(agent)
    assert "system_prompt" in state
    assert "steps" in state
    assert len(state["steps"]) == 1

    # Test load_agent_state
    new_agent = MockAgent()
    load_agent_state(new_agent, state)
    assert len(new_agent.memory.steps) == 1
    assert isinstance(new_agent.memory.steps[0], TaskStep)
    assert new_agent.memory.steps[0].task == "test task"
