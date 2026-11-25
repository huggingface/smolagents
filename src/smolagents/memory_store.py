#!/usr/bin/env python
# coding=utf-8

# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory store functionality for saving and loading agent states."""

import base64
import io
import json
import logging
from datetime import UTC, datetime
from typing import Any, Dict

from PIL import Image

from smolagents.memory import ActionStep, PlanningStep, SystemPromptStep, TaskStep, ToolCall
from smolagents.models import MessageRole

from .agent_types import AgentImage


# Constants
REQUIRED_FIELDS = ["system_prompt", "steps"]
DEFAULT_METADATA = {"created_at": None, "updated_at": None, "agent_type": None, "model_id": None}

logger = logging.getLogger(__name__)


class AgentMemoryEncoder(json.JSONEncoder):
    """Custom JSON encoder for agent memory objects.

    Args:
        None

    Returns:
        JSON-serializable data structure

    This encoder handles special cases:
        - MessageRole enums (converts to value)
        - Message content with images (formats content list)
        - Objects with dict methods (uses dict() or __dict__)
        - General objects (converts to dict excluding private attributes)
    """

    def default(self, obj):
        if hasattr(obj, "value"):  # Handle MessageRole enum
            return obj.value
        if hasattr(obj, "content") and isinstance(obj.content, list):  # Handle message content with images
            return {
                "role": obj.role.value if hasattr(obj.role, "value") else str(obj.role),
                "content": [item if isinstance(item, (str, dict)) else str(item) for item in obj.content],
            }
        if hasattr(obj, "dict"):  # Handle objects with dict method
            try:
                return obj.dict()
            except Exception:
                return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        if hasattr(obj, "__dict__"):  # Handle other objects
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        return str(obj)


class AgentMemoryStore:
    """Manages saving and loading of agent memory states.

    This class handles the serialization and deserialization of agent memory,
    including special handling for:
    - PIL Images (converts to base64)
    - Complex objects (string representations)
    - Nested dictionaries
    - Lists/tuples
    - Task steps
    - Action steps
    - Planning steps

    The memory structure includes:
    - System prompt
    - Steps (list of actions/tasks/plans)
    - Metadata (creation time, agent type, model ID)

    Attributes:
        memories (Dict[str, Any]): Structure holding the agent's memory state
            - system_prompt: The agent's system prompt
            - steps: List of memory steps (tasks, actions, plans)
            - metadata: Creation time, agent type, and model information
    """

    def __init__(self):
        """Initialize an empty memory store with basic structure."""
        self.memories: Dict[str, Any] = {
            "system_prompt": None,
            "steps": [],
            "metadata": {"created_at": None, "updated_at": None, "agent_type": None, "model_id": None},
        }

    def _clean_step_data(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean step data to ensure it's JSON serializable.

        This method handles:
        - PIL Images (converts to base64)
        - Complex objects (converts to string representations)
        - Nested dictionaries (recursively cleaned)
        - Lists/tuples (cleaned item by item)

        Args:
            step_data (Dict[str, Any]): Raw step data from agent memory

        Returns:
            Dict[str, Any]: Cleaned data that can be JSON serialized
        """
        cleaned = {}
        for key, value in step_data.items():
            if key.startswith("_"):
                continue

            # Handle observations_images specifically
            if key == "observations_images" and value is not None:
                cleaned[key] = []
                for img in value:
                    if isinstance(img, Image.Image):
                        # Convert to RGB if needed
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        # Save image to bytes buffer
                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        cleaned[key].append(f"data:image/png;base64,{img_str}")
                continue

            if isinstance(value, (str, int, float, bool, type(None))):
                cleaned[key] = value
            elif isinstance(value, (list, tuple)):
                cleaned[key] = [
                    item if isinstance(item, (dict, str, int, float, bool)) else str(item) for item in value
                ]
            elif isinstance(value, dict):
                cleaned[key] = self._clean_step_data(value)
            else:
                cleaned[key] = str(value)
        return cleaned

    def save_memory_state(self, agent: Any) -> Dict[str, Any]:
        """
        Save current agent memory state.

        This method:
        1. Cleans all memory steps
        2. Saves system prompt
        3. Adds metadata (timestamp, agent type, model ID)

        Args:
            agent: The agent whose memory state to save

        Returns:
            Dict[str, Any]: Serializable memory state
        """
        logger.debug("Saving agent memory state")
        try:
            cleaned_steps = []
            for step in agent.memory.steps:
                step_dict = step.__dict__.copy()
                cleaned_steps.append(self._clean_step_data(step_dict))

            memory_data = {
                "system_prompt": agent.memory.system_prompt.system_prompt,
                "steps": cleaned_steps,
                "metadata": {
                    "created_at": datetime.now(UTC).isoformat(),
                    "agent_type": agent.__class__.__name__,
                    "model_id": agent.model.model_id,
                },
            }
            logger.info("Successfully saved agent memory state")
            return memory_data
        except Exception as e:
            logger.error(f"Failed to save agent memory: {e}")
            raise

    def restore_memory_state(self, agent, stored_memory: Dict[str, Any]):
        """Restore agent memory from stored state."""
        self._validate_stored_memory(stored_memory)
        self._restore_system_prompt(agent, stored_memory)
        self._restore_memory_steps(agent, stored_memory)
        self._restore_metadata(agent, stored_memory)

    def _validate_stored_memory(self, stored_memory: Dict[str, Any]) -> None:
        """Validate the structure of stored memory data.

        Args:
            stored_memory: Dictionary containing the stored memory state

        Raises:
            ValueError: If required fields are missing or invalid
        """
        logger.debug("Validating stored memory structure")

        # Check required fields exist
        for field in REQUIRED_FIELDS:
            if field not in stored_memory:
                raise ValueError(f"Missing required field '{field}' in stored memory")

        # Validate system prompt
        if not isinstance(stored_memory["system_prompt"], str):
            raise ValueError("System prompt must be a string")

        # Validate steps is a list
        if not isinstance(stored_memory["steps"], list):
            raise ValueError("Steps must be a list")

    def _restore_system_prompt(self, agent: Any, stored_memory: Dict[str, Any]) -> None:
        """Restore the system prompt to the agent.

        Args:
            agent: The agent to restore to
            stored_memory: Dictionary containing the stored memory state
        """
        logger.debug("Restoring system prompt")
        agent.memory.system_prompt = SystemPromptStep(system_prompt=stored_memory["system_prompt"])

    def _restore_memory_steps(self, agent: Any, stored_memory: Dict[str, Any]) -> None:
        """Restore memory steps to the agent.

        Args:
            agent: The agent to restore to
            stored_memory: Dictionary containing the stored memory state
        """
        logger.debug("Restoring memory steps")
        agent.memory.reset()

        for step_data in stored_memory["steps"]:
            try:
                if "task" in step_data:
                    self._restore_task_step(agent, step_data)
                elif "model_output" in step_data:
                    self._restore_action_step(agent, step_data)
                elif "plan" in step_data:
                    self._restore_planning_step(agent, step_data)
            except Exception as e:
                logger.error(f"Failed to restore step: {e}")
                continue

    def _restore_task_step(self, agent: Any, step_data: Dict[str, Any]) -> None:
        """Restore a task step with any associated images."""
        task_images = None
        if step_data.get("task_images"):
            task_images = []
            for img_data in step_data["task_images"]:
                if isinstance(img_data, str) and img_data.startswith("data:image"):
                    # Extract base64 data
                    img_format, img_str = img_data.split(";base64,")
                    img_bytes = base64.b64decode(img_str)
                    # Create PIL Image from bytes
                    img = Image.open(io.BytesIO(img_bytes))
                    # Convert to RGB to ensure compatibility
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    # Create AgentImage with the PIL Image
                    agent_img = AgentImage(value=img)
                    task_images.append(agent_img)

        step = TaskStep(task=step_data["task"], task_images=task_images)

        # Add message structure
        if not hasattr(step, "messages"):
            step.messages = []
        step.messages.append({"role": MessageRole.USER, "content": [{"type": "text", "text": step_data["task"]}]})

        agent.memory.steps.append(step)

    def _restore_action_step(self, agent: Any, step_data: Dict[str, Any]) -> None:
        """Restore an action step with tool calls and observations."""
        if step_data.get("tool_calls"):
            tool_calls = []
            for tc in step_data["tool_calls"]:
                if isinstance(tc, str):
                    tool_calls.append(ToolCall(name="final_answer", arguments=tc, id="call_1"))
                elif isinstance(tc, dict):
                    if "function" in tc:
                        tool_calls.append(
                            ToolCall(name=tc["function"]["name"], arguments=tc["function"]["arguments"], id=tc["id"])
                        )
            step_data["tool_calls"] = tool_calls

        step = ActionStep(**step_data)

        # Add message structure for model output
        if not hasattr(step, "messages"):
            step.messages = []
        if step_data.get("model_output"):
            step.messages.append(
                {"role": MessageRole.ASSISTANT, "content": [{"type": "text", "text": step_data["model_output"]}]}
            )

        # Handle observations_images
        if hasattr(step, "observations_images") and step.observations_images:
            restored_images = []
            for img_data in step.observations_images:
                if isinstance(img_data, str) and img_data.startswith("data:image/png;base64,"):
                    # Convert base64 back to PIL Image
                    base64_data = img_data.split("base64,")[1]
                    img_bytes = base64.b64decode(base64_data)
                    img = Image.open(io.BytesIO(img_bytes))
                    restored_images.append(img)
            step.observations_images = restored_images if restored_images else None

        # Clean up messages
        self._clean_messages(step)

        agent.memory.steps.append(step)

    def _restore_planning_step(self, agent: Any, step_data: Dict[str, Any]) -> None:
        """Restore a planning step."""
        step = PlanningStep(**step_data)
        agent.memory.steps.append(step)

    def _clean_messages(self, step: Any) -> None:
        """Clean up message content to remove non-text items."""
        if hasattr(step, "messages"):
            filtered_messages = []
            for msg in step.messages:
                if "content" in msg:
                    msg["content"] = [
                        item for item in msg["content"] if isinstance(item, dict) and item.get("type") == "text"
                    ]
                filtered_messages.append(msg)
            step.messages = filtered_messages

        if hasattr(step, "model_input_messages"):
            filtered_input_messages = []
            for msg in step.model_input_messages:
                if "content" in msg:
                    msg["content"] = [
                        item for item in msg["content"] if isinstance(item, dict) and item.get("type") == "text"
                    ]
                filtered_input_messages.append(msg)
            step.model_input_messages = filtered_input_messages

    def _restore_metadata(self, agent: Any, stored_memory: Dict[str, Any]) -> None:
        """Restore metadata if present in stored memory."""
        if "metadata" in stored_memory:
            logger.debug("Restoring metadata")
            # Here you could restore any metadata needed
            # Currently just logging it
            logger.info(f"Restored metadata: {stored_memory['metadata']}")


def save_agent_state(agent) -> Dict[str, Any]:
    """
    Helper function to serialize agent state to a JSON-compatible dictionary.

    This function converts the agent's memory state into a serializable format
    but leaves the actual storage mechanism up to the caller. The resulting
    dictionary can be stored in:
    - Files
    - Databases
    - Redis
    - Any other storage system

    Args:
        agent: The agent whose state to serialize

    Returns:
        Dict[str, Any]: JSON-serializable dictionary containing the agent's state
    """
    memory_store = AgentMemoryStore()
    return memory_store.save_memory_state(agent)


def load_agent_state(agent, stored_memory: Dict[str, Any]):
    """
    Helper function to restore agent state from a dictionary.

    This function takes a previously serialized agent state and restores it
    to the agent. The stored_memory can come from any storage system:
    - Files
    - Databases
    - Redis
    - Any other storage system

    Args:
        agent: The agent whose state to restore
        stored_memory (Dict[str, Any]): Previously serialized agent state
    """
    memory_store = AgentMemoryStore()
    memory_store.restore_memory_state(agent, stored_memory)


__all__ = ["AgentMemoryStore", "save_agent_state", "load_agent_state"]
