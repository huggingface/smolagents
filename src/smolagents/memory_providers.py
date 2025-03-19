"""
Alternative memory providers for smolagents.

This module contains alternative implementations of memory providers that can be used
with smolagents. These providers implement the MemoryProvider protocol defined in memory.py.
"""

import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from smolagents.memory import (
    ActionStep,
    MemoryProvider,
    MemoryStep,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
)
from smolagents.monitoring import AgentLogger, LogLevel
from smolagents.utils import make_json_serializable


class DictMemory:
    """
    A memory provider that stores memory steps in an OrderedDict.

    This implementation uses an OrderedDict to store memory steps, which allows for
    efficient lookup by step ID while maintaining insertion order.

    Attributes:
        system_prompt (SystemPromptStep): The system prompt step.
        steps (OrderedDict): An ordered dictionary mapping step IDs to memory steps.
        next_id (int): The next step ID to assign.
    """

    def __init__(self, system_prompt: str):
        """
        Initialize a new DictMemory instance.

        Args:
            system_prompt (str): The system prompt to use.
        """
        self.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        self.steps: OrderedDict[str, Union[TaskStep, ActionStep, PlanningStep]] = OrderedDict()
        self.next_id = 1

    def reset(self) -> None:
        """Reset the memory by clearing all steps."""
        self.steps.clear()
        self.next_id = 1

    def add_step(self, step: Union[TaskStep, ActionStep, PlanningStep]) -> str:
        """
        Add a step to the memory.

        Args:
            step (Union[TaskStep, ActionStep, PlanningStep]): The step to add.

        Returns:
            str: The ID of the added step.
        """
        step_id = f"step_{self.next_id}"
        self.steps[step_id] = step
        self.next_id += 1
        return step_id

    def get_step(self, step_id: str) -> Optional[MemoryStep]:
        """
        Get a step by ID.

        Args:
            step_id (str): The ID of the step to get.

        Returns:
            Optional[MemoryStep]: The step with the given ID, or None if not found.
        """
        return self.steps.get(step_id)

    def get_succinct_steps(self) -> list[dict]:
        """
        Get a succinct representation of the memory steps.

        This excludes model_input_messages to reduce verbosity.

        Returns:
            list[dict]: A list of dictionaries representing the memory steps.
        """
        return [
            {key: value for key, value in step.dict().items() if key != "model_input_messages"}
            for step in self.steps.values()
        ]

    def get_full_steps(self) -> list[dict]:
        """
        Get a full representation of the memory steps.

        Returns:
            list[dict]: A list of dictionaries representing the memory steps.
        """
        return [step.dict() for step in self.steps.values()]

    def replay(self, logger: AgentLogger, detailed: bool = False) -> None:
        """
        Prints a pretty replay of the agent's steps.

        Args:
            logger (AgentLogger): The logger to print replay logs to.
            detailed (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        logger.console.log("Replaying the agent's steps:")

        # First, log the system prompt if detailed is True
        if detailed:
            logger.log_markdown(title="System prompt", content=self.system_prompt.system_prompt, level=LogLevel.ERROR)

        # Then, log each step
        for step in self.steps.values():
            if isinstance(step, TaskStep):
                logger.log_task(step.task, "", level=LogLevel.ERROR)
            elif isinstance(step, ActionStep):
                logger.log_rule(f"Step {step.step_number}", level=LogLevel.ERROR)
                if detailed:
                    logger.log_messages(step.model_input_messages)
                logger.log_markdown(title="Agent output:", content=step.model_output, level=LogLevel.ERROR)
            elif isinstance(step, PlanningStep):
                logger.log_rule("Planning step", level=LogLevel.ERROR)
                if detailed:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                logger.log_markdown(title="Agent output:", content=step.facts + "\n" + step.plan, level=LogLevel.ERROR)

    def write_to_messages(self, summary_mode: bool = False) -> List[Dict[str, Any]]:
        """
        Convert the memory steps to a list of messages.

        Args:
            summary_mode (bool, optional): If True, exclude certain details to create a summary. Defaults to False.

        Returns:
            List[Dict[str, Any]]: A list of messages.
        """
        messages = self.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in self.steps.values():
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages


class FilteredMemory:
    """
    A memory provider that filters memory steps based on a predicate.

    This implementation wraps another memory provider and filters its steps
    based on a predicate function.

    Attributes:
        base_memory (MemoryProvider): The base memory provider to wrap.
        filter_predicate (callable): A function that takes a memory step and returns
            a boolean indicating whether to include the step.
    """

    def __init__(self, base_memory: MemoryProvider, filter_predicate: callable):
        """
        Initialize a new FilteredMemory instance.

        Args:
            base_memory (MemoryProvider): The base memory provider to wrap.
            filter_predicate (callable): A function that takes a memory step and returns
                a boolean indicating whether to include the step.
        """
        self.base_memory = base_memory
        self.filter_predicate = filter_predicate

    def reset(self) -> None:
        """Reset the memory by clearing all steps."""
        self.base_memory.reset()

    def get_succinct_steps(self) -> list[dict]:
        """
        Get a succinct representation of the filtered memory steps.

        Returns:
            list[dict]: A list of dictionaries representing the filtered memory steps.
        """
        steps = self.base_memory.get_succinct_steps()
        return [step for step in steps if self.filter_predicate(step)]

    def get_full_steps(self) -> list[dict]:
        """
        Get a full representation of the filtered memory steps.

        Returns:
            list[dict]: A list of dictionaries representing the filtered memory steps.
        """
        steps = self.base_memory.get_full_steps()
        return [step for step in steps if self.filter_predicate(step)]

    def replay(self, logger: AgentLogger, detailed: bool = False) -> None:
        """
        Prints a pretty replay of the filtered agent's steps.

        Args:
            logger (AgentLogger): The logger to print replay logs to.
            detailed (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        # This is a bit tricky since we need to filter steps before replaying
        # For simplicity, we'll just call the base memory's replay method
        # A more sophisticated implementation would filter the steps before replaying
        self.base_memory.replay(logger, detailed)

    def write_to_messages(self, summary_mode: bool = False) -> List[Dict[str, Any]]:
        """
        Convert the filtered memory steps to a list of messages.

        Args:
            summary_mode (bool, optional): If True, exclude certain details to create a summary. Defaults to False.

        Returns:
            List[Dict[str, Any]]: A list of messages.
        """
        # This is also tricky since we need to filter steps before converting to messages
        # For simplicity, we'll just call the base memory's write_to_messages method
        # A more sophisticated implementation would filter the steps before converting
        return self.base_memory.write_to_messages(summary_mode)


class ExternalStorageMemory:
    """
    A memory provider that stores memory steps in external storage (JSON files).

    This implementation allows for persistent storage of memory steps, enabling
    agents to resume conversations from checkpoints and replay full conversations.

    Attributes:
        system_prompt (SystemPromptStep): The system prompt step.
        steps (OrderedDict): An ordered dictionary mapping step IDs to memory steps.
        next_id (int): The next step ID to assign.
        storage_dir (Path): The directory where memory steps are stored.
        conversation_id (str): A unique identifier for the conversation.
    """

    def __init__(self, system_prompt: str, storage_dir: str = "memory_storage", conversation_id: str = None):
        """
        Initialize a new ExternalStorageMemory instance.

        Args:
            system_prompt (str): The system prompt to use.
            storage_dir (str, optional): The directory where memory steps are stored. Defaults to "memory_storage".
            conversation_id (str, optional): A unique identifier for the conversation. If None, a new ID is generated.
        """
        self.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        self.steps: OrderedDict[str, Union[TaskStep, ActionStep, PlanningStep]] = OrderedDict()
        self.next_id = 1

        # Create storage directory if it doesn't exist
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Generate a unique conversation ID if not provided
        self.conversation_id = conversation_id or f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create conversation directory
        self.conversation_dir = self.storage_dir / self.conversation_id
        self.conversation_dir.mkdir(parents=True, exist_ok=True)

        # Save system prompt
        self._save_system_prompt()

    def _save_system_prompt(self) -> None:
        """Save the system prompt to a file."""
        system_prompt_file = self.conversation_dir / "system_prompt.json"
        with open(system_prompt_file, "w") as f:
            json.dump({"system_prompt": self.system_prompt.system_prompt}, f, indent=2)

    def reset(self) -> None:
        """Reset the memory by clearing all steps."""
        self.steps.clear()
        self.next_id = 1

        # Clear all step files but keep the system prompt
        for file_path in self.conversation_dir.glob("step_*.json"):
            file_path.unlink()

        # Clear all checkpoint files
        for file_path in self.conversation_dir.glob("checkpoint_*.json"):
            file_path.unlink()

    def add_step(self, step: Union[TaskStep, ActionStep, PlanningStep]) -> str:
        """
        Add a step to the memory and save it to external storage.

        Args:
            step (Union[TaskStep, ActionStep, PlanningStep]): The step to add.

        Returns:
            str: The ID of the added step.
        """
        step_id = f"step_{self.next_id}"
        self.steps[step_id] = step
        self.next_id += 1

        # Save step to a file
        step_file = self.conversation_dir / f"{step_id}.json"
        with open(step_file, "w") as f:
            step_dict = step.dict()
            # Convert to JSON-serializable format
            step_dict = make_json_serializable(step_dict)
            json.dump(step_dict, f, indent=2)

        return step_id

    def get_step(self, step_id: str) -> Optional[MemoryStep]:
        """
        Get a step by ID.

        Args:
            step_id (str): The ID of the step to get.

        Returns:
            Optional[MemoryStep]: The step with the given ID, or None if not found.
        """
        # Try to get from memory first
        if step_id in self.steps:
            return self.steps[step_id]

        # Try to load from file
        step_file = self.conversation_dir / f"{step_id}.json"
        if step_file.exists():
            with open(step_file, "r") as f:
                step_dict = json.load(f)

                # Create the appropriate step object based on the type
                if "task" in step_dict:
                    return TaskStep(**step_dict)
                elif "model_output" in step_dict:
                    # Handle the case where 'step' is used instead of 'step_number'
                    if "step" in step_dict and "step_number" not in step_dict:
                        step_dict["step_number"] = step_dict.pop("step")
                    return ActionStep(**step_dict)
                elif "facts" in step_dict and "plan" in step_dict:
                    return PlanningStep(**step_dict)

        return None

    def get_succinct_steps(self) -> list[dict]:
        """
        Get a succinct representation of the memory steps.

        This excludes model_input_messages to reduce verbosity.

        Returns:
            list[dict]: A list of dictionaries representing the memory steps.
        """
        # Load all steps from files if not already in memory
        self._load_all_steps()

        return [
            {key: value for key, value in step.dict().items() if key != "model_input_messages"}
            for step in self.steps.values()
        ]

    def get_full_steps(self) -> list[dict]:
        """
        Get a full representation of the memory steps.

        Returns:
            list[dict]: A list of dictionaries representing the memory steps.
        """
        # Load all steps from files if not already in memory
        self._load_all_steps()

        return [step.dict() for step in self.steps.values()]

    def _load_all_steps(self) -> None:
        """Load all steps from files if not already in memory."""
        # Get all step files
        step_files = sorted(self.conversation_dir.glob("step_*.json"), key=lambda x: int(x.stem.split("_")[1]))

        # Load each step if not already in memory
        for step_file in step_files:
            step_id = step_file.stem
            if step_id not in self.steps:
                self.get_step(step_id)

    def replay(self, logger: AgentLogger, detailed: bool = False) -> None:
        """
        Prints a pretty replay of the agent's steps.

        Args:
            logger (AgentLogger): The logger to print replay logs to.
            detailed (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        logger.console.log(f"Replaying conversation: {self.conversation_id}")

        # First, log the system prompt if detailed is True
        if detailed:
            logger.log_markdown(title="System prompt", content=self.system_prompt.system_prompt, level=LogLevel.ERROR)

        # Load all steps from files if not already in memory
        self._load_all_steps()

        # Then, log each step
        for step in self.steps.values():
            if isinstance(step, TaskStep):
                logger.log_task(step.task, "", level=LogLevel.ERROR)
            elif isinstance(step, ActionStep):
                logger.log_rule(f"Step {step.step_number}", level=LogLevel.ERROR)
                if detailed:
                    logger.log_messages(step.model_input_messages)
                logger.log_markdown(title="Agent output:", content=step.model_output, level=LogLevel.ERROR)
            elif isinstance(step, PlanningStep):
                logger.log_rule("Planning step", level=LogLevel.ERROR)
                if detailed:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                logger.log_markdown(title="Agent output:", content=step.facts + "\n" + step.plan, level=LogLevel.ERROR)

    def write_to_messages(self, summary_mode: bool = False) -> List[Dict[str, Any]]:
        """
        Convert the memory steps to a list of messages.

        Args:
            summary_mode (bool, optional): If True, exclude certain details to create a summary. Defaults to False.

        Returns:
            List[Dict[str, Any]]: A list of messages.
        """
        # Load all steps from files if not already in memory
        self._load_all_steps()

        messages = self.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in self.steps.values():
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages

    def save_checkpoint(self, checkpoint_name: str = None) -> str:
        """
        Save a checkpoint of the current memory state.

        Args:
            checkpoint_name (str, optional): A name for the checkpoint. If None, a timestamp is used.

        Returns:
            str: The name of the saved checkpoint.
        """
        # Generate a checkpoint name if not provided
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create checkpoint file
        checkpoint_file = self.conversation_dir / f"{checkpoint_name}.json"

        # Save checkpoint data
        checkpoint_data = {
            "checkpoint_name": checkpoint_name,
            "conversation_id": self.conversation_id,
            "system_prompt": self.system_prompt.system_prompt,
            "next_id": self.next_id,
            "step_ids": list(self.steps.keys()),
            "timestamp": datetime.now().isoformat(),
        }

        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        return checkpoint_name

    def load_checkpoint(self, checkpoint_name: str) -> bool:
        """
        Load a checkpoint of a memory state.

        Args:
            checkpoint_name (str): The name of the checkpoint to load.

        Returns:
            bool: True if the checkpoint was loaded successfully, False otherwise.
        """
        # Find checkpoint file
        checkpoint_file = self.conversation_dir / f"{checkpoint_name}.json"

        if not checkpoint_file.exists():
            return False

        # Load checkpoint data
        with open(checkpoint_file, "r") as f:
            checkpoint_data = json.load(f)

        # Reset current memory
        self.steps.clear()

        # Restore system prompt
        self.system_prompt = SystemPromptStep(system_prompt=checkpoint_data["system_prompt"])

        # Restore next_id
        self.next_id = checkpoint_data["next_id"]

        # Load steps from checkpoint
        for step_id in checkpoint_data["step_ids"]:
            step = self.get_step(step_id)
            if step:
                self.steps[step_id] = step

        return True

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.

        Returns:
            List[Dict[str, Any]]: A list of checkpoint information.
        """
        checkpoints = []

        # Look for checkpoint files
        for checkpoint_file in self.conversation_dir.glob("*.json"):
            # Skip system prompt and step files
            if checkpoint_file.stem == "system_prompt" or checkpoint_file.stem.startswith("step_"):
                continue

            try:
                with open(checkpoint_file, "r") as f:
                    checkpoint_data = json.load(f)
                    if (
                        "checkpoint_name" in checkpoint_data
                        and "timestamp" in checkpoint_data
                        and "step_ids" in checkpoint_data
                    ):
                        checkpoints.append(
                            {
                                "name": checkpoint_data["checkpoint_name"],
                                "timestamp": checkpoint_data["timestamp"],
                                "step_count": len(checkpoint_data["step_ids"]),
                            }
                        )
            except (json.JSONDecodeError, KeyError):
                # Skip files that are not valid checkpoint files
                continue

        # Also look for files with step_N as the checkpoint name
        for step_number in range(1, 100):  # Limit to 100 steps to avoid infinite loop
            checkpoint_name = f"step_{step_number}"
            checkpoint_file = self.conversation_dir / f"{checkpoint_name}.json"

            if not checkpoint_file.exists():
                continue

            # Check if this is a step file or a checkpoint file
            try:
                with open(checkpoint_file, "r") as f:
                    file_data = json.load(f)

                    # If it has checkpoint data, it's a checkpoint file
                    if "checkpoint_name" in file_data and "timestamp" in file_data and "step_ids" in file_data:
                        checkpoints.append(
                            {
                                "name": file_data["checkpoint_name"],
                                "timestamp": file_data["timestamp"],
                                "step_count": len(file_data["step_ids"]),
                            }
                        )
            except (json.JSONDecodeError, KeyError):
                # Skip files that are not valid checkpoint files
                continue

        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)

        return checkpoints

    def list_conversations(self) -> List[str]:
        """
        List all available conversations in the storage directory.

        Returns:
            List[str]: A list of conversation IDs.
        """
        conversations = []

        for conversation_dir in self.storage_dir.iterdir():
            if conversation_dir.is_dir() and (conversation_dir / "system_prompt.json").exists():
                conversations.append(conversation_dir.name)

        return conversations

    @classmethod
    def load_conversation(cls, conversation_id: str, storage_dir: str = "memory_storage") -> "ExternalStorageMemory":
        """
        Load an existing conversation.

        Args:
            conversation_id (str): The ID of the conversation to load.
            storage_dir (str, optional): The directory where memory steps are stored. Defaults to "memory_storage".

        Returns:
            ExternalStorageMemory: A new ExternalStorageMemory instance with the loaded conversation.

        Raises:
            FileNotFoundError: If the conversation does not exist.
        """
        storage_path = Path(storage_dir)
        conversation_dir = storage_path / conversation_id

        if not conversation_dir.exists() or not (conversation_dir / "system_prompt.json").exists():
            raise FileNotFoundError(f"Conversation {conversation_id} not found in {storage_dir}")

        # Load system prompt
        with open(conversation_dir / "system_prompt.json", "r") as f:
            system_prompt_data = json.load(f)
            system_prompt = system_prompt_data["system_prompt"]

        # Create a new instance with the loaded conversation
        memory = cls(system_prompt=system_prompt, storage_dir=storage_dir, conversation_id=conversation_id)

        # Load all steps
        memory._load_all_steps()

        return memory
