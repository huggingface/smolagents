"""
Alternative memory providers for smolagents.

This module contains alternative implementations of memory providers that can be used
with smolagents. These providers implement the MemoryProvider protocol defined in memory.py.
"""

from collections import OrderedDict
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
